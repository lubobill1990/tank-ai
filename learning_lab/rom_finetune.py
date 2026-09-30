"""Reserved on-policy normal-ROM fine-tuning of a verified imitation policy.

Training openings are fixed 0/1/0/1, never SB3 seed+n. No teacher control or BC
loss is used. Rollouts and bootstrap substeps have separate reservations.
Checkpoints contain committed policy+Adam only, never simulator state.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecFrameStack

from learning_lab import imitation, backplay_train as machinery
from learning_lab.budget import BudgetExhausted
from learning_lab.rom_curriculum_env import RomCurriculumEnv, normal_start_spec
from learning_lab.rom_player import play_development_episode
from learning_lab.victory_objective import FiniteVictoryVecEnv, SPARSE_OBJECTIVE_VERSION
from native_rollout import NativeGPURunRolloutBuffer
from policy import Grid26CNN
from rust_resources import ResourceMonitor, ResourcePressureError


ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'tank-ai-rom-finetune-checkpoint-v1'
DeadlineReached = machinery.DeadlineReached
PPO = dict(learning_rate=5e-5, gamma=1., gae_lambda=.95, clip_range=.1,
           ent_coef=.001, target_kl=.01, n_epochs=4, n_steps=256, batch_size=256)
ROLLOUT = 1024
OPENINGS = (0, 1, 0, 1)
DEV_SEEDS = (0, 1, 7, 39)
GLOBAL_DEADLINE = datetime.fromisoformat('2026-09-30T21:01:44+08:00').timestamp()


def _contract():
    return dict(observation_shape=[94, 26, 26], actions=10, n_envs=4, n_stack=2,
                extractor='Grid26CNN', normalize_images=False, net_arch={'pi': [64, 64], 'vf': [64, 64]},
                ppo=deepcopy(PPO), opening_seeds=list(OPENINGS), stage=1, frame_skip=4,
                reward_objective=SPARSE_OBJECTIVE_VERSION, shaping=False, timeout_bootstrap=False,
                teacher_control=False, bc_regularizer=False, rollout_reservation=ROLLOUT,
                save_every=8192, evaluate_every=16384, dev_seeds=list(DEV_SEEDS),
                selection='real-dev-wins-then-mean-kills', exact_simulator_resume=False)


def _sources():
    result = machinery._sources()
    for name in ('learning_lab\\rom_finetune.py', 'learning_lab\\rom_curriculum_env.py',
                 'learning_lab\\rom_player.py', 'learning_lab\\evaluation.py',
                 'learning_lab\\budget.py', 'learning_lab\\imitation.py',
                 'battle_city_env.py', 'game_state.py', 'heuristic.py', 'rust_validation.py',
                 'battle_city.nes'):
        result[name] = machinery._hash(ROOT / name)
    return result


def _state_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode() + str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class BootstrapMeter:
    """Reserve before every unreserved boot/reset substep; failures stay charged."""
    def __init__(self, budget, prefix, guard):
        self.budget, self.prefix, self.guard = budget, prefix, guard
        self.rows, self.active = [], None

    def _reserve(self, count):
        self.guard.check('bootstrap_reservation')
        token = self.budget.reserve(f'{self.prefix}-bootstrap-{len(self.rows)}', count)
        self.active = dict(reservation=token, reserved=count, attempted=0, settled=False)
        self.rows.append(self.active)

    def begin_initial(self):
        self._reserve(4096)

    def consume(self, event):
        self.guard.check('bootstrap_substep')
        if event.get('kind') != 'normal_reset_substep' or event.get('count') != 1:
            raise ValueError('Only normal reset substeps are allowed; teacher/curriculum initialization forbidden')
        if self.active is not None and self.active['attempted'] == self.active['reserved']:
            self.flush_success()
        if self.active is None:
            self._reserve(512)
        self.active['attempted'] += 1

    def flush_success(self):
        if self.active is not None:
            self.budget.settle(self.active['reservation'], self.active['attempted'])
            self.active['settled'] = True
            self.active = None


class DeferredEvaluationBudget:
    """rom_player may request settlement in finally; commit only returned episodes."""
    def __init__(self, budget, guard=None):
        self.budget, self.rows, self.guard = budget, [], guard

    def snapshot(self):
        state = dict(self.budget.snapshot())
        if self.guard is not None:
            state['remaining_seconds'] = max(0., min(
                state['remaining_seconds'], self.guard.deadline - time.monotonic(),
                self.guard.deadline_epoch - time.time()))
        return state

    def reserve(self, label, interactions):
        token = self.budget.reserve(label, interactions)
        self.rows.append(dict(reservation=token, reserved=interactions, actual=None, settled=False))
        return token

    def settle(self, token, interactions):
        row = next(row for row in self.rows if row['reservation'] == token)
        if type(interactions) is not int or not 0 <= interactions <= row['reserved']:
            raise ValueError('Invalid evaluation settlement')
        row['actual'] = interactions

    def commit(self):
        for row in self.rows:
            if not row['settled']:
                if row['actual'] is None:
                    raise ValueError('Unresolved evaluation cannot be settled')
                self.budget.settle(row['reservation'], row['actual'])
                row['settled'] = True


class DevelopmentSelection:
    def __init__(self, initial):
        if initial['status'] != 'completed':
            raise ValueError('A completed initial development reference is required')
        self.best = (initial['wins'], initial['mean_kills'])
        self.below_initial_checks = 0

    def consider(self, evaluation):
        if evaluation['status'] != 'completed':
            return False, False
        score = (evaluation['wins'], evaluation['mean_kills'])
        better = score > self.best
        if better:
            self.best = score
        self.below_initial_checks += int(evaluation['train_wins'] < 2)
        return better, self.below_initial_checks >= 2


class _CountedVec(VecEnvWrapper):
    def __init__(self, venv, guard):
        super().__init__(venv)
        self.guard, self.completed_steps, self.limit = guard, 0, 0

    def reset(self):
        self.guard.check('rom_vector_reset')
        return self.venv.reset()

    def step_wait(self):
        self.guard.check('rom_vector_step')
        if self.completed_steps + self.num_envs > self.limit:
            raise RuntimeError('ROM step has no sufficient rollout reservation')
        result = self.venv.step_wait()
        self.completed_steps += self.num_envs
        return result


class _EpisodeCallback(machinery._Callback):
    def __init__(self, guard, output):
        super().__init__(guard, SPARSE_OBJECTIVE_VERSION)
        self.output = output

    def _on_step(self):
        result = super()._on_step()
        for index, done in enumerate(self.locals['dones']):
            if done:
                row = {key: value for key, value in self.locals['infos'][index].items()
                       if key != 'terminal_observation'}
                row.update(env_index=index, collected_steps=self.model.num_timesteps)
                if row.get('source_seed') not in (0, 1):
                    raise ValueError('Training seed pool violation')
                with (self.output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(_json_safe(row), allow_nan=False) + '\n')
        return result


class _Runtime(machinery._Runtime):
    """Reuse the tested transactional optimizer, with a ROM-only constructor."""
    def __init__(self, warm, output, guard, bootstrap, *, seed, device, max_steps,
                 env_factory=RomCurriculumEnv, settings=None):
        self.settings = dict(PPO if settings is None else settings)
        self.source_pin = _sources()
        self.opening_seeds = list(OPENINGS)
        self.rollout_size = 4 * self.settings['n_steps']
        self.guard, self.optimized_steps, self.adam_steps = guard, 0, 0
        self.max_steps, self._step_count = max_steps, {'count': 0}
        self.model = self.env = self.raw = None
        self.physical_envs = []
        torch.set_num_threads(1)
        def create(opening):
            env = env_factory(normal_start_spec(opening), max_frames=12000,
                              deadline=guard.deadline, initialization_counter=bootstrap.consume)
            self.physical_envs.append(env)
            return Monitor(env)
        try:
            self.raw = DummyVecEnv([lambda opening=opening: create(opening) for opening in OPENINGS])
            self.counted = _CountedVec(self.raw, guard)
            objective = FiniteVictoryVecEnv(
                self.counted, np.full(4, 20, np.int64),
                lambda: np.asarray(self.raw.get_attr('current_kills'), np.int64),
                gamma=1., shaping=False)
            self.env = VecFrameStack(objective, n_stack=2, channels_order='first')
            self.model = MaskablePPO(
                'CnnPolicy', self.env, seed=seed, device=device, verbose=0, **self.settings,
                policy_kwargs={'features_extractor_class': Grid26CNN, 'normalize_images': False,
                               'net_arch': {'pi': [64, 64], 'vf': [64, 64]}},
                rollout_buffer_class=machinery._GuardedBuffer,
                rollout_buffer_kwargs={'channels_per_frame': 47, 'max_storage_bytes': 2 * 1024**3})
            if str(device).startswith('cuda') and self.model.device.type != 'cuda':
                raise RuntimeError('Requested CUDA must not silently fall back to CPU')
            self.model.policy.load_state_dict(warm.policy.state_dict(), strict=True)
            self.model.policy.optimizer.load_state_dict(machinery._cpu_copy(warm.policy.optimizer.state_dict()))
            self.model._n_updates = int(warm._n_updates)
            self.start_policy_sha256 = _state_hash(self.model.policy.state_dict())
            self.start_adam_sha256 = _state_hash(self.model.policy.optimizer.state_dict())
            if self.start_policy_sha256 != _state_hash(warm.policy.state_dict()) \
                    or self.start_adam_sha256 != _state_hash(warm.policy.optimizer.state_dict()):
                raise ValueError('Warm policy/Adam identity failed')
            self.model.rollout_buffer.guard = guard
            self._last_good, self.state_verified = self.export_warm(), True
            original_step = self.model.policy.optimizer.step
            def guarded_step(*args, **kwargs):
                guard.check('adam_step')
                if any(not torch.isfinite(parameter.grad).all()
                       for parameter in self.model.policy.parameters() if parameter.grad is not None):
                    raise FloatingPointError('Nonfinite PPO gradient')
                result = original_step(*args, **kwargs)
                if any(not torch.isfinite(parameter).all() for parameter in self.model.policy.parameters()):
                    raise FloatingPointError('Nonfinite PPO parameters after Adam update')
                self._step_count['count'] += 1
                return result
            self.model.policy.optimizer.step = guarded_step
            self.model.set_logger(configure(str(output / 'optimizer-logs'), ['json']))
            self.total, self.callback = self.model._setup_learn(
                max_steps, callback=_EpisodeCallback(guard, output))
            self.callback.on_training_start({}, {})
        except BaseException:
            self.close()
            raise

    def collect(self):
        self.counted.limit = self.completed_steps + self.rollout_size
        return self.model.collect_rollouts(self.env, callback=self.callback,
                                           rollout_buffer=self.model.rollout_buffer,
                                           n_rollout_steps=self.settings['n_steps'])

    def close(self):
        if self.model is not None:
            self.model.policy.optimizer.__dict__.pop('step', None)
            self.model.rollout_buffer.guard = None
        if hasattr(self, 'callback'):
            self.callback.on_training_end()
        if self.model is not None and getattr(self.model, '_logger', None) is not None:
            self.model.logger.close()
        if self.env is not None:
            self.env.close()
        elif self.raw is not None:
            self.raw.close()
        else:
            for env in self.physical_envs:
                env.close()


def _collect_reserved(runtime, budget, bootstrap, label):
    row = machinery.collect_reserved(runtime, budget, label)
    if row['settled']:
        bootstrap.flush_success()
    return row


def _parent_identity(path):
    path = Path(path).resolve()
    manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    return {'path': str(path), 'sha256': machinery._hash(path),
            'manifest_sha256': machinery._hash(path.with_suffix('.manifest.json')),
            'manifest': manifest}


def _save_checkpoint(runtime, path, parent, role):
    if not runtime.state_verified or runtime.optimized_steps % ROLLOUT:
        raise ValueError('Only initial or fully committed rollout updates may be saved')
    if runtime.source_pin != _sources():
        raise ValueError('Frozen fine-tuning sources changed during the run')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    collected = runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps = runtime.optimized_steps
        runtime.model.save(path, exclude=['_last_obs', '_last_episode_starts', '_last_original_obs'])
    finally:
        runtime.model.num_timesteps = collected
    manifest = {
        'format': FORMAT, 'contract': _contract(), 'checkpoint_file': path.name,
        'checkpoint_sha256': machinery._hash(path), 'sources': runtime.source_pin, 'parent': parent,
        'role': role, 'optimized_steps': runtime.optimized_steps, 'adam_steps': runtime.adam_steps,
        'ppo_epoch_updates': int(runtime.model._n_updates), 'exact_simulator_resume': False,
        'policy_sha256': _state_hash(runtime.model.policy.state_dict()),
        'adam_sha256': _state_hash(runtime.model.policy.optimizer.state_dict()),
        'start_policy_sha256': runtime.start_policy_sha256, 'start_adam_sha256': runtime.start_adam_sha256,
        'warning': 'On-policy normal-ROM training openings0/1 only. Development selection is not unseen-map generalization.',
    }
    machinery._write(path.with_suffix('.manifest.json'), manifest)
    return {'path': str(path), 'sha256': manifest['checkpoint_sha256'],
            'optimized_steps': runtime.optimized_steps}


def strict_load_rom_finetune_checkpoint(path, device='cpu'):
    """Strict trusted-local policy+Adam restore; small inference buffer, no env resume."""
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        parent = manifest['parent']
        if manifest['format'] != FORMAT or manifest['contract'] != _contract() \
                or manifest['sources'] != _sources() or manifest['checkpoint_file'] != path.name \
                or manifest['exact_simulator_resume'] is not False \
                or isinstance(manifest['contract']['ppo']['gamma'], bool) \
                or type(manifest['optimized_steps']) is not int or not 0 <= manifest['optimized_steps'] <= 65536 \
                or manifest['optimized_steps'] % ROLLOUT \
                or parent['manifest']['format'] != imitation._FORMAT \
                or parent['manifest']['contract'] != imitation._CONTRACT \
                or parent['manifest']['sources'] != imitation._source_hashes():
            raise ValueError('Invalid ROM fine-tune contract/source/parent')
        for name in ('sha256', 'manifest_sha256'):
            value = parent[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid parent SHA256')
        if parent['sha256'] != parent['manifest']['checkpoint_sha256']:
            raise ValueError('Parent archive identity mismatch')
        imitation._verify_provenance(parent['manifest'])
        if machinery._hash(path) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA256 mismatch')
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('Missing or malformed ROM fine-tune manifest') from error
    model = MaskablePPO.load(path, device=device, custom_objects={
        'rollout_buffer_class': NativeGPURunRolloutBuffer,
        'rollout_buffer_kwargs': {'channels_per_frame': 47, 'max_storage_bytes': 2 * 1024**3},
        'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    imitation._verify_model(model)
    if isinstance(model.gamma, bool) or model.gamma != 1. or model.gae_lambda != .95 or model.learning_rate != 5e-5 \
            or model.ent_coef != .001 or model.target_kl != .01 or model.n_epochs != 4 \
            or model.clip_range(1.) != .1 or model.num_timesteps != manifest['optimized_steps'] \
            or model._n_updates != manifest['ppo_epoch_updates'] \
            or _state_hash(model.policy.state_dict()) != manifest['policy_sha256'] \
            or _state_hash(model.policy.optimizer.state_dict()) != manifest['adam_sha256']:
        raise ValueError('Serialized policy/optimizer violates fine-tune contract')
    return model


load_rom_finetune_checkpoint = strict_load_rom_finetune_checkpoint
strictload_rom_finetune_checkpoint = strict_load_rom_finetune_checkpoint


@contextmanager
def _evaluation_bootstrap_meter(meter):
    """Read-only call profiling charges evaluator boot frames, not gameplay repeats."""
    from nes_py import NESEnv
    from battle_city_env import BattleCityEnv
    previous, depth = sys.getprofile(), 0
    if previous is not None:
        raise RuntimeError('Cannot replace an existing profiler during evaluation accounting')
    def profile(frame, event, value):
        nonlocal depth
        if frame.f_code is BattleCityEnv.step.__code__:
            depth += 1 if event == 'call' else -1 if event == 'return' else 0
        elif frame.f_code is NESEnv.step.__code__ and event == 'call':
            meter.guard.check('development_nes_frame')
            if depth == 0:
                meter.consume({'kind': 'normal_reset_substep', 'count': 1})
    try:
        sys.setprofile(profile)
        yield
    finally:
        sys.setprofile(previous)


def _evaluate(model, output, budget, guard, checkpoint, index):
    row = {'status': 'running', 'index': index, 'episodes': [], 'reservations': [],
           'bootstrap_reservations': [], 'wins': 0, 'train_wins': 0, 'mean_kills': None}
    meter = BootstrapMeter(budget, 'eval-boot-' + hashlib.sha256(str(output).encode()).hexdigest()[:16], guard)
    proxy = DeferredEvaluationBudget(budget, guard)
    row['bootstrap_reservations'], row['reservations'] = meter.rows, proxy.rows
    try:
        meter.begin_initial()
        for seed in DEV_SEEDS:
            guard.check('development_episode')
            with _evaluation_bootstrap_meter(meter):
                result = play_development_episode(
                    model, output=output / f'seed-{seed}.json', budget=proxy, stage=1,
                    seed=seed, deterministic=True, max_frames=12000, checkpoint=checkpoint['path'])
            proxy.commit()
            row['episodes'].append({'seed': seed, 'verified_full_clear': result['verified_full_clear'],
                                     'kills': result['episode']['kills'], 'outcome': result['episode']['outcome'],
                                     'decision_attempts': result['decision_attempts']})
        meter.flush_success()
        row.update(status='completed', wins=sum(e['verified_full_clear'] for e in row['episodes']),
                   train_wins=sum(e['verified_full_clear'] for e in row['episodes'] if e['seed'] in (0, 1)),
                   mean_kills=float(np.mean([e['kills'] for e in row['episodes']])))
    except BaseException as error:
        row.update(status='partial', error=f'{type(error).__name__}: {error}')
    output.mkdir(parents=True, exist_ok=True)
    machinery._write(output / 'summary.json', row)
    return row


def run_rom_finetune(initial_checkpoint, output, budget, *, steps=65536, seed=0, device='cpu'):
    """Run the fixed 4x256 ROM PPO protocol against an existing shared budget.

    Initial and every16,384-step development checks use normal ROM seeds0/1/7/39.
    Two completed checks with fewer than two training-seed wins stop the run
    (incomplete checks never count). Initial/best checkpoints are retained.
    No automatic retry; unresolved rollout/evaluation/bootstrap reservations
    stay charged. This function creates no budget and extends no deadline.
    """
    if type(steps) is not int or not ROLLOUT <= steps <= 65536 or steps % ROLLOUT:
        raise ValueError('steps must be a multiple of1024 in1024..65536')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Choose an explicit CPU/CUDA device')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format='tank-ai-rom-finetune-run-v1', status='running', stop_reason=None,
                  contract=_contract(), requested_steps=steps, seed=seed, device=str(device),
                  rollouts=[], evaluations=[], checkpoints=[], optimized_steps=0, adam_steps=0,
                  bootstrap_reservations=[], unresolved_reservations=[], train_completed_decisions=0,
                  exact_simulator_resume=False, no_teacher_control=True, best_checkpoint=None)
    runtime, bootstrap, parent = None, None, None
    prefix = 'rom-ft-' + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    machinery._write(output / 'report.json', report)
    try:
        state = budget.snapshot()
        if not 0 < state['max_interactions'] <= 4000000:
            raise ValueError('Research ledger cap must not exceed4M')
        seconds = max(0., min(state['remaining_seconds'], state['deadline_epoch'] - time.time(),
                              GLOBAL_DEADLINE - time.time()))
        monitor = ResourceMonitor(output / 'resources.jsonl', device=str(device).split(':')[0])
        guard = machinery._Guard(time.monotonic() + seconds,
                                 min(state['deadline_epoch'], GLOBAL_DEADLINE), monitor)
        guard.check('admission')
        monitor.check('admission', admission_slots=1, force=True)
        warm = imitation.load_imitation_checkpoint(initial_checkpoint, device='cpu')
        parent = _parent_identity(initial_checkpoint)
        report['parent'], report['sources'] = parent, _sources()
        bootstrap = BootstrapMeter(budget, prefix, guard)
        report['bootstrap_reservations'] = bootstrap.rows
        bootstrap.begin_initial()
        machinery._write(output / 'report.json', report)
        runtime = _Runtime(warm, output, guard, bootstrap, seed=seed, device=device, max_steps=steps)
        del warm
        bootstrap.flush_success()
        initial = _save_checkpoint(runtime, output / 'initial.zip', parent, 'initial')
        report['checkpoints'].append(initial); report['best_checkpoint'] = initial
        machinery._write(output / 'report.json', report)
        baseline = _evaluate(runtime.model, output / 'development-initial', budget, guard, initial, 0)
        report['evaluations'].append(baseline)
        if baseline['status'] != 'completed':
            report.update(status='partial', stop_reason='incomplete_initial_development')
            return report
        selection = DevelopmentSelection(baseline)
        if baseline['train_wins'] < 2:
            report.update(status='partial', stop_reason='initial_two_train_wins_not_reproduced')
            return report
        while runtime.optimized_steps < steps:
            guard.check('rollout_start')
            row = _collect_reserved(runtime, budget, bootstrap, f'{prefix}-rollout-{len(report["rollouts"])}')
            report['rollouts'].append(row)
            report['train_completed_decisions'] += row['completed_decisions']
            if not row['complete']:
                report.update(status='partial', stop_reason='unresolved_or_partial_rollout', error=row['error'])
                break
            runtime.optimize()
            report['optimized_steps'], report['adam_steps'] = runtime.optimized_steps, runtime.adam_steps
            metrics = _json_safe(dict(runtime.model.logger.name_to_value))
            with (output / 'optimizer-metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'optimized_steps': runtime.optimized_steps, **metrics},
                                         allow_nan=False) + '\n')
            runtime.model.logger.dump(runtime.optimized_steps)
            checkpoint = None
            if runtime.optimized_steps % 8192 == 0:
                checkpoint = _save_checkpoint(runtime, output / f'optimized-{runtime.optimized_steps:06d}.zip',
                                               parent, 'periodic')
                report['checkpoints'].append(checkpoint)
            if runtime.optimized_steps % 16384 == 0:
                evaluation = _evaluate(runtime.model, output / f'development-{runtime.optimized_steps:06d}',
                                       budget, guard, checkpoint, len(report['evaluations']))
                report['evaluations'].append(evaluation)
                if evaluation['status'] != 'completed':
                    report.update(status='partial', stop_reason='incomplete_development')
                    break
                better, stop = selection.consider(evaluation)
                if better:
                    report['best_checkpoint'] = checkpoint
                report['below_initial_completed_checks'] = selection.below_initial_checks
                if stop:
                    report.update(status='completed', stop_reason='two_completed_train_win_regressions')
                    break
            machinery._write(output / 'report.json', report)
        if report['status'] == 'running':
            report.update(status='completed', stop_reason='step_cap')
    except (DeadlineReached, BudgetExhausted, ResourcePressureError) as error:
        report.update(status='partial', stop_reason=type(error).__name__, error=str(error))
    except BaseException as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if runtime is not None:
            try:
                report['optimized_steps'], report['adam_steps'] = runtime.optimized_steps, runtime.adam_steps
                final = _save_checkpoint(runtime, output / 'final.zip', parent, 'final_committed')
                report['checkpoints'].append(final)
            except Exception as error:
                report['checkpoint_error'] = f'{type(error).__name__}: {error}'
            try:
                runtime.close()
            except Exception as error:
                report['cleanup_error'] = f'{type(error).__name__}: {error}'
        rows = list(report['rollouts']) + list(report['bootstrap_reservations'])
        for evaluation in report['evaluations']:
            rows.extend(evaluation['reservations']); rows.extend(evaluation['bootstrap_reservations'])
        report['unresolved_reservations'] = [row['reservation'] for row in rows if not row['settled']]
        report['budget_snapshot'] = budget.snapshot()
        machinery._write(output / 'report.json', report)
        machinery._write(output / 'best.json', report['best_checkpoint'])
    return report
