"""One-variable ROM PPO ablation: freeze the BC-trained shared visual extractor.

All ROM/PPO, evaluation, budget, retention and rollback helpers remain those of
rom_finetune. Actor/critic heads and all Adam state are retained. Feature grads
are None before Adam, so momentum and weight decay cannot move frozen weights.
Use this module's strict loader: state_dict alone does not save requires_grad.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from sb3_contrib import MaskablePPO

from learning_lab import imitation, rom_finetune as control


FORMAT = 'tank-ai-rom-frozen-features-checkpoint-v1'
FREEZE_POLICY = {
    'version': 'shared-visual-features-frozen-v1',
    'module_aliases': ['features_extractor', 'pi_features_extractor', 'vf_features_extractor'],
    'requires_grad': False, 'before_adam_gradient': 'None',
    'keep_optimizer_parameters_and_state': True, 'verify_after_adam': 'bit_exact',
    'trainable_modules': ['mlp_extractor.policy_net', 'action_net',
                          'mlp_extractor.value_net', 'value_net'],
}
_FEATURE_PREFIXES = tuple(name + '.' for name in FREEZE_POLICY['module_aliases'])


def _contract():
    return {**control._contract(), 'freeze_policy': deepcopy(FREEZE_POLICY)}


def _sources():
    return {**control._sources(),
            'learning_lab\\rom_frozen_features.py': control.machinery._hash(Path(__file__))}


def feature_state(model):
    return {name: value for name, value in model.policy.state_dict().items()
            if name.startswith(_FEATURE_PREFIXES)}


class FrozenFeatureGuard:
    """Optimizer-local hooks; no parameter removal, global patch, or Adam reset."""
    def __init__(self, model):
        imitation._verify_model(model)
        policy = model.policy
        if not policy.share_features_extractor \
                or policy.features_extractor is not policy.pi_features_extractor \
                or policy.features_extractor is not policy.vf_features_extractor:
            raise ValueError('The controlled variant requires the original shared feature extractor')
        parameters = {}
        for alias in FREEZE_POLICY['module_aliases']:
            for parameter in getattr(policy, alias).parameters():
                parameters[id(parameter)] = parameter
        if not parameters:
            raise ValueError('No feature parameters found')
        self.parameters = tuple(parameters.values())
        ids = set(parameters)
        self.parameter_names = [name for name, parameter in policy.named_parameters() if id(parameter) in ids]
        for parameter in policy.parameters():
            if id(parameter) not in ids and not parameter.requires_grad:
                raise ValueError('Actor/critic head parameters must remain trainable')
        for parameter in self.parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        self.snapshots = [parameter.detach().contiguous().view(torch.uint8).clone()
                          for parameter in self.parameters]
        self.initial_sha256 = control._state_hash(feature_state(model))
        self.pre_hook = model.policy.optimizer.register_step_pre_hook(self._before_step)
        self.post_hook = model.policy.optimizer.register_step_post_hook(self._after_step)
        self.check()

    def check(self):
        for parameter, expected in zip(self.parameters, self.snapshots):
            if parameter.requires_grad or not torch.equal(
                    parameter.detach().contiguous().view(torch.uint8), expected):
                raise RuntimeError('Frozen visual feature contract violated')

    def _before_step(self, optimizer, args, kwargs):
        self.check()
        for parameter in self.parameters:
            parameter.grad = None

    def _after_step(self, optimizer, args, kwargs):
        self.check()

    def close(self):
        self.pre_hook.remove()
        self.post_hook.remove()


class FrozenRuntime(control._Runtime):
    def __init__(self, *args, **kwargs):
        self.freeze_guard = None
        self.frozen_source_pin = _sources()
        super().__init__(*args, **kwargs)
        try:
            self.freeze_guard = FrozenFeatureGuard(self.model)
            if self.start_policy_sha256 != control._state_hash(self.model.policy.state_dict()) \
                    or self.start_adam_sha256 != control._state_hash(self.model.policy.optimizer.state_dict()):
                raise ValueError('Applying freeze changed initial policy/Adam state')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.freeze_guard is not None:
            self.freeze_guard.close()
        super().close()


def _save_checkpoint(runtime, path, parent, role):
    if not runtime.state_verified or runtime.optimized_steps % control.ROLLOUT:
        raise ValueError('Only initial or committed updates can be saved')
    if runtime.frozen_source_pin != _sources():
        raise ValueError('Frozen ablation source identity changed')
    runtime.freeze_guard.check()
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
        'format': FORMAT, 'contract': _contract(), 'freeze_policy': deepcopy(FREEZE_POLICY),
        'checkpoint_file': path.name, 'checkpoint_sha256': control.machinery._hash(path),
        'sources': runtime.frozen_source_pin, 'parent': parent, 'role': role,
        'optimized_steps': runtime.optimized_steps, 'adam_steps': runtime.adam_steps,
        'ppo_epoch_updates': int(runtime.model._n_updates), 'exact_simulator_resume': False,
        'policy_sha256': control._state_hash(runtime.model.policy.state_dict()),
        'adam_sha256': control._state_hash(runtime.model.policy.optimizer.state_dict()),
        'start_policy_sha256': runtime.start_policy_sha256, 'start_adam_sha256': runtime.start_adam_sha256,
        'initial_feature_sha256': runtime.freeze_guard.initial_sha256,
        'feature_sha256': control._state_hash(feature_state(runtime.model)),
        'frozen_parameter_names': runtime.freeze_guard.parameter_names,
        'warning': 'Only visual feature training is disabled; all other ROM fine-tune contracts are unchanged. '
                   'Development results are not unseen-map generalization evidence.',
    }
    control.machinery._write(path.with_suffix('.manifest.json'), manifest)
    return {'path': str(path), 'sha256': manifest['checkpoint_sha256'],
            'optimized_steps': runtime.optimized_steps}


def load_frozen_feature_checkpoint(path, device='cpu'):
    """Strict trusted-local loader; restores full Adam plus local freeze enforcement."""
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        parent = manifest['parent']
        if manifest['format'] != FORMAT or manifest['contract'] != _contract() \
                or manifest['freeze_policy'] != FREEZE_POLICY or manifest['sources'] != _sources() \
                or manifest['freeze_policy']['requires_grad'] is not False \
                or manifest['checkpoint_file'] != path.name or manifest['exact_simulator_resume'] is not False \
                or isinstance(manifest['contract']['ppo']['gamma'], bool) \
                or type(manifest['optimized_steps']) is not int or not 0 <= manifest['optimized_steps'] <= 65536 \
                or manifest['optimized_steps'] % control.ROLLOUT \
                or parent['manifest']['format'] != imitation._FORMAT \
                or parent['manifest']['contract'] != imitation._CONTRACT \
                or parent['manifest']['sources'] != imitation._source_hashes():
            raise ValueError('Invalid frozen-feature checkpoint contract/source/parent')
        for key in ('sha256', 'manifest_sha256'):
            value = parent[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid parent SHA256')
        if parent['sha256'] != parent['manifest']['checkpoint_sha256']:
            raise ValueError('Parent archive identity mismatch')
        imitation._verify_provenance(parent['manifest'])
        if control.machinery._hash(path) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA256 mismatch')
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('Missing or malformed frozen-feature manifest') from error
    model = MaskablePPO.load(path, device=device, custom_objects={
        'rollout_buffer_class': control.NativeGPURunRolloutBuffer,
        'rollout_buffer_kwargs': {'channels_per_frame': 47, 'max_storage_bytes': 2 * 1024**3},
        'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    imitation._verify_model(model)
    if isinstance(model.gamma, bool) or model.gamma != 1. or model.gae_lambda != .95 \
            or model.learning_rate != 5e-5 or model.ent_coef != .001 or model.target_kl != .01 \
            or model.n_epochs != 4 or model.clip_range(1.) != .1 \
            or model.num_timesteps != manifest['optimized_steps'] \
            or model._n_updates != manifest['ppo_epoch_updates'] \
            or control._state_hash(model.policy.state_dict()) != manifest['policy_sha256'] \
            or control._state_hash(model.policy.optimizer.state_dict()) != manifest['adam_sha256']:
        raise ValueError('Serialized PPO/optimizer differs from frozen-feature contract')
    features = control._state_hash(feature_state(model))
    if features != manifest.get('feature_sha256') or features != manifest.get('initial_feature_sha256'):
        raise ValueError('Frozen feature weights changed from the initial checkpoint')
    guard = FrozenFeatureGuard(model)
    if guard.parameter_names != manifest.get('frozen_parameter_names'):
        guard.close()
        raise ValueError('Frozen parameter selection differs from manifest')
    # Optimizer hook bound methods own guard for the loaded model's lifetime.
    return model


strict_load_frozen_feature_checkpoint = load_frozen_feature_checkpoint


def run_frozen_feature_finetune(initial_checkpoint, output, budget, *, steps=65536, seed=0, device='cpu'):
    """Same control orchestration, with only runtime/save/format selected locally.

    A short explicit orchestration copy is necessary because the frozen control
    driver has no runtime/checkpoint injection points. No globals are replaced.
    All substantive PPO, environment, evaluation, reservation, selection and
    rollback helpers are reused unchanged from rom_finetune.
    """
    if type(steps) is not int or not control.ROLLOUT <= steps <= 65536 or steps % control.ROLLOUT:
        raise ValueError('steps must be a multiple of1024 in1024..65536')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Choose an explicit CPU/CUDA device')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format='tank-ai-rom-frozen-features-run-v1', status='running', stop_reason=None,
                  contract=_contract(), freeze_policy=deepcopy(FREEZE_POLICY),
                  requested_steps=steps, seed=seed, device=str(device), rollouts=[], evaluations=[],
                  checkpoints=[], optimized_steps=0, adam_steps=0, bootstrap_reservations=[],
                  unresolved_reservations=[], train_completed_decisions=0, exact_simulator_resume=False,
                  no_teacher_control=True, best_checkpoint=None)
    runtime, bootstrap, parent = None, None, None
    prefix = 'rom-freeze-' + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    control.machinery._write(output / 'report.json', report)
    try:
        state = budget.snapshot()
        if not 0 < state['max_interactions'] <= 4000000:
            raise ValueError('Research ledger cap must not exceed4M')
        seconds = max(0., min(state['remaining_seconds'], state['deadline_epoch'] - time.time(),
                              control.GLOBAL_DEADLINE - time.time()))
        monitor = control.ResourceMonitor(output / 'resources.jsonl', device=str(device).split(':')[0])
        guard = control.machinery._Guard(time.monotonic() + seconds,
                                         min(state['deadline_epoch'], control.GLOBAL_DEADLINE), monitor)
        guard.check('admission')
        monitor.check('admission', admission_slots=1, force=True)
        warm = imitation.load_imitation_checkpoint(initial_checkpoint, device='cpu')
        parent = control._parent_identity(initial_checkpoint)
        report['parent'], report['sources'] = parent, _sources()
        bootstrap = control.BootstrapMeter(budget, prefix, guard)
        report['bootstrap_reservations'] = bootstrap.rows
        bootstrap.begin_initial()
        control.machinery._write(output / 'report.json', report)
        runtime = FrozenRuntime(warm, output, guard, bootstrap, seed=seed, device=device, max_steps=steps)
        del warm
        bootstrap.flush_success()
        report['initial_feature_sha256'] = runtime.freeze_guard.initial_sha256
        initial = _save_checkpoint(runtime, output / 'initial.zip', parent, 'initial')
        report['checkpoints'].append(initial); report['best_checkpoint'] = initial
        control.machinery._write(output / 'report.json', report)
        baseline = control._evaluate(runtime.model, output / 'development-initial', budget, guard, initial, 0)
        report['evaluations'].append(baseline)
        if baseline['status'] != 'completed':
            report.update(status='partial', stop_reason='incomplete_initial_development')
            return report
        selection = control.DevelopmentSelection(baseline)
        if baseline['train_wins'] < 2:
            report.update(status='partial', stop_reason='initial_two_train_wins_not_reproduced')
            return report
        while runtime.optimized_steps < steps:
            guard.check('rollout_start')
            row = control._collect_reserved(runtime, budget, bootstrap,
                                             f'{prefix}-rollout-{len(report["rollouts"])}')
            report['rollouts'].append(row)
            report['train_completed_decisions'] += row['completed_decisions']
            if not row['complete']:
                report.update(status='partial', stop_reason='unresolved_or_partial_rollout', error=row['error'])
                break
            runtime.optimize()
            runtime.freeze_guard.check()
            report['optimized_steps'], report['adam_steps'] = runtime.optimized_steps, runtime.adam_steps
            metrics = control._json_safe(dict(runtime.model.logger.name_to_value))
            with (output / 'optimizer-metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'optimized_steps': runtime.optimized_steps, **metrics},
                                         allow_nan=False) + '\n')
            runtime.model.logger.dump(runtime.optimized_steps)
            checkpoint = None
            if runtime.optimized_steps % 8192 == 0:
                checkpoint = _save_checkpoint(
                    runtime, output / f'optimized-{runtime.optimized_steps:06d}.zip', parent, 'periodic')
                report['checkpoints'].append(checkpoint)
            if runtime.optimized_steps % 16384 == 0:
                evaluation = control._evaluate(
                    runtime.model, output / f'development-{runtime.optimized_steps:06d}',
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
            control.machinery._write(output / 'report.json', report)
        if report['status'] == 'running':
            report.update(status='completed', stop_reason='step_cap')
    except (control.DeadlineReached, control.BudgetExhausted, control.ResourcePressureError) as error:
        report.update(status='partial', stop_reason=type(error).__name__, error=str(error))
    except BaseException as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if runtime is not None:
            try:
                report['optimized_steps'], report['adam_steps'] = runtime.optimized_steps, runtime.adam_steps
                final = _save_checkpoint(runtime, output / 'final.zip', parent, 'final_committed')
                report['checkpoints'].append(final)
                report['final_feature_sha256'] = control._state_hash(feature_state(runtime.model))
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
        control.machinery._write(output / 'report.json', report)
        control.machinery._write(output / 'best.json', report['best_checkpoint'])
    return report
