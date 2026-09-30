"""CPU development-only actuator/recovery probes; never final qualification.

This module has no ledger writes or inherited experiment deadline. One supplied
counter covers all initialization NES calls and all primitive control attempts.
Gameplay frame-repeat calls are reported separately, never double charged.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from battle_city_env import BattleCityEnv
from rust_resources import ResourceMonitor
from rust_train import _policy_hash
from rust_validation import PrimitiveActions, EpisodePlayer, _evaluation_state, _seed_episode
from within_stage.execution_noise import ExecutionNoiseEnv, RecoveryStressEnv


ROOT = Path(__file__).resolve().parents[1]
OLD_OPENINGS = (0, 1, 7, 39)
ROLE = 'exposed-development-recovery-probe-not-final'
RESERVATION = '353f9292f2f043598e21d0afa133a18f'


def _write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class StopProbe(RuntimeError):
    pass


class ProbeCounter:
    def __init__(self, limit, deadline):
        if type(limit) is not int or not 1 <= limit <= 20000:
            raise ValueError('Probe limit must be in1..20000')
        self.limit, self.deadline = limit, float(deadline)
        self.initialization_attempted = self.initialization_completed = 0
        self.control_attempted = self.control_completed = 0
        self.nes_frames_attempted = self.nes_frames_completed = 0

    @property
    def attempted(self):
        return self.initialization_attempted + self.control_attempted

    @property
    def completed(self):
        return self.initialization_completed + self.control_completed

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise StopProbe('deadline')

    def check(self):
        self.check_time()
        if self.attempted >= self.limit:
            raise StopProbe('decision_budget')

    def take(self, domain):
        self.check()
        name = domain + '_attempted'
        setattr(self, name, getattr(self, name) + 1)

    def complete(self, domain):
        name = domain + '_completed'
        setattr(self, name, getattr(self, name) + 1)

    def report(self):
        return {
            'initialization_attempted': self.initialization_attempted,
            'initialization_completed': self.initialization_completed,
            'control_attempted': self.control_attempted, 'control_completed': self.control_completed,
            'total_attempted': self.attempted, 'total_completed': self.completed,
            'nes_frames_attempted': self.nes_frames_attempted,
            'nes_frames_completed': self.nes_frames_completed, 'limit': self.limit,
            'deadline_monotonic': self.deadline, 'ledger_settled': False,
        }


class MeteredPrimitive(PrimitiveActions):
    def __init__(self, raw, counter):
        super().__init__(raw, deadline=counter.deadline)
        self.counter, self.resetting = counter, False
        checked = raw.nes.step
        def frame(*args, **kwargs):
            counter.check_time()
            if self.resetting:
                counter.take('initialization')
            counter.nes_frames_attempted += 1
            result = checked(*args, **kwargs)
            counter.nes_frames_completed += 1
            if self.resetting:
                counter.complete('initialization')
            return result
        raw.nes.step = frame

    def reset(self, *, seed=None, options=None):
        self.resetting = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.resetting = False

    def step(self, action):
        self.counter.take('control')
        if not self.action_masks()[action]:
            raise ValueError('Probe attempted a masked primitive action')
        result = super().step(action)
        self.counter.complete('control')
        return result


def _normal_rom():
    return BattleCityEnv(stage=1, objective='full', max_frames=12000, frame_skip=4,
                         reward_profile='sparse')


def _snapshot(raw, observation, reward=None, terminated=False, truncated=False):
    return {'ram': raw.nes.ram.copy(), 'pixels': raw.nes.screen.copy(),
            'observation': np.asarray(observation).copy(),
            'reward_dtype': None if reward is None else str(np.asarray(reward).dtype),
            'reward_bits': None if reward is None else np.asarray(reward).tobytes().hex(),
            'terminated': bool(terminated), 'truncated': bool(truncated)}


def check_probability_zero(actions, output, counter, *, env_factory=_normal_rom):
    """Compare owned RAM/pixels/observations and reward bits, counting BOTH traces."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    actions = np.asarray(actions)
    if actions.ndim != 1 or actions.dtype.kind not in 'iu' or not len(actions) \
            or np.any((actions < 0) | (actions >= 10)):
        raise ValueError('Expected a nonempty primitive-action trace')
    result = {'role': ROLE, 'passed': False, 'compared_actions': 0, 'opening_seed': 0,
              'probability': 0., 'error': None, 'status': 'running'}
    baseline = []
    try:
        for wrapped in (False, True):
            counter.check()
            raw = env_factory()
            primitive = MeteredPrimitive(raw, counter)
            env = ExecutionNoiseEnv(primitive, probability=0, deadline=counter.deadline) if wrapped else primitive
            try:
                observation, _ = env.reset(seed=0)
                snapshots = [_snapshot(raw, observation)]
                for action in actions:
                    counter.check()
                    observation, reward, terminated, truncated, _ = env.step(int(action))
                    snapshots.append(_snapshot(raw, observation, reward, terminated, truncated))
                    if terminated or truncated:
                        raise RuntimeError('Recorded parity prefix terminated unexpectedly')
                if not wrapped:
                    baseline = snapshots
                else:
                    for index, (left, right) in enumerate(zip(baseline, snapshots)):
                        for key in ('ram', 'pixels', 'observation'):
                            if left[key].dtype != right[key].dtype or left[key].shape != right[key].shape \
                                    or left[key].tobytes() != right[key].tobytes():
                                np.save(output / f'mismatch-{key}-baseline.npy', left[key], allow_pickle=False)
                                np.save(output / f'mismatch-{key}-wrapped.npy', right[key], allow_pickle=False)
                                raise ValueError(f'p0 parity {key} mismatch at prefix index {index}')
                        for key in ('reward_dtype', 'reward_bits', 'terminated', 'truncated'):
                            if left[key] != right[key]:
                                raise ValueError(f'p0 parity {key} mismatch at prefix index {index}')
                    result.update(passed=True, status='completed', compared_actions=len(actions),
                                  ram_pixels_observations_rewards_bit_exact=True)
            finally:
                env.close()
    except StopProbe as error:
        result.update(status='partial', error=str(error))
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    result['counter'] = counter.report()
    _write(output / 'report.json', result)
    return result


def evaluate_recovery_case(model, *, opening_seed, intervention_decision, noise_seed,
                           output, counter, env_factory=_normal_rom, monitor=None):
    """One deterministic CPU-model stress case; all schedules are exposed development."""
    if opening_seed not in OLD_OPENINGS:
        raise ValueError('This development probe permits only old openings0/1/7/39')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    before = counter.report()
    result = {
        'role': ROLE, 'stage': 1, 'opening_seed': opening_seed,
        'intervention_decision': intervention_decision, 'noise_seed': noise_seed,
        'decision_indices_are_zero_based': True, 'allowed_perturbation_actions': [0, 1, 2, 3, 4],
        'deterministic_policy': True, 'status': 'running', 'outcome': None,
        'full_game_clear': None, 'ordinary_pure_model_win': False,
        'intended_action_counts': [0] * 10, 'executed_action_counts': [0] * 10,
        'decisions': 0, 'kills': None, 'frames': None, 'stress': None, 'error': None}
    _write(output / 'report.json', result)
    raw = env = None
    policy_hash = _policy_hash(model.policy)
    try:
        counter.check()
        raw = env_factory()
        primitive = MeteredPrimitive(raw, counter)
        env = RecoveryStressEnv(primitive, [intervention_decision], noise_seed=noise_seed,
                                stream_id=0, allowed_perturbation_actions=(0, 1, 2, 3, 4),
                                deadline=counter.deadline)
        with _evaluation_state(model):
            _seed_episode(opening_seed)
            player = EpisodePlayer(env, model=model, nstack=2, deterministic=True, deadline=counter.deadline)
            player.reset(opening_seed)
            result['noise_provenance'] = asdict(env.episode_provenance)
            result['initial_ram_sha256'] = hashlib.sha256(raw.nes.ram.tobytes()).hexdigest()
            with (output / 'actions.jsonl').open('x', encoding='utf-8') as trace:
                for index in range(3000):
                    counter.check()
                    if monitor is not None:
                        monitor.check('development_recovery')
                    reward, done, info = player.step()
                    event = info['execution_noise']
                    intended, executed = event['intended_action'], event['executed_action']
                    result['intended_action_counts'][intended] += 1
                    result['executed_action_counts'][executed] += 1
                    result.update(decisions=index + 1, kills=int(info['kills']), frames=int(info['frames']),
                                  outcome=info['outcome'])
                    trace.write(json.dumps({'decision_index': index, 'reward': float(reward),
                                            'outcome': info['outcome'], 'kills': info['kills'],
                                            'frames': info['frames'], 'execution_noise': event}) + '\n')
                    if (index + 1) % 128 == 0:
                        trace.flush(); _write(output / 'progress.json', {**result, 'counter': counter.report()})
                    if done:
                        clear = info['outcome'] == 'clear'
                        if clear and (raw.state.remaining_enemies != 0 or raw.state.base_destroyed):
                            raise ValueError('Clear label lacks actual terminal-state confirmation')
                        result.update(status='completed', full_game_clear=bool(info['full_game_clear']),
                                      terminal_remaining_enemies=int(raw.state.remaining_enemies),
                                      terminal_base_destroyed=bool(raw.state.base_destroyed))
                        break
                else:
                    raise RuntimeError('Full-frame episode exceeded3000 four-frame decisions')
    except (StopProbe, TimeoutError) as error:
        result.update(status='partial', error=f'{type(error).__name__}: {error}')
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if env is not None:
            env.close()
            if env.last_episode is not None:
                result['stress'] = asdict(env.last_episode)
        elif raw is not None:
            raw.close()
        if result['stress'] is None:
            result['stress'] = {'completed': False, 'recovery_rate_valid': False,
                                'requested_interventions': 0, 'applied_interventions': 0,
                                'immediate_terminal_after_intervention': False,
                                'invalid_reason': 'reset_incomplete'}
        result['policy_parameters_unchanged'] = _policy_hash(model.policy) == policy_hash
        if not result['policy_parameters_unchanged']:
            result.update(status='failed', error='Policy parameters changed during CPU evaluation')
        result['attempted_decisions_including_init'] = counter.attempted - (
            before['initialization_attempted'] + before['control_attempted'])
        result['completed_decisions_including_init'] = counter.completed - (
            before['initialization_completed'] + before['control_completed'])
        result['counter'] = counter.report()
        _write(output / 'report.json', result)
    return result


def run_development_probe(checkpoint, recorded_episode, output, *, deadline, limit=20000):
    """Run the predeclared p0 check and eight cases; no ledger settlement or retries."""
    from learning_lab.imitation import load_imitation_checkpoint
    checkpoint, recorded_episode, output = map(Path, (checkpoint, recorded_episode, output))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    counter = ProbeCounter(limit, deadline)
    monitor = ResourceMonitor(output / 'resources.jsonl', device='cpu')
    sources = {str(ROOT / name): _hash(ROOT / name) for name in (
        'within_stage\\execution_noise.py', 'within_stage\\episode_eval.py',
        'battle_city_env.py', 'game_state.py', 'heuristic.py', 'rust_validation.py',
        'learning_lab\\imitation.py', 'rust_models.py', 'policy.py', 'battle_city.nes')}
    plan = [{'case_index': i, 'opening_seed': seed, 'intervention_decision': decision,
             'noise_seed': 701 + i, 'status': 'not_started'}
            for i, (seed, decision) in enumerate((seed, decision)
                for seed in OLD_OPENINGS for decision in (200, 600))]
    report = {'role': ROLE, 'reservation': RESERVATION, 'ledger_settled': False,
              'status': 'running', 'stage': 1, 'checkpoint': str(checkpoint.resolve()),
              'checkpoint_sha256': _hash(checkpoint), 'recorded_episode': str(recorded_episode.resolve()),
              'recorded_episode_sha256': _hash(recorded_episode), 'sources': sources,
              'planned_cases': plan, 'cases': [], 'parity': None, 'real_test_probe_decisions': 0,
              'warning': 'All these old openings and stress schedules are EXPOSED DEVELOPMENT, never future final qualification.'}
    _write(output / 'plan.json', report)
    try:
        counter.check(); monitor.check('admission', admission_slots=1, force=True)
        original = json.loads(recorded_episode.read_text())
        if original['stage'] != 1 or original['seed'] != 0 or not original['deterministic'] \
                or original['checkpoint_sha256'] != report['checkpoint_sha256']:
            raise ValueError('Recorded parity actions do not match the selected model/stage1/seed0')
        model = load_imitation_checkpoint(checkpoint, device='cpu')
        report['policy_sha256'] = _policy_hash(model.policy)
        report['parity'] = check_probability_zero(original['actions'][:32], output / 'probability-zero', counter)
        if not report['parity']['passed']:
            report['status'] = 'partial' if report['parity']['status'] == 'partial' else 'failed'
            return report
        for row in plan:
            counter.check()
            case = evaluate_recovery_case(
                model, opening_seed=row['opening_seed'], intervention_decision=row['intervention_decision'],
                noise_seed=row['noise_seed'], output=output / f"case-{row['case_index']:02d}",
                counter=counter, monitor=monitor)
            row['status'] = case['status']
            report['cases'].append(case)
            _write(output / 'report.json', {**report, 'counter': counter.report()})
            print(json.dumps({key: case[key] for key in (
                'opening_seed', 'intervention_decision', 'noise_seed', 'status',
                'outcome', 'kills', 'frames', 'full_game_clear', 'stress')}), flush=True)
            if case['status'] != 'completed':
                report['status'] = case['status']
                break
        else:
            report['status'] = 'completed'
    except StopProbe as error:
        report.update(status='partial', error=str(error))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        report['counter'] = counter.report()
        report['sources_unchanged'] = sources == {path: _hash(path) for path in sources}
        report['completed_cases'] = sum(case['status'] == 'completed' for case in report['cases'])
        valid = [case for case in report['cases'] if case['status'] == 'completed'
                 and case['stress']['recovery_rate_valid']]
        report['valid_recovery_cases'] = len(valid)
        report['recovery_clears'] = sum(case['stress']['recovery_success'] for case in valid)
        report['immediate_terminal_cases'] = sum(
            case['stress']['immediate_terminal_after_intervention'] for case in report['cases'])
        _write(output / 'report.json', report)
    return report
