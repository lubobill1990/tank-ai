"""Development-family evaluation only; normal and perturbed outcomes stay separate."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import gymnasium as gym
import numpy as np

from recordings import BestRecordings
from rust_learning import _hash, _write_new, runtime_signature
from rust_resources import ResourceMonitor
from rust_train import _policy_hash
from rust_validation import PrimitiveActions, _evaluation_state, play_episode
from within_stage.execution_noise import RecoveryStressEnv
from within_stage.observation import PrecisionObservation, OBSERVATION_VERSION
from within_stage.opening_pool import load_pool, make_opening_env
from within_stage.openings import ProbeAccounting


def validate_development_request(split, config_index, max_frames):
    if (type(config_index) is not int or config_index not in split['development']['all_config_indices']
            or split['stage'] != 1):
        raise ValueError('Only frozen development configurations are accepted; train/final are excluded')
    if type(max_frames) is not int or not 1 <= max_frames <= split['max_frames']:
        raise ValueError('Invalid evaluation frame cap')


class ObservationAdapter:
    def __init__(self, model, *, offsets_enabled=True):
        if model.observation_space.shape != (110, 26, 26) or model.action_space.n != 10:
            raise ValueError('Expected the explicitly migrated precision110 policy')
        if type(offsets_enabled) is not bool:
            raise ValueError('offsets_enabled must be boolean')
        self.model, self.offsets_enabled = model, offsets_enabled

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, observation, **kwargs):
        if self.offsets_enabled:
            return self.model.predict(observation, **kwargs)
        observation = observation.copy()
        observation[..., 47:55, :, :] = 0
        observation[..., 102:110, :, :] = 0
        return self.model.predict(observation, **kwargs)


def evaluate_development_case(model, *, checkpoint, split_path, config_index, output, budget,
                              offsets_enabled=True, intervention_decision=None, noise_seed=0,
                              record_video=False, max_frames=12000):
    split_path, output, checkpoint = Path(split_path), Path(output), Path(checkpoint)
    split = json.loads(split_path.read_text())
    validate_development_request(split, config_index, max_frames)
    if intervention_decision is not None and (
            type(intervention_decision) is not int or intervention_decision < 0):
        raise ValueError('Intervention must be a nonnegative decision index or None')
    if output.exists():
        raise FileExistsError(output)
    model = ObservationAdapter(model, offsets_enabled=offsets_enabled)
    pool_path = Path(split['pool_path'])
    if _hash(pool_path) != split['pool_sha256']:
        raise ValueError('Frozen opening pool hash changed')
    pool = load_pool(pool_path)
    spec = next(row for row in pool['specs'] if row['config_index'] == config_index)
    allowance = max_frames + 2048 + int(spec['spec']['title_wait']) + int(spec['spec']['stage_wait'])
    token = budget.reserve('development-' + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:24],
                           allowance)
    deadline = time.monotonic() + budget.snapshot()['remaining_seconds']
    output.parent.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(output.with_suffix('.resources.jsonl'),
                              device=str(getattr(model, 'device', 'cpu')).split(':')[0])
    account = ProbeAccounting(allowance, deadline=deadline, monitor=monitor)
    policy_sha = _policy_hash(model.policy)
    result = {
        'role': 'within-stage-development-not-final', 'config_index': config_index,
        'family_id': spec['family_id'], 'opening_spec': spec['spec'],
        'split_sha256': _hash(split_path), 'pool_sha256': split['pool_sha256'],
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': _hash(checkpoint),
        'policy_sha256': policy_sha, 'observation_version': OBSERVATION_VERSION,
        'offsets_enabled': offsets_enabled, 'deterministic': True, 'max_frames': max_frames,
        'intervention_decision': intervention_decision, 'noise_seed': noise_seed,
        'status': 'running', 'ordinary_model_clear': False, 'recovery_success': False,
        'reservation': token, 'actions': [],
    }
    raw = recorder = stress = None

    class Observed(gym.Wrapper):
        @property
        def max_frames(self):
            return self.unwrapped.max_frames

        @property
        def state(self):
            return self.unwrapped.state

        def action_masks(self):
            return self.env.get_wrapper_attr('action_masks')()

        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            result['initial_ram_sha256'] = hashlib.sha256(self.unwrapped.nes.ram.tobytes()).hexdigest()
            if result['initial_ram_sha256'] != spec['initial_ram_sha256']:
                raise ValueError('Legal opening did not reproduce its frozen initial state')
            if recorder is not None:
                recorder.begin(self.unwrapped.render(), {
                    key: result[key] for key in ('role', 'checkpoint_sha256', 'family_id',
                                                'offsets_enabled', 'intervention_decision')
                })
            return observation, info

        def step(self, action):
            transition = super().step(action)
            result['actions'].append(int(action))
            if recorder is not None:
                recorder.write(self.unwrapped.render())
                if transition[2] or transition[3]:
                    path = recorder.finish(transition[4])
                    result['video'] = str(path) if path else None
            return transition

    try:
        monitor.check('evaluation-start', force=True)
        result['source'] = {
            'runtime': runtime_signature(),
            'within_stage': {name: _hash(Path(__file__).parent / name) for name in
                             ('evaluation.py', 'observation.py', 'execution_noise.py', 'opening_pool.py')},
        }
        raw = make_opening_env(pool, config_index, account=account, max_frames=max_frames)
        raw = PrecisionObservation(PrimitiveActions(raw, deadline=deadline))
        if intervention_decision is not None:
            stress = RecoveryStressEnv(
                raw, (intervention_decision,), noise_seed=noise_seed,
                allowed_perturbation_actions=(0, 1, 2, 3, 4), deadline=deadline)
            raw = stress
        if record_video:
            recorder = BestRecordings(output.parent / 'videos', fps=15, scale=3, limit=10)
        raw = Observed(raw)
        with _evaluation_state(model):
            result['episode'] = play_episode(
                raw, model=model, nstack=2, deterministic=True, seed=0, deadline=deadline,
                resource_check=lambda: monitor.check('development'))
        if _policy_hash(model.policy) != policy_sha:
            raise ValueError('Evaluation modified policy parameters')
        state = raw.state
        clear = result['episode']['outcome'] == 'clear'
        if clear and (state.remaining_enemies != 0 or state.base_destroyed):
            raise ValueError('Clear lacks terminal game-state evidence')
        result.update(status='completed', full_game_clear=clear,
                      remaining_enemies=state.remaining_enemies, base_destroyed=state.base_destroyed,
                      ordinary_model_clear=clear and stress is None)
        if stress is not None:
            result['stress'] = asdict(stress.last_episode)
            result['recovery_success'] = result['stress']['recovery_success']
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        _write_new(output.with_suffix('.failure.json'), result)
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
            if raw is not None:
                raw.close()
        finally:
            result['accounting'] = account.snapshot()
            budget.settle(token, result['accounting']['attempted'])
    _write_new(output, result)
    return result
