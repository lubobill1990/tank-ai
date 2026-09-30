"""Authenticated DEV-only entity-policy evaluation, normal or recovery stress.

The entity wrapper already owns both precision frames: predict receives its
dict directly. No FrameStack, teacher control, RAM edits, or FINAL opening is
introduced. Failed/partial evaluations keep their reservation unresolved.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from recordings import BestRecordings
from rust_resources import ResourceMonitor, ResourcePressureError
from rust_validation import PrimitiveActions, _evaluation_state, _seed_episode, _primitive
from learning_lab.budget import BudgetExhausted
from within_stage.demonstrations import SPLIT_SHA256
from within_stage.entity_observation import EntityObservationEnv, ENTITY_VERSION
from within_stage.entity_policy import load_entity_checkpoint, _validate_model, state_sha256, file_sha256
from within_stage.execution_noise import RecoveryStressEnv
from within_stage.observation import OBSERVATION_VERSION
from within_stage.opening_pool import load_pool, make_opening_env
from within_stage.openings import ProbeAccounting, ProbeStop


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT_SHA256 = SPLIT_SHA256


def _write(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\entity_evaluation.py', 'within_stage\\entity_observation.py',
        'within_stage\\entity_policy.py', 'within_stage\\observation.py',
        'within_stage\\opening_pool.py', 'within_stage\\early_openings.py',
        'within_stage\\openings.py', 'within_stage\\execution_noise.py',
        'game_state.py', 'battle_city_env.py', 'rust_validation.py', 'battle_city.nes')}


def _request(split_path, config_index, max_frames):
    if file_sha256(split_path) != FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen split hash mismatch')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    dev = split['development']
    if type(config_index) is not int or config_index not in dev['all_config_indices'] \
            or split['stage'] != 1 or split['post_start_frames'] != 150 \
            or split['observation_version'] != OBSERVATION_VERSION \
            or len(dev['family_ids']) != 12 or len(set(dev['family_ids'])) != 12:
        raise ValueError('Only authenticated twelve-family DEV aliases are accepted; TRAIN/FINAL excluded')
    if type(max_frames) is not int or not 1 <= max_frames <= min(split['max_frames'], 12000):
        raise ValueError('Invalid evaluation frame cap')
    pool_path = Path(split['pool_path'])
    if file_sha256(pool_path) != split['pool_sha256']:
        raise ValueError('Frozen opening-pool hash mismatch')
    pool = load_pool(pool_path)
    spec = next((row for row in pool['specs'] if row['config_index'] == config_index), None)
    if spec is None or spec['family_id'] not in dev['family_ids']:
        raise ValueError('Requested configuration/family is not in DEV')
    return split, pool, spec


def evaluate_entity_development_case(model, checkpoint, split_path, config_index, output, budget,
                                     intervention_decision=None, noise_seed=0, max_frames=12000,
                                     *, record_video=False):
    """Return a persisted result for one DEV case; no extra history stacking.

    output is a fresh JSON file; actions/resource/failure files are adjacent.
    Full-game claims require12000-frame capacity and terminal base/enemy proof.
    Stress successes are separate from ordinary_model_clear. No failure retry.
    """
    checkpoint, split_path, output = map(lambda value: Path(value).resolve(),
                                         (checkpoint, split_path, output))
    if output.exists() or output.with_suffix('.actions.jsonl').exists() \
            or output.with_suffix('.resources.jsonl').exists():
        raise FileExistsError(output)
    split, pool, spec = _request(split_path, config_index, max_frames)
    if intervention_decision is not None and (
            type(intervention_decision) is not int or intervention_decision < 0):
        raise ValueError('Intervention index must be nonnegative or None')
    if type(noise_seed) is not int or not 0 <= noise_seed < 2**64 or type(record_video) is not bool:
        raise ValueError('Use uint64 noise_seed and boolean record_video')
    _validate_model(model)
    policy_hash = state_sha256(model.policy.state_dict())
    # SB3 loading reconstructs modules and reseeds global RNGs even for inference.
    with _evaluation_state(model):
        verified = load_entity_checkpoint(checkpoint, device='cpu')
        if state_sha256(verified.policy.state_dict()) != policy_hash:
            raise ValueError('Passed entity policy does not match the strict checkpoint')
        del verified
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'role': 'within-stage-entity-development-not-final', 'status': 'running',
        'config_index': config_index, 'family_id': spec['family_id'], 'opening_spec': spec['spec'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
        'policy_sha256': policy_hash, 'split_sha256': FROZEN_SPLIT_SHA256, 'pool_sha256': split['pool_sha256'],
        'sources': _sources(), 'observation_version': ENTITY_VERSION,
        'history': 'EntityObservationEnv owns current/previous55; no extra stack',
        'max_frames': max_frames, 'smoke_only': max_frames < 12000, 'deterministic': True,
        'intervention_decision': intervention_decision, 'noise_seed': noise_seed,
        'ordinary_model_clear': False, 'full_game_clear': False, 'recovery_success': False,
        'actions': [], 'executed_actions': [], 'episode': None, 'reservation': None, 'settled': False}
    _write(output, result)
    raw = env = stress = recorder = account = None
    try:
        snapshot = budget.snapshot()
        remaining = min(float(snapshot['remaining_seconds']),
                        float(snapshot['deadline_epoch']) - time.time())
        if remaining <= 0:
            raise ProbeStop('Global evaluation deadline exhausted')
        deadline = time.monotonic() + remaining
        allowance = max_frames + 2048 + int(spec['spec']['title_wait']) + int(spec['spec']['stage_wait'])
        token = budget.reserve('entity-dev-' + hashlib.sha256(str(output).encode()).hexdigest()[:24], allowance)
        result['reservation'], result['reserved'] = token, allowance
        monitor = ResourceMonitor(output.with_suffix('.resources.jsonl'),
                                  device=str(model.device).split(':')[0])
        account = ProbeAccounting(allowance, deadline=deadline, monitor=monitor)
        monitor.check('entity_dev_admission', force=True)
        with _evaluation_state(model):
            raw = make_opening_env(pool, config_index, account=account, max_frames=max_frames)
            env = EntityObservationEnv(PrimitiveActions(raw, deadline=deadline))
            if intervention_decision is not None:
                stress = RecoveryStressEnv(env, [intervention_decision], noise_seed=noise_seed,
                                            allowed_perturbation_actions=(0, 1, 2, 3, 4), deadline=deadline)
                env = stress
            if record_video:
                recorder = BestRecordings(output.parent / 'videos', fps=15, scale=3, limit=10)
            _seed_episode(0)
            observation, _ = env.reset(seed=0)
            result['initial_ram_sha256'] = hashlib.sha256(raw.nes.ram.tobytes()).hexdigest()
            if result['initial_ram_sha256'] != spec['initial_ram_sha256']:
                raise ValueError('Initial RAM did not reproduce the frozen DEV opening')
            if recorder is not None:
                recorder.begin(raw.render(), {key: result[key] for key in (
                    'role', 'checkpoint_sha256', 'family_id', 'intervention_decision')})
            total_reward = 0.
            with output.with_suffix('.actions.jsonl').open('x', encoding='utf-8') as log:
                for decision in range((max_frames + 3) // 4):
                    account.check()
                    if account.attempted >= allowance:
                        raise ProbeStop('Entity DEV allowance exhausted')
                    masks = env.get_wrapper_attr('action_masks')()
                    action = _primitive(model.predict(observation, deterministic=True, action_masks=masks)[0])
                    if not masks[action]:
                        raise ValueError('Entity model selected an illegal primitive')
                    result['actions'].append(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    event = info.get('execution_noise')
                    executed = event['executed_action'] if event is not None else action
                    result['executed_actions'].append(executed)
                    result['episode'] = {key: info[key] for key in (
                        'outcome', 'objective', 'stage', 'frames', 'kills', 'deaths') if key in info}
                    result['episode'].update(decisions=decision + 1, reward=total_reward)
                    log.write(json.dumps({'decision_index': decision, 'intended': action,
                                          'executed': executed, 'reward': float(reward),
                                          'info': info}, allow_nan=False) + '\n')
                    if recorder is not None:
                        recorder.write(raw.render())
                    if (decision + 1) % 128 == 0:
                        log.flush()
                        _write(output, {**result, 'accounting': account.snapshot()})
                    if terminated or truncated:
                        if info['outcome'] not in ('clear', 'base_lost', 'lives_exhausted', 'timeout'):
                            raise ValueError('Unexpected full-game terminal outcome')
                        clear = info['outcome'] == 'clear'
                        if clear and (raw.state.remaining_enemies != 0 or raw.state.base_destroyed):
                            raise ValueError('Clear lacks terminal ROM-state proof')
                        full_clear = clear and max_frames == 12000
                        result.update(status='completed', full_game_clear=full_clear,
                                      remaining_enemies=int(raw.state.remaining_enemies),
                                      base_destroyed=bool(raw.state.base_destroyed),
                                      ordinary_model_clear=full_clear and stress is None)
                        if stress is not None:
                            result['stress'] = asdict(stress.last_episode)
                            result['recovery_success'] = full_clear and stress.last_episode.recovery_success
                        if recorder is not None:
                            video = recorder.finish(info)
                            result['video'] = str(video) if video else None
                        break
                else:
                    raise RuntimeError('Entity episode exceeded its declared decision/frame cap')
        if state_sha256(model.policy.state_dict()) != policy_hash:
            raise ValueError('Entity policy parameters changed during evaluation')
    except (ProbeStop, BudgetExhausted, TimeoutError, ResourcePressureError) as error:
        result.update(status='partial', error=f'{type(error).__name__}: {error}')
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            if recorder is not None:
                recorder.close()
            if env is not None:
                env.close()
            elif raw is not None:
                raw.close()
        except Exception as error:
            result.update(status='failed', cleanup_error=str(error))
        if stress is not None and stress.last_episode is not None:
            result['stress'] = asdict(stress.last_episode)
        result['policy_parameters_unchanged'] = state_sha256(model.policy.state_dict()) == policy_hash
        result['sources_unchanged'] = _sources() == result['sources']
        if not result['policy_parameters_unchanged'] or not result['sources_unchanged']:
            result.update(status='failed', error='Policy/source identity changed')
        result['accounting'] = account.snapshot() if account is not None else {'attempted': 0, 'completed': 0}
        if result['status'] == 'completed':
            budget.settle(result['reservation'], result['accounting']['attempted'])
            result['settled'] = True
        else:
            result['ordinary_model_clear'] = result['recovery_success'] = result['full_game_clear'] = False
        _write(output, result)
    return result
