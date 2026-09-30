"""Normal-start ROM development episodes with replay evidence, never teacher takeover."""
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
from sb3_contrib import MaskablePPO

from battle_city_env import BattleCityEnv
from learning_lab.evaluation import _CountedPolicy
from recordings import BestRecordings
from rust_learning import _hash, _write_new, runtime_signature
from rust_resources import ResourceMonitor
from rust_train import _policy_hash
from rust_validation import PrimitiveActions, _evaluation_state, play_episode


def play_development_episode(model, *, output, budget, stage=1, seed=0,
                             deterministic=True, max_frames=12000, checkpoint=None,
                             record_video=False):
    if type(stage) is not int or stage not in (1, 2, 3):
        raise ValueError('Only development stages1/2/3 are available here')
    if type(seed) is not int or seed < 0 or int(np.random.default_rng(seed).integers(0, 24)) not in (11, 20, 22, 23):
        raise ValueError('Opening phase is outside the four development phases')
    if type(max_frames) is not int or not 1 <= max_frames <= 12000:
        raise ValueError('Invalid full-game frame cap')
    shape = model.observation_space.shape
    if shape not in ((94, 26, 26), (752, 26, 26)) or model.action_space.n != 10:
        raise ValueError('Expected a canonical ten-primitive learned/baseline policy')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    token = budget.reserve('rom-' + hashlib.sha256(str(output).encode()).hexdigest()[:24], max_frames)
    deadline = time.monotonic() + budget.snapshot()['remaining_seconds']
    counted = _CountedPolicy(model, max_frames, deadline)
    resources = ResourceMonitor(output.with_suffix('.resources.jsonl'),
                                device=str(getattr(model, 'device', 'cpu')).split(':')[0])
    result = {
        'role': 'development-not-heldout-confirmation', 'backend': 'original-ROM',
        'learned_model': isinstance(model, MaskablePPO) or getattr(model, 'learned_model', False),
        'stage': stage, 'seed': seed, 'deterministic': deterministic,
        'max_frames': max_frames, 'smoke_only': max_frames < 12000,
        'checkpoint': str(checkpoint) if checkpoint is not None else None,
        'checkpoint_sha256': _hash(checkpoint) if checkpoint is not None else None,
        'policy_sha256': _policy_hash(model.policy) if hasattr(model, 'policy') else None,
        'decoder': getattr(model, 'decoder_config', {'name': 'ordinary-policy-predict'}),
        'actions': [], 'status': 'running', 'verified_full_clear': False,
    }
    raw = recorder = None

    class ObservedROM(PrimitiveActions):
        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            result['initial_ram_sha256'] = hashlib.sha256(self.unwrapped.nes.ram.tobytes()).hexdigest()
            if recorder is not None:
                recorder.begin(self.unwrapped.render(), {
                    'checkpoint': result['checkpoint'], 'checkpoint_sha256': result['checkpoint_sha256'],
                    'policy_sha256': result['policy_sha256'], 'learned_model': result['learned_model'],
                    'stage': stage, 'seed': seed, 'deterministic': deterministic,
                    'decoder': result['decoder'], 'role': result['role'],
                })
            return observation, info

        def step(self, action):
            if not self.action_masks()[action]:
                raise ValueError('Policy attempted a masked action')
            output_step = super().step(action)
            result['actions'].append(int(action))
            if recorder is not None:
                recorder.write(self.unwrapped.render())
                if output_step[2] or output_step[3]:
                    path = recorder.finish(output_step[4])
                    result['video'] = str(path) if path is not None else None
            return output_step

    try:
        resources.check('rom-start', force=True)
        result['source'] = {'runtime': runtime_signature(), 'rom_player_sha256': _hash(__file__)}
        if record_video:
            recorder = BestRecordings(output.parent / 'videos', fps=15, scale=3, limit=10)
        raw = ObservedROM(BattleCityEnv(stage=stage, objective='full', max_frames=max_frames,
                                       reward_profile='sparse'), deadline=deadline)
        with _evaluation_state(model):
            result['episode'] = play_episode(
                raw, model=counted, nstack=shape[0] // 47, deterministic=deterministic,
                seed=seed, deadline=deadline, resource_check=lambda: resources.check('rom'))
        state = raw.unwrapped.state
        result.update(status='completed', remaining_enemies=state.remaining_enemies,
                      base_destroyed=state.base_destroyed)
        result['verified_full_clear'] = (
            result['episode']['outcome'] == 'clear' and state.remaining_enemies == 0
            and not state.base_destroyed and max_frames == 12000)
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
            budget.settle(token, counted.attempts)
    result['decision_attempts'] = counted.attempts
    _write_new(output, result)
    return result
