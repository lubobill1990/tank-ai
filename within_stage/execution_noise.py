"""On-policy actuator transition noise and separately labelled recovery stress.

Store the agent's ORIGINAL intended action/log-probability in the learner.
This wrapper changes only the action handed to the environment, not the agent
action or reward. It never uses a teacher, planner, RAM, policy RNG, or reset RNG.

Noise seeds derive only from (noise_seed, stream_id, successful_reset_index).
Assign a distinct stream_id to each vector slot; identical tuples deliberately
reproduce identical streams. Applied counts require a returned step; failed
steps retain attempted intervention counts and are invalid recovery evidence.
reset(seed=..., options=...) forwards those arguments unchanged to the wrapped
environment. Repeating a ROM seed does not restart the noise episode sequence.
Stress schedules are zero-based decision indices, one entry per episode,
cycling in the predeclared order; missed requests are never deferred/retried.
No experiment deadline is hardcoded. Optional deadline is time.monotonic();
guard, if supplied, is a callable(phase), e.g. an existing guard.check method.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from numbers import Real
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np


VERSION = 'within-stage-execution-noise-v1'


def _uint(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or not 0 <= value < 2**64:
        raise ValueError(f'{name} must be an unsigned 64-bit integer')
    return int(value)


def derive_noise_seed(noise_seed, stream_id, episode_index):
    values = [_uint(value, name) for value, name in (
        (noise_seed, 'noise_seed'), (stream_id, 'stream_id'), (episode_index, 'episode_index'))]
    payload = (VERSION + ':' + ':'.join(map(str, values))).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


@dataclass(frozen=True)
class NoiseConfiguration:
    mode: str
    probability: float | None
    noise_seed: int
    stream_id: int
    decision_indices: tuple[int, ...]
    allowed_perturbation_actions: tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True)
class EpisodeProvenance:
    version: str
    mode: str
    episode_index: int
    noise_seed: int
    stream_id: int
    episode_noise_seed: int
    rng: str
    probability: float | None
    decision_indices: tuple[int, ...]
    scheduled_decision_index: int | None
    allowed_perturbation_actions: tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True)
class EpisodeResult:
    provenance: EpisodeProvenance
    attempted_steps: int
    completed_steps: int
    requested_interventions: int
    attempted_interventions: int
    applied_interventions: int
    completed: bool
    end_reason: str
    full_game_clear: bool
    uninterrupted_model_clear: bool
    recovery_rate_valid: bool
    recovery_success: bool
    invalid_reason: str | None
    error: str | None
    immediate_terminal_after_intervention: bool = False


class _ActuatorWrapper(gym.Wrapper):
    def __init__(self, env, configuration, *, deadline, guard):
        if not isinstance(env, gym.Env) or not isinstance(env.action_space, spaces.Discrete) \
                or env.action_space.n != 10 or env.action_space.start != 0:
            raise ValueError('Actuator noise requires Discrete(10), with primitives0..9')
        if env.has_wrapper_attr('_execution_noise_config'):
            raise ValueError('Do not nest actuator-noise/stress wrappers')
        masks = env.get_wrapper_attr('action_masks')
        if not callable(masks):
            raise ValueError('The environment must expose callable action_masks')
        if deadline is not None and (isinstance(deadline, (bool, np.bool_))
                                     or not isinstance(deadline, Real) or not np.isfinite(deadline)):
            raise ValueError('deadline must be a finite absolute monotonic time')
        if guard is not None and not callable(guard):
            raise ValueError('guard must be callable(phase)')
        super().__init__(env)
        self._execution_noise_config = configuration
        self._mask_reader = masks
        self.deadline, self.guard = deadline, guard
        self._episode_index = -1
        self._provenance = self._last_episode = self._noise_rng = None
        self._active = self._closed = False
        self._clear_counts()

    @property
    def configuration(self):
        return self._execution_noise_config

    @property
    def episode_provenance(self):
        return self._provenance

    @property
    def last_episode(self):
        return self._last_episode

    def _clear_counts(self):
        self._attempted = self._completed = self._requested = 0
        self._intervention_attempts = self._applied = 0
        self._immediate_terminal = False

    def _check(self, phase):
        if self._closed:
            raise RuntimeError('Actuator wrapper is closed')
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('Actuator wrapper deadline reached')
        if self.guard is not None:
            self.guard(phase)

    def action_masks(self):
        self._check('execution_noise_masks')
        if not self._active:
            raise RuntimeError('Reset before requesting action masks')
        masks = np.asarray(self._mask_reader())
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError('Expected a nonempty boolean mask for ten primitive actions')
        return masks.copy()

    def _finish(self, *, completed, reason, info=None, error=None):
        info = {} if info is None else info
        full = bool(completed and info.get('outcome') == 'clear'
                    and info.get('objective') == 'full'
                    and not info.get('curriculum_start', False)
                    and not info.get('partial_episode', False)
                    and info.get('terminal_remaining_enemies', 0) == 0
                    and not info.get('terminal_base_destroyed', False))
        stress = self._provenance.mode == 'recovery-stress'
        valid = stress and completed and self._requested == 1 and self._applied == 1
        invalid = None
        if stress and not valid:
            invalid = (reason if not completed else
                       'scheduled_index_not_reached' if not self._requested else 'no_legal_alternative')
        self._last_episode = EpisodeResult(
            self._provenance, self._attempted, self._completed, self._requested,
            self._intervention_attempts, self._applied, completed, reason, full,
            full and not stress and self._applied == 0, valid, valid and full, invalid, error,
            immediate_terminal_after_intervention=self._immediate_terminal)
        self._active = False
        return self._last_episode

    def reset(self, *, seed=None, options=None):
        self._check('execution_noise_reset')
        if self._active:
            self._finish(completed=False, reason='reset_before_termination')
        observation, info = self.env.reset(seed=seed, options=options)
        config = self.configuration
        self._episode_index += 1
        derived = derive_noise_seed(config.noise_seed, config.stream_id, self._episode_index)
        scheduled = (config.decision_indices[self._episode_index % len(config.decision_indices)]
                     if config.mode == 'recovery-stress' else None)
        self._provenance = EpisodeProvenance(
            VERSION, config.mode, self._episode_index, config.noise_seed, config.stream_id,
            derived, f'PCG64/numpy-{np.__version__}', config.probability, config.decision_indices, scheduled,
            allowed_perturbation_actions=config.allowed_perturbation_actions)
        self._noise_rng = np.random.Generator(np.random.PCG64(derived))
        self._clear_counts()
        self._active = True
        info = deepcopy(info)
        info['execution_noise_provenance'] = asdict(self._provenance)
        return deepcopy(observation), info

    def step(self, action):
        self._check('execution_noise_step')
        if not self._active:
            raise RuntimeError('Reset before stepping an ended/uninitialized actuator episode')
        value = np.asarray(action)
        if value.shape != () or value.dtype.kind not in 'iu' or not 0 <= int(value) < 10:
            raise ValueError('Expected one integer primitive action in0..9')
        intended = int(value)
        masks = self.action_masks()
        if not masks[intended]:
            raise ValueError('Intended action must be legal; noise cannot repair an invalid policy action')
        index = self._attempted
        provenance = self._provenance
        alternatives = np.flatnonzero(
            masks & (np.arange(10) != intended)
            & np.isin(np.arange(10), provenance.allowed_perturbation_actions))
        draw = None
        if provenance.mode == 'recovery-stress':
            requested = index == provenance.scheduled_decision_index
        elif provenance.probability == 0:
            requested = False
        else:
            draw = float(self._noise_rng.random())
            requested = draw < provenance.probability
        executed = int(self._noise_rng.choice(alternatives)) if requested and len(alternatives) else intended
        changed = executed != intended
        self._attempted += 1
        self._requested += int(requested)
        self._intervention_attempts += int(changed)
        try:
            observation, reward, terminated, truncated, raw_info = self.env.step(executed)
        except Exception as error:
            self._finish(completed=False, reason='step_error', error=f'{type(error).__name__}: {error}')
            raise
        self._completed += 1
        self._applied += int(changed)
        immediate_terminal = bool(changed and (terminated or truncated))
        self._immediate_terminal = immediate_terminal
        info = deepcopy(raw_info)
        event = {
            **asdict(provenance), 'decision_index': index,
            'intended_action': intended, 'executed_action': executed,
            'legal_mask': tuple(map(bool, masks)), 'legal_alternatives': tuple(map(int, alternatives)),
            'draw': draw, 'requested': bool(requested), 'applied': changed,
            'miss_reason': 'no_legal_alternative' if requested and not changed else None,
            'requested_count': self._requested, 'applied_count': self._applied,
            'agent_action_and_logprob_unchanged': True,
            'transition_noise_not_teacher_replacement': True,
            'immediate_terminal_after_intervention': immediate_terminal,
        }
        info['execution_noise'] = event
        info['immediate_terminal_after_intervention'] = immediate_terminal
        if self._applied or provenance.mode == 'recovery-stress':
            labels = ('full_rom_win', 'verified_full_clear', 'pure_model_win', 'ordinary_pure_model_win',
                      'uninterrupted_model_clear')
            event['original_win_labels'] = {name: deepcopy(info[name]) for name in labels if name in info}
            for name in labels:
                if name in info:
                    info[name] = False
        if terminated or truncated:
            result = self._finish(completed=True, reason='terminated' if terminated else 'truncated',
                                  info=raw_info)
            info['execution_noise_episode'] = asdict(result)
            info['full_game_clear'] = result.full_game_clear
            info['uninterrupted_model_clear'] = result.uninterrupted_model_clear
            info['recovery_rate_valid'] = result.recovery_rate_valid
            info['recovery_success'] = result.recovery_success
        return deepcopy(observation), reward, terminated, truncated, info

    def close(self):
        if not self._closed:
            if self._active:
                self._finish(completed=False, reason='closed_before_termination')
            try:
                self.env.close()
            finally:
                self._closed = True


class ExecutionNoiseEnv(_ActuatorWrapper):
    """Training transition noise. probability=0 makes no draws or action changes."""
    def __init__(self, env, probability=0., *, noise_seed=0, stream_id=0, deadline=None, guard=None):
        if isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real) \
                or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('probability must be a finite number in[0,1]')
        config = NoiseConfiguration('training-actuator-noise', float(probability),
                                    _uint(noise_seed, 'noise_seed'), _uint(stream_id, 'stream_id'), ())
        super().__init__(env, config, deadline=deadline, guard=guard)


class RecoveryStressEnv(_ActuatorWrapper):
    """Exactly one scheduled request per episode; no alternative means invalid stress.

    decision_indices is a nonempty predeclared sequence of zero-based indices.
    Entry episode_index % len(schedule) is used. There is no stochastic training
    noise here, no retry on a later decision, and no ordinary-model-win label.
    Only completed episodes with one actually applied intervention enter a
    recovery-rate denominator; completed clears among those are recoveries.
    Perturbations default to movement/no-op0..4, intersected with the live mask
    and excluding the intended action. Explicit range(10) restores all-action
    stress. This is a fixed support restriction, not a teacher/planner or safety
    lookahead. Immediate terminal cases are kept, counted, and flagged.
    """
    def __init__(self, env, decision_indices, *, allowed_perturbation_actions=(0, 1, 2, 3, 4),
                 noise_seed=0, stream_id=0, deadline=None, guard=None):
        try:
            schedule = tuple(_uint(index, 'decision index') for index in decision_indices)
        except TypeError as error:
            raise ValueError('decision_indices must be a nonempty sequence of integer indices') from error
        if not schedule:
            raise ValueError('decision_indices must not be empty')
        try:
            allowed = tuple(_uint(action, 'perturbation action') for action in allowed_perturbation_actions)
        except TypeError as error:
            raise ValueError('allowed_perturbation_actions must be a nonempty primitive-action sequence') from error
        if not allowed or len(set(allowed)) != len(allowed) or any(action >= 10 for action in allowed):
            raise ValueError('allowed_perturbation_actions must contain distinct actions in0..9')
        config = NoiseConfiguration('recovery-stress', None, _uint(noise_seed, 'noise_seed'),
                                    _uint(stream_id, 'stream_id'), schedule, tuple(sorted(allowed)))
        super().__init__(env, config, deadline=deadline, guard=guard)
