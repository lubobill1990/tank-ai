"""Exact-ROM snapshot curriculum, separate from normal full-game validation.

Only stage1 training openings0/1/2 may prepare a snapshot. The teacher runs once,
RAM must match the requested SHA256, then NES._backup stores the complete live
machine. Later resets use seed23 ONLY after that verified backup (zero delay).
No RAM is filled or patched. Python counters/time_left restart for each episode.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from battle_city_env import BattleCityEnv
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_validation import PrimitiveActions, _primitive


ROOT = Path(__file__).resolve().parents[1]


def _hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def normal_start_spec(source_seed=0):
    """A normal ROM reset reference, NOT native stage_ram's two-frame-primed RAM."""
    return {'stage': 1, 'source_seed': source_seed, 'source_frame': 0,
            'ram_sha256': None, 'remaining': 20, 'full_start': True}


def start_spec_from_pool(pool, index, *, source_sha256=None):
    """Adapt a curriculum row of load_backplay_pool without touching an emulator."""
    if bool(pool['full_start'][index]):
        raise ValueError('Native full_start includes two raw prime frames; use normal_start_spec for ROM')
    ram = pool['rams'][index]
    if ram.dtype != np.uint8 or ram.shape != (2048,):
        raise ValueError('Pool RAM must be uint8[2048]')
    return {'stage': int(pool['stages'][index]), 'source_seed': int(pool['seeds'][index]),
            'source_frame': int(pool['source_frames'][index]), 'ram_sha256': _hash(ram),
            'remaining': int(pool['remaining'][index]), 'full_start': False,
            'candidate_id': int(pool['candidate_ids'][index]),
            'source_sha256': deepcopy(source_sha256 or {})}


class RomCurriculumEnv(gym.Env):
    """47-plane/10-action Gym environment compatible with Monitor and DummyVecEnv.

    start_spec keys: stage=1, source_seed in0/1/2, source_frame (multiple of4),
    ram_sha256; optional remaining, full_start=False, initial_ram_sha256 and
    source_sha256 (repo-relative source-path -> expected SHA256).

    initialization_counter(event) is called BEFORE each teacher decision and
    each NES.step outside a teacher/learner four-frame action. event contains
    kind, phase, count=1, stage, source_seed. Raising refuses that attempt.
    Internal initialization_counts also track successful completions. Learner
    calls and raw NES-frame calls have separate counters; nothing settles a
    ledger. deadline is absolute time.monotonic().

    Gym reset(seed=...) seeds only wrapper bookkeeping; it never changes the
    fixed source opening. full_start=True instead repeats normal source-seed
    resets, with no teacher and no seed23 use. These remain training starts,
    not formal ROM-win evidence. A failed preparation/restore is not retried.
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    render_mode = 'rgb_array'

    def __init__(self, start_spec, *, max_frames, deadline=None, initialization_counter=None):
        super().__init__()
        self.start_spec = deepcopy(dict(start_spec))
        spec = self.start_spec
        for name in ('stage', 'source_seed', 'source_frame'):
            value = spec.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer')
            spec[name] = int(value)
        if spec['stage'] != 1 or spec['source_seed'] not in (0, 1, 2):
            raise ValueError('Only stage1 training source seeds0/1/2 are allowed')
        self.full_start = spec.get('full_start', False)
        if type(self.full_start) is not bool or not 0 <= spec['source_frame'] < 12000 \
                or spec['source_frame'] % 4 or (self.full_start and spec['source_frame'] != 0):
            raise ValueError('Invalid source frame/full_start contract; native source_frame=-1 is not a ROM reset')
        digest = spec.get('ram_sha256')
        if not (self.full_start and digest is None) and (
                not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('A valid exact RAM SHA256 is required')
        if type(max_frames) is not int or not 1 <= max_frames <= 12000:
            raise ValueError('max_frames must be in1..12000')
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                     or not math.isfinite(deadline)):
            raise ValueError('deadline must be finite absolute monotonic time')
        if initialization_counter is not None and not callable(initialization_counter):
            raise ValueError('initialization_counter must be callable(event)')
        remaining = spec.get('remaining')
        if remaining is not None and (isinstance(remaining, (bool, np.bool_))
                                      or not isinstance(remaining, (int, np.integer))
                                      or not 1 <= remaining <= 20):
            raise ValueError('remaining must be in1..20')
        self._initial_remaining = int(remaining) if remaining is not None else None
        self.max_frames, self.deadline = max_frames, deadline
        self._counter = initialization_counter
        self._raw = self._env = None
        self._prepared = self._closed = self._ready = False
        self._failed = None
        self._phase = 'idle'
        self._baseline_kills = 0
        self._snapshot_ram = self._snapshot_pixels = None
        self._last_resource_check = float('-inf')
        self.resource_snapshot = None
        self._initialization_counts = {f'{name}_{status}': 0 for name in (
            'normal_reset_substeps', 'snapshot_reset_substeps', 'teacher_decisions')
            for status in ('attempted', 'completed')}
        self.learner_counts = {'attempted': 0, 'completed': 0}
        self.nes_frame_counts = {'attempted': 0, 'completed': 0}
        self.source_sha256 = self._verify_sources()

    def _verify_sources(self):
        expected = self.start_spec.get('source_sha256', {})
        if not isinstance(expected, dict):
            raise ValueError('source_sha256 must be a source-hash mapping')
        sources = {}
        names = set(expected) | {
            'battle_city.nes', 'battle_city_env.py', 'game_state.py', 'heuristic.py',
            'learning_lab\\teacher.py', 'learning_lab\\teacher_v2.py',
            'learning_lab\\rom_curriculum_env.py'}
        for name in sorted(names):
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('source hash paths must stay within the repository')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if name in expected and digest != expected[name]:
                raise ValueError(f'Frozen source hash mismatch: {name}')
            sources[name] = digest
        return sources

    def _make_raw_env(self):
        return BattleCityEnv(stage=1, objective='full', max_frames=12000,
                             frame_skip=4, reward_profile='sparse')

    def _make_teacher(self):
        return TeacherControllerV2()

    @property
    def initialization_counts(self):
        return dict(self._initialization_counts)

    @property
    def initial_remaining(self):
        if self._initial_remaining is None:
            raise RuntimeError('Provide remaining in start_spec or reset once before reading initial_remaining')
        return self._initial_remaining

    @property
    def current_kills(self):
        return int(self._raw.state.kills - self._baseline_kills) if self._ready else 0

    def _guard(self):
        if self._closed or self._failed is not None:
            raise RuntimeError(f'ROM curriculum environment is closed/failed: {self._failed}')
        now = time.monotonic()
        if self.deadline is not None and now >= self.deadline:
            raise TimeoutError('ROM curriculum deadline reached')
        if now - self._last_resource_check >= 1:
            from rust_resources import host_snapshot, COMMIT_RESERVE, ResourcePressureError
            try:
                snapshot = host_snapshot()
            except OSError as error:
                snapshot = {'status': 'unavailable', 'reason': str(error)}
            if snapshot.get('commit_available_bytes', COMMIT_RESERVE) < COMMIT_RESERVE:
                raise ResourcePressureError('ROM curriculum host commit reserve exhausted')
            self.resource_snapshot, self._last_resource_check = snapshot, now

    def _charge_initialization(self, kind):
        if self._counter is not None:
            self._counter({'kind': kind, 'phase': self._phase, 'count': 1,
                           'stage': 1, 'source_seed': self.start_spec['source_seed']})
        self._initialization_counts[kind + 's_attempted'] += 1

    def _make_environment(self):
        self._raw = self._make_raw_env()
        self._env = PrimitiveActions(self._raw, deadline=self.deadline)
        checked_step = self._raw.nes.step

        def counted_step(*args, **kwargs):
            self._guard()
            kind = ('normal_reset_substep' if self._phase == 'normal_reset' else
                    'snapshot_reset_substep' if self._phase == 'snapshot_reset' else None)
            if kind:
                self._charge_initialization(kind)
            self.nes_frame_counts['attempted'] += 1
            result = checked_step(*args, **kwargs)
            self.nes_frame_counts['completed'] += 1
            if kind:
                self._initialization_counts[kind + 's_completed'] += 1
            return result

        self._raw.nes.step = counted_step

    def _check_initial_remaining(self):
        remaining = int(self._raw.state.remaining_enemies)
        if not 1 <= remaining <= 20 or self._raw.state.base_destroyed or self._raw.state.game_over:
            raise ValueError('Source is terminal or has invalid remaining enemies')
        if self._initial_remaining is not None and self._initial_remaining != remaining:
            raise ValueError('Source remaining count does not match start_spec')
        self._initial_remaining = remaining

    def _prepare(self):
        self._phase = 'normal_reset'
        observation, info = self._env.reset(seed=self.start_spec['source_seed'])
        initial_hash = self.start_spec.get('initial_ram_sha256')
        if initial_hash is not None and _hash(self._raw.nes.ram) != initial_hash:
            raise ValueError('Normal initial RAM hash mismatch')
        if not self.full_start and self._raw.state.frame < self.start_spec['source_frame']:
            teacher = self._make_teacher()
            teacher.reset()
            while self._raw.state.frame < self.start_spec['source_frame']:
                self._guard()
                self._phase = 'teacher'
                self._charge_initialization('teacher_decision')
                masks = self._env.action_masks()
                action = _primitive(teacher.act(self._raw.state, masks.copy()))
                if not masks[action]:
                    raise ValueError('Initialization teacher selected a masked action')
                observation, _, terminated, truncated, info = self._env.step(action)
                self._initialization_counts['teacher_decisions_completed'] += 1
                if terminated or truncated:
                    raise RuntimeError('Teacher reached terminal before a valid source snapshot')
        if self._raw.state.frame != self.start_spec['source_frame']:
            raise ValueError('Teacher did not reach the exact source frame')
        expected = self.start_spec.get('ram_sha256')
        if expected is not None and _hash(self._raw.nes.ram) != expected:
            raise ValueError('Replayed source RAM SHA256 mismatch; no recovery or RAM patching allowed')
        self._check_initial_remaining()
        self._snapshot_ram = self._raw.nes.ram.copy()
        self._snapshot_pixels = self._raw.nes.screen.copy()
        if not self.full_start:
            self._raw.nes._backup()
        self._raw.max_frames = self.max_frames
        self._prepared = True
        return observation, info

    def _check_restore(self):
        if not np.array_equal(self._raw.nes.ram, self._snapshot_ram) \
                or not np.array_equal(self._raw.nes.screen, self._snapshot_pixels):
            raise RuntimeError('Exact ROM snapshot RAM/pixel restoration failed')
        self._check_initial_remaining()

    def _info(self, original):
        info = dict(original)
        info.update(
            original_kills=int(original.get('kills', self._raw.state.kills)),
            kills=self.current_kills, current_kills=self.current_kills,
            initial_remaining_enemies=self.initial_remaining, curriculum_start=not self.full_start,
            partial_episode=not self.full_start, normal_training_start=self.full_start,
            formal_rom_validation=False, full_rom_win=False, backend='original-ROM-snapshot-curriculum',
            source_seed=self.start_spec['source_seed'], source_frame=self.start_spec['source_frame'],
            source_ram_sha256=_hash(self._snapshot_ram), source_kills_baseline=self._baseline_kills,
            snapshot_restore_seed=None if self.full_start else 23,
            initialization_counts=self.initialization_counts,
        )
        return info

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('ROM curriculum reset options are not supported')
        try:
            self._guard()
            super().reset(seed=seed)
            if self._env is None:
                self._make_environment()
            first = not self._prepared
            if first:
                observation, info = self._prepare()
            if not self.full_start:
                if not self._prepared or not self._raw.nes._has_backup:
                    raise RuntimeError('Snapshot restore forbidden before a validated backup')
                if int(np.random.default_rng(23).integers(0, 24)) != 0:
                    raise RuntimeError('Zero-delay post-snapshot seed contract changed')
                self._phase = 'snapshot_reset'
                observation, info = self._env.reset(seed=23)
                self._check_restore()
            elif not first:
                self._phase = 'normal_reset'
                observation, info = self._env.reset(seed=self.start_spec['source_seed'])
                self._check_restore()
            self._baseline_kills = int(self._raw.state.kills)
            self._ready = True
            return np.array(observation, copy=True), self._info(info)
        except Exception as error:
            self._failed = f'{type(error).__name__}: {error}'
            raise
        finally:
            self._phase = 'idle'

    def action_masks(self):
        self._guard()
        if not self._ready:
            raise RuntimeError('Reset before querying curriculum action masks')
        return self._env.action_masks().copy()

    def step(self, action):
        self._guard()
        if not self._ready:
            raise RuntimeError('Reset before stepping curriculum')
        action = _primitive(action)
        if not self._env.action_masks()[action]:
            raise ValueError('Learner selected a masked primitive action')
        self.learner_counts['attempted'] += 1
        self._phase = 'learner'
        try:
            observation, reward, terminated, truncated, info = self._env.step(action)
            self.learner_counts['completed'] += 1
            return np.array(observation, copy=True), reward, terminated, truncated, self._info(info)
        except Exception as error:
            self._failed = f'{type(error).__name__}: {error}'
            raise
        finally:
            self._phase = 'idle'

    def render(self):
        return self._raw.render()

    def close(self):
        if not self._closed:
            if self._env is not None:
                self._env.close()
            self._closed = True
