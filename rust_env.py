"""Explicit seeded experimental backend; the ROM environment stays independent."""
import ctypes as ct
from functools import lru_cache
import hashlib
from pathlib import Path

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

BACKEND_VERSION = 'rust-seeded-experimental-v1'
RNG_VERSION = 'splitmix64-top-byte-v1'
SEED_VERSION = 'sha256-episode-v1'
CHANNELS = 47
U8P = ct.POINTER(ct.c_uint8)
U32P = ct.POINTER(ct.c_uint32)
U64P = ct.POINTER(ct.c_uint64)
F32P = ct.POINTER(ct.c_float)


def _uint(value, name, bits=64):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or not 0 <= value < 2**bits:
        raise ValueError(f'{name} must be an unsigned {bits}-bit integer')
    return int(value)


def episode_seed(seed, stream, episode):
    values = [_uint(value, name) for value, name in
              ((seed, 'seed'), (stream, 'stream'), (episode, 'episode'))]
    payload = ('tank-ai-episode-v1:' + ':'.join(map(str, values))).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


@lru_cache(maxsize=1)
def native_library():
    path = Path(__file__).parent / 'native' / 'target' / 'release' / 'battle_mirror_probe.dll'
    if not path.is_file():
        raise FileNotFoundError(f'{path}; build with cargo build --release --manifest-path native\\Cargo.toml')
    lib = ct.CDLL(str(path))
    signatures = {
        'mirror_rng_version': ([], ct.c_uint32),
        'mirror_diagnostic_build': ([], ct.c_uint32),
        'mirror_create_seeded': ([U8P, ct.c_size_t, ct.c_uint32, U64P], ct.c_void_p),
        'mirror_reset_seeded': ([ct.c_void_p, ct.c_size_t, ct.c_uint64, F32P], ct.c_int32),
        'mirror_close': ([ct.c_void_p], None),
        'mirror_ram': ([ct.c_void_p, ct.c_size_t, U8P], ct.c_int32),
        'mirror_support': ([ct.c_void_p, ct.c_size_t], ct.c_uint32),
        'mirror_observe': ([ct.c_void_p, F32P, ct.c_size_t], ct.c_int32),
        'mirror_masks': ([ct.c_void_p, U8P, ct.c_size_t], ct.c_int32),
        'mirror_step': ([ct.c_void_p, U8P, ct.c_size_t, F32P, F32P, U8P, U32P], ct.c_int32),
        'mirror_step_batch': (
            [ct.c_void_p, U8P, ct.c_size_t, F32P, F32P, U8P, U32P, U32P, U8P], ct.c_int32),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    if lib.mirror_diagnostic_build():
        raise RuntimeError('Training rejects reference-rng diagnostic builds; rebuild without that feature')
    if lib.mirror_rng_version() != 1:
        raise RuntimeError('Unsupported native RNG version; rebuild the matching backend')
    return lib


def stage_ram(stage, seed=0):
    """Initialize a stage once using the local ROM, never during Rust stepping."""
    from battle_city_env import BattleCityEnv
    _uint(seed, 'seed')
    env = BattleCityEnv(stage=stage, reward_profile='sparse')
    try:
        env.reset(seed=seed)
        for _ in range(2):
            env.nes.step(0)
        return env.nes.ram.copy()
    finally:
        env.close()


class NativeBatch:
    """Reusable native buffers; callers must copy arrays they retain across calls."""
    def __init__(self, initial, max_frames=12000, *, seeds, fast=True):
        initial = np.asarray(initial)
        if initial.dtype != np.uint8 or initial.ndim != 2 or initial.shape[1] != 2048 \
                or not 1 <= len(initial) <= 4096:
            raise ValueError('initial must be uint8 RAM with shape (1..4096, 2048)')
        self.n = len(initial)
        self.max_frames = _uint(max_frames, 'max_frames', 32)
        if not self.max_frames:
            raise ValueError('max_frames must be positive')
        seeds = np.asarray([_uint(value, 'seed') for value in seeds], dtype=np.uint64)
        if seeds.shape != (self.n,):
            raise ValueError('One seed is required per environment')
        initial = np.ascontiguousarray(initial)
        self.lib, self.fast = native_library(), bool(fast)
        self.handle = self.lib.mirror_create_seeded(
            initial.ctypes.data_as(U8P), self.n, self.max_frames, seeds.ctypes.data_as(U64P))
        if not self.handle:
            raise RuntimeError('Rust rejected seeded batch initialization')
        self.obs = np.empty((self.n, CHANNELS, 26, 26), np.float32)
        self.rewards = np.empty(self.n, np.float32)
        self.dones = np.empty(self.n, np.uint8)
        self.stats = np.empty((self.n, 5), np.uint32)
        self.support = np.empty(self.n, np.uint32)
        self.action_buffer = np.empty(self.n, np.uint8)
        self.mask_buffer = np.empty((self.n, 10), np.uint8)
        self._masks_fresh = False
        try:
            self._check_support()
        except RuntimeError:
            self.close()
            raise

    def _open(self):
        if self.handle is None:
            raise RuntimeError('Native batch is closed')

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f'Native ABI call failed: {code}')

    def _check_support(self, *, cached=False):
        if not cached:
            for index in range(self.n):
                self.support[index] = self.lib.mirror_support(self.handle, index)
        if np.any(self.support):
            raise RuntimeError(f'Unsupported native mechanics, flags per environment: {self.support.tolist()}')

    def ram(self, index=0):
        self._open()
        index = _uint(index, 'index')
        if index >= self.n:
            raise ValueError('index is outside this batch')
        ram = np.empty(2048, np.uint8)
        self._check(self.lib.mirror_ram(self.handle, index, ram.ctypes.data_as(U8P)))
        return ram

    def reset(self, index, *, seed):
        self._open()
        index, seed = _uint(index, 'index'), _uint(seed, 'seed')
        if index >= self.n:
            raise ValueError('index is outside this batch')
        self._check(self.lib.mirror_reset_seeded(
            self.handle, index, seed, self.obs[index].ctypes.data_as(F32P)))
        self._masks_fresh = False
        self._check_support()
        return self.obs[index]

    def observe(self):
        self._open()
        self._check(self.lib.mirror_observe(self.handle, self.obs.ctypes.data_as(F32P), self.obs.size))
        return self.obs

    def masks(self):
        self._open()
        if not self._masks_fresh:
            self._check(self.lib.mirror_masks(
                self.handle, self.mask_buffer.ctypes.data_as(U8P), self.mask_buffer.size))
            self._masks_fresh = True
        return self.mask_buffer.view(np.bool_)

    def step(self, actions):
        self._open()
        values = np.asarray(actions)
        if values.shape != (self.n,) or not np.issubdtype(values.dtype, np.integer) \
                or np.any(values < 0) or np.any(values >= 10):
            raise ValueError('Expected one integer primitive action in 0..9 per environment')
        np.copyto(self.action_buffer, values, casting='unsafe')
        args = (self.handle, self.action_buffer.ctypes.data_as(U8P), self.n,
                self.obs.ctypes.data_as(F32P), self.rewards.ctypes.data_as(F32P),
                self.dones.ctypes.data_as(U8P), self.stats.ctypes.data_as(U32P))
        if self.fast:
            self._check(self.lib.mirror_step_batch(
                *args, self.support.ctypes.data_as(U32P), self.mask_buffer.ctypes.data_as(U8P)))
            self._masks_fresh = True
        else:
            self._check(self.lib.mirror_step(*args))
            self._masks_fresh = False
        self._check_support(cached=self.fast)
        return self.obs, self.rewards, self.dones.view(np.bool_), self.stats

    def close(self):
        if self.handle is not None:
            self.lib.mirror_close(self.handle)
            self.handle = None


class RustVectorEnv(VecEnv):
    def __init__(self, initial, max_frames=12000, *, seed=0, fast=True):
        seed = _uint(seed, 'seed')
        n = len(initial)
        self._base_seeds = [seed] * n
        self._episodes = [0] * n
        self.batch = NativeBatch(initial, max_frames,
                                 seeds=[episode_seed(seed, i, 0) for i in range(n)], fast=fast)
        self.render_mode = None
        self._actions = None
        self._returns = np.zeros(n, np.float64)
        self._lengths = np.zeros(n, np.int64)
        super().__init__(n, spaces.Box(-1, 1, (47, 26, 26), np.float32), spaces.Discrete(10))

    def seed(self, seed=None):
        if seed is not None:
            _uint(seed, 'seed')
        values = super().seed(seed)
        for value in values:
            _uint(value, 'seed')
        return values

    def reset(self):
        if any(self._options):
            raise ValueError('Native reset options are not supported')
        self._actions = None
        for index, seed in enumerate(self._seeds):
            if seed is not None:
                self._base_seeds[index] = seed
            self._episodes[index] = 0
            self.batch.reset(index, seed=episode_seed(self._base_seeds[index], index, 0))
        self._returns.fill(0)
        self._lengths.fill(0)
        self._reset_seeds()
        self._reset_options()
        return self.batch.obs.copy()

    def step_async(self, actions):
        if self._actions is not None:
            raise RuntimeError('A native step is already pending')
        self._actions = np.array(actions, copy=True)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError('Call step_async before step_wait')
        actions, self._actions = self._actions, None
        obs, rewards, native_dones, stats = self.batch.step(actions)
        # SB3 retains episode-start flags beyond the following native call.
        dones = native_dones.copy()
        self._returns += rewards
        self._lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        for index in np.flatnonzero(dones):
            index = int(index)
            ram = self.batch.ram(index)
            frame, kills, deaths, shots, pickups = map(int, stats[index])
            if ram[0x68] != 0x80:
                outcome = 'base_lost'
            elif ram[0x51] == 0 or (ram[0x51] <= 1 and 0x10 <= ram[0xa0] < 0x80):
                outcome = 'lives_exhausted'
            elif ram[0x80] == 0:
                outcome = 'clear'
            else:
                outcome = 'timeout'
            infos[index] = {
                'terminal_observation': obs[index].copy(),
                'TimeLimit.truncated': outcome == 'timeout',
                'outcome': outcome, 'frames': frame, 'kills': kills, 'deaths': deaths,
                'shots': shots, 'pickups': pickups, 'lives': int(ram[0x51]),
                'episode_seed': episode_seed(self._base_seeds[index], index, self._episodes[index]),
                'episode': {'r': float(self._returns[index]), 'l': int(self._lengths[index])},
            }
            self._episodes[index] += 1
            self.batch.reset(index, seed=episode_seed(
                self._base_seeds[index], index, self._episodes[index]))
            self._returns[index] = self._lengths[index] = 0
        return obs.copy(), rewards.copy(), dones, infos

    def close(self):
        self.batch.close()

    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        if attr_name == 'render_mode':
            return [None for _ in indices]
        if attr_name == 'action_masks':
            return [lambda index=i: self.batch.masks()[index].copy() for i in indices]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name != 'action_masks' or args or kwargs:
            raise AttributeError(method_name)
        masks = self.batch.masks()
        return [masks[i].copy() for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
