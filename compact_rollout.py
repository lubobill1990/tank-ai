"""Model-independent lossless observation storage for MaskablePPO.

This changes storage only, not policy inputs or PPO arithmetic. Frames use
zlib level 1 with an uncompressed fallback, or opt-in raw storage; float32 bit
patterns (including signed zeros and NaN payloads) are preserved. This is not a
checkpoint interchange format.
"""

import math
import sys
import zlib
from collections.abc import Generator

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer, MaskableRolloutBufferSamples
from stable_baselines3.common.buffers import BaseBuffer


def encode_frame(frame: np.ndarray, codec: str = "zlib") -> bytes:
    """Encode native float32 without any floating-point arithmetic."""
    if codec not in ("zlib", "raw"):
        raise ValueError("codec must be 'zlib' or 'raw'")
    if not isinstance(frame, np.ndarray) or frame.dtype != np.dtype(np.float32):
        raise TypeError("Frames must be native float32 ndarrays")
    raw = frame.tobytes(order="C")
    if codec == "raw":
        return b"r" + raw
    compressed = zlib.compress(raw, level=1)
    return b"z" + compressed if len(compressed) < len(raw) else b"r" + raw


def decode_frame(payload: bytes, shape: tuple[int, ...] = (47, 26, 26)) -> np.ndarray:
    """Return a read-only frame view backed by its decoded bytes."""
    expected = math.prod(shape) * np.dtype(np.float32).itemsize
    if not payload or payload[:1] not in (b"r", b"z"):
        raise ValueError("Invalid frame encoding")
    try:
        raw = zlib.decompress(payload[1:]) if payload[:1] == b"z" else payload[1:]
    except zlib.error as exc:
        raise ValueError("Invalid compressed frame") from exc
    if len(raw) != expected:
        raise ValueError(f"Decoded frame has {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape)


class CompactMaskableRolloutBuffer(MaskableRolloutBuffer):
    """Drop-in *training* buffer for chronological channel-first frame stacks.

    Pass via ``MaskablePPO(rollout_buffer_class=CompactMaskableRolloutBuffer,
    rollout_buffer_kwargs={"max_storage_bytes": ...}, ...)``. Only native float32
    Box observations shaped ``(n_stack * channels_per_frame, 26, 26)`` and binary
    masks are supported. There is deliberately no dense ``observations`` array.
    ``codec="zlib"`` is the default; ``codec="raw"`` skips compression while
    retaining overlap deduplication, for memory/throughput comparisons.
    Subclasses may override ``_encode_frame``/``_decode_frame`` together for a
    different lossless bytes-to-native-float32 frame codec.

    Identical overlapping frames reuse IDs only within the same environment and
    adjacent rollout steps. A uint32 cache of the previous stack's last
    ``n_stack - 1`` frames makes that comparison bit-exact without decompression.
    It costs one overlap-sized observation batch, not a rollout. Reset discards
    all frame IDs/history; the first step always stores its complete stack.

    ``storage_nbytes`` includes compressed/raw payloads and every owned ndarray,
    including the overlap cache. ``python_overhead_nbytes`` adds frame-list slots,
    bytes/ndarray headers and an 8 KiB fixed estimate for instance/config objects.
    ``max_storage_bytes`` caps their sum (2 GiB by default, ``None`` opts out),
    checked before persistent allocation or committing each add. It is not an
    RSS cap: allocator fragmentation, zlib work
    space, incoming observations, add staging, minibatch tensors and permutation
    indices are transient and excluded. Minibatch decoding has no epoch cache.

    Scalar arrays retain upstream dtypes and GAE is inherited unchanged. Sampling
    gathers env-major indices without flattening/copying full-rollout arrays.
    """

    _FIXED_PYTHON_OVERHEAD = 8192

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        channels_per_frame: int = 47,
        max_storage_bytes: int | None = 2 * 1024**3,
        codec: str = "zlib",
    ):
        if codec not in ("zlib", "raw"):
            raise ValueError("codec must be 'zlib' or 'raw'")
        for name, value in (("buffer_size", buffer_size), ("n_envs", n_envs),
                            ("channels_per_frame", channels_per_frame)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_storage_bytes is not None and (
            not isinstance(max_storage_bytes, (int, np.integer)) or max_storage_bytes <= 0
        ):
            raise ValueError("max_storage_bytes must be a positive integer or None")
        if not isinstance(observation_space, spaces.Box) or observation_space.dtype != np.float32:
            raise TypeError("Observations require a native float32 Box")
        shape = observation_space.shape
        if len(shape) != 3 or shape[1:] != (26, 26) or shape[0] % channels_per_frame or shape[0] == 0:
            raise ValueError("Observation shape must be (n_stack * channels_per_frame, 26, 26)")
        if isinstance(action_space, spaces.Discrete):
            self.mask_dims = int(action_space.n)
        elif isinstance(action_space, spaces.MultiDiscrete):
            if action_space.nvec.ndim != 1:
                raise ValueError("MultiDiscrete must have a one-dimensional nvec")
            self.mask_dims = int(sum(action_space.nvec))
        elif isinstance(action_space, spaces.MultiBinary) and isinstance(action_space.n, int):
            self.mask_dims = 2 * action_space.n
        else:
            raise ValueError(f"Unsupported action space {type(action_space)}")

        # RolloutBuffer.__init__/reset allocate dense observations: never call them.
        BaseBuffer.__init__(self, buffer_size, observation_space, action_space, device, n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.channels_per_frame = channels_per_frame
        self.n_stack = shape[0] // channels_per_frame
        self.frame_shape = (channels_per_frame, 26, 26)
        self.max_storage_bytes = max_storage_bytes
        self.codec = codec
        self._max_frames = int(buffer_size) * int(n_envs) * self.n_stack
        if self._max_frames > np.iinfo(np.uint32).max:
            raise ValueError("Frame capacity exceeds uint32 frame IDs")
        scalar_shape = (buffer_size, n_envs)
        self._array_specs = [
            ("actions", (*scalar_shape, self.action_dim), action_space.dtype),
            *((name, scalar_shape, np.dtype(np.float32)) for name in
              ("rewards", "returns", "episode_starts", "values", "log_probs", "advantages")),
            ("action_masks", (*scalar_shape, self.mask_dims), np.dtype(np.bool_)),
            ("_frame_ids", (*scalar_shape, self.n_stack), np.dtype(np.uint32)),
            ("_previous_frames", (n_envs, self.n_stack - 1, *self.frame_shape), np.dtype(np.uint32)),
        ]
        self._array_nbytes = int(sum(math.prod(s) * np.dtype(d).itemsize for _, s, d in self._array_specs))
        array_headers = sum(sys.getsizeof(np.empty((0,) * len(s), dtype=d))
                            for _, s, d in self._array_specs)
        self._base_python_overhead = (
            self._FIXED_PYTHON_OVERHEAD + array_headers + sys.getsizeof([])
            + self._max_frames * np.dtype(np.intp).itemsize
        )
        self._check_capacity(self._array_nbytes + self._base_python_overhead)
        self.reset()

    @property
    def storage_nbytes(self) -> int:
        return self._array_nbytes + self._payload_nbytes

    @property
    def python_overhead_nbytes(self) -> int:
        return self._base_python_overhead + self._frame_count * sys.getsizeof(b"")

    @property
    def total_storage_nbytes(self) -> int:
        return self.storage_nbytes + self.python_overhead_nbytes

    @property
    def dense_equivalent_nbytes(self) -> int:
        entries = int(self.buffer_size) * int(self.n_envs)
        return entries * (
            math.prod(self.obs_shape) * 4 + self.action_dim * self.action_space.dtype.itemsize
            + 6 * np.dtype(np.float32).itemsize + self.mask_dims * np.dtype(np.float32).itemsize
        )

    @property
    def unique_frames(self) -> int:
        return self._frame_count

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        return encode_frame(frame, codec=self.codec)

    def _decode_frame(self, payload: bytes) -> np.ndarray:
        return decode_frame(payload, self.frame_shape)

    def _check_capacity(self, required: int) -> None:
        if self.max_storage_bytes is not None and required > self.max_storage_bytes:
            raise MemoryError(
                f"Compact rollout needs {required:,} accounted bytes; "
                f"max_storage_bytes={self.max_storage_bytes:,}"
            )

    def reset(self) -> None:
        # Drop payload ownership before allocating the next bounded frame table.
        self._frames = None
        self._payload_nbytes = 0
        self._frame_count = 0
        self._frames: list[bytes | None] = [None] * self._max_frames
        for name, shape, dtype in self._array_specs:
            array = getattr(self, name, None)
            if array is None:
                array = np.empty(shape, dtype=dtype)
                setattr(self, name, array)
            array.fill(name == "action_masks")
        self.generator_ready = False
        BaseBuffer.reset(self)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        action_masks: np.ndarray | None = None,
    ) -> None:
        if self.full:
            raise RuntimeError("Rollout is full; reset before adding another observation")
        if not isinstance(obs, np.ndarray) or obs.dtype != np.dtype(np.float32):
            raise TypeError("Observations must be native float32 ndarrays")
        if obs.shape != (self.n_envs, *self.obs_shape):
            raise ValueError(f"Expected observations shaped {(self.n_envs, *self.obs_shape)}, got {obs.shape}")
        if action_masks is not None:
            masks = np.asarray(action_masks).reshape(self.n_envs, self.mask_dims)
            if not np.all((masks == 0) | (masks == 1)):
                raise ValueError("Action masks must contain only binary 0/1 values")
            masks = masks.astype(np.bool_)
        else:
            masks = np.ones((self.n_envs, self.mask_dims), dtype=np.bool_)

        # Validate/stage all scalar conversions before any persistent mutation.
        scalars = {
            "actions": np.asarray(action).reshape(self.n_envs, self.action_dim).astype(self.actions.dtype),
            "rewards": np.broadcast_to(reward, (self.n_envs,)).astype(np.float32),
            "episode_starts": np.broadcast_to(episode_start, (self.n_envs,)).astype(np.float32),
            "values": value.clone().cpu().numpy().reshape(self.n_envs).astype(np.float32),
            "log_probs": log_prob.clone().cpu().numpy().reshape(self.n_envs).astype(np.float32),
            "action_masks": masks,
        }
        frame_bits = obs.view(np.uint32).reshape(self.n_envs, self.n_stack, *self.frame_shape)
        new_ids = np.empty((self.n_envs, self.n_stack), dtype=np.uint32)
        pending: list[bytes] = []
        pending_bytes = 0
        for env in range(self.n_envs):
            for index in range(self.n_stack):
                if self.pos and index < self.n_stack - 1 and np.array_equal(
                    frame_bits[env, index], self._previous_frames[env, index]
                ):
                    new_ids[env, index] = self._frame_ids[self.pos - 1, env, index + 1]
                else:
                    payload = self._encode_frame(frame_bits[env, index].view(np.float32))
                    pending_bytes += len(payload)
                    self._check_capacity(
                        self.total_storage_nbytes + pending_bytes
                        + (len(pending) + 1) * sys.getsizeof(b"")
                    )
                    new_ids[env, index] = self._frame_count + len(pending)
                    pending.append(payload)

        end = self._frame_count + len(pending)
        self._frames[self._frame_count:end] = pending
        self._frame_count = end
        self._payload_nbytes += pending_bytes
        self._frame_ids[self.pos] = new_ids
        np.copyto(self._previous_frames, frame_bits[:, 1:])
        for name, values in scalars.items():
            getattr(self, name)[self.pos] = values
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def get(self, batch_size: int | None = None) -> Generator[MaskableRolloutBufferSamples, None, None]:
        assert self.full, "Rollout must be full before sampling"
        count = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = count
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        indices = np.random.permutation(count)
        self.generator_ready = True
        for start in range(0, count, batch_size):
            yield self._get_samples(indices[start:start + batch_size])

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> MaskableRolloutBufferSamples:
        steps = batch_inds % self.buffer_size
        envs = batch_inds // self.buffer_size
        frame_ids = self._frame_ids[steps, envs]
        observations = np.empty((len(batch_inds), *self.obs_shape), dtype=np.float32)
        frames = observations.reshape(len(batch_inds), self.n_stack, *self.frame_shape)
        # Reuse decoded output slots within this minibatch, not a second frame cache.
        decoded: dict[int, tuple[int, int]] = {}
        for row in range(len(batch_inds)):
            for index in range(self.n_stack):
                frame_id = int(frame_ids[row, index])
                previous = decoded.get(frame_id)
                if previous is None:
                    frames[row, index] = self._decode_frame(self._frames[frame_id])
                    decoded[frame_id] = (row, index)
                else:
                    frames[row, index] = frames[previous]
        data = (
            self.actions[steps, envs],
            self.values[steps, envs].flatten(),
            self.log_probs[steps, envs].flatten(),
            self.advantages[steps, envs].flatten(),
            self.returns[steps, envs].flatten(),
            self.action_masks[steps, envs].astype(np.float32).reshape(-1, self.mask_dims),
        )
        return MaskableRolloutBufferSamples(
            self.to_torch(observations, copy=False), *map(self.to_torch, data)
        )
