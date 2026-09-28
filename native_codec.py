"""Lossless float32 bit-run storage using the current native library."""
import ctypes as ct
import math

import numpy as np

from rust_env import F32P, U8P, native_library


class NativeRunCodec:
    def __init__(self, shape=(47, 26, 26)):
        self.shape = tuple(shape)
        if not self.shape or any(isinstance(n, (bool, np.bool_))
                                 or not isinstance(n, (int, np.integer)) or n <= 0
                                 for n in self.shape):
            raise ValueError('Codec shape must contain positive integer dimensions')
        self.count = math.prod(map(int, self.shape))
        if self.count > 1_000_000:
            raise ValueError('Codec frames are limited to 1,000,000 float32 values')
        self.lib = native_library()
        self.lib.mirror_encode_f32_runs.argtypes = [F32P, ct.c_size_t, U8P, ct.c_size_t]
        self.lib.mirror_encode_f32_runs.restype = ct.c_ssize_t
        self.lib.mirror_decode_f32_runs.argtypes = [U8P, ct.c_size_t, F32P, ct.c_size_t]
        self.lib.mirror_decode_f32_runs.restype = ct.c_int32
        self.scratch = np.empty(self.count * 4 + 1, np.uint8)

    def encode(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.dtype(np.float32) \
                or frame.shape != self.shape:
            raise ValueError('Expected a native float32 frame with the configured shape')
        frame = np.require(frame, requirements=['C', 'A'])
        length = self.lib.mirror_encode_f32_runs(
            frame.ctypes.data_as(F32P), self.count, self.scratch.ctypes.data_as(U8P), self.scratch.size)
        if length < 0:
            raise ValueError(f'Native codec encode failed: {length}')
        return self.scratch[:length].tobytes()

    def decode(self, payload):
        if not isinstance(payload, bytes) or not payload:
            raise ValueError('Expected nonempty encoded bytes')
        source = np.frombuffer(payload, np.uint8)
        output = np.empty(self.shape, np.float32)
        code = self.lib.mirror_decode_f32_runs(
            source.ctypes.data_as(U8P), source.size, output.ctypes.data_as(F32P), self.count)
        if code:
            raise ValueError(f'Invalid native codec payload: {code}')
        return output
