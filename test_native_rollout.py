"""Bit-exact native/CUDA storage against the current seeded engine."""
import unittest

from gymnasium import spaces
import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

from rust_env import NativeBatch, stage_ram
from native_codec import NativeRunCodec
from native_rollout import NativeGPURunRolloutBuffer, NativeRunRolloutBuffer


class NativeCodecTests(unittest.TestCase):
    def test_payloads_preserve_sparse_arbitrary_and_unaligned_float_bits(self):
        codec = NativeRunCodec()
        sparse = np.zeros((47, 26, 26), np.float32)
        sparse.view(np.uint32)[0, 0, :6] = [0x80000000, 0x7fc00001, 0x7fa12345,
                                         0xff800000, 0x3f800000, 0]
        noise = np.random.default_rng(33).integers(0, 2**32, sparse.shape, dtype=np.uint32).view(np.float32)
        unaligned = np.ndarray(sparse.shape, np.float32, buffer=bytearray(sparse.nbytes + 1), offset=1)
        unaligned[:] = sparse
        first = codec.encode(sparse)
        for frame in (sparse, noise, unaligned, sparse[:, :, ::-1]):
            payload = codec.encode(frame)
            self.assertLessEqual(len(payload), frame.nbytes + 1)
            self.assertEqual(codec.decode(payload).tobytes(), frame.tobytes())
        self.assertEqual(codec.decode(first).tobytes(), sparse.tobytes())
        self.assertEqual(codec.encode(noise)[0], 0)
        self.assertEqual(first[0], 1)

    def test_bad_inputs_fail_explicitly(self):
        for shape in ((0, 26, 26), (-1,), (1000001,), (1.5,)):
            with self.assertRaises(ValueError):
                NativeRunCodec(shape)
        codec = NativeRunCodec()
        for payload in (b'', b'\x02', b'\x00', b'\x01' + b'\0' * 8):
            with self.assertRaises(ValueError):
                codec.decode(payload)
        with self.assertRaises(ValueError):
            codec.encode(np.zeros((47, 26, 26), np.float64))


class NativeRolloutTests(unittest.TestCase):
    def compare_samples(self, device):
        kwargs = dict(buffer_size=3, observation_space=spaces.Box(-1, 1, (94, 26, 26), np.float32),
                      action_space=spaces.Discrete(10), n_envs=2, device=device, gamma=.99, gae_lambda=.95)
        dense = MaskableRolloutBuffer(**kwargs)
        cls = NativeGPURunRolloutBuffer if device == 'cuda' else NativeRunRolloutBuffer
        compact = cls(**kwargs)
        batch = NativeBatch(np.stack([stage_ram(stage) for stage in (1, 35)]), 120, seeds=[5, 8])
        try:
            frames = batch.observe().copy()
            previous = np.zeros_like(frames)
            for step in range(3):
                obs = np.concatenate((previous, frames), axis=1)
                obs.view(np.uint32)[0, 0, 0, :4] = [0x80000000, 0x7fc00001, 0x7fa12345, 0xff800000]
                args = (obs, np.array([5, 7]), np.array([.5, -.2], np.float32),
                        np.array([step == 0, step == 0]), torch.tensor([.1, .2], device=device),
                        torch.tensor([-.2, -.3], device=device))
                for buffer in (dense, compact):
                    buffer.add(*args, action_masks=batch.masks().copy())
                previous = frames.copy()
                frames = batch.step(np.array([5, 7]))[0].copy()
            for buffer in (dense, compact):
                buffer.compute_returns_and_advantage(torch.tensor([.2, .1], device=device),
                                                     np.array([False, False]))
            np.random.seed(15)
            expected = list(dense.get(4))
            np.random.seed(15)
            actual = list(compact.get(4))
            for left, right in zip(expected, actual):
                for a, b in zip(left, right):
                    self.assertEqual(a.cpu().numpy().tobytes(), b.cpu().numpy().tobytes())
            if device == 'cuda':
                self.assertEqual(compact.gpu_decoded_minibatches, 2)
            self.assertLess(compact.total_storage_nbytes, compact.dense_equivalent_nbytes)
        finally:
            batch.close()

    def test_current_seeded_cpu_samples_and_gae_are_bit_exact(self):
        self.compare_samples('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_current_seeded_cuda_samples_and_gae_are_bit_exact(self):
        self.compare_samples('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_raw_fallback_preserves_arbitrary_bits(self):
        buffer = NativeGPURunRolloutBuffer(
            1, spaces.Box(-1, 1, (47, 26, 26), np.float32), spaces.Discrete(10), device='cuda')
        obs = np.random.default_rng(34).integers(0, 2**32, (1, 47, 26, 26), dtype=np.uint32).view(np.float32)
        buffer.add(obs, np.array([0]), np.array([0]), np.array([True]),
                   torch.zeros(1), torch.zeros(1), action_masks=np.ones((1, 10), bool))
        actual = next(buffer.get(1)).observations.cpu().numpy()
        self.assertEqual(actual.tobytes(), obs.tobytes())
        self.assertEqual(buffer.gpu_decoded_minibatches, 0)


if __name__ == '__main__':
    unittest.main()
