"""Lossless rollout-storage spike: run with ``python -m unittest test_compact_rollout``."""

import os
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

torch.set_num_threads(1)


def observation_space(stack=3):
    return spaces.Box(-np.inf, np.inf, (stack * 47, 26, 26), np.float32)


def frame(tag):
    result = np.zeros((47, 26, 26), np.float32)
    result[0] = np.float32(tag / 127)
    result[1, 3, 5] = np.float32(-1.25)
    result[2, ::2, ::3] = 1
    result[3] = np.float32(0.125)
    return result


def make_buffer(cls, steps=4, envs=2, stack=3, action_space=None, **kwargs):
    return cls(steps, observation_space(stack), action_space or spaces.Discrete(4),
               device="cpu", gamma=0.97, gae_lambda=0.93, n_envs=envs, **kwargs)


def add_step(buffer, observations, step=0, starts=None, masks=None):
    n = buffer.n_envs
    if isinstance(buffer.action_space, spaces.Discrete):
        actions = np.arange(n) % buffer.action_space.n
    else:
        actions = np.zeros((n, buffer.action_dim), buffer.action_space.dtype)
    buffer.add(
        observations, actions,
        np.arange(n, dtype=np.float32) / 3 + step / 7,
        np.zeros(n, np.float32) if starts is None else starts,
        torch.arange(n, dtype=torch.float32) / 5 + step / 11,
        torch.arange(n, dtype=torch.float32) / -13 - step / 17,
        action_masks=masks,
    )


class CodecTests(unittest.TestCase):
    def test_roundtrip_preserves_every_float32_bit(self):
        from compact_rollout import decode_frame, encode_frame

        rng = np.random.default_rng(193)
        bits = rng.integers(0, 2**32, (47, 26, 26), dtype=np.uint32)
        bits[0, 0, :8] = [
            0, 0x80000000, 0x3F800000, 0x7F800000,
            0xFF800000, 0x7FC12345, 0x7F800001, 0xFFC12345,
        ]
        for frame in (bits.view(np.float32), bits.view(np.float32)[:, :, ::-1],
                      np.zeros((47, 26, 26), np.float32)):
            with self.subTest(contiguous=frame.flags.c_contiguous):
                decoded = decode_frame(encode_frame(frame), frame.shape)
                self.assertEqual(decoded.dtype, np.float32)
                self.assertEqual(decoded.tobytes(), frame.tobytes())

    def test_codec_compresses_constant_and_sparse_planes(self):
        from compact_rollout import decode_frame, encode_frame

        frame = np.zeros((47, 26, 26), np.float32)
        frame[1] = np.float32(0.125)
        frame[2, 3, 7] = np.float32(-1.25)
        frame[3, ::2] = 1
        encoded = encode_frame(frame)
        self.assertLess(len(encoded), frame.nbytes // 20)
        self.assertEqual(decode_frame(encoded, frame.shape).tobytes(), frame.tobytes())

    def test_codec_rejects_unsupported_dtype_and_malformed_payload(self):
        from compact_rollout import decode_frame, encode_frame

        for dtype in (np.float64, np.int32, ">f4"):
            with self.assertRaises((TypeError, ValueError)):
                encode_frame(np.zeros((47, 26, 26), dtype=dtype))
        for payload in (b"", b"x", b"rshort", b"zbad"):
            with self.assertRaises(ValueError):
                decode_frame(payload, (47, 26, 26))

    def test_raw_codec_preserves_bits_without_compression(self):
        from compact_rollout import decode_frame, encode_frame

        bits = np.random.default_rng(349).integers(0, 2**32, (47, 26, 26), dtype=np.uint32)
        bits[0, 0, :4] = [0, 0x80000000, 0x7FC12345, 0x7F800001]
        for observations in (bits.view(np.float32), frame(2)):
            payload = encode_frame(observations, codec="raw")
            self.assertEqual(len(payload), observations.nbytes + 1)
            self.assertEqual(decode_frame(payload).tobytes(), observations.tobytes())
        with self.assertRaises(ValueError):
            encode_frame(frame(2), codec="unknown")


class BufferTests(unittest.TestCase):
    def compact_class(self):
        from compact_rollout import CompactMaskableRolloutBuffer
        return CompactMaskableRolloutBuffer

    def assert_samples_equal(self, baseline, compact, batch_size, seed):
        np.random.seed(seed)
        expected = list(baseline.get(batch_size))
        expected_next_random = np.random.random()
        np.random.seed(seed)
        actual = list(compact.get(batch_size))
        self.assertEqual(np.random.random(), expected_next_random)
        self.assertEqual(len(actual), len(expected))
        for expected_sample, actual_sample in zip(expected, actual):
            for name, left, right in zip(expected_sample._fields, expected_sample, actual_sample):
                with self.subTest(field=name):
                    self.assertEqual(left.dtype, right.dtype)
                    self.assertEqual(left.device, right.device)
                    self.assertEqual(left.shape, right.shape)
                    self.assertEqual(left.cpu().numpy().tobytes(), right.cpu().numpy().tobytes())

    def test_stack_overlap_resets_broken_overlap_and_rollout_boundary(self):
        cls = self.compact_class()
        compact = make_buffer(cls, steps=3)
        baseline = make_buffer(MaskableRolloutBuffer, steps=3)
        zero = np.zeros((47, 26, 26), np.float32)
        broken = frame(103)
        broken.view(np.uint32)[4, 0, 0] = 0x80000000
        rollouts = [
            [[frame(1), frame(2), frame(3)], [frame(101), frame(102), frame(103)]],
            [[zero, zero, frame(4)], [frame(102), frame(103), frame(104)]],
            [[zero, frame(4), frame(5)], [broken, frame(104), frame(105)]],
        ]
        for step, stacks in enumerate(rollouts):
            observations = np.array([np.concatenate(stack) for stack in stacks])
            starts = np.array([step == 1, False], np.float32)
            for buffer in (baseline, compact):
                add_step(buffer, observations, step, starts)
        self.assertEqual(compact.unique_frames, 13)
        self.assert_samples_equal(baseline, compact, 4, 891)
        before = compact.storage_nbytes
        for buffer in (baseline, compact):
            buffer.reset()
        self.assertEqual(compact.unique_frames, 0)
        self.assertLess(compact.storage_nbytes, before)
        for step in range(3):
            observations = np.array([
                np.concatenate([frame(300 + env * 20 + step + index) for index in range(3)])
                for env in range(2)
            ])
            for buffer in (baseline, compact):
                add_step(buffer, observations, step)
        self.assertEqual(compact.unique_frames, 10)
        self.assert_samples_equal(baseline, compact, 2, 999)

    def test_gae_and_repeated_minibatches_match_sb3_for_all_maskable_action_spaces(self):
        cls = self.compact_class()
        for stack in (1, 2, 3):
            for action_space in (spaces.Discrete(4), spaces.MultiDiscrete([2, 3]),
                                 spaces.MultiBinary(3)):
                with self.subTest(stack=stack, action_space=action_space):
                    compact = make_buffer(cls, stack=stack, action_space=action_space)
                    baseline = make_buffer(MaskableRolloutBuffer, stack=stack, action_space=action_space)
                    for step in range(4):
                        observations = np.array([
                            np.concatenate([frame(env * 30 + step + index) for index in range(stack)])
                            for env in range(2)
                        ])
                        observations.view(np.uint32)[0, -1, 0, :3] = [
                            0x80000000, 0x7FC12345, 0x7F800001,
                        ]
                        masks = np.ones((2, baseline.mask_dims), np.float32)
                        masks[:, 0] = 0
                        starts = np.array([step == 2, step == 3], np.float32)
                        for buffer in (baseline, compact):
                            add_step(buffer, observations, step, starts, masks)
                    for buffer in (baseline, compact):
                        buffer.compute_returns_and_advantage(
                            torch.tensor([0.17, -0.83]), np.array([False, True]))
                    self.assertEqual(compact.advantages.tobytes(), baseline.advantages.tobytes())
                    self.assertEqual(compact.returns.tobytes(), baseline.returns.tobytes())
                    self.assertEqual(compact.action_masks.dtype, np.bool_)
                    for seed, batch_size in ((11, 3), (12, 1), (13, 5), (14, None)):
                        self.assert_samples_equal(baseline, compact, batch_size, seed)

    def test_only_requested_minibatch_is_decoded(self):
        import compact_rollout
        compact = make_buffer(self.compact_class(), steps=8, envs=1)
        for step in range(8):
            add_step(compact, np.concatenate([frame(step + i) for i in range(3)])[None], step)
        original = compact_rollout.decode_frame
        decoded_count = 0

        def limited_decode(*args, **kwargs):
            nonlocal decoded_count
            decoded_count += 1
            self.assertLessEqual(decoded_count, 6, "decoded more than a two-observation minibatch")
            return original(*args, **kwargs)

        with patch.object(compact_rollout, "decode_frame", limited_decode):
            sample = next(compact.get(2))
        self.assertEqual(sample.observations.shape, (2, 141, 26, 26))
        self.assertGreater(decoded_count, 0)

    def test_constructor_and_reset_never_allocate_dense_rollout_observations(self):
        cls = self.compact_class()
        real_zeros, real_empty, real_ones = np.zeros, np.empty, np.ones
        forbidden = (11, 2, 141, 26, 26)

        def guarded(factory):
            def allocate(shape, *args, **kwargs):
                self.assertNotEqual(tuple(shape) if np.iterable(shape) else (shape,), forbidden)
                return factory(shape, *args, **kwargs)
            return allocate

        with patch.object(np, "zeros", guarded(real_zeros)), \
                patch.object(np, "empty", guarded(real_empty)), \
                patch.object(np, "ones", guarded(real_ones)):
            compact = make_buffer(cls, steps=11)
            for _ in range(3):
                compact.reset()
        self.assertLess(compact.storage_nbytes, np.prod(forbidden) * 4 // 4)

    def test_cap_rejects_before_mutating_and_reset_releases_payload(self):
        cls = self.compact_class()
        observations = np.concatenate([frame(i) for i in range(3)])[None]
        reference = make_buffer(cls, steps=3, envs=1)
        empty_bytes = reference.total_storage_nbytes
        add_step(reference, observations)
        cap = reference.total_storage_nbytes
        compact = make_buffer(cls, steps=3, envs=1, max_storage_bytes=cap)
        add_step(compact, observations)
        self.assertLessEqual(compact.total_storage_nbytes, cap)
        before = {name: value.copy() for name, value in vars(compact).items()
                  if isinstance(value, np.ndarray)}
        with self.assertRaisesRegex(MemoryError, "max_storage_bytes"):
            add_step(compact, np.concatenate([frame(i) for i in (10, 11, 12)])[None], 1)
        self.assertEqual(compact.pos, 1)
        self.assertFalse(compact.full)
        self.assertEqual(compact.unique_frames, 3)
        self.assertEqual(compact.total_storage_nbytes, cap)
        for name, original in before.items():
            self.assertEqual(getattr(compact, name).tobytes(), original.tobytes(), name)
        compact.reset()
        self.assertEqual(compact.total_storage_nbytes, empty_bytes)
        add_step(compact, observations)
        with self.assertRaisesRegex(MemoryError, "max_storage_bytes"):
            make_buffer(cls, max_storage_bytes=1)

    def test_storage_accounts_for_arrays_payloads_and_dense_equivalent(self):
        compact = make_buffer(self.compact_class(), steps=4)
        dense = make_buffer(MaskableRolloutBuffer, steps=4)
        for step in range(4):
            add_step(compact, np.array([
                np.concatenate([frame(step + env * 10 + i) for i in range(3)])
                for env in range(2)
            ]), step)
        actual_arrays = sum(value.nbytes for value in vars(compact).values()
                            if isinstance(value, np.ndarray))
        self.assertGreater(compact.storage_nbytes, actual_arrays)
        self.assertGreater(compact.python_overhead_nbytes, 0)
        self.assertEqual(compact.total_storage_nbytes,
                         compact.storage_nbytes + compact.python_overhead_nbytes)
        self.assertEqual(compact.dense_equivalent_nbytes,
                         sum(value.nbytes for value in vars(dense).values()
                             if isinstance(value, np.ndarray)))
        self.assertLess(compact.total_storage_nbytes, compact.dense_equivalent_nbytes // 2)

    def test_default_cap_is_finite_and_storage_metrics_are_python_ints(self):
        compact = make_buffer(
            self.compact_class(), steps=np.int64(2), envs=np.int64(1), stack=2)
        self.assertEqual(compact.max_storage_bytes, 2 * 1024**3)
        for metric in ("storage_nbytes", "dense_equivalent_nbytes", "unique_frames"):
            self.assertIs(type(getattr(compact, metric)), int, metric)

    def test_raw_codec_dedup_gae_sampling_and_cap_failure(self):
        cls = self.compact_class()
        reference = make_buffer(cls, steps=3, envs=1, stack=2, codec="raw")
        first_observation = np.concatenate([frame(1), frame(2)])[None]
        initial_bytes = reference.storage_nbytes
        add_step(reference, first_observation)
        self.assertEqual(reference.unique_frames, 2)
        self.assertEqual(reference.storage_nbytes - initial_bytes, 2 * (frame(1).nbytes + 1))
        cap = reference.total_storage_nbytes
        limited = make_buffer(
            cls, steps=3, envs=1, stack=2, codec="raw", max_storage_bytes=cap)
        add_step(limited, first_observation)
        snapshots = {name: array.copy() for name, array in vars(limited).items()
                     if isinstance(array, np.ndarray)}
        with self.assertRaisesRegex(MemoryError, "max_storage_bytes"):
            add_step(limited, np.concatenate([frame(2), frame(3)])[None], 1)
        self.assertEqual(limited.pos, 1)
        self.assertEqual(limited.unique_frames, 2)
        self.assertEqual(limited.total_storage_nbytes, cap)
        for name, original in snapshots.items():
            self.assertEqual(getattr(limited, name).tobytes(), original.tobytes(), name)
        limited.reset()
        self.assertEqual(limited.storage_nbytes, initial_bytes)
        self.assertEqual(limited.unique_frames, 0)

        baseline = make_buffer(MaskableRolloutBuffer, steps=3, envs=1, stack=2)
        add_step(baseline, first_observation)
        for step in (1, 2):
            observations = np.concatenate([frame(step + 1), frame(step + 2)])[None]
            for buffer in (baseline, reference):
                add_step(buffer, observations, step)
        self.assertEqual(reference.unique_frames, 4)
        for buffer in (baseline, reference):
            buffer.compute_returns_and_advantage(torch.tensor([0.75]), np.array([False]))
        self.assertEqual(reference.advantages.tobytes(), baseline.advantages.tobytes())
        self.assertEqual(reference.returns.tobytes(), baseline.returns.tobytes())
        self.assert_samples_equal(baseline, reference, 2, 835)
        self.assert_samples_equal(baseline, reference, 1, 836)
        with self.assertRaises(ValueError):
            make_buffer(cls, codec="unknown")

    def test_subclass_frame_codec_hooks_roundtrip_and_account_for_custom_payload(self):
        class CustomCodecBuffer(self.compact_class()):
            def _encode_frame(self, observations):
                return b"custom-frame".ljust(32, b"\0") + observations.tobytes()

            def _decode_frame(self, payload):
                if payload[:32] != b"custom-frame".ljust(32, b"\0"):
                    raise ValueError("Wrong custom codec payload")
                return np.frombuffer(payload[32:], dtype=np.float32).reshape(self.frame_shape)

        compact = make_buffer(CustomCodecBuffer, steps=2, envs=1, stack=2)
        baseline = make_buffer(MaskableRolloutBuffer, steps=2, envs=1, stack=2)
        initial_bytes = compact.storage_nbytes
        for step in range(2):
            observations = np.concatenate([frame(step + 1), frame(step + 2)])[None]
            for buffer in (baseline, compact):
                add_step(buffer, observations, step)
        self.assertEqual(compact.storage_nbytes - initial_bytes, 3 * (frame(1).nbytes + 32))
        self.assert_samples_equal(baseline, compact, 1, 937)

    def test_invalid_observations_masks_full_buffer_and_batch_size_rejected(self):
        cls = self.compact_class()
        compact = make_buffer(cls, steps=1, envs=1, stack=1)
        good = frame(1)[None]
        for bad in (good.astype(np.float64), good[:, :-1], good[0]):
            with self.assertRaises((TypeError, ValueError)):
                add_step(compact, bad)
            self.assertEqual(compact.pos, 0)
        for mask in (np.array([[0, 1, 0.5, 1]]), np.array([[0, 1, np.nan, 1]])):
            with self.assertRaises(ValueError):
                add_step(compact, good, masks=mask)
            self.assertEqual(compact.pos, 0)
        add_step(compact, good)
        self.assertTrue(next(compact.get(1)).action_masks.bool().all())
        with self.assertRaises((RuntimeError, ValueError)):
            add_step(compact, good)
        for size in (0, -1):
            with self.assertRaises(ValueError):
                next(compact.get(size))

    def test_noncontiguous_observations_remain_exact_and_do_not_alias_inputs(self):
        compact = make_buffer(self.compact_class(), steps=2, envs=1)
        dense = make_buffer(MaskableRolloutBuffer, steps=2, envs=1)
        for step in range(2):
            observations = np.concatenate([frame(i + step) for i in range(3)])[None, :, :, ::-1]
            for buffer in (compact, dense):
                add_step(buffer, observations, step)
            observations.fill(987)
        self.assert_samples_equal(dense, compact, 1, 137)
        first_sample = next(compact.get(1))
        stored = first_sample.observations.numpy().tobytes()
        compact.reset()
        self.assertEqual(first_sample.observations.numpy().tobytes(), stored)

    def test_incompressible_payload_cannot_bypass_memory_cap(self):
        compact = make_buffer(self.compact_class(), steps=2, envs=1, stack=1)
        empty_bytes = compact.total_storage_nbytes
        compact = make_buffer(
            self.compact_class(), steps=2, envs=1, stack=1,
            max_storage_bytes=empty_bytes + 1024)
        bits = np.random.default_rng(333).integers(0, 2**32, (1, 47, 26, 26), dtype=np.uint32)
        with self.assertRaisesRegex(MemoryError, "max_storage_bytes"):
            add_step(compact, bits.view(np.float32))
        self.assertEqual(compact.pos, 0)
        self.assertEqual(compact.unique_frames, 0)
        self.assertEqual(compact.total_storage_nbytes, empty_bytes)

    def test_none_masks_are_reinitialized_after_rollout_reset(self):
        compact = make_buffer(self.compact_class(), steps=1, envs=1, stack=1)
        add_step(compact, frame(1)[None], masks=np.array([[True, False, True, False]]))
        self.assertEqual(next(compact.get(1)).action_masks.tolist(), [[1, 0, 1, 0]])
        compact.reset()
        add_step(compact, frame(2)[None])
        self.assertEqual(next(compact.get(1)).action_masks.tolist(), [[1, 1, 1, 1]])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_sample_dtypes_and_values_match_upstream(self):
        for action_space in (spaces.Discrete(4), spaces.MultiDiscrete([2, 3]), spaces.MultiBinary(3)):
            compact = self.compact_class()(
                2, observation_space(1), action_space, device="cuda", n_envs=1)
            baseline = MaskableRolloutBuffer(
                2, observation_space(1), action_space, device="cuda", n_envs=1)
            for step in range(2):
                for buffer in (compact, baseline):
                    add_step(buffer, frame(step)[None], step)
            self.assert_samples_equal(baseline, compact, 1, 41)

    @unittest.skipUnless(os.environ.get("TANK_AI_OBSERVATION_SNAPSHOT"),
                         "Set TANK_AI_OBSERVATION_SNAPSHOT to an actual Rust float32 .npy observation")
    def test_optional_rust_snapshot_roundtrip(self):
        observations = np.load(os.environ["TANK_AI_OBSERVATION_SNAPSHOT"], allow_pickle=False)
        if observations.ndim == 3:
            observations = observations[None]
        self.assertEqual(observations.dtype, np.float32)
        self.assertEqual(observations.shape[-2:], (26, 26))
        self.assertEqual(observations.shape[1] % 47, 0)
        stack = observations.shape[1] // 47
        compact = make_buffer(self.compact_class(), steps=1, envs=len(observations), stack=stack)
        baseline = make_buffer(MaskableRolloutBuffer, steps=1, envs=len(observations), stack=stack)
        for buffer in (compact, baseline):
            add_step(buffer, observations)
        self.assert_samples_equal(baseline, compact, 1, 823)


class TinyMaskedEnv(gym.Env):
    """Deterministic 47-plane environment with real resets and legal-action checks."""

    def __init__(self):
        self.observation_space = observation_space(3)
        self.action_space = spaces.Discrete(4)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.frames = [np.zeros((47, 26, 26), np.float32)] * 2 + [frame(1)]
        return np.concatenate(self.frames), {}

    def action_masks(self):
        return np.array([True, self.steps % 2 == 0, True, False])

    def step(self, action):
        if not self.action_masks()[int(action)]:
            raise AssertionError(f"Illegal masked action {action}")
        self.steps += 1
        self.frames = self.frames[1:] + [frame(self.steps + int(action) + 1)]
        return np.concatenate(self.frames), float(action) / 7, self.steps == 5, False, {}


class TrainingTests(unittest.TestCase):
    def test_two_rollouts_match_real_maskableppo_outputs_and_parameters(self):
        from compact_rollout import CompactMaskableRolloutBuffer

        def train(buffer_cls, codec=None):
            env = DummyVecEnv([TinyMaskedEnv, TinyMaskedEnv])
            model = MaskablePPO(
                "MlpPolicy", env, seed=710, device="cpu",
                n_steps=4, batch_size=4, n_epochs=2, learning_rate=0.0003,
                policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
                rollout_buffer_class=buffer_cls,
                rollout_buffer_kwargs={} if codec is None else {"codec": codec},
            )
            model.learn(total_timesteps=16)
            self.assertEqual(model.num_timesteps, 16)
            sample = torch.from_numpy(env.reset())
            with torch.no_grad():
                outputs = model.policy.evaluate_actions(
                    sample, torch.tensor([0, 2]),
                    action_masks=np.array([env.envs[0].action_masks(), env.envs[1].action_masks()]),
                )
            env.close()
            return model, outputs

        baseline, expected = train(MaskableRolloutBuffer)
        for codec in ("zlib", "raw"):
            with self.subTest(codec=codec):
                compact, actual = train(CompactMaskableRolloutBuffer, codec)
                self.assertEqual(baseline._n_updates, compact._n_updates)
                for name, parameter in baseline.policy.state_dict().items():
                    self.assertTrue(torch.equal(parameter, compact.policy.state_dict()[name]), name)
                for left, right in zip(expected, actual):
                    self.assertEqual(left.numpy().tobytes(), right.numpy().tobytes())


if __name__ == "__main__":
    unittest.main()
