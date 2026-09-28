"""Regression coverage for the standalone seeded native training boundary."""
import unittest

import numpy as np

from rust_env import NativeBatch, RustVectorEnv, episode_seed, stage_ram


class RustEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = stage_ram(1, seed=0)[None, :]

    def test_episode_seed_domains_are_distinct_and_repeatable(self):
        seeds = {episode_seed(31, stream, episode)
                 for stream in range(3) for episode in range(4)}
        self.assertEqual(len(seeds), 12)
        self.assertEqual(episode_seed(31, 2, 3), episode_seed(31, 2, 3))
        self.assertNotEqual(episode_seed(32, 2, 3), episode_seed(31, 2, 3))
        for value in (-1, 2**64, 0.5, True):
            with self.assertRaises(ValueError):
                episode_seed(value, 0, 0)

    def test_fused_and_plain_step_match_latest_full_game(self):
        initial = np.repeat(self.initial, 2, axis=0)
        batches = [NativeBatch(initial, 240, seeds=[31, 32], fast=fast)
                   for fast in (False, True)]
        try:
            for action in np.random.default_rng(77).integers(0, 10, (90, 2)):
                left, right = [batch.step(action) for batch in batches]
                for a, b in zip(left, right):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(batches[0].masks(), batches[1].masks())
                for index in np.flatnonzero(left[2]):
                    for batch in batches:
                        batch.reset(int(index), seed=100 + int(index))
        finally:
            for batch in batches:
                batch.close()

    def test_vector_seeding_reproduces_after_autoresets(self):
        env = RustVectorEnv(np.repeat(self.initial, 2, axis=0), 12, seed=17)
        try:
            def trace():
                env.seed(17)
                result = [env.reset()]
                for action in ([5, 1], [6, 2], [7, 3], [8, 4], [9, 5]) * 3:
                    result.extend(env.step(np.asarray(action))[:3])
                return result
            left, right = trace(), trace()
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)
        finally:
            env.close()

    def test_time_limit_preserves_terminal_observation_and_retained_outputs(self):
        env = RustVectorEnv(self.initial, 4, seed=22)
        try:
            initial = env.reset()
            obs, rewards, dones, infos = env.step(np.array([5]))
            self.assertTrue(dones[0])
            self.assertTrue(infos[0]['TimeLimit.truncated'])
            self.assertEqual(infos[0]['outcome'], 'timeout')
            terminal = infos[0]['terminal_observation']
            self.assertEqual(float(terminal[36, 0, 0]), 0.0)
            np.testing.assert_array_equal(obs, initial)
            saved = [value.copy() for value in (obs, rewards, dones, terminal)]
            env.step(np.array([1]))
            for actual, expected in zip((obs, rewards, dones, terminal), saved):
                np.testing.assert_array_equal(actual, expected)
        finally:
            env.close()

    def test_closed_batch_and_invalid_actions_raise(self):
        batch = NativeBatch(self.initial, 40, seeds=[1])
        try:
            before = batch.ram()
            for action in ([-1], [10], [1.5], [True], [[1]], [0, 1]):
                with self.assertRaises(ValueError):
                    batch.step(action)
                np.testing.assert_array_equal(batch.ram(), before)
        finally:
            batch.close()
        with self.assertRaises(RuntimeError):
            batch.observe()


if __name__ == '__main__':
    unittest.main()
