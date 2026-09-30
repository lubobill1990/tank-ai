"""Fake-only dynamic aiming tests; zero ROM/native decisions and no GPU."""
from itertools import product
import hashlib
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from learning_lab import aiming_dynamic as dynamic
from learning_lab.aiming_task import SCENE_VERSION


def fake_generator(count, seed):
    rams = np.zeros((count, 2048), np.uint8)
    keys = list(product(range(4), range(4), (0, 2, 4, 6)))
    for index in range(count):
        rams[index, :8] = np.frombuffer(int(seed).to_bytes(8, "little"), np.uint8)
        rams[index, 8:12] = np.frombuffer(index.to_bytes(4, "little"), np.uint8)
    return dict(
        rams=rams, directions=np.array([keys[i % 64][0] for i in range(count)], np.int64),
        player_facings=np.array([keys[i % 64][1] for i in range(count)], np.int64),
        subcell_offsets=np.array([keys[i % 64][2] for i in range(count)], np.int64),
        distances=np.full(count, 64, np.int64),
        hashes=np.array([hashlib.sha256(r.tobytes()).hexdigest() for r in rams]),
        metadata={"scene_version": SCENE_VERSION, "generator_seed": seed,
                  "initialization_rom_decisions": 0, "support_flags": [0] * count},
    )


class FakeRaw(VecEnv):
    instances = []

    def __init__(self, initial, max_frames, *, seed):
        self.initial = initial.copy()
        self.marker = (int(initial[0, 8]) + 1) / 1024
        self.closed, self.ticks = False, 0
        self.batch = self
        self.stats = np.zeros((1, 5), np.uint32)
        self.support = np.zeros(1, np.uint32)
        self._masks = np.ones((1, 10), bool)
        self.instances.append(self)
        super().__init__(1, gym.spaces.Box(-1, 1, (47, 26, 26), np.float32), gym.spaces.Discrete(10))

    def ram(self, index=0):
        return self.initial[index].copy()

    def masks(self):
        return self._masks

    def reset(self):
        self.ticks = 0
        self.stats.fill(0)
        return np.full((1, 47, 26, 26), self.marker, np.float32)

    def step_async(self, action):
        self.action = action

    def step_wait(self):
        self.ticks += 1
        done = self.ticks == 2
        self.stats[0] = (self.ticks * 4, int(done), 0, int(done), 0)
        terminal = np.full((47, 26, 26), self.marker + .2, np.float32)
        info = {}
        if done:
            info = {"outcome": "clear", "episode": {"r": .5, "l": 2},
                    "terminal_observation": terminal, "TimeLimit.truncated": False,
                    "frames": 8, "kills": 1, "deaths": 0}
        # Deliberately emulate the native VecEnv's old-layout automatic reset.
        obs = np.full((1, 47, 26, 26), self.marker if done else self.marker + .1, np.float32)
        return obs, np.zeros(1, np.float32), np.array([done]), [info]

    def close(self):
        self.closed = True

    def get_attr(self, name, indices=None):
        return [None]

    def set_attr(self, name, value, indices=None):
        pass

    def env_method(self, name, *args, indices=None, **kwargs):
        if name == "action_masks":
            return [self._masks[0]]
        raise AttributeError(name)

    def env_is_wrapped(self, cls, indices=None):
        return [False]


class DynamicAimingTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(dynamic, "generate_aiming_starts", fake_generator),
                        patch.object(dynamic, "RustVectorEnv", FakeRaw)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        FakeRaw.instances = []
        self.account = dynamic.DynamicAccounting(10000, deadline=time.monotonic() + 1000)
        self.sampler = dynamic.BalancedAimingSampler(seed=0, excluded_hashes=set(),
                                                     accounting=self.account)

    def test_fresh_unique_balanced_stream_with_excluded_hash(self):
        reference = dynamic.BalancedAimingSampler(seed=0, excluded_hashes=set(),
                                                  accounting=self.account)
        blocked = reference.next_layout()["hash"]
        sampler = dynamic.BalancedAimingSampler(seed=0, excluded_hashes={blocked}, accounting=self.account)
        rows = [sampler.next_layout() for _ in range(64)]
        self.assertEqual(len({r["hash"] for r in rows}), 64)
        self.assertNotIn(blocked, {r["hash"] for r in rows})
        self.assertEqual(len({(r["direction"], r["player_facing"], r["subcell_offset"])
                              for r in rows}), 64)
        for key in ("direction", "player_facing"):
            np.testing.assert_array_equal(np.bincount([r[key] for r in rows]), [16] * 4)

    def test_terminal_is_owned_and_second_step_requires_fresh_reset(self):
        env = dynamic.DynamicAimingEnv(self.sampler, stream=0)
        self.addCleanup(env.close)
        first, first_info = env.reset(seed=0)
        raw = FakeRaw.instances[-1]
        env.step(0)
        terminal, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.)
        np.testing.assert_array_equal(terminal, np.full((47, 26, 26), raw.marker + .2, np.float32))
        terminal.fill(99)
        self.assertFalse(np.any(info["terminal_observation"] == 99))
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(0)
        new, new_info = env.reset()
        self.assertTrue(raw.closed)
        self.assertNotEqual(first_info["layout_hash"], new_info["layout_hash"])
        self.assertFalse(np.shares_memory(first, new))
        self.assertEqual(self.account.native_steps_completed, 2)

    def test_dummy_vec_autoreset_has_zero_prefix_and_terminal_history(self):
        env = dynamic.make_dynamic_aiming_env(self.sampler, n_envs=2, seed=0)
        self.addCleanup(env.close)
        initial = env.reset()
        self.assertEqual(initial.shape, (2, 94, 26, 26))
        self.assertTrue(np.all(initial[:, :47] == 0))
        env.step(np.zeros(2, np.int64))
        new, rewards, dones, infos = env.step(np.zeros(2, np.int64))
        self.assertTrue(dones.all())
        self.assertTrue(np.all(new[:, :47] == 0))
        self.assertTrue(np.all(rewards == 1))
        for info in infos:
            self.assertEqual(info["terminal_observation"].shape, (94, 26, 26))
            self.assertTrue(np.any(info["terminal_observation"][:47] != 0))
            self.assertFalse(info["TimeLimit.truncated"])
        self.assertEqual(len(FakeRaw.instances), 4)
        self.assertTrue(all(e.closed for e in FakeRaw.instances[:2]))
        self.assertEqual(self.account.native_steps_completed, 4)

    def test_masks_are_owned_and_mask_invalid_actions_rejected(self):
        env = dynamic.DynamicAimingEnv(self.sampler, stream=0)
        self.addCleanup(env.close)
        env.reset()
        masks = env.action_masks()
        masks.fill(False)
        self.assertTrue(env.action_masks().all())
        FakeRaw.instances[-1]._masks[0, 5] = False
        with self.assertRaises(ValueError):
            env.step(5)
        self.assertEqual(self.account.native_steps_attempted, 0)

    def test_budget_and_deadline_stop_before_a_native_step(self):
        env = dynamic.DynamicAimingEnv(self.sampler, stream=0)
        self.addCleanup(env.close)
        env.reset()
        self.account.limit = 1
        env.step(0)
        with self.assertRaises(dynamic.DynamicStop):
            env.step(0)
        self.assertEqual(self.account.native_steps_attempted, 1)
        self.account.limit = 10
        self.account.deadline = 0
        with self.assertRaises(dynamic.DynamicStop):
            env.step(0)
        self.assertEqual(self.account.native_steps_attempted, 1)

    def test_new_final_seed_and_hash_exclusion_apply_before_training(self):
        forbidden = {str(fake_generator(128, dynamic.FINAL_TEST_SEED)["hashes"][0])}
        holdout = dynamic._new_final_test(self.account, forbidden)
        self.assertEqual(holdout["metadata"]["generator_seed"], 20261002)
        self.assertEqual(holdout["rams"].shape, (128, 2048))
        self.assertFalse(set(holdout["hashes"]) & forbidden)
        self.assertEqual(len(set(holdout["hashes"])), 128)

    def test_final_evaluation_is_single_use_and_blocks_later_training(self):
        root = Path("runs") / ("dynamic-test-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root)
        prepared = dict(accounting=self.account, sampler=self.sampler, output=root,
                        final_test=fake_generator(128, dynamic.FINAL_TEST_SEED))
        self.account.reserved_evaluation = 6400
        with patch.object(dynamic, "evaluate_holdout", return_value={
            "status": "completed", "attempted_decisions": 256, "completed_decisions": 256,
        }) as evaluate:
            dynamic.evaluate_final_once(object(), prepared)
            with self.assertRaises(dynamic.DynamicStop):
                dynamic.evaluate_final_once(object(), prepared)
        evaluate.assert_called_once()
        self.assertEqual(self.account.evaluation_attempted, 256)
        with self.assertRaises(dynamic.DynamicStop):
            self.sampler.next_layout()
        with self.assertRaises(dynamic.DynamicStop):
            self.account.take_step()


if __name__ == "__main__":
    unittest.main()
