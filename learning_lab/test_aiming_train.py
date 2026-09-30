"""Tiny CPU/fake-native tests; no real environment decisions or GPU work."""
from pathlib import Path
import json
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from learning_lab import aiming_train as aiming


class FakeVec(VecEnv):
    def __init__(self, count=2):
        self.ticks = np.zeros(count, np.int64)
        self.batch = SimpleNamespace(stats=np.zeros((count, 5), np.uint32))
        super().__init__(count, gym.spaces.Box(-1, 1, (47, 26, 26), np.float32),
                         gym.spaces.Discrete(10))

    def reset(self):
        self.ticks.fill(0)
        self.batch.stats.fill(0)
        return np.full((self.num_envs, 47, 26, 26), .25, np.float32)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.ticks += 1
        done = self.ticks == 2
        self.batch.stats[:, 1] = done.astype(np.uint32)
        info = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(done):
            info[i] = dict(outcome="clear", episode={"r": 1., "l": 2}, frames=8,
                           kills=1, deaths=0, shots=1, lives=3,
                           terminal_observation=np.full((47, 26, 26), .5, np.float32),
                           **{"TimeLimit.truncated": False})
            self.ticks[i] = 0
        return self.reset_observation(), np.zeros(self.num_envs), done, info

    def reset_observation(self):
        return np.full((self.num_envs, 47, 26, 26), .25, np.float32)

    def close(self):
        pass

    def get_attr(self, name, indices=None):
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        pass

    def env_method(self, name, *args, indices=None, **kwargs):
        if name == "action_masks":
            return [np.ones(10, bool) for _ in self._get_indices(indices)]
        raise AttributeError(name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class FakeNative:
    def __init__(self, initial, max_frames, *, seeds):
        self.n, self.calls = len(initial), 0

    def observe(self):
        return np.zeros((self.n, 47, 26, 26), np.float32)

    def masks(self):
        return np.ones((self.n, 10), bool)

    def step(self, actions):
        self.calls += 1
        done = np.arange(self.n) < (self.n // 2 if self.calls == 1 else self.n)
        stats = np.zeros((self.n, 5), np.uint32)
        stats[:, 0] = 4 * self.calls
        return self.observe(), np.zeros(self.n), done, stats

    def ram(self, index):
        ram = np.zeros(2048, np.uint8)
        ram[0x68], ram[0x51] = 0x80, 3
        return ram

    def close(self):
        pass


class AimingTrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path("runs") / ("aiming-train-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def test_runtime_sparse_objective_zero_history_and_only_completed_updates(self):
        starts = {"rams": np.zeros((2, 2048), np.uint8), "directions": np.array([0, 1])}
        guard = SimpleNamespace(check=lambda phase: None)
        with patch.object(aiming, "make_aiming_env", return_value=FakeVec()):
            runtime = aiming.AimingRuntime(starts, guard=guard, device="cpu", output=self.root,
                                           target_steps=4, n_steps=2, batch_size=2)
            try:
                self.assertEqual(runtime.objective_version, "finite-victory-sparse-v1")
                self.assertEqual(runtime.model.gamma, 1.)
                self.assertEqual(runtime.model._last_obs.shape, (2, 94, 26, 26))
                self.assertTrue(np.all(runtime.model._last_obs[:, :47] == 0))
                self.assertTrue(runtime.collect())
                self.assertEqual(runtime.counted.completed_steps, 4)
                self.assertEqual(runtime.optimized_steps, 0)
                runtime.optimize()
                self.assertEqual(runtime.optimized_steps, 4)
                self.assertTrue(runtime.callback.episodes)
                self.assertTrue(all(row["return"] == 1 for row in runtime.callback.episodes))
                self.assertIsInstance(runtime.model.rollout_buffer, aiming.NativeGPURunRolloutBuffer)
                self.assertFalse(torch.cuda.is_initialized())
            finally:
                runtime.close()

    def test_eval_has_one_first_episode_per_template_and_counts_finished_slots(self):
        starts = {"rams": np.zeros((128, 2048), np.uint8),
                  "directions": np.tile(np.arange(4), 32)}
        policy = SimpleNamespace(predict=lambda obs, **kwargs: (np.zeros(len(obs), np.int64), None))
        guard = SimpleNamespace(check=lambda phase: None)
        with patch.object(aiming, "NativeBatch", FakeNative):
            report = aiming.evaluate_holdout(policy, starts, guard=guard)
        self.assertEqual(report["completed_decisions"], 256)
        self.assertEqual(len(report["episodes"]), 128)
        self.assertEqual(report["wins"], 128)
        self.assertEqual(len(set(row["template_index"] for row in report["episodes"])), 128)
        self.assertEqual(report["max_decisions"], 6400)

    def test_profile_and_dataset_role_are_not_full_game(self):
        contract = aiming.contract()
        self.assertFalse(contract["full_game"])
        self.assertFalse(contract["shaping"])
        self.assertEqual(contract["ppo"]["n_steps"], 128)
        self.assertEqual(contract["ppo"]["gamma"], 1.)
        self.assertEqual(contract["training_steps"], 65536)
        self.assertEqual(contract["holdout_episodes"], 128)

    def test_strict_checkpoint_reload_preserves_sparse_policy_and_adam(self):
        starts = {"rams": np.zeros((2, 2048), np.uint8), "directions": np.array([0, 1])}
        guard = SimpleNamespace(check=lambda phase: None)
        with patch.object(aiming, "make_aiming_env", return_value=FakeVec()):
            runtime = aiming.AimingRuntime(starts, guard=guard, device="cpu", output=self.root,
                                           target_steps=4, n_steps=2, batch_size=2)
            try:
                runtime.collect()
                runtime.optimize()
                # Synthetic progress tag exercises the production archive schema;
                # this fixture performs no real native/ROM decisions.
                runtime.optimized_steps = 4096
                provenance = dict(sources=aiming._sources(), datasets={
                    name: "a" * 64 for name in ("train.npz", "train.json", "test.npz", "test.json")})
                path = self.root / "fixture.zip"
                aiming.save_checkpoint(runtime, path, provenance, "cpu_fixture")
                model = aiming.load_aiming_checkpoint(path)
                self.assertEqual(model.gamma, 1.)
                self.assertTrue(model.policy.optimizer.state_dict()["state"])
                for name, value in runtime.model.policy.state_dict().items():
                    torch.testing.assert_close(value, model.policy.state_dict()[name], rtol=0, atol=0)
                sidecar = path.with_suffix(".manifest.json")
                manifest = json.loads(sidecar.read_text())
                manifest["objective_version"] = "finite-victory-potential-v1"
                sidecar.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    aiming.load_aiming_checkpoint(path)
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
