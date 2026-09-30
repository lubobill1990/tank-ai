"""Fake early-ROM factory tests: no real emulator steps or GPU/model work."""
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

import game_state as gs
from within_stage.openings import ProbeAccounting
from within_stage import training_env as training

SPLIT = Path("runs") / "20260930-within-stage" / "frozen-split.json"


class FakeEarly(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = gym.spaces.Discrete(10)
    render_mode = None
    instances = []

    def __init__(self, config, account):
        self.config, self.account = config, account
        self.closed, self.ticks = False, 0
        self.mask = np.ones(10, bool)
        self.mask[9] = False
        self.observation = np.zeros((47, 26, 26), np.float32)
        self.actions = []
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        for _ in range(2):
            self.account.before("raw")
            self.account.after("raw")
        terrain = np.zeros((26, 26), np.uint8)
        terrain[24:26, 12:14] = gs.T_BASE
        self.state = gs.GameState(0, terrain, gs.Tank(0, 88, 216, gs.UP),
                                  (gs.Tank(2, 90 + self.config % 4, 160, gs.DOWN),),
                                  (), 1, 3, 19, 0, 0, None, False, False, True,
                                  remaining_enemies=20, fire_ready=True,
                                  brick_masks=np.zeros((26, 26), np.uint8))
        self.ticks = 0
        self.observation.fill(.1)
        return self.observation, {"objective": "full", "outcome": ""}

    def action_masks(self):
        return self.mask

    def step(self, action):
        self.account.before("agent")
        self.actions.append(int(action))
        self.ticks += 1
        self.state = replace(self.state, frame=self.ticks * 4, kills=self.ticks,
                             remaining_enemies=20 - self.ticks)
        self.observation.fill(.1 + int(action) / 100)
        done = self.ticks == 3
        # Deliberately no per-step "kills" info; provider must read canonical state.
        info = dict(objective="full", outcome="timeout" if done else "",
                    frames=self.state.frame, terminal_remaining_enemies=20 - self.ticks)
        self.account.after("agent")
        self.account.agent_raw_frames += 4
        return self.observation, np.float32(.125), False, done, info

    def close(self):
        self.closed = True


class TrainingEnvTests(unittest.TestCase):
    def setUp(self):
        FakeEarly.instances = []
        self.deadline = time.monotonic() + 1000
        self.account = ProbeAccounting(1000000, deadline=self.deadline)
        self.patch = patch.object(
            training, "make_opening_env",
            side_effect=lambda pool, index, *, account, max_frames: FakeEarly(index, account))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.context = training.load_train_context(SPLIT)

    def make(self, **kwargs):
        defaults = dict(n_envs=2, account=self.account, deadline=self.deadline)
        defaults.update(kwargs)
        env = training.make_training_env(SPLIT, **defaults)
        self.addCleanup(env.close)
        return env

    def test_only_train_families_and_aliases_are_selected(self):
        env = self.make()
        allowed = set(self.context["all_train_config_indices"])
        for _ in range(30):
            env.reset()
            self.assertTrue(all(row["config_index"] in allowed for row in env.current_openings()))
        self.assertTrue(all(instance.config in allowed for instance in FakeEarly.instances))
        self.assertTrue(all(e.closed for e in FakeEarly.instances[:-2]))

    def test_family_sampling_is_uniform_not_config_count_weighted(self):
        groups = self.context["groups"]
        small = min(groups, key=lambda f: len(groups[f]))
        large = max(groups, key=lambda f: len(groups[f]))
        self.assertLess(len(groups[small]), len(groups[large]))
        env = self.make(n_envs=1, config_indices=groups[small] + groups[large])
        counts = {small: 0, large: 0}
        for _ in range(2000):
            env.reset()
            counts[env.current_openings()[0]["family_id"]] += 1
        self.assertLess(abs(counts[small] - counts[large]), 200)

    def test_dev_final_and_invalid_subsets_fail_before_any_reset(self):
        split = json.loads(SPLIT.read_text())
        for name in ("development", "final"):
            forbidden = split[name]["representative_config_indices"][0]
            with self.assertRaises(ValueError):
                self.make(config_indices=[forbidden])
        with self.assertRaises(ValueError):
            self.make(config_indices=[])
        self.assertFalse(FakeEarly.instances)
        self.assertEqual(self.account.attempted, 0)

    def test_terminal_history_zero_reset_and_counter_provider_ignore_sparse_info(self):
        env = self.make()
        observation = env.reset()
        self.assertEqual(observation.shape, (2, 110, 26, 26))
        self.assertTrue(np.all(observation[:, :55] == 0))
        for _ in range(2):
            _, rewards, dones, _ = env.step(np.array([2, 4]))
            self.assertTrue(np.all(rewards == 0))
            self.assertFalse(dones.any())
        np.testing.assert_array_equal(env.current_kills(), [2, 2])
        observation, rewards, dones, infos = env.step(np.array([6, 6]))
        self.assertTrue(dones.all())
        self.assertTrue(np.all(rewards == -1))
        self.assertTrue(np.all(observation[:, :55] == 0))
        np.testing.assert_array_equal(env.current_kills(), [3, 3])
        np.testing.assert_array_equal(env.current_state_kills(), [0, 0])
        np.testing.assert_array_equal(env.initial_remaining(), [20, 20])
        for info in infos:
            self.assertFalse(info["TimeLimit.truncated"])
            self.assertEqual(info["episode"]["r"], -1)
            self.assertEqual(info["reward_profile"], "finite-victory-sparse-v1")
            self.assertEqual(info["terminal_observation"].shape, (110, 26, 26))
            self.assertTrue(np.any(info["terminal_observation"][:55] != 0))
            self.assertEqual(info["execution_noise"]["intended_action"], 6)
            self.assertEqual(info["execution_noise"]["executed_action"], 6)
        self.assertEqual(self.account.agent_attempted, 6)
        self.assertEqual(self.account.agent_completed, 6)
        self.assertEqual(self.account.raw_initialization_attempted, 8)

    def test_noise0_preserves_parity_and_noise_seed_never_changes_openings(self):
        a = self.make(n_envs=1, noise_seed=11, noise_probability=0.)
        b = self.make(n_envs=1, noise_seed=99, noise_probability=0.)
        c = self.make(n_envs=1, noise_seed=123, noise_probability=1.)
        for _ in range(8):
            ao, bo, co = a.reset(), b.reset(), c.reset()
            self.assertEqual(a.current_openings()[0]["config_index"], b.current_openings()[0]["config_index"])
            self.assertEqual(a.current_openings()[0]["config_index"], c.current_openings()[0]["config_index"])
            np.testing.assert_array_equal(ao, bo)
            for action in (2, 4, 6):
                ax, ar, ad, ai = a.step(np.array([action]))
                bx, br, bd, bi = b.step(np.array([action]))
                c.step(np.array([action]))
                self.assertEqual(ax.tobytes(), bx.tobytes())
                self.assertEqual(ar.tobytes(), br.tobytes())
                self.assertIsNone(ai[0]["execution_noise"]["draw"])
                self.assertEqual(ai[0]["execution_noise"]["executed_action"], action)

    def test_masks_metadata_and_observations_are_owned(self):
        env = self.make(n_envs=1)
        env.reset()
        masks = env.env_method("action_masks")[0]
        masks.fill(False)
        self.assertTrue(env.env_method("action_masks")[0][0])
        info = env.current_openings()
        info[0]["config_index"] = -999
        self.assertNotEqual(env.current_openings()[0]["config_index"], -999)
        provenance = env.provenance
        self.assertEqual(provenance["split_sha256"], training.FROZEN_SPLIT_SHA256)
        self.assertTrue(provenance["sources"])
        self.assertEqual(provenance["wrapper_order"][-2:], ["FiniteVictoryVecEnv(shaping=False,gamma=1)", "VecFrameStack(first,2)"])

    def test_deadline_stops_before_bootstrap_or_control(self):
        env = self.make(n_envs=1, deadline=time.monotonic() - 1)
        with self.assertRaises(TimeoutError):
            env.reset()
        self.assertEqual(self.account.attempted, 0)

    def test_provenance_guard_rejects_changed_source_hashes(self):
        env = self.make(n_envs=1)
        self.assertTrue(env.verify_provenance())
        with patch.object(training, "_source_hashes", return_value={}):
            with self.assertRaises(ValueError):
                env.verify_provenance()
        self.assertFalse(FakeEarly.instances)

    def test_policy_global_rng_and_sb3_seed_do_not_reseed_opening_stream(self):
        saved_rng = np.random.get_state()
        self.addCleanup(np.random.set_state, saved_rng)
        a, b = self.make(n_envs=1), self.make(n_envs=1)
        first = []
        second = []
        for index in range(12):
            np.random.seed(index)
            a.seed(index)
            a.reset()
            first.append(a.current_openings()[0]["config_index"])
            np.random.seed(index + 100)
            b.seed(index + 999)
            b.reset()
            second.append(b.current_openings()[0]["config_index"])
        self.assertEqual(first, second)

    def test_failed_step_never_silently_replays_episode(self):
        env = self.make(n_envs=1)
        env.reset()
        raw = FakeEarly.instances[-1]
        with patch.object(raw, "step", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                env.step(np.array([0]))
        with self.assertRaises(RuntimeError):
            env.step(np.array([0]))
        self.assertEqual(self.account.agent_attempted, 0)


if __name__ == "__main__":
    unittest.main()
