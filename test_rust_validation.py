"""CPU-only validator contracts, including real cartridge/native smoke cases."""
import importlib
import importlib.util
from dataclasses import replace
import json
import random
import time
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from battle_city_env import BattleCityEnv
import game_state as gs
from rust_env import episode_seed, stage_ram
from skills import SkillEnv


class PrimitivePolicy:
    def __init__(self, nstack=16, action=0, error=None):
        self.observation_space = spaces.Box(-1, 1, (47 * nstack, 26, 26), np.float32)
        self.action_space = spaces.Discrete(10)
        self.policy = torch.nn.Identity()
        self.action, self.error = action, error
        self.calls = self.starts = 0

    def predict(self, observation, *, deterministic, action_masks):
        assert observation.shape == self.observation_space.shape
        assert action_masks.shape == (10,)
        assert action_masks.dtype == np.bool_
        self.calls += 1
        self.starts += int(not np.any(observation[:-47]))
        random.random()
        np.random.random()
        torch.rand(1)
        self.policy.eval()
        if self.error:
            raise self.error
        action = self.action if action_masks[self.action] else 0
        return np.asarray(action), None


class CounterEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(15)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = 1
        return np.full(self.observation_space.shape, self.frame / 10, np.float32), {}

    def step(self, action):
        self.frame += 1
        return (np.full(self.observation_space.shape, self.frame / 10, np.float32),
                0.0, False, self.frame == 4, {})

    def action_masks(self):
        return np.ones(15, dtype=bool)


class ValidationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("rust_validation"),
                             "the periodic ten-primitive validator is missing")
        return importlib.import_module("rust_validation")

    def snapshot_rng(self):
        return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()

    def assert_rng(self, saved):
        self.assertEqual(random.getstate(), saved[0])
        current = np.random.get_state()
        self.assertEqual(current[0], saved[1][0])
        np.testing.assert_array_equal(current[1], saved[1][1])
        self.assertEqual(current[2:], saved[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), saved[2]))

    def test_adapter_prohibits_macros_and_only_exposes_ten_masks(self):
        module = self.module()
        env = module.PrimitiveActions(CounterEnv())
        self.addCleanup(env.close)
        env.reset(seed=101)
        self.assertEqual(env.action_space.n, 10)
        np.testing.assert_array_equal(env.action_masks(), np.ones(10, bool))
        for action in (-1, 10, 14, 1.5, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)

    def test_history_matches_vecframestack_at_reset_steps_terminal_and_next_reset(self):
        module = self.module()
        for nstack in (2, 16):
            with self.subTest(nstack=nstack):
                env = module.PrimitiveActions(CounterEnv())
                vec = VecFrameStack(DummyVecEnv([CounterEnv]), n_stack=nstack,
                                    channels_order="first")
                player = module.EpisodePlayer(env, model=PrimitivePolicy(nstack),
                                              nstack=nstack, deterministic=True)
                try:
                    vec.seed(101)
                    np.testing.assert_array_equal(player.reset(101), vec.reset()[0])
                    for _ in range(3):
                        _, done, _ = player.step()
                        obs, _, dones, infos = vec.step(np.array([0]))
                        self.assertEqual(done, bool(dones[0]))
                        expected = infos[0]["terminal_observation"] if done else obs[0]
                        np.testing.assert_array_equal(player.observation, expected)
                    with self.assertRaises(RuntimeError):
                        player.step()
                    np.testing.assert_array_equal(player.reset(101), obs[0])
                finally:
                    env.close()
                    vec.close()

    def test_real_suite_separate_modes_counts_caps_and_history(self):
        module = self.module()
        policy = PrimitivePolicy()
        saved = self.snapshot_rng()
        result = module.evaluate_suite(policy, model_name="history", stages=(1, 4),
                                       seeds=(101, 202), max_frames=12)
        self.assert_rng(saved)
        self.assertTrue(policy.policy.training)
        self.assertEqual(set(result), {"rust", "rom", "skills", "metadata"})
        self.assertFalse(result["metadata"]["rom_parity"])
        self.assertTrue(result["metadata"]["smoke_only"])
        self.assertEqual(result["metadata"]["n_stack"], 16)
        json.dumps(result, allow_nan=False)
        for backend in ("rust", "rom"):
            self.assertEqual(result[backend]["max_frames"], 12)
            for mode in ("deterministic", "sampled"):
                data = result[backend]["modes"][mode]
                summary = data["summary"]
                self.assertEqual(summary["episodes"], 4)
                self.assertEqual(summary["wins"], 0)
                self.assertEqual(summary["mean_frames"], 12)
                self.assertEqual(set(data["per_stage"]), {"1", "4"})
                self.assertEqual(len(data["episodes"]), 4)
                for key in ("mean_kills", "mean_deaths", "mean_shots", "mean_pickups"):
                    self.assertEqual(summary[key], 0)
                for episode in data["episodes"]:
                    self.assertEqual(episode["outcome"], "timeout")
                    self.assertEqual(episode["decisions"], 3)
                    self.assertEqual(episode["diagnostics"]["sample_count"], 3)
                    self.assertEqual(episode["diagnostics"]["sampled_frames"], 12)
                    if backend == "rust":
                        self.assertEqual(episode["episode_seed"],
                                         episode_seed(episode["seed"], 0, 0))
        for mode in ("deterministic", "sampled"):
            data = result["skills"]["modes"][mode]
            self.assertEqual(data["summary"]["episodes"], 24)
            self.assertNotIn("wins", data["summary"])
            self.assertEqual(data["summary"]["mean_success_rate"], 0)
            self.assertEqual(len(data["per_skill"]), 6)
            self.assertEqual({e["variant"] for e in data["episodes"]}, {0, 1})
            self.assertTrue(all(not e["skill_success"] for e in data["episodes"]))
        self.assertEqual(policy.starts, 64)
        self.assertEqual(policy.calls, 192)

    def test_resource_pressure_between_decisions_restores_rng_and_policy_mode(self):
        from rust_resources import ResourcePressureError
        module = self.module()
        policy = PrimitivePolicy()
        saved = self.snapshot_rng()
        checks = 0

        def check():
            nonlocal checks
            checks += 1
            if checks == 4:
                raise ResourcePressureError('host commit reserve')

        with self.assertRaises(ResourcePressureError):
            module.evaluate_suite(policy, model_name='history', stages=(1,),
                                  seeds=(101,), max_frames=12, resource_check=check)
        self.assertEqual(policy.calls, 1)
        self.assertTrue(policy.policy.training)
        self.assert_rng(saved)

    def test_real_native_terminal_counters_and_history_are_not_autoreset_values(self):
        module = self.module()
        raw = module.RustEpisode(stage_ram(1)[None], stage=1, max_frames=4)
        self.addCleanup(raw.close)
        player = module.EpisodePlayer(raw, model=PrimitivePolicy(16, action=5),
                                      nstack=16, deterministic=True)
        first = player.reset(101).copy()
        _, done, info = player.step()
        self.assertTrue(done)
        self.assertEqual(info["frames"], 4)
        self.assertEqual(info["shots"], 1)
        self.assertEqual(raw.state.frame, 4)
        self.assertEqual(player.observation[-47 + 36, 0, 0], 0)
        self.assertTrue(np.any(player.observation[-94:-47]))
        np.testing.assert_array_equal(player.reset(101), first)

    def test_skill_success_is_real_primitive_success_never_a_full_win(self):
        module = self.module()
        for action, success in ((5, True), (0, False)):
            for variant in (0, 1):
                raw = module.PrimitiveActions(SkillEnv("shoot", variant=variant, max_frames=120))
                try:
                    episode = module.play_episode(raw, model=PrimitivePolicy(2, action),
                                                  nstack=2, deterministic=True, seed=101)
                    self.assertEqual(episode["skill_success"], success)
                    self.assertEqual(episode["outcome"], "objective" if success else "failure")
                    self.assertNotIn("wins", episode)
                    self.assertEqual(episode["kills"], int(success))
                finally:
                    raw.close()

    def test_preserves_rng_policy_mode_on_prediction_error_and_timeout(self):
        module = self.module()
        for training in (False, True):
            for error in (RuntimeError("predict failed"), TimeoutError("expired")):
                policy = PrimitivePolicy(2, error=error)
                policy.policy.train(training)
                saved = self.snapshot_rng()
                with self.assertRaises(type(error)):
                    module.evaluate_suite(policy, model_name="cnn", stages=(1,), seeds=(101,),
                                          max_frames=12)
                self.assert_rng(saved)
                self.assertEqual(policy.policy.training, training)
        policy = PrimitivePolicy(2)
        saved = self.snapshot_rng()
        with patch.object(module, "RustEpisode") as constructor, self.assertRaises(TimeoutError):
            module.evaluate_suite(policy, model_name="cnn", max_frames=12,
                                  deadline=time.monotonic() - 1)
        constructor.assert_not_called()
        self.assert_rng(saved)

    def test_rejects_wrong_checkpoint_spaces_before_constructing_envs(self):
        module = self.module()
        for bad in ("stack", "actions"):
            policy = PrimitivePolicy(2)
            if bad == "actions":
                policy.action_space = spaces.Discrete(15)
            with patch.object(module, "RustEpisode") as constructor, self.assertRaises(ValueError):
                module.evaluate_suite(policy, model_name="history" if bad == "stack" else "cnn",
                                      max_frames=12)
            constructor.assert_not_called()

    def test_deadline_checked_inside_rom_reset_not_just_episode_boundary(self):
        module = self.module()
        raw = module.PrimitiveActions(BattleCityEnv(stage=1), deadline=100)
        self.addCleanup(raw.close)
        with patch.object(module.time, "monotonic", side_effect=[0, 0, 101]):
            with self.assertRaises(TimeoutError):
                raw.reset(seed=101)
        self.assertFalse(raw.env.nes._has_backup)

    def test_deadline_crossed_by_prediction_prevents_the_next_step(self):
        module = self.module()
        raw = module.PrimitiveActions(CounterEnv())
        self.addCleanup(raw.close)
        policy = PrimitivePolicy(2)
        player = module.EpisodePlayer(raw, model=policy, nstack=2,
                                      deterministic=True, deadline=100)
        with patch.object(module.time, "monotonic", return_value=0) as clock:
            player.reset(101)
            original = policy.predict

            def expire(*args, **kwargs):
                result = original(*args, **kwargs)
                clock.return_value = 101
                return result

            policy.predict = expire
            with self.assertRaises(TimeoutError):
                player.step()
        self.assertEqual(raw.env.frame, 1)

    def test_diagnostic_denominators_and_requests_are_not_shot_counts(self):
        module = self.module()
        state = gs.parse(stage_ram(1))
        before = replace(state, player=gs.Tank(0, 88, 24, gs.UP), fire_ready=False)
        after = replace(before, frame=4)
        diagnostics = module._Diagnostics()
        diagnostics.update(before, after, 6, {"kills": 0, "pickups": 0})
        diagnostics.update(after, replace(after, frame=6), 0, {"kills": 0, "pickups": 0})
        result = diagnostics.result()
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["sampled_frames"], 6)
        self.assertEqual(result["player_sample_count"], 2)
        self.assertEqual(result["stationary_samples"], 2)
        self.assertEqual(result["stationary_sample_rate"], 1)
        self.assertEqual(result["boundary_waste_requests"], 1)
        self.assertEqual(result["boundary_waste_frames"], 4)
        self.assertEqual(result["boundary_waste_rate"], 4 / 6)
        self.assertEqual(result["fire_requests"], 1)
        self.assertEqual(result["requested_unavailable_fire"], 1)
        self.assertEqual(result["requested_outward_fire"], 1)
        self.assertEqual(result["ineffective_fire_requests"], 1)
        self.assertEqual(result["ineffective_fire_request_rate"], 1)
        self.assertEqual(result["stagnation_frames"], 6)
        self.assertEqual(result["action_counts"], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    def test_formal_default_passes_12000_to_real_native_not_training_cap(self):
        module = self.module()
        policy = PrimitivePolicy(2, error=RuntimeError("stop before first step"))
        with patch.object(module, "RustEpisode", wraps=module.RustEpisode) as factory:
            with self.assertRaisesRegex(RuntimeError, "stop before first step"):
                module.evaluate_suite(policy, model_name="cnn", stages=(1,), seeds=(101,))
        self.assertEqual(factory.call_args.kwargs["max_frames"], 12000)

    def test_initialized_cuda_rng_restored_without_using_a_gpu(self):
        module = self.module()
        state = {"value": [torch.tensor([7], dtype=torch.uint8)]}

        def get_state():
            return [value.clone() for value in state["value"]]

        def set_state(value):
            state["value"] = value

        with patch.object(torch.cuda, "is_initialized", return_value=True), \
                patch.object(torch.cuda, "get_rng_state_all", side_effect=get_state), \
                patch.object(torch.cuda, "set_rng_state_all", side_effect=set_state), \
                patch.object(torch.cuda, "manual_seed_all",
                             side_effect=lambda seed: set_state([torch.tensor([seed % 256])])):
            policy = PrimitivePolicy(2, error=RuntimeError("stop"))
            with self.assertRaises(RuntimeError):
                module.evaluate_suite(policy, model_name="cnn", stages=(1,), seeds=(101,),
                                      max_frames=12)
        self.assertEqual(state["value"][0].item(), 7)


if __name__ == "__main__":
    unittest.main()
