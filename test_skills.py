"""Real cartridge controls: skills must be achievable, but never timeout wins."""
import importlib
import importlib.util
import json
import random
import unittest

import gymnasium as gym
import numpy as np

import game_state as gs


class SkillTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("skills"),
                             "the real-ROM skill benchmark is not implemented")
        return importlib.import_module("skills")

    def scene(self, skill, variant=0, seed=101, max_frames=600):
        env = self.module().SkillEnv(skill, variant=variant, max_frames=max_frames)
        self.addCleanup(env.close)
        observation, info = env.reset(seed=seed)
        self.assertEqual(observation.shape, (47, 26, 26))
        self.assertEqual(env.action_space.n, 15)
        self.assertFalse(info["skill_success"])
        self.assertEqual(info["frames"], 0)
        self.assertEqual(info["shots"], 0)
        self.assertEqual(info["deaths"], 0)
        self.assertFalse(env.state.shield)
        self.assertFalse(env.state.base_destroyed)
        return env

    def play(self, env, action):
        for decision in range(env.max_frames):
            _, _, terminated, truncated, info = env.step(
                action(decision) if callable(action) else action)
            if terminated or truncated:
                self.assertIn(info["outcome"], ("objective", "failure"))
                self.assertEqual(info["skill_success"], info["outcome"] == "objective")
                with self.assertRaises(gym.error.ResetNeeded):
                    env.step(0)
                return info
        self.fail("skill exceeded its frame budget")

    def test_six_scenes_have_geometry_variants_and_clean_reproducible_reset(self):
        module = self.module()
        self.assertEqual(module.SKILL_NAMES,
                         ("shoot", "dodge", "counter", "demolish", "pickup", "defend"))
        for skill in module.SKILL_NAMES:
            with self.subTest(skill=skill):
                env = self.scene(skill)
                initial = env._observation().copy()
                first_ram = env.nes.ram.copy()
                env.step(0)
                again, info = env.reset(seed=101)
                np.testing.assert_array_equal(initial, again)
                np.testing.assert_array_equal(first_ram, env.nes.ram)
                self.assertEqual((info["frames"], info["shots"], info["deaths"]), (0, 0, 0))
                other = self.scene(skill, variant=1)
                self.assertFalse(np.array_equal(initial[:31], other._observation()[:31]))

    def test_shoot_requires_a_real_kill_and_wrong_facing_does_not_pass(self):
        for variant, correct, wrong in ((0, 5, 9), (1, 5, 6)):
            with self.subTest(variant=variant):
                env = self.scene("shoot", variant)
                self.assertEqual(len(env.state.enemies), 1)
                info = self.play(env, correct)
                self.assertTrue(info["skill_success"], info)
                self.assertEqual(info["kills"], 1)
                self.assertGreater(info["shots"], 0)
                info = self.play(self.scene("shoot", variant, max_frames=120), wrong)
                self.assertFalse(info["skill_success"], info)

    def test_dodge_requires_lateral_evasion_and_meaningful_survival_time(self):
        for variant, direction in ((0, 3), (1, 1)):
            with self.subTest(variant=variant):
                env = self.scene("dodge", variant)
                incoming = next(b for b in env.state.bullets if b.owner >= 2)
                player = env.state.player
                offset = abs(incoming.x - player.x) if incoming.vy else abs(incoming.y - player.y)
                self.assertGreater(offset, 4)
                info = self.play(env, lambda i: direction if i < 8 else 0)
                self.assertTrue(info["skill_success"], info)
                self.assertGreaterEqual(info["frames"], 120)
                self.assertEqual(info["shots"], 0)
                failed = self.play(self.scene("dodge", variant), 0)
                self.assertFalse(failed["skill_success"])
                self.assertEqual(failed["deaths"], 1)

    def test_short_dodge_counts_when_the_live_shot_passes_outside_the_impact_lane(self):
        for variant, direction in ((0, 3), (1, 1)):
            for seed in (101, 202):
                with self.subTest(variant=variant, seed=seed):
                    env = self.scene("dodge", variant, seed=seed)
                    initial = env.state.player
                    info = self.play(env, lambda i: direction if i == 0 else 0)
                    player = env.state.player
                    distance = abs(player.x - initial.x) + abs(player.y - initial.y)
                    self.assertEqual(distance, 3)
                    self.assertEqual(info["deaths"], 0)
                    self.assertEqual(info["shots"], 0)
                    self.assertTrue(info["skill_success"], info)
                    self.assertEqual(info["frames"], 120)

    def test_two_pixel_dodge_clears_the_rom_hitbox_but_one_pixel_does_not(self):
        for variant, direction in ((0, 3), (1, 1)):
            with self.subTest(variant=variant):
                env = self.scene("dodge", variant)
                env.frame_skip = 1
                initial = env.state.player
                info = self.play(env, lambda i: direction if i < 3 else 0)
                player = env.state.player
                self.assertEqual(abs(player.x - initial.x) + abs(player.y - initial.y), 2)
                self.assertTrue(info["skill_success"], info)
                self.assertEqual(info["frames"], 120)
                failed_env = self.scene("dodge", variant)
                failed_env.frame_skip = 1
                failed = self.play(failed_env, lambda i: direction if i < 2 else 0)
                self.assertFalse(failed["skill_success"])
                self.assertEqual(failed["deaths"], 1)

    def test_dodge_cannot_credit_a_shot_blocked_before_it_passes_the_player(self):
        for variant, direction in ((0, 3), (1, 1)):
            with self.subTest(variant=variant):
                env = self.scene("dodge", variant)
                grid = env.nes.ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
                if variant == 0:
                    grid[10, 9:11] = 16
                else:
                    grid[9:11, 10] = 16
                env._tracker.reset(env.nes.ram)
                info = self.play(env, lambda i: direction if i < 8 else 0)
                self.assertEqual(info["deaths"], 0)
                self.assertFalse(info["skill_success"], info)

    def test_counter_requires_pairwise_bullet_cancellation_not_dodging(self):
        for variant, dodge in ((0, 3), (1, 1)):
            with self.subTest(variant=variant):
                env = self.scene("counter", variant)
                info = self.play(env, 5)
                self.assertTrue(info["skill_success"], info)
                self.assertGreater(info["shots"], 0)
                self.assertEqual(info["deaths"], 0)
                evidence = info["interception"]
                self.assertEqual(evidence["enemy_slot"], 2)
                self.assertIn(evidence["player_slot"], (0, 8))
                self.assertGreater(evidence["frame"], 0)
                self.assertEqual(evidence["enemy_status_after"], 0)
                self.assertEqual(evidence["player_status_after"], 0)
                failed = self.play(self.scene("counter", variant),
                                   lambda i: dodge if i < 8 else 0)
                self.assertFalse(failed["skill_success"], failed)
                self.assertEqual(failed["deaths"], 0)
                self.assertIsNone(failed["interception"])

    def test_counter_counts_a_shot_created_and_cancelled_in_the_same_rom_frame(self):
        for variant in (0, 1):
            with self.subTest(variant=variant):
                env = self.scene("counter", variant)
                info = self.play(env, lambda i: 0 if i < 8 else 5)
                self.assertTrue(info["skill_success"], info)
                self.assertEqual(info["shots"], 1)
                self.assertEqual(info["deaths"], 0)
                self.assertTrue(info["interception"]["same_frame_shot"])
                self.assertEqual(info["interception"]["enemy_status_after"], 0)
                self.assertEqual(info["interception"]["player_status_after"], 0)

    def test_perpendicular_projectile_cancellation_counts_as_counter_not_dodge(self):
        for skill, actions, frame, immediate in (
            ("counter", [3] * 5 + [4] + [0] * 2 + [5], 36, False),
            ("counter", [3] * 5 + [4] + [0] * 3 + [5], 38, True),
            ("dodge", [3] * 6 + [4] + [5] + [0] * 22, 36, False),
        ):
            with self.subTest(skill=skill, immediate=immediate):
                env = self.scene(skill)
                info = self.play(env, lambda i: actions[i] if i < len(actions) else 0)
                self.assertEqual(info["shots"], 1)
                self.assertEqual(info["deaths"], 0)
                self.assertEqual(info["skill_success"], skill == "counter", info)
                self.assertIsNotNone(info["interception"])
                self.assertEqual(info["interception"]["frame"], frame)
                self.assertEqual(info["interception"]["same_frame_shot"], immediate)
                self.assertEqual(info["interception"]["enemy_status_after"], 0)
                self.assertEqual(info["interception"]["player_status_after"], 0)

    def test_counter_does_not_credit_an_enemy_bullet_hitting_a_wall(self):
        env = self.scene("counter")
        grid = env.nes.ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
        grid[10, 8:10] = 16
        env._tracker.reset(env.nes.ram)
        info = self.play(env, 5)
        self.assertFalse(info["skill_success"], info)
        self.assertEqual(info["reason"], "target_not_intercepted")
        self.assertIsNone(info["interception"])
        self.assertGreater(info["shots"], 0)
        self.assertEqual(info["deaths"], 0)

    def test_demolition_cannot_be_bypassed_and_requires_destroyed_partial_bricks(self):
        for variant, direction, firing in ((0, 4, 9), (1, 2, 7)):
            with self.subTest(variant=variant):
                env = self.scene("demolish", variant)
                partial = (env.state.brick_masks > 0) & (env.state.brick_masks < 15)
                self.assertTrue(partial.any())
                info = self.play(env, firing)
                self.assertTrue(info["skill_success"], info)
                self.assertGreater(info["shots"], 0)
                self.assertTrue(np.all(env.state.brick_masks[partial] == 0))
                failed = self.play(self.scene("demolish", variant, max_frames=120), direction)
                self.assertFalse(failed["skill_success"])
                self.assertEqual(failed["shots"], 0)

    def test_pickup_requires_distant_item_collection_and_real_power_change(self):
        for variant, action in ((0, 4), (1, 2)):
            with self.subTest(variant=variant):
                env = self.scene("pickup", variant)
                player, item = env.state.player, env.state.item
                self.assertGreater(abs(player.x - item.x) + abs(player.y - item.y), 64)
                self.assertEqual(item.kind, 3)
                info = self.play(env, action)
                self.assertTrue(info["skill_success"], info)
                self.assertEqual(info["pickups"], 1)
                self.assertEqual(env.state.player.power, 1)

    def test_defend_requires_enemy_neutralization_and_surviving_its_live_shot(self):
        for variant in (0, 1):
            with self.subTest(variant=variant):
                env = self.scene("defend", variant)
                self.assertEqual(len(env.state.enemies), 1)
                self.assertEqual(len(env.state.bullets), 1)
                info = self.play(env, 5)
                self.assertTrue(info["skill_success"], info)
                self.assertEqual(info["kills"], 1)
                self.assertFalse(env.state.base_destroyed)
                self.assertFalse(any(b.owner >= 2 for b in env.state.bullets))
                failed_env = self.scene("defend", variant)
                failed = self.play(failed_env, lambda i: 3 if i < 8 else 0)
                self.assertFalse(failed["skill_success"], failed)
                self.assertTrue(failed_env.state.base_destroyed)

    def test_noop_never_passes_any_scene_and_short_budget_is_failure(self):
        for skill in self.module().SKILL_NAMES:
            with self.subTest(skill=skill):
                result = self.play(self.scene(skill, max_frames=4), 0)
                self.assertFalse(result["skill_success"])
                self.assertEqual(result["reason"], "timeout")
                result = self.play(self.scene(skill), 0)
                self.assertFalse(result["skill_success"], result)

    def test_invalid_scene_parameters_fail_before_allocating_emulator(self):
        module = self.module()
        for args in (("unknown",), ("shoot", -1), ("shoot", 2), ("shoot", True),
                     ("shoot", 0, 0), ("shoot", 0, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.SkillEnv(*args)

    def test_evaluation_keeps_objectives_separate_and_restores_random_state(self):
        module = self.module()
        import torch

        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        py_state, np_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        result = module.evaluate_skills(seeds=(101,), variants=(0,), skills=("shoot", "counter"))
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        json.dumps(result, allow_nan=False)
        self.assertEqual(set(result["modes"]), {"deterministic", "sampled"})
        for mode in result["modes"].values():
            self.assertEqual(set(mode["per_skill"]), {"shoot", "counter"})
            episodes = mode["episodes"]
            self.assertEqual(len(episodes), 2)
            for episode in episodes:
                self.assertEqual(episode["seed"], 101)
                self.assertEqual(episode["variant"], 0)
                self.assertIsInstance(episode["skill_success"], bool)
                self.assertIn(episode["outcome"], ("objective", "failure"))
                self.assertNotIn("wins", episode)
                self.assertTrue(episode["reason"])
                for field in ("frames", "shots", "deaths"):
                    self.assertIsInstance(episode[field], int)
            successes = sum(e["skill_success"] for e in episodes)
            self.assertEqual(mode["summary"]["mean_success_rate"], successes / 2)
            self.assertNotIn("wins", mode["summary"])
            for skill in ("shoot", "counter"):
                self.assertEqual(mode["per_skill"][skill]["episodes"], 1)

    def test_model_evaluation_uses_native_stacking_masks_and_both_prediction_modes(self):
        class FirePolicy:
            def __init__(self):
                self.inputs = []

            def predict(self, observation, *, deterministic, action_masks):
                self.inputs.append((observation.copy(), deterministic, action_masks.copy()))
                return np.array(5), None

        policy = FirePolicy()
        result = self.module().evaluate_skills(model=policy, seeds=(202,), variants=(0,),
                                               nstack=2, skills=("shoot",))
        self.assertEqual(result["policy"], "MaskablePPO")
        self.assertEqual({entry[1] for entry in policy.inputs}, {True, False})
        for observation, _, masks in policy.inputs:
            self.assertEqual(observation.shape, (94, 26, 26))
            self.assertEqual(masks.shape, (15,))
            self.assertEqual(masks.dtype, np.bool_)
        self.assertFalse(policy.inputs[0][0][:47].any())
        self.assertTrue(policy.inputs[0][0][47:].any())
        for mode in result["modes"].values():
            self.assertEqual(mode["summary"]["mean_success_rate"], 1)
            self.assertEqual(mode["episodes"][0]["outcome"], "objective")

    def test_evaluation_rejects_empty_duplicate_or_invalid_dimensions(self):
        module = self.module()
        for kwargs in ({"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (-1,)},
                       {"seeds": (True,)}, {"variants": ()}, {"variants": (2,)},
                       {"skills": ()}, {"skills": ("shoot", "shoot")},
                       {"skills": ("other",)}, {"nstack": 0}, {"nstack": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                module.evaluate_skills(**kwargs)


if __name__ == "__main__":
    unittest.main()
