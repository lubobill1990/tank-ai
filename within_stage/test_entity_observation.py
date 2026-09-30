"""Synthetic entity geometry/history tests; no real emulator calls."""
from dataclasses import replace
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import game_state as gs
from within_stage.observation import encode_precision
from within_stage import entity_observation as entity


def fixture_ram():
    ram = np.zeros(2048, np.uint8)
    ram[0x68], ram[0x51], ram[0x80], ram[0x85] = 128, 3, 20, 1
    ram[0x90], ram[0x98], ram[0xA0], ram[0x101] = 82, 160, 0x80, 0x40
    ram[0x92], ram[0x9A], ram[0xA2], ram[0xAA] = 92, 80, 0xA2, 0xE4
    ram[0x93], ram[0x9B], ram[0xA3], ram[0xAB] = 40, 64, 0xA3, 0x20
    ram[0xB8], ram[0xC2], ram[0xCC], ram[0xD6] = 82, 140, 0x40, 1
    ram[0xBA], ram[0xC4], ram[0xCE] = 92, 100, 0x42
    ram[0x86], ram[0x87], ram[0x88] = 105, 110, 3
    return ram


class FakeRaw(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    def __init__(self):
        self.state = gs.StateTracker().reset(fixture_ram())
        self.obs = np.zeros((47, 26, 26), np.float32)
        self.info = {'nested': {'value': 1}}
        self.reset_arguments = []
    def reset(self, *, seed=None, options=None):
        self.reset_arguments.append((seed, options))
        self.obs.fill(.1)
        return self.obs, self.info
    def step(self, action):
        self.obs.fill(.2)
        return self.obs, np.float32(-.002), True, False, self.info
    def action_masks(self):
        return np.ones(10, bool)


class EntityObservationTests(unittest.TestCase):
    def test_named_geometry_fields_preserve_subcell_position_and_relations(self):
        state = gs.StateTracker().reset(fixture_ram())
        tokens = entity.encode_entities(state)
        fields = {name: index for index, name in enumerate(entity.ENTITY_FIELDS)}
        self.assertEqual(tokens.shape, (18, 16))
        self.assertEqual(tokens.dtype, np.float32)
        player = tokens[0]
        self.assertEqual(player[fields['valid']], 1)
        self.assertAlmostEqual(player[fields['x_over_255']], 82 / 255, places=7)
        self.assertEqual(player[fields['relative_player_x_over_255']], 0)
        self.assertAlmostEqual(player[fields['relative_base_y_over_255']], (160 - 216) / 255, places=7)
        self.assertAlmostEqual(player[fields['power_over_3']], 2 / 3, places=7)
        enemies = tokens[1:7]
        enemy = enemies[np.argmax(enemies[:, fields['armor_over_4']])]
        self.assertAlmostEqual(enemy[fields['relative_player_x_over_255']], 10 / 255, places=7)
        self.assertEqual(enemy[fields['carrier_or_bullet_team']], 1)
        self.assertEqual(enemy[fields['armor_over_4']], 1)
        bullets = tokens[7:17]
        self.assertEqual(set(bullets[bullets[:, 0] == 1, fields['carrier_or_bullet_team']]), {-1, 1})
        item = tokens[17]
        self.assertAlmostEqual(item[fields['item_kind_plus_one_over_6']], 4 / 6, places=7)
        changed = replace(state, enemies=(replace(state.enemies[0], x=94), state.enemies[1]))
        self.assertFalse(np.array_equal(tokens, entity.encode_entities(changed)))

    def test_order_slot_ids_generations_and_unrelated_counters_cannot_change_tokens(self):
        state = gs.StateTracker().reset(fixture_ram())
        changed = replace(
            state, enemies=tuple(replace(e, slot=7 - i) for i, e in enumerate(reversed(state.enemies))),
            bullets=tuple(replace(b, slot=9 - i, generation=999) for i, b in enumerate(reversed(state.bullets))),
            frame=9876, kills=19, score=999999, spawn_left=0, remaining_enemies=1, lives=9)
        np.testing.assert_array_equal(entity.encode_entities(state), entity.encode_entities(changed))
        tokens = entity.encode_entities(state)
        np.testing.assert_array_equal(tokens[3:7], 0)
        np.testing.assert_array_equal(tokens[9:17], 0)

    def test_missing_player_and_empty_entities_are_finite_and_zero_padded(self):
        state = gs.StateTracker().reset(fixture_ram())
        missing = replace(state, player=None)
        tokens = entity.encode_entities(missing)
        np.testing.assert_array_equal(tokens[0], 0)
        np.testing.assert_array_equal(tokens[:, 4:6], 0)
        self.assertTrue(np.isfinite(tokens).all())
        empty = replace(missing, enemies=(), bullets=(), item=None)
        np.testing.assert_array_equal(entity.encode_entities(empty), np.zeros((18, 16), np.float32))

    def test_ram_conversion_matches_live_encoding_and_keeps_audit_ram_out_of_model_input(self):
        ram = fixture_ram()
        state = gs.StateTracker().reset(ram)
        expected = entity.encode_entities(state)
        np.testing.assert_array_equal(entity.entities_from_ram(ram), expected)
        grids = np.zeros((2, 110, 26, 26), np.float32)
        result = entity.entity_batch_from_ram(grids, np.stack([ram, ram]))
        self.assertEqual(set(result), {'grid', 'entities'})
        np.testing.assert_array_equal(result['entities'], np.stack([expected, expected]))
        result['grid'].fill(1)
        np.testing.assert_array_equal(grids, 0)
        self.assertEqual(ram[0x90], 82)

    def test_wrapper_builds_precision_history_itself_and_returns_owned_data(self):
        raw = FakeRaw()
        wrapped = entity.EntityObservationEnv(raw)
        self.addCleanup(wrapped.close)
        reset, info = wrapped.reset(seed=123, options={'choice': 1})
        initial = encode_precision(raw.state, raw.obs)
        np.testing.assert_array_equal(reset['grid'][:55], 0)
        np.testing.assert_array_equal(reset['grid'][55:], initial)
        next_obs, reward, done, truncated, _ = wrapped.step(2)
        np.testing.assert_array_equal(next_obs['grid'][:55], initial)
        np.testing.assert_array_equal(reset['grid'][55:], initial)
        self.assertEqual(reward.tobytes(), np.float32(-.002).tobytes())
        self.assertTrue(done); self.assertFalse(truncated)
        info['nested']['value'] = 9
        self.assertEqual(raw.info['nested']['value'], 1)
        restarted, _ = wrapped.reset(seed=456)
        np.testing.assert_array_equal(restarted['grid'][:55], 0)
        self.assertEqual(raw.reset_arguments, [(123, {'choice': 1}), (456, None)])
        self.assertEqual(wrapped.action_masks().shape, (10,))
        self.assertTrue(wrapped.observation_space.contains(restarted))

    def test_invalid_entity_counts_and_raw_contracts_raise_without_silent_clipping(self):
        state = gs.StateTracker().reset(fixture_ram())
        with self.assertRaises(ValueError):
            entity.encode_entities(replace(state, enemies=state.enemies * 4))
        with self.assertRaises(ValueError):
            entity.encode_entities(replace(state, bullets=state.bullets * 6))
        with self.assertRaises(ValueError):
            entity.encode_entities(replace(state, player=replace(state.player, x=300)))
        with self.assertRaises(ValueError):
            entity.entities_from_ram(np.zeros(2048, np.float32))
        raw = FakeRaw(); raw.observation_space = spaces.Box(-1, 1, (55, 26, 26), np.float32)
        with self.assertRaises(ValueError):
            entity.EntityObservationEnv(raw)

    def test_contract_returns_owned_descriptions_and_p2_allegiance_is_distinct(self):
        schema = entity.entity_schema()
        schema['fields'][0] = 'corrupt'
        self.assertEqual(entity.entity_schema()['fields'][0], 'valid')
        state = gs.StateTracker().reset(fixture_ram())
        p2 = replace(state.bullets[0], owner=1)
        tokens = entity.encode_entities(replace(state, bullets=(p2,)))
        self.assertEqual(tokens[7, 15], -.5)
        self.assertEqual(tokens[7, 10], 0)
        self.assertEqual(tokens[7, 11], -1)
        self.assertEqual(tokens[17, 8], 0)
        self.assertEqual(tokens[17, 9], 0)

    def test_monitor_dummy_vec_supports_dict_history_without_another_frame_stack(self):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        raw = FakeRaw()
        wrapped = entity.EntityObservationEnv(raw)
        vec = DummyVecEnv([lambda: Monitor(wrapped)])
        self.addCleanup(vec.close)
        first = vec.reset()
        self.assertEqual(first['grid'].shape, (1, 110, 26, 26))
        self.assertEqual(first['entities'].shape, (1, 18, 16))
        next_obs, _, dones, infos = vec.step(np.array([0]))
        self.assertTrue(dones[0])
        terminal = infos[0]['terminal_observation']
        self.assertEqual(set(terminal), {'grid', 'entities'})
        np.testing.assert_array_equal(terminal['grid'][:55], first['grid'][0, 55:])
        np.testing.assert_array_equal(next_obs['grid'][0, :55], 0)
        self.assertEqual(vec.env_method('action_masks')[0].shape, (10,))

if __name__ == '__main__':
    unittest.main()
