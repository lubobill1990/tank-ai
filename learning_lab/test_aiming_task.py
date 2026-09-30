"""Synthetic aiming-template fixtures; no ROM/native environment decisions."""
import unittest
from unittest.mock import patch

import numpy as np

import game_state as gs
from learning_lab import aiming_task as aiming


def base_ram():
    ram = np.zeros(2048, np.uint8)
    grid = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
    grid[24:26, 12:14] = ((200, 201), (202, 203))
    ram[0x68], ram[0x51], ram[0x85] = 0x80, 3, 1
    ram[0x80], ram[0x100] = 1, 10
    ram[0xa0], ram[0xa2], ram[0xaa] = 0x80, 0xa2, 0x80
    ram[0x90], ram[0x98], ram[0x92], ram[0x9a] = 88, 160, 88, 80
    return ram


class FakeSupport:
    def __init__(self, initial, max_frames, *, seeds):
        self.initial = initial.copy()
        self.support = np.zeros(len(initial), np.uint32)

    def masks(self):
        return np.ones((len(self.initial), 10), bool)

    def ram(self, index=0):
        return self.initial[index].copy()

    def close(self):
        pass


class AimingTaskTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("_shoot_reset_ram", lambda: base_ram()), ("NativeBatch", FakeSupport)):
            item = patch.object(aiming, name, value)
            item.start()
            self.addCleanup(item.stop)

    def test_balanced_directions_facings_subcells_and_valid_bounds(self):
        data = aiming.generate_aiming_starts(32, 311)
        self.assertEqual(data["rams"].shape, (32, 2048))
        self.assertEqual(data["rams"].dtype, np.uint8)
        np.testing.assert_array_equal(np.bincount(data["directions"], minlength=4), [8] * 4)
        np.testing.assert_array_equal(np.bincount(data["player_facings"], minlength=4), [8] * 4)
        for direction in range(4):
            offsets = data["subcell_offsets"][data["directions"] == direction]
            self.assertEqual(sorted(offsets.tolist()), [0, 0, 2, 2, 4, 4, 6, 6])
            for key in ("player_facings", "enemy_facings"):
                np.testing.assert_array_equal(
                    np.bincount(data[key][data["directions"] == direction], minlength=4), [2] * 4)
        for row in range(32):
            state = gs.parse(data["rams"][row])
            player, enemy = state.player, state.enemies[0]
            dx, dy = gs.DXY[int(data["directions"][row])]
            self.assertEqual((enemy.x - player.x, enemy.y - player.y),
                             (dx * data["distances"][row], dy * data["distances"][row]))
            self.assertTrue(48 <= data["distances"][row] <= 96)
            self.assertTrue(all(24 <= v <= 216 for v in (player.x, player.y, enemy.x, enemy.y)))
            perpendicular = player.x if dy else player.y
            self.assertEqual((perpendicular - gs.ARENA) % 8, data["subcell_offsets"][row])
            self.assertTrue(aiming.valid_scene(data["rams"][row]))

    def test_freeze_basic_enemy_and_no_initial_bullets_or_pending_spawns(self):
        data = aiming.generate_aiming_starts(32, 312)
        for ram in data["rams"]:
            state = gs.parse(ram)
            self.assertEqual(len(state.enemies), 1)
            self.assertEqual(state.enemies[0].armor, 1)
            self.assertFalse(state.enemies[0].drops_item)
            self.assertFalse(state.bullets)
            self.assertEqual((state.remaining_enemies, state.spawn_left), (1, 0))
            self.assertTrue(state.freeze and state.fire_ready)
            self.assertFalse(state.base_destroyed)
            self.assertGreater((int(ram[0x100]) - 1) * 64, 200)

    def test_seed_reproducibility_disjoint_holdout_and_overlap_rejection(self):
        train = aiming.generate_aiming_starts(32, 311)
        repeat = aiming.generate_aiming_starts(32, 311)
        test = aiming.generate_aiming_starts(128, 947)
        np.testing.assert_array_equal(train["rams"], repeat["rams"])
        aiming.validate_split(train, test)
        with self.assertRaises(ValueError):
            aiming.validate_split(train, repeat)
        forged = dict(test, hashes=test["hashes"].copy())
        forged["hashes"][0] = train["hashes"][0]
        with self.assertRaises(ValueError):
            aiming.validate_split(train, forged)
        self.assertEqual(len(set(train["hashes"])), 32)
        self.assertEqual(len(set(test["hashes"])), 128)

    def test_hashes_and_metadata_are_owned_and_training_only(self):
        first = aiming.generate_aiming_starts(16, 51)
        original_hashes = first["hashes"].copy()
        first["rams"].fill(0)
        second = aiming.generate_aiming_starts(16, 51)
        np.testing.assert_array_equal(second["hashes"], original_hashes)
        self.assertEqual(second["metadata"]["scene_version"], aiming.SCENE_VERSION)
        self.assertFalse(second["metadata"]["full_game"])
        self.assertEqual(second["metadata"]["rom_boot_seed"], 0)
        self.assertEqual(second["metadata"]["native_decisions"], 0)

    def test_oracle_turns_without_wrong_heading_fire_then_fires_legally(self):
        initial = base_ram()
        initial[0xa0] = 0x83
        state = gs.parse(initial)
        masks = np.ones(10, bool)
        self.assertEqual(aiming.oracle_action(state, gs.UP, masks), 1)
        initial[0xa0] = 0x80
        self.assertEqual(aiming.oracle_action(gs.parse(initial), gs.UP, masks), 5)
        masks[5] = False
        self.assertEqual(aiming.oracle_action(gs.parse(initial), gs.UP, masks), 0)

    def test_invalid_counts_and_base_crossing_geometry_are_rejected(self):
        for count in (0, 7, 15, 17, True):
            with self.assertRaises(ValueError):
                aiming.generate_aiming_starts(count, 0)
        initial = base_ram()
        initial[0x90], initial[0x98] = 72, 216
        initial[0x92], initial[0x9a] = 168, 216
        self.assertFalse(aiming.valid_scene(initial))

    def test_training_factory_uses_existing_fixed_template_vector_env(self):
        data = aiming.generate_aiming_starts(16, 11)
        marker = object()
        with patch.object(aiming, "RustVectorEnv", return_value=marker) as constructor:
            self.assertIs(aiming.make_aiming_env(data, seed=91), marker)
        supplied = constructor.call_args.args[0]
        np.testing.assert_array_equal(supplied, data["rams"])
        self.assertFalse(np.shares_memory(supplied, data["rams"]))
        self.assertEqual(constructor.call_args.args[1], 200)
        self.assertEqual(constructor.call_args.kwargs, {"seed": 91})


if __name__ == "__main__":
    unittest.main()
