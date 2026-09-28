import unittest

import numpy as np

import game_state as gs


def ram_fixture():
    ram = np.zeros(2048, dtype=np.uint8)
    ram[0x68] = 0x80
    ram[0x51] = 3
    ram[0x80] = 20
    ram[0x7f] = 18
    ram[0x85] = 1
    ram[0xa0] = 0x80
    ram[0x90], ram[0x98] = 88, 216
    return ram


class SnapshotTests(unittest.TestCase):
    def test_tank_occupancy_bit_never_becomes_a_terrain_wall(self):
        ram = ram_fixture()
        for base in (*range(16), 16, 18, 32, 33, 34):
            with self.subTest(base=base):
                ram[0x442] = base
                plain = gs.parse(ram)
                ram[0x442] = base | 0x80
                occupied = gs.parse(ram)
                np.testing.assert_array_equal(plain.terrain, occupied.terrain)
                np.testing.assert_array_equal(plain.brick_masks, occupied.brick_masks)
                self.assertEqual(gs.decode_material(base | 0x80), gs.decode_material(base))

    def test_occupancy_normalization_preserves_base_and_unknown_bytes(self):
        for value in range(200, 208):
            self.assertEqual(gs.decode_material(value), gs.T_BASE)
        with self.assertWarnsRegex(RuntimeWarning, '0xFF'):
            self.assertEqual(gs.decode_material(255), gs.T_UNKNOWN)

    def test_enemy_drop_identity_is_decoded_independently_of_armor(self):
        ram = ram_fixture()
        ram[0xa2], ram[0x92], ram[0x9a] = 0xa2, 88, 100
        for attribute, carrier in ((0x80, False), (0x84, True), (0xa4, True),
                                   (0xc4, True), (0xe4, True), (0xe2, False)):
            with self.subTest(attribute=attribute):
                ram[0xaa] = attribute
                tank = gs.parse(ram).enemies[0]
                self.assertTrue(hasattr(tank, 'drops_item'))
                self.assertEqual(tank.drops_item, carrier)

    def test_every_residual_mask_is_preserved_but_all_still_block_tanks(self):
        ram = ram_fixture()
        for mask in range(16):
            with self.subTest(mask=mask):
                ram[0x442] = mask
                state = gs.parse(ram)
                self.assertTrue(hasattr(state, 'brick_masks'))
                self.assertEqual(state.brick_masks[0, 0], mask)
                self.assertEqual(gs.can_drive(state.terrain[0, 0]), mask == 0)
                self.assertFalse(state.brick_masks.flags.writeable)
        for value in (16, 18, 32, 33, 34, 200):
            ram[0x442] = value
            self.assertEqual(gs.parse(ram).brick_masks[0, 0], 0)

    def test_materials_preserve_native_water_ice_and_forest_semantics(self):
        ram = ram_fixture()
        for value, material, drive, shoot in (
            (15, gs.T_BRICK, False, False), (16, gs.T_STEEL, False, False),
            (18, gs.T_WATER, False, True), (33, gs.T_ICE, True, True),
            (32, gs.T_EMPTY, True, True), (34, gs.T_EMPTY, True, True),
            (200, gs.T_BASE, False, False), (204, gs.T_BASE, False, False),
            (207, gs.T_BASE, False, False),
        ):
            with self.subTest(value=value):
                ram[0x442] = value
                state = gs.parse(ram)
                self.assertEqual(state.terrain[0, 0], material)
                self.assertEqual(gs.can_drive(material), drive)
                self.assertEqual(gs.can_shoot_through(material), shoot)
                self.assertFalse(state.terrain.flags.writeable)

    def test_inactive_bullet_coordinates_are_not_projectiles(self):
        ram = ram_fixture()
        ram[0xb8], ram[0xc2] = 88, 24
        self.assertEqual(len(gs.parse(ram).bullets), 0)

    def test_inactive_and_spawning_tanks_are_not_live_enemies(self):
        ram = ram_fixture()
        ram[0x92:0x95] = [24, 120, 216]
        ram[0x9a:0x9d] = [24, 24, 24]
        ram[0xa2:0xa5] = [0, 0xe3, 0x73]
        self.assertEqual(len(gs.parse(ram).enemies), 0)

    def test_terrain_is_native_26_grid(self):
        state = gs.parse(ram_fixture())
        self.assertEqual(np.shape(state.terrain), (26, 26))

    def test_second_player_shot_has_player_ownership_and_real_velocity(self):
        ram = ram_fixture()
        ram[0xd4], ram[0xc0], ram[0xca], ram[0xde] = 0x41, 88, 100, 3
        b = gs.parse(ram).bullets[0]
        self.assertEqual((b.slot, b.owner, b.vx, b.vy), (8, 0, -4, 0))

    def test_score_is_not_kills_and_item_flag_is_not_loop_index(self):
        ram = ram_fixture()
        ram[0x19] = 5
        ram[0x49] = 0xff
        state = gs.parse(ram)
        self.assertEqual(state.score, 500)
        self.assertEqual(state.kills, 0)
        self.assertIsNone(state.item)
        ram[0x86], ram[0x87], ram[0x88] = 88, 144, 3
        self.assertEqual(gs.parse(ram).item.kind, 3)
        ram[0x62] = 20
        self.assertIsNone(gs.parse(ram).item)

    def test_fire_capacity_uses_status_and_power(self):
        ram = ram_fixture()
        ram[0xb8] = 88
        self.assertTrue(gs.parse(ram).fire_ready)
        ram[0xcc] = 0x40
        self.assertFalse(gs.parse(ram).fire_ready)
        ram[0x101] = 0x40
        self.assertTrue(gs.parse(ram).fire_ready)
        ram[0xd4] = 0x33
        self.assertFalse(gs.parse(ram).fire_ready)

    def test_flashing_heavy_tank_has_full_armor_before_item_drop(self):
        ram = ram_fixture()
        ram[0xa2], ram[0x92], ram[0x9a] = 0xa2, 88, 100
        for attribute, hits_left in ((0xe4, 4), (0xe3, 4), (0xe2, 3),
                                     (0xe1, 2), (0xe0, 1), (0x84, 1)):
            with self.subTest(attribute=attribute):
                ram[0xaa] = attribute
                self.assertEqual(gs.parse(ram).enemies[0].armor, hits_left)


class TransitionTests(unittest.TestCase):
    def test_occupancy_flip_does_not_change_material_or_remaining_bricks(self):
        ram = ram_fixture()
        ram[0x442] = 9
        tracker = gs.StateTracker()
        plain = tracker.reset(ram)
        ram[0x442] = 0x89
        occupied, _ = tracker.advance(ram)
        np.testing.assert_array_equal(plain.terrain, occupied.terrain)
        np.testing.assert_array_equal(plain.brick_masks, occupied.brick_masks)
        ram[0x442] = 8
        damaged, _ = tracker.advance(ram)
        self.assertEqual(damaged.brick_masks[0, 0], 8)

    def test_red_heavy_first_hit_records_drop_before_any_kill(self):
        ram = ram_fixture()
        ram[0xa2], ram[0x92], ram[0x9a], ram[0xaa] = 0xa2, 88, 100, 0xe4
        tracker = gs.StateTracker()
        tracker.reset(ram)
        ram[0xaa] = 0xe2
        ram[0x86], ram[0x87], ram[0x88] = 96, 192, 3
        _, events = tracker.advance(ram)
        self.assertTrue(hasattr(events, 'item_drops'))
        self.assertEqual((events.kills, events.item_drops, events.pickups), (0, 1, ()))
        self.assertEqual(tracker.advance(ram)[1].item_drops, 0)

    def test_residual_updates_are_cached_with_material_but_not_lost_when_still_brick(self):
        ram = ram_fixture()
        ram[0x442] = 15
        tracker = gs.StateTracker()
        first = tracker.reset(ram)
        self.assertTrue(hasattr(first, 'brick_masks'))
        same, _ = tracker.advance(ram)
        self.assertIs(first.brick_masks, same.brick_masks)
        ram[0x442] = 1
        changed, _ = tracker.advance(ram)
        self.assertEqual(changed.terrain[0, 0], first.terrain[0, 0])
        self.assertEqual(changed.brick_masks[0, 0], 1)
        self.assertEqual(first.brick_masks[0, 0], 15)
        self.assertIsNot(changed.brick_masks, first.brick_masks)

    def test_death_is_immediate_not_delayed_lives_decrement(self):
        ram = ram_fixture()
        tracker = gs.StateTracker()
        tracker.reset(ram)
        ram[0xa0] = 0x73
        state, events = tracker.advance(ram)
        self.assertEqual(events.deaths, 1)
        self.assertEqual(state.lives, 3)
        self.assertIsNone(state.player)
        ram[0xa0] = 0x62
        ram[0x51] = 2
        _, events = tracker.advance(ram)
        self.assertEqual(events.deaths, 0)

    def test_kill_and_pickup_can_coexist_without_score_heuristics(self):
        ram = ram_fixture()
        ram[0xa2], ram[0x92], ram[0x9a] = 0xa2, 88, 160
        ram[0x86], ram[0x87], ram[0x88] = 88, 216, 4
        tracker = gs.StateTracker()
        tracker.reset(ram)
        ram[0xa2], ram[0x62], ram[0x49], ram[0x19] = 0x73, 49, 0, 5
        state, events = tracker.advance(ram)
        self.assertEqual(events.kills, 1)
        self.assertEqual(events.pickups, (4,))
        self.assertEqual(state.kills, 1)
        _, events = tracker.advance(ram)
        self.assertEqual((events.kills, events.pickups), (0, ()))

    def test_shot_pulse_detects_same_frame_destroyed_projectile_once(self):
        ram = ram_fixture()
        tracker = gs.StateTracker()
        tracker.reset(ram)
        ram[0x30f] = 1
        _, events = tracker.advance(ram)
        self.assertEqual(len(events.shots), 1)
        _, events = tracker.advance(ram)
        self.assertEqual(len(events.shots), 0)

    def test_bullet_generation_survives_motion_and_changes_on_reuse(self):
        ram = ram_fixture()
        ram[0xce], ram[0xba], ram[0xc4] = 0x42, 88, 80
        tracker = gs.StateTracker()
        first = tracker.reset(ram).bullets[0]
        ram[0xc4] = 82
        state, _ = tracker.advance(ram)
        self.assertEqual(state.bullets[0].generation, first.generation)
        self.assertEqual((state.bullets[0].vx, state.bullets[0].vy), (0, 2))
        ram[0xce] = 0x33
        self.assertEqual(tracker.advance(ram)[0].bullets, ())
        ram[0xce], ram[0xc4] = 0x42, 80
        state, _ = tracker.advance(ram)
        self.assertNotEqual(state.bullets[0].generation, first.generation)

    def test_player_secondary_transfer_preserves_existing_bullet_identity(self):
        ram = ram_fixture()
        ram[0xcc], ram[0xb8], ram[0xc2] = 0x40, 88, 100
        tracker = gs.StateTracker()
        first = tracker.reset(ram).bullets[0]
        ram[0xd4], ram[0xc0], ram[0xca] = 0x40, 88, 98
        ram[0xc2], ram[0x30f] = 208, 1
        state, events = tracker.advance(ram)
        by_slot = {b.slot: b for b in state.bullets}
        self.assertEqual(by_slot[8].generation, first.generation)
        self.assertNotEqual(by_slot[0].generation, first.generation)
        self.assertEqual(len(events.shots), 1)


class FootprintTests(unittest.TestCase):
    def test_spawn_and_edges(self):
        self.assertEqual(set(gs.tank_cells(88, 216)),
                         {(24, 8), (24, 9), (25, 8), (25, 9)})
        for x, y, rows, cols in (
            (24, 24, (0, 1), (0, 1)), (216, 24, (0, 1), (24, 25)),
            (24, 216, (24, 25), (0, 1)), (216, 216, (24, 25), (24, 25)),
            (89, 81, (7, 8, 9), (8, 9, 10)),
        ):
            with self.subTest(x=x, y=y):
                self.assertEqual(set(gs.tank_cells(x, y)),
                                 {(r, c) for r in rows for c in cols})

    def test_clipped_and_outside_tanks(self):
        self.assertEqual(gs.tank_cells(16, 16), ((0, 0),))
        self.assertEqual(gs.tank_cells(224, 224), ((25, 25),))
        for x, y in ((8, 80), (232, 80), (80, 8), (80, 232), (0, 0)):
            with self.subTest(x=x, y=y):
                self.assertEqual(gs.tank_cells(x, y), ())


if __name__ == '__main__':
    unittest.main()
