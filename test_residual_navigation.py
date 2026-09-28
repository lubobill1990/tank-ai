"""Residual-brick navigation against real ROM collision and turn alignment."""

import unittest

import numpy as np

from battle_city_env import BattleCityEnv
import game_state as gs


class ResidualNavigationTests(unittest.TestCase):
    def arena(self, mask=1, *, row=12, player=(88, 120), item=(144, 120), wall=False):
        env = BattleCityEnv(stage=1, max_frames=800)
        self.addCleanup(env.close)
        env.reset(seed=0)
        ram = env.nes.ram
        grid = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
        grid[:] = 0
        grid[24:26, 12:14] = np.array(((200, 201), (202, 203)), dtype=np.uint8)
        if wall:
            grid[:, 10] = 16
            grid[10:16, 10] = 15
        grid[row, 10] = mask
        ram[0xa1:0xa8] = 0
        ram[0xcc:0xd6] = 0
        ram[0x7f], ram[0x80] = 0, 1
        ram[0x90], ram[0x98], ram[0xa0] = *player, 0x83
        ram[0x101] = ram[0xa8] = 0
        ram[0x86], ram[0x87], ram[0x88], ram[0x62] = *item, 0, 0
        ram[0x68] = 0x80
        env._tracker.reset(ram)
        return env, grid

    def test_empty_cannon_lane_stays_stuck_under_right_and_right_fire(self):
        env, grid = self.arena()
        for _ in range(3):
            env.step(4)
        self.assertEqual((env.state.player.x, env.state.player.y), (88, 120))
        for _ in range(20):
            env.step(9)
        self.assertEqual((env.state.player.x, env.state.player.y), (88, 120))
        self.assertEqual(int(grid[12, 10]), 1)

    def test_standalone_heuristic_actively_collects_a_safe_item_beyond_64_pixels(self):
        env, _ = self.arena(0, player=(72, 72), item=(184, 72))
        self.assertEqual(env.heuristic_action(), 4)
        for _ in range(64):
            _, _, done, truncated, info = env.step(env.heuristic_action())
            self.assertFalse(done or truncated)
            if info["pickups"]:
                break
        self.assertEqual(info["pickups"], 1)
        self.assertTrue(env.state.shield)

    def test_hit_quarters_clear_under_the_same_right_fire_inputs(self):
        for mask in (8, 15):
            with self.subTest(mask=mask):
                env, grid = self.arena(mask)
                for _ in range(20):
                    env.step(9)
                self.assertEqual(int(grid[12, 10]), 0)
                self.assertGreater(env.state.player.x, 104)

    def test_real_quarter_boundary_hits_use_center_and_negative_side_only(self):
        horizontal = (
            (1, 116, True), (1, 117, False),
            (8, 115, False), (8, 116, True),
            (8, 120, True), (8, 121, False),
        )
        vertical = (
            (1, 100, True), (1, 101, False),
            (8, 99, False), (8, 100, True),
            (8, 104, True), (8, 105, False),
        )
        for facing in (gs.UP, gs.DOWN, gs.LEFT, gs.RIGHT):
            cases = horizontal if facing in (gs.LEFT, gs.RIGHT) else vertical
            for mask, lane, hit in cases:
                with self.subTest(facing=facing, mask=mask, lane=lane):
                    if facing in (gs.LEFT, gs.RIGHT):
                        player = (88 if facing == gs.RIGHT else 120, lane)
                    else:
                        player = (lane, 88 if facing == gs.DOWN else 144)
                    env, grid = self.arena(mask, player=player)
                    env.nes.ram[0xa0] = 0x80 | gs.ROM_FACING.index(facing)
                    env._tracker.reset(env.nes.ram)
                    for _ in range(20):
                        env.step(6 + facing)
                    self.assertEqual(int(grid[12, 10]), 0 if hit else mask)

    def test_isolated_residual_is_routed_around_without_demolition(self):
        env, grid = self.arena()
        for _ in range(64):
            _, _, done, truncated, info = env.step(14)
            self.assertFalse(done or truncated)
            player = env.state.player
            self.assertNotIn((12, 10), gs.tank_cells(player.x, player.y))
            if info["pickups"]:
                break
        self.assertEqual(info["pickups"], 1)
        self.assertEqual(info["shots"], 0)
        self.assertEqual(int(grid[12, 10]), 1)

    def test_macro_repositions_and_reaches_item_without_empty_shot_loop(self):
        env, grid = self.arena(wall=True)
        visited = []
        for _ in range(64):
            _, _, done, truncated, info = env.step(14)
            player = env.state.player
            self.assertIsNotNone(player)
            visited.append((player.x, player.y))
            if int(grid[12, 10]):
                self.assertNotIn((12, 10), gs.tank_cells(player.x, player.y))
            self.assertFalse(done or truncated)
            if info["pickups"]:
                break
        self.assertEqual(info["pickups"], 1, visited)
        self.assertTrue(any(y != 120 for _, y in visited), visited)
        self.assertGreater(info["shots"], 0)
        self.assertLess(info["shots"], 8, visited)
        self.assertFalse(env.state.base_destroyed)

    def test_fortress_residual_escape_preserves_base_ring(self):
        env, grid = self.arena(row=24, player=(88, 216), item=(88, 184))
        before = grid[24:26, 10:16].copy()
        for _ in range(32):
            _, _, done, truncated, info = env.step(14)
            self.assertFalse(done or truncated)
            if info["pickups"]:
                break
        self.assertEqual(info["pickups"], 1)
        np.testing.assert_array_equal(grid[24:26, 10:16], before)
        self.assertEqual(info["shots"], 0)

    def test_turn_snap_escapes_when_frame_skip_overshoots_firing_alignment(self):
        for y in (113, 114, 115):
            with self.subTest(y=y):
                env, grid = self.arena(player=(88, y), wall=True)
                visited = []
                for _ in range(64):
                    _, _, done, truncated, info = env.step(14)
                    player = env.state.player
                    visited.append((player.x, player.y))
                    if grid[12, 10]:
                        self.assertNotIn((12, 10), gs.tank_cells(player.x, player.y))
                    self.assertFalse(done or truncated)
                    if info["pickups"]:
                        break
                self.assertEqual(info["pickups"], 1, visited)
                self.assertLess(info["shots"], 8)


if __name__ == "__main__":
    unittest.main()
