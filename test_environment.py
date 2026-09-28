import unittest

import numpy as np

from battle_city_env import BattleCityEnv
import game_state as gs


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = BattleCityEnv(stage=1, max_frames=120)
        self.addCleanup(self.env.close)

    def test_only_native_grid_observations(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs.shape[1:], (26, 26))
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(self.env.action_space.n, 15)

    def test_residual_quarters_and_subcell_offsets_are_visible(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (47, 26, 26))
        ram = self.env.nes.ram
        for mask in range(16):
            ram[0x442] = mask
            self.env._tracker.reset(ram)
            obs = self.env._observation()
            np.testing.assert_array_equal(obs[39:43, 0, 0],
                                          [bool(mask & bit) for bit in (1, 2, 4, 8)])
            self.assertEqual(obs[0, 0, 0], mask != 0)
        ram[0x90], ram[0x98] = 91, 213
        self.env._tracker.reset(ram)
        obs = self.env._observation()
        np.testing.assert_allclose(obs[43, obs[5] != 0], 3 / 7)
        np.testing.assert_allclose(obs[44, obs[5] != 0], 5 / 7)
        self.assertFalse(np.any(obs[43:45, obs[5] == 0]))
        ram[0xa0] = 0x73
        self.env._tracker.reset(ram)
        self.assertFalse(np.any(self.env._observation()[43:45]))

    def test_danger_plane_is_appended_without_reordering_existing_channels(self):
        from battle_city_env import CHANNEL_NAMES
        ram, _ = self.combat_arena()
        ram[0xce], ram[0xba], ram[0xc4], ram[0xd8] = 0x42, 96, 80, 0
        self.env._tracker.reset(ram)
        obs = self.env._observation()
        self.assertEqual(CHANNEL_NAMES[39:45],
                         ('brick_tl', 'brick_tr', 'brick_bl', 'brick_br',
                          'player_offset_x', 'player_offset_y'))
        self.assertEqual(CHANNEL_NAMES[45], 'bullet_danger')
        self.assertGreater(obs[45, 13, 9], 0)  # hits our right track
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_red_carrier_is_distinguishable_from_an_ordinary_enemy(self):
        ram, _ = self.combat_arena()
        ram[0xa2], ram[0x92], ram[0x9a], ram[0xaa] = 0xa2, 88, 80, 0x80
        self.env._tracker.reset(ram)
        normal = self.env._observation()
        ram[0xaa] = 0x84
        self.env._tracker.reset(ram)
        red = self.env._observation()
        self.assertEqual(red.shape, (47, 26, 26))
        np.testing.assert_array_equal(normal[:46], red[:46])
        self.assertFalse(normal[46].any())
        self.assertTrue(red[46].any())

    def test_danger_plane_matches_real_track_hits_and_residual_cover(self):
        for wall, hit in ((0, True), (16, False), (1, True), (8, False)):
            with self.subTest(wall=wall):
                ram, terrain = self.combat_arena()
                terrain[10, 9] = wall
                ram[0xce], ram[0xba], ram[0xc4], ram[0xd8] = 0x42, 96, 80, 0
                self.env._tracker.reset(ram)
                self.assertEqual(self.env._observation()[45, 13, 9] > 0, hit)
                for _ in range(12):
                    _, _, done, truncated, info = self.env.step(0)
                    if info['deaths'] or done or truncated:
                        break
                self.assertEqual(info['deaths'], int(hit))

    def test_reset_reproduces_state_and_mask_queries_are_pure(self):
        first, _ = self.env.reset(seed=7)
        self.env.step(6)
        second, _ = self.env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        frame = self.env.state.frame
        masks = self.env.action_masks()
        np.testing.assert_array_equal(masks, self.env.action_masks())
        self.assertEqual(self.env.state.frame, frame)

    def test_timeout_and_reward_components(self):
        self.env.reset(seed=0)
        for _ in range(30):
            obs, reward, done, truncated, info = self.env.step(0)
            self.assertAlmostEqual(reward, sum(info['reward_components'].values()))
            self.assertTrue(self.env.observation_space.contains(obs))
            if done or truncated:
                break
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(info['outcome'], 'timeout')
        self.assertEqual(info['frames'], 120)

    def test_ghost_bullet_does_not_disable_fire(self):
        self.env.reset(seed=0)
        self.env.nes.ram[0xb8], self.env.nes.ram[0xcc] = 88, 0
        self.env._tracker.reset(self.env.nes.ram)
        self.assertTrue(self.env.action_masks()[5])

    def test_death_cannot_earn_evasion_reward(self):
        self.env.reset(seed=0)
        ram = self.env.nes.ram
        ram[0x89] = 0
        ram[0xce], ram[0xba], ram[0xc4], ram[0xd8] = 0x42, 88, 206, 0
        self.env._tracker.reset(ram)
        _, _, _, _, info = self.env.step(0)
        self.assertEqual(info['deaths'], 1)
        self.assertEqual(info['reward_components']['evasion'], 0)
        self.assertEqual(info['reward_components']['death'], -3)

    def test_all_six_real_pickups_are_not_scored_as_kills(self):
        for kind in range(6):
            with self.subTest(kind=kind):
                self.env.reset(seed=0)
                ram = self.env.nes.ram
                ram[0xa2:0xa8] = 0
                ram[0x86], ram[0x87], ram[0x88], ram[0x62] = 88, 216, kind, 0
                self.env._tracker.reset(ram)
                _, _, _, _, info = self.env.step(0)
                self.assertEqual(info['pickups'], 1)
                self.assertEqual(info['kills'], 0)
                self.assertEqual(self.env.state.score, 500)
                self.assertGreater(info['reward_components']['pickup'], 0)

    def test_real_two_star_shots_include_secondary_slot(self):
        self.env.reset(seed=0)
        ram = self.env.nes.ram
        ram[0x101] = ram[0xa8] = 0x40
        ram[0xcc:0xd6] = 0
        self.env._tracker.reset(ram)
        seen = set()
        for _ in range(5):
            _, _, _, _, info = self.env.step(5)
            seen.update(b.slot for b in self.env.state.bullets if b.owner == 0)
        self.assertEqual(seen, {0, 8})
        self.assertEqual(info['shots'], 2)

    def test_actual_shot_not_repeated_fire_requests_drives_penalties(self):
        self.env.reset(seed=0)
        self.env.step(5)
        _, _, _, _, info = self.env.step(5)
        self.assertEqual(info['reward_components']['base_fire'], 0)
        self.assertEqual(info['reward_components']['steel_fire'], 0)
        self.assertEqual(info['shots'], 1)

    def test_live_sprite_alignment_after_crossing_cell_boundaries(self):
        for direction in (16, 128):
            self.env.reset(seed=0)
            for _ in range(40):
                self.env.nes.step(0)
            for frame in range(17):
                if frame:
                    self.env.nes.step(direction)
                    self.env.nes.step(0)
                    self.env.nes.step(0)
                ram = self.env.nes.ram
                x, y = int(ram[0x90]), int(ram[0x98])
                x0, x1 = max(16, x - 12), min(224, x + 12)
                y0, y1 = max(16, y - 12), min(224, y + 12)
                pixels = np.array(self.env.nes.screen)[y0:y1, x0:x1, :3]
                rows, cols = np.nonzero(np.all(pixels == [172, 124, 0], axis=2))
                self.assertGreater(len(rows), 0)
                self.env._tracker.reset(ram)
                grid = self.env._observation()
                self.assertTrue(np.all(grid[5, (rows + y0 - 16) // 8,
                                            (cols + x0 - 16) // 8] == 1))

    def test_real_bullet_kill_increments_once_without_waiting_for_score(self):
        self.env.reset(seed=0)
        ram = self.env.nes.ram
        ram[0xa2:0xa8] = 0
        ram[0xa2], ram[0x92], ram[0x9a], ram[0xaa] = 0xa2, 88, 190, 0x80
        ram[0x89] = 3
        self.env._tracker.reset(ram)
        for _ in range(8):
            _, _, done, truncated, info = self.env.step(5)
            if info['kills'] or done or truncated:
                break
        self.assertEqual(info['kills'], 1)
        self.assertEqual(info['pickups'], 0)
        self.assertEqual(int(self.env.nes.ram[0x73]), 1)

    def test_flashing_heavy_tank_requires_four_real_impacts(self):
        self.env.reset(seed=0)
        ram = self.env.nes.ram
        ram[0xa2:0xa8] = 0
        ram[0xa2], ram[0x92], ram[0x9a], ram[0xaa] = 0xa2, 88, 100, 0xe4
        self.env._tracker.reset(ram)
        for remaining in (3, 2, 1, 0):
            ram = self.env.nes.ram
            ram[0xcc], ram[0xb8], ram[0xc2], ram[0xd6] = (
                0x40, ram[0x92], int(ram[0x9a]) + 10, 0)
            _, _, _, _, info = self.env.step(0)
            enemies = [t for t in self.env.state.enemies if t.slot == 2]
            if remaining:
                self.assertEqual(enemies[0].armor, remaining)
                self.assertEqual(info['kills'], 0)
            else:
                self.assertEqual(enemies, [])
                self.assertEqual(info['kills'], 1)

    def test_base_loss_precedes_timeout(self):
        self.env.reset(seed=0)
        ram = self.env.nes.ram
        ram[0xce], ram[0xba], ram[0xc4], ram[0xd8] = 0x42, 120, 206, 0
        self.env._tracker.reset(ram)
        _, _, done, truncated, info = self.env.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'base_lost')
        self.assertEqual(info['reward_components']['base_loss'], -20)

    def combat_arena(self, *, player=(88, 120), facing=gs.UP):
        self.env.reset(seed=0)
        ram = self.env.nes.ram
        terrain = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
        terrain[:] = 0
        terrain[24:26, 12:14] = np.array(((200, 201), (202, 203)), dtype=np.uint8)
        ram[0xa1:0xa8] = ram[0xcc:0xd6] = 0
        ram[0x7f], ram[0x80], ram[0x89] = 0, 1, 0
        ram[0x90], ram[0x98] = player
        ram[0xa0] = 0x80 | gs.ROM_FACING.index(facing)
        return ram, terrain

    def test_all_four_edges_mask_only_redundant_outward_actions(self):
        for x, y, outward in ((24, 80, gs.LEFT), (216, 80, gs.RIGHT),
                               (80, 24, gs.UP), (80, 216, gs.DOWN)):
            for current in (gs.UP, gs.DOWN, gs.LEFT, gs.RIGHT):
                with self.subTest(x=x, y=y, outward=outward, current=current):
                    ram, _ = self.combat_arena(player=(x, y), facing=current)
                    self.env._tracker.reset(ram)
                    masks = self.env.action_masks()
                    self.assertTrue(masks[0])
                    self.assertEqual(bool(masks[1 + outward]), current != outward)
                    self.assertEqual(bool(masks[6 + outward]), current != outward)
                    self.assertEqual(bool(masks[5]), current != outward)
                    for other in set((gs.UP, gs.DOWN, gs.LEFT, gs.RIGHT)) - {outward}:
                        self.assertTrue(masks[1 + other])
                        self.assertTrue(masks[6 + other])

    def test_boundary_turn_alignment_and_inward_movement_stay_available(self):
        ram, _ = self.combat_arena(player=(24, 123), facing=gs.UP)
        self.env._tracker.reset(ram)
        self.assertTrue(self.env.action_masks()[3])
        self.env.step(3)
        player = self.env.state.player
        self.assertEqual((player.x, player.y, player.facing), (24, 120, gs.LEFT))
        self.assertFalse(self.env.action_masks()[3])
        self.assertFalse(self.env.action_masks()[8])
        self.assertTrue(self.env.action_masks()[4])
        self.env.step(4)
        self.assertGreater(self.env.state.player.x, 24)

    def test_outward_shots_have_a_separate_penalty_and_boundary_waste_streak(self):
        ram, terrain = self.combat_arena(player=(24, 120), facing=gs.LEFT)
        self.env._tracker.reset(ram)
        _, _, _, _, first = self.env.step(8)  # Direct stepping deliberately bypasses policy masks.
        self.assertEqual(first['shots'], 1)
        self.assertEqual(first['reward_components']['boundary_fire'], -0.05)
        self.assertEqual(first['boundary_waste_frames'], 4)
        self.assertEqual(first['boundary_waste_streak_frames'], 4)
        terrain[4, 12] = 15  # Unrelated world change must not reset the player's counters.
        _, _, _, _, second = self.env.step(8)
        self.assertEqual(second['boundary_waste_streak_frames'], 8)
        self.assertEqual(second['max_boundary_waste_frames'], 8)
        self.assertEqual(second['stagnation_frames'], 8)
        _, _, _, _, third = self.env.step(4)
        self.assertEqual(third['boundary_waste_streak_frames'], 0)
        self.assertEqual(third['boundary_waste_frames'], 8)
        self.assertEqual(third['max_boundary_waste_frames'], 8)

    def test_open_corridor_and_inward_fire_are_not_penalized_as_boundary_waste(self):
        for x, facing in ((24, gs.RIGHT), (80, gs.LEFT)):
            with self.subTest(x=x, facing=facing):
                ram, _ = self.combat_arena(player=(x, 120), facing=facing)
                self.env._tracker.reset(ram)
                self.assertTrue(self.env.action_masks()[5])
                _, _, _, _, info = self.env.step(5)
                self.assertEqual(info['shots'], 1)
                self.assertEqual(info['reward_components']['boundary_fire'], 0)
                self.assertEqual(info['boundary_waste_frames'], 0)

    def test_stationary_guarding_is_not_counted_as_a_boundary_action(self):
        ram, terrain = self.combat_arena(player=(24, 120), facing=gs.LEFT)
        self.env._tracker.reset(ram)
        self.env.step(0)
        terrain[5, 12] = 15
        _, _, _, _, info = self.env.step(0)
        self.assertEqual(info['stagnation_frames'], 8)
        self.assertEqual(info['boundary_waste_frames'], 0)

    def test_counter_mask_matches_ammo_and_actual_interception_geometry(self):
        from heuristic import incoming_bullets
        for can_fire, offset, expect_counter in (
            (False, 0, False), (True, 0, True), (True, 4, True),
            (True, 6, False), (True, 9, False),
        ):
            with self.subTest(can_fire=can_fire, offset=offset):
                ram, _ = self.combat_arena()
                ram[0xce], ram[0xba], ram[0xc4], ram[0xd8] = 0x42, 88 + offset, 80, 0
                if not can_fire:
                    ram[0xcc], ram[0xb8], ram[0xc2], ram[0xd6] = 0x40, 88, 40, 0
                self.env._tracker.reset(ram)
                self.assertTrue(incoming_bullets(self.env.state))
                self.assertEqual(bool(self.env.action_masks()[13]), expect_counter)
                self.assertEqual(self.env._bot.advise('counter', self.env.state) != 0,
                                 expect_counter)
                self.assertTrue(self.env.action_masks()[12])

    def test_real_counter_can_cancel_a_centered_bullet_but_not_a_track_hit(self):
        for offset, expected_deaths in ((0, 0), (6, 1), (9, 1)):
            with self.subTest(offset=offset):
                ram, _ = self.combat_arena()
                ram[0xce], ram[0xba], ram[0xc4], ram[0xd8] = 0x42, 88 + offset, 80, 0
                self.env._tracker.reset(ram)
                for _ in range(10):
                    _, _, done, truncated, info = self.env.step(5)
                    if info['deaths'] or done or truncated:
                        break
                self.assertEqual(info['deaths'], expected_deaths)

    def test_baseward_brick_shot_is_allowed_until_the_real_lane_is_clear(self):
        from heuristic import _safe_fire, shot_obstacle
        for left_mask, right_mask, expect_allowed in ((15, 15, True), (1, 2, False)):
            with self.subTest(left_mask=left_mask, right_mask=right_mask):
                ram, terrain = self.combat_arena(player=(120, 168), facing=gs.DOWN)
                terrain[22, 12], terrain[22, 13] = left_mask, right_mask
                self.env._tracker.reset(ram)
                self.assertEqual(_safe_fire(self.env.state, (120, 168), gs.DOWN),
                                 expect_allowed)
                self.assertEqual(shot_obstacle(self.env.state, 120, 168, gs.DOWN),
                                 'brick' if expect_allowed else 'base')
                _, _, _, _, info = self.env.step(5)
                self.assertEqual(info['shots'], 1)
                self.assertEqual(info['reward_components']['base_fire'],
                                 0 if expect_allowed else -0.5)


class OutcomeTests(unittest.TestCase):
    def red_combat(self):
        env = BattleCityEnv(objective='combat', max_frames=2000)
        self.addCleanup(env.close)
        env.reset(seed=0)
        ram = env.nes.ram
        terrain = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
        terrain[:] = 0
        terrain[24:26, 12:14] = np.array(((200, 201), (202, 203)), dtype=np.uint8)
        ram[0xa1:0xa8] = ram[0xcc:0xd6] = 0
        ram[0x7f], ram[0x80] = 0, 2
        ram[0xa2], ram[0x92], ram[0x9a], ram[0xaa] = 0xa2, 88, 100, 0x84
        ram[0xcc], ram[0xb8], ram[0xc2], ram[0xd6] = 0x40, 88, 110, 0
        env._tracker.reset(ram)
        return env

    def test_red_kill_does_not_reset_before_the_pickup_opportunity(self):
        env = self.red_combat()
        _, _, done, truncated, info = env.step(0)
        self.assertEqual(info['kills'], 1)
        self.assertIsNotNone(env.state.item)
        self.assertFalse(done or truncated)
        self.assertEqual((info['item_drops'], info['item_opportunities']), (1, 1))
        item = env.state.item
        env.nes.ram[0x90], env.nes.ram[0x98] = item.x, item.y
        for _ in range(2):
            _, _, done, _, info = env.step(0)
            if info['pickups']:
                break
        self.assertEqual(info['pickups'], 1)
        self.assertTrue(done)
        self.assertEqual(info['outcome'], 'objective')

    def test_post_kill_pickup_window_is_bounded_without_more_rewards(self):
        env = self.red_combat()
        total = 0
        for _ in range(200):
            _, reward, done, truncated, info = env.step(0)
            total += reward
            if done or truncated:
                break
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'objective')
        self.assertGreaterEqual(info['frames'], 600)
        self.assertLessEqual(info['frames'], 610)
        self.assertEqual(info['pickups'], 0)
        self.assertEqual(info['item_opportunities'], 1)

    def test_an_earlier_pickup_does_not_skip_a_new_post_kill_opportunity(self):
        env = self.red_combat()
        env._pickups = 1
        _, _, done, truncated, info = env.step(0)
        self.assertEqual(info['kills'], 1)
        self.assertIsNotNone(env.state.item)
        self.assertFalse(done or truncated)

    def test_objective_success_is_not_a_stage_clear(self):
        env = BattleCityEnv(objective='combat')
        self.addCleanup(env.close)
        env.reset(seed=0)
        ram = env.nes.ram
        ram[0xa2:0xa8] = 0
        ram[0xa2], ram[0x92], ram[0x9a], ram[0xaa] = 0xa2, 88, 190, 0x80
        ram[0x86], ram[0x87], ram[0x88], ram[0x62] = 88, 216, 4, 0
        env._tracker.reset(ram)
        _, _, done, truncated, info = env.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'objective')
        self.assertEqual(info['objective'], 'combat')
        self.assertEqual(info['kills'], 1)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({'stage': 0}, {'max_frames': 0}, {'objective': 'invalid'},
                       {'reward_profile': 'typo'}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BattleCityEnv(**kwargs)


class RewardProfileTests(unittest.TestCase):
    def test_sparse_reward_changes_only_reward_not_observation_or_events(self):
        shaped = BattleCityEnv(max_frames=240, reward_profile='shaped')
        sparse = BattleCityEnv(max_frames=240, reward_profile='sparse')
        self.addCleanup(shaped.close)
        self.addCleanup(sparse.close)
        a, _ = shaped.reset(seed=0)
        b, _ = sparse.reset(seed=0)
        np.testing.assert_array_equal(a, b)
        core = {'time', 'kill', 'death', 'clear', 'base_loss'}
        changed = False
        for action in [8, 3, 6, 9, 7, 5] * 10:
            oa, ra, ta, xa, ia = shaped.step(action)
            ob, rb, tb, xb, ib = sparse.step(action)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual((ta, xa), (tb, xb))
            for key in ia.keys() - {'reward_components'}:
                self.assertEqual(ia[key], ib[key])
            self.assertAlmostEqual(rb, sum(value for key, value in ia['reward_components'].items()
                                          if key in core))
            self.assertTrue(all(value == 0 for key, value in ib['reward_components'].items()
                                if key not in core))
            self.assertAlmostEqual(rb, sum(ib['reward_components'].values()))
            changed |= ra != rb
            if ta or xa:
                break
        self.assertTrue(changed, 'fixture must exercise a shaped-only reward')


if __name__ == '__main__':
    unittest.main()
