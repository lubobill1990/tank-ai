"""Hand-built native-grid regression fixtures; no ROM or emulator required."""

from dataclasses import replace
import math
import unittest

import numpy as np

import game_state as gs
from heuristic import HeuristicBot, incoming_bullets, shot_obstacle


def snapshot(*, player=None, terrain=None, enemies=(), bullets=(), item=None, fire_ready=True):
    if terrain is None:
        terrain = np.zeros((26, 26), dtype=np.uint8)
    terrain = terrain.copy()
    terrain[24:26, 12:14] = gs.T_BASE
    terrain.flags.writeable = False
    return gs.GameState(
        frame=100, terrain=terrain,
        player=player if player is not None else gs.Tank(0, 72, 72, gs.UP),
        enemies=tuple(enemies), bullets=tuple(bullets), stage=1, lives=3,
        spawn_left=10, kills=0, score=0, item=item, base_destroyed=False,
        game_over=False, shield=False, fire_ready=fire_ready,
    )


def residual_snapshot(mask, *, row=12, col=10, **kwargs):
    terrain = kwargs.pop("terrain", np.zeros((26, 26), dtype=np.uint8))
    terrain = terrain.copy()
    terrain[row, col] = gs.T_BRICK
    masks = np.where(terrain == gs.T_BRICK, 15, 0).astype(np.uint8)
    masks[row, col] = mask
    masks.flags.writeable = False
    return replace(snapshot(terrain=terrain, **kwargs), brick_masks=masks)


class ResidualBrickTests(unittest.TestCase):
    def test_all_masks_and_directions_hit_only_occupied_quarters(self):
        # Brick occupies x=96..103, y=112..119. The negative-side probe
        # includes the other half at 100/116, but not at 101/117.
        cases = (
            (gs.RIGHT, (72, 114), 3), (gs.LEFT, (128, 114), 3),
            (gs.RIGHT, (72, 118), 12), (gs.LEFT, (128, 118), 12),
            (gs.DOWN, (98, 88), 5), (gs.UP, (98, 144), 5),
            (gs.DOWN, (102, 88), 10), (gs.UP, (102, 144), 10),
            (gs.RIGHT, (72, 116), 15), (gs.LEFT, (128, 116), 15),
            (gs.RIGHT, (72, 117), 12), (gs.LEFT, (128, 117), 12),
            (gs.DOWN, (100, 88), 15), (gs.UP, (100, 144), 15),
            (gs.DOWN, (101, 88), 10), (gs.UP, (101, 144), 10),
            (gs.RIGHT, (72, 120), 12), (gs.LEFT, (128, 120), 12),
            (gs.RIGHT, (72, 111), 0), (gs.LEFT, (128, 121), 0),
            (gs.DOWN, (104, 88), 10), (gs.UP, (104, 144), 10),
            (gs.DOWN, (95, 88), 0), (gs.UP, (105, 144), 0),
        )
        for mask in range(1, 16):
            state = residual_snapshot(mask)
            for facing, (x, y), hit_bits in cases:
                with self.subTest(mask=mask, facing=facing, position=(x, y)):
                    self.assertEqual(
                        shot_obstacle(state, x, y, facing),
                        "brick" if mask & hit_bits else "none",
                    )

    def test_cleared_quarter_reveals_obstacle_behind_it(self):
        for material, expected in (
            (gs.T_STEEL, "steel"), (gs.T_UNKNOWN, "unknown"),
            (gs.T_WATER, "none"), (gs.T_ICE, "none"),
        ):
            with self.subTest(material=material):
                terrain = np.zeros((26, 26), dtype=np.uint8)
                terrain[12, 14] = material
                state = residual_snapshot(1, terrain=terrain)
                self.assertEqual(shot_obstacle(state, 72, 118, gs.RIGHT), expected)
        state = residual_snapshot(1, enemies=(gs.Tank(2, 152, 118, gs.LEFT),))
        self.assertEqual(shot_obstacle(state, 72, 118, gs.RIGHT), "enemy")

    def test_missing_masks_only_in_legacy_fixtures_mean_full_brick(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[12, 10] = gs.T_BRICK
        state = snapshot(terrain=terrain)
        self.assertIsNone(state.brick_masks)
        self.assertEqual(shot_obstacle(state, 72, 118, gs.RIGHT), "brick")

    def test_threat_counter_and_dodge_share_residual_ray(self):
        bullet = gs.Bullet(2, 1, 2, 136, 118, -2, 0)
        for mask, blocked in ((1, False), (8, True)):
            with self.subTest(mask=mask):
                state = residual_snapshot(
                    mask, player=gs.Tank(0, 80, 118, gs.RIGHT), bullets=(bullet,),
                )
                bot = HeuristicBot()
                self.assertEqual(incoming_bullets(state), () if blocked else (bullet,))
                self.assertEqual(bot.advise("counter", state), 0 if blocked else 5)
                self.assertEqual(bot.advise("dodge", state) == 0, blocked)

    def test_residual_off_cannon_lane_is_not_a_tunnel_edge(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_STEEL
        terrain[10:16, 10] = gs.T_BRICK
        state = residual_snapshot(
            1, terrain=terrain, player=gs.Tank(0, 88, 120, gs.RIGHT),
            item=gs.Item(136, 120, 0),
        )
        action = HeuristicBot().advise("item", state)
        self.assertIn(action, (1, 2), "Reposition instead of firing through an empty quarter")

    def test_no_residual_mask_is_drivable_even_when_cannon_lane_is_empty(self):
        for mask in range(1, 16):
            with self.subTest(mask=mask):
                state = residual_snapshot(
                    mask, player=gs.Tank(0, 88, 120, gs.RIGHT),
                    item=gs.Item(112, 120, 0),
                )
                self.assertNotEqual(HeuristicBot().advise("item", state), 4)

    def test_retained_tunnel_waypoint_is_rechecked_after_a_partial_hit(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_STEEL
        terrain[10:16, 10] = gs.T_BRICK
        options = dict(
            terrain=terrain, player=gs.Tank(0, 88, 120, gs.RIGHT),
            item=gs.Item(136, 120, 0),
        )
        bot = HeuristicBot()
        self.assertEqual(bot.advise("item", residual_snapshot(15, **options)), 9)
        state = replace(residual_snapshot(1, **options), frame=104)
        self.assertIn(bot.advise("item", state), (1, 2))
        self.assertEqual(int(state.brick_masks[12, 10]), 1)
        self.assertFalse(state.brick_masks.flags.writeable)

    def test_missed_residual_near_base_does_not_trigger_empty_tunnelling_shots(self):
        state = residual_snapshot(
            1, row=24, player=gs.Tank(0, 88, 216, gs.RIGHT),
            item=gs.Item(104, 216, 0),
        )
        self.assertLess(HeuristicBot().advise("item", state), 5)

    def test_fortress_bricks_are_legal_targets_when_the_base_is_not_first(self):
        for mask in (1, 8):
            with self.subTest(mask=mask):
                state = residual_snapshot(
                    mask, row=22, player=gs.Tank(0, 72, 197, gs.RIGHT),
                    enemies=(gs.Tank(2, 176, 197, gs.LEFT),),
                )
                self.assertGreaterEqual(HeuristicBot().advise("hunt", state), 5)


class ThreatTests(unittest.TestCase):
    def test_danger_map_includes_track_width_and_arrival_urgency(self):
        from heuristic import bullet_danger
        state = snapshot(bullets=(gs.Bullet(2, 1, 2, 88, 80, 0, 2),))
        danger = bullet_danger(state)
        self.assertEqual(danger.shape, (26, 26))
        self.assertEqual(danger.dtype, np.float32)
        self.assertGreater(danger[13, 8], 0)  # center (80,120): 8px off lane
        self.assertGreater(danger[13, 10], 0)  # center (96,120): other track
        self.assertEqual(danger[13, 7], 0)
        self.assertEqual(danger[13, 11], 0)
        self.assertGreater(danger[10, 9], danger[13, 9])
        self.assertEqual(danger[18, 9], 0)  # beyond 24-frame lookahead
        self.assertTrue(np.all((danger >= 0) & (danger <= 1)))

    def test_danger_map_respects_walls_water_and_residual_holes(self):
        from heuristic import bullet_danger
        for material, blocked in ((gs.T_STEEL, True), (gs.T_WATER, False),
                                  (gs.T_BRICK, True), (gs.T_ICE, False)):
            with self.subTest(material=material):
                terrain = np.zeros((26, 26), dtype=np.uint8)
                terrain[10, 8:10] = material
                state = snapshot(terrain=terrain,
                                 bullets=(gs.Bullet(2, 1, 2, 88, 80, 0, 2),))
                danger = bullet_danger(state)
                self.assertEqual(danger[13, 9] == 0, blocked)
        for mask, blocked in ((1, False), (8, True)):
            with self.subTest(mask=mask):
                state = residual_snapshot(mask, row=10, col=8,
                    bullets=(gs.Bullet(2, 1, 2, 88, 80, 0, 2),))
                self.assertEqual(bullet_danger(state)[13, 9] == 0, blocked)

    def test_danger_map_merges_multiple_enemy_bullets_but_ignores_friendly(self):
        from heuristic import bullet_danger
        b1 = gs.Bullet(2, 1, 2, 88, 80, 0, 2)
        b2 = gs.Bullet(3, 2, 3, 120, 120, -4, 0)
        first = bullet_danger(snapshot(bullets=(b1,)))
        second = bullet_danger(snapshot(bullets=(b2,)))
        np.testing.assert_array_equal(bullet_danger(snapshot(bullets=(b1, b2))),
                                      np.maximum(first, second))
        self.assertFalse(np.any(bullet_danger(snapshot(
            bullets=(replace(b1, owner=0), replace(b2, owner=1))))))
        self.assertFalse(np.any(bullet_danger(snapshot())))

    def test_danger_map_uses_strict_rom_track_boundary_and_never_mutates_state(self):
        from heuristic import bullet_danger
        for offset, hits in ((9, True), (10, False), (-9, True), (-10, False)):
            with self.subTest(offset=offset):
                state = snapshot(player=gs.Tank(0, 88, 120, gs.UP),
                                 bullets=(gs.Bullet(2, 1, 2, 88 + offset, 80, 0, 2),))
                before = state.terrain.copy()
                first = bullet_danger(state)
                self.assertEqual(first[13, 9] > 0, hits)
                self.assertEqual(bool(incoming_bullets(state)), hits)
                np.testing.assert_array_equal(first, bullet_danger(state))
                np.testing.assert_array_equal(before, state.terrain)

    def test_danger_intensity_agrees_with_shared_point_threat_prediction(self):
        from heuristic import _impact_time, bullet_danger
        for vx, vy in ((2, 0), (-2, 0), (0, 4), (0, -4)):
            with self.subTest(vx=vx, vy=vy):
                bullet = gs.Bullet(2, 1, 2, 88, 96, vx, vy)
                state = residual_snapshot(1, row=10, col=10, bullets=(bullet,))
                danger = bullet_danger(state)
                for row, col in ((8, 9), (9, 9), (10, 7), (10, 10), (13, 9), (18, 18)):
                    impact = _impact_time(state, bullet, 16 + col * 8, 16 + row * 8)
                    expected = (25 - impact) / 25 if math.isfinite(impact) else 0
                    self.assertAlmostEqual(float(danger[row, col]), expected, places=6)

    def test_only_closing_enemy_bullets_are_threats(self):
        incoming = gs.Bullet(2, 4, 2, 72, 32, 0, 2)
        own = gs.Bullet(0, 5, 0, 72, 40, 0, 2)
        leaving = gs.Bullet(3, 1, 3, 72, 40, 0, -2)
        unknown = gs.Bullet(4, 1, 4, 72, 40, 0, 0)
        state = snapshot(bullets=(own, leaving, incoming, unknown))
        self.assertEqual(incoming_bullets(state), (incoming,))

    def test_friendly_p2_bullet_is_not_a_threat_or_combat_target(self):
        state = snapshot(bullets=(gs.Bullet(1, 1, 1, 72, 32, 0, 2),))
        self.assertEqual(incoming_bullets(state), ())
        bot = HeuristicBot()
        self.assertEqual(bot.advise("dodge", state), 0)
        self.assertEqual(bot.advise("counter", state), 0)

    def test_hitbox_includes_side_lane_not_just_center(self):
        side = gs.Bullet(2, 1, 2, 81, 32, 0, 2)
        outside = gs.Bullet(3, 1, 3, 83, 32, 0, 2)
        self.assertEqual(incoming_bullets(snapshot(bullets=(side, outside))), (side,))

    def test_water_passes_bullets_but_solid_obstacles_stop_them(self):
        bullet = gs.Bullet(2, 1, 2, 72, 32, 0, 2)
        for material, expected in (
            (gs.T_WATER, (bullet,)), (gs.T_ICE, (bullet,)),
            (gs.T_BRICK, ()), (gs.T_STEEL, ()), (gs.T_UNKNOWN, ()),
        ):
            with self.subTest(material=material):
                terrain = np.zeros((26, 26), dtype=np.uint8)
                terrain[4, 6:8] = material
                self.assertEqual(
                    incoming_bullets(snapshot(terrain=terrain, bullets=(bullet,))),
                    expected,
                )

    def test_threats_are_sorted_by_time_not_slot_or_distance(self):
        slow = gs.Bullet(2, 1, 2, 72, 48, 0, 1)
        fast = gs.Bullet(3, 1, 3, 72, 32, 0, 4)
        self.assertEqual(incoming_bullets(snapshot(bullets=(slow, fast))), (fast, slow))

    def test_no_player_or_finished_game_has_no_threats(self):
        state = snapshot(bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),))
        self.assertEqual(incoming_bullets(replace(state, player=None)), ())
        self.assertEqual(incoming_bullets(replace(state, game_over=True)), ())


class ShotObstacleTests(unittest.TestCase):
    def test_immediate_outward_edge_shots_are_distinct_from_open_infield_rays(self):
        for x, y, face in ((24, 80, gs.LEFT), (216, 80, gs.RIGHT),
                           (80, 24, gs.UP), (80, 216, gs.DOWN)):
            with self.subTest(x=x, y=y, face=face):
                state = snapshot(player=gs.Tank(0, x, y, face))
                self.assertEqual(shot_obstacle(state, x, y, face), "boundary")
        self.assertEqual(shot_obstacle(snapshot(), 80, 80, gs.LEFT), "none")
        self.assertEqual(shot_obstacle(snapshot(), 24, 80, gs.RIGHT), "none")

    def test_water_ice_and_decoded_forest_do_not_stop_shots(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[6:8, 10] = gs.T_WATER
        terrain[6:8, 11] = gs.T_ICE
        terrain[6:8, 12] = gs.T_EMPTY
        state = snapshot(terrain=terrain)
        self.assertEqual(shot_obstacle(state, 72, 72, gs.RIGHT), "none")

    def test_solid_materials_are_classified(self):
        for material, expected in (
            (gs.T_BRICK, "brick"), (gs.T_STEEL, "steel"),
            (gs.T_UNKNOWN, "unknown"),
        ):
            with self.subTest(material=material):
                terrain = np.zeros((26, 26), dtype=np.uint8)
                terrain[6:8, 10] = material
                self.assertEqual(
                    shot_obstacle(snapshot(terrain=terrain), 72, 72, gs.RIGHT),
                    expected,
                )

    def test_closer_enemy_then_brick_occlude_base(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[20, 12:14] = gs.T_BRICK
        state = snapshot(
            terrain=terrain, enemies=(gs.Tank(2, 120, 152, gs.UP),),
        )
        self.assertEqual(shot_obstacle(state, 120, 120, gs.DOWN), "enemy")
        self.assertEqual(shot_obstacle(replace(state, enemies=()), 120, 120, gs.DOWN), "brick")
        self.assertEqual(shot_obstacle(snapshot(), 120, 120, gs.DOWN), "base")

    def test_closer_enemy_then_brick_occlude_steel(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[6:8, 10] = gs.T_BRICK
        terrain[6:8, 15] = gs.T_STEEL
        state = snapshot(
            terrain=terrain, enemies=(gs.Tank(2, 88, 72, gs.LEFT),),
        )
        self.assertEqual(shot_obstacle(state, 64, 72, gs.RIGHT), "enemy")
        self.assertEqual(shot_obstacle(replace(state, enemies=()), 64, 72, gs.RIGHT), "brick")
        terrain[6:8, 10] = gs.T_EMPTY
        self.assertEqual(
            shot_obstacle(snapshot(terrain=terrain), 64, 72, gs.RIGHT), "steel",
        )

    def test_wall_occludes_enemy_behind_it(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[6:8, 10] = gs.T_STEEL
        state = snapshot(
            terrain=terrain, enemies=(gs.Tank(2, 144, 72, gs.LEFT),),
        )
        self.assertEqual(shot_obstacle(state, 72, 72, gs.RIGHT), "steel")

    def test_event_coordinates_are_used_even_without_current_player(self):
        state = replace(
            snapshot(enemies=(gs.Tank(2, 160, 136, gs.LEFT),)), player=None,
        )
        self.assertEqual(shot_obstacle(state, 120, 136, gs.RIGHT), "enemy")

    def test_all_four_heading_codes_and_enemy_hitbox(self):
        cases = (
            (gs.UP, (120, 72)), (gs.DOWN, (120, 168)),
            (gs.LEFT, (72, 120)), (gs.RIGHT, (168, 120)),
        )
        for facing, (x, y) in cases:
            with self.subTest(facing=facing):
                state = snapshot(enemies=(gs.Tank(2, x, y, gs.UP),))
                self.assertEqual(shot_obstacle(state, 120, 120, facing), "enemy")
        state = snapshot(enemies=(gs.Tank(2, 81, 32, gs.DOWN),))
        self.assertEqual(shot_obstacle(state, 72, 72, gs.UP), "enemy")
        self.assertEqual(shot_obstacle(state, 68, 72, gs.UP), "none")

    def test_ray_misses_base_inside_conservative_controller_guard_margin(self):
        self.assertEqual(shot_obstacle(snapshot(), 103, 160, gs.DOWN), "none")
        self.assertEqual(shot_obstacle(snapshot(), 120, 160, gs.DOWN), "base")

    def test_shot_starts_at_muzzle_and_ignores_obstacles_behind_it(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[6:8, 5] = gs.T_BRICK
        state = snapshot(
            terrain=terrain, enemies=(gs.Tank(2, 40, 72, gs.RIGHT),),
        )
        self.assertEqual(shot_obstacle(state, 72, 72, gs.RIGHT), "none")

    def test_high_power_still_reports_steel_and_does_not_mutate_state(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[6:8, 10] = gs.T_STEEL
        state = snapshot(terrain=terrain)
        before = state.terrain.copy()
        for power in (0, 1, 2, 3):
            self.assertEqual(shot_obstacle(state, 72, 72, gs.RIGHT, power), "steel")
        np.testing.assert_array_equal(state.terrain, before)
        self.assertFalse(state.terrain.flags.writeable)

    def test_invalid_heading_is_an_explicit_error(self):
        for facing in (-1, 4):
            with self.subTest(facing=facing), self.assertRaises(ValueError):
                shot_obstacle(snapshot(), 72, 72, facing)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.bot = HeuristicBot()

    def test_item_route_does_not_fit_through_eight_pixel_gap(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_STEEL
        terrain[8, 10] = gs.T_EMPTY
        state = snapshot(
            player=gs.Tank(0, 88, 84, gs.RIGHT), terrain=terrain,
            item=gs.Item(120, 84, 0),
        )
        self.assertEqual(self.bot.advise("item", state), 0)

    def test_sixteen_pixel_opening_is_reachable(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_STEEL
        terrain[8:10, 10] = gs.T_EMPTY
        state = snapshot(
            player=gs.Tank(0, 88, 88, gs.RIGHT), terrain=terrain,
            item=gs.Item(120, 88, 0),
        )
        self.assertEqual(self.bot.advise("item", state), 4)

    def test_water_detour_reaches_item_without_entering_water(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[0:10, 10:12] = gs.T_WATER
        state = snapshot(
            player=gs.Tank(0, 80, 64, gs.RIGHT), terrain=terrain,
            item=gs.Item(136, 64, 0),
        )
        visited = []
        for frame in range(300):
            p = state.player
            if abs(p.x - 136) < 2 and abs(p.y - 64) < 2:
                break
            action = self.bot.advise("item", state)
            self.assertIn(action, (1, 2, 3, 4))
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
            x, y = p.x + dx, p.y + dy
            self.assertTrue(
                x + 8 <= 96 or x - 8 >= 112 or y - 8 >= 96,
                f"16px tank entered water at {(x, y)}",
            )
            visited.append((x, y))
            state = replace(state, frame=frame + 101, player=replace(p, x=x, y=y))
        self.assertLess(abs(state.player.x - 136) + abs(state.player.y - 64), 3)
        self.assertTrue(any(y >= 104 for _, y in visited))

    def test_partial_cell_positions_keep_advancing_across_boundary(self):
        for x in (41, 43, 47, 49, 53, 57, 61):
            state = snapshot(
                player=gs.Tank(0, x, 67, gs.RIGHT),
                item=gs.Item(112, 67, 0),
            )
            self.assertEqual(self.bot.advise("item", state), 4, f"x={x}")

    def test_partial_footprint_does_not_clip_third_cell(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_STEEL
        terrain[8:10, 10] = gs.T_EMPTY
        state = snapshot(
            player=gs.Tank(0, 88, 89, gs.RIGHT), terrain=terrain,
            item=gs.Item(120, 89, 0),
        )
        self.assertEqual(self.bot.advise("item", state), 1)

    def test_brick_wall_can_be_tunnelled_purposefully(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_BRICK
        state = snapshot(
            player=gs.Tank(0, 88, 88, gs.RIGHT), terrain=terrain,
            item=gs.Item(120, 88, 0),
        )
        self.assertEqual(self.bot.advise("item", state), 9)

    def test_protect_returns_above_fortress_without_firing_into_it(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[22:24, 10:16] = gs.T_BRICK
        terrain[24:26, 10:12] = gs.T_BRICK
        terrain[24:26, 14:16] = gs.T_BRICK
        state = snapshot(
            player=gs.Tank(0, 120, 160, gs.DOWN), terrain=terrain,
        )
        self.assertEqual(self.bot.advise("protect", state), 2)

    def test_baseward_demolition_edge_is_permitted_when_brick_intercepts(self):
        from heuristic import _segment_cost
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[20, 12:14] = gs.T_BRICK
        state = snapshot(
            player=gs.Tank(0, 120, 168, gs.DOWN), terrain=terrain,
            item=gs.Item(120, 192, 0),
        )
        self.assertTrue(math.isfinite(
            _segment_cost(state, (120, 168), (120, 176), tunnel=True)))


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.bot = HeuristicBot()

    def test_aim_uses_pixel_overlap_across_cell_boundaries(self):
        state = snapshot(
            player=gs.Tank(0, 71, 88, gs.UP),
            enemies=(gs.Tank(1, 79, 40, gs.DOWN),),
        )
        self.assertIn(self.bot.advise("hunt", state), (5, 6))

    def test_aim_fires_across_water_but_not_steel(self):
        for material, fires in ((gs.T_WATER, True), (gs.T_STEEL, False)):
            with self.subTest(material=material):
                terrain = np.zeros((26, 26), dtype=np.uint8)
                terrain[4, 6:8] = material
                state = snapshot(
                    terrain=terrain, enemies=(gs.Tank(1, 72, 32, gs.DOWN),),
                )
                self.assertEqual(self.bot.advise("hunt", state) >= 5, fires)

    def test_shots_missing_the_base_are_not_blocked_by_a_conservative_margin(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[20, 12:14] = gs.T_BRICK
        cases = (
            (gs.Tank(0, 103, 160, gs.DOWN), gs.Tank(1, 103, 192, gs.UP)),
            (gs.Tank(0, 80, 200, gs.RIGHT), gs.Tank(1, 152, 200, gs.LEFT)),
            (gs.Tank(0, 160, 204, gs.LEFT), gs.Tank(1, 80, 204, gs.RIGHT)),
        )
        for player, enemy in cases:
            with self.subTest(player=player):
                state = snapshot(player=player, terrain=terrain, enemies=(enemy,))
                self.assertGreaterEqual(self.bot.advise("hunt", state), 5)
                self.assertGreaterEqual(self.bot.advise("protect", state), 5)

    def test_shooting_away_from_base_is_allowed(self):
        state = snapshot(
            player=gs.Tank(0, 152, 216, gs.RIGHT),
            enemies=(gs.Tank(1, 200, 216, gs.LEFT),),
        )
        self.assertIn(self.bot.advise("hunt", state), (5, 9))

    def test_fortress_brick_can_be_destroyed_when_ray_misses_eagle(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[22:24, 10:16] = gs.T_BRICK
        state = snapshot(
            player=gs.Tank(0, 64, 192, gs.RIGHT), terrain=terrain,
            enemies=(gs.Tank(1, 168, 192, gs.LEFT),),
        )
        self.assertGreaterEqual(self.bot.advise("hunt", state), 5)

    def test_counter_is_unavailable_without_a_free_firing_slot(self):
        bullet = gs.Bullet(2, 1, 2, 72, 32, 0, 2)
        state = snapshot(bullets=(bullet,), fire_ready=False)
        self.assertEqual(self.bot.advise("counter", state), 0)
        self.assertIn(self.bot.act(state), (3, 4))

    def test_trapped_unarmed_bot_does_not_fall_back_to_attacking(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[6:8, 5] = terrain[6:8, 8] = gs.T_WATER
        terrain[5, 6:8] = terrain[8, 6:8] = gs.T_WATER
        state = snapshot(terrain=terrain, fire_ready=False,
                         bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),),
                         enemies=(gs.Tank(2, 72, 24, gs.DOWN),))
        self.assertEqual(self.bot.act(state), 0)

    def test_counter_requires_bullet_to_bullet_alignment_not_tank_overlap(self):
        aligned = gs.Bullet(2, 1, 2, 72, 32, 0, 2)
        off_axis = replace(aligned, x=81)
        self.assertIn(self.bot.advise("counter", snapshot(bullets=(aligned,))), (5, 6))
        self.assertEqual(self.bot.advise("counter", snapshot(bullets=(off_axis,))), 0)

    def test_counter_does_not_fire_through_steel(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[4, 6:8] = gs.T_STEEL
        state = snapshot(terrain=terrain, bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),))
        self.assertEqual(self.bot.advise("counter", state), 0)

    def test_counter_checks_outgoing_lane_as_well_as_incoming_lane(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[4, 6] = gs.T_STEEL
        bullet = gs.Bullet(2, 1, 2, 76, 32, 0, 2)
        state = snapshot(
            player=gs.Tank(0, 72, 88, gs.UP), terrain=terrain, bullets=(bullet,),
        )
        self.assertEqual(incoming_bullets(state), (bullet,))
        self.assertEqual(self.bot.advise("counter", state), 0)

    def test_dodge_uses_clear_side_not_water_or_brick(self):
        for material in (gs.T_WATER, gs.T_BRICK):
            with self.subTest(material=material):
                terrain = np.zeros((26, 26), dtype=np.uint8)
                terrain[6:8, 5] = material
                state = snapshot(
                    terrain=terrain, bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),),
                )
                self.assertEqual(self.bot.advise("dodge", state), 4)

    def test_dodge_does_not_enter_second_bullet_lane(self):
        state = snapshot(bullets=(
            gs.Bullet(2, 1, 2, 72, 32, 0, 2),
            gs.Bullet(3, 1, 3, 88, 16, 0, 3),
        ))
        self.assertEqual(self.bot.advise("dodge", state), 3)

    def test_dodge_does_not_avoid_friendly_p2_escape_lane(self):
        state = snapshot(bullets=(
            gs.Bullet(2, 1, 2, 72, 32, 0, 2),
            gs.Bullet(1, 1, 1, 88, 16, 0, 3),
        ))
        self.assertEqual(self.bot.advise("dodge", state), 4)

    def test_counter_does_not_fire_toward_base(self):
        state = snapshot(
            player=gs.Tank(0, 120, 168, gs.DOWN),
            bullets=(gs.Bullet(2, 1, 2, 120, 200, 0, -2),),
        )
        self.assertEqual(self.bot.advise("counter", state), 0)

    def test_act_responds_to_threat_before_chasing_item(self):
        state = snapshot(
            bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),),
            item=gs.Item(72, 32, 0),
        )
        self.assertIn(self.bot.act(state), (3, 4))


class ItemPolicyTests(unittest.TestCase):
    def test_safe_useful_item_beyond_sixty_four_pixels_beats_hunting(self):
        state = snapshot(
            enemies=(gs.Tank(2, 72, 32, gs.DOWN),),
            item=gs.Item(168, 72, 3),
        )
        self.assertEqual(HeuristicBot().act(state), 4)

    def test_item_value_depends_on_power_shield_and_lives(self):
        cases = (
            (3, 0, False, 3, 4), (3, 3, False, 3, 5),
            (0, 0, False, 3, 4), (0, 0, True, 3, 5),
            (5, 0, False, 1, 4), (5, 0, False, 5, 5),
        )
        for kind, power, shield, lives, action in cases:
            with self.subTest(kind=kind, power=power, shield=shield, lives=lives):
                state = replace(snapshot(
                    player=gs.Tank(0, 72, 72, gs.UP, power=power),
                    enemies=(gs.Tank(2, 72, 32, gs.DOWN),),
                    item=gs.Item(184, 72, kind),
                ), shield=shield, lives=lives)
                self.assertEqual(HeuristicBot().act(state), action)

    def test_existing_freeze_reduces_the_value_of_another_freeze(self):
        state = snapshot(
            enemies=(gs.Tank(2, 72, 32, gs.DOWN),), item=gs.Item(184, 72, 1),
        )
        self.assertEqual(HeuristicBot().act(state), 4)
        self.assertEqual(HeuristicBot().act(replace(state, freeze=True)), 5)

    def test_redundant_upgrade_is_still_collected_if_almost_underfoot(self):
        state = snapshot(
            player=gs.Tank(0, 72, 72, gs.UP, power=3),
            enemies=(gs.Tank(2, 72, 32, gs.DOWN),), item=gs.Item(88, 72, 3),
        )
        self.assertEqual(HeuristicBot().act(state), 4)

    def test_short_manhattan_distance_does_not_hide_an_expensive_detour(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:20, 10] = gs.T_WATER
        state = snapshot(
            player=gs.Tank(0, 88, 72, gs.UP), terrain=terrain,
            enemies=(gs.Tank(2, 88, 32, gs.DOWN),), item=gs.Item(120, 72, 0),
        )
        self.assertEqual(HeuristicBot().advise("item", state), 2)
        self.assertEqual(HeuristicBot().act(state), 5)

    def test_shared_search_reports_detour_cost_and_inclusive_budget(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:20, 10] = gs.T_WATER
        state = snapshot(terrain=terrain)
        bot = HeuristicBot()
        self.assertEqual(
            bot._find_route(state, (88, 72), (120, 72)), ((88, 80), 256),
        )
        self.assertEqual(
            bot._find_route(state, (88, 72), (120, 72), max_cost=256),
            ((88, 80), 256),
        )
        self.assertEqual(
            bot._find_route(state, (88, 72), (120, 72), max_cost=255),
            (None, math.inf),
        )

    def test_affordable_detour_is_still_worth_pursuing(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:8, 10] = gs.T_WATER
        state = snapshot(
            player=gs.Tank(0, 88, 72, gs.UP), terrain=terrain,
            enemies=(gs.Tank(2, 88, 32, gs.DOWN),), item=gs.Item(120, 72, 0),
        )
        self.assertEqual(HeuristicBot().act(state), 2)

    def test_unreachable_item_falls_back_to_hunting_on_every_snapshot(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 10] = gs.T_STEEL
        state = snapshot(
            terrain=terrain, enemies=(gs.Tank(2, 72, 32, gs.DOWN),),
            item=gs.Item(120, 72, 5),
        )
        bot = HeuristicBot()
        for frame in range(100, 120, 4):
            self.assertEqual(bot.act(replace(state, frame=frame)), 5)

    def test_imminent_bullet_and_base_emergency_override_useful_items(self):
        threatened = snapshot(
            bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),),
            item=gs.Item(72, 168, 3),
        )
        self.assertIn(HeuristicBot().act(threatened), (3, 4))
        base_threat = snapshot(
            player=gs.Tank(0, 72, 192, gs.RIGHT),
            enemies=(gs.Tank(2, 120, 192, gs.DOWN),), item=gs.Item(72, 88, 5),
        )
        self.assertEqual(HeuristicBot().act(base_threat), 5)


class CarrierPriorityTests(unittest.TestCase):
    def state(self, *, carrying=True, red_x=160, terrain=None):
        return snapshot(
            player=gs.Tank(0, 120, 80, gs.UP), terrain=terrain,
            enemies=(gs.Tank(2, 120, 40, gs.DOWN),
                     gs.Tank(3, red_x, 80, gs.LEFT, drops_item=carrying)),
        )

    def test_equally_safe_reachable_carrier_gets_priority(self):
        self.assertEqual(HeuristicBot().advise("hunt", self.state()), 9)

    def test_losing_carried_item_removes_priority_without_reset(self):
        bot = HeuristicBot()
        self.assertEqual(bot.advise("hunt", self.state()), 9)
        self.assertEqual(bot.advise("hunt", self.state(carrying=False)), 5)

    def test_remote_carrier_does_not_override_a_nearby_ordinary_target(self):
        self.assertEqual(HeuristicBot().advise("hunt", self.state(red_x=216)), 5)

    def test_inaccessible_carrier_does_not_displace_a_clear_ordinary_shot(self):
        terrain = np.zeros((26, 26), dtype=np.uint8)
        terrain[:, 16] = gs.T_STEEL
        self.assertEqual(HeuristicBot().advise("hunt", self.state(terrain=terrain)), 5)

    def test_ordinary_lethal_base_threat_beats_remote_carrier(self):
        state = snapshot(
            player=gs.Tank(0, 72, 192, gs.RIGHT),
            enemies=(gs.Tank(2, 120, 192, gs.DOWN),
                     gs.Tank(3, 72, 72, gs.DOWN, drops_item=True)),
            item=gs.Item(72, 88, 5),
        )
        self.assertEqual(HeuristicBot().act(state), 5)


class ContractTests(unittest.TestCase):
    def test_all_advisors_and_act_return_primitive_indices(self):
        state = snapshot(
            enemies=(gs.Tank(1, 120, 40, gs.DOWN),),
            bullets=(gs.Bullet(2, 1, 2, 72, 32, 0, 2),),
            item=gs.Item(160, 72, 0),
        )
        bot = HeuristicBot()
        actions = [bot.act(state)]
        actions.extend(bot.advise(name, state) for name in (
            "hunt", "protect", "dodge", "counter", "item",
        ))
        for action in actions:
            self.assertIs(type(action), int)
            self.assertIn(action, range(10))

    def test_absent_targets_and_dead_player_are_noops(self):
        bot = HeuristicBot()
        state = snapshot()
        for name in ("hunt", "dodge", "counter", "item"):
            self.assertEqual(bot.advise(name, state), 0)
        for state in (replace(state, player=None), replace(state, game_over=True)):
            for name in ("hunt", "protect", "dodge", "counter", "item"):
                self.assertEqual(bot.advise(name, state), 0)
            self.assertEqual(bot.act(state), 0)

    def test_unknown_advisor_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            HeuristicBot().advise("typo", snapshot())

    def test_repeated_calls_do_not_change_snapshot_or_threat_estimates(self):
        state = snapshot(
            bullets=(gs.Bullet(2, 8, 2, 72, 32, 0, 2),),
            enemies=(gs.Tank(1, 120, 40, gs.DOWN),),
            item=gs.Item(136, 72, 0),
        )
        before = state.terrain.copy()
        bot = HeuristicBot()
        expected = {name: bot.advise(name, state) for name in (
            "hunt", "protect", "dodge", "counter", "item",
        )}
        for _ in range(4):
            self.assertEqual(incoming_bullets(state), state.bullets)
            for name, action in expected.items():
                self.assertEqual(bot.advise(name, state), action)
        np.testing.assert_array_equal(state.terrain, before)
        self.assertFalse(state.terrain.flags.writeable)
        self.assertEqual(state.bullets[0], gs.Bullet(2, 8, 2, 72, 32, 0, 2))
        bot.reset()
        fresh = HeuristicBot()
        for name in expected:
            self.assertEqual(bot.advise(name, state), fresh.advise(name, state))


if __name__ == "__main__":
    unittest.main()
