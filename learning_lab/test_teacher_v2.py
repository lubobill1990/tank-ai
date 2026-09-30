"""Synthetic stand-off invariants; no environment interactions."""
from dataclasses import replace
import unittest

import numpy as np

import game_state as gs
from learning_lab.teacher import TeacherController as TeacherV1
from learning_lab.teacher_v2 import TeacherControllerV2


def state(**changes):
    terrain = np.zeros((26, 26), np.uint8)
    terrain[24:26, 12:14] = gs.T_BASE
    return replace(gs.GameState(
        4500, terrain, gs.Tank(0, 56, 152, gs.RIGHT),
        (gs.Tank(2, 80, 144, gs.LEFT),), (), 1, 1, 0, 18, 0, None,
        False, False, False, fire_ready=False,
        brick_masks=np.zeros((26, 26), np.uint8)), **changes)


class TeacherV2Tests(unittest.TestCase):
    def test_steps_out_of_close_offset_firing_lane_before_a_bullet_exists(self):
        current = state()
        masks = np.ones(10, bool)
        masks[5] = False
        action = TeacherControllerV2().act(current, masks)
        self.assertEqual(action, 2)
        self.assertEqual(current.bullets, ())

    def test_does_not_reenter_nearby_shoulder_lane(self):
        current = state(player=gs.Tank(0, 56, 155, gs.UP),
                        enemies=(gs.Tank(2, 100, 144, gs.LEFT),), fire_ready=False)
        action = TeacherControllerV2().act(current, np.ones(10, bool))
        self.assertNotEqual(action, 1)

    def test_retains_v1_when_shielded_frozen_or_enemy_faces_away(self):
        for current in (state(shield=True), state(freeze=True), state(fire_ready=True),
                        state(lives=3),
                        state(enemies=(gs.Tank(2, 80, 144, gs.RIGHT),)),
                        state(enemies=(gs.Tank(2, 200, 144, gs.LEFT),))):
            masks = np.ones(10, bool)
            masks[5] = current.fire_ready
            self.assertEqual(TeacherControllerV2().act(current, masks),
                             TeacherV1().act(current, masks))

    def test_does_not_abandon_ready_centered_counterfire(self):
        current = state(player=gs.Tank(0, 56, 144, gs.RIGHT), fire_ready=True)
        self.assertEqual(TeacherControllerV2().act(current, np.ones(10, bool)), 5)

    def test_never_violates_supplied_masks(self):
        current = state()
        for bits in range(1, 1024):
            masks = np.array([bool(bits & (1 << n)) for n in range(10)])
            self.assertTrue(masks[TeacherControllerV2().act(current, masks)])

    def test_no_new_baseward_fire(self):
        current = state(player=gs.Tank(0, 120, 184, gs.DOWN),
                        enemies=(gs.Tank(2, 120, 208, gs.UP),), fire_ready=True)
        action = TeacherControllerV2().act(current, np.ones(10, bool))
        self.assertFalse(TeacherV1._base_fire(current, action))

    def test_existing_bullet_response_takes_priority(self):
        current = state(bullets=(gs.Bullet(2, 1, 2, 76, 144, -2, 0),))
        masks = np.ones(10, bool)
        masks[5] = False
        controller = TeacherControllerV2()
        action = controller.act(current, masks)
        baseline = TeacherV1().act(current, masks)
        self.assertGreaterEqual(controller._danger(current, action), controller._danger(current, baseline))

    def test_no_standoff_motion_through_wall(self):
        current = state()
        current.terrain[18:20, 4:6] = gs.T_STEEL
        action = TeacherControllerV2().act(current, np.ones(10, bool))
        position, velocity = TeacherV1._motion(current, action)
        if action == 2:
            self.assertNotEqual(velocity, (0, 0))

    def test_standoff_is_rotationally_symmetric_not_a_stage_script(self):
        player = gs.Tank(0, 80, 120, gs.RIGHT)
        enemy = gs.Tank(2, 104, 112, gs.LEFT)
        turn = {gs.UP: gs.RIGHT, gs.RIGHT: gs.DOWN, gs.DOWN: gs.LEFT, gs.LEFT: gs.UP}
        masks = np.ones(10, bool)
        masks[5] = False
        for expected in (2, 3, 1, 4):
            current = state(player=player, enemies=(enemy,))
            self.assertEqual(TeacherControllerV2().act(current, masks), expected)
            player = replace(player, x=240 - player.y, y=player.x, facing=turn[player.facing])
            enemy = replace(enemy, x=240 - enemy.y, y=enemy.x, facing=turn[enemy.facing])

    def test_new_escape_does_not_enter_an_existing_bullet_path(self):
        current = state(bullets=(gs.Bullet(3, 1, 3, 80, 164, -2, 0),))
        masks = np.ones(10, bool)
        masks[5] = False
        action = TeacherControllerV2().act(current, masks)
        self.assertNotEqual(action, 2)
        self.assertEqual(TeacherV1._danger(current, action), float("inf"))


if __name__ == "__main__":
    unittest.main()
