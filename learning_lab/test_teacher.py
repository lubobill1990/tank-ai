"""Tests for the NON-LEARNED current-state development teacher."""
from dataclasses import replace
import unittest

import numpy as np

import game_state as gs
from learning_lab.teacher import TeacherController


def state(**changes):
    terrain = np.zeros((26, 26), dtype=np.uint8)
    terrain[24:26, 12:14] = gs.T_BASE
    initial = gs.GameState(0, terrain, gs.Tank(0, 104, 24, gs.RIGHT),
                           (gs.Tank(2, 120, 56, gs.DOWN),), (), 1, 3, 19, 0, 0,
                           None, False, False, False, fire_ready=False,
                           brick_masks=np.zeros((26, 26), dtype=np.uint8))
    return replace(initial, **changes)


class TeacherTests(unittest.TestCase):
    def test_never_walks_into_existing_bullet_lane(self):
        current = state(player=gs.Tank(0, 108, 24, gs.UP),
                        enemies=(gs.Tank(2, 216, 24, gs.LEFT),),
                        bullets=(gs.Bullet(2, 1, 2, 120, 28, 0, -2),))
        action = TeacherController().act(current, np.ones(10, dtype=bool))
        self.assertNotIn(action, (4, 9))

    def test_mask_legality_for_every_single_allowed_action(self):
        controller = TeacherController()
        for action in range(10):
            masks = np.zeros(10, dtype=bool)
            masks[action] = True
            self.assertEqual(controller.act(state(), masks), action)

    def test_no_baseward_fire_even_through_fortress_bricks(self):
        current = state(player=gs.Tank(0, 120, 184, gs.DOWN),
                        enemies=(gs.Tank(2, 120, 216, gs.UP),), fire_ready=True)
        current.terrain[23, 12:14] = gs.T_BRICK
        action = TeacherController().act(current, np.ones(10, dtype=bool))
        self.assertNotIn(action, (5, 7))

    def test_turn_snap_does_not_create_baseward_fire(self):
        current = state(player=gs.Tank(0, 109, 184, gs.RIGHT),
                        enemies=(gs.Tank(2, 112, 208, gs.UP),), fire_ready=True)
        self.assertTrue(TeacherController._base_fire(current, 7))

    def test_no_player_returns_legal_noop(self):
        self.assertEqual(TeacherController().act(state(player=None), np.ones(10, dtype=bool)), 0)

    def test_invalid_masks_rejected(self):
        controller = TeacherController()
        with self.assertRaises(ValueError):
            controller.act(state(), np.zeros(10, dtype=bool))
        with self.assertRaises(ValueError):
            controller.act(state(), np.ones(15, dtype=bool))

    def test_planner_does_not_drive_through_unbreakable_wall(self):
        current = state(player=gs.Tank(0, 88, 88, gs.RIGHT),
                        enemies=(gs.Tank(2, 152, 88, gs.LEFT),))
        current.terrain[:, 10:12] = gs.T_STEEL
        controller = TeacherController()
        self.assertEqual(controller.act(current, np.ones(10, dtype=bool)), 0)
        self.assertEqual(controller._motion(current, 4)[1], (0, 0))

    def test_mask_legality_for_all_nonempty_subsets(self):
        current = state(bullets=(gs.Bullet(2, 1, 2, 120, 28, 0, -2),))
        controller = TeacherController()
        for bits in range(1, 1024):
            masks = np.array([bool(bits & (1 << i)) for i in range(10)])
            action = controller.act(current, masks)
            self.assertTrue(masks[action])

    def test_reset_clears_navigation_history(self):
        controller = TeacherController()
        controller.act(state(), np.ones(10, dtype=bool))
        controller.reset()
        self.assertEqual(controller.bot._waypoints, {})


if __name__ == "__main__":
    unittest.main()
