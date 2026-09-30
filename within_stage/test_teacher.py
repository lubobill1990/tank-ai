"""Synthetic geometry tests for the TRAIN-refined non-learned teacher."""
from dataclasses import replace
import unittest

import numpy as np

import game_state as gs
from heuristic import _Waypoint
from learning_lab.teacher_v2 import TeacherControllerV2
from within_stage import teacher as teachers


def scene(player=(88, 56), enemy=(184, 144)):
    ram = np.zeros(2048, np.uint8)
    ram[0x68], ram[0x51], ram[0x80], ram[0x85] = 128, 3, 20, 1
    ram[0x90], ram[0x98], ram[0xA0] = *player, 0x80
    ram[0x92], ram[0x9A], ram[0xA2] = *enemy, 0xA2
    arena = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
    arena[24:26, 12:14] = [[200, 201], [202, 203]]
    return gs.parse(ram)


class WithinStageTeacherTests(unittest.TestCase):
    def test_proactive_defence_triggers_before_fixed_near_base_threshold_if_far_from_intercept(self):
        state = scene()
        self.assertGreater(abs(state.enemies[0].x - 120) + abs(state.enemies[0].y - 216), 80)
        self.assertTrue(teachers.ProactiveDefenceTeacher.needs_early_defence(state))
        near_guard = replace(state, player=replace(state.player, x=168, y=176))
        self.assertFalse(teachers.ProactiveDefenceTeacher.needs_early_defence(near_guard))
        self.assertFalse(teachers.ProactiveDefenceTeacher.needs_early_defence(replace(state, enemies=())))

    def test_only_current_state_is_used_and_all_primitive_masks_are_respected(self):
        teacher = teachers.WithinStageTeacher()
        state = scene()
        before = state.terrain.copy()
        for action in range(10):
            masks = np.zeros(10, bool); masks[action] = True
            teacher.reset()
            self.assertEqual(teacher.act(state, masks), action)
        np.testing.assert_array_equal(state.terrain, before)
        with self.assertRaises(ValueError):
            teacher.act(state, np.zeros(10, bool))
        with self.assertRaises(ValueError):
            teacher.act(state, np.ones(10, np.uint8))

    def test_safe_unthreatened_case_matches_frozen_teacher_and_reset_is_repeatable(self):
        state = scene(player=(120, 184), enemy=(24, 24))
        original = TeacherControllerV2()
        teacher = teachers.WithinStageTeacher()
        mask = np.ones(10, bool)
        first = teacher.act(state, mask)
        self.assertEqual(first, original.act(state, mask))
        teacher.reset()
        self.assertEqual(teacher.act(state, mask), first)
        self.assertEqual(teacher.override_count, 0)

    def test_waypoint_snap_turn_avoids_oscillating_across_an_open_corner(self):
        state = scene(player=(88, 194), enemy=(200, 200))
        state = replace(state, player=replace(state.player, facing=gs.DOWN))
        target = (120, 192)
        bot = teachers.WaypointAlignedNavigator()
        bot._waypoints['protect'] = _Waypoint((*target, state.stage, state.lives), (88, 200), (88, 192))
        self.assertEqual(bot._navigate('protect', state, target), 1 + gs.RIGHT)
        self.assertEqual(bot.snap_turns, 1)
        bot.reset()
        self.assertEqual(bot.snap_turns, 0)

    def test_waypoint_snap_does_not_turn_far_from_corner(self):
        target = (120, 192)
        state = scene(player=(88, 198), enemy=(200, 200))
        state = replace(state, player=replace(state.player, facing=gs.UP))
        bot = teachers.WaypointAlignedNavigator()
        bot._waypoints['protect'] = _Waypoint((*target, state.stage, state.lives), (88, 200), (88, 192))
        self.assertEqual(bot._navigate('protect', state, target), 1 + gs.UP)
        self.assertEqual(bot.snap_turns, 0)

    def test_selected_teacher_retains_v2_behavior_on_multiple_current_states(self):
        for p in ((88, 194), (88, 56), (120, 184)):
            for e in ((184, 144), (144, 216), (24, 24)):
                state = scene(player=p, enemy=e)
                for action in range(10):
                    masks = np.ones(10, bool); masks[action] = False
                    original, selected = TeacherControllerV2(), teachers.WithinStageTeacher()
                    self.assertEqual(selected.act(state, masks), original.act(state, masks))
                    self.assertEqual(selected.override_count, 0)


if __name__ == '__main__':
    unittest.main()
