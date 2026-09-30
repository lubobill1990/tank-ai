"""Offline geometry-only relocation tests; no native/ROM calls."""
from dataclasses import replace
import importlib
import unittest

import numpy as np

import game_state as gs
from learning_lab.teacher_v2 import TeacherControllerV2

try:
    relocation = importlib.import_module('within_stage.relocation_queries')
except ModuleNotFoundError as error:
    if error.name != 'within_stage.relocation_queries':
        raise
    relocation = None


def pair():
    first = np.zeros(2048, np.uint8)
    first[0x68], first[0x51], first[0x80], first[0x85] = 128, 3, 20, 1
    first[0x90], first[0x98], first[0xA0], first[0x101] = 82, 160, 0x80, 0x40
    first[0x92], first[0x9A], first[0xA2] = 160, 72, 0xA2
    board = first[0x400:].reshape(32, 32)
    board[26:28, 14:16] = [[200, 201], [202, 203]]
    second = first.copy()
    second[0x98] = 157
    return first, second


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(relocation, 'Offline relocation queries are not implemented')

    def test_same_translation_preserves_subcell_displacement_and_every_other_byte(self):
        old = pair()
        saved = [ram.copy() for ram in old]
        shifted = relocation.relocate_pair(*old, (16, -8))
        for original, result in zip(old, shifted):
            changed = np.flatnonzero(original != result)
            self.assertEqual(changed.tolist(), [0x90, 0x98])
            self.assertEqual(int(result[0x90]), int(original[0x90]) + 16)
            self.assertEqual(int(result[0x98]), int(original[0x98]) - 8)
            self.assertEqual(int(result[0x90]) % 8, int(original[0x90]) % 8)
            self.assertEqual(int(result[0x98]) % 8, int(original[0x98]) % 8)
            np.testing.assert_array_equal(result[0x400:], original[0x400:])
        self.assertEqual(int(shifted[1][0x98]) - int(shifted[0][0x98]), -3)
        for actual, expected in zip(old, saved):
            np.testing.assert_array_equal(actual, expected)

    def test_noop_translation_replays_fresh_teacher_label_and_observation_exactly(self):
        previous, current = pair()
        query = relocation.make_query(previous, current, (100, 104), (0, 0))
        state = relocation.current_state(current, 104)
        teacher = TeacherControllerV2(); teacher.reset()
        expected = teacher.act(state, relocation.primitive_masks(state))
        self.assertEqual(query['action'], expected)
        np.testing.assert_array_equal(query['observations'][:55], relocation.precision_frame(previous, 100))
        np.testing.assert_array_equal(query['observations'][55:], relocation.precision_frame(current, 104))
        self.assertTrue(query['action_masks'][query['action']])
        self.assertEqual(query['observations'].shape, (110, 26, 26))

    def test_pair_constraints_reject_player_bullets_inactive_player_and_changed_terrain(self):
        previous, current = pair()
        for address, value in ((0xCC, 0x40), (0xD4, 0x40), (0xA0, 0x73), (0x400 + 100, 15)):
            bad = current.copy(); bad[address] = value
            with self.subTest(address=address), self.assertRaises(relocation.RejectedGeometry):
                relocation.relocate_pair(previous, bad, (8, 0))

    def test_body_must_be_in_arena_passable_and_not_overlap_tanks_in_either_frame(self):
        previous, current = pair()
        with self.assertRaises(relocation.RejectedGeometry):
            relocation.relocate_pair(previous, current, (-80, 0))
        for ram in (previous, current):
            ram[0x93], ram[0x9B], ram[0xA3] = 98, 160, 0xA0
        with self.assertRaises(relocation.RejectedGeometry):
            relocation.relocate_pair(previous, current, (16, 0))
        previous, current = pair()
        for ram in (previous, current):
            ram[0x400 + 19 * 32 + 12] = 16
        with self.assertRaises(relocation.RejectedGeometry):
            relocation.relocate_pair(previous, current, (16, 0))
        with self.assertRaises(ValueError):
            relocation.relocate_pair(*pair(), (3, 0))

    def test_new_geometry_is_reencoded_instead_of_copying_old_grid_or_label(self):
        previous, current = pair()
        old = relocation.make_query(previous, current, (100, 104), (0, 0))
        new = relocation.make_query(previous, current, (100, 104), (16, 0))
        self.assertFalse(np.array_equal(old['observations'], new['observations']))
        state = relocation.current_state(new['ram_pair'][1], 104)
        teacher = TeacherControllerV2(); teacher.reset()
        self.assertEqual(new['action'], teacher.act(state, new['action_masks']))


if __name__ == '__main__':
    unittest.main()
