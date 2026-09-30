import unittest
import numpy as np


class DecodingTests(unittest.TestCase):
    def test_marginal_decoding_can_choose_rare_fire_without_changing_movement(self):
        from learning_lab.decoding import choose_marginal_action
        p = np.array([.3, .51, 0, 0, 0, 0, .19, 0, 0, 0])
        mask = np.ones(10, dtype=bool)
        self.assertEqual(choose_marginal_action(p, mask, .1), 6)
        self.assertEqual(choose_marginal_action(p, mask, .5), 1)

    def test_mask_is_respected_even_when_chosen_movement_has_only_one_legal_form(self):
        from learning_lab.decoding import choose_marginal_action
        p = np.array([.2, 0, 0, 0, 0, 0, .8, 0, 0, 0])
        mask = np.ones(10, dtype=bool)
        mask[1] = False
        self.assertEqual(choose_marginal_action(p, mask, .95), 6)

    def test_fire_preference_for_another_direction_does_not_leak_into_selected_movement(self):
        from learning_lab.decoding import choose_marginal_action
        p = np.array([.15, 0, .45, 0, 0, 0, 0, 0, 0, .4])
        self.assertEqual(choose_marginal_action(p, np.ones(10, dtype=bool), .1), 2)

    def test_invalid_distribution_or_threshold_is_not_silently_defaulted(self):
        from learning_lab.decoding import choose_marginal_action
        mask = np.ones(10, dtype=bool)
        for p in (np.zeros(10), np.full(10, np.nan), np.ones(9), -np.ones(10)):
            with self.assertRaises(ValueError):
                choose_marginal_action(p, mask, .1)
        for threshold in (-.1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                choose_marginal_action(np.ones(10), mask, threshold)
