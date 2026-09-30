import unittest
import numpy as np


class TemperedPolicyTests(unittest.TestCase):
    def test_temperature_sharpens_without_changing_argmax_or_reviving_masked_actions(self):
        from learning_lab.tempered_policy import tempered_probabilities
        p = np.array([.6, .3, .1, 0, 0, 0, 0, 0, 0, 0])
        q = tempered_probabilities(p, .5)
        self.assertAlmostEqual(q[0], .36 / .46)
        self.assertEqual(int(q.argmax()), 0)
        np.testing.assert_array_equal(q[3:], 0)
        np.testing.assert_allclose(tempered_probabilities(p, 1), p)

    def test_invalid_distributions_and_temperatures_fail(self):
        from learning_lab.tempered_policy import tempered_probabilities
        for temperature in (0, -.5, True, float('nan')):
            with self.assertRaises(ValueError):
                tempered_probabilities(np.ones(10), temperature)
        for p in (np.zeros(10), -np.ones(10), np.full(10, np.nan), np.ones(9)):
            with self.assertRaises(ValueError):
                tempered_probabilities(p, .5)
