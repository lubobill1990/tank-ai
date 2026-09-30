from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

import game_state as gs
from battle_city_env import BattleCityEnv


class PrecisionObservationTests(unittest.TestCase):
    def state(self):
        ram = np.zeros(2048, dtype=np.uint8)
        ram[0x68], ram[0x80], ram[0x51], ram[0x85], ram[0xaa] = 0x80, 1, 3, 1, 0x80
        ram[0x100] = 255
        arena = ram[0x400:0x800].reshape(32, 32)[2:28, 2:28]
        arena[24:26, 12:14] = ((200, 201), (202, 203))
        ram[0x90], ram[0x98], ram[0xa0] = 82, 160, 0x80 | gs.ROM_FACING.index(gs.UP)
        ram[0x92], ram[0x9a], ram[0xa2] = 90, 80, 0xa0 | gs.ROM_FACING.index(gs.DOWN)
        tracker = gs.StateTracker()
        tracker.reset(ram)
        return tracker.state

    @staticmethod
    def base(state):
        holder = SimpleNamespace(state=state, max_frames=12000, objective='full', _last_macro=-1)
        return BattleCityEnv._observation(holder)

    def test_indistinguishable_firing_lanes_are_separated_without_changing_legacy_planes(self):
        from within_stage.observation import encode_precision
        first = self.state()
        second = replace(first, enemies=(replace(first.enemies[0], x=92),))
        left, right = self.base(first), self.base(second)
        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
        a, b = encode_precision(first, left), encode_precision(second, right)
        self.assertEqual(a.shape, (55, 26, 26))
        self.assertEqual(a.dtype, np.float32)
        np.testing.assert_array_equal(a[:47].view(np.uint32), left.view(np.uint32))
        self.assertFalse(np.array_equal(a, b))
        cells = gs.tank_cells(first.enemies[0].x, first.enemies[0].y)
        for row, column in cells:
            self.assertAlmostEqual(float(a[47, row, column]), 2 / 7)
            self.assertAlmostEqual(float(b[47, row, column]), 4 / 7)

    def test_offsets_and_inputs_remain_owned_and_finite(self):
        from within_stage.observation import encode_precision
        state = self.state()
        original = self.base(state)
        saved = original.copy()
        output = encode_precision(state, original)
        output[0].fill(99)
        np.testing.assert_array_equal(original, saved)
        self.assertTrue(np.isfinite(output[47:]).all())
        self.assertTrue(((output[47:] >= 0) & (output[47:] <= 1)).all())

    def test_bad_legacy_contract_is_explicitly_rejected(self):
        from within_stage.observation import encode_precision
        state = self.state()
        for observation in (np.zeros((46, 26, 26), np.float32),
                            np.zeros((47, 26, 26), np.uint8)):
            with self.assertRaises(ValueError):
                encode_precision(state, observation)

    def test_bullet_and_item_offsets_keep_team_channels_and_clip_outside_cells(self):
        from within_stage.observation import encode_precision
        state = self.state()
        modified = replace(
            state, enemies=(), bullets=(
                SimpleNamespace(owner=0, x=83, y=86),
                SimpleNamespace(owner=2, x=92, y=95),
                SimpleNamespace(owner=1, x=100, y=100),
                SimpleNamespace(owner=2, x=255, y=255),
            ), item=SimpleNamespace(x=105, y=110))
        result = encode_precision(modified, self.base(state))
        self.assertAlmostEqual(float(result[49, 8, 8]), 3 / 7)
        self.assertAlmostEqual(float(result[50, 8, 8]), 6 / 7)
        self.assertAlmostEqual(float(result[51, 9, 9]), 4 / 7)
        self.assertAlmostEqual(float(result[52, 9, 9]), 1)
        self.assertAlmostEqual(float(result[53, 11, 11]), 1 / 7)
        self.assertAlmostEqual(float(result[54, 11, 11]), 6 / 7)
        self.assertEqual(int(np.count_nonzero(result[49:])), 6)

    def test_overlapping_enemy_offsets_are_explicit_means_not_clipped_sums(self):
        from within_stage.observation import encode_precision
        state = self.state()
        state = replace(state, enemies=(state.enemies[0], replace(state.enemies[0], x=92)))
        result = encode_precision(state, self.base(state))
        self.assertAlmostEqual(float(result[47, 8, 9]), 3 / 7)
