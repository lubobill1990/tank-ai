"""Synthetic early-handoff tests; no ROM or learning-data generation."""
from dataclasses import replace
import unittest

import numpy as np

import game_state as gs
from within_stage import early_openings as early


def state():
    terrain = np.zeros((26, 26), np.uint8)
    terrain[24:26, 12:14] = gs.T_BASE
    masks = np.zeros_like(terrain)
    return gs.GameState(0, terrain, gs.Tank(0, 88, 216, gs.UP), (), (), 1, 3, 20, 0, 0,
                        None, False, False, True, fire_ready=True, brick_masks=masks)


class EarlyOpeningTests(unittest.TestCase):
    def test_readiness_requires_live_player_full_geometry_and_normal_counters(self):
        current = state()
        reference = early.geometry_reference(current)
        self.assertTrue(early.ready_state(current, reference))
        for changed in (replace(current, player=None), replace(current, lives=2),
                        replace(current, remaining_enemies=19), replace(current, base_destroyed=True),
                        replace(current, player=replace(current.player, power=1)),
                        replace(current, player=replace(current.player, x=96))):
            self.assertFalse(early.ready_state(changed, reference))
        changed = replace(current, brick_masks=current.brick_masks.copy())
        changed.brick_masks[2, 2] = 7
        self.assertFalse(early.ready_state(changed, reference))

    def test_fixed_handoff_comes_from_shared_pristine_ready_window(self):
        traces = [
            [{"post_start_frame": i, "ready": 10 <= i < 20} for i in range(30)],
            [{"post_start_frame": i, "ready": 12 <= i < 22} for i in range(30)],
        ]
        self.assertEqual(early.choose_handoff(traces), 12)
        with self.assertRaises(ValueError):
            early.choose_handoff([traces[0], [{"post_start_frame": 25, "ready": True}]])

    def test_plan_stays_bounded_and_handoff_is_explicit(self):
        plan = early.menu_specs(100)
        self.assertEqual(len(plan), 24)
        self.assertEqual(len({(s.title_wait, s.stage_wait) for s in plan}), 24)
        self.assertTrue(all(s.post_start_frames == 100 for s in plan))
        self.assertLessEqual(early.plan_upper_bound(), 50000)
        with self.assertRaises(ValueError):
            early.EarlyOpeningSpec(post_start_frames=-1)


if __name__ == "__main__":
    unittest.main()
