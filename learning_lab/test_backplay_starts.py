"""Synthetic/fake-native snapshot-pool tests; zero real environment decisions."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np

import game_state as gs
from learning_lab import backplay_starts as backplay


def ram(remaining=4):
    result = np.zeros(2048, np.uint8)
    result[0x68], result[0x51], result[0x85] = 0x80, 3, 1
    result[0x80] = remaining
    result[0xa0], result[0xa2] = 0x80, 0x82
    result[0x90], result[0x98], result[0x92], result[0x9a] = 88, 208, 24, 24
    return result


class FakeMonitor:
    def check(self, *args, **kwargs):
        pass


class FakeNative:
    automatic_clear = False

    def __init__(self, initial, max_frames, *, seeds):
        self.initial = initial[0].copy()
        self.current = self.initial.copy()
        self.maximum, self.frame = max_frames, 0
        self.support = np.zeros(1, np.uint32)
        self.closed = False

    def reset(self, index, *, seed):
        self.current = self.initial.copy()
        self.frame = 0
        return self.observe()[0]

    def observe(self):
        observation = np.zeros((1, 47, 26, 26), np.float32)
        observation[:, 36] = max(0, 1 - self.frame / self.maximum)
        return observation

    def ram(self, index=0):
        return self.current.copy()

    def masks(self):
        return np.ones((1, 10), bool)

    def step(self, actions):
        assert actions.tolist() == [0]
        self.frame += 4
        if self.automatic_clear:
            self.current[0x80] = 0
        done = self.automatic_clear or self.frame >= self.maximum
        return self.observe(), np.zeros(1), np.array([done]), np.array([[self.frame, 0, 0, 0, 0]])

    def close(self):
        self.closed = True


class BackplayTests(unittest.TestCase):
    def test_eligibility_requires_live_unassisted_nonterminal_opposition(self):
        current = gs.parse(ram())
        self.assertTrue(backplay.eligible_snapshot(current))
        invalid = (
            replace(current, player=None), replace(current, enemies=()),
            replace(current, base_destroyed=True), replace(current, game_over=True),
            replace(current, fire_ready=False), replace(current, remaining_enemies=0),
            replace(current, stage=2),
            replace(current, bullets=(gs.Bullet(0, 1, 0, 80, 80, 0, -2),)),
            replace(current, bullets=(gs.Bullet(8, 1, 0, 80, 80, 0, -4),)),
        )
        for candidate in invalid:
            self.assertFalse(backplay.eligible_snapshot(candidate))
        self.assertTrue(backplay.eligible_snapshot(
            replace(current, bullets=(gs.Bullet(2, 1, 2, 80, 80, 0, 2),))))

    def test_selector_enforces_spacing_deduplication_and_owns_ram(self):
        selector = backplay.SnapshotSelector(per_bucket=4, min_spacing_frames=64)
        original = ram()
        current = gs.parse(original, frame=100)
        self.assertTrue(selector.consider(current, original, seed=0, episode_id=0,
                                          initial_ram_sha256="a" * 64))
        original[0] = 7
        self.assertEqual(selector.candidates[0]["ram"][0], 0)
        self.assertFalse(selector.consider(replace(current, frame=104), original, seed=0,
                                           episode_id=0, initial_ram_sha256="a" * 64))
        self.assertTrue(selector.consider(replace(current, frame=164), original, seed=0,
                                          episode_id=0, initial_ram_sha256="a" * 64))
        self.assertFalse(selector.consider(replace(current, frame=228), original, seed=0,
                                           episode_id=0, initial_ram_sha256="a" * 64))

    def test_selector_rejects_unauthorized_openings(self):
        selector = backplay.SnapshotSelector()
        with self.assertRaises(ValueError):
            selector.consider(gs.parse(ram()), ram(), seed=7, episode_id=0,
                              initial_ram_sha256="a" * 64)

    def test_same_opening_spacing_does_not_reset_with_episode_or_seed(self):
        selector = backplay.SnapshotSelector(per_bucket=4, min_spacing_frames=64)
        initial = ram()
        current = gs.parse(initial, frame=100)
        self.assertTrue(selector.consider(current, initial, seed=0, episode_id=0,
                                          initial_ram_sha256="a" * 64))
        later = initial.copy()
        later[0] = 9
        self.assertFalse(selector.consider(replace(current, frame=104), later, seed=2, episode_id=2,
                                           initial_ram_sha256="a" * 64))

    def test_combined_budget_counts_domains_and_stops_before_overspend(self):
        budget = backplay.DecisionBudget(3, deadline=float("inf"))
        budget.take("rom")
        budget.take("native")
        budget.take("native")
        self.assertEqual((budget.rom_decisions, budget.native_decisions), (1, 2))
        with self.assertRaises(backplay.BudgetStop):
            budget.take("rom")
        self.assertEqual(budget.total, 3)

    def test_native_noop_clear_is_rejected_and_counted(self):
        budget = backplay.DecisionBudget(10, deadline=float("inf"))
        with patch.object(backplay, "NativeBatch", FakeNative), \
                patch.object(FakeNative, "automatic_clear", True):
            report = backplay.probe_noop(ram(), native_seed=1000, budget=budget,
                                         monitor=FakeMonitor(), max_frames=8)
        self.assertTrue(report["automatic_clear"])
        self.assertFalse(report["retained"])
        self.assertEqual(budget.native_decisions, 1)
        self.assertTrue(report["initial_ram_byte_exact"])
        self.assertEqual(report["initial_time_left"], 1.)

    def test_supported_nonclearing_native_start_is_retained(self):
        budget = backplay.DecisionBudget(10, deadline=float("inf"))
        with patch.object(backplay, "NativeBatch", FakeNative):
            report = backplay.probe_noop(ram(), native_seed=1000, budget=budget,
                                         monitor=FakeMonitor(), max_frames=8)
        self.assertTrue(report["retained"])
        self.assertEqual(report["outcome"], "timeout")
        self.assertEqual(report["decisions"], 2)
        self.assertEqual(report["frames"], 8)

    def test_unsupported_native_import_is_retained_as_negative_evidence(self):
        budget = backplay.DecisionBudget(10, deadline=float("inf"))
        with patch.object(backplay, "NativeBatch", side_effect=RuntimeError("Unsupported native mechanics")):
            report = backplay.probe_noop(ram(), native_seed=1000, budget=budget,
                                         monitor=FakeMonitor(), max_frames=8)
        self.assertFalse(report["retained"])
        self.assertEqual(report["decisions"], 0)
        self.assertIn("Unsupported", report["error"])

    def test_pool_roundtrip_hashes_and_shapes(self):
        root = Path("runs") / ("backplay-test-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root)
        selector = backplay.SnapshotSelector()
        selector.consider(gs.parse(ram(), frame=100), ram(), seed=0, episode_id=0,
                          initial_ram_sha256="a" * 64)
        backplay.save_pool(root / "starts.npz", selector.candidates)
        loaded = backplay.load_backplay_starts(root)
        self.assertEqual(loaded["ram"].shape, (1, 2048))
        self.assertEqual(loaded["ram"].dtype, np.uint8)
        self.assertEqual(loaded["teacher_frames"].tolist(), [100])
        self.assertEqual(loaded["remaining"].tolist(), [4])
        self.assertEqual(loaded["full_start"].tolist(), [False])
        loaded["ram"][0, 0] = 9
        self.assertEqual(backplay.load_backplay_starts(root)["ram"][0, 0], 0)

    def test_public_pool_separates_remaining20_curriculum_from_exact_full_start(self):
        root = Path("runs") / ("backplay-test-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root)
        selector = backplay.SnapshotSelector()
        selector.consider(gs.parse(ram(20), frame=128), ram(20), seed=0, episode_id=0,
                          initial_ram_sha256="a" * 64)
        backplay.save_pool(root / "starts.npz", selector.candidates)
        (root / "report.json").write_text(json.dumps({
            "native_probes": [{"candidate_id": 0, "outcome": "timeout", "retained": True,
                              "automatic_clear": False, "supported": True}],
            "missing_buckets": [1, 2, 4, 8, 12, 16],
        }))
        normal = ram(20)
        normal[0] = 1
        budget = backplay.DecisionBudget(10, deadline=float("inf"))
        with patch.object(backplay, "stage_ram", return_value=normal) as initializer, \
                patch.object(backplay, "NativeBatch", FakeNative):
            reference = backplay.create_normal_start_reference(root, budget=budget, monitor=FakeMonitor())
        initializer.assert_called_once_with(1, seed=0)
        self.assertEqual(budget.rom_decisions, 2)
        self.assertEqual(budget.native_decisions, 0)
        self.assertEqual(reference["initialization_method"], "rust_env.stage_ram(1, seed=0)")
        with patch.object(backplay, "stage_ram", side_effect=AssertionError("Loader must not step ROM")):
            loaded = backplay.load_backplay_pool(root)
        self.assertEqual(loaded["rams"].shape, (2, 2048))
        self.assertEqual(loaded["remaining"].tolist(), [20, 20])
        self.assertEqual(loaded["source_frames"].tolist(), [128, -1])
        self.assertEqual(loaded["full_start"].tolist(), [False, True])
        self.assertEqual(loaded["noop_tested"].tolist(), [True, False])
        np.testing.assert_array_equal(loaded["rams"][1], normal)

    def test_public_pool_excludes_any_recorded_automatic_clear(self):
        root = Path("runs") / ("backplay-test-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root)
        selector = backplay.SnapshotSelector()
        selector.consider(gs.parse(ram()), ram(), seed=0, episode_id=0,
                          initial_ram_sha256="a" * 64)
        backplay.save_pool(root / "starts.npz", selector.candidates)
        (root / "report.json").write_text(json.dumps({
            "native_probes": [{"candidate_id": 0, "outcome": "clear", "automatic_clear": True}],
        }))
        loaded = backplay.load_backplay_pool(root)
        self.assertEqual(loaded["rams"].shape, (0, 2048))
        self.assertEqual(loaded["automatic_noop_clear_exclusions"], [0])
        self.assertIsNone(loaded["normal_start"])


if __name__ == "__main__":
    unittest.main()
