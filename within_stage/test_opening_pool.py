"""Synthetic-only opening pool tests; no emulator/data collection."""
from pathlib import Path
import shutil
import unittest
import uuid

from within_stage import opening_pool as pool


class OpeningPoolTests(unittest.TestCase):
    def test_predeclared_configs_cover_calibration_and_broad_waits_without_duplicates(self):
        configs = pool.menu_configurations()
        self.assertEqual(len(configs), 112)
        self.assertEqual(len(set(configs)), 112)
        self.assertEqual(configs[:8], pool.CALIBRATION)
        self.assertGreaterEqual(max(stage for _, stage in configs), 1023)
        self.assertLessEqual(len(configs) + len(pool.CALIBRATION), 128)

    def test_conservative_accounting_covers_raw_gameplay_and_wrapper_calls(self):
        account = pool.PoolAccounting(7, deadline=float("inf"))
        account.before("raw")
        account.after("raw")
        account.before("agent")
        account.after("agent")
        account.agent_raw_frames += 4
        self.assertEqual(account.charged_bound, 6)
        self.assertEqual(account.snapshot()["completed_all_calls"], 6)
        with self.assertRaises(pool.ProbeStop):
            account.before("agent")
        self.assertEqual(account.charged_bound, 6)

    def test_family_grouping_does_not_count_aliases_as_independent(self):
        trials = [
            {"config_index": 0, "status": "valid", "future_semantic_sha256": "a"},
            {"config_index": 1, "status": "valid", "future_semantic_sha256": "a"},
            {"config_index": 2, "status": "valid", "future_semantic_sha256": "b"},
            {"config_index": 3, "status": "invalid_initial_conditions"},
        ]
        families = pool.group_families(trials)
        self.assertEqual(len(families), 2)
        self.assertEqual(families[0]["config_indices"], [0, 1])

    def test_pool_roundtrip_is_unsplit_spec_only_and_rejects_corruption(self):
        root = Path("runs") / ("pool-test-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root)
        value = {"format": pool.FORMAT, "stage": 1, "handoff": 150, "frozen": False,
                 "partition": None, "reference": {}, "specs": [
                     {"config_index": 0, "spec": {"title_wait": 0, "stage_wait": 1,
                                                 "post_start_frames": 150, "legacy_phase": None}}],
                 "families": [{"family_id": 0, "config_indices": [0]}]}
        pool.save_pool(root, value)
        self.assertEqual(pool.load_pool(root), value)
        with (root / "pool.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            pool.load_pool(root)

    def test_uniform_handoff_cannot_silently_vary_per_case(self):
        value = {"format": pool.FORMAT, "stage": 1, "handoff": 150, "frozen": False,
                 "partition": None, "reference": {}, "specs": [
                     {"config_index": 0, "spec": {"title_wait": 0, "stage_wait": 0,
                                                 "post_start_frames": 134, "legacy_phase": None}}],
                 "families": []}
        with self.assertRaises(ValueError):
            pool.validate_pool(value)


if __name__ == "__main__":
    unittest.main()
