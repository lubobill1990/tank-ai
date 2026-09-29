import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def healthy_host():
    return {'commit_available_bytes': 64 * 1024**3, 'commit_total_bytes': 32 * 1024**3,
            'commit_limit_bytes': 96 * 1024**3, 'physical_available_bytes': 16 * 1024**3,
            'process_private_bytes': 1024**3, 'process_rss_bytes': 512 * 1024**2,
            'process_peak_private_bytes': 1024**3, 'host_budget_kind': 'windows_commit'}


class ResourceTests(unittest.TestCase):
    def test_pressure_records_sample_and_stops_at_reserve_not_after_allocation_failure(self):
        import rust_resources as resources
        host = healthy_host()
        host['commit_available_bytes'] = 8 * 1024**3 - 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(resources, 'host_snapshot', return_value=host):
            path = Path(directory) / 'resources.jsonl'
            monitor = resources.ResourceMonitor(path)
            with self.assertRaises(resources.ResourcePressureError):
                monitor.check('rollout')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['phase'], 'rollout')
            self.assertEqual(rows[0]['commit_available_bytes'], 8 * 1024**3 - 1)
            self.assertEqual(monitor.latest, rows[0])

    def test_admission_accounts_for_both_workers_starting_before_their_memory_ramps(self):
        import rust_resources as resources
        host = healthy_host()
        host['commit_available_bytes'] = 20 * 1024**3
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(resources, 'host_snapshot', return_value=host):
            monitor = resources.ResourceMonitor(Path(directory) / 'resources.jsonl')
            monitor.check('first-child', admission_slots=1, force=True)
            with self.assertRaises(resources.ResourcePressureError):
                monitor.check('second-child', admission_slots=2, force=True)

    def test_boundary_allows_reserve_and_sampling_is_throttled_but_admission_is_fresh(self):
        import rust_resources as resources
        host = healthy_host()
        host['commit_available_bytes'] = 8 * 1024**3
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(resources, 'host_snapshot', return_value=host), \
                patch.object(resources.time, 'monotonic', return_value=10):
            path = Path(directory) / 'resources.jsonl'
            monitor = resources.ResourceMonitor(path)
            for _ in range(100):
                monitor.check('rollout')
            self.assertEqual(len(path.read_text().splitlines()), 1)
            host['commit_available_bytes'] -= 1
            with self.assertRaises(resources.ResourcePressureError):
                monitor.check('optimizer', force=True)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_sampler_failure_never_becomes_zero_usage_or_a_successful_check(self):
        import rust_resources as resources
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(resources, 'host_snapshot', side_effect=OSError('counter unavailable')):
            monitor = resources.ResourceMonitor(Path(directory) / 'resources.jsonl')
            with self.assertRaisesRegex(OSError, 'counter unavailable'):
                monitor.check('rollout')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows memory counters')
    def test_live_windows_counters_measure_commit_and_private_memory_consistently(self):
        import rust_resources as resources
        row = resources.host_snapshot()
        self.assertEqual(row['commit_available_bytes'],
                         row['commit_limit_bytes'] - row['commit_total_bytes'])
        self.assertGreater(row['process_private_bytes'], 0)
        self.assertGreaterEqual(row['process_peak_private_bytes'], row['process_private_bytes'])
