"""Synthetic native-bridge tests: zero real ROM/native controller decisions."""
import json
from pathlib import Path
import shutil
import time
import unittest
import uuid

import numpy as np

import game_state as gs
from learning_lab import native_teacher_data as pilot
from learning_lab.demonstrations import DemonstrationDataset


def initial_ram(stage=1):
    ram = np.zeros(2048, np.uint8)
    ram[0x68], ram[0x51], ram[0x80], ram[0x7F], ram[0x85] = 0x80, 3, 20, 20, stage
    ram[0xA0], ram[0x90], ram[0x98] = 0x80, 88, 216
    return ram


class FakeNative:
    def __init__(self, stage=1, max_frames=8):
        self.initial = initial_ram(stage)
        self.max_frames = max_frames
        self.seed = None
        self.support_code = 0
        self.corrupt_observation = False
        self.reset(0)

    def reset(self, seed):
        self.seed = seed
        self.current = self.initial.copy()
        self.frame = 0

    def ram(self):
        return self.current.copy()

    def counters(self):
        return np.array([self.frame, 0, 0, 0, 0], np.uint32)

    def frame_step(self, control):
        self.frame += 1
        self.current[0x0B] = (int(self.current[0x0B]) + 1) % 256

    def observe(self):
        result = pilot.canonical_observation(gs.parse(self.current, frame=self.frame), self.max_frames)
        if self.corrupt_observation:
            result[36, 0, 0] -= .25
        return result

    def masks(self):
        return pilot.canonical_masks(gs.parse(self.current, frame=self.frame))

    def support(self):
        return self.support_code

    def close(self):
        pass


class FakeTeacher:
    def reset(self):
        pass

    def act(self, state, masks):
        if state.frame not in (0, 4):
            raise AssertionError('Teacher received a wrong per-environment clock')
        return 0


class NativeTeacherTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('native-teacher-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def bridge(self, stream=0, stage=1, backend=None):
        result = pilot.FrameBridge(backend or FakeNative(stage), stage=stage,
                                   stream_id=stream, master_seed=100, max_frames=8)
        self.addCleanup(result.close)
        return result

    def test_frame_tracker_advances_each_real_frame_and_terminal_snapshot_precedes_autoreset(self):
        bridge = self.bridge()
        first_seed = bridge.seed
        first = bridge.step(0)
        self.assertFalse(first['done'])
        self.assertEqual(bridge.state.frame, 4)
        self.assertEqual(first['stats']['frames'], 4)
        second = bridge.step(0)
        self.assertTrue(second['done'])
        self.assertEqual(second['outcome'], 'timeout')
        self.assertEqual(second['stats']['frames'], 8)
        self.assertEqual(second['state'].frame, 8)
        self.assertEqual(second['episode_seed'], first_seed)
        self.assertEqual(float(second['observation'][36, 0, 0]), 0.)
        self.assertEqual(bridge.state.frame, 0)
        self.assertEqual(float(bridge.observation[36, 0, 0]), 1.)
        self.assertEqual(bridge.episode_id, 1)
        self.assertNotEqual(bridge.seed, first_seed)
        self.assertNotEqual(self.bridge(stream=1).seed, first_seed)

    def test_bridge_rejects_bad_observations_support_flags_and_illegal_actions(self):
        bad = FakeNative()
        bad.corrupt_observation = True
        with self.assertRaisesRegex(pilot.BridgeError, 'observation'):
            self.bridge(backend=bad)
        unsupported = FakeNative()
        unsupported.support_code = 32
        with self.assertRaisesRegex(pilot.BridgeError, 'support'):
            self.bridge(backend=unsupported)
        bridge = self.bridge()
        for invalid in (10, -1, True):
            with self.assertRaises(ValueError):
                bridge.step(invalid)
        self.assertEqual(bridge.state.frame, 0)

    def test_same_frame_spawn_kill_keeps_canonical_observer_and_native_counts_separate(self):
        class SameFrameKill(FakeNative):
            def __init__(self):
                super().__init__()
                self.initial[0xA2] = 0xED
                self.reset(0)

            def frame_step(self, control):
                super().frame_step(control)
                self.current[0xA2] = 0x73

            def counters(self):
                result = super().counters()
                result[1] = int(self.frame > 0)
                return result

        bridge = self.bridge(backend=SameFrameKill())
        result = bridge.step(0)
        self.assertEqual(result['state'].frame, 4)
        self.assertEqual(result['state'].kills, 0)
        self.assertEqual(result['stats']['kills'], 1)
        self.assertEqual(len(bridge.observer_counter_differences), 1)
        difference = bridge.observer_counter_differences[0]
        self.assertEqual(difference['tracker_kills'], 0)
        self.assertEqual(difference['native_kills'], 1)
        self.assertEqual(bridge.max_observation_error, 0)

    def test_streamed_datasets_do_not_interleave_histories_and_quality_gate_stops(self):
        streams = [self.bridge(0, 1), self.bridge(1, 3)]
        budget = pilot.DecisionBudget(100, time.time() + 60)
        result = pilot.collect_streams(self.root / 'data', streams, budget,
                                        teacher_factory=FakeTeacher, monitor=None, max_complete=64)
        self.assertEqual(result['stop_reason'], 'pilot_quality_gate')
        self.assertEqual(result['completed_episodes'], 16)
        self.assertEqual(result['clear_episodes'], 0)
        self.assertEqual(result['pilot_gate']['mean_kills'], 0.)
        self.assertEqual(budget.counts['native_collection_attempted'], 32)
        self.assertEqual(budget.counts['native_collection_completed'], 32)
        self.assertEqual(len(result['dataset_paths']), 2)
        for path in result['dataset_paths']:
            dataset = DemonstrationDataset(path)
            self.addCleanup(dataset.close)
            batch = dataset.batch(np.arange(len(dataset)))
            self.assertEqual(len(dataset), 16)
            np.testing.assert_array_equal(batch['episode_ids'], np.repeat(np.arange(8), 2))
            np.testing.assert_array_equal(batch['observations'][::2, :47], 0)
            np.testing.assert_array_equal(batch['observations'][1::2, :47],
                                          batch['observations'][::2, 47:])
            np.testing.assert_array_equal(batch['observations'][:, 47 + 36, 0, 0], [1., .5] * 8)
            self.assertFalse(dataset.metadata['learned_model'])
            self.assertEqual(dataset.metadata['backend'], 'NATIVE-seeded-experimental')
            self.assertFalse(dataset.metadata['rom_wins'])

    def test_global_budget_counts_all_probe_collection_and_initialization_calls(self):
        budget = pilot.DecisionBudget(3, time.time() + 60)
        budget.charge('rom_initialization')
        budget.complete('rom_initialization')
        budget.charge('native_probe')
        budget.complete('native_probe')
        budget.charge('native_collection')
        self.assertEqual(budget.attempted, 3)
        self.assertEqual(budget.completed, 2)
        with self.assertRaises(pilot.StopBudget):
            budget.charge('native_collection')
        self.assertEqual(budget.attempted, 3)

    def test_budget_stop_preserves_incomplete_episode_and_owned_start_history(self):
        budget = pilot.DecisionBudget(1, time.time() + 60)
        result = pilot.collect_streams(self.root / 'limited', [self.bridge()], budget,
                                        teacher_factory=FakeTeacher, monitor=None, max_complete=64)
        self.assertEqual(result['stop_reason'], 'decision_budget')
        self.assertEqual(result['completed_episodes'], 0)
        path = Path(result['dataset_paths'][0])
        reports = json.loads((path / 'episode_reports.json').read_text())
        self.assertEqual(reports[0]['outcome'], 'decision_budget')
        self.assertEqual(reports[0]['decisions'], 1)
        dataset = DemonstrationDataset(path)
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), 1)
        np.testing.assert_array_equal(dataset.batch([0])['observations'][0, :47], 0)


if __name__ == '__main__':
    unittest.main()
