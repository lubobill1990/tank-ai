"""Synthetic collector tests: zero ROM/native controller decisions."""
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
import uuid

import numpy as np

from learning_lab import collect_teacher_v2 as collector
from learning_lab.demonstrations import DemonstrationDataset


class FakeTeacher:
    def reset(self):
        pass

    def act(self, state, masks):
        return 2 * (state.tick % 2)


class FakeROM:
    """Borrowed observations and deterministic outcomes exercise collector boundaries."""
    def __init__(self, *, length=3, fail_on_step=None):
        self.length, self.fail_on_step = length, fail_on_step
        self.observation = np.zeros((47, 26, 26), np.float32)
        self.seeds, self.executed = [], []
        self.calls = 0
        self.closed = False

    def reset(self, *, seed):
        self.seed, self.tick = seed, 0
        self.seeds.append(seed)
        self.observation.fill(0)
        self._state(False)
        return self.observation, self._info('')

    def _state(self, done):
        clear = done and self.seed != 1
        self.state = SimpleNamespace(tick=self.tick, remaining_enemies=0 if clear else 20,
                                     base_destroyed=done and not clear, stage=1)

    def _info(self, outcome):
        return {'frames': self.tick * 4, 'kills': 20 if outcome == 'clear' else 0,
                'deaths': 0, 'outcome': outcome, 'objective': 'full', 'stage': 1}

    def action_masks(self):
        mask = np.zeros(10, bool)
        mask[[0, 2, 5]] = True
        return mask

    def step(self, action):
        self.calls += 1
        self.executed.append(action)
        if self.calls == self.fail_on_step:
            raise RuntimeError('synthetic step failure')
        self.tick += 1
        self.observation[0, 0, 0] = self.tick / 1000
        done = self.tick == self.length
        outcome = ('clear' if self.seed != 1 else 'base_lost') if done else ''
        self._state(done)
        return self.observation, 0., done, False, self._info(outcome)

    def close(self):
        self.closed = True


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('collector-v2-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def collect(self, env=None, **kwargs):
        env = env or FakeROM()
        config = {'max_episodes': 8, 'max_decisions': 1000, 'deadline': time.time() + 60}
        config.update(kwargs)
        summary = collector.collect(
            self.root / 'all', env_factory=lambda: env, teacher_factory=FakeTeacher, **config)
        return summary, env

    def test_schedule_is_only_stage_one_training_seeds_and_exact_perturbation_rng(self):
        schedule = [collector.episode_config(i) for i in range(24)]
        self.assertEqual([row['seed'] for row in schedule], [0, 1, 2] * 8)
        self.assertEqual([row['perturbation_probability'] for row in schedule], [0.] * 6 + [.02] * 18)
        self.assertEqual([row['perturbation_rng_seed'] for row in schedule], list(range(601, 625)))
        self.assertTrue(all(row['stage'] == 1 for row in schedule))
        for invalid in (-1, 24, True, 1.5):
            with self.assertRaises(ValueError):
                collector.episode_config(invalid)

    def test_labels_are_before_steps_borrowed_frames_are_owned_and_failures_retained(self):
        summary, env = self.collect(FakeROM(length=40))
        self.assertEqual(summary['status'], 'completed', summary.get('error'))
        self.assertEqual(summary['attempted_decisions'], 320)
        self.assertEqual(summary['completed_decisions'], 320)
        self.assertEqual(summary['clear_episodes'], 5)
        self.assertEqual(summary['outcomes'], {'clear': 5, 'base_lost': 3})
        self.assertEqual(env.seeds, [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertTrue(env.closed)
        dataset = DemonstrationDataset(self.root / 'all')
        self.addCleanup(dataset.close)
        batch = dataset.batch(np.arange(320))
        np.testing.assert_array_equal(batch['actions'], np.tile([0, 2] * 20, 8))
        np.testing.assert_array_equal(batch['observations'][:, 47, 0, 0],
                                      np.tile(np.arange(40, dtype=np.float32) / 1000, 8))
        executed = np.load(self.root / 'all' / 'executed_actions.npy')
        np.testing.assert_array_equal(executed, env.executed)
        expected = []
        for episode in range(8):
            rng = np.random.default_rng(601 + episode)
            for tick in range(40):
                action = 2 * (tick % 2)
                if episode >= 6 and rng.random() < .02:
                    action = int(rng.choice([0, 2, 5]))
                expected.append(action)
        np.testing.assert_array_equal(executed, expected)
        self.assertTrue(dataset.metadata['development_only'])
        self.assertFalse(dataset.metadata['learned_model'])
        collector.verify_integrity(self.root / 'all')

    def test_attempts_include_failing_step_while_completed_steps_do_not(self):
        summary, env = self.collect(FakeROM(fail_on_step=2))
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['attempted_decisions'], 2)
        self.assertEqual(summary['completed_decisions'], 1)
        self.assertEqual(summary['stored_samples'], 2)
        self.assertIn('synthetic step failure', summary['error'])
        self.assertEqual(summary['outcomes'], {'aborted': 1})
        self.assertTrue(env.closed)
        reports = json.loads((self.root / 'all' / 'episode_reports.json').read_text())
        self.assertFalse(reports[0]['episode_finished'])
        self.assertEqual(reports[0]['attempted_decisions'], 2)
        self.assertEqual(reports[0]['completed_decisions'], 1)
        copied = collector.copy_success_episodes(self.root / 'all', self.root / 'success',
                                                 deadline=time.time() + 60)
        self.assertEqual(copied['copied_episodes'], 0)
        self.assertEqual(copied['copied_samples'], 0)
        collector.verify_integrity(self.root / 'success')

    def test_decision_cap_preserves_partial_episode_without_overshoot(self):
        summary, env = self.collect(FakeROM(length=40), max_decisions=5)
        self.assertEqual(summary['attempted_decisions'], 5)
        self.assertEqual(summary['completed_decisions'], 5)
        self.assertEqual(env.calls, 5)
        self.assertEqual(summary['outcomes'], {'decision_budget': 1})
        self.assertEqual(summary['stop_reason'], 'decision_budget')

    def test_expired_deadline_does_not_construct_or_step_environment(self):
        def forbidden():
            raise AssertionError('No environment should be constructed')
        summary = collector.collect(self.root / 'all', deadline=0, env_factory=forbidden,
                                    teacher_factory=FakeTeacher)
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['attempted_decisions'], 0)
        self.assertEqual(summary['completed_decisions'], 0)
        self.assertEqual(summary['stop_reason'], 'deadline')

    def test_success_copy_keeps_whole_episode_history_labels_masks_and_executed_actions(self):
        source, _ = self.collect()
        result = collector.copy_success_episodes(self.root / 'all', self.root / 'success',
                                                 deadline=time.time() + 60)
        self.assertEqual(result['new_environment_decisions'], 0)
        self.assertEqual(result['copied_episodes'], 5)
        self.assertEqual(result['copied_samples'], 15)
        full = DemonstrationDataset(self.root / 'all')
        subset = DemonstrationDataset(self.root / 'success')
        self.addCleanup(full.close)
        self.addCleanup(subset.close)
        origin = np.load(self.root / 'success' / 'origin_indices.npy')
        self.assertEqual(origin.tolist(), [0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20])
        a, b = full.batch(origin), subset.batch(np.arange(len(subset)))
        for key in ('observations', 'actions', 'action_masks'):
            self.assertEqual(a[key].tobytes(), b[key].tobytes())
        original_executed = np.load(self.root / 'all' / 'executed_actions.npy')
        copied_executed = np.load(self.root / 'success' / 'executed_actions.npy')
        np.testing.assert_array_equal(copied_executed, original_executed[origin])
        self.assertTrue(subset.metadata['success_filtered'])
        self.assertIn('not independent generalization', subset.metadata['warning'])
        self.assertEqual(subset.metadata['source_attempted_decisions'], source['attempted_decisions'])
        collector.verify_integrity(self.root / 'all')
        collector.verify_integrity(self.root / 'success')

    def test_copy_rejects_changed_source_and_never_overwrites_outputs(self):
        self.collect()
        with self.assertRaises(FileExistsError):
            self.collect()
        with (self.root / 'all' / 'frames.bin').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaises(ValueError):
            collector.copy_success_episodes(self.root / 'all', self.root / 'success',
                                             deadline=time.time() + 60)
        self.assertFalse((self.root / 'success').exists())


if __name__ == '__main__':
    unittest.main()
