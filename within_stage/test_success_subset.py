"""Synthetic-only complete-clear subset tests; no real corpus copy or environment."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from within_stage import imitation
from within_stage.demonstrations import PrecisionWriter, PrecisionDataset
from within_stage.observation import OBSERVATION_VERSION
from within_stage.precision_policy import file_sha256
from within_stage import success_subset as subset


class SuccessSubsetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('success-subset-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True); self.addCleanup(shutil.rmtree, self.root)
        self.pool = self.root / 'pool.json'
        self.pool.write_text(json.dumps({'specs': [
            {'config_index': i, 'family_id': i, 'spec': {'config': i}} for i in range(41)]}))
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps({
            'observation_version': OBSERVATION_VERSION, 'channels_per_frame': 55,
            'n_stack': 2, 'actions': 10, 'stage': 1, 'post_start_frames': 150,
            'pool_path': str(self.pool.resolve()), 'pool_sha256': file_sha256(self.pool),
            'train': {'family_ids': list(range(40)), 'all_config_indices': list(range(40))}}))
        self.split_hash = file_sha256(self.split)
        for name, value in (('FROZEN_SPLIT_PATH', self.split), ('FROZEN_SPLIT_SHA256', self.split_hash)):
            handle = patch.object(imitation, name, value)
            handle.start(); self.addCleanup(handle.stop)

    def dataset(self, name, overrides=None, outside=False):
        path = self.root / name
        writer = PrecisionWriter(path)
        reports, plan = [], []
        try:
            for episode, outcome in enumerate(('clear', 'base_lost', 'clear', 'timeout')):
                family = 40 if outside and episode == 1 else episode
                origin = {'episode_id': episode, 'config_index': family, 'family_id': family,
                          'opening_spec': {'config': family}}
                plan.append(origin)
                for index in range(3):
                    frame = np.full((55, 26, 26), (episode * 3 + index) / 20, np.float32)
                    frame.view(np.uint32)[0, 0, 0] = 0x80000000
                    ram = np.full(2048, episode * 3 + index, np.uint8)
                    masks = np.zeros(10, bool); masks[[0, 2, 5]] = True
                    writer.add(frame, 2, masks, ram=ram, frame=100 + index * 4,
                               episode_start=index == 0, executed_action=5)
                row = {**origin, 'stage': 1, 'outcome': outcome, 'clear': outcome == 'clear',
                       'episode_finished': True, 'decisions': 3, 'completed_decisions': 3,
                       'error': None, 'terminal_remaining': 0 if outcome == 'clear' else 4,
                       'terminal_base_destroyed': outcome == 'base_lost'}
                row.update((overrides or {}).get(episode, {}))
                reports.append(row)
            writer.finish(reports, {'selected_partition': 'train', 'split_sha256': self.split_hash,
                                   'plan': plan, 'status': 'completed', 'collector_sources': {'synthetic': 'fixture'}})
        finally:
            writer.close()
        return path

    def test_copy_preserves_whole_episode_history_ram_frames_labels_masks_and_origins(self):
        paths = [self.dataset('a'), self.dataset('b')]
        source_hashes = [file_sha256(path / 'integrity.json') for path in paths]
        output = self.root / 'success'
        report = subset.copy_success_episodes(paths, output, self.split)
        self.assertEqual(report['copied_episodes'], 4)
        self.assertEqual(report['copied_samples'], 12)
        self.assertEqual(report['source_samples'], 24)
        self.assertEqual(report['environment_interactions'], 0)
        target = PrecisionDataset(output); self.addCleanup(target.close)
        mapping = json.loads((output / 'origin_mapping.json').read_text())
        self.assertEqual([row['source_episode_id'] for row in mapping], [0, 2, 0, 2])
        self.assertEqual([row['family_id'] for row in mapping], [0, 2, 0, 2])
        for row in mapping:
            source = PrecisionDataset(row['source_dataset_path'])
            try:
                old = np.arange(row['source_start'], row['source_stop'])
                new = np.arange(row['destination_start'], row['destination_stop'])
                left, right = source.batch(old), target.batch(new)
                for key in ('observations', 'actions', 'action_masks'):
                    self.assertEqual(left[key].tobytes(), right[key].tobytes())
                for key in ('ram', 'frames'):
                    self.assertEqual(source.audit(old)[key].tobytes(), target.audit(new)[key].tobytes())
                np.testing.assert_array_equal(right['observations'][0, :55], 0)
                self.assertEqual(row['source_sha256'], source.fingerprint)
            finally:
                source.close()
        np.testing.assert_array_equal(np.load(output / 'executed_actions.npy'), [5] * 12)
        self.assertEqual(source_hashes, [file_sha256(path / 'integrity.json') for path in paths])
        corpus = imitation.TrainingCorpus([output], imitation._training_contract(), None)
        self.addCleanup(corpus.close)
        self.assertEqual(len(corpus.episode_ids), 12)
        self.assertTrue(target.metadata['success_filtered'])
        self.assertIn('not representative', target.metadata['warning'])

    def test_failed_truncated_or_partial_rows_cannot_be_accepted_as_complete_clears(self):
        for i, change in enumerate((
            {'episode_finished': False}, {'truncated': True}, {'error': 'failed'},
            {'completed_decisions': 2}, {'terminal_remaining': 1}, {'terminal_base_destroyed': True},
        )):
            path = self.dataset(f'bad-{i}', overrides={0: change})
            output = self.root / f'out-{i}'
            report = subset.copy_success_episodes([path], output, self.split)
            self.assertEqual(report['copied_episodes'], 1)
            rows = json.loads((output / 'origin_mapping.json').read_text())
            self.assertEqual([row['source_episode_id'] for row in rows], [2])
            skipped = json.loads((output / 'skipped_episodes.json').read_text())
            self.assertTrue(any(row['source_episode_id'] == 0 for row in skipped))

    def test_no_complete_clears_produces_no_empty_dataset(self):
        source = self.dataset('no-clears', overrides={0: {'episode_finished': False},
                                                     2: {'episode_finished': False}})
        output = self.root / 'empty'
        with self.assertRaisesRegex(ValueError, 'No complete clear'):
            subset.copy_success_episodes([source], output, self.split)
        self.assertFalse(output.exists())

    def test_outside_train_open_input_and_duplicate_paths_fail_before_creating_output(self):
        outside = self.dataset('outside', outside=True)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            subset.copy_success_episodes([outside], self.root / 'bad', self.split)
        self.assertFalse((self.root / 'bad').exists())
        opened = self.root / 'still-open'; opened.mkdir()
        (opened / 'observations.bin').touch()
        with self.assertRaises((ValueError, FileNotFoundError)):
            subset.copy_success_episodes([opened], self.root / 'open-copy', self.split)
        self.assertFalse((self.root / 'open-copy').exists())
        valid = self.dataset('duplicate')
        with self.assertRaises(ValueError):
            subset.copy_success_episodes([valid, valid], self.root / 'dup-copy', self.split)

    def test_batches_are_bounded_and_output_cannot_overwrite_source(self):
        source = self.dataset('bounded')
        sizes = []
        original = PrecisionDataset.batch
        def batch(dataset, indices):
            sizes.append(len(indices))
            return original(dataset, indices)
        with patch.object(PrecisionDataset, 'batch', batch):
            subset.copy_success_episodes([source], self.root / 'bounded-output', self.split)
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 128)
        with self.assertRaises(FileExistsError):
            subset.copy_success_episodes([source], source, self.split)

    def test_failed_step_audit_excludes_nominal_clear_and_is_hash_bound(self):
        source = self.dataset('failed-step-audit')
        flags = np.ones(12, bool)
        flags[1] = False
        np.save(source / 'step_completed.npy', flags, allow_pickle=False)
        output = self.root / 'filtered'
        report = subset.copy_success_episodes([source], output, self.split)
        self.assertEqual(report['copied_episodes'], 1)
        self.assertEqual(report['skipped_episode_reasons']['failed_step_in_episode'], 1)
        metadata = json.loads((output / 'metadata.json').read_text())
        self.assertEqual(metadata['source_datasets'][0]['step_completed_sha256'],
                         file_sha256(source / 'step_completed.npy'))

    def test_source_corruption_is_rejected_before_destination_creation(self):
        source = self.dataset('corrupted')
        with (source / 'ram.bin').open('ab') as stream:
            stream.write(b'corrupt')
        output = self.root / 'corrupted-copy'
        with self.assertRaises(ValueError):
            subset.copy_success_episodes([source], output, self.split)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
