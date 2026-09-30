"""Synthetic CPU-only weighted cloning tests; no environment is stepped."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from learning_lab import imitation as base
from learning_lab import weighted_imitation as weighted
from learning_lab.demonstrations import DemonstrationDataset, DemonstrationWriter


class WeightedImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('weighted-imitation-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.data = cls.root / 'corpus'
        frame = np.zeros((47, 26, 26), np.float32)
        frame[4, 8:16, 8:16] = 1
        masks = np.zeros(10, bool)
        masks[[0, 2, 9]] = True
        writer = DemonstrationWriter(cls.data)
        try:
            for labels in ([0, 0, 0, 0], [0, 0, 0, 2], [9, 9, 9, 9],
                           [0, 0, 0, 0], [0, 0, 0, 2]):
                for index, label in enumerate(labels):
                    writer.add(frame, label, masks, episode_start=index == 0, executed_action=label)
            writer.finish([{'episode_id': i, 'outcome': 'timeout', 'decisions': 4}
                           for i in range(5)], {'label': 'synthetic-class-imbalance-fixture'})
        finally:
            writer.close()
        parent = base.train_imitation([cls.data], cls.root / 'parent', epochs=1,
                                      batch_size=4, seed=0, device='cpu')
        if parent['status'] != 'completed':
            raise AssertionError(parent)
        cls.parent = Path(parent['checkpoints']['final']['path'])
        cls.parent_report = parent
        cls.parent_hash = base._sha256(cls.parent)
        cls.parent_manifest_hash = base._sha256(cls.parent.with_suffix('.manifest.json'))
        cls.frozen_sources = base._source_hashes()
        cls.frozen_demo_hash = base._sha256(Path('learning_lab') / 'demonstrations.py')

        def verify_frozen_inputs():
            assert base._sha256(cls.parent) == cls.parent_hash
            assert base._sha256(cls.parent.with_suffix('.manifest.json')) == cls.parent_manifest_hash
            assert base._source_hashes() == cls.frozen_sources
            assert base._sha256(Path('learning_lab') / 'demonstrations.py') == cls.frozen_demo_hash

        cls.addClassCleanup(verify_frozen_inputs)

    def setUp(self):
        self.output = self.root / ('run-' + uuid.uuid4().hex)

    def train(self, **kwargs):
        config = {'initial_checkpoint': self.parent, 'epochs': 1, 'batch_size': 4, 'seed': 0}
        config.update(kwargs)
        return weighted.train_weighted_imitation([self.data], self.output, **config)

    def test_weights_preserve_example_mean_one_with_cap_and_explicit_zero_counts(self):
        counts = np.array([81, 9] + [0] * 8, dtype=np.int64)
        weights = weighted._class_weights(counts)
        np.testing.assert_allclose(weights[:2], [5 / 6, 2.5], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(weights[2:], 0)
        capped = weighted._class_weights(np.array([9999, 1] + [0] * 8, np.int64))
        self.assertEqual(capped[1], 4.)
        self.assertAlmostEqual(capped[0], 9996 / 9999, places=12)
        self.assertAlmostEqual(np.dot(capped, [9999, 1] + [0] * 8) / 10000, 1., places=12)
        self.assertTrue(np.isfinite(capped).all())
        for invalid in ([0] * 10, [-1] + [1] * 9, [1.] * 10, [1] * 9):
            with self.assertRaises(ValueError):
                weighted._class_weights(invalid)

    def test_minority_gradient_is_larger_without_unmasking_illegal_actions(self):
        logits = torch.full((2, 10), 100., requires_grad=True)
        with torch.no_grad():
            logits[:, :2] = 0
        masks = torch.zeros_like(logits, dtype=torch.bool)
        masks[:, :2] = True
        actions = torch.tensor([0, 1])
        weights = torch.tensor(weighted._class_weights(np.array([81, 9] + [0] * 8, np.int64)),
                               dtype=torch.float32)
        loss = weighted._weighted_masked_cross_entropy(logits, actions, masks, weights)
        self.assertAlmostEqual(loss.item(), np.log(2) * (5 / 6 + 2.5) / 2, places=6)
        loss.backward()
        self.assertAlmostEqual(abs(logits.grad[1, 1].item() / logits.grad[0, 0].item()), 3., places=6)
        torch.testing.assert_close(logits.grad[:, 2:], torch.zeros((2, 8)))
        with self.assertRaisesRegex(ValueError, 'legal'):
            weighted._weighted_masked_cross_entropy(logits, torch.tensor([0, 2]), masks, weights)

    def test_counts_use_training_episodes_only_and_validation_is_unweighted(self):
        report = self.train()
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(report['split'], self.parent_report['split'])
        self.assertEqual(report['split']['validation_episode_ids'], ['dataset0:2'])
        self.assertEqual(report['train_class_counts'], [14, 0, 2, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(report['class_weights'][9], 0)
        self.assertEqual(report['class_count_samples'], 16)
        self.assertEqual(report['supervised_samples'], 16)
        self.assertEqual(report['optimizer_updates'], 4)
        self.assertEqual(report['environment_interactions'], 0)
        model = weighted.load_weighted_checkpoint(report['checkpoints']['final']['path'])
        dataset = DemonstrationDataset(self.data)
        try:
            batch = dataset.batch([8, 9, 10, 11])
        finally:
            dataset.close()
        with torch.no_grad():
            distribution = model.policy.get_distribution(torch.from_numpy(batch['observations']),
                                                         action_masks=batch['action_masks'])
            ce = F.cross_entropy(distribution.distribution.logits,
                                 torch.from_numpy(batch['actions'])).item()
            accuracy = (distribution.get_actions(deterministic=True).numpy() == batch['actions']).mean()
        self.assertGreater(ce, 0)
        self.assertAlmostEqual(report['history'][-1]['validation_ce'], ce, places=6)
        self.assertEqual(report['history'][-1]['validation_accuracy'], accuracy)

    def test_full_checkpoint_has_weighted_provenance_and_unchanged_parent(self):
        report = self.train(epochs=2, learning_rate=2e-4)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(len(report['checkpoints']['epochs']), 2)
        checkpoint = Path(report['checkpoints']['final']['path'])
        manifest = json.loads(checkpoint.with_suffix('.manifest.json').read_text())
        self.assertIn('weighted', manifest['format'])
        self.assertIn('weighted', manifest['contract']['objective'])
        self.assertIn('learning_lab/weighted_imitation.py', manifest['sources'])
        self.assertEqual(manifest['parent_checkpoint']['sha256'], self.parent_hash)
        self.assertEqual(manifest['parent_checkpoint']['manifest_sha256'], self.parent_manifest_hash)
        self.assertEqual(manifest['train_class_counts'], report['train_class_counts'])
        self.assertEqual(manifest['class_weights'], report['class_weights'])
        self.assertFalse(manifest['exact_simulator_resume'])
        model = weighted.load_weighted_checkpoint(checkpoint)
        self.assertEqual(model.observation_space.shape, (94, 26, 26))
        self.assertEqual(model.action_space.n, 10)
        self.assertFalse(model.policy.normalize_images)
        self.assertEqual(model.policy.net_arch, {'pi': [64, 64], 'vf': [64, 64]})
        self.assertTrue(model.policy.optimizer.state_dict()['state'])
        self.assertTrue(all(group['lr'] == 2e-4 for group in model.policy.optimizer.param_groups))
        self.assertEqual(report['resource_monitor']['status'], 'enabled')
        self.assertTrue(Path(report['resource_monitor']['path']).is_file())
        self.assertEqual(model.num_timesteps, 0)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            base.load_imitation_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            weighted.load_weighted_checkpoint(self.parent)
        base.load_imitation_checkpoint(self.parent)
        self.assertFalse(torch.cuda.is_initialized())

    def test_loader_rejects_tampered_weights_sources_split_dataset_and_archive(self):
        report = self.train()
        path = Path(report['checkpoints']['final']['path'])
        sidecar = path.with_suffix('.manifest.json')
        original = json.loads(sidecar.read_text())
        for change in ('weights', 'counts', 'source', 'split', 'dataset', 'parent', 'objective'):
            with self.subTest(change=change):
                data = deepcopy(original)
                if change == 'weights':
                    data['class_weights'][0] = 4
                elif change == 'counts':
                    data['train_class_counts'][0] += 1
                elif change == 'source':
                    data['sources']['learning_lab/weighted_imitation.py'] = '0' * 64
                elif change == 'split':
                    data['split']['validation_episode_ids'] = data['split']['train_episode_ids']
                elif change == 'dataset':
                    data['datasets'][0]['sha256'] = '0' * 64
                elif change == 'parent':
                    del data['parent_checkpoint']['sha256']
                else:
                    data['contract']['objective'] = 'masked_teacher_cross_entropy'
                sidecar.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    weighted.load_weighted_checkpoint(path)
        sidecar.write_text(json.dumps(original))
        with path.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            weighted.load_weighted_checkpoint(path)

    def test_expired_deadline_does_not_count_labels_or_update_model(self):
        report = self.train(deadline=0)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['stop_reason'], 'deadline')
        self.assertEqual(report['class_count_samples'], 0)
        self.assertEqual(report['supervised_samples'], 0)
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertEqual(json.loads((self.output / 'report.json').read_text()), report)

    def test_epoch_deadline_after_one_update_keeps_correct_partial_counts(self):
        model = base.load_imitation_checkpoint(self.parent)
        corpus = base._Corpus([self.data.resolve()], self.output.resolve(), None)
        self.addCleanup(corpus.close)
        indices, _, _ = base._split_episodes(corpus.episode_ids, 0)
        weights = torch.tensor(weighted._class_weights(np.array([14, 0, 2] + [0] * 7, np.int64)),
                               dtype=torch.float32)
        progress = {'optimizer_updates': 0, 'supervised_samples': 0}

        def checkpoint_deadline(deadline):
            if progress['optimizer_updates'] == 1:
                raise base._DeadlineReached('Fixture deadline after first actual update')

        with self.assertRaises(base._DeadlineReached):
            weighted._train_epoch(model, corpus, indices, weights, 4, None, None, progress,
                                  deadline_check=checkpoint_deadline)
        self.assertEqual(progress, {'optimizer_updates': 1, 'supervised_samples': 4})

    def test_unverified_parent_fails_without_updates_and_output_is_fresh(self):
        fake = self.root / ('unverified-' + uuid.uuid4().hex + '.zip')
        shutil.copyfile(self.parent, fake)
        report = self.train(initial_checkpoint=fake)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('manifest', report['error'])
        self.assertEqual(report['optimizer_updates'], 0)
        with self.assertRaises(FileExistsError):
            self.train()


if __name__ == '__main__':
    unittest.main()
