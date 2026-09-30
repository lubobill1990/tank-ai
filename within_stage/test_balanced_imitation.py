"""CPU fixture-only precision class-balanced BC tests."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from learning_lab import imitation as legacy_bc
from learning_lab.demonstrations import DemonstrationWriter
from within_stage import imitation as plain
from within_stage.demonstrations import PrecisionWriter
from within_stage.observation import OBSERVATION_VERSION
from within_stage.precision_policy import (make_precision_model, save_precision_checkpoint,
    load_precision_checkpoint, precision_schema, file_sha256, state_sha256, _json_hash)
from within_stage import balanced_imitation as balanced


class BalancedImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('balanced-precision-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True); cls.addClassCleanup(shutil.rmtree, cls.root)
        pool = cls.root / 'pool.json'
        pool.write_text(json.dumps({'specs': [
            {'config_index': i, 'family_id': i, 'spec': {'config': i}} for i in range(40)]}))
        cls.split = cls.root / 'split.json'
        cls.split.write_text(json.dumps({
            'observation_version': OBSERVATION_VERSION, 'channels_per_frame': 55, 'actions': 10,
            'n_stack': 2, 'stage': 1, 'post_start_frames': 150, 'pool_path': str(pool.resolve()),
            'pool_sha256': file_sha256(pool),
            'train': {'family_ids': list(range(40)), 'all_config_indices': list(range(40))}}))
        cls.data = cls.root / 'data'
        writer = PrecisionWriter(cls.data)
        plan, rows = [], []
        try:
            for episode in range(5):
                origin = {'episode_id': episode, 'family_id': episode, 'config_index': episode,
                          'opening_spec': {'config': episode}}
                plan.append(origin)
                for step in range(2):
                    label = 9 if episode == 2 else 2 if episode == 4 and step == 1 else 0
                    writer.add(np.full((55, 26, 26), .1, np.float32), label, np.ones(10, bool),
                               ram=np.zeros(2048, np.uint8), frame=step * 4,
                               episode_start=step == 0, executed_action=label)
                rows.append({**origin, 'stage': 1, 'outcome': 'clear', 'decisions': 2,
                             'episode_finished': True})
            writer.finish(rows, {'selected_partition': 'train', 'split_sha256': file_sha256(cls.split),
                                 'plan': plan})
        finally:
            writer.close()
        writer = DemonstrationWriter(cls.root / 'legacy-data')
        try:
            for i in range(3):
                writer.add(np.full((47, 26, 26), .1, np.float32), 0, np.ones(10, bool),
                           episode_start=True, executed_action=0)
            writer.finish([{'episode_id': i, 'outcome': 'timeout'} for i in range(3)], {})
        finally:
            writer.close()
        old = legacy_bc.train_imitation([cls.root / 'legacy-data'], cls.root / 'legacy', epochs=1, batch_size=2)
        if old['status'] != 'completed':
            raise AssertionError(old)
        model, migration = make_precision_model(old['checkpoints']['final']['path'])
        model.policy.optimizer.zero_grad(set_to_none=True)
        model.policy.predict_values(torch.full((1, 110, 26, 26), .1)).sub(1).square().mean().backward()
        model.policy.optimizer.step()
        cls.parent = cls.root / 'parent.zip'
        save_precision_checkpoint(model, cls.parent, migration_record=migration,
                                  dataset_config={'schema': precision_schema(), 'datasets': []})

    def setUp(self):
        for key, value in (('FROZEN_SPLIT_PATH', self.split), ('FROZEN_SPLIT_SHA256', file_sha256(self.split))):
            handle = patch.object(plain, key, value); handle.start(); self.addCleanup(handle.stop)

    def train(self, **kwargs):
        options = dict(epochs=1, batch=2, lr=1e-4, seed=0)
        options.update(kwargs)
        return balanced.train_balanced_precision(
            [self.data], self.parent, self.root / uuid.uuid4().hex, **options)

    def test_weighted_minority_gradient_masks_invalid_classes_and_absent_weights_are_zero(self):
        weights = balanced.class_weights(np.array([9, 1] + [0] * 8, np.int64))
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_array_equal(weights[2:], 0)
        logits = torch.zeros((2, 10), requires_grad=True)
        masks = torch.zeros((2, 10), dtype=torch.bool); masks[:, :2] = True
        loss = balanced.weighted_ce(logits, torch.tensor([0, 1]), masks, torch.tensor(weights).float())
        loss.backward()
        self.assertAlmostEqual(abs(logits.grad[1, 1].item() / logits.grad[0, 0].item()), 3., places=5)
        torch.testing.assert_close(logits.grad[:, 2:], torch.zeros((2, 8)))

    def test_same_split_train_only_weights_critic_parent_and_separate_counts(self):
        parent = load_precision_checkpoint(self.parent)
        report = self.train()
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(report['split']['validation_episode_ids'], ['dataset0:2'])
        self.assertEqual(report['train_class_counts'], [7, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(report['class_weights'][9], 0.)
        self.assertEqual(report['optimizer_updates'], 4)
        self.assertEqual(report['supervised_samples'], 8)
        self.assertEqual(report['environment_interactions'], 0)
        model = balanced.load_balanced_precision_checkpoint(report['checkpoints']['final']['path'])
        self.assertEqual(plain.critic_fingerprint(model), plain.critic_fingerprint(parent))
        self.assertEqual(model.precision_migration_record, parent.precision_migration_record)
        config = model.precision_dataset_config
        self.assertEqual(config['parent_checkpoint']['sha256'], file_sha256(self.parent))
        self.assertEqual(config['validation'], 'unweighted')
        self.assertTrue(config['offsets_enabled'])

    def test_strict_loader_recomputes_counts_not_merely_self_consistent_forged_weights(self):
        report = self.train()
        path = Path(report['checkpoints']['final']['path'])
        sidecar = path.with_suffix('.manifest.json')
        manifest = json.loads(sidecar.read_text())
        config = manifest['dataset_config']
        config['train_class_counts'] = [6, 0, 2] + [0] * 7
        config['class_weights'] = balanced.class_weights(np.array(config['train_class_counts'])).tolist()
        manifest['dataset_config_sha256'] = _json_hash(config)
        sidecar.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'counts|TRAIN'):
            balanced.load_balanced_precision_checkpoint(path)

    def test_interrupted_partial_adam_step_rolls_back_weights_optimizer_and_counts(self):
        model = load_precision_checkpoint(self.parent)
        model.policy.optimizer.param_groups[0]['lr'] = 1e-4
        original = torch.optim.Adam.step
        before = state_sha256({'policy': model.policy.state_dict(), 'adam': model.policy.optimizer.state_dict()})
        values = {'observations': np.full((2, 110, 26, 26), .1, np.float32),
                  'actions': np.array([0, 2], np.int64), 'action_masks': np.ones((2, 10), bool)}
        def fail(optimizer, *args, **kwargs):
            original(optimizer, *args, **kwargs)
            raise RuntimeError('synthetic interrupted optimizer')
        with patch.object(torch.optim.Adam, 'step', fail):
            with self.assertRaises(RuntimeError):
                balanced.atomic_actor_update(model, values, torch.ones(10), None)
        after = state_sha256({'policy': model.policy.state_dict(), 'adam': model.policy.optimizer.state_dict()})
        self.assertEqual(before, after)
        report = self.train(deadline=0)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['supervised_samples'], 0)

    def test_public_training_loop_remains_under_authorized_250_line_bound(self):
        self.assertLessEqual(len(inspect.getsource(balanced.train_balanced_precision).splitlines()), 250)

    def test_validation_remains_unweighted_even_for_absent_training_class(self):
        report = self.train()
        model = balanced.load_balanced_precision_checkpoint(report['checkpoints']['final']['path'])
        corpus = plain.TrainingCorpus([self.data], plain._training_contract(), None)
        self.addCleanup(corpus.close)
        values = corpus.batch(np.array([4, 5]))
        self.assertTrue(np.all(values['actions'] == 9))
        self.assertEqual(report['class_weights'][9], 0.)
        with torch.no_grad():
            expected, _ = plain._loss(model, values, True)
        self.assertGreater(expected.item(), 0.)
        self.assertAlmostEqual(report['history'][-1]['validation_ce'], expected.item(), places=6)

    def test_nonfinite_adam_buffer_rolls_back_even_if_policy_weights_stay_finite(self):
        model = load_precision_checkpoint(self.parent)
        values = {'observations': np.full((2, 110, 26, 26), .1, np.float32),
                  'actions': np.array([0, 2], np.int64), 'action_masks': np.ones((2, 10), bool)}
        before = state_sha256({'policy': model.policy.state_dict(), 'adam': model.policy.optimizer.state_dict()})
        original = torch.optim.Adam.step
        def corrupt(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            next(iter(optimizer.state.values()))['exp_avg'].fill_(float('inf'))
            return result
        with patch.object(torch.optim.Adam, 'step', corrupt):
            with self.assertRaises(FloatingPointError):
                balanced.atomic_actor_update(model, values, torch.ones(10), None)
        self.assertEqual(before, state_sha256({'policy': model.policy.state_dict(),
                                              'adam': model.policy.optimizer.state_dict()}))


if __name__ == '__main__':
    unittest.main()
