"""Synthetic TRAIN-family precision BC tests; never query actual opening families."""
from copy import deepcopy
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
from within_stage.demonstrations import PrecisionWriter
from within_stage.precision_policy import (
    make_precision_model, save_precision_checkpoint, load_precision_checkpoint,
    precision_schema, file_sha256, state_sha256,
)
from within_stage.observation import OBSERVATION_VERSION
from within_stage import imitation as bc


class PrecisionImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('precision-bc-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.frozen_hashes = {str(p): file_sha256(p) for p in (
            Path('within_stage') / 'precision_policy.py', Path('within_stage') / 'observation.py')}
        specs = [{'config_index': i, 'family_id': i,
                  'spec': {'stage': 1, 'label': f'synthetic-{i}', 'post_start_frames': 150}}
                 for i in range(41)]
        cls.pool = cls.root / 'synthetic-pool.json'
        cls.pool.write_text(json.dumps({'specs': specs}))
        cls.split = cls.root / 'synthetic-split.json'
        cls.split.write_text(json.dumps({
            'observation_version': OBSERVATION_VERSION, 'channels_per_frame': 55, 'actions': 10,
            'n_stack': 2, 'stage': 1, 'post_start_frames': 150,
            'pool_path': str(cls.pool.resolve()), 'pool_sha256': file_sha256(cls.pool),
            'train': {'family_ids': list(range(40)), 'all_config_indices': list(range(40)),
                      'representative_config_indices': list(range(40))}}))
        cls.split_hash = file_sha256(cls.split)
        old_writer = DemonstrationWriter(cls.root / 'legacy-data')
        try:
            for episode in range(3):
                for step in range(2):
                    old_writer.add(np.full((47, 26, 26), .1, np.float32), 0, np.ones(10, bool),
                                   episode_start=step == 0, executed_action=0)
            old_writer.finish([{'episode_id': i, 'outcome': 'timeout'} for i in range(3)],
                              {'label': 'synthetic'})
        finally:
            old_writer.close()
        previous = legacy_bc.train_imitation([cls.root / 'legacy-data'], cls.root / 'legacy-model',
                                             epochs=1, batch_size=2)
        if previous['status'] != 'completed':
            raise AssertionError(previous)
        cls.legacy_checkpoint = previous['checkpoints']['final']['path']
        model, migration = make_precision_model(cls.legacy_checkpoint, learning_rate=1e-4)
        model.policy.optimizer.zero_grad(set_to_none=True)
        (model.policy.predict_values(torch.full((1, 110, 26, 26), .1)) - 2).square().mean().backward()
        model.policy.optimizer.step()
        cls.initial = cls.root / 'initial.zip'
        save_precision_checkpoint(model, cls.initial, migration_record=migration,
                                  dataset_config={'schema': precision_schema(), 'datasets': [], 'split': None})
        cls.data = cls._dataset('training')
        def unchanged():
            assert cls.frozen_hashes == {p: file_sha256(p) for p in cls.frozen_hashes}
        cls.addClassCleanup(unchanged)

    @classmethod
    def _dataset(cls, name, *, outside=False):
        path = cls.root / name
        writer = PrecisionWriter(path)
        plan, reports = [], []
        try:
            for episode in range(5):
                family = 40 if outside and episode == 4 else episode
                config = {'episode_id': episode, 'config_index': family, 'family_id': family,
                          'opening_spec': {'stage': 1, 'label': f'synthetic-{family}', 'post_start_frames': 150}}
                plan.append(config)
                for step in range(2):
                    observation = np.full((55, 26, 26), .1, np.float32)
                    observation[47:] = .75
                    mask = np.zeros(10, bool); mask[[0, 2]] = True
                    writer.add(observation, 2, mask, ram=np.zeros(2048, np.uint8),
                               frame=step * 4, episode_start=step == 0, executed_action=2)
                reports.append({**config, 'stage': 1, 'outcome': 'timeout', 'decisions': 2,
                                'episode_finished': True})
            writer.finish(reports, {'role': 'TRAIN-only synthetic', 'selected_partition': 'train',
                                   'split_sha256': cls.split_hash, 'plan': plan,
                                   'development_or_final_evaluated': False})
        finally:
            writer.close()
        return path

    def setUp(self):
        self.path_patch = patch.object(bc, 'FROZEN_SPLIT_PATH', self.split)
        self.hash_patch = patch.object(bc, 'FROZEN_SPLIT_SHA256', self.split_hash)
        self.path_patch.start(); self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop); self.addCleanup(self.path_patch.stop)

    def train(self, **kwargs):
        config = {'epochs': 2, 'batch_size': 2, 'learning_rate': 1e-3, 'seed': 0}
        config.update(kwargs)
        data = config.pop('dataset_paths', [self.data])
        initial = config.pop('initial_checkpoint', self.initial)
        return bc.train_precision_imitation(data, initial, self.root / uuid.uuid4().hex, **config)

    def test_offsets_off_zeroes_both_frames_and_preserves_legacy_planes_owned(self):
        rng = np.random.default_rng(1)
        original = rng.uniform(-1, 1, (2, 110, 26, 26)).astype(np.float32)
        result = bc.adapt_observations(original, offsets_enabled=False)
        np.testing.assert_array_equal(result[:, :47], original[:, :47])
        np.testing.assert_array_equal(result[:, 55:102], original[:, 55:102])
        np.testing.assert_array_equal(result[:, 47:55], 0)
        np.testing.assert_array_equal(result[:, 102:], 0)
        self.assertFalse(np.shares_memory(result, original))
        np.testing.assert_array_equal(bc.adapt_observations(original, offsets_enabled=True), original)

    def test_train_updates_actor_only_and_critic_adam_state_remains_untouched(self):
        before = load_precision_checkpoint(self.initial)
        critic = bc.critic_fingerprint(before)
        actor = state_sha256(before.policy.pi_features_extractor.state_dict())
        report = self.train()
        self.assertEqual(report['status'], 'completed', report.get('error'))
        after = load_precision_checkpoint(report['checkpoints']['final']['path'])
        self.assertEqual(bc.critic_fingerprint(after), critic)
        self.assertNotEqual(state_sha256(after.policy.pi_features_extractor.state_dict()), actor)
        self.assertEqual(after.precision_migration_record, before.precision_migration_record)
        self.assertTrue(report['critic_unchanged'])
        self.assertEqual(report['optimizer_updates'], 8)
        self.assertEqual(report['supervised_samples'], 16)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertLess(report['best_validation_ce'], report['initial_validation_ce'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_off_ablation_is_persisted_in_every_saved_precision_dataset_contract(self):
        report = self.train(epochs=1, offsets_enabled=False)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        paths = report['checkpoints']['epochs'] + [report['checkpoints']['best'], report['checkpoints']['final']]
        for checkpoint in paths:
            model = load_precision_checkpoint(checkpoint['path'])
            config = model.precision_dataset_config
            self.assertFalse(config['offsets_enabled'])
            self.assertEqual(config['evaluation_adapter']['zero_channel_ranges'], [[47, 55], [102, 110]])
            self.assertTrue(config['evaluation_adapter']['required'])
            self.assertEqual(config['parent_checkpoint']['sha256'], file_sha256(self.initial))
        model = load_precision_checkpoint(report['checkpoints']['final']['path'])
        conv = model.policy.pi_features_extractor.cnn[0].weight
        self.assertEqual(torch.count_nonzero(conv[:, 47:55]).item(), 0)
        self.assertEqual(torch.count_nonzero(conv[:, 102:]).item(), 0)

    def test_namespaced_whole_episode_split_excludes_development_and_final_families(self):
        other = self._dataset('second-' + uuid.uuid4().hex)
        report = self.train(dataset_paths=[self.data, other], epochs=1)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        split = report['split']
        train, valid = set(split['train_episode_ids']), set(split['validation_episode_ids'])
        self.assertFalse(train & valid)
        self.assertEqual(len(train | valid), 10)
        self.assertTrue(all(name.startswith(('dataset0:', 'dataset1:')) for name in train | valid))
        self.assertEqual(split['train_samples'] + split['validation_samples'], 20)
        self.assertEqual(report['family_membership']['partition'], 'train')
        self.assertEqual(report['family_membership']['frozen_split_sha256'], self.split_hash)
        outside = self._dataset('outside-' + uuid.uuid4().hex, outside=True)
        bad = self.train(dataset_paths=[outside])
        self.assertEqual(bad['status'], 'failed')
        self.assertIn('TRAIN', bad['error'])
        self.assertEqual(bad['optimizer_updates'], 0)
        self.assertIsNone(bad['checkpoints']['final'])

    def test_masked_actor_loss_has_no_gradient_on_illegal_logits(self):
        logits = torch.tensor([[0., 0., 100.]], requires_grad=True)
        loss = bc.masked_ce(logits, torch.tensor([1]), torch.tensor([[True, True, False]]))
        loss.backward()
        self.assertGreater(abs(logits.grad[0, 1].item()), 0)
        self.assertEqual(logits.grad[0, 2].item(), 0)
        with self.assertRaises(ValueError):
            bc.masked_ce(logits, torch.tensor([2]), torch.tensor([[True, True, False]]))

    def test_deadlines_record_no_update_or_one_completed_update_explicitly(self):
        report = self.train(deadline=0)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['supervised_samples'], 0)
        clock = [100.]
        original = torch.optim.Adam.step
        def advance(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            clock[0] = 102.
            return result
        with patch.object(bc, 'time', type('Clock', (), {'monotonic': staticmethod(lambda: clock[0])})), \
                patch.object(torch.optim.Adam, 'step', advance):
            report = self.train(deadline=101.)
        self.assertEqual(report['status'], 'partial', report.get('error'))
        self.assertEqual(report['optimizer_updates'], 1)
        self.assertEqual(report['supervised_samples'], 2)
        self.assertEqual(report['epochs_completed'], 0)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertIsNotNone(load_precision_checkpoint(report['checkpoints']['final']['path']))

    def test_legacy_checkpoint_and_changed_frozen_split_are_rejected(self):
        report = self.train(initial_checkpoint=self.legacy_checkpoint)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['optimizer_updates'], 0)
        with patch.object(bc, 'FROZEN_SPLIT_SHA256', '0' * 64):
            report = self.train()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('split', report['error'])
        self.assertEqual(report['optimizer_updates'], 0)

    def test_conflicting_origin_reports_are_rejected_even_with_a_valid_dataset_seal(self):
        path = self._dataset('forged-origin-' + uuid.uuid4().hex)
        reports_path = path / 'episode_reports.json'
        reports = json.loads(reports_path.read_text())
        reports[3]['family_id'] = 40
        reports_path.write_text(json.dumps(reports))
        integrity_path = path / 'integrity.json'
        seal = json.loads(integrity_path.read_text())
        for entry in seal['files']:
            if entry['path'] == 'episode_reports.json':
                entry['bytes'] = reports_path.stat().st_size
                entry['sha256'] = file_sha256(reports_path)
        import hashlib
        seal['fingerprint_sha256'] = hashlib.sha256(json.dumps(seal['files'], sort_keys=True).encode()).hexdigest()
        integrity_path.write_text(json.dumps(seal))
        report = self.train(dataset_paths=[path])
        self.assertEqual(report['status'], 'failed')
        self.assertIn('TRAIN', report['error'])
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertIsNone(report['checkpoints']['final'])

    def test_all_train_and_validation_batches_use_the_selected_offset_adapter(self):
        adapted = []
        original = bc.adapt_observations
        def inspect_inputs(observation, *, offsets_enabled):
            result = original(observation, offsets_enabled=offsets_enabled)
            adapted.append((offsets_enabled, bool(np.any(result[:, 47:55])),
                            bool(np.any(result[:, 102:110]))))
            return result
        with patch.object(bc, 'adapt_observations', inspect_inputs):
            report = self.train(epochs=1, offsets_enabled=False)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(len(adapted), 6)  # Initial+final validation and four training minibatches.
        self.assertTrue(all(item == (False, False, False) for item in adapted))

    def test_precision_warm_start_records_immediate_parent_without_changing_migration(self):
        first = self.train(epochs=1)
        parent = first['checkpoints']['final']['path']
        second = self.train(initial_checkpoint=parent, epochs=1, learning_rate=2e-4, seed=9)
        self.assertEqual(second['status'], 'completed', second.get('error'))
        self.assertEqual(second['parent_checkpoint']['sha256'], file_sha256(parent))
        loaded = load_precision_checkpoint(second['checkpoints']['final']['path'])
        before = load_precision_checkpoint(self.initial)
        self.assertEqual(loaded.precision_migration_record, before.precision_migration_record)
        self.assertTrue(all(group['lr'] == 2e-4 for group in loaded.policy.optimizer.param_groups))
        self.assertEqual(critic := bc.critic_fingerprint(loaded), bc.critic_fingerprint(before))
        self.assertEqual(critic, second['initial_critic'])


if __name__ == '__main__':
    unittest.main()
