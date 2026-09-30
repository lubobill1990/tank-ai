"""CPU synthetic entity BC tests; no opening/environment construction."""
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

from within_stage import imitation as precision_bc
from within_stage.demonstrations import PrecisionWriter
from within_stage.entity_observation import entities_from_ram
from within_stage.entity_policy import load_entity_checkpoint, make_entity_model, state_sha256, file_sha256
from within_stage.observation import OBSERVATION_VERSION
from within_stage.test_entity_observation import fixture_ram
from within_stage import entity_imitation as bc


class EntityImitationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path('runs') / ('entity-bc-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.pool = self.root / 'pool.json'
        self.pool.write_text(json.dumps({'specs': [
            {'config_index': i, 'family_id': i, 'spec': {'index': i}} for i in range(41)]}))
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps({
            'observation_version': OBSERVATION_VERSION, 'channels_per_frame': 55, 'actions': 10,
            'n_stack': 2, 'stage': 1, 'post_start_frames': 150, 'pool_path': str(self.pool.resolve()),
            'pool_sha256': file_sha256(self.pool),
            'train': {'family_ids': list(range(40)), 'all_config_indices': list(range(40))}}))
        self.split_hash = file_sha256(self.split)
        for key, value in (('FROZEN_SPLIT_PATH', self.split), ('FROZEN_SPLIT_SHA256', self.split_hash)):
            item = patch.object(precision_bc, key, value)
            item.start(); self.addCleanup(item.stop)

    def dataset(self, name='data', outside=False):
        path = self.root / name
        writer = PrecisionWriter(path)
        rows, plan = [], []
        try:
            for episode in range(5):
                family = 40 if outside and episode == 4 else episode
                origin = {'episode_id': episode, 'family_id': family, 'config_index': family,
                          'opening_spec': {'index': family}}
                plan.append(origin)
                for step in range(2):
                    ram = fixture_ram()
                    ram[0x92] = 90 + step
                    masks = np.zeros(10, bool); masks[[0, 2]] = True
                    writer.add(np.full((55, 26, 26), .1, np.float32), 2, masks,
                               ram=ram, frame=step * 4, episode_start=step == 0, executed_action=2)
                rows.append({**origin, 'stage': 1, 'outcome': 'timeout', 'decisions': 2,
                             'episode_finished': True})
            writer.finish(rows, {'selected_partition': 'train', 'split_sha256': self.split_hash, 'plan': plan})
        finally:
            writer.close()
        return path

    def test_entity_batch_keeps_exact_precision_split_and_uses_only_current_ram(self):
        paths = [self.dataset('a'), self.dataset('b')]
        contract = precision_bc._training_contract()
        reference = precision_bc.TrainingCorpus(paths, contract, None)
        corpus = bc.EntityTrainingCorpus(paths, contract, None)
        self.addCleanup(reference.close); self.addCleanup(corpus.close)
        np.testing.assert_array_equal(corpus.episode_ids, reference.episode_ids)
        first_split = precision_bc._split_episodes(reference.episode_ids, 0)
        second_split = precision_bc._split_episodes(corpus.episode_ids, 0)
        self.assertEqual(first_split[2], second_split[2])
        batch = corpus.batch(np.array([1, 0, 11]))
        self.assertEqual(set(batch), {'observations', 'actions', 'action_masks'})
        self.assertEqual(set(batch['observations']), {'grid', 'entities'})
        np.testing.assert_array_equal(batch['observations']['grid'],
                                      reference.batch(np.array([1, 0, 11]))['observations'])
        ram = fixture_ram(); ram[0x92] = 91
        np.testing.assert_array_equal(batch['observations']['entities'][0], entities_from_ram(ram))
        ram[0x92] = 90
        np.testing.assert_array_equal(batch['observations']['entities'][1], entities_from_ram(ram))

    def test_actor_only_training_keeps_fresh_critic_and_checkpoints_reload(self):
        path = self.dataset()
        initial = make_entity_model(seed=3, lr=1e-3)
        critic = precision_bc.critic_fingerprint(initial)
        report = bc.train_entity_imitation([path], self.root / 'model', epochs=2, batch_size=2,
                                           learning_rate=1e-3, seed=3)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        model = load_entity_checkpoint(report['checkpoints']['final']['path'])
        self.assertEqual(precision_bc.critic_fingerprint(model), critic)
        self.assertEqual(report['supervised_samples'], 16)
        self.assertEqual(report['optimizer_updates'], 8)
        self.assertEqual(report['validation_samples'], 6)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertIsNone(report['parent_checkpoint'])
        self.assertTrue(report['critic_unchanged'])
        self.assertLess(report['best_validation_ce'], report['initial_validation_ce'])
        self.assertEqual(model.entity_dataset_config['objective'], 'actor_only_masked_entity_bc')
        self.assertIn('not', model.entity_dataset_config['deployment_selection_note'])
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(report['deployment_selection_performed'])

    def test_outside_train_and_expired_deadline_never_train(self):
        outside = self.dataset(outside=True)
        report = bc.train_entity_imitation([outside], self.root / 'outside', epochs=1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertIn('TRAIN', report['error'])
        report = bc.train_entity_imitation([outside], self.root / 'expired', deadline=0)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['supervised_samples'], 0)

    def test_partial_deadline_saves_only_completed_actor_update_counts(self):
        path = self.dataset()
        clock = [100.]
        original = torch.optim.Adam.step
        def advance(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            clock[0] = 102.
            return result
        fake_time = type('Clock', (), {'monotonic': staticmethod(lambda: clock[0])})
        with patch.object(bc, 'time', fake_time), patch.object(precision_bc, 'time', fake_time), \
                patch.object(torch.optim.Adam, 'step', advance):
            report = bc.train_entity_imitation([path], self.root / 'partial',
                                               batch_size=2, deadline=101.)
        self.assertEqual(report['status'], 'partial', report.get('error'))
        self.assertEqual(report['optimizer_updates'], 1)
        self.assertEqual(report['supervised_samples'], 2)
        self.assertEqual(report['epochs_completed'], 0)
        self.assertIsNotNone(load_entity_checkpoint(report['checkpoints']['final']['path']))

    def test_audit_episode_mismatch_cannot_attach_other_episode_tokens(self):
        path = self.dataset()
        contract = precision_bc._training_contract()
        corpus = bc.EntityTrainingCorpus([path], contract, None)
        self.addCleanup(corpus.close)
        original = corpus.datasets[0].audit
        def wrong(ids):
            batch = original(ids)
            batch['episode_ids'][:] = 999
            return batch
        with patch.object(corpus.datasets[0], 'audit', wrong):
            with self.assertRaisesRegex(ValueError, 'RAM/episode'):
                corpus.batch(np.array([0, 1]))


if __name__ == '__main__':
    unittest.main()
