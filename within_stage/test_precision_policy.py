"""CPU-only 94->110 migration proofs; spaces-only envs never step/reset."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch
from gymnasium import spaces

from learning_lab import imitation
from learning_lab.demonstrations import DemonstrationWriter
from within_stage import precision_policy as precision


class PrecisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('precision-policy-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.frozen_paths = [
            Path('policy.py'), Path('rust_models.py'), Path('battle_city_env.py'),
            Path('within_stage') / 'observation.py', Path('learning_lab') / 'imitation.py']
        cls.old_hashes = {str(p): precision.file_sha256(p) for p in cls.frozen_paths}
        writer = DemonstrationWriter(cls.root / 'legacy-data')
        try:
            for _ in range(3):
                for index in range(2):
                    writer.add(np.full((47, 26, 26), .1, np.float32), 2, np.ones(10, bool),
                               episode_start=index == 0, executed_action=2)
            writer.finish([{'episode_id': i, 'outcome': 'timeout'} for i in range(3)],
                          {'label': 'synthetic-migration-fixture'})
        finally:
            writer.close()
        report = imitation.train_imitation([cls.root / 'legacy-data'], cls.root / 'legacy',
                                           epochs=1, batch_size=2)
        if report['status'] != 'completed':
            raise AssertionError(report)
        cls.legacy = Path(report['checkpoints']['final']['path'])
        def unchanged():
            assert cls.old_hashes == {str(p): precision.file_sha256(p) for p in cls.frozen_paths}
        cls.addClassCleanup(unchanged)

    def make(self):
        model, record = precision.make_precision_model(self.legacy, device='cpu', learning_rate=1e-4, seed=7)
        return model, record

    @staticmethod
    def dataset_config():
        return {'schema': precision.precision_schema(), 'datasets': [],
                'purpose': 'migration-fixture-no-environment', 'split': None}

    def test_frame_aware_mapping_zero_offsets_and_all_other_weights_are_copied(self):
        old = imitation.load_imitation_checkpoint(self.legacy)
        new, record = self.make()
        before, after = old.policy.state_dict(), new.policy.state_dict()
        for prefix in ('features_extractor.', 'pi_features_extractor.', 'vf_features_extractor.'):
            conv = after[prefix + 'cnn.0.weight']
            original = before['features_extractor.cnn.0.weight']
            self.assertEqual(tuple(conv.shape), (32, 110, 3, 3))
            torch.testing.assert_close(conv[:, :47], original[:, :47], rtol=0, atol=0)
            torch.testing.assert_close(conv[:, 55:102], original[:, 47:94], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(conv[:, 47:55]).item(), 0)
            self.assertEqual(torch.count_nonzero(conv[:, 102:110]).item(), 0)
        for name, value in after.items():
            if name.endswith('cnn.0.weight'):
                continue
            source = ('features_extractor.' + name.split('.', 1)[1]
                      if name.startswith(('features_extractor.', 'pi_features_extractor.', 'vf_features_extractor.'))
                      else name)
            torch.testing.assert_close(value, before[source], rtol=0, atol=0)
        self.assertTrue(old.policy.optimizer.state_dict()['state'])
        self.assertFalse(new.policy.optimizer.state_dict()['state'])
        self.assertTrue(record['fresh_adam'])
        self.assertFalse(record['legacy_optimizer_resumed'])
        self.assertFalse(record['exact_simulator_resume'])
        self.assertEqual(record['parent']['sha256'], precision.file_sha256(self.legacy))

    def test_actor_logits_and_actions_ignore_arbitrary_new_offsets_at_initialization(self):
        old = imitation.load_imitation_checkpoint(self.legacy)
        new, _ = self.make()
        rng = np.random.default_rng(819)
        legacy = rng.uniform(-1, 1, (4, 94, 26, 26)).astype(np.float32)
        upgraded = rng.uniform(-20, 20, (4, 110, 26, 26)).astype(np.float32)
        upgraded[:, :47] = legacy[:, :47]
        upgraded[:, 55:102] = legacy[:, 47:]
        masks = rng.random((4, 10)) > .4
        masks[:, 0] = True
        with torch.no_grad():
            previous = precision.actor_logits(old, torch.from_numpy(legacy))
            current = precision.actor_logits(new, torch.from_numpy(upgraded))
        torch.testing.assert_close(previous, current, rtol=1e-5, atol=2e-6)
        a, _ = old.predict(legacy, deterministic=True, action_masks=masks)
        b, _ = new.predict(upgraded, deterministic=True, action_masks=masks)
        np.testing.assert_array_equal(a, b)
        upgraded[:, 47:55] = rng.uniform(-1000, 1000, upgraded[:, 47:55].shape)
        upgraded[:, 102:] = rng.uniform(-1000, 1000, upgraded[:, 102:].shape)
        with torch.no_grad():
            alternate = precision.actor_logits(new, torch.from_numpy(upgraded))
        torch.testing.assert_close(current, alternate, rtol=0, atol=0)

    def test_critic_only_gradient_cannot_change_actor_encoder_or_action_head(self):
        model, _ = self.make()
        actor = model.policy.pi_features_extractor
        critic = model.policy.vf_features_extractor
        self.assertIsNot(actor, critic)
        self.assertIs(model.policy.features_extractor, actor)
        self.assertFalse({p.data_ptr() for p in actor.parameters()} &
                         {p.data_ptr() for p in critic.parameters()})
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, critic.state_dict()[name], rtol=0, atol=0)
        before_actor = precision.state_sha256(actor.state_dict())
        before_head = precision.state_sha256(model.policy.action_net.state_dict())
        before_critic = precision.state_sha256(critic.state_dict())
        obs = torch.full((2, 110, 26, 26), .2)
        model.policy.optimizer.zero_grad(set_to_none=True)
        (model.policy.predict_values(obs) - 3).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in critic.parameters()))
        model.policy.optimizer.step()
        self.assertEqual(before_actor, precision.state_sha256(actor.state_dict()))
        self.assertEqual(before_head, precision.state_sha256(model.policy.action_net.state_dict()))
        self.assertNotEqual(before_critic, precision.state_sha256(critic.state_dict()))

    def test_factory_validates_env_spaces_without_reset_or_step(self):
        env = precision.SpaceOnlyEnv()
        model, _ = precision.make_precision_model(self.legacy, env=env)
        self.assertEqual(model.observation_space.shape, (110, 26, 26))
        with self.assertRaises(RuntimeError):
            env.reset()
        for shape, actions in (((94, 26, 26), 10), ((110, 26, 26), 15)):
            invalid = precision.SpaceOnlyEnv()
            invalid.observation_space = spaces.Box(-1, 1, shape, np.float32)
            invalid.action_space = spaces.Discrete(actions)
            with self.assertRaises(ValueError):
                precision.make_precision_model(self.legacy, env=invalid)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_legacy_encoder_sharing_and_activation_are_rejected(self):
        model = imitation.load_imitation_checkpoint(self.legacy)
        model.policy.share_features_extractor = False
        with self.assertRaises(ValueError):
            precision.validate_legacy_model(model)
        model.policy.share_features_extractor = True
        model.policy.activation_fn = torch.nn.ReLU
        with self.assertRaises(ValueError):
            precision.validate_legacy_model(model)

    def test_precision_roundtrip_preserves_new_adam_and_rejects_wrong_contracts(self):
        model, migration = self.make()
        model.policy.optimizer.zero_grad(set_to_none=True)
        model.policy.predict_values(torch.full((1, 110, 26, 26), .1)).square().mean().backward()
        model.policy.optimizer.step()
        config = self.dataset_config()
        path = self.root / 'precision.zip'
        saved = precision.save_precision_checkpoint(model, path, migration_record=migration,
                                                     dataset_config=config)
        loaded = precision.load_precision_checkpoint(path, expected_dataset_config=config)
        self.assertEqual(precision.state_sha256(model.policy.state_dict()),
                         precision.state_sha256(loaded.policy.state_dict()))
        self.assertEqual(precision.state_sha256(model.policy.optimizer.state_dict()),
                         precision.state_sha256(loaded.policy.optimizer.state_dict()))
        self.assertFalse(loaded.policy.share_features_extractor)
        self.assertIsNot(loaded.policy.pi_features_extractor, loaded.policy.vf_features_extractor)
        self.assertEqual(saved['sha256'], precision.file_sha256(path))
        with self.assertRaises(ValueError):
            imitation.load_imitation_checkpoint(path)
        wrong = deepcopy(config); wrong['schema']['frame_shape'] = [47, 26, 26]
        with self.assertRaises(ValueError):
            precision.load_precision_checkpoint(path, expected_dataset_config=wrong)
        sidecar = path.with_suffix('.manifest.json')
        original = json.loads(sidecar.read_text())
        for change in ('schema', 'source', 'parent', 'weights', 'config'):
            manifest = deepcopy(original)
            if change == 'schema': manifest['contract']['schema']['frame_shape'] = [47, 26, 26]
            elif change == 'source': manifest['sources']['within_stage\\observation.py'] = '0' * 64
            elif change == 'parent': manifest['migration']['parent']['sha256'] = 'bad'
            elif change == 'weights': manifest['policy_sha256'] = '0' * 64
            else: manifest['dataset_config']['purpose'] = 'silently changed'
            sidecar.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                precision.load_precision_checkpoint(path)

    def test_saving_requires_explicit_precision_dataset_schema_and_original_migration(self):
        model, record = self.make()
        config = self.dataset_config()
        wrong = deepcopy(config); wrong['schema']['observation_shape'] = [94, 26, 26]
        with self.assertRaises(ValueError):
            precision.save_precision_checkpoint(model, self.root / 'wrong.zip',
                                                 migration_record=record, dataset_config=wrong)
        wrong_record = deepcopy(record); wrong_record['fresh_adam'] = False
        with self.assertRaises(ValueError):
            precision.save_precision_checkpoint(model, self.root / 'wrong-parent.zip',
                                                 migration_record=wrong_record, dataset_config=config)
        self.assertFalse((self.root / 'wrong.zip').exists())

    def test_checkpoint_rejects_archive_corruption_and_saving_over_an_existing_artifact(self):
        model, record = self.make()
        path = self.root / 'corruption.zip'
        config = self.dataset_config()
        precision.save_precision_checkpoint(model, path, migration_record=record, dataset_config=config)
        with self.assertRaises(FileExistsError):
            precision.save_precision_checkpoint(model, path, migration_record=record, dataset_config=config)
        with path.open('ab') as stream:
            stream.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            precision.load_precision_checkpoint(path)

    def test_precision_loader_rejects_shared_actor_critic_configuration_in_serialized_model(self):
        model, record = self.make()
        path = self.root / 'architecture.zip'
        precision.save_precision_checkpoint(model, path, migration_record=record,
                                             dataset_config=self.dataset_config())
        model.policy_kwargs['share_features_extractor'] = True
        model.save(path)
        sidecar = path.with_suffix('.manifest.json')
        manifest = json.loads(sidecar.read_text())
        manifest['checkpoint_sha256'] = precision.file_sha256(path)
        sidecar.write_text(json.dumps(manifest))
        with self.assertRaises((ValueError, RuntimeError)):
            precision.load_precision_checkpoint(path)

    def test_dataset_config_rejects_legacy_metadata_and_source_model_config_mutation(self):
        model, record = self.make()
        config = self.dataset_config()
        config['datasets'] = [{'path': 'fixture', 'sha256': 'a' * 64,
                               'metadata': {'frame_shape': [47, 26, 26]}}]
        with self.assertRaises(ValueError):
            precision.save_precision_checkpoint(model, self.root / 'legacy-dataset.zip',
                                                 migration_record=record, dataset_config=config)
        config = self.dataset_config()
        model.policy_kwargs['share_features_extractor'] = True
        with self.assertRaises(ValueError):
            precision.save_precision_checkpoint(model, self.root / 'bad-config.zip',
                                                 migration_record=record, dataset_config=config)


if __name__ == '__main__':
    unittest.main()
