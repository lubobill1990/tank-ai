"""CPU-only attention/checkpoint tests; no real environments or training runs."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch

from within_stage.test_entity_observation import fixture_ram
from within_stage import entity_policy


class EntityPolicyTests(unittest.TestCase):
    def setUp(self):
        from within_stage import entity_observation
        self.encoding = entity_observation
        torch.set_num_threads(1)
        self.root = Path('runs') / ('entity-policy-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def observations(self, batch=2):
        tokens = self.encoding.entities_from_ram(fixture_ram())
        return {'grid': torch.full((batch, 110, 26, 26), .1),
                'entities': torch.from_numpy(np.repeat(tokens[None], batch, axis=0))}

    def test_attention_is_permutation_invariant_and_padding_is_inert(self):
        model = entity_policy.make_entity_model(seed=1)
        extractor = model.policy.pi_features_extractor
        extractor.eval()
        observations = self.observations()
        permuted = {key: value.clone() for key, value in observations.items()}
        order = [0, 6, 4, 2, 1, 5, 3, *reversed(range(7, 17)), 17]
        permuted['entities'] = permuted['entities'][:, order]
        invalid = permuted['entities'][:, :, 0] == 0
        permuted['entities'][invalid] = float('nan')
        permuted['entities'][:, :, 0].masked_fill_(invalid, 0)
        with torch.no_grad():
            expected = extractor(observations)
            actual = extractor(permuted)
        self.assertEqual(expected.shape, (2, 256))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=2e-6)
        empty = {'grid': observations['grid'], 'entities': torch.zeros((2, 18, 16))}
        extractor(empty).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in extractor.parameters() if p.grad is not None))
        self.assertEqual(len(extractor.attention), 2)
        self.assertTrue(all(layer.self_attn.num_heads == 4 for layer in extractor.attention))

    def test_value_only_gradients_cannot_change_actor_encoder_or_actor_head(self):
        model = entity_policy.make_entity_model(seed=2, lr=1e-4)
        p = model.policy
        self.assertFalse(p.share_features_extractor)
        before_actor = entity_policy.state_sha256(p.pi_features_extractor.state_dict())
        before_critic = entity_policy.state_sha256(p.vf_features_extractor.state_dict())
        before_head = entity_policy.state_sha256(p.action_net.state_dict())
        p.optimizer.zero_grad(set_to_none=True)
        p.predict_values(self.observations()).sub(2).square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in p.pi_features_extractor.parameters()))
        p.optimizer.step()
        self.assertEqual(before_actor, entity_policy.state_sha256(p.pi_features_extractor.state_dict()))
        self.assertEqual(before_head, entity_policy.state_sha256(p.action_net.state_dict()))
        self.assertNotEqual(before_critic, entity_policy.state_sha256(p.vf_features_extractor.state_dict()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_model_is_fresh_spaces_only_and_masks_control_predict(self):
        env = entity_policy.SpaceOnlyEntityEnv()
        model = entity_policy.make_entity_model(env=env)
        self.assertFalse(model.policy.optimizer.state_dict()['state'])
        with self.assertRaises(RuntimeError):
            env.reset()
        numpy_obs = {key: value[0].numpy() for key, value in self.observations().items()}
        masks = np.zeros(10, bool); masks[8] = True
        action, _ = model.predict(numpy_obs, deterministic=True, action_masks=masks)
        self.assertEqual(int(action), 8)
        self.assertEqual(model.observation_space['grid'].shape, (110, 26, 26))
        self.assertEqual(model.observation_space['entities'].shape, (18, 16))

    def test_strict_checkpoint_roundtrip_owns_dict_schema_optimizer_and_dataset_config(self):
        model = entity_policy.make_entity_model(seed=3)
        model.policy.optimizer.zero_grad(set_to_none=True)
        model.policy.predict_values(self.observations()).sub(1).square().mean().backward()
        model.policy.optimizer.step()
        config = {'schema': self.encoding.entity_schema(), 'datasets': [], 'purpose': 'synthetic'}
        path = self.root / 'entity.zip'
        saved = entity_policy.save_entity_checkpoint(model, path, dataset_config=config)
        loaded = entity_policy.load_entity_checkpoint(path, expected_dataset_config=config)
        self.assertEqual(entity_policy.state_sha256(model.policy.state_dict()),
                         entity_policy.state_sha256(loaded.policy.state_dict()))
        self.assertEqual(entity_policy.state_sha256(model.policy.optimizer.state_dict()),
                         entity_policy.state_sha256(loaded.policy.optimizer.state_dict()))
        self.assertEqual(saved['sha256'], entity_policy.file_sha256(path))
        manifest_path = path.with_suffix('.manifest.json')
        original = json.loads(manifest_path.read_text())
        for field in ('schema', 'source', 'weights', 'dataset'):
            bad = deepcopy(original)
            if field == 'schema': bad['contract']['schema']['entities_shape'] = [17, 16]
            elif field == 'source': bad['sources'] = {}
            elif field == 'weights': bad['policy_sha256'] = '0' * 64
            else: bad['dataset_config']['purpose'] = 'changed'
            manifest_path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                entity_policy.load_entity_checkpoint(path)

    def test_serialized_attention_semantics_cannot_disagree_with_the_contract(self):
        model = entity_policy.make_entity_model()
        model.policy.pi_features_extractor.attention[0].self_attn.num_heads = 2
        with self.assertRaisesRegex(ValueError, 'attention|architecture'):
            entity_policy.save_entity_checkpoint(model, self.root / 'wrong-heads.zip', dataset_config={
                'schema': self.encoding.entity_schema(), 'datasets': []})
        self.assertFalse((self.root / 'wrong-heads.zip').exists())

    def test_bad_valid_flags_and_nonfinite_live_entities_fail_but_empty_padding_is_safe(self):
        model = entity_policy.make_entity_model()
        values = self.observations()
        values['entities'][0, 0, 0] = .5
        with self.assertRaises(ValueError):
            model.policy.pi_features_extractor(values)
        values = self.observations()
        values['entities'][0, 0, 2] = float('nan')
        with self.assertRaises(ValueError):
            model.policy.pi_features_extractor(values)

    def test_checkpoint_archive_corruption_and_dataset_schema_mismatch_are_rejected(self):
        model = entity_policy.make_entity_model()
        path = self.root / 'corrupt.zip'
        config = {'schema': self.encoding.entity_schema(), 'datasets': []}
        entity_policy.save_entity_checkpoint(model, path, dataset_config=config)
        with self.assertRaises(FileExistsError):
            entity_policy.save_entity_checkpoint(model, path, dataset_config=config)
        wrong = deepcopy(config); wrong['schema']['grid_shape'] = [94, 26, 26]
        with self.assertRaises(ValueError):
            entity_policy.load_entity_checkpoint(path, expected_dataset_config=wrong)
        with path.open('ab') as stream:
            stream.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            entity_policy.load_entity_checkpoint(path)

if __name__ == '__main__':
    unittest.main()
