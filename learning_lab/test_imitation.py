"""CPU-only imitation contracts; fixture data never step ROM/native environments."""
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import sys
import types
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from learning_lab import imitation


class FixtureDataset:
    fixtures = {}

    def __init__(self, path):
        spec = self.fixtures[str(Path(path).resolve())]
        self.episode_ids = np.asarray(spec.get('episodes', [0, 0, 1, 1, 2, 2]), np.int64)
        self.metadata = {'format': 'synthetic-cpu-fixture', 'teacher': 'not-a-learned-policy'}
        self.label = spec.get('label', 2)
        self.illegal = spec.get('illegal', False)
        self.fail_at = spec.get('fail_at')
        self.calls = 0

    def __len__(self):
        return len(self.episode_ids)

    def batch(self, indices):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError('fixture read failed')
        indices = np.asarray(indices, dtype=np.int64)
        obs = np.zeros((len(indices), 94, 26, 26), np.float32)
        obs[:, 4, 8:16, 8:16] = 1
        masks = np.zeros((len(indices), 10), bool)
        masks[:, [0, 2]] = True
        if self.illegal:
            masks[:, self.label] = False
        return {'observations': obs, 'actions': np.full(len(indices), self.label, np.int64),
                'action_masks': masks, 'episode_ids': self.episode_ids[indices].copy()}

    def close(self):
        pass


class ImitationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('imitation-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.dataset_module = types.ModuleType('learning_lab.demonstrations')
        self.dataset_module.DemonstrationDataset = FixtureDataset
        module_name = 'learning_lab.demonstrations'
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = self.dataset_module

        def restore_dataset_module():
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

        self.addCleanup(restore_dataset_module)
        torch.set_num_threads(1)

    def dataset(self, name='data', **spec):
        path = self.root / name
        path.mkdir()
        (path / 'fixture.json').write_text(json.dumps(spec), encoding='utf-8')
        FixtureDataset.fixtures[str(path.resolve())] = spec
        self.addCleanup(FixtureDataset.fixtures.pop, str(path.resolve()))
        return path

    def train(self, paths=None, output='trained', **kwargs):
        paths = [self.dataset()] if paths is None else paths
        config = {'epochs': 2, 'batch_size': 2, 'learning_rate': 1e-3, 'seed': 7}
        config.update(kwargs)
        return imitation.train_imitation(paths, self.root / output, **config)

    def test_masked_cross_entropy_excludes_illegal_logits_and_gradients(self):
        logits = torch.tensor([[0., 1., 100.]], requires_grad=True)
        loss = imitation._masked_cross_entropy(
            logits, torch.tensor([1]), torch.tensor([[True, True, False]]))
        self.assertAlmostEqual(loss.item(), 0.3132616875, places=6)
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0)
        for action, mask in ((2, [True, True, False]), (0, [False, False, False])):
            with self.assertRaisesRegex(ValueError, 'legal|mask'):
                imitation._masked_cross_entropy(
                    logits, torch.tensor([action]), torch.tensor([mask]))

    def test_episode_split_is_disjoint_and_reproducible_across_dataset_namespaces(self):
        paths = [self.dataset('a', episodes=[0, 0, 1, 1]),
                 self.dataset('b', episodes=[0, 0, 1, 1])]
        report = self.train(paths, epochs=1)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        split = report['split']
        train, validation = set(split['train_episode_ids']), set(split['validation_episode_ids'])
        self.assertFalse(train & validation)
        self.assertEqual(train | validation,
                         {'dataset0:0', 'dataset0:1', 'dataset1:0', 'dataset1:1'})
        self.assertEqual(split['train_samples'] + split['validation_samples'], 8)
        self.assertEqual(report['supervised_samples'], split['train_samples'])
        self.assertEqual(report['environment_interactions'], 0)
        self.assertIn('development', report['warning'])
        self.assertIn('full-game', report['warning'])
        a = imitation._split_episodes(np.array(['a', 'a', 'b', 'c', 'c']), 123)
        b = imitation._split_episodes(np.array(['a', 'a', 'b', 'c', 'c']), 123)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_training_reduces_loss_and_exports_full_loadable_policy_optimizer(self):
        report = self.train(epochs=3)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(report['epochs_completed'], 3)
        self.assertLess(report['best_validation_ce'], report['initial_validation_ce'])
        self.assertEqual(report['optimizer_updates'], 6)
        self.assertEqual(report['supervised_samples'], 12)
        self.assertEqual(len(report['checkpoints']['epochs']), 3)
        path = Path(report['checkpoints']['final']['path'])
        model = imitation.load_imitation_checkpoint(path)
        self.assertEqual(model.observation_space.shape, (94, 26, 26))
        self.assertEqual(model.action_space.n, 10)
        self.assertFalse(model.policy.normalize_images)
        self.assertEqual(model.policy.net_arch, {'pi': [64, 64], 'vf': [64, 64]})
        self.assertEqual(model.num_timesteps, 0)
        self.assertEqual(report['actual_device'], 'cpu')
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(model.policy.optimizer.state_dict()['state'])
        batch = FixtureDataset(self.root / 'data').batch([0, 1])
        actions, _ = model.predict(batch['observations'], action_masks=batch['action_masks'],
                                   deterministic=True)
        np.testing.assert_array_equal(actions, [2, 2])
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        self.assertEqual(manifest['checkpoint_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest['training_config']['epochs'], 3)
        self.assertEqual(manifest['actual_device'], 'cpu')
        self.assertEqual(manifest['datasets'][0]['metadata']['format'], 'synthetic-cpu-fixture')
        self.assertTrue(manifest['sources']['learning_lab/imitation.py'])
        from sb3_contrib import MaskablePPO
        direct = MaskablePPO.load(path, device='cpu')
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, direct.policy.state_dict()[name], rtol=0, atol=0)

    def test_checkpoint_requires_own_manifest_and_rejects_corruption_or_contract_changes(self):
        report = self.train(epochs=1)
        path = Path(report['checkpoints']['final']['path'])
        sidecar = path.with_suffix('.manifest.json')
        manifest = sidecar.read_text()
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, 'manifest'):
            imitation.load_imitation_checkpoint(path)
        sidecar.write_text(manifest)
        for field, value in (('format', 'legacy'), ('checkpoint_sha256', '0' * 64),
                             ('contract', {'observation_shape': [47, 26, 26]})):
            changed = json.loads(manifest)
            changed[field] = value
            sidecar.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                imitation.load_imitation_checkpoint(path)
        sidecar.write_text(manifest)
        model = imitation.load_imitation_checkpoint(path)
        model.policy_kwargs['normalize_images'] = True
        model.save(path)
        changed = json.loads(manifest)
        changed['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'architecture contract'):
            imitation.load_imitation_checkpoint(path)
        with path.open('ab') as stream:
            stream.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA|sha|hash'):
            imitation.load_imitation_checkpoint(path)

    def test_loader_checks_dataset_and_split_provenance_not_just_zip_hash(self):
        report = self.train(epochs=1)
        path = Path(report['checkpoints']['final']['path'])
        sidecar = path.with_suffix('.manifest.json')
        original = json.loads(sidecar.read_text())
        for change in ('dataset_hash', 'overlapping_split', 'source_hash', 'root_type'):
            with self.subTest(change=change):
                manifest = json.loads(json.dumps(original))
                if change == 'dataset_hash':
                    manifest['datasets'][0]['sha256'] = '0' * 64
                elif change == 'overlapping_split':
                    manifest['split']['validation_episode_ids'] = manifest['split']['train_episode_ids']
                elif change == 'source_hash':
                    manifest['sources']['policy.py'] = '0' * 64
                else:
                    manifest = []
                sidecar.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    imitation.load_imitation_checkpoint(path)

    def test_validation_rejects_wrong_shapes_dtypes_nonfinite_or_mismatched_episode_ids(self):
        data = self.dataset()
        fixture = FixtureDataset(data)
        for field, replacement in (
            ('observations', np.zeros((2, 47, 26, 26), np.float32)),
            ('observations', np.full((2, 94, 26, 26), np.nan, np.float32)),
            ('actions', np.array([2, 2], np.int32)),
            ('action_masks', np.ones((2, 10), np.uint8)),
            ('episode_ids', np.array([999, 999], np.int64)),
        ):
            with self.subTest(field=field, dtype=replacement.dtype):
                batch = fixture.batch([0, 1])
                batch[field] = replacement
                with self.assertRaises(ValueError):
                    imitation._validate_batch(batch, np.array([0, 0], np.int64))

    def test_cleanup_errors_are_persisted_as_explicit_failed_reports(self):
        with patch.object(FixtureDataset, 'close', side_effect=OSError('fixture close failed')):
            report = self.train(epochs=1)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('fixture close failed', report['cleanup_error'])
        self.assertEqual(json.loads((self.root / 'trained' / 'report.json').read_text()), report)
        self.assertEqual(report['optimizer_updates'], 2)

    def test_warm_start_records_parent_hash_and_new_optimizer_learning_rate(self):
        data = self.dataset()
        first = self.train([data], output='first', epochs=1)
        parent = Path(first['checkpoints']['final']['path'])
        second = self.train([data], output='second', epochs=1, learning_rate=2e-4, seed=99,
                            initial_checkpoint=parent)
        self.assertEqual(second['status'], 'completed', second.get('error'))
        self.assertEqual(second['parent_checkpoint']['sha256'],
                         first['checkpoints']['final']['sha256'])
        self.assertEqual(second['optimizer_updates'], 2)
        model = imitation.load_imitation_checkpoint(second['checkpoints']['final']['path'])
        self.assertTrue(all(group['lr'] == 2e-4 for group in model.policy.optimizer.param_groups))
        manifest = json.loads(Path(second['checkpoints']['final']['path'])
                              .with_suffix('.manifest.json').read_text())
        self.assertEqual(manifest['training_config']['seed'], 99)
        self.assertFalse(manifest['exact_simulator_resume'])

    def test_deadline_before_start_is_explicit_and_does_no_updates(self):
        report = self.train(deadline=0)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['stop_reason'], 'deadline')
        self.assertEqual(report['supervised_samples'], 0)
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertEqual(json.loads((self.root / 'trained' / 'report.json').read_text()), report)

    def test_deadline_after_first_update_preserves_partial_model_and_accounting(self):
        clock = [100.]
        original_step = torch.optim.Adam.step

        def advance_after_update(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            clock[0] = 102.
            return result

        with patch.object(imitation, 'time', types.SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(torch.optim.Adam, 'step', advance_after_update):
            report = self.train(deadline=101.)
        self.assertEqual(report['status'], 'partial', report.get('error'))
        self.assertEqual(report['stop_reason'], 'deadline')
        self.assertEqual(report['epochs_completed'], 0)
        self.assertEqual(report['optimizer_updates'], 1)
        self.assertEqual(report['supervised_samples'], 2)
        self.assertFalse(report['checkpoints']['epochs'])
        self.assertIsNotNone(imitation.load_imitation_checkpoint(
            report['checkpoints']['final']['path']))

    def test_illegal_labels_fail_explicitly_without_any_optimizer_update(self):
        report = self.train([self.dataset(illegal=True)])
        self.assertEqual(report['status'], 'failed')
        self.assertIn('legal', report['error'])
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['environment_interactions'], 0)

    def test_read_error_after_update_keeps_partial_accounting_and_final_checkpoint(self):
        report = self.train([self.dataset(fail_at=3)])
        self.assertEqual(report['status'], 'failed')
        self.assertIn('fixture read failed', report['error'])
        self.assertEqual(report['optimizer_updates'], 1)
        self.assertEqual(report['supervised_samples'], 2)
        self.assertTrue(Path(report['checkpoints']['final']['path']).is_file())

    def test_insufficient_episodes_and_duplicate_paths_cannot_leak_across_split(self):
        one = self.dataset('one', episodes=[3, 3])
        report = self.train([one], output='one-output')
        self.assertEqual(report['status'], 'failed')
        self.assertIn('episode', report['error'])
        data = self.dataset('duplicate')
        duplicate = self.train([data, data], output='duplicate-output')
        self.assertEqual(duplicate['status'], 'failed')
        self.assertIn('duplicate', duplicate['error'].lower())

    def test_output_directory_is_never_reused(self):
        data = self.dataset()
        output = self.root / 'occupied'
        output.mkdir()
        (output / 'keep.txt').write_text('do not overwrite')
        with self.assertRaises(FileExistsError):
            self.train([data], output='occupied')
        self.assertEqual((output / 'keep.txt').read_text(), 'do not overwrite')

    def test_resource_pressure_is_partial_and_records_telemetry_without_cuda_initialization(self):
        with patch('rust_resources.host_snapshot', return_value={'commit_available_bytes': 0}), \
                patch.object(torch.cuda, 'init', side_effect=AssertionError('GPU logging initialization')):
            report = self.train()
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['stop_reason'], 'resource_pressure')
        self.assertEqual(report['resource_monitor']['status'], 'enabled')
        self.assertTrue(Path(report['resource_monitor']['path']).is_file())
        self.assertEqual(report['optimizer_updates'], 0)

    def test_cli_reports_expired_deadline_as_partial_exit_code(self):
        data = self.dataset()
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = imitation.main(['--datasets', str(data), '--output', str(self.root / 'cli'),
                                     '--epochs', '1', '--batch-size', '2', '--deadline', '0'])
        report = json.loads(stream.getvalue())
        self.assertEqual(result, 2)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['environment_interactions'], 0)


@unittest.skipUnless(os.environ.get('TANK_AI_IMITATION_DATASET_TESTS') == '1',
                     'Set TANK_AI_IMITATION_DATASET_TESTS=1 once demonstrations.py is ready')
class DatasetFormatAcceptanceTests(unittest.TestCase):
    def test_real_compact_writer_reader_feed_offline_training_without_any_environment(self):
        from learning_lab.demonstrations import DemonstrationDataset, DemonstrationWriter
        root = Path('runs') / ('imitation-format-test-' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root)
        corpus = root / 'corpus'
        writer = DemonstrationWriter(corpus)
        self.addCleanup(writer.close)
        frame = np.zeros((47, 26, 26), np.float32)
        frame[4, 8:16, 8:16] = 1
        masks = np.zeros(10, bool)
        masks[[0, 2]] = True
        for _ in range(3):
            writer.add(frame, 2, masks, episode_start=True, executed_action=2)
            writer.add(frame, 2, masks, episode_start=False, executed_action=2)
        writer.finish([{'episode_id': i, 'outcome': 'timeout', 'decisions': 2}
                       for i in range(3)], {'label': 'synthetic-format-only'})
        dataset = DemonstrationDataset(corpus)
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), 6)
        report = imitation.train_imitation([corpus], root / 'model', epochs=1, batch_size=2)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(report['optimizer_updates'], 2)
        self.assertEqual(report['environment_interactions'], 0)
        imitation.load_imitation_checkpoint(root / 'model')


if __name__ == '__main__':
    unittest.main()
