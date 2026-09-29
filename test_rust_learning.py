"""CPU-only fixtures for bounded learning; no native batch or GPU is started."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import rust_learning as learning
# Import consumers before patching their source module so they retain real native bindings.
import rust_env
import rust_train
import rust_validation
import rust_resources
from test_rust_resources import healthy_host


def suite(wins=0, skill=0.0):
    modes = {mode: {'summary': {'wins': wins, 'episodes': 4, 'win_rate': wins / 4}}
             for mode in ('deterministic', 'sampled')}
    return {'rust': {'modes': copy.deepcopy(modes)}, 'rom': {'modes': modes},
            'skills': {'modes': {mode: {'summary': {'mean_success_rate': skill}}
                                 for mode in ('deterministic', 'sampled')}}}


def fixture_vec(*args, **kwargs):
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv
    reward = kwargs.get('_fixture_reward', 0.01)

    class FixtureEnv(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (47, 26, 26), np.float32)
        action_space = gym.spaces.Discrete(10)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, np.float32), {}

        def step(self, action):
            return np.zeros(self.observation_space.shape, np.float32), reward, False, False, {}

        def action_masks(self):
            return np.ones(10, dtype=bool)

    return DummyVecEnv([FixtureEnv, FixtureEnv])


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parent / ('.learning-test-' + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)
        self.addCleanup(self.assert_native_imports_restored)
        self.enterContext(patch.object(rust_resources, 'host_snapshot', side_effect=healthy_host))

    def config(self, **kwargs):
        return learning.TrialConfig(output=str(self.directory / 'trial'), steps=16,
                                    envs=2, n_steps=2, batch_size=4, epochs=1,
                                    stages=(1, 2), checkpoint_every=4, eval_every=8,
                                    device='cpu', codec='zlib', **kwargs)

    def assert_native_imports_restored(self):
        self.assertIs(rust_validation.RustVectorEnv, rust_env.RustVectorEnv)
        self.assertIs(rust_train.RustVectorEnv, rust_env.RustVectorEnv)

    def test_approved_defaults_and_six_fresh_seeded_trials(self):
        args = learning.parse_args(['study', '--output', str(self.directory / 'study')])
        configs = learning.trial_configs(args)
        self.assertEqual([(c.model, c.seed) for c in configs],
                         [(m, s) for m in ('cnn', 'spatial', 'history') for s in (0, 1)])
        self.assertEqual(sum(c.steps for c in configs), 1572864)
        self.assertTrue(all(c.envs == 32 and c.codec == 'cuda' for c in configs))
        self.assertEqual(len({c.output for c in configs}), 6)
        self.assertTrue(all('load' not in c.to_dict() for c in configs))

    def test_rejects_overshoot_overlap_duplicate_and_unapproved_budget(self):
        valid = self.config()
        for change in ({'steps': 17}, {'eval_every': 7}, {'checkpoint_every': 3},
                       {'stages': (1, 4)}, {'confirm_stages': (5, 6)},
                       {'eval_seeds': (101, 101)}, {'steps': 262148},
                       {'max_storage_mib': 2049}, {'max_wall_seconds': 3601}, {'patience': 4}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                learning.TrialConfig(**{**valid.to_dict(), **change}).validate()
        with self.assertRaises(SystemExit):
            learning.parse_args(['study', '--output', 'unused', '--parallel', '3'])

    def test_history_memory_uses_all_sixteen_frames(self):
        basic = learning.memory_preflight(self.config(model='cnn'))
        history = learning.memory_preflight(self.config(model='history'))
        self.assertEqual(history['observation_shape'], [752, 26, 26])
        self.assertEqual(history['dense_observation_bytes'], basic['dense_observation_bytes'] * 8)
        self.assertEqual(history['channels_per_frame'], 47)
        self.assertEqual(history['max_storage_bytes'], 2 * 1024**3)

    def test_patience_counts_only_completed_after_initial_rom_selection(self):
        gate = learning.StopGate(patience=3)
        self.assertTrue(gate.observe(suite()))
        self.assertFalse(gate.observe(None, completed=False))
        for _ in range(2):
            self.assertFalse(gate.observe(suite()))
            self.assertFalse(gate.stopped)
        self.assertFalse(gate.observe(suite()))
        self.assertTrue(gate.stopped)
        self.assertEqual(gate.completed_checks, 4)
        better_rust = suite()
        better_rust['rust']['modes']['deterministic']['summary']['wins'] = 99
        self.assertEqual(learning.selection_score(better_rust), (0, 0.0))
        self.assertGreater(learning.selection_score(suite(1, 0)), learning.selection_score(suite(0, 1)))

    def test_monitor_immutable_snapshots_survive_locked_pointer_and_deduplicate(self):
        writer = learning.StatusWriter(self.directory, 'batch')
        with patch.object(learning.os, 'replace', side_effect=PermissionError('sharing lock')), \
                self.assertWarnsRegex(RuntimeWarning, r'authoritative snapshot.*batch-'):
            writer.write({'phase': 'training', 'step': 4})
            writer.write({'phase': 'training', 'step': 4})
            writer.write({'phase': 'training', 'step': 8})
        self.assertEqual(learning.read_status(self.directory, 'batch')['step'], 8)
        self.assertEqual(len(list(self.directory.glob('batch-*.json'))), 2)
        self.assertFalse(list(self.directory.glob('*.pending')))

    def test_status_and_result_writes_do_not_materialize_whole_json_strings(self):
        payload = {'episodes': [{'step': i, 'kills': i % 3} for i in range(2000)]}
        with patch.object(learning.json, 'dumps', side_effect=MemoryError('whole-document encoding')):
            writer = learning.StatusWriter(self.directory)
            writer.write(payload)
            writer.write(payload)
            learning._write_new(self.directory / 'result.json', payload)
        self.assertEqual(learning.read_status(self.directory), payload)
        self.assertEqual(json.loads((self.directory / 'result.json').read_text()), payload)
        self.assertEqual(len(list(self.directory.glob('status-*.json'))), 1)

    def test_loop_failure_preserves_traceback_without_saving_partial_weights(self):
        runtime = FixtureRuntime()

        def failed_update(deadline):
            runtime.collected_steps = 2
            raise MemoryError('allocation failed in rollout')

        runtime.update = failed_update
        report = learning.run_training_loop(
            runtime, self.config(), evaluate=lambda *a, **k: suite(),
            save=lambda role, step: {'path': role, 'step': step}, publish=lambda _: None)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['optimized_steps'], 0)
        self.assertEqual(report['collected_steps'], 2)
        self.assertIn('failed_update', report.get('traceback', ''))
        self.assertNotIn('final', report)

    def test_resource_stop_is_incomplete_and_keeps_last_checkpoint_not_partial_weights(self):
        from rust_resources import ResourcePressureError
        runtime = FixtureRuntime()

        def failed_update(deadline):
            runtime.collected_steps = 2
            raise ResourcePressureError('host commit reserve reached')

        runtime.update = failed_update
        report = learning.run_training_loop(
            runtime, self.config(), evaluate=lambda *a, **k: suite(),
            save=lambda role, step: {'path': role, 'step': step}, publish=lambda _: None)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['stop_reason'], 'resource_pressure')
        self.assertEqual(report['collected_steps'], 2)
        self.assertEqual(report['optimized_steps'], 0)
        self.assertEqual(report['latest']['step'], 0)
        self.assertNotIn('final', report)
        self.assertFalse(report['learning_effect_valid'])

    def test_low_memory_trial_does_not_allocate_model_and_writes_failure_evidence(self):
        host = healthy_host()
        host['commit_available_bytes'] = 1024**3
        config = self.config()
        with patch.object(rust_resources, 'host_snapshot', return_value=host), \
                patch.object(learning, 'TrainingRuntime', side_effect=AssertionError('allocated model')):
            report = learning.run_trial(config, {'stages': [1, 2]})
        self.assertEqual(report['stop_reason'], 'resource_pressure')
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['collected_steps'], 0)
        failure = json.loads((Path(config.output) / 'failure.json').read_text())
        self.assertIn('ResourcePressureError', failure['traceback'])
        self.assertEqual(failure['resources']['commit_available_bytes'], 1024**3)
        self.assertFalse(list(Path(config.output).glob('*.zip')))

    def test_pressure_during_real_rollout_keeps_collected_distinct_from_optimized(self):
        import numpy as np
        original_check = rust_resources.ResourceMonitor.check

        def check(monitor, phase, **kwargs):
            if phase == 'rollout':
                raise rust_resources.ResourcePressureError('fixture host pressure')
            return original_check(monitor, phase, **kwargs)

        config = self.config()
        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec), \
                patch.object(learning, 'verify_frozen', return_value=np.zeros((2, 2048), np.uint8)), \
                patch('rust_validation.evaluate_suite', return_value=suite()), \
                patch.object(rust_resources.ResourceMonitor, 'check', check):
            report = learning.run_trial(config, {'stages': [1, 2]})
        self.assertEqual(report['stop_reason'], 'resource_pressure')
        self.assertEqual(report['collected_steps'], 2)
        self.assertEqual(report['optimized_steps'], 0)
        self.assertEqual(report['latest']['step'], 0)
        self.assertNotIn('final', report)
        self.assertTrue((Path(config.output) / 'failure.json').is_file())

    def test_result_markdown_also_streams_instead_of_materializing_nested_json(self):
        path = self.directory / 'RESULTS.md'
        with patch.object(learning.json, 'dumps', side_effect=MemoryError('whole-document encoding')):
            learning.write_results(path, [{'status': 'failed', 'error': 'fixture',
                                          'progress': {'samples': list(range(1000))}}])
        self.assertIn('Error: fixture', path.read_text())
        self.assertIn('999', path.read_text())

    def test_trial_monitor_failure_does_not_reload_large_json_under_memory_pressure(self):
        import numpy as np

        class FailedOnceWriter(learning.StatusWriter):
            failed = False

            def write(self, status):
                if status.get('phase') == 'finished' and not self.failed:
                    self.failed = True
                    raise MemoryError('final monitor allocation')
                return super().write(status)

        config = self.config()
        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec), \
                patch.object(learning, 'verify_frozen', return_value=np.zeros((2, 2048), np.uint8)), \
                patch('rust_validation.evaluate_suite', return_value=suite()), \
                patch.object(learning, 'StatusWriter', FailedOnceWriter), \
                patch.object(learning, 'read_status', side_effect=AssertionError('reloaded large JSON')):
            report = learning.run_trial(config, {'stages': [1, 2]})
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['optimized_steps'], 16)
        self.assertIn('MemoryError', report['traceback'])
        self.assertTrue((Path(config.output) / 'failure.json').is_file())

    def test_supervisor_memory_error_has_small_durable_failure_and_cleans_children(self):
        record = {'path': '0', 'output': str(self.directory / 'trial'), 'model': 'cnn', 'seed': 0}
        manifest = {'trials': [record], 'provenance': {}, 'total_step_cap': 16}
        original_popen = subprocess.Popen
        processes = []

        def spawn(command, **kwargs):
            if len(command) < 2 or command[1] != str(learning.ROOT / 'rust_train.py'):
                return original_popen(command, **kwargs)
            process = original_popen([sys.executable, '-c', 'import time;time.sleep(120)'], **kwargs)
            processes.append(process)
            return process

        class FailedWriter(learning.StatusWriter):
            def write(self, status):
                if self.name == 'batch' and status['status'] == 'running':
                    raise MemoryError('monitor serialization failed')
                return super().write(status)

        args = SimpleNamespace(output=self.directory, parallel=1, max_wall_seconds=30)
        with patch.object(learning, 'prepare_study', return_value=manifest), \
                patch.object(learning, 'verify_frozen'), \
                patch.object(learning.subprocess, 'Popen', side_effect=spawn), \
                patch.object(learning, 'StatusWriter', FailedWriter):
            with self.assertRaises(MemoryError):
                learning.study(args)
        self.assertTrue(all(p.poll() is not None for p in processes))
        path = self.directory / 'supervisor-failure.json'
        self.assertTrue(path.is_file(), 'failure evidence must survive the failed normal monitor')
        report = json.loads(path.read_text())
        self.assertIn('MemoryError', report['traceback'])
        self.assertFalse(report['automatic_retry'])
        self.assertFalse(report['exposure_exact'])
        self.assertLess(path.stat().st_size, 16384)

    def test_snapshot_is_not_published_until_payload_is_fsynced(self):
        original_fsync = learning.os.fsync
        observed = []

        def fsync(descriptor):
            observed.append(list(self.directory.glob('status-*.json')))
            original_fsync(descriptor)

        with patch.object(learning.os, 'fsync', side_effect=fsync):
            learning.StatusWriter(self.directory).write({'step': 4})
        self.assertEqual(observed[0], [])
        self.assertEqual(len(observed[1]), 1)
        self.assertEqual(learning.read_status(self.directory)['step'], 4)
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_critical_snapshot_write_error_propagates_without_publishing_partial_json(self):
        with patch.object(learning.os, 'fsync', side_effect=OSError('disk failure')), \
                self.assertRaisesRegex(OSError, 'disk failure'):
            learning.StatusWriter(self.directory).write({'step': 4})
        self.assertFalse(list(self.directory.glob('status-*.json')))
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_pointer_non_permission_io_failure_propagates(self):
        with patch.object(learning.os, 'replace', side_effect=OSError('disk full')), \
                self.assertRaisesRegex(OSError, 'disk full'):
            learning.StatusWriter(self.directory).write({'step': 4})
        self.assertEqual(learning.read_status(self.directory)['step'], 4)

    def test_newest_published_corruption_never_falls_back_to_stale_status(self):
        writer = learning.StatusWriter(self.directory)
        writer.write({'step': 4})
        latest = writer.write({'step': 8})
        latest.write_text('{incomplete')
        with self.assertRaises(json.JSONDecodeError):
            learning.read_status(self.directory)

    def test_newest_snapshot_access_failure_is_not_hidden_by_older_snapshot(self):
        writer = learning.StatusWriter(self.directory)
        writer.write({'step': 4})
        latest = writer.write({'step': 8})
        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == latest:
                raise PermissionError('latest snapshot inaccessible')
            return original_read(path, *args, **kwargs)

        with patch.object(Path, 'read_text', read_text), \
                self.assertRaisesRegex(PermissionError, 'latest snapshot inaccessible'):
            learning.read_status(self.directory)

    def test_snapshot_publish_failure_propagates_and_removes_unpublished_file(self):
        with patch.object(learning.os, 'rename', side_effect=OSError('publish disk failure')), \
                self.assertRaisesRegex(OSError, 'publish disk failure'):
            learning.StatusWriter(self.directory).write({'step': 4})
        self.assertFalse(list(self.directory.iterdir()))

    def test_pointer_cleanup_permission_failure_is_visible(self):
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path.suffix == '.pending':
                raise PermissionError('cleanup sharing lock')
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', unlink), \
                self.assertWarnsRegex(RuntimeWarning, 'cleanup.*pending'):
            learning.StatusWriter(self.directory).write({'step': 4})
        self.assertEqual(learning.read_status(self.directory)['step'], 4)

    def test_non_permission_cleanup_failure_propagates(self):
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path.suffix == '.pending':
                raise OSError('cleanup disk failure')
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', unlink), \
                self.assertRaisesRegex(OSError, 'cleanup disk failure'):
            learning.StatusWriter(self.directory).write({'step': 4})
        self.assertEqual(learning.read_status(self.directory)['step'], 4)

    def test_status_newest_does_not_regress_when_wall_clock_moves_backwards(self):
        writer = learning.StatusWriter(self.directory)
        with patch.object(learning.time, 'time_ns', side_effect=[100, 50]):
            writer.write({'step': 4})
            writer.write({'step': 8})
        self.assertEqual(learning.read_status(self.directory)['step'], 8)

    def test_initial_periodic_and_final_saves_are_after_optimizer(self):
        runtime = FixtureRuntime()
        events = []

        def save(role, step):
            self.assertEqual(step, runtime.optimized_steps)
            events.append(('save', role, step))
            return {'path': role, 'step': step}

        def evaluate(step, confirmation=False):
            events.append(('confirm' if confirmation else 'eval', step))
            return suite(step // 8, step / 32)

        report = learning.run_training_loop(runtime, self.config(), evaluate=evaluate,
                                            save=save, publish=lambda _: None)
        self.assertEqual([e for e in events if e[0] == 'eval'], [('eval', 0), ('eval', 8), ('eval', 16)])
        self.assertEqual([e[2] for e in events if e[:2] == ('save', 'latest')], [0, 4, 8, 12, 16])
        self.assertLess(events.index(('save', 'latest', 8)), events.index(('eval', 8)))
        self.assertEqual(report['optimized_steps'], 16)
        self.assertEqual(report['stop_reason'], 'step_cap')
        self.assertTrue(report['learning_effect_valid'])
        self.assertEqual(report['confirmation']['step'], 16)

    def test_partial_validation_is_explicit_and_never_reuses_previous_metrics(self):
        runtime = FixtureRuntime()
        snapshots = []

        def evaluate(step, confirmation=False):
            if step:
                raise TimeoutError('validation deadline')
            return suite()

        report = learning.run_training_loop(runtime, self.config(), evaluate=evaluate,
                                            save=lambda role, step: {'path': role, 'step': step},
                                            publish=lambda s: snapshots.append(copy.deepcopy(s)))
        self.assertEqual(report['stop_reason'], 'wall_cap')
        self.assertEqual(report['evaluations'][-1]['status'], 'incomplete')
        self.assertNotIn('metrics', report['evaluations'][-1])
        self.assertEqual(report['completed_checks'], 1)
        self.assertFalse(report['learning_effect_valid'])
        self.assertEqual(report['optimized_steps'], 8)

    def test_partial_rollout_does_not_publish_unoptimized_checkpoint(self):
        runtime = FixtureRuntime(partial=True)
        saves = []
        report = learning.run_training_loop(
            runtime, self.config(), evaluate=lambda *a, **k: suite(),
            save=lambda role, step: saves.append((role, step)) or {'path': role, 'step': step},
            publish=lambda _: None)
        self.assertEqual(report['collected_steps'], 2)
        self.assertEqual(report['optimized_steps'], 0)
        self.assertTrue(all(step == 0 for _, step in saves))
        self.assertEqual(report['stop_reason'], 'wall_cap')

    def test_nonfinite_metric_stops_explicitly(self):
        runtime = FixtureRuntime()
        report = learning.run_training_loop(
            runtime, self.config(), evaluate=lambda *a, **k: suite(skill=float('nan')),
            save=lambda role, step: {'path': role, 'step': step}, publish=lambda _: None)
        self.assertEqual(report['stop_reason'], 'nonfinite')
        self.assertEqual(report['optimized_steps'], 0)
        self.assertFalse(report['learning_effect_valid'])

    def test_manifest_rejects_legacy_forgery_and_mismatched_history(self):
        config = self.config()
        manifest = learning.checkpoint_manifest(
            config, provenance={'fixture': True}, checkpoint_sha256='a' * 64,
            step=4, optimizer_updates=1)
        learning.validate_learning_manifest(manifest, verify_runtime=False)
        for change in ({'rust_learning_schema_version': 0},
                       {'contract': {**manifest['contract'], 'actions': 15}},
                       {'contract': {**manifest['contract'], 'n_stack': 16}},
                       {'checkpoint_sha256': 'not-a-hash'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                learning.validate_learning_manifest({**manifest, **change}, verify_runtime=False)

    def test_real_cpu_optimizer_checkpoint_is_small_evaluation_only_and_fresh(self):
        import numpy as np
        import torch
        config = self.config()
        Path(config.output).mkdir()
        initial = np.zeros((2, 2048), np.uint8)
        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec), \
                patch.object(learning, 'verify_frozen', return_value=initial):
            runtime = learning.TrainingRuntime(config, initial, time.monotonic() + 120)
            try:
                initial_state = {name: value.clone() for name, value in runtime.model.policy.state_dict().items()}
                self.assertTrue(runtime.update(time.monotonic() + 120))
                record = learning._save_checkpoint(runtime, config, {}, 'final', 4)
                loaded = learning.load_learning_checkpoint(record['path'])
                self.assertEqual(loaded.n_steps, 2)
                self.assertEqual(loaded.n_envs, 1)
                self.assertEqual(loaded.num_timesteps, 4)
                self.assertEqual(loaded._n_updates, 1)
                self.assertTrue(any(not torch.equal(value, initial_state[name])
                                    for name, value in loaded.policy.state_dict().items()))
                for name, value in runtime.model.policy.state_dict().items():
                    torch.testing.assert_close(value, loaded.policy.state_dict()[name])
                with self.assertRaises(FileExistsError):
                    learning._save_checkpoint(runtime, config, {}, 'final', 4)
                path = Path(record['path'])
                with path.open('ab') as stream:
                    stream.write(b'tampered')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    learning.load_learning_checkpoint(path)
            finally:
                runtime.close()
            fresh = learning.TrainingRuntime(config, initial, time.monotonic() + 120)
            try:
                self.assertEqual(fresh.model.num_timesteps, 0)
                for name, value in fresh.model.policy.state_dict().items():
                    torch.testing.assert_close(value, initial_state[name])
            finally:
                fresh.close()

    def test_constant_returns_undefined_explained_variance_is_not_training_failure(self):
        import numpy as np
        config = self.config()
        with patch('rust_env.RustVectorEnv', side_effect=lambda *a, **k: fixture_vec(_fixture_reward=0)):
            runtime = learning.TrainingRuntime(config, np.zeros((2, 2048), np.uint8), time.monotonic() + 120)
            try:
                self.assertTrue(runtime.update(time.monotonic() + 120))
                np.testing.assert_array_equal(runtime.model.rollout_buffer.returns, 0)
                diagnostics = runtime.diagnostics()
                update = diagnostics['optimizer_metrics'][-1]
                self.assertIsNone(update['train/explained_variance'])
                self.assertIn('constant returns', update['undefined_diagnostics']['train/explained_variance'])
                self.assertEqual(diagnostics['undefined_diagnostic_counts']['train/explained_variance'], 1)
                self.assertEqual(runtime.optimized_steps, 4)
                json.dumps(diagnostics, allow_nan=False)
            finally:
                runtime.close()

    def test_actual_nan_loss_stops_optimizer_despite_undefined_diagnostic_allowance(self):
        import numpy as np
        config = self.config()
        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec):
            runtime = learning.TrainingRuntime(config, np.zeros((2, 2048), np.uint8), time.monotonic() + 120)
            train = runtime.model.train

            def bad_train():
                train()
                runtime.model.logger.record('train/loss', float('nan'))
                runtime.model.logger.record('train/explained_variance', float('nan'))

            try:
                with patch.object(runtime.model, 'train', side_effect=bad_train), \
                        self.assertRaises(FloatingPointError):
                    runtime.update(time.monotonic() + 120)
                self.assertEqual(runtime.optimized_steps, 0)
                self.assertEqual(runtime.updates, [])
            finally:
                runtime.close()

    def test_nonfinite_adam_state_is_rejected_even_with_finite_policy_and_losses(self):
        import numpy as np
        import torch
        config = self.config()
        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec):
            runtime = learning.TrainingRuntime(config, np.zeros((2, 2048), np.uint8), time.monotonic() + 120)
            train = runtime.model.train

            def bad_train():
                train()
                state = next(iter(runtime.model.policy.optimizer.state.values()))
                state['exp_avg'].fill_(float('inf'))

            try:
                with patch.object(runtime.model, 'train', side_effect=bad_train), \
                        self.assertRaisesRegex(FloatingPointError, 'optimizer'):
                    runtime.update(time.monotonic() + 120)
                self.assertTrue(all(torch.isfinite(value).all()
                                    for value in runtime.model.policy.state_dict().values()))
                self.assertEqual(runtime.optimized_steps, 0)
            finally:
                runtime.close()

    def test_history_trial_uses_formal_frame_limit_and_selected_weights_for_confirmation(self):
        import numpy as np
        config = self.config(model='history', max_frames=4)
        calls = []

        def evaluate(model, **kwargs):
            self.assertEqual(model.observation_space.shape, (752, 26, 26))
            self.assertEqual(kwargs['max_frames'], 12000)
            calls.append((tuple(kwargs['stages']), model.num_timesteps))
            return suite()

        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec), \
                patch.object(learning, 'verify_frozen', return_value=np.zeros((2, 2048), np.uint8)), \
                patch('rust_validation.evaluate_suite', side_effect=evaluate):
            report = learning.run_trial(config, {'stages': [1, 2]})
        self.assert_native_imports_restored()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(calls, [((4, 5), 0), ((4, 5), 8), ((4, 5), 16), ((6, 7), 0)])
        result = json.loads((Path(config.output) / 'result.json').read_text())
        manifest = json.loads((Path(config.output) / 'result.manifest.json').read_text())
        self.assertTrue(result['learning_effect_valid'])
        self.assertEqual(manifest['result_sha256'], learning._hash(Path(config.output) / 'result.json'))
        self.assertEqual(learning.read_status(config.output)['optimized_steps'], 16)
        self.assertEqual(result['best']['step'], 0)
        self.assertEqual(result['final']['step'], 16)
        self.assertIn('rust', result['progress'])
        self.assertIn('rom', result['progress'])

    def test_cleanup_failure_keeps_durable_evidence_and_marks_trial_failed(self):
        import numpy as np
        config = learning.TrialConfig(**{**self.config().to_dict(), 'steps': 4})
        original_close = learning.TrainingRuntime.close

        def failing_close(runtime):
            original_close(runtime)
            raise RuntimeError('cleanup failed')

        with patch('rust_env.RustVectorEnv', side_effect=fixture_vec), \
                patch.object(learning, 'verify_frozen', return_value=np.zeros((2, 2048), np.uint8)), \
                patch('rust_validation.evaluate_suite', return_value=suite()), \
                patch.object(learning.TrainingRuntime, 'close', failing_close):
            report = learning.run_trial(config, {'stages': [1, 2]})
        self.assert_native_imports_restored()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['optimized_steps'], 4)
        self.assertIn('cleanup failed', report['error'])
        self.assertFalse(report['learning_effect_valid'])
        self.assertEqual(len(report['evaluations']), 2)
        self.assertTrue((Path(config.output) / 'result.manifest.json').exists())

    def test_checkpoint_rejects_serialized_fifteen_action_policy(self):
        import gymnasium as gym
        import numpy as np
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from rust_models import extractor_class

        class FifteenActionEnv(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (94, 26, 26), np.float32)
            action_space = gym.spaces.Discrete(15)

        model = MaskablePPO('CnnPolicy', DummyVecEnv([FifteenActionEnv]), n_steps=2, batch_size=2,
                            device='cpu', policy_kwargs={'features_extractor_class': extractor_class('cnn'),
                                                        'normalize_images': False})
        path = self.directory / 'forged.zip'
        model.save(path)
        manifest = learning.checkpoint_manifest(self.config(), provenance={}, checkpoint_sha256=learning._hash(path),
                                                step=0, optimizer_updates=0)
        learning._write_new(str(path) + '.manifest.json', manifest)
        with patch.object(learning, 'verify_frozen'), self.assertRaisesRegex(ValueError, 'Serialized checkpoint'):
            learning.load_learning_checkpoint(path)
        model.env.close()

    def test_wall_before_initial_validation_still_has_current_partial_result(self):
        runtime = FixtureRuntime()
        report = learning.run_training_loop(
            runtime, self.config(), evaluate=lambda *a, **k: self.fail('deadline must prevent evaluation'),
            save=lambda role, step: {'path': role, 'step': step}, publish=lambda _: None,
            clock=lambda: 3601, started=0)
        self.assertEqual(report['evaluations'], [])
        self.assertEqual(report['stop_reason'], 'wall_cap')
        self.assertEqual(report['optimized_steps'], 0)

    def test_validation_timeout_during_confirmation_does_not_claim_completed_study(self):
        def evaluate(step, confirmation=False):
            if confirmation:
                raise TimeoutError('confirmation timed out')
            return suite()

        report = learning.run_training_loop(
            FixtureRuntime(), self.config(), evaluate=evaluate,
            save=lambda role, step: {'path': role, 'step': step}, publish=lambda _: None)
        self.assertEqual(report['confirmation']['status'], 'incomplete')
        self.assertNotIn('metrics', report['confirmation'])
        self.assertFalse(report['learning_effect_valid'])
        self.assertEqual(report['stop_reason'], 'wall_cap')

    def test_checkpoint_failure_is_explicit_not_successfully_completed(self):
        def save(role, step):
            if step:
                raise OSError('disk full')
            return {'path': role, 'step': step}

        report = learning.run_training_loop(
            FixtureRuntime(), self.config(), evaluate=lambda *a, **k: suite(),
            save=save, publish=lambda _: None)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stop_reason'], 'error')
        self.assertIn('disk full', report['error'])

    def test_final_flush_deadline_is_labeled_wall_cap(self):
        clock = [0.0]

        def save(role, step):
            if role == 'final':
                clock[0] = 3601
            return {'path': role, 'step': step}

        report = learning.run_training_loop(
            FixtureRuntime(), self.config(), evaluate=lambda *a, **k: suite(),
            save=save, publish=lambda _: None, clock=lambda: clock[0])
        self.assertEqual(report['stop_reason'], 'wall_cap')
        self.assertEqual(report['status'], 'incomplete')

    def test_patience_stops_after_three_unimproved_completed_checks(self):
        config = self.config()
        config = learning.TrialConfig(**{**config.to_dict(), 'steps': 32})
        runtime = FixtureRuntime()
        report = learning.run_training_loop(
            runtime, config, evaluate=lambda *a, **k: suite(),
            save=lambda role, step: {'path': role, 'step': step}, publish=lambda _: None)
        self.assertEqual(report['optimized_steps'], 24)
        self.assertEqual(report['stop_reason'], 'patience')
        self.assertEqual(report['completed_checks'], 4)
        self.assertEqual(report['best']['step'], 0)
        self.assertEqual(report['confirmation']['step'], 0)

    def test_interrupted_study_has_durable_last_known_partial_result(self):
        record = {'path': '0', 'output': str(self.directory / 'trial'),
                  'model': 'cnn', 'seed': 0}
        manifest = {'trials': [record], 'provenance': {}, 'total_step_cap': 16}
        original_popen = subprocess.Popen
        processes = []

        def spawn(command, **kwargs):
            if len(command) < 2 or command[1] != str(learning.ROOT / 'rust_train.py'):
                return original_popen(command, **kwargs)
            process = original_popen([sys.executable, '-c', 'import time;time.sleep(120)'], **kwargs)
            processes.append(process)
            return process

        args = SimpleNamespace(output=self.directory, parallel=1, max_wall_seconds=30)
        with patch.object(learning, 'prepare_study', return_value=manifest), \
                patch.object(learning, 'verify_frozen'), patch.object(learning.subprocess, 'Popen', side_effect=spawn), \
                patch.object(learning.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                learning.study(args)
        self.assertTrue(all(p.poll() is not None for p in processes))
        self.assertEqual(learning.read_status(self.directory, 'batch')['status'], 'interrupted')

    def test_trial_record_cannot_be_edited_or_reused_as_unsigned_config(self):
        config_dir = self.directory / 'trial-configs'
        config_dir.mkdir()
        config_path = config_dir / 'trial.json'
        config = self.config()
        learning._write_new(config_path, {'config': config.to_dict(), 'provenance': {}})
        learning._write_new(self.directory / 'study.manifest.json',
                            {'rust_learning_schema_version': 1, 'provenance': {},
                             'trials': [{'path': str(config_path), 'sha256': 'a' * 64}]})
        with self.assertRaisesRegex(ValueError, 'hash-bound'):
            learning.load_trial_record(config_path)

    def test_runtime_freeze_hashes_nested_native_sources_not_build_outputs(self):
        for path in ('native\\src\\engine.rs', 'native\\tests\\test_case.rs', 'native\\target\\generated.rs',
                     'runner.py', 'battle_city.nes', 'engine.dll'):
            file = self.directory / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'fixture')
        with patch.object(learning, 'ROOT', self.directory), \
                patch('rust_env.native_library', return_value=SimpleNamespace(_name=self.directory / 'engine.dll')), \
                patch.object(learning.subprocess, 'run', return_value=SimpleNamespace(stdout='commit\n')):
            signature = learning.runtime_signature()
        self.assertIn('native\\tests\\test_case.rs', signature['native_sources'])
        self.assertNotIn('native\\target\\generated.rs', signature['native_sources'])
        self.assertEqual(signature['source_commit'], 'commit')

    def test_all_six_signed_trials_share_one_fixed_seed_stage_initialization(self):
        import numpy as np
        calls = []

        def stage_ram(stage, seed):
            calls.append((stage, seed))
            return np.full(2048, stage + seed, np.uint8)

        args = learning.parse_args(['study', '--output', str(self.directory / 'study')])
        with patch.object(learning, 'runtime_signature', return_value={'fixture': True}), \
                patch('rust_env.stage_ram', side_effect=stage_ram):
            manifest = learning.prepare_study(args)
            configs = [learning.load_trial_record(record['path'])[0] for record in manifest['trials']]
            self.assertEqual(calls, [(1, 0), (2, 0), (3, 0)])
            self.assertEqual(len({c.initial_ram_file for c in configs}), 1)
            initial = learning.verify_frozen(manifest['provenance'])
            np.testing.assert_array_equal(initial[:, 0], [1, 2, 3])
            with self.assertRaises(FileExistsError):
                learning.prepare_study(args)
        with patch.object(learning, 'runtime_signature', return_value={'fixture': False}), \
                self.assertRaisesRegex(ValueError, 'signature changed'):
            learning.verify_frozen(manifest['provenance'])

    def test_critical_result_write_failure_propagates(self):
        path = self.directory / 'result.json'
        learning._write_new(path, {'status': 'first'})
        with self.assertRaises(FileExistsError):
            learning._write_new(path, {'status': 'second'})
        self.assertEqual(json.loads(path.read_text())['status'], 'first')

    def test_study_continues_failed_child_without_retry_and_bounds_owned_processes(self):
        records = [{'path': str(index), 'output': str(self.directory / f'trial{index}'),
                    'model': 'cnn', 'seed': index} for index in range(4)]
        manifest = {'trials': records, 'provenance': {}, 'total_step_cap': 16}
        original_popen = subprocess.Popen
        processes = []
        observed_concurrency = []

        def spawn(command, **kwargs):
            if len(command) < 2 or command[1] != str(learning.ROOT / 'rust_train.py'):
                return original_popen(command, **kwargs)
            index = int(command[-1])
            active = sum(p.poll() is None for p in processes)
            observed_concurrency.append(active + 1)
            if index == 0:
                code = (
                    "import pathlib,sys;"
                    f"p=pathlib.Path({records[index]['output']!r});p.mkdir();"
                    "(p/'result.json').write_text('{\"status\":\"failed\"}');sys.exit(3)")
            else:
                code = (
                    "import pathlib,json,hashlib,time;"
                    "time.sleep(0.1);"
                    f"p=pathlib.Path({records[index]['output']!r});p.mkdir();"
                    f"r={{'model':'cnn','seed':{index},'status':'completed',"
                    "'stop_reason':'step_cap','optimized_steps':4,'collected_steps':4,"
                    "'learning_effect_valid':True};"
                    "b=json.dumps(r).encode();(p/'result.json').write_bytes(b);"
                    "(p/'RESULTS.md').write_text('fixture');"
                    "(p/'result.manifest.json').write_text(json.dumps({"
                    "'result_sha256':hashlib.sha256(b).hexdigest(),"
                    "'report_sha256':hashlib.sha256(b'fixture').hexdigest()}))")
            process = original_popen([sys.executable, '-c', code], **kwargs)
            processes.append(process)
            return process

        args = SimpleNamespace(output=self.directory, parallel=2, max_wall_seconds=30)
        with patch.object(learning, 'prepare_study', return_value=manifest), \
                patch.object(learning, 'verify_frozen'), patch.object(learning.subprocess, 'Popen', side_effect=spawn):
            result = learning.study(args)
        self.assertEqual(len(processes), 4)
        self.assertEqual(max(observed_concurrency), 2)
        self.assertEqual(sum(r['status'] == 'failed' for r in result['trials']), 1)
        self.assertEqual(result['optimized_steps'], 12)
        self.assertFalse(result['automatic_retry'])
        self.assertFalse(result['all_learning_effect_evaluations_valid'])
        self.assertIsNone(result['automatic_winner'])
        self.assertTrue((self.directory / 'RESULTS.md').is_file())

    def test_hard_wall_cleanup_records_partial_trial_not_generic_child_failure(self):
        record = {'path': '0', 'output': str(self.directory / 'trial'),
                  'model': 'cnn', 'seed': 0}
        manifest = {'trials': [record], 'provenance': {}, 'total_step_cap': 16}
        original_popen = subprocess.Popen
        clock = [0.0]

        def spawn(command, **kwargs):
            if len(command) < 2 or command[1] != str(learning.ROOT / 'rust_train.py'):
                return original_popen(command, **kwargs)
            return original_popen([sys.executable, '-c', 'import time;time.sleep(120)'], **kwargs)

        def tick():
            clock[0] += 100
            return clock[0]

        args = SimpleNamespace(output=self.directory, parallel=1, max_wall_seconds=1)
        with patch.object(learning, 'prepare_study', return_value=manifest), \
                patch.object(learning, 'verify_frozen'), patch.object(learning.subprocess, 'Popen', side_effect=spawn), \
                patch.object(learning.time, 'monotonic', side_effect=tick):
            result = learning.study(args)
        self.assertEqual(result['trials'][0]['stop_reason'], 'wall_cap')
        self.assertEqual(result['trials'][0]['status'], 'incomplete')
        self.assertFalse(result['trials'][0]['learning_effect_valid'])

    @unittest.skipUnless(sys.platform == 'win32', 'Windows owned process-tree regression')
    def test_cleanup_owns_venv_launcher_and_descendant_processes(self):
        marker = self.directory / 'child-pid.json'
        source = ("import os,time,pathlib;"
                  f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(120)")
        process = subprocess.Popen([sys.executable, '-c', source],
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            deadline = time.monotonic() + 20
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            child_pid = int(marker.read_text())
            learning.terminate_owned_process(process)
            self.assertIsNotNone(process.poll())
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, 0, child_pid)
            if handle:
                try:
                    code = ctypes.c_uint32()
                    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                    self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                    self.assertNotEqual(code.value, 259, 'venv child is still active')
                finally:
                    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                    kernel.CloseHandle(handle)
        finally:
            learning.terminate_owned_process(process)


class FixtureRuntime:
    def __init__(self, partial=False):
        self.optimized_steps = 0
        self.collected_steps = 0
        self.optimizer_updates = 0
        self.partial = partial

    def update(self, deadline):
        self.collected_steps += 2 if self.partial else 4
        if self.partial:
            return False
        self.optimized_steps += 4
        self.optimizer_updates += 1
        return True

    def diagnostics(self):
        return {'rollout_storage_bytes': 100, 'completed_episodes': []}


if __name__ == '__main__':
    unittest.main()
