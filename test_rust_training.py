"""Acceptance tests for the experimental backend, not training-quality claims."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack

from rust_env import RustVectorEnv, stage_ram
from rust_train import contract, parse_args, train, evaluate_rom, validate_manifest


class RustTrainingTests(unittest.TestCase):
    def test_budget_requires_complete_rollouts(self):
        with self.assertRaises(SystemExit):
            parse_args(['train', '--output', 'unused', '--steps', '65',
                        '--envs', '2', '--n-steps', '16', '--batch-size', '8'])

    def test_manifest_rejects_old_action_contract(self):
        manifest = {'contract': contract()}
        validate_manifest(manifest)
        for key, value in (('actions', 15), ('rng', 'rom-equivalent'),
                           ('observation_shape', [47, 26, 26]), ('seed_derivation', 'unknown')):
            bad = {'contract': {**manifest['contract'], key: value}}
            with self.assertRaises(ValueError):
                validate_manifest(bad)

    def test_comparison_resolves_output_in_callers_working_directory(self):
        root = Path(__file__).parent
        with tempfile.TemporaryDirectory(dir=root, prefix='.rust-cwd-check-') as native_temp, \
                tempfile.TemporaryDirectory() as caller:
            relative = Path(Path(native_temp).name)
            result = subprocess.run(
                [sys.executable, str(root / 'rust_train.py'), 'compare', '--output', str(relative),
                 '--codec', 'zlib', '--steps', '4', '--envs', '2', '--n-steps', '2',
                 '--batch-size', '4', '--epochs', '1', '--max-frames', '4', '--device', 'cpu'],
                cwd=caller, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = Path(caller) / relative
            report = json.loads((output / 'comparison.json').read_text(encoding='utf-8'))
            self.assertTrue(report['same_final_parameters'])
            self.assertTrue((output / 'dense' / 'model.zip').is_file())
            self.assertTrue((output / 'zlib' / 'model.zip').is_file())
            self.assertEqual(list(Path(native_temp).iterdir()), [])

    def test_dense_and_compact_current_backend_updates_are_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            reports = []
            for codec in ('dense', 'zlib', 'native'):
                args = parse_args(['train', '--output', str(Path(temp) / codec),
                                   '--codec', codec, '--steps', '32', '--envs', '2',
                                   '--n-steps', '8', '--batch-size', '8', '--epochs', '2',
                                   '--max-frames', '12', '--device', 'cpu'])
                reports.append(train(args))
            self.assertEqual(len({r['policy_sha256'] for r in reports}), 1)
            self.assertEqual([r['steps'] for r in reports], [32, 32, 32])
            self.assertLess(reports[1]['rollout_storage_bytes'], reports[0]['rollout_storage_bytes'])
            eval_args = parse_args(['eval', '--model', str(Path(temp) / 'native' / 'model.zip'),
                                   '--output', str(Path(temp) / 'transfer.json'),
                                   '--stages', '1', '--seeds', '0', '--device', 'cpu'])
            transfer = evaluate_rom(eval_args)
            self.assertEqual(len(transfer['episodes']), 2)
            self.assertEqual(transfer['backend'], 'original-rom-transfer')
            self.assertTrue(all(case['frames'] <= 12 for case in transfer['episodes']))

    def test_frame_stack_keeps_terminal_and_next_episode_separate(self):
        initial = stage_ram(1)[None, :]
        env = VecFrameStack(RustVectorEnv(initial, 4), n_stack=2, channels_order='first')
        try:
            before = env.reset()
            obs, _, dones, infos = env.step(np.array([5]))
            self.assertTrue(dones[0])
            np.testing.assert_array_equal(obs[:, :47], 0)
            np.testing.assert_array_equal(infos[0]['terminal_observation'][:47], before[0, 47:])
            self.assertEqual(float(infos[0]['terminal_observation'][83, 0, 0]), 0.0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
