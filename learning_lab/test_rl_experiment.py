import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from learning_lab.budget import ResearchBudget
from test_rust_learning import fixture_vec
from test_rust_resources import healthy_host


class RLExperimentTests(unittest.TestCase):
    def test_budget_alignment_rejected_before_any_model_or_reservation(self):
        from learning_lab.rl_experiment import run_trial
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=1000, wall_seconds=3600)
            with self.assertRaises(ValueError):
                run_trial(output=Path(directory) / 'bad', budget=budget, profile='baseline',
                          steps=7, envs=2, n_steps=2, batch_size=4, device='cpu', evaluate=False)
            self.assertEqual(budget.snapshot()['charged_interactions'], 0)

    def test_small_real_ppo_updates_settle_exact_budget_and_save_reloadable_checkpoint(self):
        from learning_lab.rl_experiment import run_trial, load_checkpoint
        with tempfile.TemporaryDirectory() as directory, \
                patch('rust_env.RustVectorEnv', side_effect=fixture_vec), \
                patch('learning_lab.rl_experiment.stage_ram', return_value=np.zeros(2048, np.uint8)), \
                patch('rust_resources.host_snapshot', side_effect=healthy_host):
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=1000, wall_seconds=3600)
            report = run_trial(output=Path(directory) / 'trial', budget=budget,
                               profile='loss_balanced_v1', steps=8, envs=2, n_steps=2,
                               batch_size=4, epochs=1, checkpoint_every=4,
                               device='cpu', evaluate=False)
            self.assertEqual(report['optimized_steps'], 8)
            self.assertEqual(report['collected_steps'], 8)
            self.assertEqual(budget.snapshot()['settled_interactions'], 8)
            self.assertEqual(report['status'], 'completed')
            self.assertFalse(report['learning_effect_valid'])
            model = load_checkpoint(report['final_checkpoint']['path'], device='cpu')
            self.assertEqual(model.num_timesteps, 8)
            self.assertEqual(model.action_space.n, 10)
            self.assertEqual(model.observation_space.shape, (94, 26, 26))
