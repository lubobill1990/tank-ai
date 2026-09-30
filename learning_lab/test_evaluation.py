from pathlib import Path
import tempfile
import unittest
import numpy as np

from learning_lab.budget import ResearchBudget


class EvaluationTests(unittest.TestCase):
    def test_up_fire_baseline_respects_masks_without_selecting_a_macro(self):
        from learning_lab.evaluation import BaselinePolicy
        policy = BaselinePolicy('up_fire')
        mask = np.ones(10, dtype=bool)
        self.assertEqual(policy.predict(None, deterministic=True, action_masks=mask)[0], 6)
        mask[6] = False
        self.assertEqual(policy.predict(None, deterministic=True, action_masks=mask)[0], 5)
        mask[5] = False
        self.assertEqual(policy.predict(None, deterministic=True, action_masks=mask)[0], 0)
        mask[:] = False
        with self.assertRaises(ValueError):
            policy.predict(None, deterministic=True, action_masks=mask)

    def test_development_cannot_consume_sealed_generalization_stages(self):
        from learning_lab.evaluation import BaselinePolicy, evaluate_candidate
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=1000, wall_seconds=3600)
            with self.assertRaises(ValueError):
                evaluate_candidate(BaselinePolicy('up_fire'), model_name='cnn',
                                   output=Path(directory) / 'bad.json', budget=budget,
                                   stages=(8,), seeds=(0,), max_frames=4)
            self.assertEqual(budget.snapshot()['charged_interactions'], 0)

    def test_real_smoke_counts_decisions_and_preserves_the_unmodified_rom_contract(self):
        from learning_lab.evaluation import BaselinePolicy, evaluate_candidate
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=1000, wall_seconds=3600)
            result = evaluate_candidate(
                BaselinePolicy('up_fire'), model_name='cnn',
                output=Path(directory) / 'smoke.json', budget=budget,
                stages=(1,), seeds=(0,), max_frames=4)
            self.assertEqual(result['decision_attempts'], 28)
            self.assertEqual(budget.snapshot()['settled_interactions'], 28)
            self.assertTrue(result['metrics']['metadata']['smoke_only'])
            self.assertFalse(result['learned_model'])
            self.assertEqual(result['metrics']['rom']['modes']['deterministic']['summary']['episodes'], 1)
            self.assertTrue((Path(directory) / 'smoke.json').is_file())
