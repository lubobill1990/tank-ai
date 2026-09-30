import tempfile
from pathlib import Path
import unittest


class BudgetTests(unittest.TestCase):
    def test_reserved_work_counts_until_settled_and_never_exceeds_cap(self):
        from learning_lab.budget import ResearchBudget, BudgetExhausted
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=100, wall_seconds=3600)
            first = budget.reserve('teacher', 60)
            self.assertEqual(budget.snapshot()['remaining_interactions'], 40)
            with self.assertRaises(BudgetExhausted):
                budget.reserve('too-large', 41)
            budget.settle(first, 25)
            self.assertEqual(budget.snapshot()['remaining_interactions'], 75)
            budget.reserve('ppo', 75)
            self.assertEqual(budget.snapshot()['charged_interactions'], 100)
            with self.assertRaises(BudgetExhausted):
                budget.reserve('one-too-many', 1)

    def test_reopening_preserves_unknown_work_as_conservative_charge(self):
        from learning_lab.budget import ResearchBudget
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.sqlite'
            first = ResearchBudget.create(path, max_interactions=100, wall_seconds=3600)
            first.reserve('crashed-worker', 75)
            second = ResearchBudget(path)
            self.assertEqual(second.snapshot()['charged_interactions'], 75)
            self.assertEqual(second.snapshot()['settled_interactions'], 0)
            self.assertEqual(second.snapshot()['remaining_interactions'], 25)

    def test_invalid_or_conflicting_settlement_does_not_release_reservation(self):
        from learning_lab.budget import ResearchBudget
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=100, wall_seconds=3600)
            token = budget.reserve('trial', 40)
            for value in (-1, 41, True, 1.5):
                with self.assertRaises(ValueError):
                    budget.settle(token, value)
            self.assertEqual(budget.snapshot()['charged_interactions'], 40)
            budget.settle(token, 30)
            budget.settle(token, 30)
            with self.assertRaises(ValueError):
                budget.settle(token, 20)
            self.assertEqual(budget.snapshot()['charged_interactions'], 30)

    def test_elapsed_wall_budget_blocks_new_work_without_changing_existing_rows(self):
        from learning_lab.budget import ResearchBudget, BudgetExhausted
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=100, wall_seconds=1,
                                          started_epoch=1)
            with self.assertRaises(BudgetExhausted):
                budget.reserve('late', 1)
            self.assertEqual(budget.snapshot()['charged_interactions'], 0)

    def test_budget_cannot_overwrite_prior_authorization_or_request_larger_mandate(self):
        from learning_lab.budget import ResearchBudget
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.sqlite'
            ResearchBudget.create(path, max_interactions=100, wall_seconds=3600)
            with self.assertRaises(FileExistsError):
                ResearchBudget.create(path, max_interactions=200, wall_seconds=3600)
            for interactions, seconds in ((4000001, 3600), (100, 28801), (True, 3600), (100, 0)):
                with self.assertRaises(ValueError):
                    ResearchBudget.create(Path(directory) / 'invalid.sqlite',
                                          max_interactions=interactions, wall_seconds=seconds)
