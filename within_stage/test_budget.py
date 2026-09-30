from pathlib import Path
import tempfile
import time
import unittest


class WithinStageBudgetTests(unittest.TestCase):
    def test_fresh_twelve_hour_authorization_and_existing_reservation_rules(self):
        from within_stage.budget import WithinStageBudget
        from learning_lab.budget import BudgetExhausted
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.sqlite'
            budget = WithinStageBudget.create(path, max_interactions=100, wall_seconds=43200)
            self.assertGreater(budget.snapshot()['remaining_seconds'], 43190)
            token = budget.reserve('probe', 80)
            with self.assertRaises(BudgetExhausted):
                budget.reserve('over', 21)
            budget.settle(token, 40)
            self.assertEqual(WithinStageBudget(path).snapshot()['remaining_interactions'], 60)

    def test_cannot_extend_or_overwrite_an_existing_tranche(self):
        from within_stage.budget import WithinStageBudget
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.sqlite'
            WithinStageBudget.create(path, max_interactions=100, wall_seconds=3600)
            with self.assertRaises(FileExistsError):
                WithinStageBudget.create(path, max_interactions=200, wall_seconds=43200)
            for steps, seconds in ((4000001, 43200), (100, 43201), (True, 1)):
                with self.assertRaises(ValueError):
                    WithinStageBudget.create(Path(directory) / 'bad.sqlite',
                                             max_interactions=steps, wall_seconds=seconds)

    def test_expired_new_authorization_does_not_start_more_work(self):
        from within_stage.budget import WithinStageBudget
        from learning_lab.budget import BudgetExhausted
        with tempfile.TemporaryDirectory() as directory:
            budget = WithinStageBudget.create(Path(directory) / 'budget.sqlite',
                                             max_interactions=100, wall_seconds=1,
                                             started_epoch=time.time() - 10)
            with self.assertRaises(BudgetExhausted):
                budget.reserve('late', 1)
