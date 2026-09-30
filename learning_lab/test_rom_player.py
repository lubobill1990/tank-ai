from pathlib import Path
import tempfile
import unittest

from learning_lab.budget import ResearchBudget
from learning_lab.evaluation import BaselinePolicy


class RomPlayerTests(unittest.TestCase):
    def test_development_rejects_sealed_map_or_opening_before_reserving(self):
        from learning_lab.rom_player import play_development_episode
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=100, wall_seconds=3600)
            for stage, seed in ((8, 0), (1, 23)):
                with self.assertRaises(ValueError):
                    play_development_episode(
                        BaselinePolicy('up_fire'), output=Path(directory) / 'bad.json',
                        budget=budget, stage=stage, seed=seed, max_frames=4)
            self.assertEqual(budget.snapshot()['charged_interactions'], 0)

    def test_one_real_episode_records_initial_state_actions_and_actual_terminal_evidence(self):
        from learning_lab.rom_player import play_development_episode
        with tempfile.TemporaryDirectory() as directory:
            budget = ResearchBudget.create(Path(directory) / 'budget.sqlite',
                                          max_interactions=100, wall_seconds=3600)
            report = play_development_episode(
                BaselinePolicy('up_fire'), output=Path(directory) / 'episode.json',
                budget=budget, stage=1, seed=0, max_frames=4)
            self.assertEqual(report['episode']['outcome'], 'timeout')
            self.assertFalse(report['learned_model'])
            self.assertFalse(report['verified_full_clear'])
            self.assertEqual(len(report['initial_ram_sha256']), 64)
            self.assertEqual(len(report['actions']), 1)
            self.assertEqual(budget.snapshot()['settled_interactions'], 1)
