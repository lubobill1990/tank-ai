"""Synthetic development evaluator tests; zero real ROM/native decisions."""
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
import uuid

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from within_stage import episode_eval as evaluation


class FakeNES:
    def __init__(self):
        self.ram = np.zeros(2048, np.uint8)
        self.screen = np.zeros((4, 4, 3), np.uint8)
    def step(self, action):
        self.ram[0] += 1
        self.screen.fill(self.ram[0])
        return self.screen, 0., False, {}


class FakeROM(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(15)
    def __init__(self, length=4, only_action=None):
        self.nes, self.length, self.only_action = FakeNES(), length, only_action
        self.max_frames = 12000
        self.position = 0
    def reset(self, *, seed=None, options=None):
        self.position = 0
        self.nes.ram.fill(0)
        self.nes.step(0)
        self._state()
        return np.zeros((47, 26, 26), np.float32), {}
    def _state(self):
        self.state = SimpleNamespace(frame=self.position * 4, remaining_enemies=20 if self.position < self.length else 0,
                                     base_destroyed=False, kills=self.position, lives=3)
    def action_masks(self):
        masks = np.ones(15, bool)
        if self.only_action is not None:
            masks.fill(False); masks[self.only_action] = True
        return masks
    def step(self, action):
        for _ in range(4):
            self.nes.step(action)
        self.position += 1
        self._state()
        done = self.position >= self.length
        info = dict(objective='full', outcome='clear' if done else '', frames=self.position * 4,
                    kills=self.position, deaths=0, lives=3)
        return np.full((47, 26, 26), action / 10, np.float32), np.float32(-0.), done, False, info
    def close(self):
        pass


class FakeModel:
    observation_space = spaces.Box(-1, 1, (94, 26, 26), np.float32)
    device = 'cpu'
    def __init__(self):
        self.policy = torch.nn.Linear(1, 1)
    def predict(self, observation, *, deterministic, action_masks):
        if not deterministic:
            raise AssertionError('Probe must be deterministic')
        return np.asarray(0, np.int64), None


class EpisodeEvalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('episode-eval-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def counter(self, limit=100):
        return evaluation.ProbeCounter(limit, time.monotonic() + 60)

    def test_scripted_parity_counts_both_runs_and_reset_substeps(self):
        counter = self.counter()
        result = evaluation.check_probability_zero(
            [0, 1, 2], self.root / 'parity', counter, env_factory=lambda: FakeROM(length=10))
        self.assertTrue(result['passed'])
        self.assertEqual(result['compared_actions'], 3)
        self.assertEqual(counter.initialization_attempted, 2)
        self.assertEqual(counter.control_attempted, 6)
        self.assertEqual(counter.attempted, 8)
        self.assertEqual(counter.completed, 8)

    def test_stress_records_intended_executed_and_keeps_unapplied_case_invalid(self):
        counter = self.counter()
        report = evaluation.evaluate_recovery_case(
            FakeModel(), opening_seed=0, intervention_decision=1, noise_seed=701,
            output=self.root / 'stress', counter=counter, env_factory=lambda: FakeROM())
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['stress']['requested_interventions'], 1)
        self.assertEqual(report['stress']['applied_interventions'], 1)
        self.assertTrue(report['stress']['recovery_rate_valid'])
        self.assertTrue(report['full_game_clear'])
        self.assertFalse(report['ordinary_pure_model_win'])
        self.assertEqual(sum(report['executed_action_counts'][1:5]), 1)
        invalid = evaluation.evaluate_recovery_case(
            FakeModel(), opening_seed=1, intervention_decision=1, noise_seed=702,
            output=self.root / 'invalid', counter=counter, env_factory=lambda: FakeROM(only_action=0))
        self.assertEqual(invalid['status'], 'completed')
        self.assertFalse(invalid['stress']['recovery_rate_valid'])
        self.assertEqual(invalid['stress']['invalid_reason'], 'no_legal_alternative')
        self.assertEqual(counter.attempted, 10)

    def test_budget_expiry_is_partial_and_never_overruns(self):
        counter = self.counter(3)
        report = evaluation.evaluate_recovery_case(
            FakeModel(), opening_seed=7, intervention_decision=3, noise_seed=703,
            output=self.root / 'partial', counter=counter, env_factory=lambda: FakeROM(length=10))
        self.assertEqual(report['status'], 'partial')
        self.assertFalse(report['stress']['recovery_rate_valid'])
        self.assertIsNone(report['full_game_clear'])
        self.assertEqual(counter.attempted, 3)
        self.assertEqual(counter.completed, 3)
        self.assertEqual(json.loads((self.root / 'partial' / 'report.json').read_text())['status'], 'partial')


if __name__ == '__main__':
    unittest.main()
