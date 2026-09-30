"""Fake-opening entity development evaluation; never initialize ROM/test families."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

import game_state as gs
from within_stage.entity_observation import entity_schema
from within_stage.entity_policy import make_entity_model, save_entity_checkpoint, state_sha256, file_sha256
from within_stage.observation import OBSERVATION_VERSION
from within_stage.test_entity_observation import fixture_ram
from learning_lab.test_rom_finetune import FakeBudget
from within_stage import entity_evaluation as evaluator


class FakeOpening(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    def __init__(self, account):
        self.account = account
        self.nes = SimpleNamespace(ram=fixture_ram(), step=lambda action: None)
        self.state = gs.StateTracker().reset(self.nes.ram)
    def reset(self, *, seed=None, options=None):
        self.account.before('raw'); self.account.after('raw')
        self.state = gs.StateTracker().reset(self.nes.ram)
        self.steps = 0
        return np.full((47, 26, 26), .1, np.float32), {}
    def action_masks(self):
        return np.ones(10, bool)
    def step(self, action):
        self.account.before('agent')
        self.steps += 1
        done = self.steps == 2
        self.state = replace(self.state, frame=self.steps * 4,
                             remaining_enemies=0 if done else 20, kills=20 if done else 0)
        self.account.after('agent')
        return np.full((47, 26, 26), self.steps / 10, np.float32), 0., done, False, {
            'outcome': 'clear' if done else '', 'objective': 'full', 'stage': 1,
            'kills': self.state.kills, 'deaths': 0, 'frames': self.state.frame}
    def close(self):
        pass


class EntityEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path('runs') / ('entity-eval-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True); self.addCleanup(shutil.rmtree, self.root)
        self.model = make_entity_model(seed=0)
        self.checkpoint = self.root / 'model.zip'
        save_entity_checkpoint(self.model, self.checkpoint,
                               dataset_config={'schema': entity_schema(), 'datasets': []})
        self.pool = self.root / 'pool.json'
        specs = [{'config_index': i, 'family_id': i, 'spec': {'title_wait': 0, 'stage_wait': 0},
                  'initial_ram_sha256': hashlib.sha256(fixture_ram().tobytes()).hexdigest()}
                 for i in range(52)]
        self.pool.write_text(json.dumps({'specs': specs}))
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps({
            'stage': 1, 'post_start_frames': 150, 'max_frames': 12000,
            'pool_path': str(self.pool.resolve()), 'pool_sha256': file_sha256(self.pool),
            'observation_version': OBSERVATION_VERSION,
            'train': {'all_config_indices': list(range(40))},
            'development': {'all_config_indices': list(range(40, 52)), 'family_ids': list(range(40, 52))}}))
        value = patch.object(evaluator, 'FROZEN_SPLIT_SHA256', file_sha256(self.split))
        value.start(); self.addCleanup(value.stop)
        self.created = []

    def run_case(self, config=40, budget=None, intervention=None, **kwargs):
        budget = budget or FakeBudget()
        output = self.root / (uuid.uuid4().hex + '.json')
        def factory(pool, config_index, *, account, max_frames):
            self.created.append(config_index)
            return FakeOpening(account)
        with patch.object(evaluator, 'load_pool', lambda path: json.loads(self.pool.read_text())), \
                patch.object(evaluator, 'make_opening_env', factory):
            result = evaluator.evaluate_entity_development_case(
                self.model, self.checkpoint, self.split, config, output, budget,
                intervention_decision=intervention, **kwargs)
        return result, budget

    def test_dict_history_is_owned_once_and_normal_accounting_and_hashes_match(self):
        inputs = []
        original = self.model.predict
        def record(obs, **kwargs):
            inputs.append({k: v.copy() for k, v in obs.items()})
            return original(obs, **kwargs)
        before = state_sha256(self.model.policy.state_dict())
        with patch.object(self.model, 'predict', record):
            result, budget = self.run_case()
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertTrue(result['ordinary_model_clear'])
        self.assertEqual(result['accounting']['attempted'], 3)
        self.assertEqual(budget.rows[0]['actual'], 3)
        self.assertEqual(inputs[0]['grid'].shape, (110, 26, 26))
        self.assertEqual(inputs[0]['entities'].shape, (18, 16))
        np.testing.assert_array_equal(inputs[0]['grid'][:55], 0)
        np.testing.assert_array_equal(inputs[1]['grid'][:55], inputs[0]['grid'][55:])
        self.assertEqual(before, state_sha256(self.model.policy.state_dict()))

    def test_stress_clear_is_not_an_ordinary_model_clear(self):
        result, _ = self.run_case(intervention=0)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertTrue(result['full_game_clear'])
        self.assertFalse(result['ordinary_model_clear'])
        self.assertTrue(result['stress']['recovery_rate_valid'])
        self.assertTrue(result['recovery_success'])
        self.assertEqual(result['stress']['applied_interventions'], 1)

    def test_train_final_and_bad_hash_fail_before_any_environment(self):
        for config in (0, 52):
            with self.assertRaises(ValueError):
                self.run_case(config=config)
        self.assertEqual(self.created, [])
        with self.checkpoint.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaises(ValueError):
            self.run_case()
        self.assertEqual(self.created, [])

    def test_expired_budget_returns_partial_without_environment_and_no_clear_claim(self):
        class Expired(FakeBudget):
            def snapshot(self):
                state = super().snapshot(); state['remaining_seconds'] = 0
                return state
        result, _ = self.run_case(budget=Expired())
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['ordinary_model_clear'])
        self.assertFalse(result['recovery_success'])
        self.assertEqual(self.created, [])

    def test_missed_stress_is_not_a_recovery_success_and_global_rng_is_restored(self):
        np.random.seed(123)
        expected = np.random.RandomState(123).rand(3)
        result, _ = self.run_case(intervention=200)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertTrue(result['full_game_clear'])
        self.assertFalse(result['ordinary_model_clear'])
        self.assertFalse(result['recovery_success'])
        self.assertFalse(result['stress']['recovery_rate_valid'])
        self.assertEqual(result['stress']['invalid_reason'], 'scheduled_index_not_reached')
        np.testing.assert_array_equal(np.random.rand(3), expected)

    def test_wrong_initial_ram_stops_case_and_preserves_unsettled_reservation(self):
        value = json.loads(self.pool.read_text())
        value['specs'][40]['initial_ram_sha256'] = '0' * 64
        self.pool.write_text(json.dumps(value))
        split = json.loads(self.split.read_text())
        split['pool_sha256'] = file_sha256(self.pool)
        self.split.write_text(json.dumps(split))
        with patch.object(evaluator, 'FROZEN_SPLIT_SHA256', file_sha256(self.split)):
            result, budget = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Initial RAM', result['error'])
        self.assertEqual(result['actions'], [])
        self.assertEqual(result['accounting']['attempted'], 1)
        self.assertFalse(result['settled'])
        self.assertIsNone(budget.rows[0]['actual'])

    def test_parameter_mutation_during_case_invalidates_clear_claim(self):
        original = self.model.predict
        def mutate(obs, **kwargs):
            with torch.no_grad():
                self.model.policy.action_net.weight.add_(.1)
            return original(obs, **kwargs)
        with patch.object(self.model, 'predict', mutate):
            result, budget = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['policy_parameters_unchanged'])
        self.assertFalse(result['ordinary_model_clear'])
        self.assertFalse(result['full_game_clear'])
        self.assertIsNone(budget.rows[0]['actual'])


if __name__ == '__main__':
    unittest.main()
