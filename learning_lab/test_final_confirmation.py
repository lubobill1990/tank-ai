"""Synthetic final-evaluator tests. Never read real held-out seeds or start NES."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

import game_state as gs
from learning_lab.test_rom_finetune import FakeBudget
from learning_lab import final_confirmation as final


class FakeModel:
    learned_model = True
    observation_space = spaces.Box(-1, 1, (94, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    device = 'cpu'
    decoder_config = {'name': 'synthetic-deterministic'}
    def __init__(self, weight=1):
        self.policy = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.policy.weight.fill_(weight)
    def predict(self, observation, *, deterministic, action_masks):
        assert deterministic and action_masks[0]
        return np.asarray(0, np.int64), None


def fake_loader(path, device='cpu'):
    return FakeModel(json.loads(Path(path).read_text())['weight'])


class FakeNES:
    def __init__(self):
        self.ram = np.zeros(2048, np.uint8)
        self.screen = np.zeros((4, 4, 3), np.uint8)
    def step(self, action):
        return self.screen, 0., False, {}


class FakeEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(15)
    def __init__(self, stage, *, duplicate=False, mutate=None):
        self.stage, self.max_frames, self.frame_skip = stage, 12000, 4
        self.nes = FakeNES()
        self.duplicate, self.mutate = duplicate, mutate
        self.frame = 0
    def _state(self):
        state = gs.parse(self.nes.ram, frame=self.frame)
        self.state = state
    def reset(self, *, seed=None, options=None):
        self.nes.ram.fill(0)
        self.nes.ram[0x68], self.nes.ram[0x51], self.nes.ram[0x80], self.nes.ram[0x85] = 128, 3, 20, self.stage
        if not self.duplicate:
            self.nes.ram[:8] = np.frombuffer(int(seed).to_bytes(8, 'little'), np.uint8)
        self.frame = 0
        self.nes.step(0)
        self._state()
        return np.zeros((47, 26, 26), np.float32), {}
    def action_masks(self):
        return np.ones(15, bool)
    def step(self, action):
        self.frame += 4
        self.nes.step(action)
        self.nes.ram[0x80], self.nes.ram[0x73] = 0, 20
        self._state()
        if self.mutate:
            self.mutate()
        info = dict(outcome='clear', objective='full', stage=self.stage, frames=self.frame,
                    kills=20, deaths=0, shots=0, pickups=0, lives=3)
        return np.zeros((47, 26, 26), np.float32), 1., True, False, info
    def close(self):
        pass


class FinalBudget(FakeBudget):
    def __init__(self, cases=None, path=None):
        super().__init__()
        self.cases = cases
        self.path = Path(path) if path is not None else None
    def reserve(self, label, interactions):
        if any(row['label'] == label for row in self.rows):
            raise ValueError('final-once label already claimed')
        if interactions == final.CASE_RESERVATION and self.cases is not None:
            if self.cases == 0:
                raise final.BudgetExhausted('synthetic case budget expired')
            self.cases -= 1
        return super().reserve(label, interactions)


class FinalConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('final-confirmation-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.checkpoint = self.root / 'fake-checkpoint.json'
        self.checkpoint.write_text('{"weight":1}')
        self.model = FakeModel()
        self.selection = final.make_selection_record(
            self.model, self.checkpoint, loader=fake_loader,
            development_gate_passed=False, rationale='best available synthetic candidate')
        dev = [0, 1, 7, 39]
        used = {int(np.random.default_rng(s).integers(0, 24)) for s in dev}
        rows = []
        for seed in range(100, 10000):
            delay = int(np.random.default_rng(seed).integers(0, 24))
            if delay not in used:
                used.add(delay); rows.append({'seed': seed, 'reset_delay_frames': delay})
            if len(rows) == 20:
                break
        self.pool = self.root / 'synthetic-pool.json'
        self.pool.write_text(json.dumps({
            'version': 'synthetic-v1', 'training_rom_seeds': [0, 1, 2], 'development_rom_seeds': dev,
            'sealed_stages': [8, 18, 28], 'stable_stage_and_generalization_test': rows,
            'reset_delay_source_sha256': final._hash(final.ROOT / 'battle_city_env.py')}))
        self.created = []

    def run_final(self, *, budget=None, duplicate=False, mutate=None, selection=None, output='final',
                  stages=(1, 8, 18, 28)):
        destination = self.root / output
        budget = budget or FinalBudget()
        if budget.path is None:
            budget.path = self.root / 'budget.sqlite'
        def factory(stage):
            self.assertTrue((destination / 'selection.json').is_file())
            self.assertTrue((destination / 'SOURCE.SIG.json').is_file())
            saved = json.loads((destination / 'selection.json').read_text())
            self.assertEqual(saved['role'], 'heldout-final-once')
            receipt = json.loads((budget.path.parent / 'final-test-used.json').read_text())
            self.assertEqual(receipt['selection_sha256'], final._hash(destination / 'selection.json'))
            self.assertEqual(receipt['opening_pool_sha256'], final._hash(self.pool))
            self.created.append(stage)
            return FakeEnv(stage, duplicate=duplicate, mutate=mutate)
        with patch.object(final, '_make_environment', factory):
            return final.run_final_confirmation(
                self.model, self.checkpoint, destination, budget,
                selection_record=self.selection if selection is None else selection,
                opening_pool_path=self.pool, stages=stages)

    def test_unqualified_candidate_is_measured_once_with_eighty_complete_unique_cases(self):
        budget = FinalBudget()
        report = self.run_final(budget=budget)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['completed_cases'], 80)
        self.assertTrue(report['overall_passed'])
        self.assertEqual(self.created, [1] * 20 + [8] * 20 + [18] * 20 + [28] * 20)
        self.assertFalse(report['selection']['development_gate_passed'])
        self.assertEqual(report['selection']['role'], 'heldout-final-once')
        self.assertEqual(report['actual_decisions'], 160)
        self.assertEqual(report['per_stage']['1']['threshold'], 16)
        self.assertEqual(report['per_stage']['8']['threshold'], 6)
        self.assertTrue(all(len(row['initial_ram_hashes']) == 20 for row in report['per_stage'].values()))
        self.created.clear()
        again = self.run_final(budget=budget, output='second-model-forbidden')
        self.assertEqual(again['status'], 'failed')
        self.assertEqual(self.created, [])

    def test_wrong_checkpoint_hash_and_wrong_policy_identity_never_create_environment(self):
        bad = deepcopy(self.selection); bad['checkpoint_sha256'] = '0' * 64
        report = self.run_final(selection=bad)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.created, [])
        with torch.no_grad():
            self.model.policy.weight.add_(1)
        report = self.run_final(output='changed')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.created, [])

    def test_changed_parameters_during_episode_stop_without_valid_case_or_retry(self):
        def mutate():
            with torch.no_grad():
                self.model.policy.weight.add_(1)
        report = self.run_final(mutate=mutate)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completed_cases'], 0)
        self.assertEqual(len(self.created), 1)
        self.assertFalse(report['overall_passed'])

    def test_duplicate_actual_ram_does_not_become_a_fabricated_episode(self):
        report = self.run_final(duplicate=True)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completed_cases'], 1)
        self.assertEqual(len(self.created), 2)
        self.assertIn('duplicate', report['error'].lower())
        self.assertEqual(report['per_stage']['1']['completed'], 1)
        self.assertFalse(report['per_stage']['1']['passed'])

    def test_budget_expiry_preserves_partial_results_without_eighty_case_claim(self):
        report = self.run_final(budget=FinalBudget(cases=3))
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['completed_cases'], 3)
        self.assertEqual(report['planned_cases'], 80)
        self.assertFalse(report['overall_passed'])
        self.assertTrue(all(not row['passed'] for row in report['per_stage'].values()))
        self.assertEqual(len(self.created), 3)

    def test_bad_seed_set_and_stage_scope_are_rejected_before_environment_creation(self):
        pool = json.loads(self.pool.read_text())
        pool['stable_stage_and_generalization_test'] = pool['stable_stage_and_generalization_test'][:19]
        self.pool.write_text(json.dumps(pool))
        report = self.run_final()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.created, [])
        report = final.run_final_confirmation(
            self.model, self.checkpoint, self.root / 'bad-stages', FinalBudget(),
            selection_record=self.selection, opening_pool_path=self.pool, stages=(1, 3))
        self.assertEqual(report['status'], 'failed')

    def test_partial_attempt_receipt_blocks_new_budget_output_and_checkpoint(self):
        report = self.run_final(budget=FinalBudget(cases=1))
        self.assertEqual(report['status'], 'partial')
        receipt_path = self.root / 'final-test-used.json'
        original = receipt_path.read_bytes()
        self.created.clear()
        self.checkpoint = self.root / 'different-checkpoint.json'
        self.checkpoint.write_text('{"weight":2}')
        self.model = FakeModel(2)
        self.selection = final.make_selection_record(
            self.model, self.checkpoint, loader=fake_loader,
            development_gate_passed=False, rationale='must not retarget after final exposure')
        again = self.run_final(budget=FinalBudget(path=self.root / 'new-budget.sqlite'),
                               output='different-output')
        self.assertEqual(again['status'], 'failed')
        self.assertIn('final-test-used.json', again['error'])
        self.assertEqual(self.created, [])
        self.assertEqual(receipt_path.read_bytes(), original)

    def test_receipt_creation_is_exclusive_and_never_overwrites_existing_bytes(self):
        budget = FinalBudget(path=self.root / 'budget.sqlite')
        receipt = {'role': final.ROLE, 'selection_sha256': 'a' * 64,
                   'opening_pool_sha256': 'b' * 64}
        path = final._create_final_receipt(budget, receipt)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            final._create_final_receipt(budget, {**receipt, 'selection_sha256': 'c' * 64})
        self.assertEqual(path.read_bytes(), original)
        path.write_bytes(b'partial/crash receipt')
        with self.assertRaises(FileExistsError):
            final._create_final_receipt(budget, receipt)
        self.assertEqual(path.read_bytes(), b'partial/crash receipt')

    def test_stage_three_primary_is_predeclared_and_receives_the_sixteen_win_gate(self):
        selection = final.make_selection_record(
            self.model, self.checkpoint, loader=fake_loader, development_gate_passed=False,
            rationale='synthetic predeclared primary map3', primary_stage=3)
        report = self.run_final(selection=selection, stages=(3, 8, 18, 28))
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['completed_cases'], 80)
        self.assertEqual(report['primary_stage'], 3)
        self.assertEqual(self.created, [3] * 20 + [8] * 20 + [18] * 20 + [28] * 20)
        self.assertNotIn('1', report['per_stage'])
        self.assertEqual(report['per_stage']['3']['threshold'], 16)
        self.assertTrue(all(report['per_stage'][str(stage)]['threshold'] == 6 for stage in (8, 18, 28)))
        receipt = json.loads((self.root / 'final-test-used.json').read_text())
        self.assertEqual(receipt['primary_stage'], 3)
        self.assertEqual(receipt['stages'], [3, 8, 18, 28])
        self.created.clear()
        again = self.run_final(output='cannot-switch-primary', budget=FinalBudget())
        self.assertEqual(again['status'], 'failed')
        self.assertEqual(self.created, [])

    def test_primary_mismatch_is_rejected_before_pool_read_or_receipt(self):
        with patch.object(final, '_pool', side_effect=AssertionError('Pool must remain unopened')):
            report = self.run_final(stages=(3, 8, 18, 28))
        self.assertEqual(report['status'], 'failed')
        self.assertIn('primary', report['error'].lower())
        self.assertEqual(self.created, [])
        self.assertFalse((self.root / 'final-test-used.json').exists())


if __name__ == '__main__':
    unittest.main()
