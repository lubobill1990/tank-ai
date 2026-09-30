"""Synthetic collector tests. No real opening, ROM, GPU, or teacher run."""
from copy import deepcopy
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

from within_stage.demonstrations import PrecisionDataset
from within_stage.observation import OBSERVATION_VERSION
from within_stage.openings import ProbeAccounting
from within_stage.precision_policy import file_sha256
from within_stage import recovery_data as recovery


class FakeEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (55, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    def __init__(self, account, config, length=3):
        self.account, self.config, self.length = account, config, length
        self.nes = SimpleNamespace(ram=np.zeros(2048, np.uint8))
        self.obs = np.zeros((55, 26, 26), np.float32)
        self.received = []
        self.fail_step = False
        self.max_frames = 12000

    def reset(self, *, seed=None, options=None):
        self.account.before('raw'); self.account.after('raw')
        self.nes.ram.fill(0); self.obs.fill(0)
        self.state = SimpleNamespace(frame=0, tick=0, remaining_enemies=20, base_destroyed=False)
        return self.obs, {}

    def action_masks(self):
        mask = np.zeros(10, bool); mask[[0, 2, 4, 5]] = True
        return mask

    def step(self, action):
        self.account.before('agent')
        self.received.append(int(action))
        if self.fail_step:
            raise RuntimeError('synthetic failed step')
        tick = self.state.tick + 1
        self.state = SimpleNamespace(frame=tick * 4, tick=tick,
                                     remaining_enemies=0 if tick == self.length else 20,
                                     base_destroyed=False)
        self.nes.ram[0] = tick
        self.obs.fill(tick / 10)
        self.account.after('agent')
        done = tick == self.length
        return self.obs, 0., done, False, {'outcome': 'clear' if done else '', 'kills': tick,
                                         'frames': tick * 4, 'deaths': 0}


class FakeTeacher:
    states = []
    def reset(self):
        pass
    def act(self, state, masks):
        self.states.append(state.tick)
        return 2 if state.tick % 2 == 0 else 5


class FakeStudent:
    observation_space = spaces.Box(-1, 1, (110, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    device = 'cpu'
    def __init__(self):
        self.policy = torch.nn.Linear(1, 1)
        self.inputs = []
    def predict(self, observation, *, deterministic, action_masks):
        assert deterministic and action_masks[0]
        self.inputs.append(observation.copy())
        return np.asarray(0, np.int64), None


class RecoveryDataTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('runs') / ('recovery-data-test-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        specs = [{'config_index': family * 2 + alias, 'family_id': family,
                  'spec': {'family': family, 'alias': alias, 'post_start_frames': 150}}
                 for family in range(41) for alias in range(2)]
        self.pool = self.root / 'pool.json'
        self.pool.write_text(json.dumps({'specs': specs}))
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps({
            'stage': 1, 'post_start_frames': 150, 'channels_per_frame': 55, 'n_stack': 2, 'actions': 10,
            'observation_version': OBSERVATION_VERSION,
            'pool_path': str(self.pool.resolve()), 'pool_sha256': file_sha256(self.pool),
            'train': {'family_ids': list(range(40)), 'all_config_indices': list(range(80)),
                      'representative_config_indices': list(range(0, 80, 2))}}))
        patcher = patch.object(recovery, 'FROZEN_SPLIT_SHA256', file_sha256(self.split))
        patcher.start(); self.addCleanup(patcher.stop)
        FakeTeacher.states = []
        self.instances = []
        def factory(pool, config_index, account, deadline):
            env = FakeEnv(account, config_index)
            self.instances.append(env)
            return env
        self.factory = factory

    def collect(self, *, model='default', episodes=2, **kwargs):
        if model == 'default':
            model = FakeStudent()
        account = kwargs.pop('account', ProbeAccounting(1000, deadline=time.monotonic() + 60))
        output = self.root / uuid.uuid4().hex
        with patch.object(recovery, '_make_precision_env', self.factory), \
                patch.object(recovery, 'TeacherControllerV2', FakeTeacher):
            report = recovery.collect_recovery_data(
                model, output, self.split, account=account, episodes=episodes,
                seed=27, deadline=time.monotonic() + 60, **kwargs)
        return report, output, account, model

    def test_first_forty_episodes_cover_only_shuffled_train_representatives(self):
        _, plan = recovery.training_plan(self.split, episodes=80, seed=27)
        self.assertEqual(len(plan), 80)
        self.assertEqual({row['family_id'] for row in plan[:40]}, set(range(40)))
        self.assertTrue(all(row['config_index'] % 2 == 0 for row in plan[:40]))
        self.assertEqual({row['family_id'] for row in plan[40:]}, set(range(40)))
        self.assertTrue(all(row['config_index'] < 80 for row in plan))
        self.assertTrue(all(row['opening_spec']['family'] == row['family_id'] for row in plan))
        self.assertEqual(recovery.training_plan(self.split, episodes=80, seed=27)[1], plan)

    def test_pre_action_teacher_labels_precision_history_and_independent_execution_records(self):
        report, path, account, model = self.collect(teacher_probability=.5, noise_probability=1.)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(FakeTeacher.states, [0, 1, 2, 0, 1, 2])
        data = PrecisionDataset(path)
        self.addCleanup(data.close)
        batch = data.batch(np.arange(6))
        np.testing.assert_array_equal(batch['actions'], [2, 5, 2, 2, 5, 2])
        np.testing.assert_array_equal(batch['observations'][:, 55, 0, 0],
                                      np.array([0, .1, .2, 0, .1, .2], np.float32))
        np.testing.assert_array_equal(data.audit(np.arange(6))['ram'][:, 0], [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(model.inputs[0][:55], 0)
        np.testing.assert_array_equal(model.inputs[3][:55], 0)
        np.testing.assert_array_equal(model.inputs[1][:55], model.inputs[0][55:])
        np.testing.assert_array_equal(np.load(path / 'student_proposals.npy'), [0] * 6)
        selected = np.load(path / 'selected_actions.npy')
        executed = np.load(path / 'executed_actions.npy')
        self.assertTrue(np.all(np.isin(executed, [0, 2, 4])))
        self.assertTrue(np.all(executed != selected))
        self.assertEqual(account.agent_attempted, 6)
        self.assertEqual(account.raw_initialization_attempted, 2)
        self.assertEqual(report['learner_wins'], 0)
        self.assertEqual(report['mixed_trajectory_clears'], 2)
        self.assertEqual(data.metadata['selected_partition'], 'train')
        self.assertEqual(data.metadata['plan'], report['plan'])
        recovery.verify_recovery_integrity(path)

    def test_beta_and_noise_draws_are_independent_of_each_other_and_policy_rng(self):
        np.random.seed(999)
        expected = np.random.RandomState(999).random_sample(3)
        _, first, _, _ = self.collect(teacher_probability=.2, noise_probability=0.)
        _, second, _, _ = self.collect(teacher_probability=.8, noise_probability=1.)
        a = [json.loads(line) for line in (first / 'decisions.jsonl').read_text().splitlines()]
        b = [json.loads(line) for line in (second / 'decisions.jsonl').read_text().splitlines()]
        self.assertEqual([row['beta_draw'] for row in a], [row['beta_draw'] for row in b])
        self.assertEqual([row['noise_draw'] for row in a], [row['noise_draw'] for row in b])
        self.assertNotEqual([row['beta_draw'] for row in a], [row['noise_draw'] for row in a])
        np.testing.assert_array_equal(np.random.random_sample(3), expected)

    def test_no_student_mode_is_explicit_teacher_data_not_a_learned_win(self):
        report, path, _, _ = self.collect(model=None, teacher_probability=.8, noise_probability=0.)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(report['learner_wins'], 0)
        self.assertEqual(report['teacher_only_trajectory_clears'], 2)
        np.testing.assert_array_equal(np.load(path / 'student_proposals.npy'), [-1] * 6)
        self.assertEqual(set(np.load(path / 'selected_sources.npy')), {'teacher'})
        self.assertFalse(report['student_present'])

    def test_failed_tail_and_budget_cutoff_are_preserved_with_exact_accounting(self):
        def fail_factory(pool, config_index, account, deadline):
            env = FakeEnv(account, config_index)
            env.fail_step = True
            return env
        self.factory = fail_factory
        report, path, account, _ = self.collect()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['samples'], 1)
        self.assertEqual(account.agent_attempted, 1)
        self.assertEqual(account.agent_completed, 0)
        self.assertFalse(np.load(path / 'step_completed.npy')[0])
        self.assertIn('failed step', report['error'])
        data = PrecisionDataset(path); self.addCleanup(data.close)
        self.assertEqual(len(data), 1)
        self.factory = lambda pool, config_index, account, deadline: FakeEnv(account, config_index)
        account = ProbeAccounting(3, deadline=time.monotonic() + 60)
        report, path, _, _ = self.collect(account=account)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['samples'], 2)
        self.assertEqual(account.attempted, 3)
        self.assertEqual(report['episode_reports'][0]['outcome'], 'budget_or_deadline')

    def test_invalid_split_or_precision_student_fails_before_any_environment(self):
        self.split.write_text(self.split.read_text() + ' ')
        report, _, account, _ = self.collect()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('split', report['error'])
        self.assertEqual(account.attempted, 0)
        self.assertEqual(self.instances, [])

    def test_collector_deadline_applies_inside_initialization_without_changing_account_deadline(self):
        clock = [100.]
        account = ProbeAccounting(100, deadline=time.monotonic() + 60)
        original_deadline = account.deadline
        def factory(pool, config_index, account, deadline):
            env = FakeEnv(account, config_index)
            original = env.reset
            def slow_reset(**kwargs):
                result = original(**kwargs)
                clock[0] = 102.
                account.before('raw')
                raise AssertionError('Should stop before second initialization substep')
            env.reset = slow_reset
            return env
        with patch.object(recovery, '_make_precision_env', factory), \
                patch.object(recovery, 'TeacherControllerV2', FakeTeacher), \
                patch.object(recovery, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
            report = recovery.collect_recovery_data(
                None, self.root / 'deadline-init', self.split, account=account, episodes=1,
                deadline=101., seed=27)
        self.assertEqual(report['status'], 'partial', report.get('error'))
        self.assertEqual(account.raw_initialization_attempted, 1)
        self.assertEqual(account.agent_attempted, 0)
        self.assertEqual(account.deadline, original_deadline)

    def test_resource_admission_failure_publishes_explicit_empty_precision_corpus(self):
        from rust_resources import ResourcePressureError
        with patch.object(recovery.ResourceMonitor, 'check', side_effect=ResourcePressureError('fixture reserve')):
            report, path, account, _ = self.collect()
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(account.attempted, 0)
        self.assertEqual(report['samples'], 0)
        data = PrecisionDataset(path); self.addCleanup(data.close)
        self.assertEqual(len(data), 0)
        self.assertEqual(self.instances, [])

    def test_collected_dataset_is_accepted_by_frozen_actor_bc_train_membership_validator(self):
        from within_stage import imitation
        report, path, _, _ = self.collect()
        with patch.object(imitation, 'FROZEN_SPLIT_PATH', self.split), \
                patch.object(imitation, 'FROZEN_SPLIT_SHA256', file_sha256(self.split)):
            contract = imitation._training_contract()
            corpus = imitation.TrainingCorpus([path], contract, time.monotonic() + 60)
        self.addCleanup(corpus.close)
        self.assertEqual(len(corpus.episode_ids), 6)
        self.assertEqual(len(corpus.membership), 2)
        self.assertEqual(set(corpus.membership[0]) >= {'family_id', 'config_index'}, True)
        self.assertEqual(report['learner_wins'], 0)

    def test_student_off_adapter_does_not_zero_precision_corpus_and_student_weights_are_checked(self):
        model = FakeStudent()
        model.precision_dataset_config = {'offsets_enabled': False}
        report, path, _, _ = self.collect(model=model)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertFalse(np.any(model.inputs[1][47:55]))
        self.assertFalse(np.any(model.inputs[1][102:110]))
        data = PrecisionDataset(path); self.addCleanup(data.close)
        self.assertTrue(np.any(data.batch([1])['observations'][:, 102:110]))
        changed = FakeStudent()
        original = changed.predict
        def change(*args, **kwargs):
            with torch.no_grad():
                changed.policy.weight.add_(1)
            return original(*args, **kwargs)
        changed.predict = change
        report, _, _, _ = self.collect(model=changed)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stop_reason'], 'parameter_change')
        self.assertEqual(report['samples'], 3)

    def test_noise_with_no_alternative_is_not_retried_and_bad_legacy_student_never_opens_env(self):
        model = FakeStudent()
        model.observation_space = spaces.Box(-1, 1, (94, 26, 26), np.float32)
        report, _, account, _ = self.collect(model=model)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(account.attempted, 0)
        self.assertEqual(self.instances, [])
        def only_teacher(pool, config_index, account, deadline):
            env = FakeEnv(account, config_index, length=1)
            mask = np.zeros(10, bool); mask[2] = True
            env.action_masks = lambda: mask.copy()
            return env
        self.factory = only_teacher
        report, path, _, _ = self.collect(model=None, episodes=1, noise_probability=1.)
        self.assertEqual(report['status'], 'completed', report.get('error'))
        row = json.loads((path / 'decisions.jsonl').read_text().strip())
        self.assertTrue(row['noise_requested'])
        self.assertFalse(row['noise_applied'])
        self.assertEqual(row['noise_miss_reason'], 'no_legal_movement_alternative')
        self.assertEqual(row['teacher_action'], row['executed_action'])


if __name__ == '__main__':
    unittest.main()
