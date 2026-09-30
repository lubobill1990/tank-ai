"""Tiny synthetic CPU PPO fixtures; no ROM, GPU, or research-budget work."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from learning_lab import imitation
from learning_lab import rom_finetune as finetune
from learning_lab.demonstrations import DemonstrationWriter

class FakeBudget:
    def __init__(self):
        self.rows = []

    def snapshot(self):
        charged = sum(row['reserved'] if row['actual'] is None else row['actual'] for row in self.rows)
        return {'max_interactions': 1000000, 'remaining_interactions': 1000000 - charged,
                'remaining_seconds': 1000., 'deadline_epoch': time.time() + 1000,
                'reservations': deepcopy(self.rows)}

    def reserve(self, label, interactions):
        token = str(len(self.rows))
        self.rows.append({'id': token, 'label': label, 'reserved': interactions, 'actual': None})
        return token

    def settle(self, token, interactions):
        row = self.rows[int(token)]
        if not 0 <= interactions <= row['reserved']:
            raise ValueError('overrun')
        row['actual'] = interactions


class FakeROM(gym.Env):
    observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = spaces.Discrete(10)
    render_mode = None

    def __init__(self, spec, *, max_frames, deadline, initialization_counter):
        if not spec['full_start'] or spec['source_seed'] not in (0, 1):
            raise AssertionError('Training opening pool leaked')
        self.source_seed = spec['source_seed']
        self.initial_remaining = 20
        self.current_kills = 0
        self.counter = initialization_counter
        self.requested_seeds = []
        self.calls = 0

    def reset(self, *, seed=None, options=None):
        self.requested_seeds.append(seed)
        self.counter({'kind': 'normal_reset_substep', 'phase': 'normal_reset', 'count': 1})
        self.current_kills, self.calls = 0, 0
        return np.zeros((47, 26, 26), np.float32), {}

    def step(self, action):
        self.calls += 1
        done = self.calls == 2
        self.current_kills = 20 if done else 0
        return np.full((47, 26, 26), .1, np.float32), -.002, done, False, {
            'outcome': 'clear' if done else '', 'kills': self.current_kills,
            'source_seed': self.source_seed, 'frames': self.calls * 4,
            'normal_training_start': True, 'curriculum_start': False}

    def action_masks(self):
        return np.ones(10, bool)


class RomFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('rom-finetune-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        data = cls.root / 'data'
        writer = DemonstrationWriter(data)
        try:
            for episode in range(3):
                for index in range(2):
                    writer.add(np.zeros((47, 26, 26), np.float32), 0, np.ones(10, bool),
                               episode_start=index == 0, executed_action=0)
            writer.finish([{'episode_id': i, 'outcome': 'timeout'} for i in range(3)],
                          {'label': 'synthetic-fixture'})
        finally:
            writer.close()
        report = imitation.train_imitation([data], cls.root / 'parent', epochs=1, batch_size=2)
        if report['status'] != 'completed':
            raise AssertionError(report)
        cls.parent = Path(report['checkpoints']['final']['path'])

    def runtime(self, guard=None):
        output = self.root / uuid.uuid4().hex
        output.mkdir()
        budget = FakeBudget()
        guard = guard or SimpleNamespace(deadline=time.monotonic() + 1000, check=lambda phase: None)
        meter = finetune.BootstrapMeter(budget, 'fixture-' + uuid.uuid4().hex, guard)
        meter.begin_initial()
        warm = imitation.load_imitation_checkpoint(self.parent)
        runtime = finetune._Runtime(warm, output, guard, meter, seed=37, device='cpu',
                                    max_steps=8, env_factory=FakeROM,
                                    settings={**finetune.PPO, 'n_steps': 2, 'batch_size': 4})
        self.addCleanup(runtime.close)
        meter.flush_success()
        return runtime, warm, budget, meter, output

    def test_policy_and_adam_identity_at_start_and_fixed_seed_pool(self):
        runtime, parent, budget, _, _ = self.runtime()
        self.assertEqual(runtime.opening_seeds, [0, 1, 0, 1])
        self.assertEqual(runtime.raw.get_attr('source_seed'), [0, 1, 0, 1])
        for name, value in parent.policy.state_dict().items():
            torch.testing.assert_close(runtime.model.policy.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(finetune._state_hash(runtime.model.policy.optimizer.state_dict()),
                         finetune._state_hash(parent.policy.optimizer.state_dict()))
        self.assertEqual(runtime.model.gamma, 1.)
        self.assertEqual(runtime.model.gae_lambda, .95)
        self.assertEqual(runtime.model.clip_range(1.), .1)
        self.assertEqual(runtime.model.ent_coef, .001)
        self.assertEqual(runtime.model.target_kl, .01)
        self.assertEqual(runtime.model.learning_rate, 5e-5)
        self.assertEqual(budget.rows[0]['actual'], 4)

    def test_compact_rollout_sparse_rewards_episode_stats_and_budget(self):
        runtime, _, budget, meter, output = self.runtime()
        result = finetune._collect_reserved(runtime, budget, meter, 'rollout-fixture')
        self.assertTrue(result['complete'])
        self.assertEqual(result['completed_decisions'], 8)
        self.assertEqual(budget.rows[1]['reserved'], 8)
        self.assertEqual(budget.rows[1]['actual'], 8)
        runtime.optimize()
        self.assertEqual(runtime.optimized_steps, 8)
        self.assertGreater(runtime.adam_steps, 0)
        self.assertEqual(runtime.model.rollout_buffer.__class__.__name__, '_GuardedBuffer')
        rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row['episode']['r'] == 1 for row in rows))
        self.assertTrue(all(row['TimeLimit.truncated'] is False for row in rows))
        self.assertFalse(torch.cuda.is_initialized())

    def test_partial_optimizer_update_rolls_back_weights_adam_and_counters(self):
        holder = {}
        def check(phase):
            if phase == 'adam_step' and holder['runtime']._step_count['count']:
                raise finetune.DeadlineReached('fixture deadline')
        guard = SimpleNamespace(deadline=time.monotonic() + 1000, check=lambda phase: None)
        runtime, _, budget, meter, _ = self.runtime(guard)
        holder['runtime'] = runtime
        finetune._collect_reserved(runtime, budget, meter, 'rollback-fixture')
        before = runtime.export_warm()
        guard.check = check
        with self.assertRaises(finetune.DeadlineReached):
            runtime.optimize()
        after = runtime.export_warm()
        self.assertEqual(finetune._state_hash(before), finetune._state_hash(after))
        self.assertEqual(runtime.optimized_steps, 0)
        self.assertEqual(runtime._step_count['count'], 0)
        self.assertTrue(runtime.state_verified)

    def test_exceptional_rollout_and_failed_evaluation_keep_reservations_charged(self):
        class FailedRuntime:
            rollout_size = 1024
            completed_steps = 0
            def collect(self):
                self.completed_steps = 4
                raise RuntimeError('partial vector failure')
        budget = FakeBudget()
        guard = SimpleNamespace(check=lambda phase: None)
        meter = finetune.BootstrapMeter(budget, 'init', guard)
        row = finetune._collect_reserved(FailedRuntime(), budget, meter, 'failed')
        self.assertFalse(row['settled'])
        self.assertIsNone(budget.rows[0]['actual'])
        proxy = finetune.DeferredEvaluationBudget(budget)
        token = proxy.reserve('evaluation', 12000)
        proxy.settle(token, 17)
        self.assertIsNone(budget.rows[-1]['actual'])
        proxy.commit()
        self.assertEqual(budget.rows[-1]['actual'], 17)

    def test_development_selection_uses_wins_then_kills_and_two_completed_regressions(self):
        initial = {'status': 'completed', 'train_wins': 2, 'wins': 2, 'mean_kills': 12.}
        gate = finetune.DevelopmentSelection(initial)
        incomplete = {'status': 'partial', 'train_wins': 0, 'wins': 0, 'mean_kills': 0.}
        self.assertEqual(gate.consider(incomplete), (False, False))
        worse = {'status': 'completed', 'train_wins': 1, 'wins': 1, 'mean_kills': 20.}
        self.assertEqual(gate.consider(worse), (False, False))
        better = {'status': 'completed', 'train_wins': 2, 'wins': 3, 'mean_kills': 10.}
        self.assertEqual(gate.consider(better), (True, False))
        self.assertEqual(gate.consider(worse), (False, True))

    def test_strict_initial_checkpoint_roundtrip_and_identity_rejections(self):
        runtime, _, _, _, output = self.runtime()
        parent = finetune._parent_identity(self.parent)
        checkpoint = finetune._save_checkpoint(runtime, output / 'initial.zip', parent, 'initial')
        loaded = finetune.strict_load_rom_finetune_checkpoint(checkpoint['path'])
        self.assertEqual(finetune._state_hash(runtime.model.policy.state_dict()),
                         finetune._state_hash(loaded.policy.state_dict()))
        self.assertEqual(finetune._state_hash(runtime.model.policy.optimizer.state_dict()),
                         finetune._state_hash(loaded.policy.optimizer.state_dict()))
        self.assertEqual(loaded.gamma, 1.)
        self.assertEqual(loaded.ent_coef, .001)
        path = Path(checkpoint['path']); sidecar = path.with_suffix('.manifest.json')
        original = json.loads(sidecar.read_text())
        for change in ('source', 'gamma', 'parent', 'archive'):
            manifest = deepcopy(original)
            if change == 'source': manifest['sources'] = {}
            elif change == 'gamma': manifest['contract']['ppo']['gamma'] = .99
            elif change == 'parent': manifest['parent']['sha256'] = 'bad'
            else: manifest['checkpoint_sha256'] = '0' * 64
            sidecar.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                finetune.strict_load_rom_finetune_checkpoint(path)
        with self.assertRaises(ValueError):
            imitation.load_imitation_checkpoint(path)

    def test_public_driver_stops_on_two_real_dev_regressions_and_keeps_initial_best(self):
        class SimulatedRuntime:
            rollout_size = 1024
            def __init__(self, warm, output, guard, bootstrap, **kwargs):
                from stable_baselines3.common.logger import configure
                self.model = warm
                self.source_pin = finetune._sources()
                for key, value in finetune.PPO.items():
                    if key != 'clip_range':
                        setattr(self.model, key, value)
                self.model.clip_range = lambda _: .1
                self.model.set_logger(configure(str(output / 'fixture-logs'), []))
                self.optimized_steps = self.adam_steps = self.completed_steps = 0
                self.state_verified = True
                self.start_policy_sha256 = finetune._state_hash(warm.policy.state_dict())
                self.start_adam_sha256 = finetune._state_hash(warm.policy.optimizer.state_dict())
            def collect(self):
                self.completed_steps += 1024
                self.model.num_timesteps = self.completed_steps
                return True
            def optimize(self):
                self.optimized_steps += 1024
                self.adam_steps += 1
                self.model._n_updates += 4
            def close(self):
                self.model.logger.close()
        def development(model, output, budget, guard, checkpoint, index):
            wins = [2, 1, 0][index]
            return {'status': 'completed', 'train_wins': wins, 'wins': wins, 'mean_kills': 12.,
                    'reservations': [], 'bootstrap_reservations': [], 'episodes': []}
        output = self.root / 'driver'
        with patch.object(finetune, '_Runtime', SimulatedRuntime), \
                patch.object(finetune, '_evaluate', development):
            result = finetune.run_rom_finetune(self.parent, output, FakeBudget())
        self.assertEqual(result['optimized_steps'], 32768)
        self.assertEqual(result['stop_reason'], 'two_completed_train_win_regressions')
        self.assertEqual(len(result['rollouts']), 32)
        self.assertEqual(len(result['evaluations']), 3)
        self.assertEqual(Path(result['best_checkpoint']['path']).name, 'initial.zip')
        self.assertTrue(all(row['settled'] for row in result['rollouts']))
        self.assertEqual(len([row for row in result['checkpoints'] if 'optimized-' in row['path']]), 4)
        self.assertEqual(result['unresolved_reservations'], [])

    def test_expired_global_budget_stops_public_driver_before_bootstrap(self):
        class ExpiredBudget(FakeBudget):
            def snapshot(self):
                state = super().snapshot()
                state['remaining_seconds'] = 0
                return state
        budget = ExpiredBudget()
        report = finetune.run_rom_finetune(self.parent, self.root / 'expired', budget, steps=1024)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['optimized_steps'], 0)
        self.assertEqual(budget.rows, [])

    def test_development_adapter_checks_seed_pool_settlement_and_global_deadline(self):
        budget = FakeBudget()
        guard = SimpleNamespace(deadline=time.monotonic() + 10, deadline_epoch=time.time() + 10,
                                check=lambda phase: None)
        seen = []
        def play(model, *, output, budget, stage, seed, **kwargs):
            self.assertEqual(stage, 1)
            self.assertLessEqual(budget.snapshot()['remaining_seconds'], 10)
            seen.append(seed)
            token = budget.reserve(f'fixture-dev-{seed}', 12000)
            budget.settle(token, 3)
            return {'verified_full_clear': seed in (0, 1), 'decision_attempts': 3,
                    'episode': {'kills': 20 if seed in (0, 1) else 5,
                                'outcome': 'clear' if seed in (0, 1) else 'base_lost'}}
        with patch.object(finetune, 'play_development_episode', play):
            report = finetune._evaluate(None, self.root / 'dev-adapter', budget, guard,
                                        {'path': 'synthetic.zip'}, 0)
        self.assertEqual(seen, [0, 1, 7, 39])
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['train_wins'], 2)
        self.assertEqual(report['wins'], 2)
        self.assertEqual(report['mean_kills'], 12.5)
        self.assertTrue(all(row['actual'] is not None for row in budget.rows))
        meter = finetune.BootstrapMeter(budget, 'forbidden-teacher', guard)
        with self.assertRaises(ValueError):
            meter.consume({'kind': 'teacher_decision', 'count': 1})


if __name__ == '__main__':
    unittest.main()
