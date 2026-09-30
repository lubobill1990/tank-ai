"""CPU-only freeze ablation fixtures; no ROM/native interactions."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
import uuid

import numpy as np
import torch

from learning_lab import imitation, rom_finetune as control
from learning_lab import rom_frozen_features as frozen
from learning_lab.demonstrations import DemonstrationWriter
from learning_lab.test_rom_finetune import FakeBudget, FakeROM

class FrozenFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('frozen-feature-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.old_sources = control._sources()
        writer = DemonstrationWriter(cls.root / 'data')
        try:
            for _ in range(3):
                for step in range(2):
                    writer.add(np.full((47, 26, 26), .1, np.float32), 0,
                               np.ones(10, bool), episode_start=step == 0, executed_action=0)
            writer.finish([{'episode_id': i, 'outcome': 'timeout'} for i in range(3)],
                          {'label': 'synthetic-freeze-fixture'})
        finally:
            writer.close()
        report = imitation.train_imitation([cls.root / 'data'], cls.root / 'parent',
                                           epochs=1, batch_size=2)
        if report['status'] != 'completed':
            raise AssertionError(report)
        cls.parent = Path(report['checkpoints']['final']['path'])
        def verify_frozen_sources():
            assert control._sources() == cls.old_sources
        cls.addClassCleanup(verify_frozen_sources)

    def model(self):
        return imitation.load_imitation_checkpoint(self.parent)

    def runtime(self, guard=None):
        output = self.root / uuid.uuid4().hex
        output.mkdir()
        guard = guard or SimpleNamespace(deadline=time.monotonic() + 1000, check=lambda phase: None)
        budget = FakeBudget()
        meter = control.BootstrapMeter(budget, 'freeze-' + uuid.uuid4().hex, guard)
        meter.begin_initial()
        parent = self.model()
        runtime = frozen.FrozenRuntime(
            parent, output, guard, meter, seed=0, device='cpu', max_steps=8,
            env_factory=FakeROM, settings={**control.PPO, 'n_steps': 2, 'batch_size': 4})
        self.addCleanup(runtime.close)
        meter.flush_success()
        return runtime, parent, budget, meter, output

    def test_freeze_preserves_initial_policy_adam_and_only_changes_feature_grad_flags(self):
        model = self.model()
        policy_hash = control._state_hash(model.policy.state_dict())
        adam_hash = control._state_hash(model.policy.optimizer.state_dict())
        guard = frozen.FrozenFeatureGuard(model)
        self.addCleanup(guard.close)
        self.assertEqual(policy_hash, control._state_hash(model.policy.state_dict()))
        self.assertEqual(adam_hash, control._state_hash(model.policy.optimizer.state_dict()))
        self.assertIs(model.policy.features_extractor, model.policy.pi_features_extractor)
        self.assertIs(model.policy.features_extractor, model.policy.vf_features_extractor)
        frozen_ids = {id(p) for p in model.policy.features_extractor.parameters()}
        for parameter in model.policy.parameters():
            self.assertEqual(parameter.requires_grad, id(parameter) not in frozen_ids)
        self.assertEqual(frozen._contract()['ppo'], control._contract()['ppo'])
        self.assertEqual(frozen._contract()['opening_seeds'], [0, 1, 0, 1])

    def test_value_only_loss_cannot_change_features_even_with_stale_gradients_and_decay(self):
        model = self.model()
        guard = frozen.FrozenFeatureGuard(model)
        self.addCleanup(guard.close)
        before_features = control._state_hash(frozen.feature_state(model))
        before_adam = [deepcopy(model.policy.optimizer.state[p])
                       for p in model.policy.features_extractor.parameters()]
        self.assertTrue(any(torch.count_nonzero(state['exp_avg']) for state in before_adam))
        before_value = model.policy.value_net.weight.detach().clone()
        before_actor = model.policy.action_net.weight.detach().clone()
        model.policy.optimizer.param_groups[0]['weight_decay'] = .2
        for _ in range(3):
            model.policy.optimizer.zero_grad(set_to_none=True)
            values = model.policy.predict_values(torch.full((2, 94, 26, 26), .1))
            (values - 3).square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.policy.features_extractor.parameters()))
            for parameter in model.policy.features_extractor.parameters():
                parameter.grad = torch.ones_like(parameter)
            model.policy.optimizer.step()
            guard.check()
            self.assertEqual(before_features, control._state_hash(frozen.feature_state(model)))
        self.assertFalse(torch.equal(before_value, model.policy.value_net.weight))
        torch.testing.assert_close(before_actor, model.policy.action_net.weight, rtol=0, atol=0)
        after_adam = [model.policy.optimizer.state[p] for p in model.policy.features_extractor.parameters()]
        self.assertEqual(control._state_hash(before_adam), control._state_hash(after_adam))

    def test_actor_and_critic_heads_can_update_while_all_feature_aliases_stay_bit_exact(self):
        model = self.model()
        guard = frozen.FrozenFeatureGuard(model)
        self.addCleanup(guard.close)
        features = control._state_hash(frozen.feature_state(model))
        actor = model.policy.action_net.weight.detach().clone()
        critic = model.policy.value_net.weight.detach().clone()
        observations = torch.full((2, 94, 26, 26), .2)
        model.policy.optimizer.zero_grad(set_to_none=True)
        _, log_prob, _ = model.policy.evaluate_actions(observations, torch.tensor([2, 2]),
                                                     action_masks=np.ones((2, 10), bool))
        loss = -log_prob.mean() + model.policy.predict_values(observations).sub(2).square().mean()
        loss.backward()
        model.policy.optimizer.step()
        self.assertFalse(torch.equal(actor, model.policy.action_net.weight))
        self.assertFalse(torch.equal(critic, model.policy.value_net.weight))
        self.assertEqual(features, control._state_hash(frozen.feature_state(model)))

    def test_real_tiny_cpu_ppo_retains_frozen_features_and_old_protocol(self):
        runtime, parent, budget, meter, _ = self.runtime()
        self.assertEqual(runtime.start_policy_sha256, control._state_hash(parent.policy.state_dict()))
        self.assertEqual(runtime.start_adam_sha256, control._state_hash(parent.policy.optimizer.state_dict()))
        frozen_hash = runtime.freeze_guard.initial_sha256
        row = control._collect_reserved(runtime, budget, meter, 'tiny-frozen-rollout')
        self.assertTrue(row['complete'])
        runtime.optimize()
        self.assertEqual(runtime.optimized_steps, 8)
        self.assertGreater(runtime.adam_steps, 0)
        self.assertEqual(frozen_hash, control._state_hash(frozen.feature_state(runtime.model)))
        self.assertEqual(runtime.opening_seeds, [0, 1, 0, 1])
        self.assertFalse(torch.cuda.is_initialized())

    def test_rollback_and_frozen_flags_survive_an_interrupted_update(self):
        guard = SimpleNamespace(deadline=time.monotonic() + 1000, check=lambda phase: None)
        runtime, _, budget, meter, _ = self.runtime(guard)
        control._collect_reserved(runtime, budget, meter, 'interrupted-frozen-rollout')
        previous = control._state_hash(runtime.export_warm())
        def stop(phase):
            if phase == 'adam_step' and runtime._step_count['count'] > 0:
                raise control.DeadlineReached('synthetic interrupted update')
        guard.check = stop
        with self.assertRaises(control.DeadlineReached):
            runtime.optimize()
        self.assertEqual(previous, control._state_hash(runtime.export_warm()))
        self.assertEqual(runtime.optimized_steps, 0)
        runtime.freeze_guard.check()
        self.assertTrue(all(not p.requires_grad for p in runtime.model.policy.features_extractor.parameters()))

    def test_strict_checkpoint_restores_freeze_flags_hooks_and_rejects_wrong_contract(self):
        runtime, _, _, _, output = self.runtime()
        identity = control._parent_identity(self.parent)
        checkpoint = frozen._save_checkpoint(runtime, output / 'initial.zip', identity, 'initial')
        loaded = frozen.load_frozen_feature_checkpoint(checkpoint['path'])
        self.assertEqual(control._state_hash(loaded.policy.state_dict()),
                         control._state_hash(runtime.model.policy.state_dict()))
        self.assertEqual(control._state_hash(loaded.policy.optimizer.state_dict()),
                         control._state_hash(runtime.model.policy.optimizer.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in loaded.policy.features_extractor.parameters()))
        before = control._state_hash(frozen.feature_state(loaded))
        for p in loaded.policy.features_extractor.parameters():
            p.grad = torch.ones_like(p)
        loaded.policy.optimizer.param_groups[0]['weight_decay'] = 1.
        loaded.policy.optimizer.step()
        self.assertEqual(before, control._state_hash(frozen.feature_state(loaded)))
        path = Path(checkpoint['path']); sidecar = path.with_suffix('.manifest.json')
        original = json.loads(sidecar.read_text())
        for change in ('freeze', 'source', 'features'):
            value = deepcopy(original)
            if change == 'freeze': value['freeze_policy']['requires_grad'] = True
            elif change == 'source': value['sources'] = {}
            else: value['initial_feature_sha256'] = '0' * 64
            sidecar.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                frozen.load_frozen_feature_checkpoint(path)
        with self.assertRaises(ValueError):
            control.load_rom_finetune_checkpoint(path)

    def test_guard_detects_bit_changes_even_when_floats_compare_equal(self):
        model = self.model()
        parameter = next(model.policy.features_extractor.parameters())
        with torch.no_grad():
            parameter.reshape(-1)[0] = 0.
        guard = frozen.FrozenFeatureGuard(model)
        self.addCleanup(guard.close)
        with torch.no_grad():
            parameter.reshape(-1)[0] = -0.
        self.assertEqual(float(parameter.reshape(-1)[0]), 0.)
        with self.assertRaisesRegex(RuntimeError, 'Frozen'):
            guard.check()

    def test_public_driver_respects_expired_budget_without_environment_work(self):
        class ExpiredBudget(FakeBudget):
            def snapshot(self):
                result = super().snapshot()
                result['remaining_seconds'] = 0.
                return result
        budget = ExpiredBudget()
        report = frozen.run_frozen_feature_finetune(
            self.parent, self.root / 'expired-driver', budget, steps=1024)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['optimized_steps'], 0)
        self.assertEqual(budget.rows, [])
        self.assertEqual(report['contract']['freeze_policy'], frozen.FREEZE_POLICY)


if __name__ == '__main__':
    unittest.main()
