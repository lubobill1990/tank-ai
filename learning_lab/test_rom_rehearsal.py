"""Tiny CPU rehearsal/transaction tests; no real environment interactions."""
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
from learning_lab import rom_rehearsal as rehearsal
from learning_lab.demonstrations import DemonstrationWriter
from learning_lab.collect_teacher_v2 import _seal
from learning_lab.test_rom_finetune import FakeBudget, FakeROM

class RehearsalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('rehearsal-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.source_hashes = control._sources()
        cls.anchor_path = cls.root / 'anchors'
        writer = DemonstrationWriter(cls.anchor_path)
        reports = []
        try:
            for episode, count in enumerate((1159, 1206, 3)):
                frame = np.full((47, 26, 26), .1 if episode < 2 else .9, np.float32)
                masks = np.zeros(10, bool); masks[[0, 2, 9]] = True
                for index in range(count):
                    label = 2 if episode < 2 else 9
                    writer.add(frame, label, masks, episode_start=index == 0, executed_action=label)
                reports.append({'episode_id': episode, 'stage': 1, 'seed': episode if episode < 2 else 39,
                                'outcome': 'clear', 'clear': True, 'episode_finished': True,
                                'decisions': count, 'attempted_decisions': count, 'completed_decisions': count,
                                'perturbation_probability': 0., 'changed_actions': 0, 'error': None,
                                'terminal_remaining_enemies': 0, 'terminal_base_destroyed': False})
            writer.finish(reports, {'teacher_class': 'learning_lab.teacher_v2.TeacherControllerV2',
                                   'learned_model': False, 'development_only': True,
                                   'provenance': 'Normal original-ROM reset; synthetic fixture, no real emulator'})
        finally:
            writer.close()
        _seal(cls.anchor_path)
        tiny = DemonstrationWriter(cls.root / 'tiny')
        try:
            for _ in range(3):
                for step in range(2):
                    tiny.add(np.full((47, 26, 26), .1, np.float32), 0, np.ones(10, bool),
                             episode_start=step == 0, executed_action=0)
            tiny.finish([{'episode_id': i, 'outcome': 'timeout'} for i in range(3)], {'label': 'synthetic'})
        finally:
            tiny.close()
        report = imitation.train_imitation([cls.root / 'tiny'], cls.root / 'parent', epochs=1, batch_size=2)
        if report['status'] != 'completed':
            raise AssertionError(report)
        cls.parent = Path(report['checkpoints']['final']['path'])
        def check_sources():
            assert control._sources() == cls.source_hashes
        cls.addClassCleanup(check_sources)

    def anchors(self):
        anchors = rehearsal.AnchorDataset(self.anchor_path)
        self.addCleanup(anchors.close)
        return anchors

    def runtime(self, guard=None):
        output = self.root / uuid.uuid4().hex; output.mkdir()
        budget = FakeBudget()
        guard = guard or SimpleNamespace(deadline=time.monotonic() + 1000, check=lambda phase: None)
        meter = control.BootstrapMeter(budget, 'replay-' + uuid.uuid4().hex, guard)
        meter.begin_initial()
        model = imitation.load_imitation_checkpoint(self.parent)
        runtime = rehearsal.RehearsalRuntime(
            model, output, guard, meter, anchors=self.anchors(), seed=7, device='cpu', max_steps=8,
            env_factory=FakeROM, settings={**control.PPO, 'n_steps': 2, 'batch_size': 4},
            replay_batch_size=4)
        self.addCleanup(runtime.close)
        meter.flush_success()
        return runtime, budget, meter, output

    def test_anchor_selection_excludes_every_unselected_episode(self):
        anchors = self.anchors()
        self.assertEqual(len(anchors.indices), 2365)
        rng = np.random.default_rng(17)
        selected = anchors.sample_indices(rng, 1000)
        self.assertTrue(np.all(selected < 2365))
        batch = anchors.batch(selected)
        self.assertEqual(set(batch['episode_ids']), {0, 1})
        self.assertTrue(np.all(batch['actions'] == 2))
        self.assertEqual(anchors.identity['selected_episode_ids'], [0, 1])
        self.assertEqual(anchors.identity['selected_episode_counts'], [1159, 1206])
        with self.assertRaises(ValueError):
            anchors.batch(np.array([2365]))

    def test_replay_has_actor_gradient_legal_masks_and_no_environment_calls(self):
        anchors = self.anchors()
        model = imitation.load_imitation_checkpoint(self.parent)
        before = model.policy.action_net.weight.detach().clone()
        illegal = model.policy.action_net.weight[[1, 3, 4, 5, 6, 7, 8]].detach().clone()
        progress = {'bc_updates': 0, 'supervised_samples': 0, 'losses': []}
        result = rehearsal.rehearse(
            model, anchors, np.random.default_rng(9), SimpleNamespace(check=lambda phase: None),
            progress, batch_size=4, minibatches=4)
        self.assertFalse(torch.equal(before, model.policy.action_net.weight))
        # Adam may retain BC momentum for these rows, so legality is asserted on
        # the fresh loss gradient rather than assuming optimizer state is zero.
        batch = anchors.batch(np.array([0, 1]))
        model.policy.optimizer.zero_grad(set_to_none=True)
        loss, _ = imitation._batch_loss(model, batch)
        loss.backward()
        torch.testing.assert_close(model.policy.action_net.weight.grad[[1, 3, 4, 5, 6, 7, 8]],
                                   torch.zeros_like(illegal))
        self.assertEqual(result['bc_updates'], 4)
        self.assertEqual(result['supervised_samples'], 16)
        self.assertEqual(model.num_timesteps, 0)
        self.assertTrue(all(p.requires_grad for p in model.policy.parameters()))
        self.assertTrue(all(group['lr'] == 5e-5 for group in model.policy.optimizer.param_groups))

    def test_ppo_and_replay_commit_separate_exact_counters(self):
        runtime, budget, meter, _ = self.runtime()
        control._collect_reserved(runtime, budget, meter, 'combined-fixture')
        before_env = runtime.completed_steps
        runtime.optimize()
        self.assertEqual(runtime.completed_steps, before_env)
        self.assertEqual(runtime.optimized_steps, 8)
        self.assertEqual(runtime.bc_updates, 4)
        self.assertEqual(runtime.supervised_samples, 16)
        self.assertEqual(runtime.adam_steps, runtime.ppo_adam_steps + 4)
        self.assertTrue(runtime.state_verified)
        self.assertFalse(torch.cuda.is_initialized())

    def test_interrupted_replay_rolls_back_successful_ppo_and_replay_adam(self):
        guard = SimpleNamespace(deadline=time.monotonic() + 1000, check=lambda phase: None)
        runtime, budget, meter, _ = self.runtime(guard)
        control._collect_reserved(runtime, budget, meter, 'interrupted-combination')
        before = control._state_hash(runtime.export_warm())
        rng = deepcopy(runtime.replay_rng.bit_generator.state)
        def stop(phase):
            if phase == 'rehearsal_minibatch' and runtime.pending_replay['bc_updates'] == 1:
                raise control.DeadlineReached('interrupt second replay minibatch')
        guard.check = stop
        with self.assertRaises(control.DeadlineReached):
            runtime.optimize()
        self.assertEqual(before, control._state_hash(runtime.export_warm()))
        self.assertEqual(runtime.replay_rng.bit_generator.state, rng)
        self.assertEqual(runtime.optimized_steps, 0)
        self.assertEqual(runtime.bc_updates, 0)
        self.assertEqual(runtime.supervised_samples, 0)
        self.assertEqual(runtime.adam_steps, 0)
        self.assertEqual(runtime.ppo_adam_steps, 0)
        self.assertEqual(runtime.discarded_replay['bc_updates'], 1)
        self.assertTrue(runtime.state_verified)

    def test_checkpoint_requires_anchor_identity_and_combined_commit_contract(self):
        runtime, _, _, output = self.runtime()
        checkpoint = rehearsal._save_checkpoint(
            runtime, output / 'initial.zip', control._parent_identity(self.parent), 'initial')
        loaded = rehearsal.load_rehearsal_checkpoint(checkpoint['path'])
        self.assertEqual(control._state_hash(runtime.model.policy.state_dict()),
                         control._state_hash(loaded.policy.state_dict()))
        self.assertEqual(control._state_hash(runtime.model.policy.optimizer.state_dict()),
                         control._state_hash(loaded.policy.optimizer.state_dict()))
        self.assertTrue(all(p.requires_grad for p in loaded.policy.parameters()))
        path = Path(checkpoint['path']); sidecar = path.with_suffix('.manifest.json')
        original = json.loads(sidecar.read_text())
        for change in ('episodes', 'weight', 'counts', 'source', 'anchor_hash'):
            value = deepcopy(original)
            if change == 'episodes': value['anchors']['selected_episode_ids'] = [0, 39]
            elif change == 'weight': value['contract']['rehearsal']['loss_weight'] = .1
            elif change == 'counts': value['bc_updates'] = 1
            elif change == 'source': value['sources'] = {}
            else: value['anchors']['dataset']['sha256'] = '0' * 64
            sidecar.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                rehearsal.load_rehearsal_checkpoint(path)
        with self.assertRaises(ValueError):
            control.load_rom_finetune_checkpoint(path)

    def test_nonfinite_replay_rolls_back_the_preceding_ppo_update(self):
        runtime, budget, meter, _ = self.runtime()
        control._collect_reserved(runtime, budget, meter, 'nonfinite-combination')
        original = runtime.anchors
        class CorruptBatch:
            identity = original.identity
            def sample_indices(self, rng, size):
                return original.sample_indices(rng, size)
            def batch(self, indices):
                batch = original.batch(indices)
                batch['observations'][0, 0, 0, 0] = np.nan
                return batch
        runtime.anchors = CorruptBatch()
        before = control._state_hash(runtime.export_warm())
        with self.assertRaises((FloatingPointError, ValueError)):
            runtime.optimize()
        self.assertEqual(before, control._state_hash(runtime.export_warm()))
        self.assertEqual(runtime.optimized_steps, 0)
        self.assertEqual(runtime.bc_updates, 0)
        self.assertEqual(runtime.ppo_adam_steps, 0)
        self.assertEqual(runtime.adam_steps, 0)

    def test_perturbed_or_unsealed_anchors_are_rejected(self):
        path = self.root / 'bad-anchors'
        shutil.copytree(self.anchor_path, path)
        reports_path = path / 'episode_reports.json'
        reports = json.loads(reports_path.read_text())
        reports[0]['perturbation_probability'] = .02
        reports_path.write_text(json.dumps(reports))
        with self.assertRaises(ValueError):
            rehearsal.AnchorDataset(path)
        _seal(path)
        with self.assertRaisesRegex(ValueError, 'clean'):
            rehearsal.AnchorDataset(path)

    def test_expired_public_driver_never_starts_an_environment(self):
        class ExpiredBudget(FakeBudget):
            def snapshot(self):
                result = super().snapshot()
                result['remaining_seconds'] = 0
                return result
        budget = ExpiredBudget()
        report = rehearsal.run_rehearsal_finetune(
            self.parent, self.anchor_path, self.root / 'expired-driver', budget, steps=1024)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['optimized_steps'], 0)
        self.assertEqual(report['bc_updates'], 0)
        self.assertEqual(budget.rows, [])


if __name__ == '__main__':
    unittest.main()
