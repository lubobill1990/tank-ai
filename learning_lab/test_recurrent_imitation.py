"""Tiny CPU-only causal/TBPTT contracts; never construct a gameplay environment."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch

from learning_lab import imitation as base
from learning_lab import recurrent_imitation as recurrent
from learning_lab.demonstrations import DemonstrationWriter


class RecurrentImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path('runs') / ('recurrent-imitation-test-' + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.data = cls.root / 'corpus'
        writer = DemonstrationWriter(cls.data)
        masks = np.zeros(10, bool)
        masks[[0, 2]] = True
        try:
            for episode in range(5):
                for step in range(3):
                    frame = np.zeros((47, 26, 26), np.float32)
                    frame[4, 8:16, 8:16] = 1
                    frame[0, 0, 0] = episode / 10 + step / 100
                    writer.add(frame, 2, masks, episode_start=step == 0, executed_action=2)
            writer.finish([{'episode_id': i, 'outcome': 'timeout', 'decisions': 3}
                           for i in range(5)], {'label': 'synthetic-recurrent-fixture'})
        finally:
            writer.close()
        cls.frozen = base._source_hashes()
        cls.frozen.update({str(path): base._sha256(path) for path in (
            Path('learning_lab') / 'demonstrations.py',
            Path('learning_lab') / 'weighted_imitation.py',
            Path('learning_lab') / 'rom_player.py')})
        cls.report = recurrent.train_recurrent_imitation(
            [cls.data], cls.root / 'trained', epochs=2, sequence_length=2,
            learning_rate=1e-3, seed=0)
        if cls.report['status'] != 'completed':
            raise AssertionError(cls.report)
        cls.checkpoint = Path(cls.report['checkpoints']['final']['path'])

        def check_frozen():
            for name, digest in cls.frozen.items():
                assert base._sha256(Path(*name.replace('\\', '/').split('/'))) == digest

        cls.addClassCleanup(check_frozen)

    def output(self, name='output'):
        return self.root / (name + '-' + uuid.uuid4().hex)

    @staticmethod
    def observation(prefix=False, marker=0.2):
        observation = np.zeros((94, 26, 26), np.float32)
        if not prefix:
            observation[:47] = 0.1
        observation[51, 8:16, 8:16] = 1
        observation[47, 0, 0] = marker
        return observation

    @staticmethod
    def progress():
        return {'supervised_samples': 0, 'optimizer_updates': 0, 'validation_samples': 0,
                'training_episodes_completed': 0, 'validation_episodes_completed': 0}

    def test_lstm_is_causal_and_earlier_logits_cannot_read_future_frames(self):
        torch.manual_seed(13)
        policy = recurrent.RecurrentActor().eval()
        frames = torch.randn(4, 47, 26, 26) * 0.1
        changed = frames.clone()
        changed[2:] = 0.9
        with torch.no_grad():
            original, _ = policy(frames)
            altered, _ = policy(changed)
        torch.testing.assert_close(original[:2], altered[:2], rtol=0, atol=1e-6)
        self.assertIsInstance(policy.lstm, torch.nn.LSTM)
        self.assertEqual(policy.lstm.hidden_size, 128)
        self.assertFalse(policy.lstm.bidirectional)
        self.assertEqual(policy.lstm.num_layers, 1)
        self.assertEqual(original.shape, (4, 10))
        frames.requires_grad_()
        logits, _ = policy(frames)
        logits[0].sum().backward()
        torch.testing.assert_close(frames.grad[1:], torch.zeros_like(frames.grad[1:]))

    def test_chunked_forward_matches_whole_sequence_and_detachment_cuts_graph(self):
        policy = recurrent.RecurrentActor().eval()
        frames = torch.randn(4, 47, 26, 26) * 0.1
        whole, state = policy(frames)
        first, hidden = policy(frames[:2])
        self.assertTrue(all(item.grad_fn is not None for item in hidden))
        detached = recurrent.detach_state(hidden)
        self.assertTrue(all(item.grad_fn is None and not item.requires_grad for item in detached))
        second, final = policy(frames[2:], detached)
        torch.testing.assert_close(torch.cat((first, second)), whole, rtol=1e-5, atol=1e-6)
        for a, b in zip(final, state):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        self.assertIsNone(recurrent.detach_state(None))

    def test_predict_uses_current_frame_and_resets_on_prefix_or_explicit_reset(self):
        wrapper = recurrent.load_recurrent_checkpoint(self.checkpoint)
        masks = np.ones(10, bool)
        current = self.observation()
        prefix = self.observation(prefix=True)
        wrapper.predict(current, deterministic=True, action_masks=masks)
        previous_hidden = tuple(item.clone() for item in wrapper._hidden)
        wrapper.predict(current, deterministic=True, action_masks=masks)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(previous_hidden, wrapper._hidden)))
        action, state = wrapper.predict(prefix, deterministic=True, action_masks=masks)
        expected_hidden = tuple(item.clone() for item in wrapper._hidden)
        self.assertIsNone(state)
        wrapper.reset()
        other_previous = current.copy()
        other_previous[:47] = 0.8
        repeated, _ = wrapper.predict(other_previous, deterministic=True, action_masks=masks)
        self.assertEqual(int(action), int(repeated))
        for actual, expected in zip(wrapper._hidden, expected_hidden):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(item.grad_fn is None for item in wrapper._hidden))

    def test_nonzero_prefix_keeps_memory_even_when_player_channels_are_zero(self):
        wrapper = recurrent.load_recurrent_checkpoint(self.checkpoint)
        observation = self.observation()
        masks = np.ones(10, bool)
        wrapper.predict(observation, deterministic=True, action_masks=masks)
        hidden = tuple(item.clone() for item in wrapper._hidden)
        with torch.no_grad():
            _, expected = wrapper.policy(torch.from_numpy(observation[47:])[None], hidden)
        wrapper.predict(observation, deterministic=True, action_masks=masks)
        for actual, desired in zip(wrapper._hidden, expected):
            torch.testing.assert_close(actual, desired, rtol=0, atol=0)

    def test_single_observation_masking_never_selects_illegal_actions(self):
        wrapper = recurrent.load_recurrent_checkpoint(self.checkpoint)
        masks = np.zeros(10, bool)
        masks[7] = True
        for deterministic in (True, False):
            action, _ = wrapper.predict(self.observation(prefix=True),
                                        deterministic=deterministic, action_masks=masks)
            self.assertEqual(int(action), 7)
        action, _ = wrapper.predict(self.observation(prefix=True)[None],
                                    deterministic=True, action_masks=masks[None])
        np.testing.assert_array_equal(action, [7])
        before = tuple(item.clone() for item in wrapper._hidden)
        for invalid in (np.zeros(10, bool), np.ones(10, np.float32), np.ones(9, bool)):
            with self.assertRaises(ValueError):
                wrapper.predict(self.observation(), deterministic=True, action_masks=invalid)
        for actual, expected in zip(wrapper._hidden, before):
            torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, 'single'):
            wrapper.predict(np.stack([self.observation()] * 2),
                            deterministic=True, action_masks=np.ones((2, 10), bool))

    def test_tbptt_is_chronological_resets_between_episodes_and_detaches_between_chunks(self):
        policy = recurrent.RecurrentActor()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        illegal_rows = [1, 3, 4, 5, 6, 7, 8, 9]
        untouched_actor = policy.actor.weight[illegal_rows].detach().clone()
        corpus = base._Corpus([self.data.resolve()], self.output().resolve(), None)
        self.addCleanup(corpus.close)
        episodes = recurrent._episode_indices(corpus, ['dataset0:3', 'dataset0:0'])
        hidden_inputs, frame_order = [], []

        def lstm_input(module, inputs):
            state = inputs[1]
            hidden_inputs.append(None if state is None else
                                 [(item.requires_grad, item.grad_fn is not None) for item in state])

        lstm_hook = policy.lstm.register_forward_pre_hook(lstm_input)
        cnn_hook = policy.extractor.register_forward_pre_hook(
            lambda module, inputs: frame_order.extend(inputs[0][:, 0, 0, 0].detach().cpu().tolist()))
        self.addCleanup(lstm_hook.remove)
        self.addCleanup(cnn_hook.remove)
        progress = self.progress()
        recurrent._run_episodes(policy, corpus, episodes, 2, None, None, progress, optimizer=optimizer)
        np.testing.assert_allclose(frame_order, [.30, .31, .32, .00, .01, .02], atol=1e-7)
        self.assertEqual(hidden_inputs, [None, [(False, False)] * 2, None, [(False, False)] * 2])
        self.assertEqual(progress['optimizer_updates'], 4)
        self.assertEqual(progress['supervised_samples'], 6)
        self.assertEqual(progress['training_episodes_completed'], 2)
        torch.testing.assert_close(policy.actor.weight[illegal_rows], untouched_actor, rtol=0, atol=0)

    def test_training_accounting_episode_split_and_full_checkpoint_reload(self):
        report = self.report
        self.assertEqual(report['supervised_samples'], 24)
        self.assertEqual(report['optimizer_updates'], 16)
        self.assertEqual(report['validation_samples'], 9)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertEqual(report['split']['validation_episode_ids'], ['dataset0:2'])
        self.assertEqual(len(report['checkpoints']['epochs']), 2)
        for epoch in report['history']:
            self.assertEqual(set(epoch['train_episode_order']),
                             set(report['split']['train_episode_ids']))
        self.assertLess(report['best_validation_ce'], report['initial_validation_ce'])
        wrapper = recurrent.load_recurrent_checkpoint(self.checkpoint)
        self.assertTrue(wrapper.learned_model)
        self.assertEqual(wrapper.observation_space.shape, (94, 26, 26))
        self.assertEqual(wrapper.action_space.n, 10)
        self.assertIsInstance(wrapper.policy, torch.nn.Module)
        self.assertEqual(str(wrapper.device), 'cpu')
        self.assertEqual(report['resource_monitor']['status'], 'enabled')
        self.assertTrue(Path(report['resource_monitor']['path']).is_file())
        self.assertTrue(wrapper.optimizer.state_dict()['state'])
        self.assertIsNone(wrapper._hidden)
        self.assertFalse(torch.cuda.is_initialized())
        manifest = json.loads(self.checkpoint.with_suffix('.manifest.json').read_text())
        self.assertEqual(manifest['contract']['recurrent_input_shape'], [47, 26, 26])
        self.assertFalse(manifest['contract']['bidirectional'])
        self.assertFalse(manifest['hidden_state_saved'])
        self.assertEqual(manifest['checkpoint_sha256'], base._sha256(self.checkpoint))
        self.assertIn('learning_lab/recurrent_imitation.py', manifest['sources'])

    def test_loader_rejects_altered_contract_sources_integrity_and_parameter_hash(self):
        path = self.output('copy').with_suffix('.pt')
        shutil.copyfile(self.checkpoint, path)
        source = json.loads(self.checkpoint.with_suffix('.manifest.json').read_text())
        source['checkpoint_file'] = path.name
        sidecar = path.with_suffix('.manifest.json')
        for change in ('future', 'sources', 'dataset', 'split', 'weights'):
            with self.subTest(change=change):
                manifest = deepcopy(source)
                if change == 'future':
                    manifest['contract']['bidirectional'] = True
                elif change == 'sources':
                    manifest['sources']['learning_lab/recurrent_imitation.py'] = '0' * 64
                elif change == 'dataset':
                    manifest['datasets'][0]['sha256'] = '0' * 64
                elif change == 'split':
                    manifest['split']['validation_episode_ids'] = manifest['split']['train_episode_ids']
                else:
                    manifest['policy_sha256'] = '0' * 64
                sidecar.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    recurrent.load_recurrent_checkpoint(path)
        sidecar.write_text(json.dumps(source))
        with path.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            recurrent.load_recurrent_checkpoint(path)
        with self.assertRaises(ValueError):
            base.load_imitation_checkpoint(self.checkpoint)

    def test_expired_deadline_is_partial_without_optimizer_or_environment_work(self):
        output = self.output()
        report = recurrent.train_recurrent_imitation([self.data], output, deadline=0)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['stop_reason'], 'deadline')
        self.assertEqual(report['supervised_samples'], 0)
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertEqual(json.loads((output / 'report.json').read_text()), report)

    def test_chunk_deadline_preserves_exact_partial_update_counts(self):
        policy = recurrent.RecurrentActor()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        corpus = base._Corpus([self.data.resolve()], self.output().resolve(), None)
        self.addCleanup(corpus.close)
        episodes = recurrent._episode_indices(corpus, ['dataset0:0'])
        progress = self.progress()

        def check_deadline(deadline):
            if progress['optimizer_updates']:
                raise base._DeadlineReached('Stop after one completed chunk')

        with self.assertRaises(base._DeadlineReached):
            recurrent._run_episodes(policy, corpus, episodes, 2, None, None, progress,
                                    optimizer=optimizer, deadline_check=check_deadline)
        self.assertEqual(progress['supervised_samples'], 2)
        self.assertEqual(progress['optimizer_updates'], 1)
        self.assertEqual(progress['training_episodes_completed'], 0)

    def test_validation_keeps_memory_across_chunks_and_starts_each_episode_from_zero(self):
        wrapper = recurrent.load_recurrent_checkpoint(self.checkpoint)
        corpus = base._Corpus([self.data.resolve()], self.output().resolve(), None)
        self.addCleanup(corpus.close)
        episodes = recurrent._episode_indices(corpus, ['dataset0:2'])
        chunked = recurrent._run_episodes(wrapper.policy, corpus, episodes, 1,
                                         None, None, self.progress())
        whole = recurrent._run_episodes(wrapper.policy, corpus, episodes, 64,
                                       None, None, self.progress())
        self.assertAlmostEqual(chunked['ce'], whole['ce'], places=6)
        self.assertEqual(chunked['accuracy'], whole['accuracy'])
        self.assertEqual(chunked['samples'], 3)

    def test_failed_data_contract_is_explicit_and_outputs_cannot_be_reused(self):
        output = self.output()
        report = recurrent.train_recurrent_imitation([self.root / 'missing-corpus'], output)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Dataset', report['error'])
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['environment_interactions'], 0)
        self.assertEqual(json.loads((output / 'report.json').read_text()), report)
        with self.assertRaises(FileExistsError):
            recurrent.train_recurrent_imitation([self.data], output)


if __name__ == '__main__':
    unittest.main()
