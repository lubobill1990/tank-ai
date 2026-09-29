"""CPU acceptance tests for the explicit, finite Rust-model history contract."""
import importlib.util
import io
import json
import os
import unittest

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from compact_rollout import CompactMaskableRolloutBuffer
from policy import Grid26CNN


torch.set_num_threads(1)


def observation_space(stack):
    return spaces.Box(-np.inf, np.inf, (stack * 47, 26, 26), np.float32)


def tagged_frame(tag):
    frame = np.zeros((47, 26, 26), np.float32)
    frame[0, 24:26, 12:14] = 1
    frame[1, 4:8, 8:12] = np.float32(tag / 100)
    frame[2, 6:10, 16:20] = np.float32((tag % 7) / 7)
    return frame


def history_batch(tags):
    return torch.from_numpy(np.stack([
        np.concatenate([np.zeros((47, 26, 26), np.float32) if tag is None
                        else tagged_frame(tag) for tag in sequence])
        for sequence in tags
    ]))


class TaggedEpisodeEnv(gym.Env):
    """Real VecFrameStack input with distinguishable environments and episodes."""
    observation_space = observation_space(1)
    action_space = spaces.Discrete(10)

    def __init__(self, stream=0, length=2):
        self.stream, self.length = stream, length
        self.episode, self.step_count = -1, 0

    def observation(self):
        return tagged_frame(1 + self.stream * 40 + self.episode * 7 + self.step_count)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_count = 0
        return self.observation(), {}

    def step(self, action):
        self.step_count += 1
        return self.observation(), float(action == 5), False, self.step_count == self.length, {}

    def action_masks(self):
        mask = np.ones(10, bool)
        mask[0] = False
        return mask


def tagged_vec(stack=16):
    return VecFrameStack(DummyVecEnv([
        lambda: TaggedEpisodeEnv(0, 2),
        lambda: TaggedEpisodeEnv(1, 3),
    ]), n_stack=stack, channels_order="first")


class ModelContractTests(unittest.TestCase):
    def test_public_registry_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("rust_models"),
                             "The approved Rust model registry must exist")
        from rust_models import MODEL_NAMES, extractor_class, model_spec
        self.assertEqual(MODEL_NAMES, ("cnn", "spatial", "history"))
        self.assertIs(extractor_class("cnn"), Grid26CNN)
        for name, stack in zip(MODEL_NAMES, (2, 2, 16)):
            spec = model_spec(name)
            self.assertEqual(spec, dict(name=name, n_stack=stack, features_dim=256,
                                        version="rust-models-v1"))
            self.assertEqual(json.loads(json.dumps(spec)), spec)
            spec["n_stack"] = -1
            self.assertEqual(model_spec(name)["n_stack"], stack)
        for invalid in ("", "transformer", "HISTORY"):
            for lookup in (model_spec, extractor_class):
                with self.assertRaises(ValueError):
                    lookup(invalid)

    def test_feature_shapes_parameters_and_gradients(self):
        from rust_models import MODEL_NAMES, extractor_class, model_spec
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                torch.manual_seed(41)
                stack = model_spec(name)["n_stack"]
                encoder = extractor_class(name)(observation_space(stack))
                inputs = history_batch([[None] * (stack - 1) + [5], list(range(stack))])
                outputs = encoder(inputs)
                self.assertEqual(outputs.shape, (2, 256))
                self.assertTrue(torch.isfinite(outputs).all())
                outputs.square().mean().backward()
                gradients = [p.grad for p in encoder.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
                count = sum(p.numel() for p in encoder.parameters())
                self.assertGreater(count, 0)
                if name != "cnn":
                    self.assertLess(count, 300_000)

    def test_transformer_shapes_and_fixed_position_buffers(self):
        from rust_models import HistoryTransformer, SpatialTransformer
        for cls, stack, tokens in ((SpatialTransformer, 2, 169), (HistoryTransformer, 16, 16)):
            with self.subTest(model=cls.__name__):
                model = cls(observation_space(stack))
                positions = model.positions
                self.assertEqual(positions.shape, (1, tokens, 64))
                self.assertFalse(positions.requires_grad)
                self.assertIn("positions", dict(model.named_buffers()))
                self.assertNotIn("positions", dict(model.named_parameters()))
                self.assertFalse(torch.equal(positions[:, 0], positions[:, 1]))
                self.assertEqual(len(model.layers), 2)
                for layer in model.layers:
                    self.assertEqual(layer.self_attn.num_heads, 4)
                    self.assertEqual(layer.self_attn.embed_dim, 64)
                    self.assertEqual(layer.linear1.out_features, 256)
                    self.assertTrue(layer.norm_first)
                    self.assertEqual(layer.dropout.p, 0)
                    self.assertEqual(layer.self_attn.dropout, 0)
                with self.assertRaises(ValueError):
                    cls(observation_space(1))
                with self.assertRaises(ValueError):
                    cls(observation_space(stack), features_dim=128)
                with self.assertRaises(ValueError):
                    model(torch.zeros(1, stack * 47, 25, 26))
        spatial = SpatialTransformer(observation_space(2))
        self.assertEqual(spatial.stem(torch.zeros(1, 94, 26, 26)).shape, (1, 64, 13, 13))
        positions = spatial.positions.reshape(13, 13, 64)
        self.assertFalse(torch.equal(positions[0, 0], positions[1, 0]))
        self.assertFalse(torch.equal(positions[0, 0], positions[0, 1]))
        self.assertFalse(torch.equal(positions[1, 0], positions[0, 1]))


class HistoryTests(unittest.TestCase):
    def setUp(self):
        from rust_models import HistoryTransformer
        torch.manual_seed(73)
        self.encoder = HistoryTransformer(observation_space(16))

    def test_frame_pool_matches_adaptive_bins_and_input_gradients_on_cpu(self):
        pool = self.encoder.frame_projection[0]
        actual_input = torch.randn(2, 64, 13, 13, requires_grad=True)
        reference_input = actual_input.detach().clone().requires_grad_(True)
        actual = pool(actual_input)
        expected = torch.nn.functional.adaptive_avg_pool2d(reference_input, (3, 3))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        gradient = torch.randn_like(expected)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-6, atol=1e-7)

    def test_future_frames_cannot_change_earlier_tokens_or_gradients(self):
        first = history_batch([list(range(16))]).requires_grad_(True)
        changed = first.detach().clone()
        changed[:, 8 * 47:] = history_batch([[100 + i for i in range(16)]])[:, 8 * 47:]
        before = self.encoder.encode_sequence(first)
        after = self.encoder.encode_sequence(changed)
        self.assertEqual(before.shape, (1, 16, 256))
        torch.testing.assert_close(before[:, :8], after[:, :8], rtol=0, atol=0)
        self.assertFalse(torch.allclose(before[:, -1], after[:, -1]))
        before[:, 7].sum().backward()
        self.assertEqual(torch.count_nonzero(first.grad[:, 8 * 47:]).item(), 0)
        self.assertGreater(torch.count_nonzero(first.grad[:, :8 * 47]).item(), 0)

    def test_past_content_and_temporal_order_affect_latest_output(self):
        baseline = history_batch([list(range(16))])
        changed = baseline.clone()
        changed[:, :47] = torch.from_numpy(tagged_frame(300))
        reordered = baseline.reshape(1, 16, 47, 26, 26).clone()
        reordered[:, [0, 14]] = reordered[:, [14, 0]]
        with torch.no_grad():
            expected = self.encoder(baseline)
            self.assertFalse(torch.allclose(expected, self.encoder(changed), rtol=1e-5, atol=1e-7))
            self.assertFalse(torch.allclose(expected, self.encoder(reordered.reshape_as(baseline)),
                                            rtol=1e-5, atol=1e-7))
            torch.testing.assert_close(expected, self.encoder.encode_sequence(baseline)[:, -1])

    def test_padding_never_adds_features_or_gradients_and_empty_input_is_finite(self):
        inputs = history_batch([[None] * 16, [None] * 15 + [6],
                                [None] * 14 + [5, 6]]).requires_grad_(True)
        output = self.encoder.encode_sequence(inputs)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(torch.count_nonzero(output[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[1, :15]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[2, :14]).item(), 0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertEqual(torch.count_nonzero(inputs.grad[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(inputs.grad[1, :15 * 47]).item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.encoder.parameters()))
        self.assertFalse(torch.allclose(output[1, -1], output[2, -1]))

    def test_padding_positions_cannot_influence_valid_tokens(self):
        inputs = history_batch([[None] * 14 + [3, 4]])
        with torch.no_grad():
            baseline = self.encoder(inputs)
            self.encoder.positions[:, :14].add_(10000)
            torch.testing.assert_close(self.encoder(inputs), baseline, rtol=0, atol=0)

    def test_minibatch_order_other_environments_and_train_eval_do_not_change_history(self):
        inputs = history_batch([list(range(16)), [None] * 15 + [100]])
        with torch.no_grad():
            expected = self.encoder(inputs)
            torch.testing.assert_close(self.encoder(inputs.flip(0)).flip(0), expected)
            torch.testing.assert_close(self.encoder(inputs[:1]), expected[:1], rtol=1e-5, atol=1e-6)
            unrelated = history_batch([[400] * 16])
            self.encoder(unrelated)
            torch.testing.assert_close(self.encoder(inputs), expected, rtol=0, atol=0)
            self.encoder.eval()
            torch.testing.assert_close(self.encoder(inputs), expected, rtol=1e-5, atol=1e-6)

    def test_actual_vec_frame_stack_preserves_terminal_and_resets_only_finished_env(self):
        env = tagged_vec()
        try:
            initial = env.reset()
            np.testing.assert_array_equal(initial[:, :15 * 47], 0)
            first, _, dones, _ = env.step(np.array([5, 5]))
            self.assertFalse(dones.any())
            np.testing.assert_array_equal(first[:, -94:-47], initial[:, -47:])
            second, _, dones, infos = env.step(np.array([5, 5]))
            np.testing.assert_array_equal(dones, [True, False])
            terminal = infos[0]["terminal_observation"]
            np.testing.assert_array_equal(terminal[:-47], first[0, 47:])
            np.testing.assert_array_equal(second[0, :-47], 0)
            np.testing.assert_array_equal(second[1, :-47], first[1, 47:])
            expected_reset = history_batch([[None] * 15 + [8]])
            with torch.no_grad():
                encoded = self.encoder(torch.from_numpy(second))
                torch.testing.assert_close(encoded[:1], self.encoder(expected_reset),
                                           rtol=1e-5, atol=1e-6)
                terminal_encoded = self.encoder(torch.from_numpy(terminal[None]))
                self.assertFalse(torch.allclose(terminal_encoded, encoded[:1]))
                new_first, _, _, _ = env.step(np.array([5, 5]))
                np.testing.assert_array_equal(new_first[0, -94:-47], second[0, -47:])
                np.testing.assert_array_equal(new_first[1, :-47], 0)
                torch.testing.assert_close(
                    self.encoder(torch.from_numpy(new_first[:1])),
                    self.encoder(history_batch([[None] * 14 + [8, 9]])),
                    rtol=1e-5, atol=1e-6,
                )
                explicit_reset = env.reset()
                np.testing.assert_array_equal(explicit_reset[:, :-47], 0)
        finally:
            env.close()


@unittest.skipUnless(os.environ.get("TANK_AI_TEST_CUDA_MODELS") == "1",
                     "Opt in with TANK_AI_TEST_CUDA_MODELS=1 for short CUDA checks")
class DeterministicHistoryCUDATests(unittest.TestCase):
    def setUp(self):
        from rust_train import _configure
        _configure("cuda")

    def test_pool_matches_adaptive_outputs_and_cpu_reference_gradients(self):
        from rust_models import HistoryTransformer
        pool = HistoryTransformer(observation_space(16)).frame_projection[0].cuda()
        reference_input = torch.randn(2, 64, 13, 13, requires_grad=True)
        cuda_input = reference_input.detach().cuda().requires_grad_(True)
        reference = torch.nn.functional.adaptive_avg_pool2d(reference_input, (3, 3))
        actual = pool(cuda_input)
        with torch.no_grad():
            cuda_reference = torch.nn.functional.adaptive_avg_pool2d(cuda_input, (3, 3))
        torch.testing.assert_close(actual, cuda_reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-5, atol=1e-6)
        gradient = torch.randn_like(reference)
        reference.backward(gradient)
        actual.backward(gradient.cuda())
        torch.testing.assert_close(cuda_input.grad.cpu(), reference_input.grad, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_history_backward_and_optimizer_step_keep_strict_determinism(self):
        from rust_models import HistoryTransformer
        torch.manual_seed(311)
        model = HistoryTransformer(observation_space(16)).cuda().train()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 222656)
        inputs = history_batch([list(range(16)), [None] * 15 + [7]]).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        before = model.projection[0].weight.detach().clone()
        output = model(inputs)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer.step()
        torch.cuda.synchronize()
        self.assertFalse(torch.equal(model.projection[0].weight, before))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())


class PPOIntegrationTests(unittest.TestCase):
    def make_model(self, name, env, buffer_class=CompactMaskableRolloutBuffer, codec="raw"):
        from rust_models import extractor_class
        kwargs = {"channels_per_frame": 47, "max_storage_bytes": 128 * 1024**2}
        if buffer_class is CompactMaskableRolloutBuffer:
            kwargs["codec"] = codec
        return MaskablePPO(
            "CnnPolicy", env, seed=31, device="cpu", n_steps=4, batch_size=4, n_epochs=1,
            learning_rate=5e-5, rollout_buffer_class=buffer_class, rollout_buffer_kwargs=kwargs,
            policy_kwargs={"features_extractor_class": extractor_class(name),
                           "features_extractor_kwargs": {"features_dim": 256},
                           "normalize_images": False},
        )

    def test_raw_and_native_minibatches_preserve_history_bits_and_policy_outputs(self):
        from native_rollout import NativeRunRolloutBuffer
        env = tagged_vec()
        try:
            model = self.make_model("history", env)
            model.policy.set_training_mode(False)
            buffers = [
                CompactMaskableRolloutBuffer(4, env.observation_space, env.action_space,
                                            device="cpu", n_envs=2, channels_per_frame=47, codec="raw"),
                NativeRunRolloutBuffer(4, env.observation_space, env.action_space,
                                      device="cpu", n_envs=2, channels_per_frame=47),
            ]
            observations = env.reset()
            records = []
            starts = np.ones(2, bool)
            for _ in range(4):
                masks = np.array(env.env_method("action_masks"))
                with torch.no_grad():
                    actions, values, logs = model.policy(
                        torch.from_numpy(observations), deterministic=True, action_masks=masks)
                records.append((observations.copy(), actions, values, logs, masks))
                for buffer in buffers:
                    buffer.add(observations, actions.numpy(), np.zeros(2, np.float32), starts,
                               values, logs, action_masks=masks)
                observations, _, starts, _ = env.step(actions.numpy())
            expected_obs = np.stack([r[0] for r in records]).swapaxes(0, 1).reshape(
                8, 752, 26, 26)
            expected_values = torch.stack([r[2] for r in records]).transpose(0, 1).reshape(8)
            expected_logs = torch.stack([r[3] for r in records]).transpose(0, 1).reshape(8)
            for buffer in buffers:
                buffer.compute_returns_and_advantage(torch.zeros(2), starts)
                np.random.seed(61)
                indices = np.random.permutation(8)
                np.random.seed(61)
                for offset, sample in zip(range(0, 8, 3), buffer.get(3)):
                    rows = indices[offset:offset + 3]
                    self.assertEqual(sample.observations.numpy().tobytes(), expected_obs[rows].tobytes())
                    with torch.no_grad():
                        values, logs, _ = model.policy.evaluate_actions(
                            sample.observations, sample.actions.flatten().long(),
                            action_masks=sample.action_masks)
                    torch.testing.assert_close(values.flatten(), expected_values[rows],
                                               rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(logs, expected_logs[rows], rtol=1e-5, atol=1e-6)
        finally:
            env.close()

    def test_all_models_native_ppo_update_and_save_load(self):
        from native_rollout import NativeRunRolloutBuffer
        from rust_env import RustVectorEnv, stage_ram
        from rust_models import MODEL_NAMES, model_spec
        initial = np.stack([stage_ram(1), stage_ram(2)])
        for name in MODEL_NAMES:
            raw_parameters = None
            for cls in (CompactMaskableRolloutBuffer, NativeRunRolloutBuffer):
                with self.subTest(model=name, codec=cls.__name__):
                    env = VecFrameStack(RustVectorEnv(initial, max_frames=8, seed=19),
                                        n_stack=model_spec(name)["n_stack"], channels_order="first")
                    try:
                        model = self.make_model(name, env, cls)
                        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
                        model.learn(total_timesteps=16)
                        self.assertEqual(model.num_timesteps, 16)
                        self.assertEqual(model._n_updates, 2)
                        self.assertTrue(any(not torch.equal(before[key], value)
                                            for key, value in model.policy.state_dict().items()))
                        if raw_parameters is None:
                            raw_parameters = {key: value.clone()
                                              for key, value in model.policy.state_dict().items()}
                        else:
                            for key, value in model.policy.state_dict().items():
                                torch.testing.assert_close(value, raw_parameters[key], rtol=0, atol=0)
                        self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in model.policy.parameters()))
                        obs = env.reset()
                        masks = np.array(env.env_method("action_masks"))
                        actions, _ = model.predict(obs, deterministic=True, action_masks=masks)
                        self.assertTrue(masks[np.arange(2), actions].all())
                        checkpoint = io.BytesIO()
                        model.save(checkpoint)
                        checkpoint.seek(0)
                        loaded = MaskablePPO.load(checkpoint, env=env, device="cpu")
                        reloaded, _ = loaded.predict(obs, deterministic=True, action_masks=masks)
                        np.testing.assert_array_equal(actions, reloaded)
                        with torch.no_grad():
                            for a, b in zip(
                                model.policy.evaluate_actions(torch.from_numpy(obs),
                                                              torch.from_numpy(actions), action_masks=masks),
                                loaded.policy.evaluate_actions(torch.from_numpy(obs),
                                                               torch.from_numpy(actions), action_masks=masks),
                            ):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
                    finally:
                        env.close()


if __name__ == "__main__":
    unittest.main()
