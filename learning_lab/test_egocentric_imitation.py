"""Tiny CPU fixtures only: no ROM/native interactions and no CUDA initialization."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from learning_lab import egocentric_imitation as ego
from learning_lab import imitation as frozen
from learning_lab.ego_encoder import EgocentricGridCNN, center_observations


class FixtureDataset:
    specifications = {}

    def __init__(self, path):
        self.spec = self.specifications[str(Path(path).resolve())]
        self.episode_ids = np.array([0, 0, 1, 1, 2, 2], np.int64)
        self.metadata = {"format": "tiny-egocentric-cpu-test", "environment_interactions": 0}

    def __len__(self):
        return 6

    def batch(self, indices):
        indices = np.asarray(indices, np.int64)
        observations = np.zeros((len(indices), 94, 26, 26), np.float32)
        observations[:, 52, 10:12, 8:10] = 1
        observations[:, 51, 24:26, 12:14] = 1
        observations[:, 67] = .3
        observations[:, 71] = 1
        masks = np.zeros((len(indices), 10), bool)
        masks[:, [0, 2]] = True
        if self.spec.get("illegal"):
            masks[:, 2] = False
        return dict(observations=observations, actions=np.full(len(indices), 2, np.int64),
                    action_masks=masks, episode_ids=self.episode_ids[indices].copy())

    def close(self):
        pass


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_both_frames_use_current_centroid_and_preserve_relative_motion(self):
        observation = torch.zeros(1, 94, 26, 26)
        observation[0, 52, 8, 10] = 1
        observation[0, 5, 8, 8] = 1
        observation[0, 8, 6, 13] = .5
        observation[0, 55, 6, 13] = .75
        centered, scalars = center_observations(observation)
        self.assertEqual(tuple(centered.shape), (1, 94, 25, 25))
        self.assertEqual(centered[0, 52, 12, 12].item(), 1)
        self.assertEqual(centered[0, 5, 12, 10].item(), 1)
        self.assertEqual(centered[0, 8, 10, 15].item(), .5)
        self.assertEqual(centered[0, 55, 10, 15].item(), .75)
        torch.testing.assert_close(scalars[0, :2], torch.tensor([2 * 10 / 25 - 1, 2 * 8 / 25 - 1]))

    def test_integer_translation_preserves_centered_local_geometry(self):
        a, b = torch.zeros(1, 94, 26, 26), torch.zeros(1, 94, 26, 26)
        a[0, 52, 8:10, 10:12] = 1
        a[0, 55, 6, 13] = 1
        b[0, 52, 11:13, 14:16] = 1
        b[0, 55, 9, 17] = 1
        ac, _ = center_observations(a)
        bc, _ = center_observations(b)
        torch.testing.assert_close(ac, bc, rtol=0, atol=0)

    def test_dead_player_falls_back_previous_then_arena_center(self):
        observation = torch.zeros(2, 94, 26, 26)
        observation[0, 5, 4, 7] = 1
        centered, scalars = center_observations(observation)
        self.assertTrue(torch.isfinite(centered).all())
        self.assertTrue(torch.isfinite(scalars).all())
        self.assertEqual(centered[0, 5, 12, 12].item(), 1)
        torch.testing.assert_close(scalars[1, :2], torch.zeros(2))

    def test_zero_reset_prefix_and_outside_padding_stay_zero(self):
        observation = torch.zeros(1, 94, 26, 26)
        observation[:, 52, 0, 0] = 1
        observation[:, 47, :, :] = 1
        centered, _ = center_observations(observation)
        self.assertTrue(torch.all(centered[:, :47] == 0))
        self.assertTrue(torch.all(centered[:, :, :12, :] == 0))
        self.assertTrue(torch.all(centered[:, :, :, :12] == 0))
        self.assertEqual(centered[0, 47, 12, 12].item(), 1)

    def test_observed_fire_ready_and_lives_are_the_only_extra_scalars(self):
        observation = torch.zeros(1, 94, 26, 26)
        observation[:, 71] = 1
        observation[:, 67] = .4
        _, scalars = center_observations(observation)
        self.assertEqual(tuple(scalars.shape), (1, 4))
        torch.testing.assert_close(scalars[0, 2:], torch.tensor([1., .4]))

    def test_encoder_shape_input_only_geometry_and_trainable_branch_gradients(self):
        space = gym.spaces.Box(-1, 1, (94, 26, 26), np.float32)
        encoder = EgocentricGridCNN(space)
        observation = torch.rand(2, 94, 26, 26, requires_grad=True)
        centered, scalars = center_observations(observation)
        self.assertFalse(centered.requires_grad)
        self.assertFalse(scalars.requires_grad)
        features = encoder(observation)
        self.assertEqual(tuple(features.shape), (2, 256))
        features.sum().backward()
        self.assertIsNone(observation.grad)
        for branch in (encoder.local_cnn, encoder.global_cnn):
            convolution = next(m for m in branch.modules() if isinstance(m, nn.Conv2d))
            self.assertIsNotNone(convolution.weight.grad)
            self.assertGreater(convolution.weight.grad.abs().sum().item(), 0)
        self.assertFalse(any(isinstance(m, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d))
                             for m in encoder.modules()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_rejects_noncanonical_channel_count(self):
        with self.assertRaises(ValueError):
            center_observations(torch.zeros(1, 47, 26, 26))
        with self.assertRaises(ValueError):
            EgocentricGridCNN(gym.spaces.Box(-1, 1, (47, 26, 26), np.float32))

    def test_distant_information_remains_available_to_original_map_branch(self):
        encoder = EgocentricGridCNN(gym.spaces.Box(-1, 1, (94, 26, 26), np.float32))
        a = torch.zeros(1, 94, 26, 26)
        a[0, 52, 3, 3] = 1
        b = a.clone()
        b[0, 55, 24, 24] = 1
        ac, _ = center_observations(a)
        bc, _ = center_observations(b)
        torch.testing.assert_close(ac, bc, rtol=0, atol=0)
        with torch.no_grad():
            self.assertFalse(torch.equal(encoder.global_cnn(a), encoder.global_cnn(b)))
            self.assertFalse(torch.equal(encoder(a), encoder(b)))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path("runs") / ("egocentric-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        fake = types.ModuleType("learning_lab.demonstrations")
        fake.DemonstrationDataset = FixtureDataset
        name = "learning_lab.demonstrations"
        previous = sys.modules.get(name)
        sys.modules[name] = fake

        def restore_dataset():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

        self.addCleanup(restore_dataset)

    def dataset(self, **spec):
        path = self.root / "fixture"
        path.mkdir()
        (path / "fixture.json").write_text(json.dumps(spec))
        FixtureDataset.specifications[str(path.resolve())] = spec
        self.addCleanup(FixtureDataset.specifications.pop, str(path.resolve()))
        return path

    def train(self, **kwargs):
        paths = kwargs.pop("dataset_paths", None)
        if paths is None:
            paths = [self.dataset()]
        config = dict(epochs=2, batch_size=2, learning_rate=1e-3, seed=7, device="cpu")
        config.update(kwargs)
        return ego.train_egocentric_imitation(paths, self.root / "trained", **config)

    def test_legal_mask_loss_has_zero_illegal_gradient_and_rejects_illegal_label(self):
        logits = torch.tensor([[0., 1., 100.]], requires_grad=True)
        masks = torch.tensor([[True, True, False]])
        loss = frozen._masked_cross_entropy(logits, torch.tensor([1]), masks)
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0)
        with self.assertRaises(ValueError):
            frozen._masked_cross_entropy(logits, torch.tensor([2]), masks)

    def test_cpu_training_exact_split_checkpoint_and_optimizer_reload(self):
        report = self.train()
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertEqual(report["environment_interactions"], 0)
        self.assertEqual(report["optimizer_updates"], 4)
        self.assertEqual(report["actual_device"], "cpu")
        expected = frozen._split_episodes(
            np.array(["dataset0:0"] * 2 + ["dataset0:1"] * 2 + ["dataset0:2"] * 2), 7)[2]
        self.assertEqual(report["split"], expected)
        self.assertLess(report["best_validation_ce"], report["initial_validation_ce"])
        checkpoint = Path(report["checkpoints"]["final"]["path"])
        model = ego.load_egocentric_checkpoint(checkpoint)
        self.assertIsInstance(model.policy.features_extractor, EgocentricGridCNN)
        self.assertEqual(model.policy.net_arch, {"pi": [64, 64], "vf": [64, 64]})
        self.assertFalse(model.policy.normalize_images)
        self.assertEqual(model.num_timesteps, 0)
        self.assertTrue(model.policy.optimizer.state_dict()["state"])
        batch = FixtureDataset(self.root / "fixture").batch([0, 1])
        actions, _ = model.predict(batch["observations"], deterministic=True,
                                   action_masks=batch["action_masks"])
        self.assertTrue(batch["action_masks"][np.arange(2), actions].all())
        single, _ = model.predict(batch["observations"][0], deterministic=True,
                                  action_masks=batch["action_masks"][0])
        self.assertTrue(batch["action_masks"][0, int(single)])
        from sb3_contrib import MaskablePPO
        direct = MaskablePPO.load(checkpoint, device="cpu")
        for name, parameter in model.policy.state_dict().items():
            torch.testing.assert_close(parameter, direct.policy.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_strict_manifest_rejects_missing_changed_hash_contract_and_sources(self):
        report = self.train(epochs=1)
        checkpoint = Path(report["checkpoints"]["final"]["path"])
        sidecar = checkpoint.with_suffix(".manifest.json")
        text = sidecar.read_text()
        sidecar.unlink()
        with self.assertRaises(ValueError):
            ego.load_egocentric_checkpoint(checkpoint)
        for field, value in (("format", "tank-ai-imitation-checkpoint-v1"),
                             ("checkpoint_sha256", "0" * 64), ("contract", {}),
                             ("sources", {})):
            changed = json.loads(text)
            changed[field] = value
            sidecar.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                ego.load_egocentric_checkpoint(checkpoint)
        sidecar.write_text(text)
        ego.load_egocentric_checkpoint(checkpoint)
        with self.assertRaises(ValueError):
            frozen.load_imitation_checkpoint(checkpoint)
        self.assertEqual(json.loads(text)["checkpoint_sha256"],
                         hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_deadline_before_start_has_no_updates_or_model(self):
        report = self.train(deadline=0)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stop_reason"], "deadline")
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["environment_interactions"], 0)
        self.assertIsNone(report["checkpoints"]["final"])

    def test_deadline_after_first_update_preserves_partial_checkpoint(self):
        clock = [100.]
        original_step = torch.optim.Adam.step

        def advance_after_update(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            clock[0] = 102.
            return result

        fake_time = types.SimpleNamespace(monotonic=lambda: clock[0])
        with patch.object(frozen, "time", fake_time), patch.object(ego, "time", fake_time), \
                patch.object(torch.optim.Adam, "step", advance_after_update):
            report = self.train(deadline=101.)
        self.assertEqual(report["status"], "partial", report.get("error"))
        self.assertEqual(report["stop_reason"], "deadline")
        self.assertEqual(report["optimizer_updates"], 1)
        self.assertEqual(report["supervised_samples"], 2)
        self.assertEqual(report["epochs_completed"], 0)
        self.assertFalse(report["checkpoints"]["epochs"])
        ego.load_egocentric_checkpoint(report["checkpoints"]["final"]["path"])

    def test_bad_episode_split_in_manifest_is_not_accepted(self):
        report = self.train(epochs=1)
        checkpoint = Path(report["checkpoints"]["final"]["path"])
        sidecar = checkpoint.with_suffix(".manifest.json")
        manifest = json.loads(sidecar.read_text())
        manifest["split"]["validation_episode_ids"] = manifest["split"]["train_episode_ids"][:1]
        sidecar.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            ego.load_egocentric_checkpoint(checkpoint)

    def test_illegal_corpus_labels_fail_without_update(self):
        path = self.dataset(illegal=True)
        report = self.train(dataset_paths=[path], epochs=1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
