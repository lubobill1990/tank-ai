"""CPU-only fixtures for the two-candidate comparison; no real interactions."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from learning_lab import aiming_compare as compare
from learning_lab.test_aiming_train import FakeVec


def bundle(seed, count, marker):
    rams = np.zeros((count, 2048), np.uint8)
    rams[:, 0] = np.arange(count)
    rams[:, 1] = marker
    return dict(rams=rams, hashes=np.array([hashlib.sha256(r.tobytes()).hexdigest() for r in rams]),
                metadata={"generator_seed": seed})


class AimingCompareTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path("runs") / ("aiming-compare-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def test_new_holdout_excludes_training_and_old_development_identities(self):
        train, old, new = bundle(311, 32, 1), bundle(947, 128, 2), bundle(999, 128, 3)
        compare.validate_new_holdout(train, old, new)
        new["rams"][0] = old["rams"][0]
        new["hashes"][0] = old["hashes"][0]
        with self.assertRaises(ValueError):
            compare.validate_new_holdout(train, old, new)

    def test_same_settings_only_representation_and_length_differ(self):
        cnn, ego = compare.experiment_contract("cnn"), compare.experiment_contract("egocentric")
        self.assertEqual(cnn["ppo"], ego["ppo"])
        self.assertEqual(cnn["ppo"], compare.base.PPO)
        self.assertEqual(cnn["training_steps"], 262144)
        self.assertEqual(ego["training_steps"], 262144)
        self.assertEqual(cnn["new_holdout_generator_seed"], 999)
        self.assertFalse(cnn["full_game"])
        self.assertNotEqual(cnn["extractor"], ego["extractor"])

    def test_both_real_extractors_cpu_updates_and_strict_reload(self):
        starts = {"rams": np.zeros((2, 2048), np.uint8), "directions": np.array([0, 1])}
        guard = SimpleNamespace(check=lambda phase: None)
        for architecture in ("cnn", "egocentric"):
            directory = self.root / architecture
            directory.mkdir()
            with patch.object(compare, "make_aiming_env", return_value=FakeVec()):
                runtime = compare.CompareRuntime(
                    starts, architecture=architecture, guard=guard, device="cpu", output=directory,
                    target_steps=4, n_steps=2, batch_size=2)
                try:
                    self.assertEqual(runtime.objective_version, compare.base.OBJECTIVE)
                    self.assertTrue(np.all(runtime.model._last_obs[:, :47] == 0))
                    self.assertTrue(runtime.collect())
                    runtime.optimize()
                    self.assertEqual(runtime.optimized_steps, 4)
                    self.assertIs(type(runtime.model.policy.features_extractor),
                                  compare.EXTRACTORS[architecture])
                    runtime.optimized_steps = 4096
                    provenance = dict(sources=compare.source_hashes(), datasets={
                        name: "a" * 64 for name in compare.DATASET_KEYS})
                    record = compare.save_checkpoint(runtime, directory / "fixture.zip",
                                                      provenance, "cpu_fixture")
                    model = compare.load_compare_checkpoint(record["path"])
                    self.assertIs(type(model.policy.features_extractor), compare.EXTRACTORS[architecture])
                    for name, parameter in runtime.model.policy.state_dict().items():
                        torch.testing.assert_close(parameter, model.policy.state_dict()[name], rtol=0, atol=0)
                    sidecar = Path(record["path"]).with_suffix(".manifest.json")
                    manifest = json.loads(sidecar.read_text())
                    manifest["architecture"] = "egocentric" if architecture == "cnn" else "cnn"
                    sidecar.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        compare.load_compare_checkpoint(record["path"])
                finally:
                    runtime.close()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
