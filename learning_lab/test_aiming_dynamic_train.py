"""CPU/fake-environment dynamic runner tests; zero real interactions."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

import torch

from learning_lab import aiming_dynamic as dynamic
from learning_lab import aiming_dynamic_train as runner
from learning_lab.test_aiming_dynamic import FakeRaw, fake_generator


class DynamicRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path("runs") / ("dynamic-runner-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        for name, value in (("generate_aiming_starts", fake_generator), ("RustVectorEnv", FakeRaw)):
            p = patch.object(dynamic, name, value)
            p.start()
            self.addCleanup(p.stop)
        account = dynamic.DynamicAccounting(10000, deadline=time.monotonic() + 1000)
        sampler = dynamic.BalancedAimingSampler(seed=0, excluded_hashes=set(), accounting=account)
        self.prepared = dict(accounting=account, sampler=sampler, output=self.root,
                             final_test=fake_generator(128, dynamic.FINAL_TEST_SEED),
                             manifest={"format": dynamic.VERSION, "excluded_hashes_sha256": "a" * 64,
                                       "final_test_sha256": "b" * 64,
                                       "objective": "finite-victory-sparse-v1",
                                       "gamma": 1, "shaping": False})

    def test_allowance_requires_a_finite_deadline_and_no_autoextension(self):
        allowance = runner.DynamicRunAllowance(268546, time.monotonic() + 600)
        self.assertEqual(allowance.max_decisions, 268546)
        with self.assertRaises(ValueError):
            runner.DynamicRunAllowance(0, time.monotonic() + 600)
        with self.assertRaises(ValueError):
            runner.DynamicRunAllowance(268546, float("inf"))

    def test_cpu_update_counter_dynamic_callback_and_valid_saved_model(self):
        runtime = runner.DynamicRuntime(self.prepared, device="cpu", steps=4,
                                        n_envs=2, n_steps=2, batch_size=2, seed=0)
        try:
            self.assertTrue(runtime.collect())
            self.assertEqual(self.prepared["accounting"].native_steps_completed, 4)
            self.assertEqual(runtime.optimized_steps, 0)
            runtime.optimize()
            self.assertEqual(runtime.optimized_steps, 4)
            self.assertTrue(all(row["layout_hash"] for row in runtime.callback.episodes))
            self.assertIs(type(runtime.model.policy.features_extractor), runner.EgocentricGridCNN)
            # Synthetic archive accounting tag; fixtures performed no real decisions.
            runtime.optimized_steps = 4096
            provenance = {"sources": runner.source_hashes(), "environment": self.prepared["manifest"]}
            path = self.root / "fixture.zip"
            runner.save_dynamic_checkpoint(runtime, path, provenance, "cpu_fixture")
            model = runner.load_dynamic_checkpoint(path)
            self.assertEqual(model.gamma, 1.)
            self.assertTrue(model.policy.optimizer.state_dict()["state"])
            for name, value in runtime.model.policy.state_dict().items():
                torch.testing.assert_close(value, model.policy.state_dict()[name], rtol=0, atol=0)
            manifest = json.loads(path.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            manifest["environment"]["format"] = "fixed-template"
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                runner.load_dynamic_checkpoint(path)
        finally:
            runtime.close()
        self.assertFalse(torch.cuda.is_initialized())

    def test_expired_deadline_prevents_any_collection_or_update(self):
        runtime = runner.DynamicRuntime(self.prepared, device="cpu", steps=4,
                                        n_envs=2, n_steps=2, batch_size=2, seed=0)
        try:
            self.prepared["accounting"].deadline = 0
            with self.assertRaises(dynamic.DynamicStop):
                runtime.collect()
            self.assertEqual(runtime.optimized_steps, 0)
            self.assertEqual(self.prepared["accounting"].native_steps_attempted, 0)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
