"""Bounded-run contracts; fixtures never substitute for real gameplay evidence."""
import copy
import json
import shutil
import unittest
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import ablation as lab


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f".ablation-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)


def reports(wins=0, skill=0.0):
    full = {"modes": {mode: {
        "summary": {"wins": wins, "episodes": 2, "win_rate": wins / 2},
        "episodes": [
            {"objective": "full", "outcome": "clear" if i < wins else "timeout",
             "stage": 1, "seed": seed}
            for i, seed in enumerate((101, 202))
        ],
    } for mode in ("deterministic", "sampled")}}
    skills = {"modes": {mode: {
        "summary": {"mean_success_rate": skill}, "per_skill": {},
        "episodes": [],
    } for mode in ("deterministic", "sampled")}}
    return {"full_stage": full, "skills": skills}


class ConfigTests(unittest.TestCase):
    def test_exact_arms_change_one_factor_and_share_budgets(self):
        arms = lab.ARMS
        self.assertEqual(tuple(arms), ("baseline", "warm", "horizon", "sparse"))
        baseline = asdict(arms["baseline"])
        for name, field in (("warm", "initialization"), ("horizon", "gamma"),
                            ("sparse", "reward_profile")):
            self.assertEqual([k for k in baseline if baseline[k] != asdict(arms[name])[k]], [field])
        tasks = lab.trial_configs()
        self.assertEqual(len(tasks), 8)
        self.assertEqual(sum(t.steps_per_trial for t in tasks), 400000)
        self.assertEqual({t.actual_step_cap for t in tasks}, {50176})
        self.assertEqual({t.evaluation_steps for t in tasks}, {(0, 25088, 50176)})
        self.assertEqual({t.seed for t in tasks}, {0, 1})
        self.assertEqual(lab.SELECTION_SEEDS, (101, 202))
        self.assertEqual(lab.CONFIRMATION_SEEDS, (1001, 2002))
        self.assertFalse(set(lab.SELECTION_SEEDS) & set(lab.CONFIRMATION_SEEDS))

    def test_invalid_configuration_cannot_expand_approval(self):
        config = lab.TrialConfig()
        for changes in (
            {"steps_per_trial": 50001}, {"steps_per_trial": 0}, {"seed": 2},
            {"max_wall_seconds": float("nan")}, {"max_wall_seconds": 0},
            {"patience": 0}, {"eval_every": 0}, {"batch_size": 65},
            {"n_envs": 3}, {"nstack": 3}, {"arm": "other"}, {"device": "gpu"},
            {"n_steps": True}, {"n_steps": 512},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes).validate()
        for seeds in ((0, 0), (), (0, 1, 2)):
            with self.assertRaises(ValueError):
                lab.trial_configs(seeds=seeds)
        with self.assertRaises(ValueError):
            lab.validate_parallel(3)

    def test_rounding_never_accumulates_extra_halfway_rollouts(self):
        c = lab.TrialConfig(steps_per_trial=16, eval_every=7, n_steps=4, batch_size=8)
        self.assertEqual(c.validate().evaluation_steps, (0, 8, 16))
        c = replace(c, steps_per_trial=15, eval_every=5)
        self.assertEqual(c.evaluation_steps, (0, 8, 16))


class StopAndMetricTests(unittest.TestCase):
    def test_zero_baseline_valid_patience_and_lexicographic_wins(self):
        gate = lab.StopGate(patience=3, max_wall_seconds=10, started=0)
        self.assertTrue(gate.observe((0, 0.0)))
        self.assertIsNone(gate.reason(steps=0, cap=512, now=1))
        self.assertFalse(gate.observe((0, 0.0)))
        self.assertTrue(gate.observe((0, 0.5)))
        self.assertTrue(gate.observe((1, 0.0)))
        self.assertFalse(gate.observe((0, 1.0)))
        self.assertFalse(gate.observe((1, 0.0)))
        self.assertFalse(gate.observe((1, 0.0)))
        self.assertEqual(gate.reason(steps=0, cap=512, now=2), "patience")
        self.assertEqual(gate.reason(steps=512, cap=512, now=2), "step_cap")
        self.assertEqual(gate.reason(steps=0, cap=512, now=10), "wall_clock")

    def test_metric_keeps_true_wins_and_skills_separate(self):
        report = reports(wins=1, skill=0.75)
        report["full_stage"]["modes"]["deterministic"]["summary"]["mean_reward"] = 999999
        self.assertEqual(lab.selection_metric(report), (1, 0.75))
        report["full_stage"]["modes"]["deterministic"]["episodes"][0]["objective"] = "combat"
        with self.assertRaises(ValueError):
            lab.selection_metric(report)

    def test_nonfinite_rejected_in_all_metrics_not_only_selection(self):
        report = reports()
        report["skills"]["modes"]["sampled"]["summary"]["extra"] = float("nan")
        with self.assertRaises(lab.TrialStop) as caught:
            lab.selection_metric(report)
        self.assertEqual(caught.exception.reason, "nonfinite")


class AtomicJsonTests(WorkspaceTest):
    def test_transient_replace_permission_error_retries_and_publishes(self):
        target = self.root / "batch.json"
        lab.write_json(target, {"status": "old"})
        real_replace = lab.os.replace
        attempts = []

        def replace_after_reader_releases(source, destination):
            attempts.append((source, destination))
            if len(attempts) <= 2:
                self.assertEqual(lab.read_json(target), {"status": "old"})
                raise PermissionError("reader is holding target")
            return real_replace(source, destination)

        with patch.object(lab.os, "replace", side_effect=replace_after_reader_releases), \
                patch.object(lab.time, "sleep") as sleep, \
                self.assertWarnsRegex(RuntimeWarning, "retrying") as warning:
            lab.write_json(target, {"status": "new"}, replace=True)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(warning.warnings), 1)
        self.assertIn(str(target), str(warning.warning))
        self.assertEqual(lab.read_json(target), {"status": "new"})
        self.assertFalse((self.root / ".batch.json.writing").exists())

    def test_persistent_replace_permission_error_preserves_target_and_cleans_staging(self):
        target = self.root / "batch.json"
        lab.write_json(target, {"status": "old"})
        original = target.read_bytes()
        error = PermissionError("persistent sharing violation")
        with patch.object(lab.os, "replace", side_effect=error) as replace_file, \
                patch.object(lab.time, "sleep") as sleep, \
                self.assertWarnsRegex(RuntimeWarning, "retrying") as warning:
            with self.assertRaises(PermissionError) as caught:
                lab.write_json(target, {"status": "new"}, replace=True)
        self.assertIs(caught.exception, error)
        self.assertEqual(replace_file.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 0.51)
        self.assertEqual(len(warning.warnings), 1)
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse((self.root / ".batch.json.writing").exists())

    def test_non_permission_replace_error_propagates_without_retry(self):
        target = self.root / "batch.json"
        lab.write_json(target, {"status": "old"})
        with patch.object(lab.os, "replace", side_effect=OSError("disk failure")) as replace_file, \
                patch.object(lab.time, "sleep") as sleep:
            with self.assertRaisesRegex(OSError, "disk failure"):
                lab.write_json(target, {"status": "new"}, replace=True)
        self.assertEqual(replace_file.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(lab.read_json(target), {"status": "old"})
        self.assertFalse((self.root / ".batch.json.writing").exists())

    @unittest.skipUnless(lab.os.name == "nt", "Windows file sharing semantics")
    def test_real_windows_reader_releases_without_delete_sharing(self):
        import ctypes
        import threading
        from ctypes import wintypes

        target = self.root / "batch.json"
        lab.write_json(target, {"status": "old"})
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create_file.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        # GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE (no FILE_SHARE_DELETE).
        handle = create_file(str(target), 0x80000000, 0x3, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        blocked = threading.Event()
        released = threading.Event()
        close_errors = []

        def release_reader():
            blocked.wait(5)
            if not close_handle(handle):
                close_errors.append(ctypes.get_last_error())
            released.set()

        reader = threading.Thread(target=release_reader)
        reader.start()
        real_replace = lab.os.replace
        errors = []

        def observed_replace(source, destination):
            try:
                return real_replace(source, destination)
            except PermissionError as exc:
                errors.append(exc.winerror)
                blocked.set()
                self.assertTrue(released.wait(5), "reader did not release its handle")
                raise

        try:
            with patch.object(lab.os, "replace", side_effect=observed_replace), \
                    self.assertWarnsRegex(RuntimeWarning, "retrying"):
                lab.write_json(target, {"status": "new"}, replace=True)
        finally:
            blocked.set()
            reader.join(timeout=5)
        self.assertFalse(reader.is_alive())
        self.assertEqual(close_errors, [])
        self.assertTrue(errors)
        self.assertTrue(all(code in (5, 32, 33) for code in errors))
        self.assertEqual(lab.read_json(target), {"status": "new"})
        self.assertFalse((self.root / ".batch.json.writing").exists())


class PersistenceTests(WorkspaceTest):
    def signature(self):
        sources = {}
        for name in ("ablation.py", "skills.py", "experiments.py"):
            path = self.root / name
            path.write_text("fixture\n")
            sources[name] = lab.file_hash(path)
        rom = self.root / "battle_city.nes"
        rom.write_bytes(b"fixture bytes, never a ROM")
        return {"source_sha256": sources, "rom_sha256": lab.file_hash(rom),
                "reward_profile": "shaped"}

    def test_freeze_detects_source_and_rom_changes_not_docs(self):
        signature = self.signature()
        lab.verify_frozen(signature, root=self.root)
        (self.root / "README.md").write_text("unsigned docs")
        lab.verify_frozen(signature, root=self.root)
        for name in ("skills.py", "ablation.py", "battle_city.nes"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "frozen runtime"):
                lab.verify_frozen(signature, root=self.root)
            path.write_bytes(original)

    def test_batch_paths_are_absolute_persisted_and_never_overwritten(self):
        signature = self.signature()
        source = self.root / "source.zip"
        source.write_bytes(b"fixture")
        signatures = {"shaped": signature, "sparse": dict(signature, reward_profile="sparse")}
        batch = lab.prepare_batch(self.root / "batch", warm_source=source,
                                  signatures=signatures)
        self.assertEqual(len(batch["trials"]), 8)
        self.assertEqual({t["status"] for t in batch["trials"]}, {"pending"})
        for task in batch["trials"]:
            for field in ("config_path", "run_dir", "log_path"):
                self.assertTrue(Path(task[field]).is_absolute())
            saved = json.loads(Path(task["config_path"]).read_text())
            self.assertEqual(saved["trial_id"], task["trial_id"])
            self.assertEqual(saved["warm_source"], str(source.resolve()))
            self.assertEqual(saved["warm_source_sha256"], lab.file_hash(source))
        with self.assertRaises(FileExistsError):
            lab.prepare_batch(self.root / "batch", warm_source=source, signatures=signatures)

    def test_lab_manifest_is_not_a_normal_training_manifest(self):
        signature = self.signature()
        config = lab.TrialConfig()
        manifest = lab.make_lab_manifest(config, signature, trial_id="baseline-seed-0")
        self.assertNotIn("schema_version", manifest)
        self.assertEqual(manifest["selection"]["stages"], [1])
        self.assertEqual(manifest["confirmation"]["stages"], [1])
        lab.validate_lab_manifest(manifest, signature)
        changed = copy.deepcopy(manifest)
        changed["selection"]["seeds"] = [1001, 2002]
        with self.assertRaises(ValueError):
            lab.validate_lab_manifest(changed, signature)
        with self.assertRaises(ValueError):
            lab.validate_lab_manifest(manifest, dict(signature, reward_profile="sparse"))

    def test_batch_rejects_arm_runtime_differences_before_creating_artifacts(self):
        signature = self.signature()
        source = self.root / "source.zip"
        source.write_bytes(b"fixture")
        sparse = copy.deepcopy(signature)
        sparse.update(reward_profile="sparse", rom_sha256="different")
        output = self.root / "invalid-batch"
        with self.assertRaises(ValueError):
            lab.prepare_batch(output, warm_source=source,
                              signatures={"shaped": signature, "sparse": sparse})
        self.assertFalse(output.exists())

    def test_manifest_rejects_checkpoint_progress_and_mismatched_trial_identity(self):
        signature = self.signature()
        manifest = lab.make_lab_manifest(lab.TrialConfig(), signature, trial_id="baseline-seed-0")
        for update in (
            {"trial_id": "sparse-seed-1"},
            {"timesteps": 1, "phase": "full", "progress": {"full": 1}},
            {"timesteps": 512, "phase": "combat", "progress": {"full": 512}},
            {"timesteps": 512, "phase": "full", "progress": {"full": 0}},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                lab.validate_lab_manifest(dict(manifest, **update), signature)

    def test_child_success_requires_matching_persistent_result(self):
        task = {"trial_id": "baseline-seed-0", "run_dir": str(self.root / "run"),
                "status": "running"}
        Path(task["run_dir"]).mkdir()
        failed = lab.finish_child(task, 7)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["returncode"], 7)
        self.assertEqual(lab.finish_child(task, 0)["status"], "failed")
        result = {"trial_id": task["trial_id"], "status": "completed", "actual_timesteps": 0,
                  "stop_reason": "step_cap"}
        (Path(task["run_dir"]) / "result.json").write_text(json.dumps(result))
        self.assertEqual(lab.finish_child(task, 0)["status"], "completed")

    def test_aggregation_preserves_failed_tasks_and_does_not_rank_zeros(self):
        result = {"trial_id": "baseline-seed-0", "status": "completed",
                  "actual_timesteps": 50176, "stop_reason": "step_cap",
                  "initial_metric": {"full_stage_wins": 0, "mean_skill_success_rate": 0.0},
                  "best_metric": {"full_stage_wins": 0, "mean_skill_success_rate": 0.0}}
        run = self.root / "run"
        run.mkdir()
        (run / "result.json").write_text(json.dumps(result))
        tasks = [{"trial_id": result["trial_id"], "status": "completed", "run_dir": str(run),
                  "arm": "baseline", "seed": 0},
                 {"trial_id": "warm-seed-0", "status": "failed",
                  "run_dir": str(self.root / "absent"), "arm": "warm", "seed": 0}]
        report = lab.aggregate_results({"trials": tasks})
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_trials"], 1)
        self.assertIn("24", " ".join(report["limitations"]))
        self.assertNotIn("winner", report)

    def test_scheduler_limits_children_and_finishes_all_eight_after_failure(self):
        signature = self.signature()
        source = self.root / "source.zip"
        source.write_bytes(b"fixture")
        batch = lab.prepare_batch(
            self.root / "batch", warm_source=source,
            signatures={"shaped": signature, "sparse": dict(signature, reward_profile="sparse")})
        children = []
        live = peak = 0

        class Child:
            def __init__(self, command, *, cwd, env, stdout, stderr):
                nonlocal live, peak
                live += 1
                peak = max(peak, live)
                self.pid, self.ticks, self.returncode = 100 + len(children), 0, None
                self.command, self.cwd, self.environment = command, cwd, env
                self.record = json.loads(Path(command[-1]).read_text())
                children.append(self)

            def poll(self):
                nonlocal live
                self.ticks += 1
                if self.ticks >= 2 and self.returncode is None:
                    live -= 1
                    self.returncode = 3 if self.record["trial_id"] == "warm-seed-0" else 0
                    run = Path(self.record["run_dir"])
                    run.mkdir()
                    (run / "result.json").write_text(json.dumps({
                        "trial_id": self.record["trial_id"],
                        "status": "completed" if self.returncode == 0 else "failed",
                        "stop_reason": "step_cap" if self.returncode == 0 else "nonfinite",
                    }))
                return self.returncode

        with patch.object(lab, "verify_frozen"), patch.object(lab.subprocess, "Popen", Child), \
                patch.object(lab.time, "sleep"):
            code = lab.run_batch(batch)
        self.assertEqual(code, 1)
        self.assertEqual(peak, 2)
        self.assertEqual(len(children), 8)
        self.assertEqual(sum(t["status"] == "failed" for t in batch["trials"]), 1)
        self.assertEqual(sum(t["status"] == "completed" for t in batch["trials"]), 7)
        for child in children:
            self.assertEqual(child.command[:3], [lab.sys.executable, str(lab.ROOT / "ablation.py"),
                                                 "--trial"])
            self.assertEqual(child.cwd, lab.ROOT)
            self.assertEqual(child.environment["OMP_NUM_THREADS"], "1")
            self.assertEqual(child.environment["MKL_NUM_THREADS"], "1")
        self.assertEqual(json.loads((self.root / "batch" / "report.json").read_text())["status"],
                         "failed")


class TrainingTests(WorkspaceTest):
    def config(self):
        return lab.TrialConfig(steps_per_trial=16, eval_every=8, n_steps=4,
                               batch_size=8, n_epochs=1, device="cpu")

    def fixture_vector(self):
        import gymnasium as gym
        import numpy as np
        from gymnasium import spaces
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

        class ContinuingEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(0, 1, (47, 26, 26), np.float32)
                self.action_space = spaces.Discrete(15)
                self.resets = self.decisions = 0
                self.nonfinite_after = None

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.resets += 1
                return np.zeros((47, 26, 26), np.float32), {}

            def step(self, action):
                self.decisions += 1
                reward = float("nan") if self.nonfinite_after is not None and \
                    self.decisions > self.nonfinite_after else 0.1
                return np.zeros((47, 26, 26), np.float32), reward, False, False, {}

            def action_masks(self):
                return np.ones(15, dtype=bool)

        self.envs = [ContinuingEnv(), ContinuingEnv()]
        vector = VecFrameStack(DummyVecEnv([lambda e=e: e for e in self.envs]),
                               n_stack=2, channels_order="first")
        self.addCleanup(vector.close)
        return vector

    def signature(self):
        return {"stacked_observation_shape": [94, 26, 26], "reward_profile": "shaped",
                "source_sha256": {"ablation.py": "a" * 64, "skills.py": "b" * 64},
                "rom_sha256": "c" * 64}

    def test_actual_optimizer_chunks_preserve_episode_and_save_lab_checkpoints(self):
        vector, config, signature = self.fixture_vector(), self.config(), self.signature()
        run = self.root / "trial"
        record = {"trial_id": "baseline-seed-0", "config": asdict(config),
                  "runtime": signature, "run_dir": str(run),
                  "warm_source": str(self.root / "unused.zip"), "warm_source_sha256": "unused"}
        config_path = self.root / "trial-config.json"
        config_path.write_text(json.dumps(record))
        seen = []

        def evaluation(model, config, gate, *, seeds):
            seen.append((model.num_timesteps, model._n_updates, tuple(seeds)))
            return reports(skill=model.num_timesteps / 32)

        with patch.object(lab, "verify_frozen"), \
                patch.object(lab, "build_signature", return_value=signature), \
                patch.object(lab, "make_training_env", return_value=vector), \
                patch.object(lab, "evaluate_checkpoint", side_effect=evaluation):
            result = lab.run_trial(config_path)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stop_reason"], "step_cap")
        self.assertEqual(result["actual_timesteps"], 16)
        self.assertEqual(seen, [(0, 0, (101, 202)), (8, 1, (101, 202)),
                                (16, 2, (101, 202)), (16, 2, (1001, 2002))])
        self.assertEqual([env.resets for env in self.envs], [1, 1])
        loaded, manifest = lab.load_lab_checkpoint(run / "final.zip", signature, device="cpu")
        self.assertEqual(loaded.num_timesteps, 16)
        self.assertEqual(manifest["timesteps"], 16)
        self.assertEqual(len(list((run / "checks").glob("*.json"))), 3)
        self.assertEqual(result["best_metric"]["mean_skill_success_rate"], 0.5)
        self.assertTrue((run / "confirmation.json").is_file())
        with self.assertRaises(FileExistsError):
            lab.run_trial(config_path)

    def test_fresh_seed_repeatability_and_weights_only_warm_optimizer_reset(self):
        import torch
        torch.set_num_threads(1)
        config, vector = self.config(), self.fixture_vector()
        first, _ = lab.build_model(config, vector, self.signature())
        first.learn(total_timesteps=8)
        original_parameters = copy.deepcopy(first.policy.state_dict())
        with patch("migration.load_recovery_weights", return_value=(first, {"fixture": True}),
                   create=True):
            warm, provenance = lab.build_model(replace(config, arm="warm"), vector,
                                               self.signature(), warm_source="fixture.zip")
        self.assertEqual(provenance, {"fixture": True})
        self.assertEqual(warm.num_timesteps, 0)
        self.assertEqual(warm._n_updates, 0)
        self.assertFalse(warm.policy.optimizer.state_dict()["state"])
        for key, value in original_parameters.items():
            self.assertTrue(torch.equal(value, warm.policy.state_dict()[key]), key)
        fresh1, _ = lab.build_model(config, vector, self.signature())
        fresh2, _ = lab.build_model(config, vector, self.signature())
        for key, value in fresh1.policy.state_dict().items():
            self.assertTrue(torch.equal(value, fresh2.policy.state_dict()[key]), key)

    def test_evaluation_preserves_python_numpy_torch_rng_and_policy_mode(self):
        import random
        import numpy as np
        import torch
        model, _ = lab.build_model(self.config(), self.fixture_vector(), self.signature())
        model.policy.set_training_mode(True)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        expected = random.random(), np.random.random(), torch.rand(()).item()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        with lab.preserved_evaluation(model):
            random.random()
            np.random.random()
            torch.rand(())
            model.policy.set_training_mode(False)
        self.assertEqual((random.random(), np.random.random(), torch.rand(()).item()), expected)
        self.assertTrue(model.policy.training)

    def test_sparse_training_is_evaluated_on_common_shaped_profile(self):
        from types import SimpleNamespace
        config = replace(self.config(), arm="sparse")
        model, _ = lab.build_model(config, self.fixture_vector(), self.signature())
        calls = []
        diagnostics = reports()

        def evaluate(**kwargs):
            calls.append(kwargs)
            return diagnostics["full_stage"]

        skill_module = SimpleNamespace(evaluate_skills=lambda **kwargs: diagnostics["skills"])
        with patch("experiments.evaluate", side_effect=evaluate), \
                patch.dict(lab.sys.modules, {"skills": skill_module}):
            report = lab.evaluate_checkpoint(
                model, config, lab.StopGate(3, 2700, lab.time.monotonic()),
                seeds=lab.SELECTION_SEEDS)
        self.assertEqual(calls[0]["env_factory"].keywords["reward_profile"], "shaped")
        self.assertEqual(report["full_stage"]["reward_profile"], "shaped")
        signature = dict(self.signature(), reward_profile="sparse")
        manifest = lab.make_lab_manifest(config, signature, trial_id="sparse-seed-0")
        self.assertEqual(manifest["arm"]["reward_profile"], "sparse")
        self.assertEqual(manifest["runtime"]["reward_profile"], "sparse")
        self.assertEqual(manifest["selection"]["reward_profile"], "shaped")
        self.assertEqual(manifest["confirmation"]["reward_profile"], "shaped")

    def test_nonfinite_mid_rollout_stops_and_keeps_last_optimized_checkpoint(self):
        vector, config, signature = self.fixture_vector(), self.config(), self.signature()
        self.envs[0].nonfinite_after = 4
        run = self.root / "trial"
        record = {"trial_id": "baseline-seed-0", "config": asdict(config),
                  "runtime": signature, "run_dir": str(run),
                  "warm_source": str(self.root / "unused.zip"), "warm_source_sha256": "unused"}
        config_path = self.root / "trial-config.json"
        config_path.write_text(json.dumps(record))
        with patch.object(lab, "verify_frozen"), \
                patch.object(lab, "build_signature", return_value=signature), \
                patch.object(lab, "make_training_env", return_value=vector), \
                patch.object(lab, "evaluate_checkpoint", return_value=reports()):
            result = lab.run_trial(config_path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stop_reason"], "nonfinite")
        self.assertEqual(result["actual_timesteps"], 10)
        self.assertEqual(result["optimized_timesteps"], 8)
        self.assertEqual(result["final_checkpoint_timesteps"], 8)
        loaded, _ = lab.load_lab_checkpoint(run / "final.zip", signature)
        self.assertEqual(loaded.num_timesteps, 8)
        self.assertEqual(loaded._n_updates, 1)
        self.assertEqual(result["confirmation"]["status"], "not_run")

    def test_wall_gate_mid_rollout_writes_only_completed_optimizer_steps(self):
        vector, config, signature = self.fixture_vector(), self.config(), self.signature()
        run = self.root / "trial"
        record = {"trial_id": "baseline-seed-0", "config": asdict(config),
                  "runtime": signature, "run_dir": str(run),
                  "warm_source": str(self.root / "unused.zip"), "warm_source_sha256": "unused"}
        config_path = self.root / "trial-config.json"
        config_path.write_text(json.dumps(record))

        def deadline(gate):
            if self.envs[0].decisions > 4:
                raise lab.TrialStop("wall_clock")

        with patch.object(lab, "verify_frozen"), \
                patch.object(lab, "build_signature", return_value=signature), \
                patch.object(lab, "make_training_env", return_value=vector), \
                patch.object(lab.StopGate, "check_deadline", deadline), \
                patch.object(lab, "evaluate_checkpoint", return_value=reports()):
            result = lab.run_trial(config_path)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stop_reason"], "wall_clock")
        self.assertEqual(result["actual_timesteps"], 10)
        self.assertEqual(result["final_checkpoint_timesteps"], 8)
        loaded, _ = lab.load_lab_checkpoint(run / "final.zip", signature)
        self.assertEqual(loaded.num_timesteps, 8)
        self.assertEqual(loaded._n_updates, 1)


if __name__ == "__main__":
    unittest.main()
