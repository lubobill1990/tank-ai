"""Workflow regression tests; the small fixture is not a gameplay substitute."""
import copy
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr
from functools import partial
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from experiments import (
    EpisodePlayer, TrainConfig, create_run, evaluate, make_manifest,
    runtime_signature, save_checkpoint, validate_manifest, load_checkpoint,
    summarize_episodes,
)


class FixtureEnv(gym.Env):
    """Two decisions, seeded observations, and explicit terminal counters."""
    def __init__(self, stage=1, objective="full", max_frames=12,
                 render_mode="rgb_array"):
        self.stage = stage
        self.objective = objective
        self.observation_space = spaces.Box(0, 1, (2, 26, 26), np.float32)
        self.action_space = spaces.Discrete(15)
        self.count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return self.np_random.random((2, 26, 26), dtype=np.float32), {}

    def action_masks(self):
        result = np.zeros(15, dtype=bool)
        result[:2] = True
        return result

    def heuristic_action(self):
        return 1

    def step(self, action):
        if not self.action_masks()[action]:
            raise ValueError("illegal fixture action")
        self.count += 1
        done = self.count == 2
        outcome = ("clear" if self.objective == "full" else "objective") if done else ""
        info = dict(
            stage=self.stage, objective=self.objective, outcome=outcome,
            kills=self.count, lives=2, frames=self.count * 4,
            shots=self.count + 1, deaths=1, pickups=1, stagnation_frames=4,
            item_drops=1, item_opportunities=1,
            boundary_waste_frames=4, boundary_waste_streak_frames=0,
            max_boundary_waste_frames=4,
            reward_components={"kill": 1.0, "time": -0.25},
        )
        return self.np_random.random((2, 26, 26), dtype=np.float32), 0.75, done, False, info


class SeededPolicy:
    def __init__(self):
        self.actions = []
        self.observations = []

    def predict(self, observation, *, deterministic, action_masks):
        action = 0 if deterministic else int(torch.randint(0, 2, ()).item())
        self.actions.append(action)
        self.observations.append(observation.copy())
        return np.array(action), None


class ContinuingFixtureEnv(FixtureEnv):
    """Keep one episode open across several optimizer/evaluation chunks."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, _, _, info = super().step(action)
        info["outcome"] = ""
        return observation, reward, False, False, info


class ExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("game_state.py", "battle_city_env.py", "heuristic.py",
                     "policy.py", "experiments.py", "train.py", "eval.py", "watch.py",
                     "recordings.py", "migration.py", "skills.py", "ablation.py"):
            (self.root / name).write_text("# fixture runtime\n", encoding="utf-8")
        (self.root / "battle_city.nes").write_bytes(b"fixture ROM, not an emulator")

    def signature(self, nstack=2):
        return runtime_signature(
            FixtureEnv(), nstack=nstack, root=self.root,
            env_version="fixture-v1", obs_channels=2,
        )

    def test_actual_shapes_and_rom_are_signed_but_tests_and_docs_are_not(self):
        signature = self.signature()
        self.assertEqual(signature["observation_shape"], [2, 26, 26])
        self.assertEqual(signature["stacked_observation_shape"], [4, 26, 26])
        self.assertEqual(signature["action_space"], {"type": "Discrete", "shape": [], "n": 15})
        (self.root / "test_experiments.py").write_text("changed test", encoding="utf-8")
        (self.root / "README.md").write_text("changed documentation", encoding="utf-8")
        self.assertEqual(signature, self.signature())
        (self.root / "migration.py").write_text("# changed migration\n", encoding="utf-8")
        self.assertNotEqual(signature, self.signature())
        (self.root / "battle_city.nes").write_bytes(b"different ROM")
        self.assertNotEqual(signature, self.signature())

    def test_recovery_signature_includes_profile_and_skill_protocol(self):
        raw = FixtureEnv()
        raw.reward_profile = "shaped"
        options = dict(nstack=2, root=self.root, env_version="grid26-events-v7-clean-terrain",
                       obs_channels=2, channel_names=("first", "second"))
        shaped = runtime_signature(raw, **options)
        raw.reward_profile = "sparse"
        sparse = runtime_signature(raw, **options)
        self.assertNotEqual(shaped, sparse)
        self.assertEqual(shaped["reward_profile"], "shaped")
        with self.assertRaisesRegex(ValueError, "incompatible checkpoint runtime"):
            validate_manifest(make_manifest(TrainConfig(), shaped), sparse)
        self.assertIn("skills.py", shaped["source_sha256"])
        self.assertIn("ablation.py", shaped["source_sha256"])
        (self.root / "skills.py").write_text("# different protocol\n", encoding="utf-8")
        self.assertNotEqual(sparse, runtime_signature(raw, **options))

    def test_manifest_rejects_every_incompatible_runtime_dimension(self):
        signature = self.signature()
        manifest = make_manifest(TrainConfig(), signature)
        for field, changed in [
            ("env_version", "other"),
            ("obs_channels", 3),
            ("observation_shape", [3, 26, 26]),
            ("stacked_observation_shape", [8, 26, 26]),
            ("action_space", {"type": "Discrete", "shape": [], "n": 10}),
            ("rom_sha256", "changed"),
            ("source_sha256", {}),
        ]:
            with self.subTest(field=field):
                incompatible = copy.deepcopy(signature)
                incompatible[field] = changed
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    validate_manifest(manifest, incompatible)
        validate_manifest(manifest, signature)

    def test_resume_rejects_changed_gamma_training_stages_or_fixed_evaluation(self):
        manifest = make_manifest(TrainConfig(), self.signature())
        for config in [
            TrainConfig(gamma=0.95), TrainConfig(train_stages=(1, 3)),
            TrainConfig(eval_seeds=(19, 20)), TrainConfig(benchmark_stages=(8,)),
        ]:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    validate_manifest(manifest, self.signature(), resume_config=config)
        validate_manifest(manifest, self.signature(), resume_config=TrainConfig(steps=2_000_000, device="cpu"))

    def test_missing_legacy_manifest_is_rejected_before_loading(self):
        checkpoint = self.root / "arbitrary-name.zip"
        checkpoint.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_checkpoint(checkpoint, self.signature(), device="cpu")

    def test_fresh_run_never_overwrites_even_an_empty_existing_directory(self):
        run = self.root / "run"
        manifest = make_manifest(TrainConfig(), self.signature())
        create_run(run, manifest)
        original = (run / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            create_run(run, manifest)
        self.assertEqual((run / "manifest.json").read_bytes(), original)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            create_run(empty, manifest)

    def test_objective_success_never_becomes_a_win(self):
        episodes = [
            dict(outcome="objective", objective="combat", stage=1, reward=1.0,
                 kills=1, shots=2, deaths=0, pickups=1, stagnation_frames=0, frames=4,
                 item_drops=2, item_opportunities=2,
                 boundary_waste_frames=4, boundary_waste_streak_frames=4,
                 max_boundary_waste_frames=4,
                 reward_components={"kill": 1.0}),
            dict(outcome="clear", objective="full", stage=1, reward=2.0,
                 kills=20, shots=40, deaths=1, pickups=2, stagnation_frames=8, frames=40,
                 item_drops=4, item_opportunities=4,
                 boundary_waste_frames=0, boundary_waste_streak_frames=0,
                 max_boundary_waste_frames=0,
                 reward_components={"kill": 2.0}),
        ]
        summary = summarize_episodes(episodes)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["win_rate"], 0.5)
        self.assertEqual(summary["outcomes"]["objective"], 1)
        self.assertEqual(summary["mean_kills"], 10.5)
        self.assertEqual(summary["pickup_rate"], 0.5)
        self.assertAlmostEqual(summary["boundary_waste_rate"], 4 / 44)
        self.assertEqual(summary["mean_max_boundary_waste_frames"], 2)
        episodes[0]["outcome"] = "clear"
        with self.assertRaisesRegex(ValueError, "full"):
            summarize_episodes(episodes)

    def test_pickup_rate_is_undefined_when_no_items_were_available(self):
        episode = dict(outcome="timeout", objective="full", stage=1, reward=0,
                       kills=0, shots=0, deaths=0, pickups=0, frames=120,
                       stagnation_frames=120, reward_components={},
                       item_drops=0, item_opportunities=0)
        episode.update(boundary_waste_frames=0, boundary_waste_streak_frames=0,
                       max_boundary_waste_frames=0)
        self.assertIsNone(summarize_episodes([episode])["pickup_rate"])

    def test_evaluation_reports_both_modes_and_is_seed_reproducible(self):
        policy = SeededPolicy()
        kwargs = dict(
            model=policy, stages=(1, 2), seeds=(123, 456),
            nstack=2, max_frames=12, env_factory=FixtureEnv,
        )
        first = evaluate(**kwargs)
        first_actions = list(policy.actions)
        first_observations = list(policy.observations)
        policy.actions.clear()
        policy.observations.clear()
        second = evaluate(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_actions, policy.actions)
        for first_observation, second_observation in zip(first_observations, policy.observations):
            np.testing.assert_array_equal(first_observation, second_observation)
        self.assertFalse(np.array_equal(first_observations[0], first_observations[2]))
        self.assertEqual(set(first["modes"]), {"deterministic", "sampled"})
        for result in first["modes"].values():
            self.assertEqual(result["summary"]["wins"], 4)
            self.assertEqual(result["per_stage"]["2"]["episodes"], 2)
            self.assertEqual([e["seed"] for e in result["episodes"]], [123, 456, 123, 456])
            self.assertEqual(result["episodes"][0]["reward_components"], {"kill": 2.0, "time": -0.5})
            self.assertEqual(result["episodes"][0]["shots"], 3)
            self.assertEqual(result["episodes"][0]["item_drops"], 1)
            self.assertEqual(result["summary"]["pickup_rate"], 1.0)
            self.assertEqual(result["episodes"][0]["max_boundary_waste_frames"], 4)

    def test_evaluation_preserves_training_random_states(self):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        expected = random.random(), np.random.random(), torch.rand(1)
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        evaluate(model=SeededPolicy(), stages=(1,), seeds=(123,), nstack=2,
                 max_frames=12, env_factory=FixtureEnv)
        actual = random.random(), np.random.random(), torch.rand(1)
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_eval_and_watch_stack_matches_training_vec_frame_stack(self):
        player = EpisodePlayer(FixtureEnv(), nstack=2, model=SeededPolicy())
        vec = VecFrameStack(DummyVecEnv([FixtureEnv]), n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        vec.seed(42)
        expected = vec.reset()[0]
        np.testing.assert_array_equal(player.reset(42), expected)
        player.step()
        expected, _, _, _ = vec.step(np.array([0]))
        np.testing.assert_array_equal(player.observation, expected[0])

    def test_config_rejects_invalid_or_leaking_benchmarks(self):
        for kwargs in [
            {"gamma": 0}, {"gamma": float("nan")}, {"steps": 0}, {"envs": 0},
            {"nstack": 0}, {"train_stages": (0,)}, {"eval_seeds": (-1,)},
            {"train_stages": (1, 1)}, {"eval_stages": (1,)},
            {"benchmark_stages": (4,)}, {"combat_steps": 1_000_000},
            {"batch_size": 3}, {"n_steps": 1},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TrainConfig(**kwargs).validate()

    def test_training_flags_and_curriculum_budgets(self):
        import train
        args = train.parse_args([
            "--output-dir", str(self.root / "training"),
            "--train-stages", "1,3", "--eval-stages", "5",
            "--benchmark-stages", "7", "--eval-seeds", "9,10",
            "--steps", "128", "--envs", "2", "--n-steps", "8",
            "--batch-size", "8", "--combat-steps", "32",
            "--defense-steps", "32", "--gamma", ".98", "--seed", "11",
            "--device", "cpu",
        ])
        config = train.config_from_args(args)
        self.assertEqual(config.phases(), [("combat", 32), ("defense", 32), ("full", 64)])
        self.assertEqual(config.train_stages, (1, 3))
        self.assertEqual(config.gamma, 0.98)

    def test_weights_only_start_is_explicit_and_exclusive_with_resume(self):
        import train
        output = ["--output-dir", str(self.root / "new-run")]
        with redirect_stderr(io.StringIO()):
            try:
                args = train.parse_args([*output, "--warm-start", "source.zip"])
            except SystemExit:
                self.fail("training must accept an explicit --warm-start checkpoint")
            with self.assertRaises(SystemExit):
                train.parse_args([*output, "--warm-start", "source.zip", "--resume", "source.zip"])
            with self.assertRaises(SystemExit):
                train.parse_args(["--warm-start", "source.zip"])
        self.assertEqual(args.warm_start, Path("source.zip"))
        self.assertIsNone(args.resume)

    def test_single_worker_cycles_all_configured_training_stages(self):
        import train
        config = TrainConfig(train_stages=(1, 3), n_steps=4, batch_size=4)
        with patch("train.make_env", FixtureEnv):
            vec = train.training_env(config, "combat")
            self.addCleanup(vec.close)
            vec.reset()
            stages = []
            for _ in range(6):
                _, _, dones, infos = vec.step(np.array([0]))
                if dones[0]:
                    stages.append(infos[0]["stage"])
                    self.assertEqual(infos[0]["objective"], "combat")
        self.assertEqual(stages, [1, 3, 1])

    def test_incomplete_training_configuration_is_rejected(self):
        manifest = make_manifest(TrainConfig(), self.signature())
        del manifest["training"]["nstack"]
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_manifest(manifest, self.signature())

    def test_checkpoint_integrity_and_cpu_model_roundtrip(self):
        from sb3_contrib import MaskablePPO
        config = TrainConfig(steps=8, n_steps=4, batch_size=4, n_epochs=1, device="cpu")
        vec = VecFrameStack(DummyVecEnv([FixtureEnv]), n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        model = MaskablePPO("MlpPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                            policy_kwargs={"net_arch": [8], "ortho_init": False},
                            seed=123, device="cpu")
        model.learn(total_timesteps=8)
        manifest = make_manifest(config, self.signature())
        path = self.root / "renamed-policy.zip"
        save_checkpoint(path, model, manifest, phase="full", progress={"combat": 0, "defense": 0, "full": 8})
        loaded, saved = load_checkpoint(path, self.signature(), device="cpu", resume_config=config)
        self.assertEqual(saved["timesteps"], 8)
        kwargs = dict(stages=(1,), seeds=(123,), env_factory=FixtureEnv, max_frames=12)
        self.assertEqual(evaluate(model=model, **kwargs), evaluate(model=loaded, **kwargs))
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_checkpoint(path, self.signature(), device="cpu")

    def test_periodic_evaluation_uses_full_episodes_and_saves_best_not_latest(self):
        from sb3_contrib import MaskablePPO
        import train
        config = TrainConfig(steps=8, n_steps=4, batch_size=4, n_epochs=1,
                             eval_stages=(4,), eval_seeds=(123,), eval_every=4, device="cpu")
        vec = VecFrameStack(DummyVecEnv([partial(FixtureEnv, objective="combat")]),
                            n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        model = MaskablePPO("MlpPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                            policy_kwargs={"net_arch": [8], "ortho_init": False},
                            seed=123, device="cpu")
        initial_weights = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        manifest = make_manifest(config, self.signature())
        run = create_run(self.root / "periodic", manifest)
        evaluation = train.FullStageEvaluation(config, run, manifest, {"combat": 0, "defense": 0, "full": 0})
        with patch("train.evaluate", partial(evaluate, env_factory=FixtureEnv)):
            train.learn_phase(model, "combat", 8, evaluation)
        reports = sorted(run.glob("eval-*.json"))
        self.assertEqual(len(reports), 3)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["role"], "checkpoint_selection")
        self.assertEqual(report["phase"], "combat")
        self.assertEqual(report["objective"], "full")
        self.assertEqual(report["seeds"], [123])
        self.assertEqual(report["stages"], [4])
        saved = json.loads((run / "best.zip.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["timesteps"], 4)
        self.assertEqual(saved["progress"], {"combat": 4, "defense": 0, "full": 0})
        checkpoint, _ = load_checkpoint(run / "best.zip", self.signature(), device="cpu")
        optimizer_states = checkpoint.policy.optimizer.state_dict()["state"]
        self.assertTrue(optimizer_states, "saved progress must include an actual optimizer update")
        self.assertTrue(all(int(state["step"]) == 1 for state in optimizer_states.values()))
        self.assertTrue(any(not torch.equal(value, initial_weights[name])
                            for name, value in checkpoint.policy.state_dict().items()))
        final_report = json.loads((run / "eval-000000000008-phase-end.json").read_text(encoding="utf-8"))
        self.assertEqual(final_report["evaluation_point"], "phase_end")

    def test_resume_only_skips_combat_that_checkpoint_has_optimized(self):
        from sb3_contrib import MaskablePPO
        import train
        config = TrainConfig(steps=8, combat_steps=4, n_steps=4, batch_size=4, n_epochs=1,
                             eval_stages=(4,), eval_seeds=(123,), eval_every=4, device="cpu")
        vec = VecFrameStack(DummyVecEnv([partial(FixtureEnv, objective="combat")]),
                            n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        model = MaskablePPO("MlpPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                            policy_kwargs={"net_arch": [8], "ortho_init": False},
                            seed=123, device="cpu")
        manifest = make_manifest(config, self.signature())
        run = create_run(self.root / "curriculum", manifest)
        progress = {"combat": 0, "defense": 0, "full": 0}
        evaluation = train.FullStageEvaluation(config, run, manifest, progress)
        with patch("train.evaluate", partial(evaluate, env_factory=FixtureEnv)):
            train.learn_phase(model, "combat", 4, evaluation)
        full_vec = VecFrameStack(DummyVecEnv([FixtureEnv]), n_stack=2, channels_order="first")
        self.addCleanup(full_vec.close)
        resumed, saved = load_checkpoint(run / "best.zip", self.signature(),
                                         device="cpu", resume_config=config, env=full_vec)
        states = resumed.policy.optimizer.state_dict()["state"]
        self.assertTrue(states)
        self.assertTrue(all(int(state["step"]) == 1 for state in states.values()))
        remaining = [(phase, budget - saved["progress"][phase])
                     for phase, budget in config.phases() if budget > saved["progress"][phase]]
        self.assertEqual(remaining, [("full", 4)])
        resumed_run = create_run(self.root / "resumed", manifest)
        resumed_evaluation = train.FullStageEvaluation(config, resumed_run, manifest, saved["progress"])
        with patch("train.evaluate", partial(evaluate, env_factory=FixtureEnv)):
            for phase, budget in remaining:
                train.learn_phase(resumed, phase, budget, resumed_evaluation)
        final, final_manifest = load_checkpoint(resumed_run / "best.zip", self.signature(), device="cpu")
        self.assertEqual(final_manifest["progress"], {"combat": 4, "defense": 0, "full": 4})
        self.assertEqual(final_manifest["timesteps"], 8)
        self.assertTrue(all(int(state["step"]) == 2
                            for state in final.policy.optimizer.state_dict()["state"].values()))

    def test_chunk_boundaries_preserve_episode_and_round_evaluation_to_rollouts(self):
        from sb3_contrib import MaskablePPO
        import train
        config = TrainConfig(steps=8, n_steps=4, batch_size=4, n_epochs=1,
                             eval_stages=(4,), eval_seeds=(123,), eval_every=3, device="cpu")
        raw = ContinuingFixtureEnv()
        vec = VecFrameStack(DummyVecEnv([lambda: raw]), n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        model = MaskablePPO("MlpPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                            policy_kwargs={"net_arch": [8], "ortho_init": False},
                            seed=123, device="cpu")
        manifest = make_manifest(config, self.signature())
        run = create_run(self.root / "continuous", manifest)
        progress = {"combat": 0, "defense": 0, "full": 0}
        evaluation = train.FullStageEvaluation(config, run, manifest, progress)
        with patch("train.evaluate", partial(evaluate, env_factory=FixtureEnv)):
            train.learn_phase(model, "full", 7, evaluation)
        self.assertEqual(model.num_timesteps, 8)
        self.assertEqual(progress, {"combat": 0, "defense": 0, "full": 8})
        self.assertEqual(raw.reset_count, 1)
        self.assertEqual(raw.count, 8)
        self.assertEqual(sorted(path.name for path in run.glob("eval-*.json")), [
            "eval-000000000004.json", "eval-000000000008-phase-end.json", "eval-000000000008.json",
        ])

    def test_training_stage_cycles_replay_identical_seeds(self):
        import train
        first = train.TrainingStages((1, 3), 0, "full", 12, FixtureEnv)
        second = train.TrainingStages((1, 3), 0, "full", 12, FixtureEnv)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        for seed in (123, None, None):
            first_observation, _ = first.reset(seed=seed)
            second_observation, _ = second.reset(seed=seed)
            np.testing.assert_array_equal(first_observation, second_observation)

    def test_eval_and_watch_require_explicit_agent_and_reject_legacy_flags(self):
        import eval as evaluation_cli
        import watch
        for module in (evaluation_cli, watch):
            required = ["--output", str(self.root / "evaluation.json")] if module is evaluation_cli else []
            with self.subTest(module=module.__name__), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(required)
                with self.assertRaises(SystemExit):
                    module.parse_args(["--heuristic", "--kind", "pixel", *required])
                argv = ["--heuristic", *required]
                self.assertTrue(module.parse_args(argv).heuristic)

    def test_watch_can_follow_one_explicit_run(self):
        import watch
        args = watch.parse_args(["--run-dir", str(self.root / "run")])
        self.assertEqual(args.run_dir, self.root / "run")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            watch.parse_args(["--run-dir", str(self.root / "run"), "--heuristic"])

    def test_watch_recording_directory_and_opt_out_are_explicit(self):
        import watch
        directory = self.root / "clips"
        args = watch.parse_args(["--heuristic", "--record-dir", str(directory)])
        self.assertEqual(args.record_dir, directory)
        self.assertFalse(args.no_record)
        self.assertTrue(watch.parse_args(["--heuristic", "--no-record"]).no_record)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            watch.parse_args(["--heuristic", "--record-dir", str(directory), "--no-record"])

    def test_latest_publication_is_optimized_bounded_and_independent_of_evaluation(self):
        from sb3_contrib import MaskablePPO
        from experiments import latest_checkpoint
        import train
        config = TrainConfig(steps=16, n_steps=4, batch_size=4, n_epochs=1,
                             checkpoint_every=4, eval_every=100,
                             eval_stages=(4,), eval_seeds=(123,), device="cpu")
        vec = VecFrameStack(DummyVecEnv([FixtureEnv]), n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        model = MaskablePPO("MlpPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                            policy_kwargs={"net_arch": [8], "ortho_init": False},
                            seed=123, device="cpu")
        manifest = make_manifest(config, self.signature())
        run = create_run(self.root / "live", manifest)
        evaluation = train.FullStageEvaluation(
            config, run, manifest, {"combat": 0, "defense": 0, "full": 0})
        with patch("train.evaluate", partial(evaluate, env_factory=FixtureEnv)):
            train.learn_phase(model, "full", 16, evaluation)
        latest = latest_checkpoint(run)
        self.assertEqual(latest.name, "step-000000000016.zip")
        loaded, saved = load_checkpoint(latest, self.signature(), device="cpu")
        self.assertEqual(loaded._n_updates, 4)
        self.assertEqual(saved["progress"]["full"], 16)
        self.assertEqual(sorted(p.name for p in (run / "checkpoints").glob("*.zip")),
                         [f"step-{step:012d}.zip" for step in (8, 12, 16)])
        self.assertEqual(len(list(run.glob("eval-*.json"))), 1)

    def test_failed_publication_keeps_previous_model_and_watch_switches_only_between_episodes(self):
        from sb3_contrib import MaskablePPO
        from experiments import latest_checkpoint, publish_latest
        from watch import RunFollower
        vec = VecFrameStack(DummyVecEnv([FixtureEnv]), n_stack=2, channels_order="first")
        self.addCleanup(vec.close)
        model = MaskablePPO("MlpPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                            policy_kwargs={"net_arch": [8], "ortho_init": False},
                            seed=123, device="cpu")
        config = TrainConfig(steps=8, n_steps=4, batch_size=4, n_epochs=1, device="cpu")
        manifest = make_manifest(config, self.signature())
        run = create_run(self.root / "publication", manifest)
        progress = {"combat": 0, "defense": 0, "full": 4}
        model.learn(4)
        first = publish_latest(run, model, manifest, phase="full", progress=progress)
        old_pointer = (run / "latest.json").read_bytes()
        old_model, _ = load_checkpoint(first, self.signature(), device="cpu")
        player = EpisodePlayer(FixtureEnv(), model=old_model)
        player.reset(123)
        follower = RunFollower(run, self.signature(), first, device="cpu")
        model.learn(4, reset_num_timesteps=False)
        progress["full"] = 8
        with patch.object(model, "save", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                publish_latest(run, model, manifest, phase="full", progress=progress)
        self.assertEqual((run / "latest.json").read_bytes(), old_pointer)
        publish_latest(run, model, manifest, phase="full", progress=progress)
        self.assertFalse(follower.refresh(player))
        self.assertIs(player.model, old_model)
        player.step()
        player.step()
        self.assertTrue(follower.refresh(player))
        self.assertEqual(player.model.num_timesteps, 8)
        self.assertEqual(follower.checkpoint, latest_checkpoint(run))
        self.assertFalse(follower.refresh(player))
        (run / "latest.json").write_text(
            json.dumps({"checkpoint": "../external.zip", "timesteps": 8}),
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "latest"):
            latest_checkpoint(run)


if __name__ == "__main__":
    unittest.main()
