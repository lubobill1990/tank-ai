"""Weights-only migration must preserve outputs without inheriting training state."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

from experiments import TrainConfig, load_checkpoint, make_manifest, save_checkpoint, sha256
from policy import Grid26CNN
import migration


SOURCE_RUNTIME = {
    "action_space": {"n": 15, "shape": [], "type": "Discrete"},
    "env_version": "grid26-events-v3",
    "obs_channels": 39,
    "observation_dtype": "float32",
    "observation_shape": [39, 26, 26],
    "rom_sha256": "a869aead5b6957fc62002ff9636e048cc34baf0100d629b07dc51aa18f220c0c",
    "source_sha256": {
        "battle_city_env.py": "5d35e4e52113ae06ee2b75bc51b3f386d27490a20378fbac27114384ef75f319",
        "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
        "experiments.py": "4463ebb58550350d72ac5492ba943a0ada037186efe614fe068727fd46eb4721",
        "game_state.py": "d9ea6739d9653b4502aba5dfe12f2fd3b5ec56ac57fd03be08fbc6dc793b31d6",
        "heuristic.py": "6eddf3f3aa4b6cfab0d7492141caf8bc44a32f3f9ded2b9536e5621dee00ca82",
        "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
        "recordings.py": "bd99a4b532cfb020fe861c85e62bfa96a1e22c984ea6bb5142b4c1e096ae797d",
        "train.py": "308e65027686c2828db9b12450479345b46784e9b873a1af767add9d6b4cd9a2",
        "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
    },
    "stack_axis": "first",
    "stacked_observation_shape": [78, 26, 26],
}
OLD_NAMES = (
    "brick", "steel", "water", "ice", "base",
    "player", "player_fx", "player_fy", "enemy", "enemy_fx", "enemy_fy", "armor",
    "pbullet", "pbullet_vx", "pbullet_vy", "ebullet", "ebullet_vx", "ebullet_vy",
    "remaining", "spawn_left", "lives", "power", "shield", "freeze", "fire_ready",
    "item_shield", "item_freeze", "item_shovel", "item_star", "item_grenade", "item_life",
    "last_hunt", "last_protect", "last_dodge", "last_counter", "last_item",
    "time_left", "objective_combat", "objective_defense",
)
NEW_NAMES = OLD_NAMES + (
    "brick_tl", "brick_tr", "brick_bl", "brick_br", "player_offset_x", "player_offset_y",
)
TACTICS_NAMES = NEW_NAMES + ("bullet_danger", "enemy_drops_item")
V4_SOURCE_HASHES = {
    "battle_city_env.py": "68777edd5614bf3343bfba7ce9666eb25d3bd6217c2c38f39acce02e49ed421c",
    "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
    "experiments.py": "f81ed9d20cd6bbf45a8b7bde3a6309de7e3e62e3e5fcc4d3161f6fe8272b0352",
    "game_state.py": "aac9b13995d64932d41da39a1ddad168cc86879c204612ed2ae416cd49f621f7",
    "heuristic.py": "9318dffa324891c2d54404f3cb3004ca3779f56f73e5dd9d6f574943f2cefe1e",
    "migration.py": "97cbf0d543f4525d9a4f9ae369164731afade76710a7af9766b17ffae8e8c94a",
    "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
    "recordings.py": "dbf521d152e96f1fe5977d37845fa6d3e0b3daec703af60aeca8022d4d9c11fe",
    "train.py": "318e497daaf49b9a27f466492b6e8b9ff4a187215deb7387ef4d2e851e977bd6",
    "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
}
V5_SOURCE_HASHES = {
    "battle_city_env.py": "6fa10fe40c4284469ecd93c29b5cccbde6324dae025b00e0428335ac93491d4a",
    "eval.py": "3366688012b4239f2026ea349004ee40eb683ee2c533f2838f8ba4b30b3b838c",
    "experiments.py": "31eea375337093cf67054baf4b8207237dd6e9a6190d84696164901d70d224b8",
    "game_state.py": "8553d9a163760c4b7b2d91e31f2e796cb69f47eff9b5602f08d3c288db593a52",
    "heuristic.py": "3170dfa8391a7f29c806cfe3e29c02100e11e0be34ea95a15532c2b9c6c8fda8",
    "migration.py": "1e616f06315a4ff33fabaff4b8dd36ce143822e5d30afe6b61923a6692f148c3",
    "policy.py": "b9a5d30bc3e2f0bd591d84ded7f52d9b470a0ce7578244d0574e6abf64c64b19",
    "recordings.py": "dbf521d152e96f1fe5977d37845fa6d3e0b3daec703af60aeca8022d4d9c11fe",
    "train.py": "318e497daaf49b9a27f466492b6e8b9ff4a187215deb7387ef4d2e851e977bd6",
    "watch.py": "ccb7b02e73955f254e5d3b458a45b2b3f54567f894548d927280350751159eca",
}
INPUT_KEYS = tuple(f"{prefix}.cnn.0.weight" for prefix in (
    "features_extractor", "pi_features_extractor", "vf_features_extractor",
))


class TensorEnv(gym.Env):
    def __init__(self, channels):
        self.observation_space = spaces.Box(-1, 1, (channels, 26, 26), np.float32)
        self.action_space = spaces.Discrete(15)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.np_random.uniform(-1, 1, self.observation_space.shape).astype(np.float32), {}

    def step(self, action):
        observation, _ = self.reset()
        return observation, 1.0, True, False, {}

    def action_masks(self):
        return np.ones(15, dtype=bool)


def make_model(channels):
    return MaskablePPO(
        "CnnPolicy", TensorEnv(channels), device="cpu", seed=12,
        policy_kwargs=dict(features_extractor_class=Grid26CNN,
                           features_extractor_kwargs=dict(features_dim=256),
                           normalize_images=False),
        n_steps=2, batch_size=2, n_epochs=1,
    )


def signatures(nstack=2):
    old = copy.deepcopy(SOURCE_RUNTIME)
    old["stacked_observation_shape"] = [39 * nstack, 26, 26]
    new = copy.deepcopy(old)
    new.update(env_version="grid26-events-v4-residual", obs_channels=45,
               observation_shape=[45, 26, 26], stacked_observation_shape=[45 * nstack, 26, 26],
               channel_names=list(NEW_NAMES))
    new["source_sha256"]["migration.py"] = "a" * 64
    return old, new


def tactics_signatures(nstack=2):
    _, old = signatures(nstack)
    old["source_sha256"] = dict(V4_SOURCE_HASHES)
    new = copy.deepcopy(old)
    new.update(env_version="grid26-events-v5-tactics", obs_channels=47,
               observation_shape=[47, 26, 26], stacked_observation_shape=[47 * nstack, 26, 26],
               channel_names=list(TACTICS_NAMES))
    for name in ("battle_city_env.py", "heuristic.py", "migration.py", "game_state.py", "experiments.py"):
        new["source_sha256"][name] = "a" * 64
    return old, new


def boundary_signatures():
    _, old = tactics_signatures()
    old["source_sha256"] = dict(V5_SOURCE_HASHES)
    new = copy.deepcopy(old)
    new["env_version"] = "grid26-events-v6-boundary"
    for name in ("migration.py", "train.py", "experiments.py", "heuristic.py", "battle_city_env.py"):
        new["source_sha256"][name] = "a" * 64
    return old, new


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = TrainConfig(steps=8, combat_steps=4, defense_steps=2, device="cpu",
                                  n_steps=2, batch_size=2, n_epochs=1, seed=12,
                                  eval_stages=(4,), eval_seeds=(1,))
        self.old, self.new = boundary_signatures()

    def checkpoint(self, steps=2):
        model = make_model(94)
        model.learn(total_timesteps=steps)
        manifest = make_manifest(self.config, self.old)
        manifest["warm_start"] = {
            "method": "v4-to-v5-per-frame-tactics-input-expansion-weights-only",
            "source_num_timesteps": 53760,
            "source_warm_start": {"source_num_timesteps": 176640},
        }
        path = self.root / "source.zip"
        save_checkpoint(path, model, manifest, phase="combat",
                        progress={"combat": steps, "defense": 0, "full": 0})
        return model, path

    def load(self, path, **kwargs):
        options = dict(nstack=2, device="cpu", resume_config=self.config)
        options.update(kwargs)
        return migration.load_runtime_continuation(path, self.new, **options)

    def assert_exact_state(self, expected, actual):
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.detach().cpu().numpy().tobytes(),
                             actual.detach().cpu().numpy().tobytes())
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_exact_state(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_exact_state(left, right)
        else:
            self.assertEqual(expected, actual)

    def assert_model_preserved(self, source, loaded):
        self.assert_exact_state(source.policy.state_dict(), loaded.policy.state_dict())
        self.assert_exact_state(source.policy.optimizer.state_dict(), loaded.policy.optimizer.state_dict())
        for name in ("num_timesteps", "_n_updates", "_current_progress_remaining", "_total_timesteps",
                     "learning_rate", "n_steps", "n_envs", "batch_size", "n_epochs", "gamma",
                     "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "seed"):
            self.assertEqual(getattr(source, name), getattr(loaded, name), name)
        for progress in (0, 0.5, 1):
            self.assertEqual(source.lr_schedule(progress), loaded.lr_schedule(progress))
            self.assertEqual(source.clip_range(progress), loaded.clip_range(progress))
        self.assertIs(loaded.policy.features_extractor, loaded.policy.pi_features_extractor)
        self.assertIs(loaded.policy.features_extractor, loaded.policy.vf_features_extractor)

    def test_continuation_preserves_full_state_and_roundtrips_new_manifest(self):
        self.assertTrue(callable(getattr(migration, "load_runtime_continuation", None)),
                        "explicit full-state continuation loader is missing")
        source, path = self.checkpoint()
        before = path.read_bytes()
        loaded, manifest, provenance = self.load(path)
        self.assert_model_preserved(source, loaded)
        self.assertIsNone(loaded.get_env())
        self.assertIsNone(loaded._last_obs)
        self.assertEqual(manifest["progress"], {"combat": 2, "defense": 0, "full": 0})
        self.assertEqual(provenance["method"], "v5-to-v6-same-layout-full-state-continuation")
        self.assertEqual(provenance["source_checkpoint"], str(path.resolve()))
        self.assertEqual(provenance["source_checkpoint_sha256"], sha256(path))
        self.assertEqual(provenance["source_runtime"], self.old)
        self.assertEqual(provenance["source_num_timesteps"], 2)
        self.assertEqual(provenance["source_n_updates"], 1)
        self.assertEqual(provenance["source_progress"], manifest["progress"])
        target = make_manifest(self.config, self.new)
        target.update(continuation=provenance, warm_start=copy.deepcopy(manifest["warm_start"]))
        continued_path = self.root / "continued.zip"
        save_checkpoint(continued_path, loaded, target, phase="combat", progress=manifest["progress"])
        reloaded, saved_manifest = load_checkpoint(continued_path, self.new, device="cpu",
                                                  resume_config=self.config)
        self.assert_model_preserved(source, reloaded)
        self.assertEqual(saved_manifest["continuation"], provenance)
        self.assertEqual(saved_manifest["warm_start"], manifest["warm_start"])
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "runtime"):
            load_checkpoint(path, self.new, device="cpu", resume_config=self.config)

    def test_cli_continuation_is_explicit_and_mutually_exclusive(self):
        import train
        args = train.parse_args(["--output-dir", str(self.root / "run"),
                                 "--continue-from", "source.zip"])
        self.assertEqual(args.continue_from, Path("source.zip"))
        for flag in ("--resume", "--warm-start"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                train.parse_args(["--output-dir", str(self.root / "run"),
                                  "--continue-from", "source.zip", flag, "other.zip"])
            self.assertEqual(error.exception.code, 2)

    def test_only_reviewed_same_layout_runtime_profile_is_accepted(self):
        manifest = make_manifest(self.config, self.old)
        migration.validate_runtime_continuation_manifest(
            manifest, self.new, nstack=2, resume_config=self.config)
        for side in ("source", "destination"):
            for field, value in [
                ("env_version", "grid26-events-v5-fire-safety"),
                ("obs_channels", 46), ("observation_shape", [47, 13, 13]),
                ("observation_dtype", "float64"), ("stacked_observation_shape", [47, 26, 26]),
                ("stack_axis", "last"), ("rom_sha256", "b" * 64),
                ("channel_names", list(reversed(TACTICS_NAMES))),
                ("action_space", {"type": "Discrete", "shape": [], "n": 14}),
                ("source_sha256", {}), ("unexpected", True),
            ]:
                with self.subTest(side=side, field=field):
                    changed, signature = copy.deepcopy(manifest), copy.deepcopy(self.new)
                    (changed["runtime"] if side == "source" else signature)[field] = value
                    with self.assertRaises(ValueError):
                        migration.validate_runtime_continuation_manifest(
                            changed, signature, nstack=2, resume_config=self.config)
            for name in V5_SOURCE_HASHES:
                if side == "destination" and name in (
                    "migration.py", "train.py", "experiments.py", "heuristic.py", "battle_city_env.py",
                ):
                    continue
                with self.subTest(side=side, name=name):
                    changed, signature = copy.deepcopy(manifest), copy.deepcopy(self.new)
                    runtime = changed["runtime"] if side == "source" else signature
                    runtime["source_sha256"][name] = "b" * 64
                    with self.assertRaises(ValueError):
                        migration.validate_runtime_continuation_manifest(changed, signature, nstack=2)
        for bad_hashes in (
            dict(self.new["source_sha256"], unknown="a" * 64),
            dict(self.new["source_sha256"], **{"train.py": "z" * 64}),
            dict(self.new["source_sha256"], **{"train.py": "a"}),
            dict(self.new["source_sha256"], **{"train.py": None}),
        ):
            with self.subTest(hashes=bad_hashes), self.assertRaises(ValueError):
                migration.validate_runtime_continuation_manifest(
                    manifest, dict(self.new, source_sha256=bad_hashes), nstack=2)
        for nstack in (0, 1, 3, -1, True, 2.0):
            with self.subTest(nstack=nstack), self.assertRaises(ValueError):
                migration.validate_runtime_continuation_manifest(manifest, self.new, nstack=nstack)
        changed = copy.deepcopy(manifest)
        changed["training"]["nstack"] = 1
        with self.assertRaises(ValueError):
            migration.validate_runtime_continuation_manifest(changed, self.new, nstack=2)
        for old in (signatures()[0], tactics_signatures()[0]):
            with self.assertRaises(ValueError):
                migration.validate_runtime_continuation_manifest(
                    make_manifest(self.config, old), self.new, nstack=2)
        with self.assertRaises(ValueError):
            migration.validate_warm_start_manifest(manifest, self.new, nstack=2)

    def test_continuation_retains_strict_resume_config_constraints(self):
        manifest = make_manifest(self.config, self.old)
        migration.validate_runtime_continuation_manifest(
            manifest, self.new, nstack=2,
            resume_config=replace(self.config, steps=20_000_000, device="auto",
                                  eval_every=200, checkpoint_every=100))
        for changes in (
            {"gamma": 0.98}, {"seed": 15}, {"n_steps": 4}, {"batch_size": 4, "envs": 2},
            {"n_epochs": 2}, {"combat_steps": 3}, {"defense_steps": 1}, {"max_frames": 500},
            {"eval_seeds": (3,)}, {"train_stages": (1, 2)}, {"steps": 7},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "resume"):
                migration.validate_runtime_continuation_manifest(
                    manifest, self.new, nstack=2, resume_config=replace(self.config, **changes))

    def rewrite_source(self, path, mutate):
        data, params, variables = load_from_zip_file(path, device="cpu")
        mutate(data, params)
        save_to_zip_file(path, data=data, params=params, pytorch_variables=variables)
        sidecar = path.with_suffix(".zip.manifest.json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        manifest["checkpoint_sha256"] = sha256(path)
        sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    def test_continuation_rejects_stale_counters_or_invalid_curriculum_progress(self):
        _, path = self.checkpoint()
        sidecar = path.with_suffix(".zip.manifest.json")
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        for changes in (
            {"timesteps": 3}, {"timesteps": True},
            {"progress": {"combat": 1, "defense": 0, "full": 0}},
            {"progress": {"combat": 3, "defense": -1, "full": 0}},
            {"progress": {"combat": 2, "defense": 0}},
            {"progress": {"combat": 1, "defense": True, "full": 0}},
            {"progress": {"combat": 2, "defense": 0.0, "full": 0}},
        ):
            with self.subTest(changes=changes):
                sidecar.write_text(json.dumps(dict(original, **changes)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "progress|timestep"):
                    self.load(path)

    def test_continuation_rejects_unoptimized_or_disguised_source_model(self):
        mutations = [
            ("updates", lambda data, params: data.update(_n_updates=0)),
            ("negative updates", lambda data, params: data.update(_n_updates=-1)),
            ("optimizer", lambda data, params: params["policy.optimizer"].update(state={})),
            ("missing optimizer", lambda data, params: params.pop("policy.optimizer")),
            ("model config", lambda data, params: data.update(gamma=0.5)),
            ("model workers", lambda data, params: data.update(n_envs=2)),
            ("architecture", lambda data, params: data["policy_kwargs"].update(activation_fn=torch.nn.ReLU)),
            ("alias", lambda data, params: params["policy"].update(
                {INPUT_KEYS[0]: params["policy"][INPUT_KEYS[0]] + 1})),
            ("dtype", lambda data, params: params["policy"].update(
                {"value_net.bias": params["policy"]["value_net.bias"].double()})),
            ("optimizer dtype", lambda data, params: params["policy.optimizer"]["state"][0].update(
                exp_avg=params["policy.optimizer"]["state"][0]["exp_avg"].double())),
            ("missing moment", lambda data, params: params["policy.optimizer"]["state"][0].pop("exp_avg")),
            ("moment shape", lambda data, params: params["policy.optimizer"]["state"][0].update(
                exp_avg=torch.zeros(1))),
            ("unused optimizer", lambda data, params: params["policy.optimizer"]["state"][0].update(
                step=torch.tensor(0.))),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                _, path = self.checkpoint()
                self.rewrite_source(path, mutate)
                with self.assertRaises(ValueError):
                    self.load(path)

    def training_argv(self, path, run):
        return [
            "--continue-from", str(path), "--output-dir", str(run),
            "--steps", str(self.config.steps), "--combat-steps", str(self.config.combat_steps),
            "--defense-steps", str(self.config.defense_steps), "--n-steps", "2", "--batch-size", "2",
            "--n-epochs", "1", "--envs", "1", "--device", "cpu", "--nstack", "2",
            "--seed", "12", "--eval-stages", "4", "--eval-seeds", "1",
        ]

    @staticmethod
    def fixture_factory(**kwargs):
        from test_experiments import FixtureEnv

        class BoundaryFixtureEnv(FixtureEnv):
            def __init__(self, **options):
                super().__init__(**options)
                self.observation_space = spaces.Box(-1, 1, (47, 26, 26), np.float32)

            @staticmethod
            def expand(observation):
                result = np.zeros((47, 26, 26), dtype=np.float32)
                result[:2] = observation
                return result

            def reset(self, **options):
                observation, info = super().reset(**options)
                return self.expand(observation), info

            def step(self, action):
                observation, *rest = super().step(action)
                return self.expand(observation), *rest

        return BoundaryFixtureEnv(**kwargs)

    def test_training_preserves_starting_step_and_only_remaining_phase_budgets(self):
        import train
        self.config = replace(self.config, steps=6, defense_steps=0)
        for source_steps in (2, 4):
            with self.subTest(source_steps=source_steps):
                source, path = self.checkpoint(steps=source_steps)
                run = self.root / f"continue-{source_steps}"
                with patch("train.make_env", self.fixture_factory), \
                        patch("train.evaluate", partial(train.evaluate, env_factory=self.fixture_factory)), \
                        patch("train.runtime_signature", return_value=self.new), redirect_stdout(io.StringIO()):
                    train.main(self.training_argv(path, run))
                initial, manifest = load_checkpoint(
                    run / "checkpoints" / f"step-{source_steps:012d}.zip", self.new, device="cpu")
                self.assert_model_preserved(source, initial)
                self.assertEqual(manifest["progress"], {"combat": source_steps, "defense": 0, "full": 0})
                self.assertEqual(manifest["phase"], "combat" if source_steps == 2 else "full")
                self.assertFalse((run / "defense.zip").exists())
                self.assertEqual((run / "combat.zip").exists(), source_steps == 2)
                final, final_manifest = load_checkpoint(run / "final.zip", self.new, device="cpu")
                self.assertEqual(final.num_timesteps, 6)
                self.assertEqual(final._n_updates, 3)
                self.assertEqual(final_manifest["progress"], {"combat": 4, "defense": 0, "full": 2})
                self.assertFalse(torch.equal(source.policy.value_net.weight, final.policy.value_net.weight))
                self.assertGreater(final.policy.optimizer.state_dict()["state"][0]["step"].item(),
                                   source.policy.optimizer.state_dict()["state"][0]["step"].item())
                run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(final_manifest["continuation"], manifest["continuation"])
                self.assertEqual(run_manifest["continuation"], manifest["continuation"])
                self.assertEqual(final_manifest["warm_start"], manifest["warm_start"])
                self.assertEqual(run_manifest["warm_start"], manifest["warm_start"])
                self.assertEqual(manifest["warm_start"]["source_num_timesteps"], 53760)
                self.assertEqual(manifest["warm_start"]["source_warm_start"]["source_num_timesteps"], 176640)
                resumed_run = self.root / f"resumed-{source_steps}"
                argv = self.training_argv(run / "final.zip", resumed_run)
                argv[0] = "--resume"
                argv[argv.index("--steps") + 1] = "8"
                with patch("train.make_env", self.fixture_factory), \
                        patch("train.evaluate", partial(train.evaluate, env_factory=self.fixture_factory)), \
                        patch("train.runtime_signature", return_value=self.new), redirect_stdout(io.StringIO()):
                    train.main(argv)
                resumed, resumed_manifest = load_checkpoint(resumed_run / "final.zip", self.new, device="cpu")
                self.assertEqual(resumed.num_timesteps, 8)
                self.assertEqual(resumed._n_updates, 4)
                self.assertEqual(resumed_manifest["progress"], {"combat": 4, "defense": 0, "full": 4})
                self.assertEqual(resumed_manifest["warm_start"], manifest["warm_start"])
                self.assertEqual(resumed_manifest["continuation"], manifest["continuation"])

    def test_invalid_continuation_does_not_create_output_run(self):
        import train
        _, path = self.checkpoint()
        self.rewrite_source(path, lambda data, params: data.update(_n_updates=0))
        run = self.root / "invalid-run"
        with patch("train.make_env", self.fixture_factory), \
                patch("train.runtime_signature", return_value=self.new), self.assertRaises(ValueError):
            train.main(self.training_argv(path, run))
        self.assertFalse(run.exists())

    def test_actual_347136_checkpoint_is_loaded_read_only_without_losing_combat_budget(self):
        path = (Path(__file__).resolve().parent / "runs" / "20260928-tactics-warm53760-g099"
                / "checkpoints" / "step-000000347136.zip")
        if not path.is_file():
            self.skipTest("approved local source checkpoint is not present")
        before_hash = sha256(path)
        self.assertEqual(before_hash, "abf44ff717cb7856139c9bfae8163d2455087026a5651d7b5da88f2ca1180600")
        sidecar = path.with_suffix(".zip.manifest.json")
        before_manifest = sidecar.read_bytes()
        manifest = json.loads(before_manifest)
        config = TrainConfig(**{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in manifest["training"].items()
        })
        source = MaskablePPO.load(path, device="cpu")
        loaded, previous, provenance = self.load(path, resume_config=replace(config, device="cpu"))
        self.assert_model_preserved(source, loaded)
        self.assertEqual(loaded.num_timesteps, 347136)
        self.assertEqual(loaded._n_updates, 904)
        self.assertEqual(previous, manifest)
        self.assertEqual(config.combat_steps - previous["progress"]["combat"], 152864)
        self.assertEqual(provenance["source_progress"], {"combat": 347136, "defense": 0, "full": 0})
        self.assertEqual(previous["warm_start"]["source_num_timesteps"], 53760)
        self.assertEqual(previous["warm_start"]["source_warm_start"]["source_num_timesteps"], 176640)
        self.assertEqual(sha256(path), before_hash)
        self.assertEqual(sidecar.read_bytes(), before_manifest)


class RecoveryWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_recovery_profile_copies_only_policy_into_a_fresh_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_model(94)
            source.learn(total_timesteps=2)
            _, old = boundary_signatures()
            old["source_sha256"] = dict(migration.V6_SOURCE_HASHES)
            config = TrainConfig(n_steps=2, batch_size=2, n_epochs=1, seed=12)
            manifest = make_manifest(config, old)
            manifest["continuation"] = {"source_num_timesteps": 347136}
            path = Path(directory) / "source.zip"
            save_checkpoint(path, source, manifest, phase="full",
                            progress={"combat": 0, "defense": 0, "full": 2})
            signature = copy.deepcopy(old)
            signature.update(env_version="grid26-events-v7-clean-terrain", reward_profile="shaped")
            signature["source_sha256"].update({
                name: "a" * 64 for name in
                ("game_state.py", "battle_city_env.py", "experiments.py", "migration.py",
                 "skills.py", "ablation.py")
            })
            loaded, provenance = migration.load_recovery_weights(path, signature)
            target = make_model(94)
            target.policy.load_state_dict(loaded.policy.state_dict(), strict=True)
            for name, tensor in source.policy.state_dict().items():
                torch.testing.assert_close(target.policy.state_dict()[name], tensor, rtol=0, atol=0)
            self.assertEqual(target.num_timesteps, 0)
            self.assertEqual(target._n_updates, 0)
            self.assertFalse(target.policy.optimizer.state)
            self.assertEqual(provenance["source_num_timesteps"], 2)
            self.assertEqual(provenance["source_continuation"]["source_num_timesteps"], 347136)
            self.assertTrue(provenance["reset"]["optimizer"])
            with self.assertRaisesRegex(ValueError, "incompatible checkpoint runtime"):
                load_checkpoint(path, signature, device="cpu")
            for field, value in (("env_version", "unreviewed"), ("reward_profile", "bad"),
                                 ("rom_sha256", "changed"), ("stacked_observation_shape", [47, 26, 26])):
                with self.subTest(field=field):
                    invalid = copy.deepcopy(signature)
                    invalid[field] = value
                    with self.assertRaises(ValueError):
                        migration.load_recovery_weights(path, invalid)
            signature["reward_profile"] = "sparse"
            migration.load_recovery_weights(path, signature)
            manifest["runtime"]["source_sha256"]["game_state.py"] = "a" * 64
            save_checkpoint(path, source, manifest, phase="full",
                            progress={"combat": 0, "defense": 0, "full": 2})
            with self.assertRaisesRegex(ValueError, "incompatible checkpoint runtime"):
                migration.load_recovery_weights(path, signature)


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.migration = migration
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_all_weights_and_policy_outputs_survive_per_frame_expansion(self):
        for nstack in (1, 2):
            with self.subTest(nstack=nstack):
                source = make_model(39 * nstack)
                source.learn(total_timesteps=2)
                source.num_timesteps = 176640
                source._n_updates = 460
                target = make_model(45 * nstack)
                self.migration.copy_warm_start_weights(source, target, nstack=nstack)
                old, new = source.policy.state_dict(), target.policy.state_dict()
                self.assertEqual(old.keys(), new.keys())
                for key in old:
                    if key in INPUT_KEYS:
                        for frame in range(nstack):
                            torch.testing.assert_close(new[key][:, frame * 45:frame * 45 + 39],
                                                       old[key][:, frame * 39:(frame + 1) * 39],
                                                       rtol=0, atol=0)
                            self.assertEqual(torch.count_nonzero(
                                new[key][:, frame * 45 + 39:(frame + 1) * 45]).item(), 0)
                    else:
                        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
                old_obs = torch.randn(3, 39 * nstack, 26, 26)
                new_obs = torch.randn(3, 45 * nstack, 26, 26) * 17
                for frame in range(nstack):
                    new_obs[:, frame * 45:frame * 45 + 39] = old_obs[:, frame * 39:(frame + 1) * 39]
                with torch.no_grad():
                    torch.testing.assert_close(source.policy.predict_values(old_obs),
                                               target.policy.predict_values(new_obs), rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(
                        source.policy.get_distribution(old_obs).distribution.logits,
                        target.policy.get_distribution(new_obs).distribution.logits, rtol=1e-5, atol=1e-6)
                self.assertTrue(source.policy.optimizer.state)
                self.assertEqual(target.policy.optimizer.state, {})
                self.assertEqual(target.num_timesteps, 0)
                self.assertEqual(target._n_updates, 0)
                self.assertEqual(target._current_progress_remaining, 1)

    def test_tactics_expansion_copies_every_tensor_alias_and_ignores_arbitrary_new_inputs(self):
        for nstack in (1, 2):
            with self.subTest(nstack=nstack):
                source = make_model(45 * nstack)
                source.learn(total_timesteps=2)
                source.num_timesteps, source._n_updates = 53760, 140
                target = make_model(47 * nstack)
                self.migration.copy_warm_start_weights(source, target, nstack=nstack)
                old, new = source.policy.state_dict(), target.policy.state_dict()
                self.assertEqual(old.keys(), new.keys())
                for name in old:
                    if name in INPUT_KEYS:
                        for frame in range(nstack):
                            torch.testing.assert_close(
                                new[name][:, frame * 47:frame * 47 + 45],
                                old[name][:, frame * 45:(frame + 1) * 45], rtol=0, atol=0)
                            self.assertEqual(torch.count_nonzero(
                                new[name][:, frame * 47 + 45:(frame + 1) * 47]).item(), 0)
                    else:
                        torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
                    self.assertNotEqual(old[name].data_ptr(), new[name].data_ptr())
                observation = torch.randn(3, 45 * nstack, 26, 26)
                expanded = torch.randn(3, 47 * nstack, 26, 26) * 17
                for frame in range(nstack):
                    expanded[:, frame * 47:frame * 47 + 45] = observation[:, frame * 45:(frame + 1) * 45]
                with torch.no_grad():
                    values = target.policy.predict_values(expanded)
                    logits = target.policy.get_distribution(expanded).distribution.logits
                    torch.testing.assert_close(source.policy.predict_values(observation),
                                               values, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(
                        source.policy.get_distribution(observation).distribution.logits,
                        logits, rtol=1e-5, atol=1e-6)
                    for frame in range(nstack):
                        expanded[:, frame * 47 + 45:(frame + 1) * 47] = 0
                    torch.testing.assert_close(target.policy.predict_values(expanded), values, rtol=0, atol=0)
                    torch.testing.assert_close(
                        target.policy.get_distribution(expanded).distribution.logits, logits, rtol=0, atol=0)
                self.assertTrue(source.policy.optimizer.state)
                self.assertEqual(target.policy.optimizer.state, {})
                self.assertEqual(target.num_timesteps, 0)
                self.assertEqual(target._n_updates, 0)
                self.assertEqual(target._current_progress_remaining, 1)
                self.assertIsNone(target._last_obs)

    def test_tactics_only_accepts_reviewed_transition_hashes_and_appended_channels(self):
        old, new = tactics_signatures()
        manifest = make_manifest(TrainConfig(), old)
        self.migration.validate_warm_start_manifest(manifest, new, nstack=2)
        for side in ("source", "destination"):
            for field, value in [
                ("env_version", "grid26-events-unreviewed"),
                ("obs_channels", 39), ("observation_shape", [39, 26, 26]),
                ("observation_dtype", "float64"), ("stacked_observation_shape", [45, 26, 26]),
                ("stack_axis", "last"), ("rom_sha256", "b" * 64),
                ("channel_names", list(reversed(NEW_NAMES))),
                ("channel_names", list(NEW_NAMES[:-1]) + ["unknown"]),
                ("action_space", {"type": "Discrete", "shape": [], "n": 14}),
                ("source_sha256", {}), ("unexpected", True),
            ]:
                with self.subTest(side=side, field=field):
                    changed_manifest, changed_signature = copy.deepcopy(manifest), copy.deepcopy(new)
                    runtime = changed_manifest["runtime"] if side == "source" else changed_signature
                    runtime[field] = value
                    with self.assertRaises(ValueError):
                        self.migration.validate_warm_start_manifest(
                            changed_manifest, changed_signature, nstack=2)
            for name in V4_SOURCE_HASHES:
                if side == "destination" and name in (
                    "battle_city_env.py", "heuristic.py", "migration.py", "game_state.py", "experiments.py",
                ):
                    continue
                with self.subTest(side=side, source_file=name):
                    changed_manifest, changed_signature = copy.deepcopy(manifest), copy.deepcopy(new)
                    runtime = changed_manifest["runtime"] if side == "source" else changed_signature
                    runtime["source_sha256"][name] = "f" * 64
                    with self.assertRaises(ValueError):
                        self.migration.validate_warm_start_manifest(
                            changed_manifest, changed_signature, nstack=2)
        for nstack in (0, -1, True, 1, 3):
            with self.subTest(nstack=nstack), self.assertRaises(ValueError):
                self.migration.validate_warm_start_manifest(manifest, new, nstack=nstack)
        changed = copy.deepcopy(manifest)
        changed["training"]["nstack"] = 1
        with self.assertRaisesRegex(ValueError, "nstack"):
            self.migration.validate_warm_start_manifest(changed, new, nstack=2)
        for destination in (
            "grid26-events-v4-residual", "grid26-events-v5-fire-safety",
            "grid26-events-v5-threat", "grid26-events-v6",
        ):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.migration.validate_warm_start_manifest(
                    manifest, dict(new, env_version=destination), nstack=2)
        for names in (
            list(NEW_NAMES), list(NEW_NAMES) + ["bullet_danger"],
            list(NEW_NAMES) + ["bullet_danger", "unknown"],
            list(NEW_NAMES) + ["enemy_drops_item", "bullet_danger"],
            ["bullet_danger", "enemy_drops_item"] + list(NEW_NAMES),
        ):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.migration.validate_warm_start_manifest(
                    manifest, dict(new, channel_names=names), nstack=2)
        same_layout = dict(old, env_version="grid26-events-v5-fire-safety",
                           source_sha256=new["source_sha256"])
        with self.assertRaises(ValueError):
            self.migration.validate_warm_start_manifest(manifest, same_layout, nstack=2)
        intermediate = dict(new, env_version="grid26-events-v5-threat", obs_channels=46,
                            observation_shape=[46, 26, 26], stacked_observation_shape=[92, 26, 26],
                            channel_names=list(NEW_NAMES) + ["bullet_danger"])
        with self.assertRaises(ValueError):
            self.migration.validate_warm_start_manifest(manifest, intermediate, nstack=2)
        with self.assertRaises(ValueError):
            self.migration.validate_warm_start_manifest(
                make_manifest(TrainConfig(), signatures()[0]), new, nstack=2)

    def test_tactics_load_preserves_ancestry_and_v5_roundtrip_without_relaxing_resume(self):
        source = make_model(90)
        source.learn(total_timesteps=2)
        source.num_timesteps, source._n_updates = 53760, 140
        old, new = tactics_signatures()
        path = self.root / "source-v4.zip"
        manifest = make_manifest(TrainConfig(), old)
        manifest["warm_start"] = {
            "method": "v3-to-v4-per-frame-input-expansion-zero-new-channels-weights-only",
            "source_num_timesteps": 176640, "source_n_updates": 460,
            "source_runtime": copy.deepcopy(SOURCE_RUNTIME),
            "source_checkpoint": "grandparent.zip", "source_checkpoint_sha256": "c" * 64,
        }
        save_checkpoint(path, source, manifest, phase="combat",
                        progress={"combat": 53760, "defense": 0, "full": 0})
        before = path.read_bytes(), path.with_suffix(".zip.manifest.json").read_bytes()
        loaded, provenance = self.migration.load_warm_start(path, new, nstack=2)
        self.assertEqual(str(loaded.device), "cpu")
        self.assertIsNone(loaded.get_env())
        self.assertEqual(provenance["method"], "v4-to-v5-per-frame-tactics-input-expansion-weights-only")
        self.assertEqual(provenance["source_num_timesteps"], 53760)
        self.assertEqual(provenance["source_n_updates"], 140)
        self.assertEqual(provenance["source_runtime"], old)
        self.assertEqual(provenance["source_warm_start"], manifest["warm_start"])
        self.assertEqual(provenance["old_channel_names"], list(NEW_NAMES))
        self.assertEqual(provenance["new_channel_names"], list(TACTICS_NAMES))
        self.assertEqual(provenance["frame_mapping"], [
            {"source_slice": [0, 45], "destination_slice": [0, 45], "zero_slice": [45, 47]},
            {"source_slice": [45, 90], "destination_slice": [47, 92], "zero_slice": [92, 94]},
        ])
        self.assertEqual(provenance["reset"], {
            "optimizer": True, "num_timesteps": 0, "n_updates": 0,
            "curriculum_progress": {"combat": 0, "defense": 0, "full": 0},
        })
        target = make_model(94)
        self.migration.copy_warm_start_weights(loaded, target, nstack=2)
        v5_manifest = make_manifest(TrainConfig(), new)
        v5_manifest["warm_start"] = provenance
        destination = self.root / "destination-v5.zip"
        save_checkpoint(destination, target, v5_manifest, phase="combat",
                        progress={"combat": 0, "defense": 0, "full": 0})
        roundtrip, saved = load_checkpoint(
            destination, new, device="cpu", resume_config=TrainConfig())
        self.assertEqual(saved["runtime"], new)
        self.assertEqual(saved["warm_start"], provenance)
        self.assertEqual(saved["progress"], {"combat": 0, "defense": 0, "full": 0})
        self.assertEqual(roundtrip.num_timesteps, 0)
        self.assertEqual(roundtrip._n_updates, 0)
        self.assertEqual(roundtrip._current_progress_remaining, 1)
        self.assertEqual(roundtrip.policy.optimizer.state, {})
        for name, tensor in target.policy.state_dict().items():
            torch.testing.assert_close(roundtrip.policy.state_dict()[name], tensor, rtol=0, atol=0)
        for checkpoint, signature in ((path, new), (destination, old)):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "runtime"):
                load_checkpoint(checkpoint, signature, device="cpu", resume_config=TrainConfig())
        self.assertEqual(before, (path.read_bytes(), path.with_suffix(".zip.manifest.json").read_bytes()))
        source.num_timesteps = 1
        save_checkpoint(path, source, manifest, phase="combat",
                        progress={"combat": 1, "defense": 0, "full": 0})
        sidecar = path.with_suffix(".zip.manifest.json")
        changed = json.loads(sidecar.read_text(encoding="utf-8"))
        changed["timesteps"] = 2
        sidecar.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.migration.load_warm_start(path, new, nstack=2)

    def test_state_mapping_rejects_unknown_missing_misshaped_or_inconsistent_aliases(self):
        mutations = [
            ("missing", lambda state: state.pop("value_net.bias")),
            ("extra", lambda state: state.update(unexpected=torch.zeros(1))),
            ("non-input shape", lambda state: state.update({"value_net.weight": torch.zeros(1, 65)})),
            ("input shape", lambda state: state.update({INPUT_KEYS[0]: torch.zeros(32, 77, 3, 3)})),
            ("alias", lambda state: state.update({INPUT_KEYS[1]: state[INPUT_KEYS[1]] + 1})),
            ("dtype", lambda state: state.update({"value_net.bias": state["value_net.bias"].double()})),
        ]
        for channels in (78, 90):
            old = make_model(channels).policy.state_dict()
            new = make_model(90 if channels == 78 else 94).policy.state_dict()
            for name, mutate in mutations:
                with self.subTest(channels=channels, name=name):
                    changed = dict(old)
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        self.migration.expand_policy_state_dict(changed, new, nstack=2)

    def test_copy_rejects_intermediate_same_layout_and_unreviewed_expansions(self):
        for source_channels, target_channels in (
            (45, 45), (45, 46), (39, 47), (45, 48), (46, 47), (47, 47),
        ):
            with self.subTest(source=source_channels, target=target_channels):
                source, target = make_model(source_channels * 2), make_model(target_channels * 2)
                with self.assertRaises(ValueError):
                    self.migration.copy_warm_start_weights(source, target, nstack=2)
                with self.assertRaises(ValueError):
                    self.migration.expand_policy_state_dict(
                        source.policy.state_dict(), target.policy.state_dict(), nstack=2)

    def test_tactics_load_rejects_serialized_alias_and_dtype_loss(self):
        source = make_model(90)
        old, new = tactics_signatures()
        path = self.root / "source-v4.zip"
        mutations = [
            ("aliases", lambda state: state.update({INPUT_KEYS[0]: state[INPUT_KEYS[0]] + 1})),
            ("dtype", lambda state: state.update({"value_net.bias": state["value_net.bias"].double()})),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                save_checkpoint(path, source, make_manifest(TrainConfig(), old),
                                phase="full", progress={"combat": 0, "defense": 0, "full": 0})
                data, params, variables = load_from_zip_file(path, device="cpu")
                mutate(params["policy"])
                save_to_zip_file(path, data=data, params=params, pytorch_variables=variables)
                sidecar = path.with_suffix(".zip.manifest.json")
                manifest = json.loads(sidecar.read_text(encoding="utf-8"))
                manifest["checkpoint_sha256"] = sha256(path)
                sidecar.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "tensor|alias"):
                    self.migration.load_warm_start(path, new, nstack=2)

    def test_tactics_load_rejects_model_layout_or_architecture_disguised_by_manifest(self):
        old, new = tactics_signatures()
        path = self.root / "source-v4.zip"
        for mutation in ("layout", "architecture"):
            with self.subTest(mutation=mutation):
                source = make_model(78 if mutation == "layout" else 90)
                if mutation == "architecture":
                    source.policy_kwargs["activation_fn"] = torch.nn.ReLU
                save_checkpoint(path, source, make_manifest(TrainConfig(), old),
                                phase="full", progress={"combat": 0, "defense": 0, "full": 0})
                with self.assertRaisesRegex(ValueError, "observation|architecture"):
                    self.migration.load_warm_start(path, new, nstack=2)

    def test_only_pinned_source_runtime_and_exact_target_layout_are_accepted(self):
        old, new = signatures()
        manifest = make_manifest(TrainConfig(), old)
        self.migration.validate_warm_start_manifest(manifest, new, nstack=2)
        for field, value in [
            ("env_version", "grid26-events-v2"), ("obs_channels", 38),
            ("observation_shape", [39, 13, 13]), ("observation_dtype", "float64"),
            ("stacked_observation_shape", [39, 26, 26]), ("stack_axis", "last"),
            ("rom_sha256", "b" * 64), ("source_sha256", {}),
            ("action_space", {"type": "Discrete", "shape": [], "n": 14}),
        ]:
            with self.subTest(source=field):
                changed = copy.deepcopy(manifest)
                changed["runtime"][field] = value
                with self.assertRaises(ValueError):
                    self.migration.validate_warm_start_manifest(changed, new, nstack=2)
        for name in old["source_sha256"]:
            with self.subTest(source_file=name):
                changed = copy.deepcopy(manifest)
                changed["runtime"]["source_sha256"][name] = "f" * 64
                with self.assertRaises(ValueError):
                    self.migration.validate_warm_start_manifest(changed, new, nstack=2)
        for field, value in [
            ("env_version", "grid26-events-v5"), ("obs_channels", 46),
            ("observation_shape", [45, 13, 13]), ("observation_dtype", "float64"),
            ("stacked_observation_shape", [45, 26, 26]), ("stack_axis", "last"),
            ("rom_sha256", "c" * 64), ("source_sha256", {}),
            ("action_space", {"type": "Discrete", "shape": [], "n": 14}),
            ("channel_names", list(reversed(NEW_NAMES))),
            ("channel_names", list(OLD_NAMES) + ["unknown"] * 6),
        ]:
            with self.subTest(target=field, value=value):
                changed = copy.deepcopy(new)
                changed[field] = value
                with self.assertRaises(ValueError):
                    self.migration.validate_warm_start_manifest(manifest, changed, nstack=2)
        with self.assertRaises(ValueError):
            self.migration.validate_warm_start_manifest(manifest, new, nstack=1)
        with self.assertRaisesRegex(ValueError, "runtime"):
            from experiments import validate_manifest
            validate_manifest(manifest, new, resume_config=TrainConfig())

    def test_checkpoint_load_is_cpu_envless_and_provenance_preserves_source_steps(self):
        source = make_model(78)
        source.learn(total_timesteps=2)
        source.num_timesteps = 176640
        old, new = signatures()
        path = self.root / "source.zip"
        save_checkpoint(path, source, make_manifest(TrainConfig(), old),
                        phase="combat", progress={"combat": 176640, "defense": 0, "full": 0})
        loaded, provenance = self.migration.load_warm_start(path, new, nstack=2)
        self.assertEqual(str(loaded.device), "cpu")
        self.assertIsNone(loaded.get_env())
        self.assertEqual(provenance["source_num_timesteps"], 176640)
        self.assertEqual(provenance["source_checkpoint"], str(path.resolve()))
        self.assertEqual(provenance["source_checkpoint_sha256"], sha256(path))
        self.assertEqual(provenance["source_runtime"], old)
        self.assertEqual(provenance["old_channel_names"], list(OLD_NAMES))
        self.assertEqual(provenance["new_channel_names"], list(NEW_NAMES))
        self.assertEqual(provenance["reset"], {
            "optimizer": True, "num_timesteps": 0, "n_updates": 0,
            "curriculum_progress": {"combat": 0, "defense": 0, "full": 0},
        })
        with self.assertRaisesRegex(ValueError, "runtime"):
            load_checkpoint(path, new, device="cpu", resume_config=TrainConfig())
        with path.open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.migration.load_warm_start(path, new, nstack=2)

    def test_copy_rejects_nonfresh_target_or_unsupported_architecture(self):
        source = make_model(78)
        target = make_model(90)
        target.num_timesteps = 2
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.migration.copy_warm_start_weights(source, target, nstack=2)
        target.num_timesteps = 0
        target._n_updates = 1
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.migration.copy_warm_start_weights(source, target, nstack=2)
        target._n_updates = 0
        target.learn(total_timesteps=2)
        target.num_timesteps = target._n_updates = 0
        target._current_progress_remaining = 1
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.migration.copy_warm_start_weights(source, target, nstack=2)
        target = make_model(90)
        target.policy.normalize_images = True
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.migration.copy_warm_start_weights(source, target, nstack=2)
        target.policy.normalize_images = False
        target.policy.features_extractor.cnn[1] = torch.nn.Sigmoid()
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.migration.copy_warm_start_weights(source, target, nstack=2)
        target.policy.features_extractor.cnn[1] = torch.nn.ReLU()
        target.policy.mlp_extractor.value_net[1] = torch.nn.Sigmoid()
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.migration.copy_warm_start_weights(source, target, nstack=2)

    def test_training_publishes_transferred_step_zero_and_restarts_curriculum(self):
        self._check_training_warm_start(source_channels=39, source_steps=176640)

    def test_tactics_training_publishes_expanded_weights_and_restarts_curriculum(self):
        self._check_training_warm_start(source_channels=45, source_steps=53760)

    def _check_training_warm_start(self, *, source_channels, source_steps):
        import train
        from test_experiments import FixtureEnv

        target_channels = 45 if source_channels == 39 else 47

        class ResidualFixtureEnv(FixtureEnv):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observation_space = spaces.Box(0, 1, (target_channels, 26, 26), np.float32)

            @staticmethod
            def expand(observation):
                result = np.zeros((target_channels, 26, 26), dtype=np.float32)
                result[:2] = observation
                return result

            def reset(self, **kwargs):
                observation, info = super().reset(**kwargs)
                return self.expand(observation), info

            def step(self, action):
                observation, *rest = super().step(action)
                return self.expand(observation), *rest

        source = make_model(source_channels * 2)
        source.learn(total_timesteps=2)
        source.num_timesteps, source._n_updates = source_steps, 460
        old, new = signatures() if source_channels == 39 else tactics_signatures()
        checkpoint = self.root / "source.zip"
        save_checkpoint(checkpoint, source, make_manifest(TrainConfig(), old),
                        phase="combat", progress={"combat": source_steps, "defense": 0, "full": 0})
        run = self.root / "warm-run"
        argv = [
            "--warm-start", str(checkpoint), "--output-dir", str(run),
            "--steps", "4", "--combat-steps", "2", "--n-steps", "2", "--batch-size", "2",
            "--n-epochs", "1", "--envs", "1", "--device", "cpu", "--nstack", "2",
            "--eval-stages", "4", "--eval-seeds", "1",
        ]
        with patch("train.make_env", ResidualFixtureEnv), \
                patch("train.evaluate", partial(train.evaluate, env_factory=ResidualFixtureEnv)), \
                patch("train.runtime_signature", return_value=new), redirect_stdout(io.StringIO()):
            train.main(argv)
        initial, initial_manifest = load_checkpoint(
            run / "checkpoints" / "step-000000000000.zip", new, device="cpu")
        self.assertEqual(initial.num_timesteps, 0)
        self.assertEqual(initial._n_updates, 0)
        self.assertEqual(initial.policy.optimizer.state, {})
        self.assertEqual(initial._current_progress_remaining, 1)
        self.assertEqual(initial_manifest["phase"], "combat")
        self.assertEqual(initial_manifest["progress"], {"combat": 0, "defense": 0, "full": 0})
        for name, weight in source.policy.state_dict().items():
            actual = initial.policy.state_dict()[name]
            if name in INPUT_KEYS:
                for frame in range(2):
                    torch.testing.assert_close(
                        actual[:, frame * target_channels:frame * target_channels + source_channels],
                        weight[:, frame * source_channels:(frame + 1) * source_channels], rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(
                        actual[:, frame * target_channels + source_channels:(frame + 1) * target_channels]), 0)
            else:
                torch.testing.assert_close(actual, weight, rtol=0, atol=0)
        final, final_manifest = load_checkpoint(run / "final.zip", new, device="cpu")
        self.assertEqual(final.num_timesteps, 4)
        self.assertEqual(final._n_updates, 2)
        self.assertTrue(final.policy.optimizer.state)
        self.assertEqual(final_manifest["progress"], {"combat": 2, "defense": 0, "full": 2})
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["warm_start"], initial_manifest["warm_start"])
        self.assertEqual(manifest["warm_start"], final_manifest["warm_start"])
        self.assertEqual(manifest["warm_start"]["source_num_timesteps"], source_steps)
        before = (run / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            train.main(argv)
        self.assertEqual((run / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
