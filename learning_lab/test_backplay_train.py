"""CPU/fake-vector acceptance tests; zero ROM/native environment decisions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from learning_lab import backplay_train as train
from learning_lab.victory_objective import OBJECTIVE_VERSION, SPARSE_OBJECTIVE_VERSION


def fixture_pool():
    remaining = np.array([1, 2, 4, 8, 12, 16, 20, 20], np.int64)
    rams = np.zeros((8, 2048), np.uint8)
    rams[:, 0] = np.arange(8)
    return dict(
        rams=rams, remaining=remaining, source_frames=np.array([100] * 7 + [-1], np.int64),
        seeds=np.zeros(8, np.int64), stages=np.ones(8, np.int64),
        candidate_ids=np.arange(8, dtype=np.int64),
        full_start=np.array([False] * 7 + [True]), noop_tested=np.array([True] * 7 + [False]),
        automatic_noop_clear_exclusions=[], unsupported_exclusions=[], missing_buckets=[],
        normal_start={"initialization_method": "rust_env.stage_ram(1, seed=0)",
                      "full_start": True, "stage": 1, "seed": 0,
                      "native_import_byte_exact": True, "native_support_flags": [0],
                      "ram_sha256": hashlib.sha256(rams[-1].tobytes()).hexdigest()},
    )


class FakeBudget:
    def __init__(self, remaining_seconds=3600):
        self.rows, self.remaining_seconds = [], remaining_seconds

    def snapshot(self):
        charged = sum(row["reserved"] if row["actual"] is None else row["actual"] for row in self.rows)
        return dict(max_interactions=4000000, remaining_interactions=4000000 - charged,
                    remaining_seconds=self.remaining_seconds, deadline_epoch=1e20,
                    charged_interactions=charged)

    def reserve(self, label, interactions):
        token = str(len(self.rows))
        self.rows.append(dict(token=token, label=label, reserved=interactions, actual=None))
        return token

    def settle(self, token, interactions):
        row = self.rows[int(token)]
        assert 0 <= interactions <= row["reserved"]
        row["actual"] = interactions


class FakeRuntime:
    rollout_size = 8192
    instances = []

    def __init__(self, *, warm=None, **kwargs):
        self.completed_steps = self.optimized_steps = self.adam_steps = 0
        self.collect_amount, self.collect_ok, self.fail_collect = self.rollout_size, True, False
        self.warm_input = deepcopy(warm)
        self.policy_version = 0 if warm is None else warm["version"]
        self.model = FakePolicy()
        self.instances.append(self)

    def collect(self):
        self.completed_steps += self.collect_amount
        if self.fail_collect:
            raise RuntimeError("uncertain native collection failure")
        return self.collect_ok

    def optimize(self):
        self.optimized_steps += self.rollout_size
        self.adam_steps += 1
        self.policy_version += 1

    def export_warm(self):
        return {"version": self.policy_version, "adam": self.adam_steps}

    def close(self):
        pass


class FakeMonitor:
    def __init__(self, path, *, device):
        self.latest = {}
        Path(path).touch(exist_ok=False)

    def check(self, *args, **kwargs):
        pass


class FakeNative:
    def __init__(self, initial, max_frames, *, seeds):
        self.n, self.calls = len(initial), 0
        self.initial = initial.copy()
        self.max_frames, self.seeds = max_frames, seeds
        self.states = np.zeros((self.n, 2048), np.uint8)
        self.states[:, 0x68] = 0x80
        self.states[:, 0x51] = 3
        self.states[13:, 0x68] = 0
        self.closed = False

    def observe(self):
        return np.zeros((self.n, 47, 26, 26), np.float32)

    def masks(self):
        return np.ones((self.n, 10), bool)

    def step(self, actions):
        self.calls += 1
        dones = np.arange(self.n) < (8 if self.calls == 1 else self.n)
        stats = np.zeros((self.n, 5), np.uint32)
        stats[:, 0] = self.calls * 4
        return self.observe(), np.zeros(self.n), dones, stats

    def ram(self, index):
        return self.states[index].copy()

    def close(self):
        self.closed = True


class FakePolicy:
    def predict(self, observation, *, deterministic, action_masks):
        assert deterministic
        return np.zeros(len(observation), np.int64), None


class CounterVec(VecEnv):
    def __init__(self, count=1):
        self.ticks = np.zeros(count, np.int64)
        self.batch = SimpleNamespace(stats=np.zeros((count, 5), np.uint32))
        super().__init__(count, gym.spaces.Box(-1, 1, (47, 26, 26), np.float32),
                         gym.spaces.Discrete(10))

    def reset(self):
        self.ticks.fill(0)
        self.batch.stats.fill(0)
        return np.zeros((self.num_envs, 47, 26, 26), np.float32)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.ticks += 1
        self.batch.stats[:, 1] = self.ticks
        dones = self.ticks == 2
        infos = [{} for _ in range(self.num_envs)]
        for index in np.flatnonzero(dones):
            infos[index] = dict(outcome="timeout", episode={"r": 0., "l": 2},
                                terminal_observation=np.zeros((47, 26, 26), np.float32),
                                **{"TimeLimit.truncated": True})
            self.ticks[index] = 0
        return (np.zeros((self.num_envs, 47, 26, 26), np.float32),
                np.zeros(self.num_envs, np.float32), dones, infos)

    def close(self):
        pass

    def get_attr(self, name, indices=None):
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        pass

    def env_method(self, name, *args, indices=None, **kwargs):
        if name == "action_masks":
            return [np.ones(10, bool) for _ in self._get_indices(indices)]
        raise AttributeError(name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class BackplayTrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path("runs") / ("backplay-train-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.pool_path = self.root / "pool"
        self.pool_path.mkdir()
        for name in ("starts.npz", "normal_start.npz", "normal_start.json", "report.json"):
            (self.pool_path / name).write_bytes(b"synthetic fixture identity")
        FakeRuntime.instances = []

    def test_levels_distinguish_remaining20_curriculum_and_normal(self):
        levels = train.build_levels(fixture_pool())
        self.assertEqual([v["remaining"] for v in levels], [1, 2, 4, 8, 12, 16, 20, 20])
        self.assertEqual([v["full_start"] for v in levels], [False] * 7 + [True])
        self.assertEqual(levels[-2]["indices"].tolist(), [6])
        self.assertEqual(levels[-1]["indices"].tolist(), [7])
        self.assertEqual([v["max_frames"] for v in levels], [2500, 3000, 4000, 6000, 8000, 10000, 12000, 12000])

    def test_bad_pool_automatic_clear_missing_bucket_and_fake_normal_rejected(self):
        for field, value in (("automatic_noop_clear_exclusions", [0]),
                             ("unsupported_exclusions", [0])):
            pool = fixture_pool()
            pool[field] = value
            with self.assertRaises(ValueError):
                train.build_levels(pool)
        pool = fixture_pool()
        pool["normal_start"]["initialization_method"] = "teacher remaining20"
        with self.assertRaises(ValueError):
            train.build_levels(pool)

    def test_reward_profile_and_gamma_are_explicit(self):
        contract = train.experiment_contract()
        self.assertEqual(contract["reward_objective"], SPARSE_OBJECTIVE_VERSION)
        self.assertFalse(contract["shaping"])
        self.assertEqual(contract["ppo"]["gamma"], 1.)
        self.assertEqual(contract["ppo"]["n_envs"], 32)
        self.assertEqual(contract["ppo"]["n_steps"], 256)
        self.assertFalse(contract["timeout_bootstrap"])
        self.assertFalse(contract["exact_simulator_resume"])
        with patch.object(train, "_PPO", dict(train._PPO, gamma=.99)):
            with self.assertRaises(ValueError):
                train._Runtime(pool=fixture_pool(), level=train.build_levels(fixture_pool())[0],
                               seed=0, device="cpu", guard=None, max_steps=32768)

    def test_counter_provider_uses_native_stats_without_per_step_kill_info(self):
        raw = CounterVec()
        wrapped = train.wrap_victory(raw, np.array([4]))
        self.addCleanup(wrapped.close)
        wrapped.reset()
        _, first, _, _ = wrapped.step(np.array([0]))
        _, last, dones, infos = wrapped.step(np.array([0]))
        self.assertEqual(float(first[0]), 0.)
        self.assertEqual(float(last[0]), -1.)
        self.assertEqual(infos[0]["episode"]["r"], -1.)
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertEqual(infos[0]["reward_profile"], wrapped.objective_version)
        self.assertEqual(wrapped.objective_version, SPARSE_OBJECTIVE_VERSION)

    def test_rollout_settles_clean_partial_but_failed_collection_stays_reserved(self):
        runtime, budget = FakeRuntime(), FakeBudget()
        runtime.collect_amount, runtime.collect_ok = 64, False
        row = train.collect_reserved(runtime, budget, "partial")
        self.assertFalse(row["complete"])
        self.assertEqual(budget.rows[0]["actual"], 64)
        runtime.fail_collect = True
        row = train.collect_reserved(runtime, budget, "failed")
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(budget.rows[1]["actual"])
        self.assertEqual(budget.rows[1]["reserved"], 8192)

    def test_real_fixture_ledger_retains_failed_rollout_upper_charge(self):
        ledger = train.ResearchBudget.create(self.root / "fixture-budget.sqlite",
                                             max_interactions=20000, wall_seconds=60)
        runtime = FakeRuntime()
        runtime.collect_amount, runtime.fail_collect = 32, True
        result = train.collect_reserved(runtime, ledger, "synthetic-failure")
        snapshot = ledger.snapshot()
        self.assertFalse(result["settled"])
        self.assertEqual(snapshot["charged_interactions"], 8192)
        self.assertEqual(snapshot["settled_interactions"], 0)
        self.assertIsNone(snapshot["reservations"][0]["actual"])

    def test_unverified_or_unoptimized_weights_are_not_saved(self):
        level = train.build_levels(fixture_pool())[0]
        for runtime in (SimpleNamespace(optimized_steps=0, state_verified=True),
                        SimpleNamespace(optimized_steps=8192, state_verified=False)):
            with self.assertRaises(ValueError):
                train._save_checkpoint(runtime, self.root, level, {},
                                       optimized_total=8192, lineage=None, role="forbidden")

    def test_eval_counts_finished_slots_and_only_first_16_episodes(self):
        budget = FakeBudget()
        guard = SimpleNamespace(check=lambda phase: None)
        level = train.build_levels(fixture_pool())[0]
        with patch.object(train, "NativeBatch", FakeNative):
            result = train.evaluate_native(FakePolicy(), fixture_pool(), level, budget=budget,
                                           guard=guard, seed=0, evaluation_index=0, label="eval")
        self.assertEqual(result["completed_decisions"], 32)
        self.assertEqual(len(result["episodes"]), 16)
        self.assertEqual(result["clears"], 13)
        self.assertEqual(budget.rows[0]["reserved"], 16 * 625)
        self.assertEqual(budget.rows[0]["actual"], 32)
        self.assertTrue(train.passes_gate(result))
        self.assertFalse(train.passes_gate(dict(result, status="partial")))

    def run_fake(self, evaluations, **kwargs):
        iterator = iter(evaluations)

        def save(runtime, output, level, provenance, *, optimized_total, lineage, role):
            path = output / f"fake-{level['index']}-{optimized_total}-{role}.zip"
            path.write_bytes(b"optimized fixture")
            return {"path": str(path), "sha256": "a" * 64, "optimized_steps": optimized_total}

        with patch.object(train, "load_backplay_pool", return_value=fixture_pool()), \
                patch.object(train, "_make_runtime", side_effect=lambda **kw: FakeRuntime(**kw)), \
                patch.object(train, "_save_checkpoint", side_effect=save), \
                patch.object(train, "evaluate_native", side_effect=lambda *a, **kw: next(iterator)), \
                patch.object(train, "ResourceMonitor", FakeMonitor), \
                patch.object(train, "_GLOBAL_DEADLINE_EPOCH", 1e20):
            return train.run_backplay(self.pool_path, self.root / "out", kwargs.pop("budget", FakeBudget()),
                                      max_steps_per_level=32768, **kwargs)

    def test_no_advance_when_131072_style_cap_has_no_gate(self):
        result = self.run_fake([dict(status="completed", clears=12, episodes=[{}] * 16,
                                     completed_decisions=16, reserved=16)])
        self.assertEqual(result["stop_reason"], "level_gate_not_met")
        self.assertEqual(len(result["levels"]), 1)
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(result["optimized_steps"], 32768)

    def test_policy_and_adam_lineage_carries_only_after_gate(self):
        result = self.run_fake([
            dict(status="completed", clears=13, episodes=[{}] * 16, completed_decisions=16, reserved=16),
            dict(status="completed", clears=0, episodes=[{}] * 16, completed_decisions=16, reserved=16),
        ])
        self.assertEqual(len(result["levels"]), 2)
        self.assertIsNone(FakeRuntime.instances[0].warm_input)
        self.assertEqual(FakeRuntime.instances[1].warm_input, {"version": 4, "adam": 4})
        self.assertIsNotNone(result["levels"][1]["parent_checkpoint"])
        self.assertFalse(result["exact_simulator_resume"])

    def test_expired_global_budget_does_not_initialize_a_runtime(self):
        result = self.run_fake([], budget=FakeBudget(remaining_seconds=0))
        self.assertEqual(result["stop_reason"], "deadline")
        self.assertEqual(result["optimized_steps"], 0)
        self.assertFalse(FakeRuntime.instances)

    def test_clean_partial_rollout_never_optimizes_or_saves(self):
        def runtime_factory(**kwargs):
            result = FakeRuntime(**kwargs)
            result.collect_amount, result.collect_ok = 32, False
            return result

        budget = FakeBudget()
        with patch.object(train, "load_backplay_pool", return_value=fixture_pool()), \
                patch.object(train, "_make_runtime", side_effect=runtime_factory), \
                patch.object(train, "ResourceMonitor", FakeMonitor), \
                patch.object(train, "_GLOBAL_DEADLINE_EPOCH", 1e20):
            report = train.run_backplay(self.pool_path, self.root / "partial", budget,
                                        max_steps_per_level=32768)
        self.assertEqual(report["optimized_steps"], 0)
        self.assertFalse(report["checkpoints"])
        self.assertEqual(budget.rows[0]["actual"], 32)
        self.assertEqual(FakeRuntime.instances[0].adam_steps, 0)

    def test_tiny_real_cpu_ppo_update_uses_compact_buffer_and_fake_native_stats(self):
        pool = fixture_pool()
        level = train.build_levels(pool)[2]
        guard = SimpleNamespace(check=lambda phase: None)
        config = dict(train._PPO, n_steps=2, batch_size=32, n_epochs=1)
        with patch.object(train, "_PPO", config), \
                patch.object(train, "RustVectorEnv", side_effect=lambda initial, *a, **kw: CounterVec(len(initial))):
            runtime = train._Runtime(pool=pool, level=level, seed=0, device="cpu",
                                     guard=guard, max_steps=64)
            try:
                runtime.rollout_size = 64
                result = train.collect_reserved(runtime, FakeBudget(), "tiny-fixture")
                self.assertTrue(result["complete"])
                self.assertEqual(result["completed_decisions"], 64)
                runtime.optimize()
                self.assertEqual(runtime.optimized_steps, 64)
                self.assertEqual(runtime.adam_steps, 2)
                self.assertIsInstance(runtime.model.rollout_buffer, train.NativeGPURunRolloutBuffer)
                self.assertFalse(torch.cuda.is_initialized())
            finally:
                runtime.close()

    def test_cpu_checkpoint_strict_reload_warm_adam_and_interrupted_update_rollback(self):
        pool = fixture_pool()
        level = train.build_levels(pool)[0]
        guard = SimpleNamespace(check=lambda phase: None)
        with patch.object(train, "RustVectorEnv", side_effect=lambda initial, *a, **kw: CounterVec(len(initial))):
            runtime = train._Runtime(pool=pool, level=level, seed=0, device="cpu",
                                     guard=guard, max_steps=32768)
            second = None
            try:
                # A single synthetic Adam step builds a tiny checkpoint fixture, not a research rollout.
                observations = torch.zeros(1, 94, 26, 26)
                logits = runtime.model.policy.get_distribution(observations).distribution.logits
                runtime.model.policy.optimizer.zero_grad(set_to_none=True)
                (-logits[:, 0].mean()).backward()
                runtime.model.policy.optimizer.step()
                runtime.optimized_steps, runtime.adam_steps = 8192, 1
                runtime.model._n_updates = 1
                warm = runtime.export_warm()
                runtime._last_good = warm
                second = train._Runtime(pool=pool, level=train.build_levels(pool)[1], seed=1,
                                        device="cpu", guard=guard, max_steps=32768, warm=warm)
                for name, value in warm["policy"].items():
                    torch.testing.assert_close(value, second.model.policy.state_dict()[name], rtol=0, atol=0)
                self.assertTrue(second.model.policy.optimizer.state_dict()["state"])
                actual_optimizer = second.model.policy.optimizer.state_dict()
                for index, fields in warm["optimizer"]["state"].items():
                    for key, value in fields.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, actual_optimizer["state"][index][key],
                                                       rtol=0, atol=0)
                        else:
                            self.assertEqual(value, actual_optimizer["state"][index][key])
                identity, start_ids = train._pool_identity(self.pool_path, pool, train.build_levels(pool))
                provenance = dict(sources=train._sources(), pool=identity, start_identities=start_ids)
                checkpoint = train._save_checkpoint(runtime, self.root, level, provenance,
                                                     optimized_total=8192, lineage=None, role="cpu_fixture")
                loaded = train.load_backplay_checkpoint(checkpoint["path"])
                self.assertEqual(loaded.num_timesteps, 8192)
                self.assertEqual(loaded.gamma, 1.)
                self.assertEqual(loaded.n_steps, 2)
                self.assertEqual(loaded.n_envs, 1)
                self.assertTrue(loaded.policy.optimizer.state_dict()["state"])
                for name, value in runtime.model.policy.state_dict().items():
                    torch.testing.assert_close(value, loaded.policy.state_dict()[name], rtol=0, atol=0)
                sidecar = Path(checkpoint["path"]).with_suffix(".manifest.json")
                original = sidecar.read_text()
                self.assertEqual(json.loads(original)["contract"]["reward_objective"],
                                 runtime.objective_version)
                self.assertEqual(runtime.objective_version, SPARSE_OBJECTIVE_VERSION)
                for field, value in (("format", "legacy"), ("sources", {}), ("pool", {})):
                    manifest = json.loads(original)
                    manifest[field] = value
                    sidecar.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        train.load_backplay_checkpoint(checkpoint["path"])
                manifest = json.loads(original)
                manifest["contract"]["ppo"]["gamma"] = .99
                sidecar.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    train.load_backplay_checkpoint(checkpoint["path"])
                manifest = json.loads(original)
                manifest["contract"]["reward_objective"] = OBJECTIVE_VERSION
                sidecar.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    train.load_backplay_checkpoint(checkpoint["path"])
                sidecar.write_text(original)

                def interrupted():
                    with torch.no_grad():
                        next(runtime.model.policy.parameters()).add_(1)
                    raise train.DeadlineReached("synthetic optimizer deadline")

                with patch.object(runtime.model, "train", side_effect=interrupted):
                    with self.assertRaises(train.DeadlineReached):
                        runtime.optimize()
                for name, value in warm["policy"].items():
                    torch.testing.assert_close(value, runtime.model.policy.state_dict()[name], rtol=0, atol=0)
                self.assertEqual(runtime.optimized_steps, 8192)
            finally:
                if second is not None:
                    second.close()
                runtime.close()


if __name__ == "__main__":
    unittest.main()
