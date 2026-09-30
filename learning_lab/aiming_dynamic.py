"""Reset-safe dynamic aiming environment/factory; intentionally no PPO runner.

Use prepare_dynamic_aiming(...) for monitored real use, then
make_dynamic_aiming_env(prepared["sampler"]) for a32-environment, two-frame VecEnv.
The caller owns the pre-reserved allowance, training, optimized checkpoints and
strict model manifests. This component never launches a GPU job or writes a
shared ledger. Final holdout evaluation is separately guarded and single-use.
"""
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from learning_lab import aiming_task as scenes
from learning_lab.aiming_task import generate_aiming_starts
from learning_lab.aiming_train import evaluate_holdout
from learning_lab.victory_objective import FiniteVictoryVecEnv
from rust_env import RustVectorEnv, episode_seed
from rust_resources import ResourceMonitor

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dynamic-aiming-environment-v1"
FINAL_TEST_SEED = 20261002
PREVIOUS_HOLDOUTS = (
    ROOT / "runs" / "20260930-victory-aiming-task" / "test.npz",
    ROOT / "runs" / "20260930-victory-lab" / "aiming-compare-v1" / "new-test.npz",
)


class DynamicStop(RuntimeError):
    pass


class DynamicAccounting:
    def __init__(self, limit, *, deadline, monitor=None):
        if type(limit) is not int or not 1 <= limit <= 4000000:
            raise ValueError("Pass an already-reserved allowance in1..4000000")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) \
                or not math.isfinite(deadline):
            raise ValueError("deadline must be an absolute finite monotonic time")
        self.limit, self.deadline, self.monitor = limit, deadline, monitor
        self.rom_initialization_decisions = 0
        self.native_steps_attempted = self.native_steps_completed = 0
        self.evaluation_attempted = self.evaluation_completed = 0
        self.generator_batches = self.generated_templates = 0
        self.native_fresh_initializations = self.native_explicit_resets = 0
        self.native_old_layout_autoresets = 0
        self.reserved_evaluation = 0
        self.evaluation_started = False

    @property
    def total_decisions(self):
        return self.rom_initialization_decisions + self.native_steps_attempted + self.evaluation_attempted

    def check(self, phase):
        if time.monotonic() >= self.deadline:
            raise DynamicStop("Absolute dynamic-aiming deadline reached")
        if self.monitor is not None:
            self.monitor.check(phase)

    def take_step(self):
        self.check("native_training_step")
        if self.evaluation_started:
            raise DynamicStop("Training is forbidden after final-test evaluation starts")
        if self.total_decisions + self.reserved_evaluation + 1 > self.limit:
            raise DynamicStop("Reserved environment-decision allowance exhausted")
        self.native_steps_attempted += 1

    def generate(self, count, seed):
        self.check("layout_generation")
        boot_bound = 0 if scenes._shoot_reset_ram.cache_info().currsize else 2
        if self.total_decisions + self.reserved_evaluation + boot_bound > self.limit:
            raise DynamicStop("Insufficient allowance for possible seed0 ROM initialization")
        before = scenes._ROM_INITIALIZATION_DECISIONS
        try:
            data = generate_aiming_starts(count, int(seed))
        finally:
            self.rom_initialization_decisions += scenes._ROM_INITIALIZATION_DECISIONS - before
        self.generator_batches += 1
        self.generated_templates += count
        return data

    def snapshot(self):
        return dict(
            limit=self.limit, deadline_monotonic=self.deadline,
            rom_initialization_decisions=self.rom_initialization_decisions,
            native_steps_attempted=self.native_steps_attempted,
            native_steps_completed=self.native_steps_completed,
            evaluation_attempted=self.evaluation_attempted,
            evaluation_completed=self.evaluation_completed,
            total_decisions=self.total_decisions,
            generator_batches=self.generator_batches,
            generated_templates_validated_without_stepping=self.generated_templates,
            generator_native_validation_initializations=self.generated_templates,
            native_fresh_initializations=self.native_fresh_initializations,
            native_explicit_resets=self.native_explicit_resets,
            native_old_layout_autoresets=self.native_old_layout_autoresets,
            initialization_note="Native imports/resets advance zero frames; first cached SkillEnv setup costs2 raw ROM NOOPs.",
        )


class BalancedAimingSampler:
    def __init__(self, *, seed, excluded_hashes, accounting):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Sampler seed must be uint64")
        self.seed, self.accounting = seed, accounting
        self.excluded = frozenset(str(h) for h in excluded_hashes)
        self.seen = set()
        self.rng = np.random.default_rng(seed)
        self.keys = list(product(range(4), range(4), (0, 2, 4, 6)))
        self.queues = {key: deque() for key in self.keys}
        self.schedule, self.position = [], 0
        self.draws = 0
        self.rejections = Counter()

    def _refill(self):
        seed = int(self.rng.integers(0, 2**63))
        data = self.accounting.generate(1024, seed)
        for i, ram in enumerate(data["rams"]):
            digest = str(data["hashes"][i])
            if hashlib.sha256(ram.tobytes()).hexdigest() != digest:
                raise ValueError("Frozen generator returned an invalid RAM hash")
            if digest in self.excluded or digest in self.seen:
                self.rejections["excluded_or_seen"] += 1
                continue
            key = (int(data["directions"][i]), int(data["player_facings"][i]),
                   int(data["subcell_offsets"][i]))
            self.queues[key].append(dict(
                ram=ram.copy(), hash=digest, direction=key[0], player_facing=key[1],
                subcell_offset=key[2], distance=int(data["distances"][i]),
                generator_seed=seed, generator_index=i,
            ))

    def next_layout(self):
        self.accounting.check("fresh_layout")
        if self.accounting.evaluation_started:
            raise DynamicStop("No training layout may be drawn after the final evaluation")
        if self.position == len(self.schedule):
            self.schedule = [self.keys[int(i)] for i in self.rng.permutation(len(self.keys))]
            self.position = 0
        key = self.schedule[self.position]
        for _ in range(100):
            while self.queues[key]:
                row = self.queues[key].popleft()
                if row["hash"] in self.excluded or row["hash"] in self.seen:
                    self.rejections["excluded_or_seen"] += 1
                    continue
                self.seen.add(row["hash"])
                self.draws += 1
                self.position += 1
                return dict(row, ram=row["ram"].copy(), layout_number=self.draws - 1)
            self._refill()
        raise RuntimeError("Could not supply a fresh nonexcluded balanced layout")


class DynamicAimingEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (47, 26, 26), np.float32)
    action_space = gym.spaces.Discrete(10)
    render_mode = None

    def __init__(self, sampler, *, stream, seed=0):
        super().__init__()
        self.sampler, self.accounting = sampler, sampler.accounting
        self.stream, self.base_seed, self.episode = int(stream), int(seed), 0
        self.raw = self.vector = None
        self.needs_reset = True
        self.layout = None

    def close(self):
        if self.vector is not None:
            self.vector.close()
        elif self.raw is not None:
            self.raw.close()
        self.vector = self.raw = None
        self.needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Dynamic aiming reset does not accept scene overrides")
        self.close()
        self.accounting.check("episode_reset")
        if seed is not None:
            self.base_seed = int(seed)
        self.layout = self.sampler.next_layout()
        native_seed = episode_seed(self.base_seed, self.stream, self.episode)
        self.episode += 1
        self.accounting.native_fresh_initializations += 1
        try:
            self.raw = RustVectorEnv(self.layout["ram"][None].copy(), 200, seed=native_seed)
            self.vector = FiniteVictoryVecEnv(
                self.raw, np.ones(1, np.int64),
                lambda: self.raw.batch.stats[:, 1].astype(np.int64, copy=True),
                gamma=1., shaping=False)
            self.objective_version = self.vector.objective_version
            if self.objective_version != "finite-victory-sparse-v1":
                raise ValueError("Dynamic aiming requires sparse finite victory")
            self.accounting.native_explicit_resets += 1
            observation = self.vector.reset()[0].copy()
            if not np.array_equal(self.raw.batch.ram(0), self.layout["ram"]):
                raise ValueError("Fresh native handle altered initialization RAM")
            self.needs_reset = False
            info = self._context()
            info["native_base_seed"] = native_seed
            return observation, info
        except BaseException:
            self.close()
            raise

    def _context(self):
        return dict(scene_version=VERSION, training_only=True, full_game=False,
                    layout_hash=self.layout["hash"], layout_number=self.layout["layout_number"],
                    aiming_direction=self.layout["direction"],
                    player_facing=self.layout["player_facing"],
                    subcell_offset=self.layout["subcell_offset"],
                    layout_generator_seed=self.layout["generator_seed"],
                    stream=self.stream, dynamic_episode=self.episode - 1)

    def action_masks(self):
        if self.needs_reset:
            raise gym.error.ResetNeeded("Reset before requesting dynamic masks")
        return self.raw.batch.masks()[0].copy()

    def step(self, action):
        if self.needs_reset:
            raise gym.error.ResetNeeded("Terminal dynamic episode must be reset before stepping")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) \
                or not 0 <= action < 10 or not self.action_masks()[action]:
            raise ValueError("Expected a mask-legal primitive action")
        self.accounting.take_step()
        try:
            observations, rewards, dones, infos = self.vector.step(np.array([action], np.int64))
        except BaseException:
            self.needs_reset = True
            raise
        self.accounting.native_steps_completed += 1
        info = deepcopy(infos[0])
        info.update(self._context())
        done = bool(dones[0])
        if done:
            self.needs_reset = True
            # RustVectorEnv already autoreset the OLD layout. Return its saved
            # terminal state, not that reset observation, to Gym/DummyVecEnv.
            self.accounting.native_old_layout_autoresets += 1
            terminal = info.get("terminal_observation")
            if not isinstance(terminal, np.ndarray) or terminal.shape != (47, 26, 26):
                raise ValueError("Native terminal observation is missing")
            if info.get("TimeLimit.truncated") is not False or float(rewards[0]) not in (-1., 1.):
                raise ValueError("Finite sparse terminal contract changed")
            observation = terminal.copy()
            info["terminal_observation"] = terminal.copy()
        else:
            if float(rewards[0]) != 0.:
                raise ValueError("Sparse aiming nonterminal reward must be zero")
            observation = observations[0].copy()
        return observation, float(rewards[0]), done, False, info


def make_dynamic_aiming_env(sampler, *, n_envs=32, seed=0):
    """32 Gym instances -> DummyVecEnv -> first-channel two-frame stack."""
    if type(n_envs) is not int or not 1 <= n_envs <= 32:
        raise ValueError("n_envs must be in1..32")
    raw = DummyVecEnv([lambda i=i: DynamicAimingEnv(sampler, stream=i, seed=seed)
                       for i in range(n_envs)])
    return VecFrameStack(raw, n_stack=2, channels_order="first")


def _read_hashes(path):
    with np.load(path, allow_pickle=False) as archive:
        rams, hashes = archive["rams"], archive["hashes"]
        if len(rams) != len(hashes) or any(
            hashlib.sha256(ram.tobytes()).hexdigest() != digest for ram, digest in zip(rams, hashes)
        ):
            raise ValueError(f"Invalid excluded dataset identity: {path}")
        return set(map(str, hashes))


def _new_final_test(accounting, excluded):
    for count in (128, 256, 512, 1024):
        data = accounting.generate(count, FINAL_TEST_SEED)
        allowed = ~np.isin(data["hashes"], list(excluded))
        if count == 128 and allowed.all():
            return data
        ids = []
        for direction, facing, offset in product(range(4), range(4), (0, 2, 4, 6)):
            available = np.flatnonzero(allowed & (data["directions"] == direction)
                                       & (data["player_facings"] == facing)
                                       & (data["subcell_offsets"] == offset))
            if len(available) < 2:
                break
            ids.extend(available[:2].tolist())
        if len(ids) == 128:
            result = {key: value[ids].copy() for key, value in data.items()
                      if isinstance(value, np.ndarray)}
            result["metadata"] = dict(data["metadata"], count=128, candidate_count=count,
                                      selection="identity-only balanced exclusion")
            return result
    raise RuntimeError("Cannot construct a disjoint final seed20261002 holdout")


def prepare_dynamic_aiming(output, *, allowance, deadline, seed=0,
                           previous_holdouts=PREVIOUS_HOLDOUTS):
    """Prepare immutable exclusions/final test and a monitored fresh-layout factory.

    The caller supplies an already-reserved decision allowance and absolute
    monotonic deadline. Reserves6400 decisions locally for the ONE future final
    test. Preparation may cost2 ROM flush decisions on the first process call;
    every later refill reuses the frozen generator's cached base RAM.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    global_epoch = datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp()
    actual_deadline = min(float(deadline), time.monotonic() + max(0., global_epoch - time.time()))
    accounting = DynamicAccounting(allowance, deadline=actual_deadline, monitor=monitor)
    accounting.reserved_evaluation = 6400
    accounting.check("prepare")
    monitor.check("admission", admission_slots=1, force=True)
    excluded, identities = set(), {}
    # Old train layouts are also excluded to prevent recycling the32 memorized starts.
    paths = list(previous_holdouts) + [ROOT / "runs" / "20260930-victory-aiming-task" / "train.npz"]
    for path in paths:
        path = Path(path)
        excluded.update(_read_hashes(path))
        identities[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    final_test = _new_final_test(accounting, excluded)
    if len(final_test["rams"]) != 128 or set(final_test["hashes"]) & excluded:
        raise ValueError("Final-test exclusion failed")
    excluded.update(map(str, final_test["hashes"]))
    with (output / "final-test.npz").open("xb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in final_test.items() if isinstance(v, np.ndarray)})
    (output / "final-test.json").write_text(json.dumps(final_test["metadata"], indent=2), encoding="utf-8")
    manifest = dict(
        format=VERSION, training_only=True, full_game=False, final_test_seed=FINAL_TEST_SEED,
        final_test_count=128, previous_dataset_identities=identities,
        final_test_sha256=hashlib.sha256((output / "final-test.npz").read_bytes()).hexdigest(),
        excluded_ram_hashes=sorted(excluded),
        excluded_hashes_sha256=hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(),
        reset_policy="fresh unique layout and fresh one-env native handle every episode",
        balance="one each of64 direction/player-facing/subcell combinations per complete64-layout cycle",
        objective="finite-victory-sparse-v1", gamma=1, shaping=False, n_stack=2,
        recommended_extractor="learning_lab.ego_encoder.EgocentricGridCNN",
        runner_included=False, final_test_protocol="one evaluation after training only",
        accounting=accounting.snapshot(),
        sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "learning_lab\\aiming_dynamic.py", "learning_lab\\aiming_task.py",
            "learning_lab\\victory_objective.py", "learning_lab\\rewards.py",
            "skills.py", "game_state.py", "heuristic.py", "rust_env.py",
            "native\\target\\release\\battle_mirror_probe.dll")},
    )
    (output / "environment.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sampler = BalancedAimingSampler(seed=seed, excluded_hashes=excluded, accounting=accounting)
    return dict(sampler=sampler, accounting=accounting, final_test=final_test,
                output=output, manifest=manifest)


def evaluate_final_once(model, prepared):
    """Optional parent-invoked one-shot evaluator; no training or model selection."""
    account = prepared["accounting"]
    account.check("final_test")
    if account.evaluation_started:
        raise DynamicStop("Final holdout evaluation is single-use")
    if account.total_decisions + 6400 > account.limit:
        raise DynamicStop("Insufficient pre-reserved final-test allowance")
    account.evaluation_started = True
    result = evaluate_holdout(model, prepared["final_test"], guard=account)
    account.evaluation_attempted += result["attempted_decisions"]
    account.evaluation_completed += result["completed_decisions"]
    account.reserved_evaluation = 0
    (prepared["output"] / "final-evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result
