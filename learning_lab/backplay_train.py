"""Reserved sparse finite-victory PPO backplay; every clear is NATIVE ONLY.

run_backplay(pool_path, output, existing_budget, ...) never creates or extends a
research budget. A failed unresolved collection retains its whole reservation.
Warm starts move policy+Adam only into new envs/RNG/history, never simulator state.
Checkpoint loading uses a tiny inference buffer and is not exact training resume.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecFrameStack

from native_rollout import NativeGPURunRolloutBuffer
from policy import Grid26CNN
from rust_env import NativeBatch, RustVectorEnv, episode_seed, BACKEND_VERSION, RNG_VERSION
from rust_resources import ResourceMonitor, ResourcePressureError
from learning_lab.backplay_starts import load_backplay_pool
from learning_lab.budget import BudgetExhausted, ResearchBudget
from learning_lab.victory_objective import FiniteVictoryVecEnv

_ROOT = Path(__file__).resolve().parents[1]
_GLOBAL_DEADLINE_EPOCH = datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp()
_FORMAT = "tank-ai-backplay-sparse-victory-checkpoint-v1"
_EXPECTED_OBJECTIVE = "finite-victory-sparse-v1"
_PPO = dict(learning_rate=3e-4, gamma=1., gae_lambda=.95, clip_range=.2,
            ent_coef=.01, target_kl=.02, n_epochs=4, batch_size=256, n_envs=32, n_steps=256)
_ROLLOUT = 32 * 256
_EVAL_EVERY, _SAVE_EVERY, _EVAL_ENVS = 32768, 16384, 16
_WARNING = "Native curriculum/normal-native clears are NOT original-ROM full wins or generalization."


class DeadlineReached(RuntimeError):
    pass


def _write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sources():
    return {name: _hash(_ROOT / name) for name in (
        "learning_lab\\backplay_train.py", "learning_lab\\backplay_starts.py",
        "learning_lab\\victory_objective.py", "learning_lab\\rewards.py",
        "policy.py", "rust_env.py", "native_codec.py", "native_rollout.py", "compact_rollout.py",
        "native\\target\\release\\battle_mirror_probe.dll")}


def experiment_contract(objective_version=None):
    objective_version = _EXPECTED_OBJECTIVE if objective_version is None else objective_version
    if objective_version != _EXPECTED_OBJECTIVE:
        raise ValueError("This first backplay experiment requires the sparse victory objective")
    return dict(
        observation_shape=[94, 26, 26], observation_dtype="float32", actions=10,
        n_stack=2, channels_order="first", frame_skip=4, extractor="Grid26CNN", features_dim=256,
        net_arch={"pi": [64, 64], "vf": [64, 64]}, normalize_images=False,
        reward_objective=objective_version, gamma_required=1., shaping=False, timeout_bootstrap=False,
        episode_return="nonterminal0; terminal clear+1; base/lives/timeout-1; no potential shaping",
        ppo=deepcopy(_PPO), buffer="NativeGPURunRolloutBuffer", storage_cap_bytes=2 * 1024**3,
        backend=BACKEND_VERSION, rng=RNG_VERSION,
        curriculum=[1, 2, 4, 8, 12, 16, 20, "normal_full_start"],
        per_level_step_cap=131072, evaluation_every=_EVAL_EVERY,
        minimum_gate_steps=_EVAL_EVERY, evaluation_episodes=16, gate_clears=13,
        checkpoint_every=_SAVE_EVERY, exact_simulator_resume=False,
        warm_start="policy+Adam only; new native env, RNG, episode counters and zero history",
        formal_rom_validation=False,
    )


def build_levels(pool):
    rams = pool.get("rams")
    if not isinstance(rams, np.ndarray) or rams.dtype != np.uint8 or rams.ndim != 2 \
            or rams.shape[1:] != (2048,):
        raise ValueError("Pool requires uint8 rams(N,2048)")
    n = len(rams)
    for key, dtype in (("remaining", np.int64), ("source_frames", np.int64),
                       ("seeds", np.int64), ("stages", np.int64), ("candidate_ids", np.int64),
                       ("full_start", np.bool_), ("noop_tested", np.bool_)):
        if not isinstance(pool.get(key), np.ndarray) or pool[key].shape != (n,) \
                or pool[key].dtype != dtype:
            raise ValueError(f"Invalid pool field {key}")
    if np.any(pool["stages"] != 1) or not np.isin(pool["seeds"], [0, 1, 2]).all():
        raise ValueError("Backplay permits only stage1 training openings0/1/2")
    excluded = set(pool.get("automatic_noop_clear_exclusions", [])) | set(pool.get("unsupported_exclusions", []))
    allowed = ~np.isin(pool["candidate_ids"], list(excluded))
    levels = []
    for remaining in (1, 2, 4, 8, 12, 16, 20):
        ids = np.flatnonzero(allowed & ~pool["full_start"] & pool["noop_tested"]
                             & (pool["remaining"] == remaining))
        if not len(ids):
            raise ValueError(f"No NOOP-screened supported curriculum starts for remaining={remaining}")
        levels.append(dict(index=len(levels), name=f"remaining_{remaining}", remaining=remaining,
                           full_start=False, indices=ids, max_frames=min(12000, 2000 + 500 * remaining)))
    full = np.flatnonzero(allowed & pool["full_start"])
    reference = pool.get("normal_start") or {}
    if len(full) != 1 or pool["remaining"][full[0]] != 20 or pool["source_frames"][full[0]] != -1 \
            or reference.get("initialization_method") != "rust_env.stage_ram(1, seed=0)" \
            or reference.get("full_start") is not True or reference.get("stage") != 1 \
            or reference.get("seed") != 0 or reference.get("native_import_byte_exact") is not True \
            or any(reference.get("native_support_flags", [1])) \
            or hashlib.sha256(rams[full[0]].tobytes()).hexdigest() != reference.get("ram_sha256"):
        raise ValueError("A separate verified exact normal full_start reference is required")
    levels.append(dict(index=7, name="normal_full_start", remaining=20, full_start=True,
                       indices=full, max_frames=12000))
    return levels


def wrap_victory(raw, initial_remaining):
    return FiniteVictoryVecEnv(
        raw, initial_remaining,
        lambda: raw.batch.stats[:, 1].astype(np.int64, copy=True), gamma=_PPO["gamma"], shaping=False)


def collect_reserved(runtime, budget, label):
    reserved = runtime.rollout_size
    token = budget.reserve(label, reserved)
    before = runtime.completed_steps
    result = dict(reservation=token, reserved=reserved, completed_decisions=0,
                  settled=False, complete=False, status="running", error=None)
    try:
        complete = bool(runtime.collect())
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      completed_decisions=runtime.completed_steps - before)
        return result
    actual = runtime.completed_steps - before
    if not 0 <= actual <= reserved:
        result.update(status="failed", error="Collection exceeded reserved bounds",
                      completed_decisions=actual)
        return result
    budget.settle(token, actual)
    result.update(status="completed" if complete and actual == reserved else "partial",
                  complete=complete and actual == reserved, completed_decisions=actual, settled=True)
    return result


def _select_indices(level, count, seed):
    source = level["indices"]
    order = np.random.default_rng(seed).permutation(len(source))
    return source[order[np.arange(count) % len(source)]]


def _outcome(ram):
    if int(ram[0x68]) != 0x80:
        return "base_lost"
    if int(ram[0x51]) == 0 or (int(ram[0x51]) <= 1 and 0x10 <= int(ram[0xa0]) < 0x80):
        return "lives_exhausted"
    return "clear" if int(ram[0x80]) == 0 else "timeout"


def evaluate_native(model, pool, level, *, budget, guard, seed, evaluation_index, label):
    count = _EVAL_ENVS
    reserved = count * math.ceil(level["max_frames"] / 4)
    token = budget.reserve(label, reserved)
    seeds = [episode_seed(seed, level["index"], 1000000 + evaluation_index * count + i)
             for i in range(count)]
    ids = _select_indices(level, count, seeds[0])
    report = dict(status="running", reservation=token, reserved=reserved, settled=False,
                  completed_decisions=0, attempted_decisions=0, episodes=[], clears=0,
                  native_episode_seeds=seeds, start_indices=ids.tolist(), warning=_WARNING)
    batch = None
    try:
        guard.check("evaluation_init")
        batch = NativeBatch(pool["rams"][ids].copy(), level["max_frames"], seeds=seeds)
        history = np.zeros((count, 94, 26, 26), np.float32)
        history[:, 47:] = batch.observe().copy()
        finished = np.zeros(count, bool)
        for _ in range(math.ceil(level["max_frames"] / 4)):
            guard.check("evaluation_step")
            masks = batch.masks().copy()
            actions, _ = model.predict(history, deterministic=True, action_masks=masks)
            actions = np.asarray(actions)
            if actions.shape != (count,) or actions.dtype.kind not in "iu" \
                    or np.any(actions < 0) or np.any(actions >= 10) \
                    or not masks[np.arange(count), actions].all():
                raise ValueError("Evaluation returned illegal primitive actions")
            guard.check("evaluation_step")
            report["attempted_decisions"] += count
            observations, _, dones, stats = batch.step(actions)
            report["completed_decisions"] += count
            history = np.roll(history, -47, axis=1)
            history[:, 47:] = observations
            for index in np.flatnonzero(dones & ~finished):
                outcome = _outcome(batch.ram(int(index)))
                report["episodes"].append(dict(
                    env_index=int(index), seed=seeds[index], outcome=outcome,
                    frames=int(stats[index, 0]), kills=int(stats[index, 1]),
                    deaths=int(stats[index, 2]), full_start=level["full_start"],
                    formal_rom_win=False,
                ))
                report["clears"] += int(outcome == "clear")
            finished |= dones
            if finished.all():
                break
        report["status"] = "completed" if finished.all() else "partial"
        budget.settle(token, report["completed_decisions"])
        report["settled"] = True
    except (DeadlineReached, ResourcePressureError) as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
        if report["attempted_decisions"] == report["completed_decisions"]:
            budget.settle(token, report["completed_decisions"])
            report["settled"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if batch is not None:
            batch.close()
    return report


def passes_gate(result):
    return bool(result.get("status") == "completed" and len(result.get("episodes", [])) == 16
                and type(result.get("clears")) is int and 13 <= result["clears"] <= 16)


class _Guard:
    def __init__(self, deadline, deadline_epoch, monitor):
        self.deadline, self.deadline_epoch, self.monitor = deadline, deadline_epoch, monitor

    def check(self, phase):
        if time.monotonic() >= self.deadline or time.time() >= self.deadline_epoch:
            raise DeadlineReached("Existing research deadline reached; no extension permitted")
        self.monitor.check(phase)


class _CountedVec(VecEnvWrapper):
    def __init__(self, venv, guard):
        super().__init__(venv)
        self.guard, self.completed_steps, self.limit = guard, 0, 0

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.guard.check("native_train_step")
        if self.completed_steps + self.num_envs > self.limit:
            raise RuntimeError("Native step lacks a sufficient rollout reservation")
        result = self.venv.step_wait()
        self.completed_steps += self.num_envs
        return result


class _GuardedBuffer(NativeGPURunRolloutBuffer):
    guard = None

    def get(self, batch_size=None):
        iterator = super().get(batch_size)
        while True:
            if self.guard is not None:
                self.guard.check("ppo_decode")
            try:
                sample = next(iterator)
            except StopIteration:
                return
            if self.guard is not None:
                self.guard.check("ppo_minibatch")
            yield sample


class _Callback(BaseCallback):
    def __init__(self, guard, objective_version):
        super().__init__()
        self.guard = guard
        self.objective_version = objective_version

    def _on_step(self):
        rewards, infos, dones = self.locals["rewards"], self.locals["infos"], self.locals["dones"]
        if not np.isfinite(rewards).all():
            raise FloatingPointError("Nonfinite finite-victory reward")
        for index, info in enumerate(infos):
            if info.get("reward_profile") != self.objective_version:
                raise ValueError("Wrong reward objective in PPO collection")
            if not dones[index] and float(rewards[index]) != 0.:
                raise ValueError("Sparse victory requires zero nonterminal reward")
            if dones[index]:
                target = 1. if info.get("outcome") == "clear" else -1.
                if info.get("TimeLimit.truncated") is not False \
                        or float(rewards[index]) != target \
                        or not np.isclose(info["episode"]["r"], target, atol=1e-5, rtol=0):
                    raise ValueError("Finite-victory terminal/bootstrap contract violated")
        try:
            self.guard.check("rollout_callback")
        except (DeadlineReached, ResourcePressureError):
            return False
        return True


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return deepcopy(value)


class _Runtime:
    rollout_size = _ROLLOUT

    def __init__(self, *, pool, level, seed, device, guard, max_steps, warm=None):
        if isinstance(_PPO["gamma"], bool) or _PPO["gamma"] != 1.:
            raise ValueError("Finite-victory backplay requires gamma=1")
        self.guard, self.optimized_steps, self.adam_steps = guard, 0, 0
        self.max_steps, self._step_count = max_steps, {"count": 0}
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        ids = _select_indices(level, 32, seed)
        self.start_indices = ids.tolist()
        self.native_rng_base = episode_seed(seed, level["index"], 0) % (2**64 - 32)
        self.initial_native_episode_seeds = [episode_seed(self.native_rng_base + i, i, 0)
                                             for i in range(32)]
        raw = RustVectorEnv(pool["rams"][ids].copy(), level["max_frames"],
                            seed=self.native_rng_base)
        self.raw = raw
        self.model, self.env = None, None
        try:
            counted = _CountedVec(raw, guard)
            self.counted = counted
            # read_kills must read raw native stats, not sparse terminal-only infos.
            objective = FiniteVictoryVecEnv(
                counted, pool["remaining"][ids],
                lambda: raw.batch.stats[:, 1].astype(np.int64, copy=True), gamma=1., shaping=False)
            self.objective_version = objective.objective_version
            experiment_contract(self.objective_version)
            self.env = VecFrameStack(objective, n_stack=2, channels_order="first")
            arguments = {key: value for key, value in _PPO.items() if key != "n_envs"}
            self.model = MaskablePPO(
                "CnnPolicy", self.env, seed=(seed + 104729 * level["index"]) % 2**32,
                device=device, verbose=0, **arguments,
                policy_kwargs={"features_extractor_class": Grid26CNN, "normalize_images": False,
                               "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                rollout_buffer_class=_GuardedBuffer,
                rollout_buffer_kwargs={"channels_per_frame": 47, "max_storage_bytes": 2 * 1024**3},
            )
            if str(device).startswith("cuda") and self.model.device.type != "cuda":
                raise RuntimeError("Requested CUDA cannot silently fall back to CPU")
            if warm is not None:
                self.model.policy.load_state_dict(warm["policy"], strict=True)
                self.model.policy.optimizer.load_state_dict(warm["optimizer"])
                self.model._n_updates = warm["ppo_epoch_updates"]
            self.model.rollout_buffer.guard = guard
            self._last_good = self.export_warm()
            self.state_verified = True
            original_step = self.model.policy.optimizer.step
            counter = self._step_count

            def guarded_step(*args, **kwargs):
                guard.check("adam_step")
                result = original_step(*args, **kwargs)
                counter["count"] += 1
                return result

            self.model.policy.optimizer.step = guarded_step
            # SB3 seeds its env while initializing the policy; restore the
            # independently recorded native RNG stream before the first reset.
            raw.seed(self.native_rng_base)
            self.total, self.callback = self.model._setup_learn(
                max_steps, callback=_Callback(guard, self.objective_version))
            self.callback.on_training_start({}, {})
        except BaseException:
            self.close()
            raise

    @property
    def completed_steps(self):
        return self.counted.completed_steps

    def collect(self):
        self.counted.limit = self.completed_steps + self.rollout_size
        return self.model.collect_rollouts(
            self.env, callback=self.callback, rollout_buffer=self.model.rollout_buffer,
            n_rollout_steps=_PPO["n_steps"])

    def export_warm(self):
        return dict(policy=_cpu_copy(self.model.policy.state_dict()),
                    optimizer=_cpu_copy(self.model.policy.optimizer.state_dict()),
                    ppo_epoch_updates=int(self.model._n_updates))

    def optimize(self):
        self.guard.check("ppo_update")
        self.model._update_current_progress_remaining(self.model.num_timesteps, self.total)
        previous_adam_steps = self._step_count["count"]
        self.state_verified = False
        try:
            self.model.train()
            committed = self.export_warm()
        except BaseException:
            self.model.policy.load_state_dict(self._last_good["policy"], strict=True)
            self.model.policy.optimizer.load_state_dict(self._last_good["optimizer"])
            self.model.policy.optimizer.zero_grad(set_to_none=True)
            self.model._n_updates = self._last_good["ppo_epoch_updates"]
            self._step_count["count"] = previous_adam_steps
            self.state_verified = True
            raise
        self.optimized_steps += self.rollout_size
        self.adam_steps = self._step_count["count"]
        self._last_good = committed
        self.state_verified = True

    def close(self):
        if self.model is not None:
            self.model.policy.optimizer.__dict__.pop("step", None)
            self.model.rollout_buffer.guard = None
        if hasattr(self, "callback"):
            self.callback.on_training_end()
        if self.env is not None:
            self.env.close()
        else:
            self.raw.close()


def _make_runtime(**kwargs):
    return _Runtime(**kwargs)


def _save_checkpoint(runtime, output, level, provenance, *, optimized_total, lineage, role):
    if runtime.optimized_steps <= 0 or optimized_total <= 0 or not runtime.state_verified:
        raise ValueError("Only completed optimized updates may be checkpointed")
    if _sources() != provenance["sources"]:
        raise ValueError("Frozen source identity changed during training")
    path = output / f"{level['name']}-optimized-{optimized_total:09d}-{role}.zip"
    if path.exists():
        raise FileExistsError(path)
    model = runtime.model
    collected = model.num_timesteps
    try:
        model.num_timesteps = optimized_total
        model.save(path, exclude=["_last_obs", "_last_episode_starts", "_last_original_obs"])
    finally:
        model.num_timesteps = collected
    manifest = dict(
        format=_FORMAT, contract=experiment_contract(runtime.objective_version), checkpoint_file=path.name,
        environment_objective_version=runtime.objective_version,
        checkpoint_sha256=_hash(path), sources=provenance["sources"],
        pool=provenance["pool"], level={k: v for k, v in level.items() if k != "indices"},
        start_identities=provenance["start_identities"][level["name"]],
        environment_start_indices=runtime.start_indices,
        initial_native_episode_seeds=runtime.initial_native_episode_seeds,
        native_rng_base=runtime.native_rng_base,
        training_device=str(model.device),
        optimized_steps=optimized_total, level_optimized_steps=runtime.optimized_steps,
        adam_steps=runtime.adam_steps, ppo_epoch_updates=int(model._n_updates),
        parent_checkpoint=lineage, role=role, exact_simulator_resume=False,
        adam_steps_scope="current level; parent Adam state retained through lineage",
        load_purpose="policy+Adam only; small inference buffer; no simulator resume",
        warning=_WARNING,
    )
    _write(path.with_suffix(".manifest.json"), manifest)
    return dict(path=str(path), sha256=manifest["checkpoint_sha256"], optimized_steps=optimized_total)


def _pool_identity(path, pool, levels):
    path = Path(path).resolve()
    identity = dict(type="real-rom-curriculum-plus-exact-normal-v1",
                    normal_initialization="rust_env.stage_ram(1, seed=0)",
                    path=str(path), files={
        name: _hash(path / name) for name in ("starts.npz", "normal_start.npz", "normal_start.json", "report.json")})
    starts = {}
    for level in levels:
        starts[level["name"]] = [dict(
            pool_index=int(i), candidate_id=int(pool["candidate_ids"][i]),
            remaining=int(pool["remaining"][i]), full_start=bool(pool["full_start"][i]),
            source_frame=int(pool["source_frames"][i]), stage=int(pool["stages"][i]),
            seed=int(pool["seeds"][i]), ram_sha256=hashlib.sha256(pool["rams"][i].tobytes()).hexdigest(),
        ) for i in level["indices"]]
    return identity, starts


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_checkpoint_identities(manifest):
    pool, level = manifest.get("pool"), manifest.get("level")
    if not isinstance(pool, dict) or pool.get("type") != "real-rom-curriculum-plus-exact-normal-v1" \
            or pool.get("normal_initialization") != "rust_env.stage_ram(1, seed=0)" \
            or not isinstance(pool.get("path"), str) or not pool["path"] \
            or not isinstance(pool.get("files"), dict) \
            or set(pool["files"]) != {"starts.npz", "normal_start.npz", "normal_start.json", "report.json"} \
            or not all(_valid_hash(v) for v in pool["files"].values()) \
            or not isinstance(level, dict) or type(level.get("index")) is not int \
            or not 0 <= level["index"] <= 7 or type(level.get("full_start")) is not bool:
        raise ValueError("Invalid checkpoint pool/type/level identity")
    expected_remaining = (1, 2, 4, 8, 12, 16, 20, 20)[level["index"]]
    expected_full = level["index"] == 7
    expected_name = "normal_full_start" if expected_full else f"remaining_{expected_remaining}"
    if level.get("remaining") != expected_remaining or level["full_start"] != expected_full \
            or level.get("name") != expected_name \
            or level.get("max_frames") != min(12000, 2000 + 500 * expected_remaining):
        raise ValueError("Checkpoint curriculum progression contract is invalid")
    starts = manifest.get("start_identities")
    if not isinstance(starts, list) or not starts or (expected_full and len(starts) != 1):
        raise ValueError("Missing checkpoint start identities")
    for start in starts:
        if not isinstance(start, dict) or start.get("full_start") is not expected_full \
                or type(start.get("remaining")) is not int or start["remaining"] != expected_remaining \
                or type(start.get("pool_index")) is not int or start["pool_index"] < 0 \
                or type(start.get("candidate_id")) is not int \
                or start.get("stage") != 1 or type(start.get("seed")) is not int \
                or start["seed"] not in (0, 1, 2) or type(start.get("source_frame")) is not int \
                or not _valid_hash(start.get("ram_sha256")) \
                or (expected_full and (start["source_frame"] != -1 or start["seed"] != 0)) \
                or (not expected_full and start["source_frame"] < 0):
            raise ValueError("Checkpoint mixes invalid curriculum/normal start identities")
    for key in ("level_optimized_steps", "adam_steps", "ppo_epoch_updates"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError("Checkpoint lacks completed optimization accounting")
    if manifest["level_optimized_steps"] % _ROLLOUT \
            or manifest["level_optimized_steps"] > 131072 \
            or manifest["level_optimized_steps"] > manifest["optimized_steps"]:
        raise ValueError("Invalid per-level optimized step count")
    if manifest["ppo_epoch_updates"] > 4 * (manifest["optimized_steps"] // _ROLLOUT):
        raise ValueError("PPO epoch accounting exceeds the declared update profile")
    indices = manifest.get("environment_start_indices")
    seeds = manifest.get("initial_native_episode_seeds")
    allowed_indices = {s["pool_index"] for s in starts}
    if not isinstance(indices, list) or len(indices) != 32 \
            or any(type(i) is not int or i not in allowed_indices for i in indices) \
            or not isinstance(seeds, list) or len(seeds) != 32 \
            or any(type(s) is not int or not 0 <= s < 2**64 for s in seeds):
        raise ValueError("Missing per-environment start/RNG identities")
    parent = manifest.get("parent_checkpoint")
    if expected_full or level["index"] > 0:
        if not isinstance(parent, dict) or not _valid_hash(parent.get("sha256")) \
                or not isinstance(parent.get("path"), str) \
                or type(parent.get("optimized_steps")) is not int \
                or parent["optimized_steps"] + manifest["level_optimized_steps"] != manifest["optimized_steps"]:
            raise ValueError("Missing policy+Adam warm-start lineage")
    elif parent is not None:
        raise ValueError("First backplay level must start fresh")
    elif manifest["optimized_steps"] != manifest["level_optimized_steps"]:
        raise ValueError("First-level optimization accounting must start from zero")


def run_backplay(pool_path, output, budget, *, seed=0, device="cpu", max_steps_per_level=131072):
    """Gated native curriculum with atomic reservations and strict optimized checkpoints."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if str(device) == "auto" or torch.device(device).type not in ("cpu", "cuda"):
        raise ValueError("Choose an explicit CPU or CUDA device for this experiment")
    if type(max_steps_per_level) is not int or not _ROLLOUT <= max_steps_per_level <= 131072 \
            or max_steps_per_level % _ROLLOUT:
        raise ValueError("Per-level cap must be a multiple of8192 in8192..131072")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(
        format="tank-ai-backplay-run-v1", status="running", stop_reason=None, error=None,
        warning=_WARNING, exact_simulator_resume=False, formal_rom_wins=0, seed=seed,
        device=str(device), max_steps_per_level=max_steps_per_level, contract=experiment_contract(),
        levels=[], rollouts=[], evaluations=[], checkpoints=[], optimized_steps=0,
        train_completed_decisions=0, eval_completed_decisions=0, unresolved_reservations=[],
    )
    runtime, lineage, warm, provenance = None, None, None, None
    last_checkpoint_steps = -1
    try:
        state = budget.snapshot()
        if type(state["max_interactions"]) is not int or not 0 < state["max_interactions"] <= 4000000:
            raise ValueError("Global research ledger may not exceed4M interactions")
        monitor = ResourceMonitor(output / "resources.jsonl", device=str(device).split(":")[0])
        deadline = started + min(28800., max(0., state["remaining_seconds"]))
        epoch = min(float(state["deadline_epoch"]), _GLOBAL_DEADLINE_EPOCH)
        guard = _Guard(deadline, epoch, monitor)
        guard.check("admission")
        monitor.check("admission", admission_slots=1, force=True)
        pool = load_backplay_pool(pool_path)
        levels = build_levels(pool)
        identity, start_ids = _pool_identity(pool_path, pool, levels)
        provenance = dict(sources=_sources(), pool=identity, start_identities=start_ids)
        report["provenance"] = provenance
        label_prefix = "backplay-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
        for level in levels:
            guard.check("level_start")
            level_report = dict(
                name=level["name"], remaining=level["remaining"], full_start=level["full_start"],
                max_frames=level["max_frames"], parent_checkpoint=lineage,
                optimized_steps=0, gate_passed=False, status="running", evaluations=[],
            )
            report["levels"].append(level_report)
            runtime = _make_runtime(pool=pool, level=level, seed=seed, device=device,
                                    guard=guard, max_steps=max_steps_per_level, warm=warm)
            warm = None
            level_report["warm_start"] = lineage is not None
            level_report["environment_objective_version"] = getattr(runtime, "objective_version", _EXPECTED_OBJECTIVE)
            level_report["native_rng_base"] = getattr(runtime, "native_rng_base", None)
            level_report["environment_start_indices"] = getattr(runtime, "start_indices", None)
            rollout_index, evaluation_index = 0, 0
            last_checkpoint_steps = -1
            while runtime.optimized_steps < max_steps_per_level:
                guard.check("rollout_start")
                row = collect_reserved(runtime, budget,
                                       f"{label_prefix}-{level['index']}-rollout-{rollout_index}")
                rollout_index += 1
                row["level"] = level["name"]
                report["rollouts"].append(row)
                report["train_completed_decisions"] += row["completed_decisions"]
                if not row["settled"]:
                    report["unresolved_reservations"].append(row["reservation"])
                if not row["complete"]:
                    report["status"] = "failed" if row["status"] == "failed" else "partial"
                    report["stop_reason"] = "unresolved_collection" if not row["settled"] else "partial_collection"
                    report["error"] = row["error"]
                    break
                guard.check("optimizer_start")
                runtime.optimize()
                report["optimized_steps"] += _ROLLOUT
                level_report["optimized_steps"] = runtime.optimized_steps
                if runtime.optimized_steps % _SAVE_EVERY == 0:
                    checkpoint = _save_checkpoint(
                        runtime, output, level, provenance, optimized_total=report["optimized_steps"],
                        lineage=lineage, role="periodic")
                    report["checkpoints"].append(checkpoint)
                    last_checkpoint_steps = runtime.optimized_steps
                _write(output / "report.json", report)
                if runtime.optimized_steps >= _EVAL_EVERY and runtime.optimized_steps % _EVAL_EVERY == 0:
                    evaluation = evaluate_native(
                        runtime.model, pool, level, budget=budget, guard=guard, seed=seed,
                        evaluation_index=evaluation_index,
                        label=f"{label_prefix}-{level['index']}-eval-{evaluation_index}")
                    evaluation_index += 1
                    report["evaluations"].append(evaluation)
                    level_report["evaluations"].append(evaluation)
                    report["eval_completed_decisions"] += evaluation["completed_decisions"]
                    if not evaluation.get("settled", True) and evaluation.get("reservation"):
                        report["unresolved_reservations"].append(evaluation["reservation"])
                    if evaluation["status"] != "completed":
                        report.update(status="partial", stop_reason="incomplete_evaluation")
                        break
                    if passes_gate(evaluation):
                        level_report["gate_passed"] = True
                        break
            if runtime.optimized_steps > 0:
                checkpoint = _save_checkpoint(
                    runtime, output, level, provenance, optimized_total=report["optimized_steps"],
                    lineage=lineage, role="level_end")
                report["checkpoints"].append(checkpoint)
                last_checkpoint_steps = runtime.optimized_steps
                level_report["checkpoint"] = checkpoint
            level_report["status"] = "passed" if level_report["gate_passed"] else "stopped"
            if report["status"] != "running":
                break
            if not level_report["gate_passed"]:
                report.update(status="completed", stop_reason="level_gate_not_met")
                break
            lineage = level_report["checkpoint"]
            warm = runtime.export_warm()
            runtime.close()
            runtime = None
            _write(output / "report.json", report)
        if report["status"] == "running":
            report.update(status="completed", stop_reason="all_native_gates_passed")
    except (DeadlineReached, BudgetExhausted, ResourcePressureError) as error:
        reason = "deadline" if isinstance(error, DeadlineReached) else (
            "budget_exhausted" if isinstance(error, BudgetExhausted) else "resource_pressure")
        report.update(status="partial", stop_reason=reason, error=f"{type(error).__name__}: {error}")
    except BaseException as error:
        report.update(status="failed", stop_reason="error", error=f"{type(error).__name__}: {error}")
    finally:
        if runtime is not None:
            if report["levels"] and report["levels"][-1]["status"] == "running":
                report["levels"][-1]["status"] = "stopped"
                report["levels"][-1]["optimized_steps"] = runtime.optimized_steps
            try:
                if runtime.optimized_steps > 0 and runtime.optimized_steps != last_checkpoint_steps \
                        and provenance is not None:
                    report["checkpoints"].append(_save_checkpoint(
                        runtime, output, level, provenance, optimized_total=report["optimized_steps"],
                        lineage=lineage, role="partial_optimized"))
            except Exception as error:
                report["checkpoint_error"] = f"{type(error).__name__}: {error}"
            runtime.close()
        report["elapsed_seconds"] = time.monotonic() - started
        report["budget_snapshot"] = budget.snapshot()
        _write(output / "report.json", report)
    return report


def load_backplay_checkpoint(path, device="cpu"):
    """Strict trusted-local policy/Adam loader; never restores native simulator state."""
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Missing/invalid backplay manifest") from error
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT \
            or manifest.get("contract") != experiment_contract() \
            or manifest.get("environment_objective_version") != _EXPECTED_OBJECTIVE \
            or isinstance(manifest["contract"]["ppo"]["gamma"], bool) \
            or manifest.get("sources") != _sources() \
            or manifest.get("checkpoint_file") != path.name \
            or manifest.get("exact_simulator_resume") is not False \
            or type(manifest.get("optimized_steps")) is not int \
            or manifest["optimized_steps"] <= 0 or manifest["optimized_steps"] % _ROLLOUT \
            or not manifest.get("start_identities") or not isinstance(manifest.get("pool"), dict):
        raise ValueError("Incompatible backplay spaces/reward/gamma/source/pool contract")
    _validate_checkpoint_identities(manifest)
    if not path.is_file() or _hash(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Backplay checkpoint SHA256 mismatch")
    model = MaskablePPO.load(
        path, device=device, custom_objects={
            "rollout_buffer_class": NativeGPURunRolloutBuffer,
            "rollout_buffer_kwargs": {"channels_per_frame": 47, "max_storage_bytes": 2 * 1024**3},
            "n_steps": 2, "n_envs": 1, "batch_size": 2,
        })
    policy = model.policy
    from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
    if not isinstance(model.observation_space, gym.spaces.Box) \
            or model.observation_space.shape != (94, 26, 26) \
            or model.observation_space.dtype != np.float32 \
            or not np.all(model.observation_space.low == -1) or not np.all(model.observation_space.high == 1) \
            or not isinstance(model.action_space, gym.spaces.Discrete) \
            or model.action_space.n != 10 or model.action_space.start != 0 \
            or type(policy) is not MaskableActorCriticCnnPolicy \
            or type(policy.features_extractor) is not Grid26CNN \
            or policy.features_extractor.features_dim != 256 or not policy.share_features_extractor \
            or policy.normalize_images or policy.net_arch != {"pi": [64, 64], "vf": [64, 64]} \
            or not isinstance(policy.optimizer, torch.optim.Adam) \
            or model.gamma != 1. or model.gae_lambda != .95 or model.ent_coef != .01 \
            or model.target_kl != .02 or model.n_epochs != 4 or model.clip_range(1.) != .2 \
            or isinstance(model.gamma, bool) \
            or model.learning_rate != 3e-4 or model.num_timesteps != manifest["optimized_steps"] \
            or model._n_updates != manifest.get("ppo_epoch_updates"):
        raise ValueError("Serialized policy violates the finite-victory PPO contract")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", required=True, help="Existing shared ResearchBudget SQLite file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps-per-level", type=int, default=131072)
    args = parser.parse_args(argv)
    report = run_backplay(args.pool, args.output, ResearchBudget(args.budget), seed=args.seed,
                          device=args.device, max_steps_per_level=args.max_steps_per_level)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
