"""One fixed-seed conditional-aiming PPO run, with one final heldout evaluation.

Uses saved layouts only (no ROM boot), zero test-layout gradients, and an
already-reserved parent allowance. No ledger is created, reserved or settled.
This task is NOT a full-game win or generalization result.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecFrameStack

from native_rollout import NativeGPURunRolloutBuffer
from policy import Grid26CNN
from rust_env import NativeBatch, episode_seed
from rust_resources import ResourceMonitor, ResourcePressureError
from rust_learning import _training_metrics
from learning_lab.aiming_task import make_aiming_env, validate_split, SCENE_VERSION
from learning_lab.backplay_train import _Guard, _GuardedBuffer, _Callback, _cpu_copy, DeadlineReached
from learning_lab.victory_objective import FiniteVictoryVecEnv

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "runs" / "20260930-victory-aiming-task"
OUTPUT = ROOT / "runs" / "20260930-victory-lab" / "aiming-ppo-v1"
FORMAT = "tank-ai-aiming-ppo-checkpoint-v1"
PPO = dict(learning_rate=3e-4, gamma=1., gae_lambda=.95, clip_range=.2,
           ent_coef=.01, target_kl=.02, n_epochs=4, n_steps=128, batch_size=256)
TRAIN_STEPS, TOTAL_CAP = 65536, 72000
OBJECTIVE = "finite-victory-sparse-v1"


def _write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    return {name: _hash(ROOT / name) for name in (
        "learning_lab\\aiming_train.py", "learning_lab\\aiming_task.py",
        "learning_lab\\backplay_train.py", "learning_lab\\victory_objective.py",
        "learning_lab\\rewards.py", "policy.py", "rust_env.py", "rust_learning.py",
        "native_rollout.py", "native_codec.py", "compact_rollout.py",
        "native\\target\\release\\battle_mirror_probe.dll")}


def contract():
    return dict(scene_version=SCENE_VERSION, full_game=False, observation_shape=[94, 26, 26],
                observation_dtype="float32", actions=10, n_stack=2, channels_order="first",
                extractor="Grid26CNN", features_dim=256, net_arch={"pi": [64, 64], "vf": [64, 64]},
                normalize_images=False, objective=OBJECTIVE, shaping=False, timeout_bootstrap=False,
                ppo=deepcopy(PPO), n_envs=32, frame_cap=200, training_steps=TRAIN_STEPS,
                holdout_episodes=128, holdout_max_decisions=6400,
                test_protocol="one first episode per template, once after training; never gradient selection",
                buffer="NativeGPURunRolloutBuffer", checkpoint_every=16384,
                exact_simulator_resume=False)


def load_dataset(root, name):
    with np.load(Path(root) / f"{name}.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data["metadata"] = json.loads((Path(root) / f"{name}.json").read_text())
    if data["metadata"]["scene_version"] != SCENE_VERSION or data["metadata"]["full_game"]:
        raise ValueError("Expected versioned training-only aiming layouts")
    return data


class _Counter(VecEnvWrapper):
    def __init__(self, raw, guard):
        super().__init__(raw)
        self.guard, self.attempted_steps, self.completed_steps, self.limit = guard, 0, 0, 0

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.guard.check("training_step")
        if self.attempted_steps + self.num_envs > self.limit:
            raise RuntimeError("Training rollout exceeds its fixed allowance")
        self.attempted_steps += self.num_envs
        result = self.venv.step_wait()
        self.completed_steps += self.num_envs
        return result


class _TrainCallback(_Callback):
    def __init__(self, guard, objective, starts, output):
        super().__init__(guard, objective)
        self.episodes, self.starts, self.output = [], starts, output

    def _on_step(self):
        keep_going = super()._on_step()
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if done:
                row = dict(env_index=i, direction=int(self.starts["directions"][i]),
                           outcome=info["outcome"], clear=info["outcome"] == "clear",
                           frames=int(info.get("frames", 0)), kills=int(info.get("kills", 0)),
                           deaths=int(info.get("deaths", 0)), **{"return":float(info["episode"]["r"])},
                           collected_step=int(self.model.num_timesteps))
                self.episodes.append(row)
                with (self.output / "train_episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
        return keep_going


class AimingRuntime:
    def __init__(self, starts, *, guard, device, output, target_steps=TRAIN_STEPS,
                 n_steps=128, batch_size=256):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.guard, self.optimized_steps, self.state_verified = guard, 0, True
        self.model, self.env, self.raw = None, None, None
        self.n_steps, self.target_steps = n_steps, target_steps
        self.optim_stats = dict(adam_steps=0, last_gradient_norm_post_clip=None)
        try:
            self.raw = make_aiming_env(starts, seed=0)
            self.counted = _Counter(self.raw, guard)
            objective = FiniteVictoryVecEnv(
                self.counted, np.ones(self.raw.num_envs, np.int64),
                lambda: self.raw.batch.stats[:, 1].astype(np.int64, copy=True),
                gamma=1., shaping=False)
            self.objective_version = objective.objective_version
            if self.objective_version != OBJECTIVE:
                raise ValueError("Wrong aiming reward objective")
            self.env = VecFrameStack(objective, n_stack=2, channels_order="first")
            self.rollout_size = self.raw.num_envs * n_steps
            self.model = MaskablePPO(
                "CnnPolicy", self.env, seed=0, device=device, verbose=0,
                **dict(PPO, n_steps=n_steps, batch_size=batch_size),
                policy_kwargs={"features_extractor_class": Grid26CNN, "normalize_images": False,
                               "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                rollout_buffer_class=_GuardedBuffer,
                rollout_buffer_kwargs={"channels_per_frame": 47, "max_storage_bytes": 2 * 1024**3})
            if str(device).startswith("cuda") and self.model.device.type != "cuda":
                raise RuntimeError("CUDA cannot silently fall back to CPU")
            self.model.rollout_buffer.guard = guard
            self._last_good = self._snapshot()
            original_step = self.model.policy.optimizer.step
            parameters, stats = list(self.model.policy.parameters()), self.optim_stats

            def guarded_step(*args, **kwargs):
                guard.check("adam_step")
                gradients = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
                norm = float(torch.stack(gradients).norm(2).item()) if gradients else 0.
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite clipped gradient norm")
                result = original_step(*args, **kwargs)
                stats["adam_steps"] += 1
                stats["last_gradient_norm_post_clip"] = norm
                return result

            self.model.policy.optimizer.step = guarded_step
            self.callback = _TrainCallback(guard, self.objective_version, starts, output)
            self.raw.seed(0)
            self.total, self.callback = self.model._setup_learn(target_steps, callback=self.callback)
            self.callback.on_training_start({}, {})
        except BaseException:
            self.close()
            raise

    def _snapshot(self):
        return dict(policy=_cpu_copy(self.model.policy.state_dict()),
                    optimizer=_cpu_copy(self.model.policy.optimizer.state_dict()),
                    updates=int(self.model._n_updates))

    def collect(self):
        if self.counted.attempted_steps + self.rollout_size > self.target_steps:
            raise RuntimeError("Fixed training cap would be exceeded")
        self.counted.limit = self.counted.attempted_steps + self.rollout_size
        return self.model.collect_rollouts(
            self.env, callback=self.callback, rollout_buffer=self.model.rollout_buffer,
            n_rollout_steps=self.n_steps)

    def optimize(self):
        self.guard.check("ppo_update")
        self.model._update_current_progress_remaining(self.model.num_timesteps, self.total)
        previous = deepcopy(self.optim_stats)
        self.state_verified = False
        try:
            self.model.train()
            metrics, undefined = _training_metrics(self.model)
            committed = self._snapshot()
        except BaseException:
            self.model.policy.load_state_dict(self._last_good["policy"], strict=True)
            self.model.policy.optimizer.load_state_dict(self._last_good["optimizer"])
            self.model.policy.optimizer.zero_grad(set_to_none=True)
            self.model._n_updates = self._last_good["updates"]
            self.optim_stats.update(previous)
            self.state_verified = True
            raise
        self.optimized_steps += self.rollout_size
        self._last_good, self.state_verified = committed, True
        return dict(step=self.optimized_steps, metrics=metrics, undefined_metrics=undefined,
                    **self.optim_stats)

    def close(self):
        if self.model is not None:
            self.model.policy.optimizer.__dict__.pop("step", None)
            self.model.rollout_buffer.guard = None
        if hasattr(self, "callback"):
            self.callback.on_training_end()
        if self.env is not None:
            self.env.close()
        elif self.raw is not None:
            self.raw.close()


def _outcome(ram):
    if int(ram[0x68]) != 0x80:
        return "base_lost"
    if int(ram[0x51]) == 0 or (int(ram[0x51]) <= 1 and 0x10 <= int(ram[0xa0]) < 0x80):
        return "lives_exhausted"
    return "clear" if int(ram[0x80]) == 0 else "timeout"


def evaluate_holdout(model, starts, *, guard):
    if len(starts["rams"]) != 128:
        raise ValueError("Exactly128 fixed heldout templates are required")
    seeds = [episode_seed(0, 991, index) for index in range(128)]
    report = dict(status="running", attempted_decisions=0, completed_decisions=0,
                  max_decisions=6400, episodes=[], wins=0, native_episode_seeds=seeds,
                  protocol="only first episode per template; finished slots still charged",
                  learned_full_game_win=False)
    batch = None
    try:
        guard.check("holdout_init")
        batch = NativeBatch(starts["rams"], 200, seeds=seeds)
        history = np.zeros((128, 94, 26, 26), np.float32)
        history[:, 47:] = batch.observe().copy()
        finished = np.zeros(128, bool)
        for _ in range(50):
            guard.check("holdout_predict")
            masks = batch.masks().copy()
            actions, _ = model.predict(history, deterministic=True, action_masks=masks)
            actions = np.asarray(actions)
            if actions.shape != (128,) or actions.dtype.kind not in "iu" \
                    or np.any(actions < 0) or np.any(actions >= 10) \
                    or not masks[np.arange(128), actions].all():
                raise ValueError("Heldout policy emitted a mask-invalid primitive")
            guard.check("holdout_step")
            report["attempted_decisions"] += 128
            observations, _, dones, stats = batch.step(actions)
            report["completed_decisions"] += 128
            history = np.roll(history, -47, axis=1)
            history[:, 47:] = observations
            for index in np.flatnonzero(dones & ~finished):
                outcome = _outcome(batch.ram(int(index)))
                row = dict(template_index=int(index), direction=int(starts["directions"][index]),
                           outcome=outcome, clear=outcome == "clear", frames=int(stats[index, 0]),
                           kills=int(stats[index, 1]), deaths=int(stats[index, 2]))
                report["episodes"].append(row)
                report["wins"] += int(row["clear"])
            finished |= dones
            if finished.all():
                break
        report["status"] = "completed" if finished.all() else "partial"
    except BaseException as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        if batch is not None:
            batch.close()
    report["per_direction"] = {}
    for direction, name in enumerate(("up", "down", "left", "right")):
        rows = [e for e in report["episodes"] if e["direction"] == direction]
        report["per_direction"][name] = dict(finished=len(rows), expected=32,
                                             wins=sum(r["clear"] for r in rows))
    return report


def save_checkpoint(runtime, path, provenance, role):
    if runtime.optimized_steps <= 0 or not runtime.state_verified:
        raise ValueError("Only verified completed optimizer updates may be saved")
    if _sources() != provenance["sources"]:
        raise ValueError("Frozen experiment sources changed during the run")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    old_steps = runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps = runtime.optimized_steps
        runtime.model.save(path, exclude=["_last_obs", "_last_episode_starts", "_last_original_obs"])
    finally:
        runtime.model.num_timesteps = old_steps
    manifest = dict(
        format=FORMAT, contract=contract(), objective_version=runtime.objective_version,
        checkpoint_file=path.name, checkpoint_sha256=_hash(path), sources=provenance["sources"],
        datasets=provenance["datasets"], optimized_steps=runtime.optimized_steps,
        optimizer_stats=deepcopy(runtime.optim_stats), ppo_epoch_updates=int(runtime.model._n_updates),
        role=role, actual_device=str(runtime.model.device), exact_simulator_resume=False,
        holdout_used_for_gradients_or_checkpoint_selection=False,
    )
    _write(path.with_suffix(".manifest.json"), manifest)
    return dict(path=str(path), sha256=manifest["checkpoint_sha256"], step=runtime.optimized_steps)


def load_aiming_checkpoint(path, device="cpu"):
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Missing/invalid aiming checkpoint manifest") from error
    if manifest.get("format") != FORMAT or manifest.get("contract") != contract() \
            or manifest.get("objective_version") != OBJECTIVE \
            or manifest.get("sources") != _sources() or manifest.get("checkpoint_file") != path.name \
            or manifest.get("exact_simulator_resume") is not False \
            or manifest.get("holdout_used_for_gradients_or_checkpoint_selection") is not False \
            or not isinstance(manifest.get("datasets"), dict) \
            or set(manifest["datasets"]) != {"train.npz", "train.json", "test.npz", "test.json"} \
            or type(manifest.get("optimized_steps")) is not int \
            or not 0 < manifest["optimized_steps"] <= TRAIN_STEPS \
            or manifest["optimized_steps"] % 4096 or _hash(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Invalid aiming source/dataset/objective/PPO checkpoint contract")
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in "0123456789abcdef" for c in value)
           for value in manifest["datasets"].values()):
        raise ValueError("Malformed aiming dataset identities")
    model = MaskablePPO.load(
        path, device=device, custom_objects={
            "rollout_buffer_class": NativeGPURunRolloutBuffer,
            "rollout_buffer_kwargs": {"channels_per_frame": 47, "max_storage_bytes": 2 * 1024**3},
            "n_steps": 2, "n_envs": 1, "batch_size": 2})
    from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
    if model.observation_space.shape != (94, 26, 26) or model.observation_space.dtype != np.float32 \
            or not np.all(model.observation_space.low == -1) or not np.all(model.observation_space.high == 1) \
            or model.action_space.n != 10 or model.action_space.start != 0 \
            or type(model.policy) is not MaskableActorCriticCnnPolicy \
            or type(model.policy.features_extractor) is not Grid26CNN \
            or model.policy.features_extractor.features_dim != 256 \
            or model.policy.normalize_images or model.policy.net_arch != {"pi": [64, 64], "vf": [64, 64]} \
            or not isinstance(model.policy.optimizer, torch.optim.Adam) \
            or isinstance(model.gamma, bool) or model.gamma != 1. or model.gae_lambda != .95 \
            or model.ent_coef != .01 or model.target_kl != .02 or model.n_epochs != 4 \
            or model.clip_range(1.) != .2 or model.learning_rate != 3e-4 \
            or model.num_timesteps != manifest["optimized_steps"] \
            or model._n_updates != manifest["ppo_epoch_updates"]:
        raise ValueError("Serialized model violates aiming PPO profile")
    return model


def run_aiming(datasets=DATASETS, output=OUTPUT, *, device="cuda"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, epoch = time.monotonic(), time.time()
    global_deadline = datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp()
    monitor = ResourceMonitor(output / "resources.jsonl", device=str(device).split(":")[0])
    guard = _Guard(started + 900, min(epoch + 900, global_deadline), monitor)
    report = dict(
        format="aiming-ppo-report-v1", reservation="babefd4ac8da4be3941238ee08297c85",
        status="running", stop_reason=None, error=None, seed=0, requested_device=str(device),
        contract=contract(), rom_decisions=0, init_probe_decisions=0,
        train_attempted_decisions=0, train_completed_decisions=0,
        optimized_steps=0, evaluation=None, heldout_evaluations=0, updates=[], checkpoints=[],
        full_game=False, learned_full_game_wins=0, max_total_decisions=TOTAL_CAP,
        warning="Conditional aiming microcontrol only; not original-ROM/full-game success.",
    )
    runtime, evaluation_model, provenance = None, None, None
    final_checkpoint = None
    try:
        monitor.check("admission", admission_slots=1, force=True)
        guard.check("loading_datasets")
        train, test = load_dataset(datasets, "train"), load_dataset(datasets, "test")
        if len(train["rams"]) != 32 or len(test["rams"]) != 128:
            raise ValueError("This candidate requires exactly32train/128test saved layouts")
        validate_split(train, test)
        provenance = dict(
            sources=_sources(), datasets={name: _hash(Path(datasets) / name)
                                          for name in ("train.npz", "train.json", "test.npz", "test.json")})
        report["provenance"] = provenance
        _write(output / "config.json", dict(contract=contract(), provenance=provenance, seed=0,
                                            experiment_runtime_cap_seconds=900))
        runtime = AimingRuntime(train, guard=guard, device=device, output=output)
        report["actual_device"] = str(runtime.model.device)
        if runtime.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        while runtime.optimized_steps < TRAIN_STEPS:
            guard.check("rollout_start")
            before = runtime.counted.completed_steps
            complete = runtime.collect()
            delta = runtime.counted.completed_steps - before
            report["train_attempted_decisions"] = runtime.counted.attempted_steps
            report["train_completed_decisions"] = runtime.counted.completed_steps
            if not complete or delta != 4096:
                raise DeadlineReached("Partial rollout is not eligible for optimization")
            guard.check("optimizer_start")
            update = runtime.optimize()
            report["optimized_steps"] = runtime.optimized_steps
            tail = runtime.callback.episodes[-100:]
            update["training_last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
            report["updates"].append(update)
            if runtime.optimized_steps % 16384 == 0:
                report["checkpoints"].append(save_checkpoint(
                    runtime, output / f"optimized-{runtime.optimized_steps:06d}.zip", provenance, "periodic"))
            _write(output / "report.json", report)
            print(json.dumps(update), flush=True)
        final_checkpoint = save_checkpoint(runtime, output / "final.zip", provenance, "final")
        report["checkpoints"].append(final_checkpoint)
        tail = runtime.callback.episodes[-100:]
        report["training_last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
        report["training_completed_episodes"] = len(runtime.callback.episodes)
        report["gpu_decoded_minibatches"] = runtime.model.rollout_buffer.gpu_decoded_minibatches
        runtime.close()
        runtime = None
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        guard.check("final_checkpoint_reload")
        evaluation_model = load_aiming_checkpoint(final_checkpoint["path"], device=device)
        report["heldout_evaluations"] = 1
        report["evaluation"] = evaluate_holdout(evaluation_model, test, guard=guard)
        if report["evaluation"]["status"] != "completed":
            raise DeadlineReached("Final heldout evaluation incomplete; not relaunched")
        report.update(status="completed", stop_reason="fixed_train_then_single_holdout")
    except BaseException as error:
        report.update(status="partial", stop_reason="stopped_no_relaunch",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if runtime is not None:
            report["train_attempted_decisions"] = runtime.counted.attempted_steps
            report["train_completed_decisions"] = runtime.counted.completed_steps
            report["optimized_steps"] = runtime.optimized_steps
            tail = runtime.callback.episodes[-100:]
            report["training_last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
            if runtime.optimized_steps and runtime.state_verified and provenance is not None:
                try:
                    report["checkpoints"].append(save_checkpoint(
                        runtime, output / "partial-optimized.zip", provenance, "partial_optimized"))
                except Exception as error:
                    report["checkpoint_error"] = f"{type(error).__name__}: {error}"
            runtime.close()
        evaluation = report["evaluation"] or {}
        report["evaluation_attempted_decisions"] = evaluation.get("attempted_decisions", 0)
        report["evaluation_completed_decisions"] = evaluation.get("completed_decisions", 0)
        report["total_decisions"] = report["train_attempted_decisions"] + report["evaluation_attempted_decisions"]
        report["completed_decisions"] = report["train_completed_decisions"] + report["evaluation_completed_decisions"]
        report["elapsed_seconds"] = time.monotonic() - started
        if report["total_decisions"] > TOTAL_CAP:
            report.update(status="failed", error="Hard interaction cap exceeded")
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        _write(output / "report.json", report)
        evaluation_model, runtime = None, None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    monitor.check("complete", force=True)
    print(json.dumps({key: report.get(key) for key in (
        "status", "error", "optimized_steps", "training_last100", "total_decisions",
        "elapsed_seconds")}, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = run_aiming(**vars(args))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
