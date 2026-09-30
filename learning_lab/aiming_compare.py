"""Two predeclared aiming representations; fresh seeds, no test-guided selection.

The frozen aiming runtime methods are reused, but construction explicitly accepts
the representation; no global monkeypatch changes the old trainer or encoder.
The old128 layouts are development-only. New seed999 layouts exclude both old
sets and are evaluated once per final candidate, never used for gradients.
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
from stable_baselines3.common.vec_env import VecFrameStack

from learning_lab import aiming_train as base
from learning_lab import aiming_task as scenes
from learning_lab.aiming_task import make_aiming_env
from learning_lab.ego_encoder import EgocentricGridCNN
from learning_lab.victory_objective import FiniteVictoryVecEnv
from policy import Grid26CNN
from rust_resources import ResourceMonitor
from native_rollout import NativeGPURunRolloutBuffer

EXTRACTORS = {"cnn": Grid26CNN, "egocentric": EgocentricGridCNN}
DATASET_KEYS = ("train.npz", "train.json", "old-test.npz", "old-test.json", "new-test.npz", "new-test.json")
TRAIN_STEPS = 262144
TOTAL_CAP = 540000
DEADLINE_EPOCH = datetime.fromisoformat("2026-09-30T18:57:07+08:00").timestamp()
FORMAT = "tank-ai-aiming-comparison-checkpoint-v1"
OUTPUT = base.ROOT / "runs" / "20260930-victory-lab" / "aiming-compare-v1"


def validate_new_holdout(train, old_test, new_test):
    if len(new_test["rams"]) != 128 or new_test["metadata"]["generator_seed"] != 999:
        raise ValueError("New holdout must contain128 seed999 layouts")
    scenes.validate_split(train, new_test)
    scenes.validate_split(old_test, new_test)


def experiment_contract(architecture):
    if architecture not in EXTRACTORS:
        raise ValueError("Only the two predeclared representations are authorized")
    result = base.contract()
    result.update(training_steps=TRAIN_STEPS, representation=architecture,
                  extractor=EXTRACTORS[architecture].__name__,
                  new_holdout_generator_seed=999, old_test_role="development_only",
                  exclusion="all training and old-test full RAM hashes",
                  comparison_seed=0, test_tuning_between_candidates=False)
    return result


def source_hashes():
    result = base._sources()
    result.update({name: base._hash(base.ROOT / name) for name in (
        "learning_lab\\aiming_compare.py", "learning_lab\\ego_encoder.py")})
    return result


def prepare_new_holdout(train, old_test):
    """Identity exclusion only; never query a policy or simulator outcome."""
    blocked = set(train["hashes"]) | set(old_test["hashes"])
    for count in (128, 256, 512, 1024):
        candidate = scenes.generate_aiming_starts(count, 999)
        allowed = ~np.isin(candidate["hashes"], list(blocked))
        if count == 128 and allowed.all():
            validate_new_holdout(train, old_test, candidate)
            return candidate
        selected = []
        for direction in range(4):
            for facing in range(4):
                for offset in (0, 2, 4, 6):
                    ids = np.flatnonzero(allowed & (candidate["directions"] == direction)
                                        & (candidate["player_facings"] == facing)
                                        & (candidate["subcell_offsets"] == offset))
                    if len(ids) < 2:
                        break
                    selected.extend(ids[:2].tolist())
        if len(selected) == 128:
            selected = np.random.default_rng(999).permutation(selected)
            result = {key: value[selected].copy() for key, value in candidate.items()
                      if isinstance(value, np.ndarray)}
            metadata = deepcopy(candidate["metadata"])
            metadata.update(count=128, oversampled_count=count, selected_indices=selected.tolist(),
                            selection="balanced direction/facing/subcell identity exclusion, no performance selection",
                            facing_design="Player facings stratified; enemy facings random from frozen generator")
            result["metadata"] = metadata
            validate_new_holdout(train, old_test, result)
            return result
    raise ValueError("Could not construct128 disjoint seed999 holdout layouts")


class CompareRuntime(base.AimingRuntime):
    def __init__(self, starts, *, architecture, guard, device, output,
                 target_steps=TRAIN_STEPS, n_steps=128, batch_size=256):
        experiment_contract(architecture)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.architecture = architecture
        self.guard, self.optimized_steps, self.state_verified = guard, 0, True
        self.model, self.env, self.raw = None, None, None
        self.n_steps, self.target_steps = n_steps, target_steps
        self.optim_stats = dict(adam_steps=0, last_gradient_norm_post_clip=None)
        try:
            self.raw = make_aiming_env(starts, seed=0)
            self.counted = base._Counter(self.raw, guard)
            objective = FiniteVictoryVecEnv(
                self.counted, np.ones(self.raw.num_envs, np.int64),
                lambda: self.raw.batch.stats[:, 1].astype(np.int64, copy=True),
                gamma=1., shaping=False)
            self.objective_version = objective.objective_version
            if self.objective_version != base.OBJECTIVE:
                raise ValueError("Wrong sparse objective")
            self.env = VecFrameStack(objective, n_stack=2, channels_order="first")
            self.rollout_size = self.raw.num_envs * n_steps
            self.model = MaskablePPO(
                "CnnPolicy", self.env, seed=0, device=device, verbose=0,
                **dict(base.PPO, n_steps=n_steps, batch_size=batch_size),
                policy_kwargs={"features_extractor_class": EXTRACTORS[architecture],
                               "normalize_images": False,
                               "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                rollout_buffer_class=base._GuardedBuffer,
                rollout_buffer_kwargs={"channels_per_frame": 47, "max_storage_bytes": 2 * 1024**3})
            if str(device).startswith("cuda") and self.model.device.type != "cuda":
                raise RuntimeError("Requested CUDA cannot fall back to CPU")
            self.model.rollout_buffer.guard = guard
            self._last_good = self._snapshot()
            original_step = self.model.policy.optimizer.step
            parameters, stats = list(self.model.policy.parameters()), self.optim_stats

            def guarded_step(*args, **kwargs):
                guard.check("adam_step")
                norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
                norm = float(torch.stack(norms).norm(2).item()) if norms else 0.
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                result = original_step(*args, **kwargs)
                stats["adam_steps"] += 1
                stats["last_gradient_norm_post_clip"] = norm
                return result

            self.model.policy.optimizer.step = guarded_step
            self.callback = base._TrainCallback(guard, self.objective_version, starts, output)
            self.raw.seed(0)
            self.total, self.callback = self.model._setup_learn(target_steps, callback=self.callback)
            self.callback.on_training_start({}, {})
        except BaseException:
            self.close()
            raise


def save_checkpoint(runtime, path, provenance, role):
    if runtime.optimized_steps <= 0 or not runtime.state_verified:
        raise ValueError("Only verified optimized weights may be saved")
    if source_hashes() != provenance["sources"]:
        raise ValueError("Frozen source identity changed")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    collected = runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps = runtime.optimized_steps
        runtime.model.save(path, exclude=["_last_obs", "_last_episode_starts", "_last_original_obs"])
    finally:
        runtime.model.num_timesteps = collected
    manifest = dict(
        format=FORMAT, architecture=runtime.architecture, contract=experiment_contract(runtime.architecture),
        objective_version=runtime.objective_version, checkpoint_file=path.name,
        checkpoint_sha256=base._hash(path), sources=provenance["sources"], datasets=provenance["datasets"],
        optimized_steps=runtime.optimized_steps, optimizer_stats=deepcopy(runtime.optim_stats),
        ppo_epoch_updates=int(runtime.model._n_updates), actual_device=str(runtime.model.device),
        role=role, fresh_initialization_seed=0, exact_simulator_resume=False,
        holdout_used_for_gradients_or_selection=False,
    )
    base._write(path.with_suffix(".manifest.json"), manifest)
    return dict(path=str(path), sha256=manifest["checkpoint_sha256"], step=runtime.optimized_steps)


def load_compare_checkpoint(path, device="cpu"):
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Missing/invalid aiming-comparison manifest") from error
    architecture = manifest.get("architecture")
    if architecture not in EXTRACTORS or manifest.get("format") != FORMAT \
            or manifest.get("contract") != experiment_contract(architecture) \
            or manifest.get("sources") != source_hashes() or manifest.get("objective_version") != base.OBJECTIVE \
            or manifest.get("checkpoint_file") != path.name \
            or manifest.get("exact_simulator_resume") is not False \
            or manifest.get("fresh_initialization_seed") != 0 \
            or manifest.get("holdout_used_for_gradients_or_selection") is not False \
            or type(manifest.get("optimized_steps")) is not int \
            or not 0 < manifest["optimized_steps"] <= TRAIN_STEPS or manifest["optimized_steps"] % 4096 \
            or not isinstance(manifest.get("datasets"), dict) or set(manifest["datasets"]) != set(DATASET_KEYS) \
            or base._hash(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Invalid comparison representation/source/profile/identity contract")
    if any(not isinstance(d, str) or len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
           for d in manifest["datasets"].values()):
        raise ValueError("Malformed comparison dataset hashes")
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
            or type(model.policy.features_extractor) is not EXTRACTORS[architecture] \
            or model.policy.features_extractor.features_dim != 256 \
            or model.policy.normalize_images or model.policy.net_arch != {"pi": [64, 64], "vf": [64, 64]} \
            or not isinstance(model.policy.optimizer, torch.optim.Adam) \
            or isinstance(model.gamma, bool) or model.gamma != 1. or model.gae_lambda != .95 \
            or model.learning_rate != 3e-4 or model.ent_coef != .01 or model.target_kl != .02 \
            or model.n_epochs != 4 or model.clip_range(1.) != .2 \
            or model.num_timesteps != manifest["optimized_steps"] \
            or model._n_updates != manifest["ppo_epoch_updates"]:
        raise ValueError("Serialized model violates the comparison contract")
    return model


def _release(runtime=None):
    if runtime is not None:
        runtime.close()
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def run_candidate(architecture, train, test, output, *, guard, provenance, device, allowance):
    output.mkdir(exist_ok=False)
    report = dict(
        architecture=architecture, status="running", error=None, contract=experiment_contract(architecture),
        train_attempted_decisions=0, train_completed_decisions=0, optimized_steps=0,
        updates=[], checkpoints=[], evaluation=None, heldout_evaluations=0,
        warning="Conditional microcontrol only; never a full-game win.",
    )
    runtime, evaluation_model = None, None
    started = time.monotonic()
    try:
        guard.check("candidate_start")
        if allowance < TRAIN_STEPS + 6400:
            raise RuntimeError("Insufficient remaining comparison allowance")
        runtime = CompareRuntime(train, architecture=architecture, guard=guard,
                                 device=device, output=output)
        report["actual_device"] = str(runtime.model.device)
        torch.cuda.reset_peak_memory_stats() if runtime.model.device.type == "cuda" else None
        while runtime.optimized_steps < TRAIN_STEPS:
            guard.check("rollout_start")
            before = runtime.counted.completed_steps
            completed = runtime.collect()
            report["train_attempted_decisions"] = runtime.counted.attempted_steps
            report["train_completed_decisions"] = runtime.counted.completed_steps
            if not completed or runtime.counted.completed_steps - before != 4096:
                raise base.DeadlineReached("Partial collection will not be optimized")
            update = runtime.optimize()
            report["optimized_steps"] = runtime.optimized_steps
            tail = runtime.callback.episodes[-100:]
            update["training_last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
            report["updates"].append(update)
            if runtime.optimized_steps % 16384 == 0:
                report["checkpoints"].append(save_checkpoint(
                    runtime, output / f"optimized-{runtime.optimized_steps:06d}.zip", provenance, "periodic"))
            base._write(output / "report.json", report)
            if runtime.optimized_steps % 32768 == 0:
                print(json.dumps(dict(architecture=architecture, **update)), flush=True)
        final = save_checkpoint(runtime, output / "final.zip", provenance, "final")
        report["checkpoints"].append(final)
        tail = runtime.callback.episodes[-100:]
        report["training_last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
        report["training_completed_episodes"] = len(runtime.callback.episodes)
        report["gpu_decoded_minibatches"] = runtime.model.rollout_buffer.gpu_decoded_minibatches
        runtime.close()
        runtime = None
        _release()
        guard.check("final_reload")
        evaluation_model = load_compare_checkpoint(final["path"], device=device)
        report["heldout_evaluations"] = 1
        report["evaluation"] = base.evaluate_holdout(evaluation_model, test, guard=guard)
        if report["evaluation"]["status"] != "completed":
            raise base.DeadlineReached("Final evaluation incomplete; no retry or next candidate")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        if runtime is not None:
            report["train_attempted_decisions"] = runtime.counted.attempted_steps
            report["train_completed_decisions"] = runtime.counted.completed_steps
            report["optimized_steps"] = runtime.optimized_steps
            tail = runtime.callback.episodes[-100:]
            report["training_last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
            if runtime.optimized_steps and runtime.state_verified:
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
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        base._write(output / "report.json", report)
        runtime, evaluation_model = None, None
        _release()
    print(json.dumps({k: report.get(k) for k in (
        "architecture", "status", "error", "optimized_steps", "training_last100",
        "total_decisions", "elapsed_seconds")}), flush=True)
    return report


def run_comparison(output=OUTPUT, *, datasets=base.DATASETS, device="cuda"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    epoch = min(DEADLINE_EPOCH, datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp())
    monitor = ResourceMonitor(output / "resources.jsonl", device=str(device).split(":")[0])
    guard = base._Guard(started + max(0., epoch - time.time()), epoch, monitor)
    before_init = scenes._ROM_INITIALIZATION_DECISIONS
    report = dict(
        reservation="e1d5661269384c609f4401e6828b0613", status="running", error=None,
        candidate_order=["cnn", "egocentric"], candidates=[], rom_initialization_decisions=0,
        max_total_decisions=TOTAL_CAP, new_holdout_seed=999, old_test_role="development_only",
        full_game=False, no_tuning_between_candidates=True, total_decisions=0,
    )
    try:
        guard.check("prepare")
        monitor.check("admission", admission_slots=1, force=True)
        train, old = base.load_dataset(datasets, "train"), base.load_dataset(datasets, "test")
        if len(train["rams"]) != 32 or len(old["rams"]) != 128:
            raise ValueError("Expected exact frozen32train/128old-development layouts")
        new = prepare_new_holdout(train, old)
        report["rom_initialization_decisions"] = scenes._ROM_INITIALIZATION_DECISIONS - before_init
        with (output / "new-test.npz").open("xb") as stream:
            np.savez_compressed(stream, **{k: v for k, v in new.items() if isinstance(v, np.ndarray)})
        base._write(output / "new-test.json", new["metadata"])
        report["new_holdout_identity"] = dict(
            count=128, generator_seed=999, overlap_train=0, overlap_old_test=0,
            source_metadata=new["metadata"])
        provenance = dict(sources=source_hashes(), datasets={
            "train.npz": base._hash(Path(datasets) / "train.npz"),
            "train.json": base._hash(Path(datasets) / "train.json"),
            "old-test.npz": base._hash(Path(datasets) / "test.npz"),
            "old-test.json": base._hash(Path(datasets) / "test.json"),
            "new-test.npz": base._hash(output / "new-test.npz"),
            "new-test.json": base._hash(output / "new-test.json"),
        })
        report["provenance"] = provenance
        base._write(output / "config.json", dict(
            contracts={name: experiment_contract(name) for name in EXTRACTORS},
            fixed_order=list(EXTRACTORS), provenance=provenance, deadline_epoch=epoch,
            maximum_planned_decisions=2 * TRAIN_STEPS + 2 * 6400 + report["rom_initialization_decisions"]))
        for architecture in ("cnn", "egocentric"):
            guard.check("next_candidate")
            used = report["rom_initialization_decisions"] + sum(r["total_decisions"] for r in report["candidates"])
            result = run_candidate(architecture, train, new, output / architecture,
                                   guard=guard, provenance=provenance, device=device,
                                   allowance=TOTAL_CAP - used)
            report["candidates"].append(result)
            report["total_decisions"] = used + result["total_decisions"]
            base._write(output / "report.json", report)
            if result["status"] != "completed":
                raise RuntimeError("Candidate stopped; no further candidate or budget extension")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        report["rom_initialization_decisions"] = scenes._ROM_INITIALIZATION_DECISIONS - before_init
        report["native_decisions"] = sum(r["total_decisions"] for r in report["candidates"])
        report["completed_native_decisions"] = sum(r["completed_decisions"] for r in report["candidates"])
        report["total_decisions"] = report["rom_initialization_decisions"] + report["native_decisions"]
        report["elapsed_seconds"] = time.monotonic() - started
        if report["total_decisions"] > TOTAL_CAP:
            report.update(status="failed", error="Hard comparison interaction cap exceeded")
        base._write(output / "report.json", report)
    monitor.check("complete", force=True)
    print(json.dumps({key: report.get(key) for key in (
        "status", "error", "rom_initialization_decisions", "native_decisions",
        "total_decisions", "elapsed_seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datasets", type=Path, default=base.DATASETS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_comparison(**vars(args))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
