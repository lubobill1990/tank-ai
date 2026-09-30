"""Minimal dynamic-layout PPO runner; caller owns reservation and GPU permission.

API:
  allowance = DynamicRunAllowance(max_decisions=268546, deadline=time.monotonic()+seconds)
  run_dynamic_aiming(output, allowance, steps=262144, device="cuda", seed=0)

262144 train +6400 one-shot final evaluation +2 possible ROM flush decisions.
No shared ledger writes, no budget extension, no full-game claims. Preparation
returns sampler/accounting/final_test/output/manifest; the runtime consumes that
object directly. Model checkpoints are policy+Adam only, not simulator resumes.
"""
import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from learning_lab import aiming_train as base
from learning_lab import aiming_task as scenes
from learning_lab.aiming_dynamic import (
    VERSION, DynamicStop, prepare_dynamic_aiming, make_dynamic_aiming_env, evaluate_final_once,
)
from learning_lab.ego_encoder import EgocentricGridCNN
from native_rollout import NativeGPURunRolloutBuffer

FORMAT = "dynamic-aiming-egocentric-ppo-checkpoint-v1"

@dataclass(frozen=True)
class DynamicRunAllowance:
    max_decisions: int
    deadline: float

    def __post_init__(self):
        if type(self.max_decisions) is not int or not 1 <= self.max_decisions <= 4000000:
            raise ValueError("Supply a pre-reserved decision allowance in1..4000000")
        if isinstance(self.deadline, bool) or not isinstance(self.deadline, (int, float)) \
                or not math.isfinite(self.deadline):
            raise ValueError("deadline must be an absolute finite time.monotonic() value")


def contract():
    value = base.contract()
    value.update(scene_version=VERSION, extractor="EgocentricGridCNN", training_steps=262144,
                 layout_policy="new unique balanced layout at every reset",
                 final_holdout_seed=20261002, dynamic_resets=True, exact_simulator_resume=False)
    return value


class DynamicCallback(base._Callback):
    def __init__(self, guard, output):
        super().__init__(guard, base.OBJECTIVE)
        self.output, self.episodes = output, []

    def _on_step(self):
        keep = super()._on_step()
        for index, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if done:
                if info.get("scene_version") != VERSION or not info.get("layout_hash"):
                    raise ValueError("Missing dynamic reset identity")
                row = dict(env_index=index, layout_hash=info["layout_hash"],
                           layout_number=info["layout_number"], direction=info["aiming_direction"],
                           outcome=info["outcome"], clear=info["outcome"] == "clear",
                           episode_return=float(info["episode"]["r"]), frames=info.get("frames"),
                           collected_step=int(self.model.num_timesteps))
                self.episodes.append(row)
                with (self.output / "train_episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
        return keep


class DynamicRuntime(base.AimingRuntime):
    def __init__(self, prepared, *, device, steps, seed=0, n_envs=32, n_steps=128, batch_size=256):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.prepared, self.guard = prepared, prepared["accounting"]
        self.target_steps, self.n_steps = steps, n_steps
        self.optimized_steps, self.state_verified = 0, True
        self.model = self.env = self.raw = None
        self.optim_stats = dict(adam_steps=0, last_gradient_norm_post_clip=None)
        if prepared["manifest"].get("objective") != base.OBJECTIVE \
                or prepared["manifest"].get("shaping") is not False:
            raise ValueError("Dynamic runner requires the prepared sparse objective")
        self.objective_version = prepared["manifest"]["objective"]
        try:
            stacked = make_dynamic_aiming_env(prepared["sampler"], n_envs=n_envs, seed=seed)
            self.counted = base._Counter(stacked, self.guard)
            self.env = self.counted
            self.rollout_size = n_envs * n_steps
            self.model = MaskablePPO(
                "CnnPolicy", self.env, seed=seed, device=device, verbose=0,
                **dict(base.PPO, n_steps=n_steps, batch_size=batch_size),
                policy_kwargs={"features_extractor_class": EgocentricGridCNN, "normalize_images": False,
                               "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                rollout_buffer_class=base._GuardedBuffer,
                rollout_buffer_kwargs={"channels_per_frame": 47, "max_storage_bytes": 2 * 1024**3})
            if str(device).startswith("cuda") and self.model.device.type != "cuda":
                raise RuntimeError("CUDA cannot silently fall back to CPU")
            self.model.rollout_buffer.guard = self.guard
            self._last_good = self._snapshot()
            original_step = self.model.policy.optimizer.step
            guard, parameters, statistics = self.guard, list(self.model.policy.parameters()), self.optim_stats

            def guarded_step(*args, **kwargs):
                guard.check("adam_step")
                values = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
                norm = float(torch.stack(values).norm(2).item()) if values else 0.
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite dynamic PPO gradient")
                result = original_step(*args, **kwargs)
                statistics["adam_steps"] += 1
                statistics["last_gradient_norm_post_clip"] = norm
                return result

            self.model.policy.optimizer.step = guarded_step
            self.callback = DynamicCallback(self.guard, prepared["output"])
            self.total, self.callback = self.model._setup_learn(steps, callback=self.callback)
            self.callback.on_training_start({}, {})
        except BaseException:
            self.close()
            raise


def source_hashes():
    result = base._sources()
    for name in ("learning_lab\\aiming_dynamic_train.py", "learning_lab\\aiming_dynamic.py",
                 "learning_lab\\ego_encoder.py", "skills.py", "game_state.py", "heuristic.py"):
        result[name] = base._hash(base.ROOT / name)
    return result


def save_dynamic_checkpoint(runtime, path, provenance, role):
    if runtime.optimized_steps <= 0 or not runtime.state_verified:
        raise ValueError("Only verified completed optimization may be checkpointed")
    if source_hashes() != provenance["sources"]:
        raise ValueError("Dynamic experiment sources changed")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    before = runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps = runtime.optimized_steps
        runtime.model.save(path, exclude=["_last_obs", "_last_episode_starts", "_last_original_obs"])
    finally:
        runtime.model.num_timesteps = before
    sampler = runtime.prepared["sampler"]
    if sampler.seen & sampler.excluded:
        raise ValueError("Training overlapped a heldout RAM identity")
    manifest = dict(
        format=FORMAT, contract=contract(), sources=provenance["sources"],
        environment=provenance["environment"], checkpoint_file=path.name,
        checkpoint_sha256=base._hash(path), optimized_steps=runtime.optimized_steps,
        objective_version=runtime.objective_version, ppo_epoch_updates=int(runtime.model._n_updates),
        optimizer_stats=runtime.optim_stats, observed_layouts=len(sampler.seen),
        observed_layout_hashes_sha256=hashlib.sha256("\n".join(sorted(sampler.seen)).encode()).hexdigest(),
        observed_layouts_scope="Includes reset-only and any partial collection; no excluded layouts",
        heldout_overlap=0, role=role, exact_simulator_resume=False,
        final_test_used_for_gradients_or_selection=False,
    )
    base._write(path.with_suffix(".manifest.json"), manifest)
    return dict(path=str(path), sha256=manifest["checkpoint_sha256"], step=runtime.optimized_steps)


def load_dynamic_checkpoint(path, device="cpu"):
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Missing/invalid dynamic aiming checkpoint manifest") from error
    environment = manifest.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("Invalid dynamic environment manifest")
    if manifest.get("format") != FORMAT or manifest.get("contract") != contract() \
            or manifest.get("sources") != source_hashes() \
            or manifest.get("objective_version") != base.OBJECTIVE \
            or manifest.get("exact_simulator_resume") is not False \
            or manifest.get("final_test_used_for_gradients_or_selection") is not False \
            or manifest.get("heldout_overlap") != 0 \
            or environment.get("format") != VERSION \
            or environment.get("objective") != base.OBJECTIVE or environment.get("shaping") is not False \
            or not all(isinstance(environment.get(k), str) and len(environment[k]) == 64
                       for k in ("excluded_hashes_sha256", "final_test_sha256")) \
            or type(manifest.get("optimized_steps")) is not int \
            or not 0 < manifest["optimized_steps"] <= 262144 or manifest["optimized_steps"] % 4096 \
            or manifest.get("checkpoint_file") != path.name or base._hash(path) != manifest.get("checkpoint_sha256"):
        raise ValueError("Invalid dynamic scene/exclusion/source/PPO checkpoint contract")
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
            or type(model.policy.features_extractor) is not EgocentricGridCNN \
            or model.policy.features_extractor.features_dim != 256 \
            or model.policy.normalize_images or model.policy.net_arch != {"pi": [64, 64], "vf": [64, 64]} \
            or not isinstance(model.policy.optimizer, torch.optim.Adam) \
            or model.gamma != 1 or isinstance(model.gamma, bool) or model.gae_lambda != .95 \
            or model.ent_coef != .01 or model.target_kl != .02 or model.n_epochs != 4 \
            or model.learning_rate != 3e-4 or model.clip_range(1.) != .2 \
            or model.num_timesteps != manifest["optimized_steps"] \
            or model._n_updates != manifest["ppo_epoch_updates"]:
        raise ValueError("Serialized dynamic policy violates its manifest")
    return model


def run_dynamic_aiming(output, budget_or_allowance, *, steps=262144, device="cpu", seed=0):
    """Prepare/train/evaluate once; caller supplies reservation, never an extension.

    DynamicRunAllowance(steps+6400+2, absolute_monotonic_deadline) covers a fresh
    process's possible2 ROM setup frames. Preparation returns a dict containing:
    sampler, accounting, final_test (arrays+metadata), output (Path), manifest.
    The final test is reserved locally, excluded before training, and single-use.
    """
    if not isinstance(budget_or_allowance, DynamicRunAllowance):
        raise ValueError("Pass DynamicRunAllowance(max_decisions, absolute_monotonic_deadline)")
    if type(steps) is not int or not 4096 <= steps <= 262144 or steps % 4096:
        raise ValueError("steps must be a multiple of4096, no greater than262144")
    if budget_or_allowance.max_decisions < steps + 6400 + 2:
        raise ValueError("Allowance must cover train+6400 final-test decisions+2 ROM setup frames")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Dynamic run output must be fresh")
    report = dict(format="dynamic-aiming-run-v1", status="running", error=None,
                  contract=contract(), seed=seed, requested_steps=steps, optimized_steps=0,
                  updates=[], checkpoints=[], evaluation=None, final_evaluations=0,
                  full_game=False, shared_ledger_written=False)
    prepared, runtime, model, provenance = None, None, None, None
    started = time.monotonic()
    rom_before_preparation = scenes._ROM_INITIALIZATION_DECISIONS
    try:
        prepared = prepare_dynamic_aiming(
            output, allowance=budget_or_allowance.max_decisions,
            deadline=budget_or_allowance.deadline, seed=seed)
        prepared["accounting"].monitor.device = str(device).split(":")[0]
        provenance = dict(sources=source_hashes(), environment=prepared["manifest"])
        base._write(output / "config.json", dict(contract=contract(), provenance=provenance,
                                                seed=seed, steps=steps, device=str(device)))
        runtime = DynamicRuntime(prepared, device=device, steps=steps, seed=seed)
        report["actual_device"] = str(runtime.model.device)
        account = prepared["accounting"]
        while runtime.optimized_steps < steps:
            account.check("rollout_start")
            before = account.native_steps_completed
            complete = runtime.collect()
            if not complete or account.native_steps_completed - before != 4096:
                raise DynamicStop("Partial dynamic rollout is not optimized")
            update = runtime.optimize()
            report["optimized_steps"] = runtime.optimized_steps
            tail = runtime.callback.episodes[-100:]
            update["last100"] = dict(episodes=len(tail), wins=sum(e["clear"] for e in tail))
            update["unique_layouts"] = len(prepared["sampler"].seen)
            report["updates"].append(update)
            if runtime.optimized_steps % 16384 == 0:
                report["checkpoints"].append(save_dynamic_checkpoint(
                    runtime, output / f"optimized-{runtime.optimized_steps:06d}.zip", provenance, "periodic"))
            report["accounting"] = account.snapshot()
            base._write(output / "report.json", report)
        final = save_dynamic_checkpoint(runtime, output / "final.zip", provenance, "final")
        report["checkpoints"].append(final)
        report["training_last100"] = report["updates"][-1]["last100"]
        runtime.close()
        runtime = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        account.check("final_checkpoint_reload")
        model = load_dynamic_checkpoint(final["path"], device=device)
        report["final_evaluations"] = 1
        report["evaluation"] = evaluate_final_once(model, prepared)
        report["status"] = "completed" if report["evaluation"]["status"] == "completed" else "partial"
    except BaseException as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        if runtime is not None:
            report["optimized_steps"] = runtime.optimized_steps
            if runtime.optimized_steps and runtime.state_verified and provenance is not None:
                try:
                    report["checkpoints"].append(save_dynamic_checkpoint(
                        runtime, output / "partial-optimized.zip", provenance, "partial_optimized"))
                except Exception as error:
                    report["checkpoint_error"] = f"{type(error).__name__}: {error}"
            runtime.close()
        if prepared is not None:
            report["accounting"] = prepared["accounting"].snapshot()
            report["unique_layouts"] = len(prepared["sampler"].seen)
            report["heldout_overlap"] = len(prepared["sampler"].seen & prepared["sampler"].excluded)
            base._write(output / "observed_layout_hashes.json", sorted(prepared["sampler"].seen))
        else:
            report["accounting"] = dict(
                rom_initialization_decisions=scenes._ROM_INITIALIZATION_DECISIONS - rom_before_preparation,
                native_steps_attempted=0, native_steps_completed=0,
                evaluation_attempted=0, evaluation_completed=0,
                total_decisions=scenes._ROM_INITIALIZATION_DECISIONS - rom_before_preparation)
        report["elapsed_seconds"] = time.monotonic() - started
        if output.is_dir():
            base._write(output / "report.json", report)
        runtime, model = None, None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowance", required=True, type=int)
    parser.add_argument("--deadline", required=True, type=float, help="Absolute time.monotonic() value")
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = run_dynamic_aiming(
        args.output, DynamicRunAllowance(args.allowance, args.deadline),
        steps=args.steps, device=args.device, seed=args.seed)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
