"""Bounded, stage-1 recovery comparisons; never resumes the preserved long run.

Lab checkpoints deliberately use ``lab_schema_version``, not the normal training
manifest: stage-1 checkpoint selection is not a disjoint-stage benchmark. Load
them explicitly with load_lab_checkpoint(), never train.py --resume/watch.py.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTION_SEEDS = (101, 202)
CONFIRMATION_SEEDS = (1001, 2002)
LIMITATIONS = [
    "Stage 1 is used for both training and selection; this is not unseen-stage generalization.",
    "Stage-1 confirmation uses separate seeds 1001,2002, never checkpoint selection.",
    "Full-game seeds change only initial delays of at most 24 frames; seed diversity is limited.",
    "Skill scenes are synthetic real-ROM diagnostics, not full-stage wins.",
    "Two training seeds and short budgets provide limited evidence; all-zero results are inconclusive.",
    "Requested steps round up to complete PPO rollouts; no automatic budget extension or retry.",
    "Only deterministic full-stage wins then deterministic mean skill success select checkpoints.",
    "All arms are evaluated with shaped rewards; only training uses each arm's reward profile.",
]


@dataclass(frozen=True)
class Arm:
    initialization: str
    gamma: float
    reward_profile: str


ARMS = {
    "baseline": Arm("fresh", 0.99, "shaped"),
    "warm": Arm("reviewed_weights_only", 0.99, "shaped"),
    "horizon": Arm("fresh", 0.997, "shaped"),
    "sparse": Arm("fresh", 0.99, "sparse"),
}


@dataclass(frozen=True)
class TrialConfig:
    arm: str = "baseline"
    seed: int = 0
    steps_per_trial: int = 50000
    eval_every: int = 25000
    max_wall_seconds: float = 2700
    patience: int = 3
    n_envs: int = 2
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    nstack: int = 2
    max_frames: int = 12000
    device: str = "cuda"

    def validate(self):
        if self.arm not in ARMS:
            raise ValueError("unknown ablation arm")
        if type(self.seed) is not int or self.seed not in (0, 1):
            raise ValueError("approved training seeds are 0 and 1")
        for name in ("steps_per_trial", "eval_every", "patience", "n_envs", "n_steps",
                     "batch_size", "n_epochs", "nstack", "max_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps_per_trial > 50000:
            raise ValueError("steps_per_trial exceeds the approved 50000 requested-step cap")
        if not isinstance(self.max_wall_seconds, (int, float)) or \
                isinstance(self.max_wall_seconds, bool) or \
                not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be finite and positive")
        if self.n_envs != 2 or self.nstack != 2:
            raise ValueError("all trials require n_envs=2 and nstack=2")
        if not 2 <= self.n_steps <= 256 or self.batch_size < 2 or \
                self.rollout_size % self.batch_size:
            raise ValueError("batch_size >= 2 must divide n_steps * n_envs, with n_steps in 2..256")
        if self.device not in ("cpu", "cuda", "auto") and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        return self

    @property
    def rollout_size(self):
        return self.n_steps * self.n_envs

    @property
    def actual_step_cap(self):
        return self.round_steps(self.steps_per_trial)

    def round_steps(self, steps):
        return ((steps + self.rollout_size - 1) // self.rollout_size) * self.rollout_size

    @property
    def evaluation_steps(self):
        requested = range(self.eval_every, self.steps_per_trial, self.eval_every)
        return tuple(sorted({0, self.actual_step_cap, *(self.round_steps(n) for n in requested)}))


def trial_configs(*, seeds=(0, 1), **kwargs):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be distinct and nonempty")
    return [TrialConfig(arm=arm, seed=seed, **kwargs).validate()
            for arm in ARMS for seed in seeds]


def validate_parallel(value):
    if type(value) is not int or not 1 <= value <= 2:
        raise ValueError("parallel must be 1 or 2; never more than two GPU workers")
    return value


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value, *, replace=False):
    """Publish JSON atomically, allowing five attempts for reader sharing conflicts."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        return
    staging = path.with_name("." + path.name + ".writing")
    try:
        with staging.open("x", encoding="utf-8") as stream:
            stream.write(text)
        for attempt in range(5):
            try:
                os.replace(staging, path)
                break
            except PermissionError as exc:
                if attempt == 4:
                    raise
                if attempt == 0:
                    warnings.warn(
                        f"Atomic JSON replacement blocked for {path}; retrying: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )
                time.sleep(0.05 * (attempt + 1))
    finally:
        staging.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_frozen(signature, *, root=ROOT):
    root = Path(root)
    sources = signature.get("source_sha256", {})
    if not sources or not {"ablation.py", "skills.py"}.issubset(sources):
        raise ValueError("frozen runtime must include skills.py and ablation.py")
    expected = dict(sources, **{"battle_city.nes": signature["rom_sha256"]})
    changed = [name for name, digest in expected.items()
               if not (root / name).is_file() or file_hash(root / name) != digest]
    if changed:
        raise ValueError(f"frozen runtime changed: {', '.join(changed)}")


class TrialStop(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def require_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            require_finite(child)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise TrialStop("nonfinite")


def selection_metric(report):
    require_finite(report)
    mode = report["full_stage"]["modes"]["deterministic"]
    episodes, summary = mode["episodes"], mode["summary"]
    if not episodes or any(e["objective"] != "full" or e["stage"] != 1 for e in episodes):
        raise ValueError("selection requires true full-stage-1 episodes")
    wins = sum(e["outcome"] == "clear" for e in episodes)
    if summary["wins"] != wins or summary["episodes"] != len(episodes):
        raise ValueError("full-stage win count disagrees with raw episodes")
    skill = report["skills"]["modes"]["deterministic"]["summary"]["mean_success_rate"]
    if not 0 <= skill <= 1:
        raise ValueError("skill success rate must be in [0,1]")
    return wins, float(skill)


def metric_parts(metric):
    return None if metric is None else {
        "full_stage_wins": metric[0], "mean_skill_success_rate": metric[1],
    }


@dataclass
class StopGate:
    patience: int
    max_wall_seconds: float
    started: float
    best: tuple | None = None
    nonimprovements: int = 0

    def observe(self, metric):
        require_finite(metric)
        if self.best is None or metric > self.best:
            self.best, self.nonimprovements = metric, 0
            return True
        self.nonimprovements += 1
        return False

    def check_deadline(self):
        if time.monotonic() - self.started >= self.max_wall_seconds:
            raise TrialStop("wall_clock")

    def reason(self, *, steps, cap, now):
        if now - self.started >= self.max_wall_seconds:
            return "wall_clock"
        if steps >= cap:
            return "step_cap"
        if self.nonimprovements >= self.patience:
            return "patience"
        return None


def make_lab_manifest(config, signature, *, trial_id, provenance=None):
    config.validate()
    if trial_id != f"{config.arm}-seed-{config.seed}":
        raise ValueError("lab trial identity does not match configuration")
    return {
        "lab_schema_version": 1, "experiment": "stage1-recovery-ablation",
        "algorithm": "MaskablePPO", "trial_id": trial_id,
        "runtime": copy.deepcopy(signature), "config": asdict(config),
        "arm": asdict(ARMS[config.arm]),
        "training": {"stages": [1], "objective": "full"},
        "selection": {"stages": [1], "seeds": list(SELECTION_SEEDS),
                      "objective": "full", "mode": "deterministic", "reward_profile": "shaped",
                      "metric": ["full_stage_wins", "mean_skill_success_rate"]},
        "confirmation": {"stages": [1], "seeds": list(CONFIRMATION_SEEDS),
                         "objective": "full", "used_for_selection": False,
                         "reward_profile": "shaped"},
        "skill_variants": [0, 1],
        "budget": {"requested_steps": config.steps_per_trial,
                   "actual_step_cap": config.actual_step_cap,
                   "rollout_size": config.rollout_size,
                   "evaluation_steps": list(config.evaluation_steps)},
        "warm_start": provenance, "limitations": LIMITATIONS,
        "resume_supported": False,
    }


def validate_lab_manifest(manifest, signature):
    if not isinstance(manifest, dict) or manifest.get("lab_schema_version") != 1:
        raise ValueError("missing or incompatible lab manifest")
    if manifest.get("runtime") != signature:
        raise ValueError("incompatible lab checkpoint runtime")
    try:
        config = TrialConfig(**manifest["config"]).validate()
        canonical = make_lab_manifest(config, signature, trial_id=manifest["trial_id"],
                                      provenance=manifest["warm_start"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid lab checkpoint configuration") from exc
    if any(manifest.get(key) != value for key, value in canonical.items()):
        raise ValueError("incompatible lab checkpoint configuration or selection")
    if signature.get("reward_profile") != ARMS[config.arm].reward_profile:
        raise ValueError("lab checkpoint reward profile mismatch")
    if "timesteps" in manifest:
        steps = manifest["timesteps"]
        if type(steps) is not int or not 0 <= steps <= config.actual_step_cap or \
                steps % config.rollout_size or manifest.get("phase") != "full" or \
                manifest.get("progress") != {"full": steps}:
            raise ValueError("lab checkpoint requires completed full-stage optimizer rollouts")
    return manifest


def load_lab_checkpoint(path, signature, *, device="cpu", env=None):
    """Explicit, strict lab-only loader for evaluation, not training continuation."""
    from experiments import checkpoint_manifest
    from sb3_contrib import MaskablePPO
    from gymnasium import spaces
    manifest = validate_lab_manifest(checkpoint_manifest(path), signature)
    model = MaskablePPO.load(path, env=env, device=device)
    if list(model.observation_space.shape) != signature["stacked_observation_shape"] or \
            not isinstance(model.action_space, spaces.Discrete) or model.action_space.n != 15:
        raise ValueError("incompatible lab checkpoint spaces")
    if model.num_timesteps != manifest.get("timesteps"):
        raise ValueError("lab checkpoint timestep mismatch")
    return model, manifest


def prepare_batch(output_dir, *, warm_source, signatures, seeds=(0, 1), parallel=2, **kwargs):
    validate_parallel(parallel)
    configs = trial_configs(seeds=seeds, **kwargs)
    output, source = Path(output_dir).resolve(), Path(warm_source).resolve()
    if set(signatures) != {"shaped", "sparse"} or any(
        signatures[profile].get("reward_profile") != profile for profile in signatures
    ) or {k: v for k, v in signatures["shaped"].items() if k != "reward_profile"} != {
        k: v for k, v in signatures["sparse"].items() if k != "reward_profile"
    }:
        raise ValueError("arm runtime signatures must differ only in reward_profile")
    if not source.is_file() or source.suffix != ".zip":
        raise ValueError("warm_source must be an existing explicit .zip checkpoint")
    source_hash = file_hash(source)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("configs", "logs", "trials"):
        (output / name).mkdir()
    batch = {
        "lab_batch_schema_version": 1, "status": "prepared", "parallel": parallel,
        "output_dir": str(output), "warm_source": str(source),
        "warm_source_sha256": source_hash, "signatures": copy.deepcopy(signatures),
        "requested_steps": sum(c.steps_per_trial for c in configs),
        "rounded_step_cap": sum(c.actual_step_cap for c in configs),
        "trials": [], "limitations": LIMITATIONS,
    }
    for config in configs:
        trial_id = f"{config.arm}-seed-{config.seed}"
        task = {
            "trial_id": trial_id, "arm": config.arm, "seed": config.seed, "status": "pending",
            "config_path": str(output / "configs" / f"{trial_id}.json"),
            "run_dir": str(output / "trials" / trial_id),
            "log_path": str(output / "logs" / f"{trial_id}.log"),
        }
        record = dict(task, config=asdict(config), warm_source=str(source),
                      warm_source_sha256=source_hash,
                      runtime=copy.deepcopy(signatures[ARMS[config.arm].reward_profile]))
        write_json(task["config_path"], record)
        task["config_sha256"] = file_hash(task["config_path"])
        batch["trials"].append(task)
    write_json(output / "batch.json", batch)
    return batch


def finish_child(task, returncode):
    task = dict(task, returncode=returncode, status="failed")
    try:
        result = read_json(Path(task["run_dir"]) / "result.json")
        if returncode != 0:
            task["error"] = f"child exited {returncode}; inspect log and result"
        elif result.get("trial_id") != task["trial_id"] or result.get("status") != "completed":
            task["error"] = "child result is mismatched or unsuccessful"
        else:
            task["status"] = "completed"
        task["stop_reason"] = result.get("stop_reason")
    except (OSError, ValueError, TypeError) as exc:
        task["error"] = f"child exited {returncode} without a valid persistent result: {exc}"
    return task


def aggregate_results(batch):
    trials = []
    for task in batch["trials"]:
        row = copy.deepcopy(task)
        try:
            row["result"] = read_json(Path(task["run_dir"]) / "result.json")
        except (OSError, ValueError):
            row["result"] = None
        trials.append(row)
    completed = sum(t["status"] == "completed" for t in trials)
    return {
        "lab_report_schema_version": 1,
        "status": "completed" if completed == len(trials) else "failed",
        "completed_trials": completed, "total_trials": len(trials), "trials": trials,
        "requested_steps": batch.get("requested_steps"),
        "rounded_step_cap": batch.get("rounded_step_cap"), "limitations": LIMITATIONS,
        "interpretation": "No automatic winner claim; compare both seeds and held-out confirmation.",
    }


def run_batch(batch):
    """Own only direct Popen children, launch at most two, never retry a trial."""
    output = Path(batch["output_dir"])
    active = {}
    batch["status"] = "running"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        while active or any(t["status"] == "pending" for t in batch["trials"]):
            for signature in batch["signatures"].values():
                verify_frozen(signature)
            for index, task in enumerate(batch["trials"]):
                if len(active) >= batch["parallel"]:
                    break
                if task["status"] != "pending":
                    continue
                if file_hash(task["config_path"]) != task["config_sha256"]:
                    raise ValueError(f"frozen trial config changed: {task['trial_id']}")
                command = [sys.executable, str(ROOT / "ablation.py"), "--trial", task["config_path"]]
                log = Path(task["log_path"]).open("x", encoding="utf-8")
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT)
                except Exception as exc:
                    log.close()
                    task.update(status="failed", error=f"could not start child: {exc}")
                    continue
                task.update(status="running", pid=child.pid, command=command)
                config = read_json(task["config_path"])["config"]
                active[index] = (child, log, time.monotonic(), config["max_wall_seconds"])
            for index, (child, log, started, wall) in list(active.items()):
                code = child.poll()
                if code is None and time.monotonic() - started > wall + 60:
                    # A stuck native call gets one finite cleanup grace period.
                    child.terminate()
                    try:
                        code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        code = child.wait(timeout=10)
                    batch["trials"][index]["error"] = "wall watchdog exceeded budget + 60s grace"
                if code is not None:
                    log.close()
                    batch["trials"][index] = finish_child(batch["trials"][index], code)
                    del active[index]
            write_json(output / "batch.json", batch, replace=True)
            if active:
                time.sleep(0.2)
    except BaseException as exc:
        batch["error"] = f"{type(exc).__name__}: {exc}"
        for task in batch["trials"]:
            if task["status"] == "pending":
                task.update(status="cancelled", error="batch interrupted before launch")
    finally:
        for index, (child, log, _, _) in active.items():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            log.close()
            batch["trials"][index] = finish_child(batch["trials"][index], child.returncode)
        report = aggregate_results(batch)
        batch["status"] = report["status"]
        write_json(output / "batch.json", batch, replace=True)
        write_json(output / "report.json", report)
    return 0 if batch["status"] == "completed" else 1


def build_signature(config):
    from battle_city_env import BattleCityEnv
    from experiments import runtime_signature
    raw = BattleCityEnv(stage=1, objective="full", max_frames=config.max_frames,
                        reward_profile=ARMS[config.arm].reward_profile)
    try:
        return runtime_signature(raw, nstack=config.nstack)
    finally:
        raw.close()


def make_training_env(config):
    from functools import partial
    from battle_city_env import BattleCityEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
    # Two ROMs in each worker; no grandchildren that could outlive an owned worker.
    factory = partial(BattleCityEnv, stage=1, objective="full", max_frames=config.max_frames,
                      reward_profile=ARMS[config.arm].reward_profile)
    vec = VecMonitor(VecFrameStack(DummyVecEnv([factory] * config.n_envs),
                                  n_stack=config.nstack, channels_order="first"))
    vec.seed(config.seed)
    return vec


@contextmanager
def preserved_evaluation(model):
    from experiments import evaluation_rng
    training = model.policy.training
    try:
        with evaluation_rng(0):
            yield
    finally:
        model.policy.set_training_mode(training)


def build_model(config, vec, signature, *, warm_source=None):
    import torch
    from policy import Grid26CNN
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    model = MaskablePPO(
        "CnnPolicy", vec, seed=config.seed, device=config.device, verbose=0,
        policy_kwargs=dict(features_extractor_class=Grid26CNN, normalize_images=False),
        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
        learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.1, gamma=ARMS[config.arm].gamma,
        gae_lambda=0.95, vf_coef=0.5,
    )
    provenance = None
    if config.arm == "warm":
        if warm_source is None:
            raise ValueError("warm arm requires reviewed weights source")
        from migration import load_recovery_weights
        # Loading the old PPO must not perturb fresh-run exploration RNG streams.
        with preserved_evaluation(model):
            old, provenance = load_recovery_weights(warm_source, signature, nstack=config.nstack)
            model.policy.load_state_dict(old.policy.state_dict(), strict=True)
            del old
        if model.num_timesteps != 0 or model.policy.optimizer.state_dict()["state"]:
            raise ValueError("warm initialization must not restore optimizer or counters")
    return model, provenance


def check_model_finite(model):
    import torch

    def check(value):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise TrialStop("nonfinite")
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
        elif isinstance(value, (int, float)):
            require_finite(value)

    check(model.policy.state_dict())
    check(model.policy.optimizer.state_dict())
    # Explained variance can legitimately be undefined for constant returns.
    values = getattr(getattr(model, "_logger", None), "name_to_value", {})
    for key in ("train/loss", "train/value_loss", "train/entropy_loss",
                "train/approx_kl", "train/policy_gradient_loss", "train/clip_fraction"):
        if key in values:
            require_finite(float(values[key]))


class EvaluationPolicy:
    """Bound every evaluation decision without changing canonical evaluation code."""
    def __init__(self, model, gate):
        self.model, self.gate = model, gate

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, observation, **kwargs):
        import numpy as np
        self.gate.check_deadline()
        if not np.isfinite(observation).all():
            raise TrialStop("nonfinite")
        result = self.model.predict(observation, **kwargs)
        if not np.isfinite(result[0]).all():
            raise TrialStop("nonfinite")
        return result


def evaluate_checkpoint(model, config, gate, *, seeds):
    from functools import partial
    from battle_city_env import BattleCityEnv
    from experiments import evaluate
    from skills import evaluate_skills
    gate.check_deadline()
    bounded_model = EvaluationPolicy(model, gate)
    with preserved_evaluation(model):
        full = evaluate(model=bounded_model, stages=(1,), seeds=seeds, nstack=config.nstack,
                        max_frames=config.max_frames,
                        env_factory=partial(BattleCityEnv, reward_profile="shaped"))
        full["reward_profile"] = "shaped"
        gate.check_deadline()
        skills = evaluate_skills(model=bounded_model, seeds=seeds, variants=(0, 1),
                                 nstack=config.nstack)
    report = {"full_stage": full, "skills": skills}
    require_finite(report)
    gate.check_deadline()
    return report


def evaluation_summary(report):
    return {
        mode: {"full_stage": report["full_stage"]["modes"][mode]["summary"],
               "skills": report["skills"]["modes"][mode]["summary"],
               "per_skill": report["skills"]["modes"][mode]["per_skill"]}
        for mode in ("deterministic", "sampled")
    }


def copy_checkpoint(source, target):
    source, target = Path(source), Path(target)
    shutil.copyfile(source, target)
    shutil.copyfile(source.with_suffix(".zip.manifest.json"),
                    target.with_suffix(".zip.manifest.json"))


def run_trial(config_path):
    """Run one immutable trial record. Reports/checkpoints survive all handled stops."""
    started = time.monotonic()
    record = read_json(config_path)
    config = TrialConfig(**record["config"]).validate()
    trial_id = f"{config.arm}-seed-{config.seed}"
    if record["trial_id"] != trial_id:
        raise ValueError("trial identity does not match arm/seed")
    run, signature = Path(record["run_dir"]).resolve(), record["runtime"]
    run.mkdir(parents=True, exist_ok=False)
    (run / "checks").mkdir()
    (run / "checkpoints").mkdir()
    write_json(run / "config.json", record)
    gate = StopGate(config.patience, config.max_wall_seconds, started)
    result = {
        "trial_id": trial_id, "arm": config.arm, "seed": config.seed, "status": "running",
        "requested_steps": config.steps_per_trial, "actual_step_cap": config.actual_step_cap,
        "actual_timesteps": 0, "optimized_timesteps": 0, "stop_reason": None,
        "initial_metric": None, "best_metric": None, "final_metric": None,
        "checks": [], "confirmation": {"status": "not_run"}, "warm_start": None,
    }
    model = vec = manifest = last_safe = None
    optimized_steps = 0

    def save_boundary(path):
        from experiments import save_checkpoint
        save_checkpoint(path, model, manifest, phase="full", progress={"full": optimized_steps})

    def selection_check():
        nonlocal last_safe
        gate.check_deadline()
        verify_frozen(signature)
        check_model_finite(model)
        checkpoint = run / "checkpoints" / f"step-{optimized_steps:012d}.zip"
        if checkpoint.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint}")
        save_boundary(checkpoint)
        last_safe = checkpoint
        evaluation = evaluate_checkpoint(model, config, gate, seeds=SELECTION_SEEDS)
        verify_frozen(signature)
        metric = selection_metric(evaluation)
        improved = gate.observe(metric)
        evaluation.update(role="checkpoint_selection", timesteps=optimized_steps,
                          metric=metric_parts(metric), improved=improved)
        path = run / "checks" / f"step-{optimized_steps:012d}.json"
        write_json(path, evaluation)
        result["checks"].append({"timesteps": optimized_steps, "report": str(path),
                                 "checkpoint": str(checkpoint), "metric": metric_parts(metric)})
        if result["initial_metric"] is None:
            result["initial_metric"] = metric_parts(metric)
            result["initial_summary"] = evaluation_summary(evaluation)
        result["final_metric"] = metric_parts(metric)
        result["final_summary"] = evaluation_summary(evaluation)
        if improved:
            result["best_metric"] = metric_parts(metric)
            result["best_timesteps"] = optimized_steps
            result["best_summary"] = evaluation_summary(evaluation)
            copy_checkpoint(checkpoint, run / "best.zip")
        write_json(run / "progress.json", dict(result, actual_timesteps=optimized_steps,
                                              elapsed_seconds=time.monotonic() - started),
                   replace=True)
        print(f"{trial_id} steps={optimized_steps} wins={metric[0]} skill={metric[1]:.4f}",
              flush=True)

    try:
        verify_frozen(signature)
        if build_signature(config) != signature:
            raise ValueError("child runtime signature differs from frozen host signature")
        gate.check_deadline()
        if config.arm == "warm" and file_hash(record["warm_source"]) != record["warm_source_sha256"]:
            raise ValueError("frozen warm source checkpoint changed")
        vec = make_training_env(config)
        model, provenance = build_model(config, vec, signature, warm_source=record["warm_source"])
        result["warm_start"] = provenance
        manifest = make_lab_manifest(config, signature, trial_id=trial_id, provenance=provenance)
        write_json(run / "lab-manifest.json", manifest)

        from stable_baselines3.common.callbacks import BaseCallback
        import numpy as np

        class BudgetCallback(BaseCallback):
            def _on_step(self):
                gate.check_deadline()
                for name in ("rewards", "new_obs"):
                    if not np.isfinite(self.locals[name]).all():
                        raise TrialStop("nonfinite")
                return True

        selection_check()
        while True:
            reason = gate.reason(steps=optimized_steps, cap=config.actual_step_cap,
                                 now=time.monotonic())
            if reason:
                result["stop_reason"] = reason
                break
            verify_frozen(signature)
            # One complete optimizer rollout; same model/env/last observation throughout.
            model.learn(total_timesteps=config.rollout_size, reset_num_timesteps=False,
                        callback=BudgetCallback())
            if model.num_timesteps != optimized_steps + config.rollout_size:
                raise RuntimeError("PPO did not finish exactly one complete rollout")
            check_model_finite(model)
            optimized_steps = int(model.num_timesteps)
            if optimized_steps in config.evaluation_steps:
                selection_check()
        result["actual_timesteps"] = int(model.num_timesteps)
        model.num_timesteps = optimized_steps
        verify_frozen(signature)
        save_boundary(run / "final.zip")
        result["final_checkpoint_timesteps"] = optimized_steps
        if (run / "best.zip").exists() and result["stop_reason"] != "wall_clock":
            best, _ = load_lab_checkpoint(run / "best.zip", signature, device=config.device)
            with preserved_evaluation(model):
                confirmation = evaluate_checkpoint(best, config, gate, seeds=CONFIRMATION_SEEDS)
            verify_frozen(signature)
            confirmation.update(role="held_out_confirmation", used_for_selection=False,
                                timesteps=best.num_timesteps)
            write_json(run / "confirmation.json", confirmation)
            result["confirmation"] = {
                "status": "completed", "path": str(run / "confirmation.json"),
                "timesteps": best.num_timesteps, "summary": evaluation_summary(confirmation),
            }
            del best
        result["status"] = "completed"
    except TrialStop as exc:
        result.update(status="failed" if exc.reason == "nonfinite" else "completed",
                      stop_reason=exc.reason)
        if model is not None:
            result["actual_timesteps"] = int(model.num_timesteps)
            model.num_timesteps = optimized_steps
            if exc.reason == "wall_clock" and manifest is not None:
                check_model_finite(model)
                save_boundary(run / "final.zip")
                result["final_checkpoint_timesteps"] = optimized_steps
            elif last_safe is not None:
                copy_checkpoint(last_safe, run / "final.zip")
                result["final_checkpoint_timesteps"] = read_json(
                    last_safe.with_suffix(".zip.manifest.json"))["timesteps"]
        result["confirmation"].setdefault("reason", exc.reason)
    except BaseException as exc:
        result.update(status="failed", stop_reason="error",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if model is not None:
            result["actual_timesteps"] = int(model.num_timesteps)
        if last_safe is not None and not (run / "final.zip").exists():
            copy_checkpoint(last_safe, run / "final.zip")
            result["final_checkpoint_timesteps"] = read_json(
                last_safe.with_suffix(".zip.manifest.json"))["timesteps"]
    finally:
        result["optimized_timesteps"] = optimized_steps
        result["elapsed_seconds"] = time.monotonic() - started
        result["final_metric_timesteps"] = (
            result["checks"][-1]["timesteps"] if result["checks"] else None)
        write_json(run / "result.json", result)
        if vec is not None:
            vec.close()
    print(f"{trial_id}: {result['status']} ({result['stop_reason']}) -> {run}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, help="internal: one immutable trial config JSON")
    parser.add_argument("--output-dir", type=Path, help="new unique batch directory, never reused")
    parser.add_argument("--warm-source", type=Path, help="explicit preserved v6 checkpoint")
    parser.add_argument("--seeds", default="0,1", help="approved training seeds, default 0,1")
    parser.add_argument("--parallel", type=int, default=2)
    defaults = TrialConfig()
    for name in ("steps_per_trial", "eval_every", "patience", "n_steps", "batch_size",
                 "n_epochs", "max_frames"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--max-wall-seconds", type=float, default=defaults.max_wall_seconds)
    parser.add_argument("--device", default=defaults.device)
    args = parser.parse_args(argv)
    if args.trial:
        if args.output_dir or args.warm_source:
            parser.error("--trial cannot be combined with batch paths")
        return args
    if args.output_dir is None or args.warm_source is None:
        parser.error("--output-dir and --warm-source are required for a batch")
    try:
        args.seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
        validate_parallel(args.parallel)
        trial_configs(seeds=args.seeds, **config_arguments(args))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def config_arguments(args):
    return {name: getattr(args, name) for name in TrialConfig.__dataclass_fields__
            if name not in ("arm", "seed", "n_envs", "nstack")}


def main(argv=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    args = parse_args(argv)
    if args.trial:
        result = run_trial(args.trial.resolve())
        return 0 if result["status"] == "completed" else 1
    signatures = {profile: build_signature(TrialConfig(arm=arm, device=args.device))
                  for profile, arm in (("shaped", "baseline"), ("sparse", "sparse"))}
    for signature in signatures.values():
        verify_frozen(signature)
    batch = prepare_batch(args.output_dir, warm_source=args.warm_source, signatures=signatures,
                          seeds=args.seeds, parallel=args.parallel, **config_arguments(args))
    print(f"Batch: {batch['requested_steps']} requested, {batch['rounded_step_cap']} rounded cap; "
          f"at most {batch['parallel']} workers -> {batch['output_dir']}", flush=True)
    return run_batch(batch)


if __name__ == "__main__":
    raise SystemExit(main())
