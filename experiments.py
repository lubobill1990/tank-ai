"""Shared, versioned MaskablePPO workflow and full-stage evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = (
    "game_state.py", "battle_city_env.py", "heuristic.py", "policy.py",
    "experiments.py", "train.py", "eval.py", "watch.py", "recordings.py", "migration.py",
)
RECOVERY_RUNTIME_FILES = RUNTIME_FILES + ("skills.py", "ablation.py")
OUTCOMES = ("clear", "base_lost", "lives_exhausted", "timeout", "objective")
DEFAULT_BENCHMARK_STAGES = (6, 7)
DEFAULT_BENCHMARK_SEEDS = (1001, 2002)


def integer_list(value):
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected distinct comma-separated integers")
    return values


def validate_stages_seeds(stages, seeds):
    if not stages or len(stages) != len(set(stages)) or any(
        type(stage) is not int or not 1 <= stage <= 35 for stage in stages
    ):
        raise ValueError("stages must be distinct integers in 1..35")
    if not seeds or len(seeds) != len(set(seeds)) or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError("seeds must be distinct integers in 0..2**32-1")


@dataclass(frozen=True)
class TrainConfig:
    train_stages: tuple[int, ...] = (1, 2, 3)
    eval_stages: tuple[int, ...] = (4, 5)
    eval_seeds: tuple[int, ...] = (101, 202)
    benchmark_stages: tuple[int, ...] = DEFAULT_BENCHMARK_STAGES
    benchmark_seeds: tuple[int, ...] = DEFAULT_BENCHMARK_SEEDS
    seed: int = 0
    steps: int = 1_000_000
    envs: int = 1
    nstack: int = 2
    gamma: float = 0.99
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    device: str = "auto"
    max_frames: int = 12000
    eval_every: int = 100_000
    checkpoint_every: int = 10_000
    combat_steps: int = 0
    defense_steps: int = 0

    def validate(self):
        validate_stages_seeds(self.train_stages, (self.seed,))
        validate_stages_seeds(self.eval_stages, self.eval_seeds)
        validate_stages_seeds(self.benchmark_stages, self.benchmark_seeds)
        stage_sets = [set(self.train_stages), set(self.eval_stages), set(self.benchmark_stages)]
        if any(stage_sets[i] & stage_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("training, selection-eval and held-out benchmark stages must be disjoint")
        for name in ("steps", "envs", "nstack", "n_steps", "batch_size",
                     "n_epochs", "max_frames", "eval_every", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be finite and in (0, 1]")
        if self.n_steps < 2 or self.batch_size < 2:
            raise ValueError("n_steps and batch_size must be at least 2")
        if (self.n_steps * self.envs) % self.batch_size:
            raise ValueError("batch_size must divide n_steps * envs")
        if self.device not in ("auto", "cpu", "cuda") and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        if any(type(value) is not int or value < 0 for value in (self.combat_steps, self.defense_steps)):
            raise ValueError("curriculum budgets must be nonnegative integers")
        if self.combat_steps + self.defense_steps >= self.steps:
            raise ValueError("curriculum must leave a positive full-stage training budget")
        return self

    def phases(self):
        return [
            (name, budget) for name, budget in (
                ("combat", self.combat_steps), ("defense", self.defense_steps),
                ("full", self.steps - self.combat_steps - self.defense_steps),
            ) if budget
        ]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_signature(raw, *, nstack, root=ROOT, env_version=None, obs_channels=None,
                      channel_names=None):
    if env_version is None or obs_channels is None:
        from battle_city_env import CHANNEL_NAMES, ENV_VERSION, OBS_CHANNELS
        env_version, obs_channels = ENV_VERSION, OBS_CHANNELS
        channel_names = CHANNEL_NAMES
    if nstack <= 0:
        raise ValueError("nstack must be positive")
    shape = raw.observation_space.shape
    if not isinstance(raw.observation_space, spaces.Box) or shape != (obs_channels, 26, 26):
        raise ValueError(f"incompatible environment observation space: {raw.observation_space}")
    if not isinstance(raw.action_space, spaces.Discrete) or raw.action_space.n != 15:
        raise ValueError(f"incompatible environment action space: {raw.action_space}")
    root = Path(root)
    recovery = env_version == "grid26-events-v7-clean-terrain"
    files = RECOVERY_RUNTIME_FILES if recovery else RUNTIME_FILES
    signature = {
        "env_version": env_version,
        "obs_channels": obs_channels,
        "observation_shape": list(shape),
        "observation_dtype": str(raw.observation_space.dtype),
        "stacked_observation_shape": [shape[0] * nstack, *shape[1:]],
        "stack_axis": "first",
        "action_space": {"type": "Discrete", "shape": list(raw.action_space.shape),
                         "n": int(raw.action_space.n)},
        "source_sha256": {name: sha256(root / name) for name in files},
        "rom_sha256": sha256(root / "battle_city.nes"),
    }
    if channel_names is not None:
        if len(channel_names) != obs_channels or len(set(channel_names)) != obs_channels:
            raise ValueError("incompatible environment channel names")
        signature["channel_names"] = list(channel_names)
    if recovery:
        if raw.reward_profile not in ("shaped", "sparse"):
            raise ValueError("incompatible environment reward profile")
        signature["reward_profile"] = raw.reward_profile
    return signature


def make_manifest(config, signature):
    config.validate()
    return {
        "schema_version": 1,
        "algorithm": "MaskablePPO",
        "runtime": copy.deepcopy(signature),
        "training": json.loads(json.dumps(asdict(config))),
        "selection": {"objective": "full", "metric": "deterministic_full_stage_win_rate",
                      "stages": list(config.eval_stages), "seeds": list(config.eval_seeds)},
        "benchmark": {"objective": "full", "stages": list(config.benchmark_stages),
                      "seeds": list(config.benchmark_seeds)},
    }


def validate_manifest(manifest, signature, *, resume_config=None):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("missing or incompatible checkpoint manifest schema")
    if manifest.get("algorithm") != "MaskablePPO":
        raise ValueError("incompatible checkpoint algorithm")
    if manifest.get("runtime") != signature:
        actual = manifest.get("runtime", {})
        changed = sorted(key for key in signature if not isinstance(actual, dict) or actual.get(key) != signature[key])
        raise ValueError(f"incompatible checkpoint runtime: {', '.join(changed)}")
    if not isinstance(manifest.get("training"), dict) or set(manifest["training"]) != set(TrainConfig.__dataclass_fields__):
        raise ValueError("missing or invalid checkpoint training configuration")
    saved_config = TrainConfig(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in manifest["training"].items()
    }).validate()
    canonical = make_manifest(saved_config, signature)
    for name in ("selection", "benchmark"):
        if manifest.get(name) != canonical[name]:
            raise ValueError(f"incompatible checkpoint {name} configuration")
    if resume_config is not None:
        resume_config.validate()
        current = make_manifest(resume_config, signature)["training"]
        # Resume can extend the total target and change execution/eval frequency.
        allowed = {"steps", "device", "eval_every", "checkpoint_every"}
        changed = [name for name, value in current.items()
                   if name not in allowed and value != manifest["training"].get(name)]
        if changed:
            raise ValueError(f"incompatible resume configuration: {', '.join(changed)}")
        if resume_config.steps < saved_config.steps:
            raise ValueError("resume steps cannot reduce the original total target")
    return manifest


def write_json(path, data, *, overwrite=False):
    path = Path(path)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def create_run(path, manifest):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "manifest.json", manifest)
    return path


def checkpoint_manifest(path):
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("provide the explicit .zip checkpoint path")
    sidecar = path.with_suffix(".zip.manifest.json")
    if not sidecar.is_file():
        raise ValueError(f"checkpoint manifest required; legacy checkpoints are unsupported: {sidecar}")
    with sidecar.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("checkpoint_sha256") != sha256(path):
        raise ValueError("checkpoint integrity mismatch")
    return manifest


def save_checkpoint(path, model, manifest, *, phase, progress):
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("checkpoint filename must end in .zip")
    model.save(path)
    saved = copy.deepcopy(manifest)
    saved.update(
        checkpoint_sha256=sha256(path), phase=phase,
        progress=dict(progress), timesteps=int(model.num_timesteps),
    )
    write_json(path.with_suffix(".zip.manifest.json"), saved, overwrite=True)


def latest_checkpoint(run):
    run = Path(run).resolve()
    with (run / "latest.json").open(encoding="utf-8") as stream:
        latest = json.load(stream)
    if not isinstance(latest, dict) or type(latest.get("timesteps")) is not int \
            or latest["timesteps"] < 0 \
            or latest.get("checkpoint") != f"step-{latest['timesteps']:012d}.zip":
        raise ValueError("invalid latest checkpoint pointer")
    return run / "checkpoints" / latest["checkpoint"]


def publish_latest(run, model, manifest, *, phase, progress):
    """Publish immutable weights+manifest before atomically moving the pointer."""
    run = Path(run).resolve()
    directory = run / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"step-{model.num_timesteps:012d}.zip"
    if path.exists():
        raise FileExistsError(f"checkpoint already published: {path}")
    save_checkpoint(path, model, manifest, phase=phase, progress=progress)
    temporary = run / ".latest.json.tmp"
    try:
        write_json(temporary, {"checkpoint": path.name,
                               "timesteps": int(model.num_timesteps)}, overwrite=True)
        for attempt in range(5):
            try:
                os.replace(temporary, run / "latest.json")
                break
            except PermissionError:
                if attempt == 4:
                    raise
                warnings.warn("Latest pointer is in use; retrying publication", RuntimeWarning)
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)
    candidates = sorted(p for p in directory.glob("step-*.zip")
                        if len(p.stem) == 17 and p.stem[5:].isdigit())
    for old in candidates[:-3]:
        try:
            old.unlink(missing_ok=True)
            old.with_suffix(".zip.manifest.json").unlink(missing_ok=True)
        except PermissionError:
            warnings.warn(f"Checkpoint still in use; retention deferred: {old}", RuntimeWarning)
    return path


def load_checkpoint(path, signature, *, device, resume_config=None, env=None):
    from sb3_contrib import MaskablePPO
    manifest = checkpoint_manifest(path)
    validate_manifest(manifest, signature, resume_config=resume_config)
    model = MaskablePPO.load(path, env=env, device=device)
    if list(model.observation_space.shape) != signature["stacked_observation_shape"]:
        raise ValueError("incompatible checkpoint observation shape")
    if not isinstance(model.action_space, spaces.Discrete) or model.action_space.n != signature["action_space"]["n"]:
        raise ValueError("incompatible checkpoint action space")
    return model, manifest


def make_env(*, stage=1, objective="full", max_frames=12000, render_mode="rgb_array",
             reward_profile="shaped"):
    from battle_city_env import BattleCityEnv
    return BattleCityEnv(stage=stage, objective=objective, max_frames=max_frames,
                         render_mode=render_mode, reward_profile=reward_profile)


class EpisodePlayer:
    """Single-env counterpart to VecFrameStack(first), without terminal auto-reset."""
    def __init__(self, raw, *, nstack=2, model=None, deterministic=True):
        if nstack <= 0:
            raise ValueError("nstack must be positive")
        self.raw = raw
        self.model = model
        self.deterministic = deterministic
        self.channels = raw.observation_space.shape[0]
        self.observation = np.zeros(
            (self.channels * nstack, *raw.observation_space.shape[1:]),
            dtype=raw.observation_space.dtype,
        )
        self.done = True

    def reset(self, seed):
        observation, _ = self.raw.reset(seed=seed)
        self.observation.fill(0)
        self.observation[-self.channels:] = observation
        self.done = False
        return self.observation

    def step(self):
        if self.done:
            raise RuntimeError("reset the episode before stepping")
        if self.model is None:
            action = self.raw.heuristic_action()
        else:
            action, _ = self.model.predict(
                self.observation, deterministic=self.deterministic,
                action_masks=self.raw.action_masks(),
            )
        observation, reward, terminated, truncated, info = self.raw.step(int(np.asarray(action).item()))
        self.observation = np.roll(self.observation, -self.channels, axis=0)
        self.observation[-self.channels:] = observation
        self.done = bool(terminated or truncated)
        return reward, self.done, info


def validate_outcome(episode):
    outcome = episode["outcome"]
    if outcome not in OUTCOMES:
        raise ValueError(f"invalid terminal outcome: {outcome!r}")
    if outcome == "clear" and episode["objective"] != "full":
        raise ValueError("clear is only a win for a full-stage episode")
    if outcome == "objective" and episode["objective"] == "full":
        raise ValueError("full-stage episodes cannot end in scenario objective success")


def summarize_episodes(episodes):
    if not episodes:
        raise ValueError("cannot summarize zero episodes")
    outcomes = dict.fromkeys(OUTCOMES, 0)
    for episode in episodes:
        validate_outcome(episode)
        outcomes[episode["outcome"]] += 1
    summary = {
        "episodes": len(episodes), "wins": outcomes["clear"],
        "win_rate": outcomes["clear"] / len(episodes), "outcomes": outcomes,
    }
    for field in ("reward", "kills", "shots", "deaths", "pickups", "frames", "stagnation_frames",
                  "item_drops", "item_opportunities", "boundary_waste_frames",
                  "boundary_waste_streak_frames", "max_boundary_waste_frames"):
        summary[f"mean_{field}"] = float(np.mean([episode[field] for episode in episodes]))
    opportunities = sum(episode["item_opportunities"] for episode in episodes)
    summary["pickup_rate"] = sum(episode["pickups"] for episode in episodes) / opportunities \
        if opportunities else None
    total_frames = sum(episode["frames"] for episode in episodes)
    summary["boundary_waste_rate"] = sum(
        episode["boundary_waste_frames"] for episode in episodes) / total_frames \
        if total_frames else None
    components = set().union(*(episode["reward_components"] for episode in episodes))
    summary["mean_reward_components"] = {
        name: sum(episode["reward_components"].get(name, 0.0) for episode in episodes) / len(episodes)
        for name in sorted(components)
    }
    return summary


@contextmanager
def evaluation_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate(*, model=None, stages=DEFAULT_BENCHMARK_STAGES, seeds=DEFAULT_BENCHMARK_SEEDS,
             nstack=2, max_frames=12000, env_factory=make_env):
    validate_stages_seeds(stages, seeds)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    modes = {}
    for mode in ("deterministic", "sampled"):
        episodes = []
        for stage in stages:
            raw = env_factory(stage=stage, objective="full", max_frames=max_frames,
                              render_mode="rgb_array")
            try:
                player = EpisodePlayer(raw, nstack=nstack, model=model, deterministic=mode == "deterministic")
                for seed in seeds:
                    with evaluation_rng(seed):
                        player.reset(seed)
                        total, components, peak_stagnation = 0.0, {}, 0
                        for decision in range(1, max_frames + 1):
                            reward, done, info = player.step()
                            total += float(reward)
                            for name, value in info["reward_components"].items():
                                components[name] = components.get(name, 0.0) + float(value)
                            peak_stagnation = max(peak_stagnation, int(info["stagnation_frames"]))
                            if done:
                                break
                        else:
                            raise RuntimeError("environment exceeded max_frames decisions without terminating")
                        episode = {key: info[key] for key in (
                            "outcome", "objective", "stage", "kills", "lives", "frames",
                            "shots", "deaths", "pickups", "stagnation_frames",
                            "item_drops", "item_opportunities",
                            "boundary_waste_frames", "boundary_waste_streak_frames",
                            "max_boundary_waste_frames",
                        )}
                        validate_outcome(episode)
                        if episode["stage"] != stage or episode["objective"] != "full":
                            raise ValueError("evaluation environment changed its stage or full objective")
                        episode.update(seed=seed, reward=total, decisions=decision,
                                       reward_components=components,
                                       max_stagnation_frames=peak_stagnation)
                        episodes.append(episode)
            finally:
                raw.close()
        modes[mode] = {
            "summary": summarize_episodes(episodes),
            "per_stage": {str(stage): summarize_episodes([e for e in episodes if e["stage"] == stage])
                          for stage in stages},
            "episodes": episodes,
        }
    return {
        "schema_version": 1, "agent": "heuristic" if model is None else "MaskablePPO",
        "objective": "full", "stages": list(stages), "seeds": list(seeds),
        "max_frames": max_frames, "nstack": nstack, "modes": modes,
    }
