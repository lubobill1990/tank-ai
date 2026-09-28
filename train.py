"""Train a grid policy: strict resume, reviewed continuation, or weights-only warm start."""
import argparse
import copy
from functools import partial
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from experiments import (
    TrainConfig, create_run, evaluate, integer_list, load_checkpoint, make_env,
    make_manifest, publish_latest, runtime_signature, save_checkpoint, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, unique directory; never reused")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path,
                                help="explicit compatible .zip; --steps is the total target")
    initialization.add_argument("--warm-start", type=Path,
                                help="allowlisted legacy .zip weights only; optimizer/curriculum/steps restart at zero")
    initialization.add_argument("--continue-from", type=Path,
                                help="reviewed v5 -> v6 same-layout .zip; preserve optimizer/curriculum/steps; "
                                     "--steps is the total target")
    defaults = TrainConfig()
    for flag in ("train_stages", "eval_stages", "eval_seeds", "benchmark_stages", "benchmark_seeds"):
        parser.add_argument("--" + flag.replace("_", "-"), type=integer_list, default=getattr(defaults, flag))
    for flag in ("steps", "envs", "nstack", "seed", "n_steps", "batch_size", "n_epochs",
                 "max_frames", "eval_every", "checkpoint_every", "combat_steps", "defense_steps"):
        parser.add_argument("--" + flag.replace("_", "-"), type=int, default=getattr(defaults, flag))
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--device", default=defaults.device)
    args = parser.parse_args(argv)
    try:
        config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def config_from_args(args):
    return TrainConfig(**{field: getattr(args, field) for field in TrainConfig.__dataclass_fields__}).validate()


class TrainingStages(gym.Env):
    """Cycle every requested stage, including when there are fewer workers than stages."""
    def __init__(self, stages, offset, objective, max_frames, env_factory):
        super().__init__()
        self.stages, self.index = stages, offset % len(stages)
        self.factory = partial(env_factory, objective=objective, max_frames=max_frames)
        self.raw = self.factory(stage=stages[self.index])
        self.observation_space, self.action_space = self.raw.observation_space, self.raw.action_space
        self.metadata, self.render_mode = self.raw.metadata, "rgb_array"
        self.started = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.started and len(self.stages) > 1:
            self.raw.close()
            self.index = (self.index + 1) % len(self.stages)
            self.raw = self.factory(stage=self.stages[self.index])
        self.started = True
        return self.raw.reset(seed=int(self.np_random.integers(0, 2**32)), options=options)

    def step(self, action):
        return self.raw.step(action)

    def action_masks(self):
        return self.raw.action_masks()

    def heuristic_action(self):
        return self.raw.heuristic_action()

    def render(self):
        return self.raw.render()

    def close(self):
        self.raw.close()


def training_env(config, phase):
    factories = [
        partial(TrainingStages, stages=config.train_stages, offset=index,
                objective="full" if phase != "full" and config.envs > 1 and index == 0 else phase,
                max_frames=config.max_frames, env_factory=make_env)
        for index in range(config.envs)
    ]
    vec = SubprocVecEnv(factories, start_method="spawn") if config.envs > 1 else DummyVecEnv(factories)
    vec = VecMonitor(VecFrameStack(vec, n_stack=config.nstack, channels_order="first"))
    vec.seed(config.seed)
    return vec


class FullStageEvaluation:
    def __init__(self, config, run, manifest, progress, *, best_rate=-1.0):
        self.config, self.run, self.manifest = config, run, manifest
        self.progress = progress
        self.best_rate = best_rate
        self.last_eval = -1
        self.last_published = -1

    def publish(self, model, phase):
        if self.last_published != model.num_timesteps:
            checkpoint = publish_latest(self.run, model, self.manifest,
                                        phase=phase, progress=self.progress)
            self.last_published = model.num_timesteps
            print(f"published latest: {checkpoint.name}", flush=True)

    def run_evaluation(self, model, phase, *, phase_end=False):
        if self.last_eval == model.num_timesteps and not phase_end:
            return
        report = evaluate(model=model, stages=self.config.eval_stages, seeds=self.config.eval_seeds,
                          nstack=self.config.nstack, max_frames=self.config.max_frames)
        report.update(role="checkpoint_selection", phase=phase, timesteps=model.num_timesteps,
                      evaluation_point="phase_end" if phase_end else "periodic")
        suffix = "-phase-end" if phase_end else ""
        write_json(self.run / f"eval-{model.num_timesteps:012d}{suffix}.json", report)
        self.last_eval = model.num_timesteps
        rate = report["modes"]["deterministic"]["summary"]["win_rate"]
        print(f"phase={phase} steps={model.num_timesteps} full-stage win rate={rate:.3f}", flush=True)
        if rate > self.best_rate:
            self.best_rate = rate
            save_checkpoint(self.run / "best.zip", model, self.manifest,
                            phase=phase, progress=self.progress)
            write_json(self.run / "best-evaluation.json", report, overwrite=True)


def learn_phase(model, phase, budget, evaluation):
    """Checkpoint only completed optimizer work, never a partially collected rollout."""
    if budget <= 0:
        raise ValueError("phase budget must be positive")
    rollout = model.n_steps * model.n_envs
    target = model.num_timesteps + ((budget + rollout - 1) // rollout) * rollout
    next_eval = model.num_timesteps + evaluation.config.eval_every
    next_checkpoint = model.num_timesteps + evaluation.config.checkpoint_every
    while model.num_timesteps < target:
        boundary = min(next_eval, next_checkpoint)
        until_boundary = ((boundary - model.num_timesteps + rollout - 1) // rollout) * rollout
        chunk = min(target - model.num_timesteps, until_boundary)
        before = model.num_timesteps
        # Reuse the same env and last observation so episodes span chunk boundaries.
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        completed = model.num_timesteps - before
        if completed != chunk:
            raise RuntimeError(f"training chunk completed {completed} steps, expected {chunk}")
        evaluation.progress[phase] += completed
        evaluation.publish(model, phase)
        next_checkpoint = model.num_timesteps + evaluation.config.checkpoint_every
        if model.num_timesteps >= next_eval:
            evaluation.run_evaluation(model, phase)
            next_eval = model.num_timesteps + evaluation.config.eval_every
    evaluation.run_evaluation(model, phase, phase_end=True)


def main(argv=None):
    args = parse_args(argv)
    config = config_from_args(args)
    if args.output_dir.exists():
        raise FileExistsError(f"run directory already exists: {args.output_dir}")
    raw = make_env(stage=config.train_stages[0], max_frames=config.max_frames)
    try:
        signature = runtime_signature(raw, nstack=config.nstack)
    finally:
        raw.close()
    manifest = make_manifest(config, signature)
    progress = {"combat": 0, "defense": 0, "full": 0}
    model = None
    warm_source = None
    previous = None
    if args.warm_start:
        from migration import load_warm_start
        warm_source, manifest["warm_start"] = load_warm_start(
            args.warm_start, signature, nstack=config.nstack)
    if args.continue_from:
        from migration import load_runtime_continuation
        model, previous, manifest["continuation"] = load_runtime_continuation(
            args.continue_from, signature, nstack=config.nstack,
            device=config.device, resume_config=config)
    elif args.resume:
        model, previous = load_checkpoint(args.resume, signature, device=config.device, resume_config=config)
        manifest["resumed_from"] = str(args.resume.resolve())
    if previous is not None:
        if set(previous.get("progress", {})) != set(progress) or any(
            type(value) is not int or value < 0 for value in previous["progress"].values()
        ):
            raise ValueError("checkpoint has invalid curriculum progress")
        progress.update(previous["progress"])
        for name in ("warm_start", "continuation"):
            if name in previous and name not in manifest:
                manifest[name] = copy.deepcopy(previous[name])
    remaining = [(phase, budget - progress[phase]) for phase, budget in config.phases()
                 if budget > progress[phase]]
    if not remaining:
        raise ValueError("resume target already completed; increase --steps")
    run = create_run(args.output_dir, manifest)
    evaluation = FullStageEvaluation(config, run, manifest, progress)
    vec = None
    try:
        for phase, budget in remaining:
            vec = training_env(config, phase)
            if model is None:
                from policy import Grid26CNN
                model = MaskablePPO(
                    "CnnPolicy", vec, device=config.device, verbose=1, seed=config.seed,
                    policy_kwargs=dict(features_extractor_class=Grid26CNN,
                                       features_extractor_kwargs=dict(features_dim=256),
                                       normalize_images=False),
                    n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
                    learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.1, gamma=config.gamma,
                    gae_lambda=0.95, vf_coef=0.5,
                )
                if warm_source is not None:
                    from migration import copy_warm_start_weights
                    copy_warm_start_weights(warm_source, model, nstack=config.nstack)
                    warm_source = None
            else:
                model.set_env(vec, force_reset=True)
            if evaluation.last_published < 0:
                evaluation.publish(model, phase)
            print(f"phase={phase} requested_steps={budget} full_worker={config.envs > 1 or phase == 'full'} "
                  "(budgets round up to complete PPO rollouts)", flush=True)
            learn_phase(model, phase, budget, evaluation)
            save_checkpoint(run / f"{phase}.zip", model, manifest, phase=phase, progress=progress)
            vec.close()
            vec = None
        save_checkpoint(run / "final.zip", model, manifest, phase="full", progress=progress)
        write_json(run / "training-result.json", {
            "requested_steps": config.steps, "actual_timesteps": model.num_timesteps,
            "phase_timesteps": progress, "best_full_stage_win_rate": evaluation.best_rate,
            "benchmark_evaluated": False,
        })
        print(f"training complete: {run / 'final.zip'}", flush=True)
    finally:
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    main()
