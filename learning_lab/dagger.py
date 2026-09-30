"""Budget-bounded DAgger collection: mixed training trajectories, NOT model wins.

``collect_dagger(model, output, ..., deadline=absolute_time_monotonic)`` accepts
an already loaded masked policy. The CLI uses the verified lab imitation loader:
``uv run --no-sync python -m learning_lab.dagger --model BC_ZIP --output NEW_DIR``.
Only the launching parent reserves/settles interactions and chooses CPU/CUDA.
Raw pre-action frames use DemonstrationWriter; extra arrays/contexts describe
which controller supplied each action. Even beta=0 is collection, not validation.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np

from battle_city_env import BattleCityEnv
from learning_lab.demonstrations import DemonstrationWriter
from learning_lab.teacher import TeacherController
from rust_resources import ResourceMonitor, ResourcePressureError

_ROOT = Path(__file__).resolve().parents[1]
_GLOBAL_DEADLINE_EPOCH = datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp()
_LABEL = "Mixed training trajectory / not model win; no formal model validation"


class _DeadlineReached(RuntimeError):
    pass


class _DecisionBudgetReached(RuntimeError):
    pass


def _json(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise _DeadlineReached("Absolute monotonic deadline reached")


def _validate(stages, seeds, max_episodes, max_decisions, beta, rng_seed, deadline):
    stages, seeds = tuple(stages), tuple(seeds)
    for values, allowed, name in ((stages, {1, 2, 3}, "stages"), (seeds, {0, 1, 2}, "seeds")):
        if not values or len(set(values)) != len(values) or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v not in allowed
            for v in values
        ):
            raise ValueError(f"DAgger {name} must be distinct values from {sorted(allowed)}")
    for value, limit, name in ((max_episodes, 20, "max_episodes"),
                               (max_decisions, 32000, "max_decisions")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) \
                or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer in 1..{limit}")
    if isinstance(beta, (bool, np.bool_)) or not isinstance(beta, Real) \
            or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("teacher_probability must be finite and in [0,1]")
    if isinstance(rng_seed, (bool, np.bool_)) or not isinstance(rng_seed, Integral) \
            or not 0 <= rng_seed < 2**64:
        raise ValueError("rng_seed must be a nonnegative uint64 integer")
    if deadline is not None and (
        isinstance(deadline, (bool, np.bool_)) or not isinstance(deadline, Real)
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be an absolute finite time.monotonic() value")
    return tuple(map(int, stages)), tuple(map(int, seeds))


def _deadline(deadline):
    now = time.monotonic()
    requested = now + 45 * 60 if deadline is None else float(deadline)
    return min(requested, now + max(0., _GLOBAL_DEADLINE_EPOCH - time.time()))


def _primitive(action, masks, owner):
    action = np.asarray(action)
    if action.size != 1 or action.dtype.kind not in "iu":
        raise ValueError(f"{owner} must return a single integer primitive")
    value = int(action.item())
    if not 0 <= value < 10 or not masks[value]:
        raise ValueError(f"{owner} action {value} violates the legal primitive mask")
    return value


def _observation(observation):
    if not isinstance(observation, np.ndarray) or observation.shape != (47, 26, 26) \
            or observation.dtype != np.float32:
        raise ValueError("Expected canonical float32 (47,26,26) ROM observation")
    return observation


def _reason(error):
    if isinstance(error, _DeadlineReached):
        return "deadline"
    if isinstance(error, _DecisionBudgetReached):
        return "decision_budget"
    if isinstance(error, ResourcePressureError):
        return "resource_pressure"
    return "failed"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def collect_dagger(model, output, *, stages=(1,), seeds=(0, 1, 2), max_episodes=20,
                   max_decisions=32000, teacher_probability=.5, rng_seed=271, deadline=None):
    """Return exact env.step invocation counts; never reserve or settle a ledger.

    The deadline is absolute monotonic time, capped by the research global stop;
    None supplies a 45-minute local cap. Policy prediction is always deterministic
    and mask-validated, including teacher-controlled decisions. Teacher state is
    updated once for every labelled visited state. Stops/errors preserve reports
    and any collected dataset, returning status partial/failed rather than wins.
    Invalid configuration and non-fresh output paths raise before any reset.
    """
    stages, seeds = _validate(stages, seeds, max_episodes, max_decisions,
                              teacher_probability, rng_seed, deadline)
    if not callable(getattr(model, "predict", None)):
        raise ValueError("model must expose predict(observation, deterministic, action_masks)")
    output = Path(output).resolve()
    started = time.monotonic()
    deadline = _deadline(deadline)
    writer = DemonstrationWriter(output)
    device = str(getattr(model, "device", "cpu"))
    report = {
        "format": "dagger-collection-report-v1", "label": _LABEL,
        "formal_model_validation": False, "verified_model_wins": 0,
        "status": "running", "stop_reason": None, "error": None,
        "actual_decisions": 0, "completed_decisions": 0, "labelled_samples": 0,
        "decision_count_definition": "env.step invocations; completed calls counted separately",
        "student_control_decisions": 0, "teacher_control_decisions": 0,
        "overridden_student_decisions": 0, "teacher_student_disagreements": 0,
        "training_trajectory_clears": 0, "gameplay_frames": 0,
        "episodes": [], "stages": list(stages), "seeds": list(seeds),
        "max_episodes": int(max_episodes), "max_decisions": int(max_decisions),
        "teacher_probability": float(teacher_probability), "rng_seed": int(rng_seed),
        "control_rng": "one independent numpy.default_rng(rng_seed), one draw per decision",
        "deadline_monotonic": deadline, "history": "previous47,current47; zero prefix at every reset",
        "normal_rom_start": True, "ram_writes": False, "oracle_lookahead": False,
        "frame_skip": 4, "max_frames_per_episode": 12000,
        "native_environment_decisions": 0, "device": device,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "checkpoint_provenance": deepcopy(getattr(model, "_dagger_checkpoint_provenance", None)),
        "intervention_definition": "teacher selected by beta draw, even when student agrees",
        "override_definition": "teacher selected AND teacher_action differs from student_action",
        "executed_actions_semantics": "requested env.step primitives; contexts.step_completed confirms return",
        "source_sha256": {name: _hash(_ROOT / name) for name in (
            "battle_city.nes", "battle_city_env.py", "game_state.py", "heuristic.py",
            "learning_lab\\teacher.py", "learning_lab\\demonstrations.py",
            "learning_lab\\imitation.py", "learning_lab\\dagger.py")},
    }
    student_actions, interventions = [], []
    monitor = None
    rng = np.random.default_rng(rng_seed)
    schedule = [(stage, seed) for stage in stages for seed in seeds]
    try:
        _check_deadline(deadline)
        monitor = ResourceMonitor(output / "resources.jsonl", device=device.split(":")[0])
        monitor.check("admission", admission_slots=1, force=True)
        teacher = TeacherController()
        for episode_id in range(max_episodes):
            _check_deadline(deadline)
            if report["actual_decisions"] >= max_decisions:
                raise _DecisionBudgetReached("Reserved decision allowance exhausted")
            stage, seed = schedule[episode_id % len(schedule)]
            episode_started = time.monotonic()
            episode = dict(
                episode_id=episode_id, stage=stage, seed=seed, outcome="running",
                label=_LABEL, formal_model_validation=False, verified_model_win=False,
                trajectory_clear=False, decisions=0, completed_decisions=0,
                student_control_decisions=0, teacher_control_decisions=0,
                overridden_student_decisions=0, kills=0, deaths=0, frames=0,
                failure_cause=None,
            )
            report["episodes"].append(episode)
            env, info = None, {}
            try:
                monitor.check("episode-start", force=True)
                _check_deadline(deadline)
                env = BattleCityEnv(stage=stage, objective="full", max_frames=12000, frame_skip=4)
                teacher.reset()
                observation, info = env.reset(seed=seed)
                observation = _observation(observation)
                history = np.zeros((94, 26, 26), np.float32)
                history[47:] = observation
                previous_frame = 0
                while True:
                    _check_deadline(deadline)
                    if report["actual_decisions"] >= max_decisions:
                        raise _DecisionBudgetReached("Reserved decision allowance exhausted")
                    monitor.check("collection")
                    masks = env.action_masks()[:10].copy()
                    if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
                        raise ValueError("Expected ten nonempty boolean primitive masks")
                    teacher_action = _primitive(teacher.act(env.state, masks.copy()), masks, "Teacher")
                    _check_deadline(deadline)
                    predicted, _ = model.predict(history.copy(), deterministic=True,
                                                 action_masks=masks.copy())
                    student_action = _primitive(predicted, masks, "Student")
                    _check_deadline(deadline)
                    intervention = bool(rng.random() < teacher_probability)
                    action = teacher_action if intervention else student_action
                    context = dict(
                        episode_id=episode_id, stage=stage, seed=seed,
                        sample_index=report["labelled_samples"], frame=int(env.state.frame),
                        teacher_action=teacher_action, student_action=student_action,
                        executed_action=action, control_source="teacher" if intervention else "student",
                        teacher_mode=getattr(teacher, "mode", None), step_completed=False,
                    )
                    writer.add(observation, teacher_action, masks,
                               episode_start=episode["decisions"] == 0, executed_action=action)
                    student_actions.append(student_action)
                    interventions.append(intervention)
                    report["labelled_samples"] += 1
                    report["actual_decisions"] += 1
                    episode["decisions"] += 1
                    source_key = "teacher_control_decisions" if intervention else "student_control_decisions"
                    report[source_key] += 1
                    episode[source_key] += 1
                    overridden = int(intervention and teacher_action != student_action)
                    report["overridden_student_decisions"] += overridden
                    episode["overridden_student_decisions"] += overridden
                    report["teacher_student_disagreements"] += int(teacher_action != student_action)
                    try:
                        observation, _, terminated, truncated, info = env.step(action)
                        context["step_completed"] = True
                        report["completed_decisions"] += 1
                        episode["completed_decisions"] += 1
                    finally:
                        with (output / "contexts.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(context) + "\n")
                    observation = _observation(observation)
                    history = np.roll(history, -47, axis=0)
                    history[47:] = observation
                    current_frame = int(info["frames"])
                    report["gameplay_frames"] += current_frame - previous_frame
                    previous_frame = current_frame
                    if report["actual_decisions"] % 128 == 0:
                        _json(output / "progress.json", {
                            "actual_decisions": report["actual_decisions"],
                            "completed_decisions": report["completed_decisions"],
                            "episode_id": episode_id,
                        })
                    if terminated or truncated:
                        episode["outcome"] = info.get("outcome") or "unknown_terminal"
                        episode["trajectory_clear"] = episode["outcome"] == "clear"
                        report["training_trajectory_clears"] += int(episode["trajectory_clear"])
                        break
            except Exception as error:
                episode["outcome"] = _reason(error)
                episode["failure_cause"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                for key in ("kills", "deaths", "frames"):
                    episode[key] = int(info.get(key, 0))
                episode["time_seconds"] = time.monotonic() - episode_started
                episode["cumulative_decisions"] = report["actual_decisions"]
                with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(episode) + "\n")
                if env is not None:
                    env.close()
        monitor.check("complete", force=True)
        report["status"] = "completed"
    except Exception as error:
        report["stop_reason"] = _reason(error)
        report["status"] = "failed" if report["stop_reason"] == "failed" else "partial"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["reported_episodes"] = len(report["episodes"])
        try:
            np.save(output / "student_actions.npy", np.asarray(student_actions, np.int64),
                    allow_pickle=False)
            np.save(output / "teacher_interventions.npy", np.asarray(interventions, bool),
                    allow_pickle=False)
            np.save(output / "control_sources.npy",
                    np.asarray(["teacher" if flag else "student" for flag in interventions], dtype="<U7"),
                    allow_pickle=False)
            metadata = {key: deepcopy(value) for key, value in report.items() if key != "episodes"}
            writer.finish(report["episodes"], metadata)
        except Exception as error:
            report.update(status="failed", stop_reason="dataset_finalization",
                          error=f"{type(error).__name__}: {error}")
        finally:
            writer.close()
        _json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=32000)
    parser.add_argument("--teacher-probability", type=float, default=.5)
    parser.add_argument("--rng-seed", type=int, default=271)
    parser.add_argument("--deadline", type=float, help="Absolute time.monotonic() deadline")
    parser.add_argument("--device", default="cpu", help="Parent-selected inference device; default cpu")
    args = parser.parse_args(argv)
    _validate(args.stages, args.seeds, args.max_episodes, args.max_decisions,
              args.teacher_probability, args.rng_seed, args.deadline)
    _check_deadline(_deadline(args.deadline))
    from learning_lab.imitation import load_imitation_checkpoint
    model = load_imitation_checkpoint(args.model, device=args.device)
    checkpoint = args.model.resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / ("best.zip" if (checkpoint / "best.zip").exists() else "final.zip")
    model._dagger_checkpoint_provenance = {
        "path": str(checkpoint), "sha256": _hash(checkpoint),
        "manifest_sha256": _hash(checkpoint.with_suffix(".manifest.json")),
        "loader": "learning_lab.imitation.load_imitation_checkpoint",
    }
    report = collect_dagger(
        model, args.output, stages=args.stages, seeds=args.seeds,
        max_episodes=args.max_episodes, max_decisions=args.max_decisions,
        teacher_probability=args.teacher_probability, rng_seed=args.rng_seed,
        deadline=args.deadline,
    )
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
