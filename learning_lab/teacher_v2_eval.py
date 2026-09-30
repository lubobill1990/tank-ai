"""CPU-only real-ROM development comparison for NON-LEARNED teacher v2."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from battle_city_env import BattleCityEnv
from learning_lab.teacher import TeacherController
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_resources import ResourceMonitor


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs" / "20260930-victory-teacher-v2-dev"
TASK_DEADLINE = datetime.fromisoformat("2026-09-30T15:10:40+08:00").timestamp()
MAX_EPISODES, MAX_DECISIONS = 24, 50000


def _write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("v1", "v2"), default="v2")
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(0, 1, 7, 39), default=[0, 1, 7, 39])
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    budget_path = EVIDENCE / "budget.json"
    if budget_path.exists():
        budget = json.loads(budget_path.read_text())
    else:
        budget = dict(started_at=time.time(),
                      deadline_monotonic=time.monotonic() + max(0., TASK_DEADLINE - time.time()),
                      episodes=0, decisions=0, completed_decisions=0, frames=0)
        _write(budget_path, budget)
    source = ROOT / "learning_lab" / ("teacher.py" if args.method == "v1" else "teacher_v2.py")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.stage != 1:
        v2_hash = hashlib.sha256((ROOT / "learning_lab" / "teacher_v2.py").read_bytes()).hexdigest()
        previous = [json.loads(line) for line in (EVIDENCE / "episodes.jsonl").read_text().splitlines()]
        clears = [r for r in previous if r["stage"] == 1 and r["method"] == "v2"
                  and r["clear"] and r["source_sha256"] == v2_hash]
        if len(clears) < 2:
            raise ValueError("Two stage-1 v2 clears are required before stage2/3 development")
    run = EVIDENCE / args.run
    run.mkdir(exist_ok=False)
    monitor = ResourceMonitor(run / "resources.jsonl", device="cpu")
    monitor.check("admission", admission_slots=1, force=True)
    (run / "controller_source.txt").write_bytes(source.read_bytes())
    provenance = dict(
        method=args.method, learned=False, stage=args.stage, source_sha256=source_hash,
        frozen_v1_sha256=hashlib.sha256((ROOT / "learning_lab" / "teacher.py").read_bytes()).hexdigest(),
        rom_sha256=hashlib.sha256((ROOT / "battle_city.nes").read_bytes()).hexdigest(),
        reset="normal original-ROM reset", observation="current canonical GameState",
        primitive_actions=10, frame_skip=4, max_frames=12000, full_enemies=20,
        ram_writes=False, oracle_rng=False, branch_rollouts=False, development_only=True,
    )
    _write(run / "provenance.json", provenance)
    rows = []
    for seed in args.seeds:
        if (budget["episodes"] >= MAX_EPISODES or budget["decisions"] >= MAX_DECISIONS
                or time.monotonic() >= budget["deadline_monotonic"]):
            break
        budget["episodes"] += 1
        _write(budget_path, budget)
        started = time.monotonic()
        env, info, error = None, dict(kills=0, deaths=0, frames=0, outcome=""), None
        bot = TeacherController() if args.method == "v1" else TeacherControllerV2()
        bot.reset()
        decisions, completed, previous_frame = 0, 0, 0
        modes = Counter()
        try:
            env = BattleCityEnv(stage=args.stage, objective="full", max_frames=12000, frame_skip=4)
            env.reset(seed=seed)
            with (run / f"stage{args.stage}-seed{seed}.jsonl").open("x", encoding="utf-8") as trace:
                while not info["outcome"]:
                    monitor.check("episode")
                    if time.monotonic() >= budget["deadline_monotonic"]:
                        raise TimeoutError("Absolute monotonic teacher-v2 deadline")
                    if budget["decisions"] >= MAX_DECISIONS:
                        raise TimeoutError("Teacher-v2 controller decision limit")
                    state = env.state
                    masks = env.action_masks()[:10]
                    action = bot.act(state, masks)
                    if not 0 <= action < 10 or not masks[action]:
                        raise AssertionError(f"Illegal teacher action {action}")
                    modes[bot.mode] += 1
                    sample = dict(
                        frame=state.frame, player=asdict(state.player) if state.player else None,
                        enemies=[asdict(e) for e in state.enemies],
                        bullets=[asdict(b) for b in state.bullets],
                        item=asdict(state.item) if state.item else None,
                        kills=state.kills, lives=state.lives, remaining=state.remaining_enemies,
                        shield=state.shield, freeze=state.freeze, fire_ready=state.fire_ready,
                        action=action, mode=bot.mode,
                    )
                    if decisions % 64 == 0:
                        sample.update(terrain=state.terrain.tolist(), brick_masks=state.brick_masks.tolist())
                    trace.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    decisions += 1
                    budget["decisions"] += 1
                    _, _, _, _, info = env.step(action)
                    completed += 1
                    budget["completed_decisions"] += 1
                    budget["frames"] += info["frames"] - previous_frame
                    previous_frame = info["frames"]
                    if decisions % 128 == 0:
                        _write(budget_path, budget)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            info["outcome"] = "aborted"
        finally:
            terminal = env.state if env is not None and not error else None
            row = dict(
                provenance, seed=seed, clear=info["outcome"] == "clear",
                outcome=info["outcome"], kills=info["kills"], deaths=info["deaths"],
                frames=info["frames"], decisions=decisions, completed_decisions=completed,
                time_seconds=time.monotonic() - started, modes=dict(modes),
                failure_cause=error or (None if info["outcome"] == "clear" else info["outcome"]),
                remaining_enemies=getattr(terminal, "remaining_enemies", None),
                base_destroyed=getattr(terminal, "base_destroyed", None),
                cumulative_episodes=budget["episodes"], cumulative_decisions=budget["decisions"],
                cumulative_frames=budget["frames"],
            )
            rows.append(row)
            _write(budget_path, budget)
            _write(run / "report.json", rows)
            with (EVIDENCE / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if env is not None:
                env.close()
        if error:
            raise RuntimeError(error)
    monitor.check("complete", force=True)


if __name__ == "__main__":
    main()
