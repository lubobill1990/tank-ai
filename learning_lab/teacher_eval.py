"""Budgeted real-ROM development evaluation; every controller here is NON-LEARNED."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from battle_city_env import BattleCityEnv
from heuristic import HeuristicBot
from rust_resources import ResourceMonitor


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs" / "20260930-victory-teacher-dev"
MAX_EPISODES = 40
MAX_DECISIONS = 120000


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("heuristic", "teacher"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--frame-skip", type=int, choices=(4,), default=4)
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ledger_path = EVIDENCE / "budget.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = dict(started_at=time.time(), deadline=time.time() + 85 * 60,
                      episodes=0, decisions=0, frames=0)
        _write(ledger_path, ledger)
    run = EVIDENCE / args.run
    run.mkdir(exist_ok=False)
    monitor = ResourceMonitor(run / "resources.jsonl")
    monitor.check("admission", admission_slots=1, force=True)
    if args.method == "teacher":
        from learning_lab.teacher import TeacherController
        bot = TeacherController()
        source = ROOT / "learning_lab" / "teacher.py"
    else:
        bot = HeuristicBot()
        source = ROOT / "heuristic.py"
    (run / "controller_source.txt").write_text(source.read_text(encoding="utf-8"),
                                               encoding="utf-8")
    provenance = dict(method=args.method, learned=False, stage=args.stage,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      rom_sha256=hashlib.sha256((ROOT / "battle_city.nes").read_bytes()).hexdigest(),
                      episode_start="BattleCityEnv.reset normal ROM boot; no RAM writes",
                      observation="canonical current GameState only",
                      action_space="10 primitive actions", frame_skip=args.frame_skip,
                      mask_adapter="masked heuristic requests become NOOP; teacher must obey masks",
                      max_frames=12000, max_enemies=20, development_only=True)
    _write(run / "provenance.json", provenance)
    rows = []
    env = BattleCityEnv(stage=args.stage, objective="full", max_frames=12000,
                        frame_skip=args.frame_skip)
    try:
        for seed in args.seeds:
            if (ledger["episodes"] >= MAX_EPISODES or ledger["decisions"] >= MAX_DECISIONS
                    or time.time() >= ledger["deadline"]):
                break
            ledger["episodes"] += 1
            _write(ledger_path, ledger)
            bot.reset()
            started = time.time()
            decisions, masked_requests = 0, 0
            outcome, error = "", None
            trace_path = run / f"stage{args.stage}-seed{seed}.jsonl"
            previous_frames = 0
            info = dict(kills=0, deaths=0, frames=0)
            try:
                _, info = env.reset(seed=seed)
                with trace_path.open("x", encoding="utf-8") as trace:
                    while not outcome:
                        monitor.check("episode")
                        if time.time() >= ledger["deadline"]:
                            raise TimeoutError("teacher experiment deadline")
                        if ledger["decisions"] >= MAX_DECISIONS:
                            raise TimeoutError("teacher decision budget")
                        state = env.state
                        masks = env.action_masks()[:10]
                        requested = bot.act(state, masks) if args.method == "teacher" else bot.act(state)
                        if not masks[requested]:
                            masked_requests += 1
                            if args.method == "teacher":
                                raise AssertionError("Teacher violated a primitive action mask")
                        action = requested if masks[requested] else 0
                        row = dict(frame=state.frame, player=asdict(state.player) if state.player else None,
                                   enemies=[asdict(e) for e in state.enemies],
                                   bullets=[asdict(b) for b in state.bullets],
                                   item=asdict(state.item) if state.item else None,
                                   kills=state.kills, lives=state.lives,
                                   remaining=state.remaining_enemies, spawn_left=state.spawn_left,
                                   fire_ready=state.fire_ready, shield=state.shield,
                                   action=action, requested=requested,
                                   controller_mode=getattr(bot, "mode", "baseline"))
                        if decisions % 64 == 0:
                            row["terrain"] = state.terrain.tolist()
                            row["brick_masks"] = state.brick_masks.tolist()
                        trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                        decisions += 1
                        ledger["decisions"] += 1
                        _, _, _, _, info = env.step(action)
                        ledger["frames"] += info["frames"] - previous_frames
                        previous_frames = info["frames"]
                        outcome = info["outcome"]
                        if decisions % 128 == 0:
                            _write(ledger_path, ledger)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                outcome = "aborted"
            terminal = env.state if not error else None
            row = dict(provenance, seed=seed, clear=outcome == "clear",
                       outcome=outcome, failure_cause=None if outcome == "clear" else error or outcome,
                       kills=info["kills"], deaths=info["deaths"], frames=info["frames"],
                       decisions=decisions, time_seconds=time.time() - started,
                       masked_requests=masked_requests,
                       cumulative_decisions=ledger["decisions"],
                       cumulative_episodes=ledger["episodes"],
                       cumulative_frames=ledger["frames"], error=error,
                       terminal_remaining_enemies=getattr(terminal, "remaining_enemies", None),
                       terminal_base_destroyed=getattr(terminal, "base_destroyed", None),
                       terminal_lives=getattr(terminal, "lives", None),
                       terminal_stage=getattr(terminal, "stage", None))
            rows.append(row)
            _write(ledger_path, ledger)
            _write(run / "report.json", rows)
            with (EVIDENCE / "episodes.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if error:
                raise RuntimeError(error)
    finally:
        env.close()
        _write(ledger_path, ledger)
    monitor.check("complete", force=True)


if __name__ == "__main__":
    main()
