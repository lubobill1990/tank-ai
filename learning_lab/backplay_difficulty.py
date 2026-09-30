"""Matched native difficulty probes; NON-LEARNED controls, no RAM correction.

Reuses exact RAM indices and native seeds from the first remaining1/2
evaluations. Teacher waypoints reset at import; this is not a continuation of
the original expert's hidden controller state. Failures do not prove a start
unsolvable. ROM decisions, neural training and GPU work are all zero.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import game_state as gs
from heuristic import _impact_time, incoming_bullets, shot_obstacle
from learning_lab.backplay_starts import load_backplay_pool
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_env import NativeBatch, BACKEND_VERSION, RNG_VERSION
from rust_resources import ResourceMonitor, ResourcePressureError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "20260930-victory-lab" / "backplay-v1" / "report.json"
OUTPUT = ROOT / "runs" / "20260930-victory-backplay-difficulty"
DEADLINE_EPOCH = datetime.fromisoformat("2026-09-30T17:08:31+08:00").timestamp()
METHODS = ("teacher_v2", "fire", "up_fire")
MAX_DECISIONS = 30000


def _write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_action(method, masks):
    target = {"fire": 5, "up_fire": 6}[method]
    if masks[target]:
        return target
    return 0 if masks[0] else int(np.flatnonzero(masks)[0])


def initial_hazards(state):
    player = state.player
    assert player is not None
    hostile = [b for b in state.bullets if b.owner >= 2]
    impacts = [_impact_time(state, b, player.x, player.y) for b in hostile]
    first = min(impacts, default=math.inf)
    return dict(
        lives=state.lives, last_life=state.lives <= 1, player=asdict(player),
        active_enemies=[asdict(e) for e in state.enemies],
        hostile_bullets=[asdict(b) for b in hostile],
        player_bullets=[asdict(b) for b in state.bullets if b.owner == 0],
        shield=state.shield, freeze=state.freeze, fire_ready=state.fire_ready,
        nearest_hostile_bullet_euclidean=min(
            (math.hypot(b.x - player.x, b.y - player.y) for b in hostile), default=None),
        nearest_hostile_bullet_manhattan=min(
            (abs(b.x - player.x) + abs(b.y - player.y) for b in hostile), default=None),
        incoming_bullets_within_24_frames=[b.slot for b in incoming_bullets(state)],
        stationary_impact_frames=first if math.isfinite(first) else None,
        nearest_enemy_distance=min(
            math.hypot(e.x - player.x, e.y - player.y) for e in state.enemies),
        current_heading_first_shot_hit=shot_obstacle(
            state, player.x, player.y, player.facing, player.power),
        hazard_note="Impact estimates use only currently visible bullets and stationary-player geometry; no future firing/RNG oracle.",
    )


def validate_bridge(ram, observation, frame, kills, max_frames):
    """Use native episode counters, not the snapshot's historical ROM kill HUD."""
    state = gs.parse(ram, frame=int(frame), kills=int(kills))
    if state.stage != 1:
        raise ValueError("Only stage1 difficulty starts are authorized")
    expected_player, expected_enemies = np.zeros((26, 26)), np.zeros((26, 26))
    if state.player:
        for cell in gs.tank_cells(state.player.x, state.player.y):
            expected_player[cell] = 1
    for enemy in state.enemies:
        for cell in gs.tank_cells(enemy.x, enemy.y):
            expected_enemies[cell] += 1 / 6
    if not np.array_equal(observation[5], expected_player) \
            or not np.allclose(observation[8], np.clip(expected_enemies, 0, 1), atol=1e-7, rtol=0) \
            or not np.allclose(observation[18], state.remaining_enemies / 20, atol=1e-7, rtol=0) \
            or not np.allclose(observation[20], min(1., state.lives / 10), atol=1e-7, rtol=0) \
            or not np.all(observation[24] == float(state.fire_ready)) \
            or not np.allclose(observation[36], max(0., 1 - frame / max_frames), atol=1e-7, rtol=0):
        raise ValueError("Canonical GameState/native observation/counter bridge mismatch")
    return state


def load_cases(source_path):
    source = json.loads(Path(source_path).read_text())
    identity = source["provenance"]["pool"]
    pool_path = Path(identity["path"])
    for name, digest in identity["files"].items():
        if _sha(pool_path / name) != digest:
            raise ValueError(f"Pool file differs from original backplay evaluation: {name}")
    pool = load_backplay_pool(pool_path)
    cases = []
    for remaining, cap, distinct in ((1, 2500, 2), (2, 3000, 3)):
        level = next(level for level in source["levels"] if level["name"] == f"remaining_{remaining}")
        if level["max_frames"] != cap:
            raise ValueError("Source evaluation cap differs from the authorized probe")
        evaluation = level["evaluations"][0]
        if evaluation["status"] != "completed":
            raise ValueError("Expected completed first source evaluations")
        results = {e["env_index"]: e for e in evaluation["episodes"]}
        selected = evaluation["start_indices"][:6]
        if len(selected) != 6 or len(set(selected)) != distinct:
            raise ValueError("First six cases do not represent the expected snapshot types")
        for index, (start, seed) in enumerate(zip(selected, evaluation["native_episode_seeds"][:6])):
            if pool["remaining"][start] != remaining or pool["full_start"][start] \
                    or pool["stages"][start] != 1 or pool["seeds"][start] not in (0, 1, 2):
                raise ValueError("Source evaluation selects an unauthorized start")
            if results[index]["seed"] != seed:
                raise ValueError("Source episode seed identity mismatch")
            cases.append(dict(
                case_id=f"remaining{remaining}-case{index}", remaining=remaining,
                case_index=index, pool_index=int(start), candidate_id=int(pool["candidate_ids"][start]),
                source_frame=int(pool["source_frames"][start]), rom_opening_seed=int(pool["seeds"][start]),
                native_seed=int(seed), max_frames=cap,
                ram_sha256=hashlib.sha256(pool["rams"][start].tobytes()).hexdigest(),
                source_learned_result=results[index],
                source_first_eval_clears=evaluation["clears"], source_first_eval_episodes=16,
            ))
    if sum(math.ceil(c["max_frames"] / 4) * len(METHODS) for c in cases) > MAX_DECISIONS:
        raise ValueError("Probe plan exceeds the hard interaction cap")
    return pool, cases


def _outcome(ram):
    if int(ram[0x68]) != 0x80:
        return "base_lost"
    if int(ram[0x51]) == 0 or (int(ram[0x51]) <= 1 and 0x10 <= int(ram[0xa0]) < 0x80):
        return "lives_exhausted"
    return "clear" if int(ram[0x80]) == 0 else "timeout"


def run_probe(source_path=SOURCE, output=OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    started = time.monotonic()
    deadline = started + max(0., DEADLINE_EPOCH - time.time())
    report = dict(
        reservation="2e33b50f04a846d39ace7b72d7de97bc", status="running", error=None,
        label="NON-LEARNED matched native difficulty bounds; not training, parity, ROM wins, or proof of impossibility",
        source_report=str(Path(source_path).resolve()), source_report_sha256=_sha(source_path),
        backend=BACKEND_VERSION, native_rng=RNG_VERSION, max_decisions=MAX_DECISIONS,
        deadline_monotonic=deadline, rom_decisions=0, native_decisions=0,
        completed_native_decisions=0, gpu=False, neural_training=False,
        ram_edits=False, per_step_ram_corrections=False, oracle_rng=False,
        fixed_action_fallback="requested primitive if legal, else NOOP, else first legal",
        teacher_initialization="TeacherControllerV2.reset(), no inherited historical navigation waypoints",
        source_sha256={name: _sha(ROOT / name) for name in (
            "learning_lab\\backplay_difficulty.py", "learning_lab\\teacher_v2.py",
            "learning_lab\\teacher.py", "game_state.py", "heuristic.py", "rust_env.py",
            "native\\target\\release\\battle_mirror_probe.dll")},
        episodes=[],
    )
    try:
        monitor.check("admission", admission_slots=1, force=True)
        pool, cases = load_cases(source_path)
        report["cases"] = cases
        report["planned_upper_decisions"] = sum(
            math.ceil(c["max_frames"] / 4) * len(METHODS) for c in cases)
        for case in cases:
            for method in METHODS:
                if time.monotonic() >= deadline or report["native_decisions"] >= MAX_DECISIONS:
                    raise TimeoutError("Difficulty probe deadline/decision cap")
                batch, error, done = None, None, False
                row = dict(case, method=method, decisions=0, completed_decisions=0, frames=0,
                           kills=0, deaths=0, shots=0, pickups=0, outcome="running",
                           first_death_frame=None, first_kill_frame=None, clear=False)
                mode_counts, actions = Counter(), Counter()
                start = time.monotonic()
                teacher = TeacherControllerV2()
                teacher.reset()
                try:
                    initial = pool["rams"][case["pool_index"]].copy()
                    batch = NativeBatch(initial[None], case["max_frames"], seeds=[case["native_seed"]])
                    observation = batch.reset(0, seed=case["native_seed"]).copy()
                    current_ram = batch.ram()
                    if not np.array_equal(initial, current_ram):
                        raise ValueError("Native import/reset changed initial RAM")
                    state = validate_bridge(current_ram, observation, 0, 0, case["max_frames"])
                    if state.player is None or not state.enemies or state.base_destroyed \
                            or not state.fire_ready or any(b.owner == 0 for b in state.bullets):
                        raise ValueError("Imported snapshot violates the expert-pool eligibility contract")
                    row.update(initial_ram_byte_exact=True, counter_bridge_verified=True,
                               initial_time_left=float(observation[36, 0, 0]),
                               initial_hazards=initial_hazards(state),
                               native_support_flags_at_start=batch.support.tolist())
                    with (output / f"{case['case_id']}-{method}.jsonl").open("x", encoding="utf-8") as trace:
                        while not done:
                            monitor.check("native_probe")
                            if time.monotonic() >= deadline or report["native_decisions"] >= MAX_DECISIONS:
                                raise TimeoutError("Difficulty probe deadline/decision cap")
                            masks = batch.masks()[0].copy()
                            action = teacher.act(state, masks) if method == "teacher_v2" else fixed_action(method, masks)
                            if not 0 <= action < 10 or not masks[action]:
                                raise ValueError(f"Controller returned illegal primitive {action}")
                            mode = teacher.mode if method == "teacher_v2" else method
                            mode_counts[mode] += 1
                            actions[int(action)] += 1
                            trace.write(json.dumps(dict(
                                frame=state.frame, lives=state.lives, remaining=state.remaining_enemies,
                                player=asdict(state.player) if state.player else None,
                                enemies=[asdict(e) for e in state.enemies],
                                bullets=[asdict(b) for b in state.bullets],
                                fire_ready=state.fire_ready, shield=state.shield,
                                action=int(action), mode=mode,
                            ), separators=(",", ":")) + "\n")
                            row["decisions"] += 1
                            report["native_decisions"] += 1
                            observations, _, dones, stats = batch.step(np.array([action], np.int64))
                            row["completed_decisions"] += 1
                            report["completed_native_decisions"] += 1
                            done = bool(dones[0])
                            previous_kills, previous_deaths = row["kills"], row["deaths"]
                            for name, value in zip(("frames", "kills", "deaths", "shots", "pickups"), stats[0]):
                                row[name] = int(value)
                            if row["kills"] > previous_kills and row["first_kill_frame"] is None:
                                row["first_kill_frame"] = row["frames"]
                            if row["deaths"] > previous_deaths and row["first_death_frame"] is None:
                                row["first_death_frame"] = row["frames"]
                            current_ram = batch.ram()
                            state = validate_bridge(
                                current_ram, observations[0], row["frames"], row["kills"], case["max_frames"])
                            if row["frames"] > case["max_frames"]:
                                raise ValueError("Native horizon exceeded the case cap")
                            if row["decisions"] % 128 == 0:
                                _write(output / "progress.json", dict(
                                    native_decisions=report["native_decisions"],
                                    completed_native_decisions=report["completed_native_decisions"],
                                    case_id=case["case_id"], method=method))
                    row.update(outcome=_outcome(current_ram), terminal_remaining=state.remaining_enemies,
                               terminal_base_destroyed=state.base_destroyed, terminal_lives=state.lives,
                               native_support_flags_at_end=batch.support.tolist())
                    row["clear"] = row["outcome"] == "clear"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    row.update(outcome="aborted", error=error)
                finally:
                    if batch is not None:
                        batch.close()
                    row.update(time_seconds=time.monotonic() - start,
                               action_counts=dict(actions), modes=dict(mode_counts),
                               failure_cause=error or (None if row["clear"] else row["outcome"]))
                    report["episodes"].append(row)
                    with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    _write(output / "report.json", report)
                    print(json.dumps({key: row[key] for key in (
                        "case_id", "pool_index", "method", "outcome", "kills", "deaths",
                        "frames", "decisions")}), flush=True)
                if error:
                    raise RuntimeError(error)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        summaries = {}
        for remaining in (1, 2):
            for method in METHODS:
                rows = [r for r in report["episodes"] if r["remaining"] == remaining and r["method"] == method]
                if rows:
                    summaries[f"remaining{remaining}/{method}"] = dict(
                        episodes=len(rows), clears=sum(r["clear"] for r in rows),
                        outcomes=dict(Counter(r["outcome"] for r in rows)),
                        decisions=sum(r["decisions"] for r in rows), frames=sum(r["frames"] for r in rows))
        report.update(summary=summaries, elapsed_seconds=time.monotonic() - started,
                      native_frames=sum(r["frames"] for r in report["episodes"]))
        _write(output / "report.json", report)
        _write(output / "progress.json", dict(status=report["status"],
              native_decisions=report["native_decisions"],
              completed_native_decisions=report["completed_native_decisions"], rom_decisions=0))
    monitor.check("complete", force=True)
    print(json.dumps(dict(status=report["status"], native_decisions=report["native_decisions"],
                         native_frames=report["native_frames"], summary=report["summary"])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run_probe(args.source_report, args.output)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
