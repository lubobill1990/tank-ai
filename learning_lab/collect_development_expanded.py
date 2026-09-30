"""Prospective protocol amendment: exposed openings become training/resubstitution.

NON-LEARNED TeacherControllerV2 labels current observations before executed
actions. No official final phases/maps are opened; no old run is relabelled.
No GPU, neural training, RAM edits, or shared-ledger writes.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from battle_city_env import BattleCityEnv
from learning_lab.teacher_v2 import TeacherControllerV2
from learning_lab.demonstrations import DemonstrationWriter, _action
from learning_lab.collect_teacher_v2 import (
    _write, _sha, _seal, _verify_dataset, verify_integrity, copy_success_episodes,
)
from rust_resources import ResourceMonitor

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "runs" / "20260930-victory-lab" / "development-expansion.json"
FULL = ROOT / "runs" / "20260930-victory-development-expanded"
SUCCESS = ROOT / "runs" / "20260930-victory-development-expanded-success"
LOCAL_DEADLINE = datetime.fromisoformat("2026-09-30T19:18:20+08:00").timestamp()
RESERVATION = "3350c109d5c24c3aba1b8e809c244a40"
WARNING = (
    "PROTOCOL AMENDMENT: stages1/3 and already-exposed opening phases11/20/22/23 "
    "(seeds0/1/7/39) are now TRAINING/RESUBSTITUTION data, NOT generalization. "
    "This does not retroactively change earlier runs. Official20 other phases and "
    "maps8/18/28 remain sealed. Final models must be selected before ONE sealed "
    "measurement; no retraining after unsealing."
)


def episode_plan(amendment):
    if amendment.get("version") != "development-training-expansion-v1" \
            or amendment.get("stages") != [1, 3] \
            or amendment.get("opening_seeds") != [0, 1, 7, 39] \
            or amendment.get("opening_phases") != [20, 11, 22, 23] \
            or amendment.get("maximum_episodes") != 24 \
            or amendment.get("maximum_environment_decisions") != 36000:
        raise ValueError("Explicit expansion authorization does not match this collector")
    rows = []
    for repetition in range(3):
        for stage in (1, 3):
            for seed, phase in zip(amendment["opening_seeds"], amendment["opening_phases"]):
                if int(np.random.default_rng(seed).integers(0, 24)) != phase:
                    raise ValueError("Approved reset seed/phase mapping differs")
                episode = len(rows)
                rows.append(dict(episode_id=episode, stage=stage, seed=seed, opening_phase=phase,
                                 repetition=repetition, perturbation_probability=0. if repetition == 0 else .01,
                                 perturbation_rng_seed=911 + episode))
    return rows


def _sources():
    return {name: _sha(ROOT / name) for name in (
        "battle_city.nes", "battle_city_env.py", "game_state.py", "heuristic.py", "native_codec.py",
        "learning_lab\\teacher.py", "learning_lab\\teacher_v2.py",
        "learning_lab\\demonstrations.py", "learning_lab\\collect_teacher_v2.py",
        "learning_lab\\collect_development_expanded.py",
        "runs\\20260930-victory-lab\\development-expansion.json")}


def collect_expansion(output=FULL, success_output=SUCCESS):
    output, success_output = Path(output).resolve(), Path(success_output).resolve()
    if output.exists() or success_output.exists():
        raise FileExistsError("Both expansion outputs must be fresh")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    plan = episode_plan(amendment)
    deadline = min(LOCAL_DEADLINE, datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp())
    collection_epoch = deadline - 60
    collection_deadline = time.monotonic() + max(0., collection_epoch - time.time())
    started = time.monotonic()
    writer = DemonstrationWriter(output)
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    sources = _sources()
    metadata = dict(
        label="NON-LEARNED TeacherControllerV2 amended final-training corpus",
        learned_model=False, development_only=True, training_resubstitution=True, warning=WARNING,
        reservation=RESERVATION, protocol_amendment=amendment,
        protocol_amendment_sha256=_sha(AMENDMENT), source_sha256=sources,
        stages=[1, 3], opening_seeds=[0, 1, 7, 39], opening_phases=[20, 11, 22, 23],
        max_episodes=24, max_decisions=36000, frame_skip=4, max_frames_per_episode=12000,
        label_timing="Teacher label BEFORE executed step; both actions and legal masks stored",
        attempt_definition="Increment before teacher.act; completed only after env.step returns",
        perturbation="One clean and two1%-uniform-legal-action-perturbed episodes per combination",
        perturbation_rng="independent numpy.default_rng(911+episode_id)",
        ram_edits=False, official_final_data_read=False, native_environment_decisions=0,
        cpu_only=True, deadline_epoch=deadline, episode_plan=plan,
    )
    _write(output / "protocol-amendment.json", amendment)
    _write(output / "collection_config.json", metadata)
    reports, attempted, completed, frames = [], 0, 0, 0
    env, active_stage, failure, stop_reason = None, None, None, "episode_cap"
    try:
        monitor.check("admission", admission_slots=1, force=True)
        for config in plan:
            if attempted >= 36000:
                stop_reason = "decision_cap"
                break
            if time.monotonic() >= collection_deadline:
                stop_reason = "deadline"
                break
            if active_stage != config["stage"]:
                if env is not None:
                    env.close()
                env = BattleCityEnv(stage=config["stage"], objective="full", max_frames=12000, frame_skip=4)
                active_stage = config["stage"]
            teacher = TeacherControllerV2()
            teacher.reset()
            rng = np.random.default_rng(config["perturbation_rng_seed"])
            ep_attempts = ep_completed = stored = changed = perturbations = 0
            episode_started = time.monotonic()
            info = dict(kills=0, deaths=0, frames=0, outcome="")
            finished, episode_error, initial_hash = False, None, None
            terminal_remaining, terminal_base = None, None
            previous_frame = 0
            try:
                observation, info = env.reset(seed=config["seed"])
                initial_hash = hashlib.sha256(env.nes.ram.copy().tobytes()).hexdigest()
                while not finished:
                    monitor.check("rom_collection")
                    if attempted >= 36000 or time.monotonic() >= collection_deadline:
                        stop_reason = "decision_cap" if attempted >= 36000 else "deadline"
                        info["outcome"] = "decision_budget" if attempted >= 36000 else "deadline"
                        break
                    masks = env.action_masks()[:10].copy()
                    ep_attempts += 1
                    attempted += 1
                    teacher_action = _action(teacher.act(env.state, masks))
                    if not masks[teacher_action]:
                        raise ValueError("Teacher label violates mask")
                    executed_action = teacher_action
                    if config["perturbation_probability"] and rng.random() < config["perturbation_probability"]:
                        perturbations += 1
                        executed_action = int(rng.choice(np.flatnonzero(masks)))
                    changed += int(executed_action != teacher_action)
                    writer.add(observation, teacher_action, masks, episode_start=stored == 0,
                               executed_action=executed_action)
                    stored += 1
                    observation, _, terminated, truncated, info = env.step(executed_action)
                    ep_completed += 1
                    completed += 1
                    frames += info["frames"] - previous_frame
                    previous_frame = info["frames"]
                    finished = bool(terminated or truncated)
                    if attempted % 128 == 0:
                        _write(output / "progress.json", dict(
                            attempted_decisions=attempted, completed_decisions=completed,
                            episode_id=config["episode_id"], stored_samples=len(writer.actions)))
                terminal_remaining = int(env.state.remaining_enemies)
                terminal_base = bool(env.state.base_destroyed)
            except Exception as error:
                episode_error = f"{type(error).__name__}: {error}"
                info["outcome"] = "aborted"
                failure, stop_reason = episode_error, "error"
            row = dict(
                config, outcome=info["outcome"], clear=finished and info["outcome"] == "clear",
                episode_finished=finished, attempted_decisions=ep_attempts,
                completed_decisions=ep_completed, decisions=stored, kills=int(info["kills"]),
                deaths=int(info["deaths"]), frames=int(info["frames"]),
                perturbation_attempts=perturbations, changed_actions=changed, error=episode_error,
                failure_cause=episode_error or (None if info["outcome"] == "clear" else info["outcome"]),
                terminal_remaining_enemies=terminal_remaining, terminal_base_destroyed=terminal_base,
                initial_ram_sha256=initial_hash, elapsed_seconds=time.monotonic() - episode_started,
                cumulative_attempted_decisions=attempted, cumulative_completed_decisions=completed,
            )
            reports.append(row)
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if episode_error or not finished:
                break
    except Exception as error:
        failure, stop_reason = f"{type(error).__name__}: {error}", "error"
    finally:
        if env is not None:
            env.close()
        if _sources() != sources:
            failure, stop_reason = "Frozen sources/amendment changed during collection", "error"
        summary = dict(
            status="failed" if failure else "completed" if len(reports) == 24 and stop_reason == "episode_cap" else "partial",
            stop_reason=stop_reason, error=failure, attempted_decisions=attempted,
            completed_decisions=completed, stored_samples=len(writer.actions),
            reported_episodes=len(reports), clear_episodes=sum(r["clear"] for r in reports),
            outcomes=dict(Counter(r["outcome"] for r in reports)), gameplay_frames=frames,
            changed_actions=sum(r["changed_actions"] for r in reports),
            elapsed_seconds=time.monotonic() - started, rom_environment_decisions=attempted,
            native_environment_decisions=0, warning=WARNING, training_resubstitution=True,
        )
        metadata.update(summary)
        writer.finish(reports, metadata)
        _write(output / "summary.json", summary)
        _write(output / "progress.json", dict(summary, finished=True))
    _verify_dataset(output, deadline, monitor)
    monitor.check("full_complete", force=True)
    full_fingerprint = _seal(output, deadline)
    full = dict(summary, path=str(output), fingerprint_sha256=full_fingerprint)
    success = copy_success_episodes(output, success_output, deadline=deadline)
    # The copy helper preserves whole episodes and verifies histories/actions.
    # Add the prospective amendment explicitly without changing any source run.
    copied_metadata = json.loads((success_output / "metadata.json").read_text())
    copied_metadata.update(warning=WARNING, training_resubstitution=True, protocol_amendment=amendment,
                           protocol_amendment_sha256=_sha(AMENDMENT))
    _write(success_output / "metadata.json", copied_metadata)
    _write(success_output / "protocol-amendment.json", amendment)
    success.update(warning=WARNING, training_resubstitution=True)
    success.pop("fingerprint_sha256", None)
    _write(success_output / "summary.json", success)
    success["fingerprint_sha256"] = _seal(success_output, deadline)
    assert verify_integrity(output, deadline) == full["fingerprint_sha256"]
    assert verify_integrity(success_output, deadline) == success["fingerprint_sha256"]
    result = dict(reservation=RESERVATION, shared_ledger_written=False, full=full, success=success)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    collect_expansion()
