"""Unsplit natural stage1 pool, generated solely by legal menu inputs.

Public load_pool/save_pool exchange specifications and provenance, NOT RAM
initializations. make_opening_env replays legal inputs in a cold ROM emulator.
Diagnostic RAM/screens/futures must never become learner-side oracle features.
No model/teacher performance affects eligibility or configuration generation.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

import game_state as gs
from within_stage.openings import ProbeStop, ProbeAccounting, InvalidOpening
from within_stage.openings import semantic_payload, semantic_hash, _hash, _write, ROOT, NEW_ROOT
from within_stage.early_openings import (
    EarlyOpeningSpec, EarlyOpeningEnv, ready_state, geometry_reference, REFERENCE,
)
from rust_resources import ResourceMonitor, ResourcePressureError

FORMAT = "natural-opening-pool-v1"
CALIBRATION = [(0, 0), (1, 1), (7, 7), (7, 31), (31, 7), (31, 31), (63, 7), (63, 31)]
REPEATS = (0, 7, 15, 31, 47, 63, 79, 95)
LOCAL_DEADLINE = datetime.fromisoformat("2026-09-30T22:03:56+08:00").timestamp()


def menu_configurations():
    rows = list(CALIBRATION)
    index = 0
    while len(rows) < 104:
        candidate = ((0, 1, 7, 31)[index % 4], (53 * index) % 384)
        if candidate not in rows:
            rows.append(candidate)
        index += 1
    rows.extend(((0, 1, 7, 31)[i % 4], 512 + 73 * i) for i in range(8))
    return rows


class PoolAccounting(ProbeAccounting):
    @property
    def charged_bound(self):
        return self.raw_initialization_attempted + 5 * self.agent_attempted

    def before(self, kind):
        cost = 1 if kind == "raw" else 5 if kind == "agent" else 0
        if not cost:
            raise ValueError("Unknown interaction kind")
        if self.charged_bound + cost > self.limit:
            raise ProbeStop("Conservative all-frame-and-controller-call cap exhausted")
        super().before(kind)

    def snapshot(self):
        result = super().snapshot()
        result.update(
            charged_conservative_bound=self.charged_bound,
            completed_raw_emulator_calls=self.raw_initialization_completed + self.agent_raw_frames,
            completed_all_calls=self.raw_initialization_completed + self.agent_raw_frames + self.agent_completed,
            pool_cap_convention="Conservatively charge1 per boot/wait NES.step and5 per controller call (wrapper+up to4 raw gameplay frames). Exact completed raw frames are separately recorded.",
        )
        return result


def group_families(trials):
    groups = {}
    for row in trials:
        if row["status"] != "valid":
            continue
        groups.setdefault(row["future_semantic_sha256"], []).append(row["config_index"])
    return [dict(family_id=index, future_semantic_sha256=digest,
                 config_indices=sorted(set(indices)))
            for index, (digest, indices) in enumerate(sorted(groups.items()))]


def validate_pool(value):
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("stage") != 1 \
            or value.get("handoff") not in (134, 150) or value.get("frozen") is not False \
            or value.get("partition") is not None or not isinstance(value.get("specs"), list) \
            or not isinstance(value.get("families"), list) or not isinstance(value.get("reference"), dict):
        raise ValueError("Invalid unsplit natural opening-pool schema")
    ids = set()
    for item in value["specs"]:
        if type(item.get("config_index")) is not int or item["config_index"] in ids:
            raise ValueError("Opening configuration IDs must be unique integers")
        ids.add(item["config_index"])
        spec = EarlyOpeningSpec(**item["spec"])
        if spec.post_start_frames != value["handoff"]:
            raise ValueError("A pool must use one fixed handoff for every configuration")
        if any(name in item for name in ("ram", "rams", "initial_ram", "controller_actions")):
            raise ValueError("Runtime pool entries must be specs, not RAM/action oracles")
    membership = [i for family in value["families"] for i in family["config_indices"]]
    if set(membership) != ids or len(membership) != len(ids):
        raise ValueError("Every admitted configuration must belong to exactly one family")


def save_pool(path, value):
    path = Path(path)
    validate_pool(value)
    if (path / "pool.json").exists() or (path / "pool.integrity.json").exists():
        raise FileExistsError("Refusing to overwrite a pool")
    _write(path / "pool.json", value)
    _write(path / "pool.integrity.json",
           dict(format=FORMAT, sha256=_hash((path / "pool.json").read_bytes())))


def load_pool(path):
    path = Path(path)
    root = path if path.is_dir() else path.parent
    integrity = json.loads((root / "pool.integrity.json").read_text())
    if integrity.get("format") != FORMAT or _hash((root / "pool.json").read_bytes()) != integrity.get("sha256"):
        raise ValueError("Pool hash integrity mismatch")
    value = json.loads((root / "pool.json").read_text())
    validate_pool(value)
    for name, digest in value.get("sources", {}).items():
        if _hash((ROOT / name).read_bytes()) != digest:
            raise ValueError(f"Pool source changed: {name}")
    return value


class PoolOpeningEnv(EarlyOpeningEnv):
    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        if not ready_state(self.state, self.reference):
            raise ValueError("Opening no longer satisfies the pool's strict initial conditions")
        return observation, info


def make_opening_env(pool, config_index, *, account, max_frames=12000):
    """Reconstruct by inputs only; neither diagnostic RAM nor future traces are read."""
    validate_pool(pool)
    row = next((r for r in pool["specs"] if r["config_index"] == config_index), None)
    if row is None:
        raise ValueError("Configuration is not an admitted pool member")
    return PoolOpeningEnv(EarlyOpeningSpec(**row["spec"]), account=account,
                          reference=pool["reference"], max_frames=max_frames)


def _world_progress(states):
    projections = [json.dumps({key: state[key] for key in
                               ("enemies", "bullets", "spawn_left", "remaining", "lives", "base_destroyed")},
                              sort_keys=True) for state in states]
    return len(set(projections)) > 1 and any(s["enemies"] or s["bullets"] for s in states)


def _trial(spec, account, reference, prefix, *, config_index, repeat, role):
    before = account.snapshot()
    row = dict(config_index=config_index, repeat=repeat, role=role, spec=asdict(spec),
               status="running", error=None, initial_criteria_valid=False)
    env = None
    states, pixel_hashes = [], []
    try:
        env = EarlyOpeningEnv(spec, account=account, reference=reference)
        env.reset(seed=0)
        ram, screen = env.nes.ram.copy(), env.nes.screen.copy()
        row.update(
            initial_criteria_valid=ready_state(env.state, reference),
            initial_state=semantic_payload(env.state),
            initial_ram_sha256=_hash(ram.tobytes()),
            initial_pixels_sha256=_hash(screen.tobytes()),
            initial_semantic_sha256=semantic_hash(env.state),
        )
        np.savez_compressed(prefix.with_suffix(".npz"), ram=ram, pixels=screen)
        if row["initial_criteria_valid"]:
            states.append(semantic_payload(env.state))
            pixel_hashes.append(_hash(screen.tobytes()))
            for _ in range(128):
                _, _, terminated, truncated, info = env.step(0)
                states.append(semantic_payload(env.state))
                pixel_hashes.append(_hash(env.nes.screen.copy().tobytes()))
                if terminated or truncated:
                    row["natural_terminal_outcome"] = info["outcome"]
                    break
            row.update(
                noop_decisions=len(states) - 1,
                gameplay_progress_observed=_world_progress(states),
                future_semantic_sha256=_hash(json.dumps(states, sort_keys=True, separators=(",", ":")).encode()),
                future_pixels_sha256=_hash(json.dumps(pixel_hashes).encode()),
                terminal_state=states[-1],
            )
            row["status"] = "valid" if row["gameplay_progress_observed"] else "paused_or_progress_unverified"
        else:
            row["status"] = "invalid_initial_conditions"
    except ProbeStop as error:
        row.update(status="budget_or_deadline_stop", error=str(error))
    except InvalidOpening as error:
        row.update(status="invalid_menu_wait", error=str(error))
    except ResourcePressureError as error:
        row.update(status="resource_stop", error=str(error))
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        if env is not None:
            row.update(checkpoints=env.checkpoints, wait_status=env.wait_status)
            env.close()
        after = account.snapshot()
        row.update(
            raw_initialization_attempted=after["raw_initialization_attempted"] - before["raw_initialization_attempted"],
            raw_initialization_completed=after["raw_initialization_completed"] - before["raw_initialization_completed"],
            controller_attempted=after["agent_attempted"] - before["agent_attempted"],
            controller_completed=after["agent_completed"] - before["agent_completed"],
            raw_gameplay_frames=after["agent_raw_frames"] - before["agent_raw_frames"],
            conservative_calls=after["charged_conservative_bound"] - before["charged_conservative_bound"],
            cumulative_accounting=after,
        )
        _write(prefix.with_suffix(".json"), row)
        with prefix.with_name(prefix.name + "-future.jsonl").open("w", encoding="utf-8") as stream:
            for index, state in enumerate(states):
                stream.write(json.dumps(dict(noop_index=index, state=state), sort_keys=True) + "\n")
    return row


def _trial_bound(title, stage, handoff):
    return 181 + 10 * 31 + 1 + handoff + title + stage + 128 * 5


def run_pool_expansion(output=NEW_ROOT / "natural-opening-pool"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = menu_configurations()
    authorization = json.loads((NEW_ROOT / "authorization.json").read_text())
    epoch = min(LOCAL_DEADLINE, datetime.fromisoformat(authorization["deadline"]).timestamp())
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    account = PoolAccounting(160000, deadline=time.monotonic() + max(0., epoch - time.time()), monitor=monitor)
    with np.load(REFERENCE, allow_pickle=False) as data:
        reference = geometry_reference(gs.parse(data["ram"]))
    plan = dict(
        format="natural-opening-generation-plan-v1",
        calibration_indices=list(range(8)), calibration_pairs=CALIBRATION,
        handoff_rule="Choose150 globally only if all8 calibration cases preserve strict conditions and show progression. Otherwise use134 globally and reject failures; never switch per case.",
        menu_pairs=[dict(config_index=i, title_wait=t, stage_wait=s) for i, (t, s) in enumerate(configs)],
        generation_parameters="First8 readiness calibrations; deterministic(53*i mod384) stage sweep with four title waits, then eight stage waits512+73*i. No wins/model/teacher filtering.",
        repeat_indices=list(REPEATS), max_unique_specs_including_fallback=120,
        future_controller="NOOP only, up to128 decisions, stop on natural terminal",
        post_handoff_other_actions=False, proposed_handoff=150, fallback_handoff=134,
        raw_and_controller_conservative_cap=160000, deadline_epoch=epoch,
        reference=reference, reference_file_sha256=_hash(REFERENCE.read_bytes()),
        budget_admission="Before each trial require enough for worst original10-handshake boot plus waits plus128*(4 raw frames+1controller call). Stop rather than extend.",
    )
    _write(output / "plan.json", plan)
    report = dict(
        reservation="f5e98e2e504d409eaa14ceeb7fc3cbeb", status="running", error=None,
        fixed_handoff=None, calibration=[], trials=[], repeats=[], frozen=False, partition=None,
        target_families=60, stage=1, models_or_teachers=False, gpu=False, ram_writes=False,
        sources={name: _hash((ROOT / name).read_bytes()) for name in (
            "within_stage\\opening_pool.py", "within_stage\\early_openings.py",
            "within_stage\\openings.py", "battle_city_env.py", "game_state.py",
            "heuristic.py", "battle_city.nes")},
    )
    started = time.monotonic()
    selected_handoff = None
    def admit(index, handoff):
        account.check()
        title, stage = configs[index]
        if account.charged_bound + _trial_bound(title, stage, handoff) > account.limit:
            raise ProbeStop("Insufficient conservative allowance for a complete next trial")
    try:
        monitor.check("admission", admission_slots=1, force=True)
        for index in range(8):
            admit(index, 150)
            title, stage = configs[index]
            row = _trial(EarlyOpeningSpec(title_wait=title, stage_wait=stage, post_start_frames=150),
                         account, reference, output / f"calibration-{index:03d}", config_index=index,
                         repeat=0, role="handoff150_calibration")
            report["calibration"].append(row)
            _write(output / "progress.json", account.snapshot())
            if row["status"] in ("budget_or_deadline_stop", "resource_stop", "error"):
                raise ProbeStop(row["error"])
        selected_handoff = 150 if all(r["status"] == "valid" for r in report["calibration"]) else 134
        report["fixed_handoff"] = selected_handoff
        _write(output / "handoff-decision.json", dict(
            handoff=selected_handoff, calibration_statuses=[r["status"] for r in report["calibration"]],
            selection_uses_only_initial_conditions_and_gameplay_progress=True))
        if selected_handoff == 150:
            report["trials"].extend(report["calibration"])
        for index in range(8 if selected_handoff == 150 else 0, len(configs)):
            admit(index, selected_handoff)
            title, stage = configs[index]
            row = _trial(EarlyOpeningSpec(title_wait=title, stage_wait=stage,
                                         post_start_frames=selected_handoff),
                         account, reference, output / f"opening-{index:03d}", config_index=index,
                         repeat=0, role="primary_pool_probe")
            report["trials"].append(row)
            _write(output / "progress.json", account.snapshot())
            if row["status"] in ("budget_or_deadline_stop", "resource_stop", "error"):
                raise ProbeStop(row["error"])
        for index in REPEATS:
            admit(index, selected_handoff)
            title, stage = configs[index]
            row = _trial(EarlyOpeningSpec(title_wait=title, stage_wait=stage,
                                         post_start_frames=selected_handoff),
                         account, reference, output / f"repeat-{index:03d}", config_index=index,
                         repeat=1, role="predeclared_reproducibility_subset")
            report["repeats"].append(row)
            if row["status"] in ("budget_or_deadline_stop", "resource_stop", "error"):
                raise ProbeStop(row["error"])
        report["status"] = "completed"
    except ProbeStop as error:
        report.update(status="partial_at_cap_or_deadline", error=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        families = group_families(report["trials"])
        repeat_results = []
        originals = {r["config_index"]: r for r in report["trials"]}
        rejected_repeat = set()
        for row in report["repeats"]:
            original = originals[row["config_index"]]
            keys = ("initial_ram_sha256", "initial_pixels_sha256", "initial_semantic_sha256",
                    "future_semantic_sha256", "future_pixels_sha256")
            equality = {k: original.get(k) == row.get(k) for k in keys}
            matched = original["status"] == row["status"] and all(equality.values())
            repeat_results.append(dict(config_index=row["config_index"], matched=matched, equality=equality))
            if not matched:
                rejected_repeat.add(row["config_index"])
        admitted = [r for r in report["trials"] if r["status"] == "valid"
                    and r["config_index"] not in rejected_repeat]
        families = group_families(admitted)
        family_ids = {i: family["family_id"] for family in families for i in family["config_indices"]}
        pool = dict(
            format=FORMAT, stage=1, handoff=selected_handoff, frozen=False, partition=None,
            reference=reference, sources=report["sources"], generation_plan_sha256=_hash((output / "plan.json").read_bytes()),
            specs=[dict(config_index=r["config_index"], spec=r["spec"],
                        family_id=family_ids[r["config_index"]],
                        initial_ram_sha256=r["initial_ram_sha256"],
                        initial_pixels_sha256=r["initial_pixels_sha256"],
                        initial_semantic_sha256=r["initial_semantic_sha256"],
                        future_semantic_sha256=r["future_semantic_sha256"],
                        noop_decisions=r["noop_decisions"],
                        natural_terminal_outcome=r.get("natural_terminal_outcome"),
                        reproduction_checked=any(x["config_index"] == r["config_index"] for x in repeat_results))
                   for r in admitted],
            families=families, repeat_results=repeat_results,
            runtime_policy="Replay legal inputs only; no diagnostic RAM/future-trace access by learner.",
            independence_warning="Each equivalence family is one unit, regardless of how many timing specs alias it.128-NOOP distinction is not proof of full-game independence.",
        )
        if selected_handoff is not None:
            save_pool(output, pool)
        report.update(
            accounting=account.snapshot(), elapsed_seconds=time.monotonic() - started,
            primary_configs_completed=len(report["trials"]),
            admitted_specs=len(admitted), distinct_families=len(families),
            target_reached=len(families) >= 60, repeat_results=repeat_results,
            unattempted_indices=[i for i in range(len(configs)) if i not in originals],
            pending_repeat_indices=[i for i in REPEATS if not any(r["config_index"] == i for r in report["repeats"])],
            shared_ledger_written=False, cpu_test_real_decisions=0,
        )
        _write(output / "report.json", report)
        _write(output / "progress.json", account.snapshot())
    monitor.check("complete", force=True)
    print(json.dumps({key: report.get(key) for key in (
        "status", "error", "fixed_handoff", "primary_configs_completed", "admitted_specs",
        "distinct_families", "target_reached", "accounting", "elapsed_seconds")}), flush=True)
    return report


if __name__ == "__main__":
    run_pool_expansion()
