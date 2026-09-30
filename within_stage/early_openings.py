"""Trace initialization, then test a fixed earlier legal stage1 handoff.

No root/old evidence is changed. Canonical pristine geometry comes from an
actually observed intact stage1 snapshot in the first probe, not fabricated RAM.
Natural same-material starts with changed quarters are a SEPARATE family.
No split is frozen, no teacher/model/GPU is used, and no RAM is corrected.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

import game_state as gs
from heuristic import _cell_shot_time
from within_stage.openings import (
    OpeningSpec, NaturalOpeningEnv, ProbeAccounting, ProbeStop, InvalidOpening,
    semantic_payload, semantic_hash, _hash, _write, ROOT, NEW_ROOT,
)
from rust_resources import ResourceMonitor

LOCAL_DEADLINE = datetime.fromisoformat("2026-09-30T21:32:37+08:00").timestamp()
REFERENCE = NEW_ROOT / "opening-probe" / "menu-08-repeat0.npz"


@dataclass(frozen=True)
class EarlyOpeningSpec(OpeningSpec):
    post_start_frames: int = 300

    def __post_init__(self):
        super().__post_init__()
        if self.legacy_phase is not None or type(self.post_start_frames) is not int \
                or not 0 <= self.post_start_frames <= 300:
            raise ValueError("Early handoff is0..300 fixed post-START frames, never legacy jitter")


def geometry_reference(state):
    if state.player is None or ((state.brick_masks != 0) & (state.brick_masks != 15)).any():
        raise ValueError("Reference needs a live ordinary player and intact brick quarters")
    return dict(terrain_sha256=_hash(state.terrain.tobytes()),
                brick_masks_sha256=_hash(state.brick_masks.tobytes()),
                player=dict(x=state.player.x, y=state.player.y, facing=state.player.facing, power=0))


def ready_state(state, reference):
    return ordinary_state(state, reference) and _hash(state.brick_masks.tobytes()) == reference["brick_masks_sha256"]


def ordinary_state(state, reference):
    if state is None or state.player is None:
        return False
    player = dict(x=state.player.x, y=state.player.y, facing=state.player.facing, power=state.player.power)
    return bool(state.stage == 1 and state.lives == 3 and state.remaining_enemies == 20
                and state.kills == 0 and state.score == 0 and state.item is None
                and not state.base_destroyed and not state.game_over
                and player == reference["player"]
                and _hash(state.terrain.tobytes()) == reference["terrain_sha256"])


def choose_handoff(traces):
    ready = [{row["post_start_frame"] for row in trace if row["ready"]} for trace in traces]
    shared = set.intersection(*ready) if ready else set()
    if not shared:
        raise ValueError("No shared live-player/pristine-map control window")
    return min(shared)


def menu_specs(handoff):
    return [EarlyOpeningSpec(title_wait=t, stage_wait=s, post_start_frames=handoff)
            for t in (0, 1, 7, 31, 63, 127) for s in (0, 1, 7, 31)]


def plan_upper_bound():
    core = 181 + 10 * 31 + 1 + 300
    discovery = sum(2 * (core + title + 33) for title in (0, 1, 31))
    comparisons = sum(2 * (core + spec.title_wait + spec.stage_wait + 33)
                      for spec in menu_specs(300))
    return discovery + comparisons


def parse_if_known(ram, frame):
    values = {gs.terrain_value(v) for v in np.unique(ram[0x400:0x800].reshape(32, 32)[2:28, 2:28])}
    known = set(range(16)) | {16, 18, 32, 33, 34} | set(range(200, 208))
    return gs.parse(ram, frame=frame) if values <= known else None


def describe_changes(previous, current, frame, geometry_was_complete):
    if previous is None or current is None:
        return None
    cells = np.argwhere((previous.terrain != current.terrain)
                        | (previous.brick_masks != current.brick_masks))
    if not len(cells):
        return None
    contacts = []
    for row, col in cells:
        for bullet in previous.bullets:
            contact = _cell_shot_time(previous, int(row), int(col), bullet.x, bullet.y, bullet.vx, bullet.vy)
            if 0 <= contact <= 1:
                contacts.append(dict(cell=[int(row), int(col)], bullet_slot=bullet.slot,
                                     owner=bullet.owner, x=bullet.x, y=bullet.y,
                                     vx=bullet.vx, vy=bullet.vy, geometric_contact_in_frames=contact))
    old_slots = {t.slot for t in previous.enemies}
    activated = [t.slot for t in current.enemies if t.slot not in old_slots]
    player_activated = previous.player is None and current.player is not None
    return dict(
        post_start_frame=frame,
        phase="after_complete_geometry" if geometry_was_complete else "map_initialization",
        cells=[dict(cell=[int(r), int(c)], before_material=int(previous.terrain[r, c]),
                    after_material=int(current.terrain[r, c]),
                    before_quarters=int(previous.brick_masks[r, c]),
                    after_quarters=int(current.brick_masks[r, c])) for r, c in cells],
        preceding_frame_projectile_contacts=contacts,
        player_activated_this_frame=player_activated, enemy_slots_activated_this_frame=activated,
        interpretation=("observed_projectile_contact_coincident_with_map_change" if contacts else
                        "coincident_activation_or_initialization_not_assigned_to_fire"
                        if player_activated or activated or not geometry_was_complete else
                        "map_change_without_recorded_projectile_contact"),
    )


class EarlyOpeningEnv(NaturalOpeningEnv):
    def __init__(self, opening_spec, *, account, reference, trace_initialization=False, max_frames=12000):
        if not isinstance(opening_spec, EarlyOpeningSpec):
            raise ValueError("opening_spec must be EarlyOpeningSpec")
        self.reference, self.trace_initialization = reference, trace_initialization
        self.initialization_rows, self.initialization_rams, self.geometry_events = [], [], []
        super().__init__(opening_spec, account=account, max_frames=max_frames)

    def _boot(self):
        self.initialization_rows, self.initialization_rams, self.geometry_events = [], [], []
        self._wait(181)
        self._checkpoint("title_ready")
        self._wait(self.spec.title_wait, phase="title_wait")
        for _ in range(10):
            if int(self.nes.ram[0x85]) == 1:
                break
            self._raw_step(8)
            self._wait(30)
        if int(self.nes.ram[0x85]) != 1:
            raise InvalidOpening("Stage1 selection handshake failed")
        self._checkpoint("stage_ready")
        self._wait(self.spec.stage_wait, phase="stage_wait")
        self._checkpoint("stage_after_wait")
        self._raw_step(8)
        previous, geometry_seen = None, False
        for frame in range(self.spec.post_start_frames + 1):
            if frame:
                self._raw_step(0)
            if self.trace_initialization or frame == self.spec.post_start_frames:
                ram = self.nes.ram.copy()
                state = parse_if_known(ram, frame)
                pristine_geometry = bool(state is not None
                    and _hash(state.terrain.tobytes()) == self.reference["terrain_sha256"]
                    and _hash(state.brick_masks.tobytes()) == self.reference["brick_masks_sha256"])
                row = dict(
                    post_start_frame=frame, ram_sha256=_hash(ram.tobytes()),
                    pixels_sha256=_hash(self.nes.screen.copy().tobytes()),
                    ready=state is not None and ready_state(state, self.reference),
                    pristine_geometry=pristine_geometry, canonical_parse_supported=state is not None,
                    state=semantic_payload(state) if state is not None else None,
                    raw_player_status=int(ram[0xa0]), raw_remaining=int(ram[0x80]),
                    raw_lives=int(ram[0x51]), raw_base=int(ram[0x68]),
                    partial_brick_cells=int(((state.brick_masks != 0) & (state.brick_masks != 15)).sum())
                    if state is not None else None,
                )
                self.initialization_rows.append(row)
                self.initialization_rams.append(ram)
                event = describe_changes(previous, state, frame, geometry_seen)
                if event:
                    self.geometry_events.append(event)
                geometry_seen |= pristine_geometry
                previous = state
        self._checkpoint("fixed_early_handoff")
        self.nes._backup()


def _case(env, prefix, role, repeat):
    before = env.account.snapshot()
    row = dict(spec=asdict(env.spec), role=role, repeat=repeat, status="running", error=None)
    future = []
    try:
        env.reset(seed=0)
        initial = env.state
        row.update(
            initial_state=semantic_payload(initial), initial_ram_sha256=_hash(env.nes.ram.copy().tobytes()),
            initial_pixels_sha256=_hash(env.nes.screen.copy().tobytes()),
            initial_semantic_sha256=semantic_hash(initial),
            strict_pristine_ready=ready_state(initial, env.reference),
            ordinary_same_material_map=ordinary_state(initial, env.reference),
        )
        row["family"] = ("strict_pristine" if row["strict_pristine_ready"] else
                         "natural_same_map_changed_quarters" if row["ordinary_same_material_map"]
                         else "not_ready_or_invalid_initial_conditions")
        np.savez_compressed(prefix.with_suffix(".npz"), ram=env.nes.ram.copy(),
                            pixels=env.nes.screen.copy(), terrain=initial.terrain,
                            brick_masks=initial.brick_masks)
        future.append(semantic_payload(initial))
        if row["family"] != "not_ready_or_invalid_initial_conditions":
            ended = False
            for _ in range(32):
                _, _, terminated, truncated, info = env.step(0)
                future.append(semantic_payload(env.state))
                if terminated or truncated:
                    ended = True
                    row["future_terminal_outcome"] = info["outcome"]
                    break
            row["future_semantic_sha256"] = _hash(json.dumps(future, sort_keys=True,
                                                             separators=(",", ":")).encode())
            row["noop_decisions"] = len(future) - 1
            row["noop_semantics_advance"] = len({json.dumps(s, sort_keys=True) for s in future}) > 1
            row["up_control_moves_player"] = False
            if not ended and env.state.player is not None:
                p = env.state.player
                row["control_before"] = [p.x, p.y, p.facing]
                _, _, _, _, _ = env.step(1)
                q = env.state.player
                row["control_after"] = [q.x, q.y, q.facing] if q else None
                row["up_control_moves_player"] = bool(q and q.y < p.y)
            row["normal_gameplay_verified_without_extra_start"] = (
                row["noop_semantics_advance"] and row["up_control_moves_player"])
            row["status"] = "valid" if row["normal_gameplay_verified_without_extra_start"] else "progress_or_control_unverified"
        else:
            row["status"] = "invalid_handoff"
    except InvalidOpening as error:
        row.update(status="invalid_menu_wait", error=str(error))
    except Exception as error:
        row.update(status="aborted", error=f"{type(error).__name__}: {error}")
    finally:
        after = env.account.snapshot()
        row.update(checkpoints=env.checkpoints, wait_status=env.wait_status,
                   raw_initialization_attempted=after["raw_initialization_attempted"] - before["raw_initialization_attempted"],
                   raw_initialization_completed=after["raw_initialization_completed"] - before["raw_initialization_completed"],
                   controller_attempted=after["agent_attempted"] - before["agent_attempted"],
                   controller_completed=after["agent_completed"] - before["agent_completed"],
                   cumulative_accounting=after)
        _write(prefix.with_suffix(".json"), row)
        with prefix.with_name(prefix.name + "-future.jsonl").open("w", encoding="utf-8") as stream:
            for index, state in enumerate(future):
                stream.write(json.dumps(dict(noop_index=index, state=state), sort_keys=True) + "\n")
        if env.trace_initialization and env.initialization_rows:
            _write(prefix.with_name(prefix.name + "-initialization.json"),
                   dict(frames=env.initialization_rows, geometry_events=env.geometry_events))
            np.savez_compressed(prefix.with_name(prefix.name + "-initialization-rams.npz"),
                                rams=np.stack(env.initialization_rams))
    return row


def run_early_probe(output=NEW_ROOT / "early-opening-probe"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    authorization = json.loads((NEW_ROOT / "authorization.json").read_text())
    epoch = min(LOCAL_DEADLINE, datetime.fromisoformat(authorization["deadline"]).timestamp())
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    account = ProbeAccounting(50000, deadline=time.monotonic() + max(0., epoch - time.time()), monitor=monitor)
    with np.load(REFERENCE, allow_pickle=False) as data:
        reference_state = gs.parse(data["ram"])
    reference = geometry_reference(reference_state)
    plan = dict(
        discovery_menu_titles=[0, 1, 31], discovery_stage_wait=0, discovery_post_start_frames=300,
        repeats=2, comparison_titles=[0, 1, 7, 31, 63, 127], comparison_stage_waits=[0, 1, 7, 31],
        handoff_selection="Earliest frame in shared pristine/live-P1 readiness intersection of six discovery traces",
        future_noop_decisions=32, additional_up_control_decisions=1,
        unique_configuration_upper_bound=27, interaction_upper_bound=plan_upper_bound(),
        pristine_reference=str(REFERENCE), pristine_reference_file_sha256=_hash(REFERENCE.read_bytes()),
        reference=reference, deadline_epoch=epoch,
    )
    _write(output / "plan.json", plan)
    if plan_upper_bound() > 50000:
        raise ValueError("Plan exceeds reservation")
    report = dict(
        reservation="406f4dad380b432ca594aa1e5fb2615b", status="running", error=None,
        stage=1, ram_writes=False, gpu=False, learning=False, pool_frozen=False,
        sources={name: _hash((ROOT / name).read_bytes()) for name in
                 ("within_stage\\early_openings.py", "within_stage\\openings.py",
                  "battle_city_env.py", "game_state.py", "heuristic.py", "battle_city.nes")},
        initialization_trials=[], trials=[], geometry_events=[],
    )
    traces = []
    started = time.monotonic()
    try:
        monitor.check("admission", admission_slots=1, force=True)
        for title in (0, 1, 31):
            spec = EarlyOpeningSpec(title_wait=title, stage_wait=0, post_start_frames=300)
            env = EarlyOpeningEnv(spec, account=account, reference=reference, trace_initialization=True)
            try:
                for repeat in range(2):
                    row = _case(env, output / f"trace-title{title}-repeat{repeat}", "initialization_trace", repeat)
                    row["trace_index"] = len(traces)
                    report["initialization_trials"].append(row)
                    if row["status"] == "aborted":
                        raise RuntimeError(row["error"])
                    traces.append(env.initialization_rows.copy())
                    report["geometry_events"].append(dict(title_wait=title, repeat=repeat,
                                                          events=env.geometry_events.copy()))
                    _write(output / "progress.json", account.snapshot())
            finally:
                env.close()
        try:
            handoff = choose_handoff(traces)
        except ValueError as error:
            report.update(status="completed_negative", error=str(error))
            handoff = None
        report["selected_handoff"] = handoff
        report["readiness"] = [
            dict(trace_index=i, first_ready=min((r["post_start_frame"] for r in trace if r["ready"]), default=None),
                 ready_frames=[r["post_start_frame"] for r in trace if r["ready"]],
                 first_complete_geometry=min((r["post_start_frame"] for r in trace if r["pristine_geometry"]), default=None))
            for i, trace in enumerate(traces)]
        if handoff is not None:
            _write(output / "chosen-handoff.json", dict(handoff=handoff, readiness=report["readiness"]))
            for index, spec in enumerate(menu_specs(handoff)):
                env = EarlyOpeningEnv(spec, account=account, reference=reference)
                try:
                    for repeat in range(2):
                        row = _case(env, output / f"early-{index:02d}-repeat{repeat}", "early_fixed_handoff", repeat)
                        row["config_index"] = index
                        report["trials"].append(row)
                        _write(output / "progress.json", account.snapshot())
                        if row["status"] == "aborted":
                            raise RuntimeError(row["error"])
                finally:
                    env.close()
            report["status"] = "completed"
    except Exception as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        grouped = defaultdict(list)
        for row in report["trials"]:
            grouped[row["config_index"]].append(row)
        configs = []
        for index, rows in sorted(grouped.items()):
            repeated = {key: rows[0].get(key) == rows[1].get(key) for key in
                        ("initial_ram_sha256", "initial_pixels_sha256", "initial_semantic_sha256",
                         "future_semantic_sha256")} if len(rows) == 2 else {}
            valid = len(rows) == 2 and all(r["status"] == "valid" for r in rows)
            configs.append(dict(config_index=index, spec=rows[0]["spec"], valid_both=valid,
                                family=rows[0].get("family"), repeatability=repeated,
                                future_trace=rows[0].get("future_semantic_sha256"),
                                statuses=[r["status"] for r in rows]))
        report.update(accounting=account.snapshot(), configurations=configs,
                      elapsed_seconds=time.monotonic() - started, cpu_test_real_decisions=0,
                      shared_ledger_written=False)
        for family in ("strict_pristine", "natural_same_map_changed_quarters"):
            good = [c for c in configs if c["valid_both"] and c["family"] == family
                    and c["repeatability"] and all(c["repeatability"].values())]
            report[family] = dict(configs=len(good),
                                  distinct_semantic_future_traces=len({c["future_trace"] for c in good}))
        _write(output / "report.json", report)
        _write(output / "progress.json", account.snapshot())
    monitor.check("complete", force=True)
    print(json.dumps({key: report.get(key) for key in (
        "status", "error", "selected_handoff", "accounting", "strict_pristine",
        "natural_same_map_changed_quarters", "elapsed_seconds")}), flush=True)
    return report


if __name__ == "__main__":
    run_early_probe()
