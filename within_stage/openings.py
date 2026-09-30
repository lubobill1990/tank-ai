"""Stage1 legal menu-wait openings; new tranche, no learning or RAM writes.

NaturalOpeningEnv(OpeningSpec(title_wait=..., stage_wait=...), account=...)
always uses a fresh emulator on reset. It follows the original181-frame title
boot, START/30-frame stage handshake, START and300-frame gameplay convention.
No random postboot delay is added. legacy_phase is explicitly reference-only;
the probe reuses the original unmodified emulator backup for those24 known phases.
Only configurations reported valid/repeatable should enter a future frozen split.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
from nes_py import NESEnv
import numpy as np

from battle_city_env import BattleCityEnv
import game_state as gs
from rust_resources import ResourceMonitor

ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = ROOT / "runs" / "20260930-within-stage"
LOCAL_DEADLINE = datetime.fromisoformat("2026-09-30T21:05:44+08:00").timestamp()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


class ProbeStop(RuntimeError):
    pass


class InvalidOpening(RuntimeError):
    pass


@dataclass(frozen=True)
class OpeningSpec:
    title_wait: int = 0
    stage_wait: int = 0
    legacy_phase: int | None = None

    def __post_init__(self):
        if any(type(v) is not int or not 0 <= v <= 4096 for v in (self.title_wait, self.stage_wait)):
            raise ValueError("Menu waits must be integers in0..4096")
        if self.legacy_phase is not None and (
            type(self.legacy_phase) is not int or not 0 <= self.legacy_phase < 24
            or self.title_wait != 0 or self.stage_wait != 0
        ):
            raise ValueError("Known legacy0..23 phases cannot be mixed with new menu waits")


class ProbeAccounting:
    def __init__(self, limit, *, deadline, monitor=None):
        if type(limit) is not int or not 1 <= limit <= 4000000:
            raise ValueError("Explicit positive interaction allowance required")
        self.limit, self.deadline, self.monitor = limit, deadline, monitor
        self.raw_initialization_attempted = self.raw_initialization_completed = 0
        self.agent_attempted = self.agent_completed = 0
        self.agent_raw_frames = self.emulator_resets = 0

    @property
    def attempted(self):
        return self.raw_initialization_attempted + self.agent_attempted

    @property
    def completed(self):
        return self.raw_initialization_completed + self.agent_completed

    def check(self):
        if time.monotonic() >= self.deadline:
            raise ProbeStop("Absolute NEW-tranche/local deadline reached")
        if self.monitor is not None:
            self.monitor.check("opening_probe")

    def before(self, kind):
        self.check()
        if self.attempted >= self.limit:
            raise ProbeStop("Opening-probe interaction allowance exhausted")
        if kind == "raw":
            self.raw_initialization_attempted += 1
        elif kind == "agent":
            self.agent_attempted += 1
        else:
            raise ValueError("Unknown interaction kind")

    def after(self, kind):
        if kind == "raw":
            self.raw_initialization_completed += 1
        elif kind == "agent":
            self.agent_completed += 1
        else:
            raise ValueError("Unknown interaction kind")

    def snapshot(self):
        return dict(
            attempted=self.attempted, completed=self.completed,
            raw_initialization_attempted=self.raw_initialization_attempted,
            raw_initialization_completed=self.raw_initialization_completed,
            agent_attempted=self.agent_attempted, agent_completed=self.agent_completed,
            agent_raw_frames=self.agent_raw_frames, emulator_resets=self.emulator_resets,
            convention="Every explicit NES boot/menu/postboot step counts1; each env.step counts1 separately. Gameplay raw frames are also reported, not double-charged. NES reset/restore performs no controller frame call.",
        )


class NaturalOpeningEnv(BattleCityEnv):
    def __init__(self, opening_spec, *, account, max_frames=12000, frame_skip=4,
                 reuse_legacy_backup=False):
        if not isinstance(opening_spec, OpeningSpec):
            raise ValueError("opening_spec must be OpeningSpec")
        if reuse_legacy_backup and opening_spec.legacy_phase is None:
            raise ValueError("Backup reuse is only for the already-exposed legacy reference pool")
        self.spec, self.account = opening_spec, account
        self._reuse_legacy_backup, self._ever_reset = reuse_legacy_backup, False
        self.checkpoints, self.wait_status = [], {}
        super().__init__(stage=1, objective="full", max_frames=max_frames,
                         frame_skip=frame_skip, reward_profile="sparse")
        self.action_space = gym.spaces.Discrete(10)

    @staticmethod
    def _gameplay_active(ram):
        active = lambda value: 0x80 <= int(value) < 0xe0
        return bool(int(ram[0x85]) == 1 and int(ram[0x68]) == 0x80
                    and int(ram[0x51]) > 0 and int(ram[0x80]) > 0
                    and active(ram[0xa0]) and any(active(v) for v in ram[0xa2:0xa8]))

    def _checkpoint(self, name):
        ram = self.nes.ram.copy()
        screen = self.nes.screen.copy()
        self.checkpoints.append(dict(
            name=name, ram_sha256=_hash(ram.tobytes()), pixels_sha256=_hash(screen.tobytes()),
            stage=int(ram[0x85]), lives=int(ram[0x51]), remaining=int(ram[0x80]),
            player_status=int(ram[0xa0]), active_enemy_statuses=[int(v) for v in ram[0xa2:0xa8]],
            gameplay_active=self._gameplay_active(ram)))

    def _raw_step(self, action):
        if action not in (0, 8):
            raise ValueError("This stage1 opening protocol only allows NOOP and START")
        self.account.before("raw")
        self.nes.step(action)
        self.account.after("raw")

    def _wait(self, frames, *, phase=None):
        if phase and frames and self._gameplay_active(self.nes.ram):
            self.wait_status[f"{phase}_started_active"] = True
            raise InvalidOpening(f"{phase}: active gameplay was already present; cannot certify a menu wait")
        for index in range(frames):
            self._raw_step(0)
            if phase and self._gameplay_active(self.nes.ram):
                self.wait_status[f"{phase}_advanced_game"] = True
                self.wait_status[f"{phase}_first_active_frame"] = index + 1
                self._checkpoint(f"{phase}_gameplay_detected")
                raise InvalidOpening(f"{phase}: gameplay began during extra menu wait")

    def _boot(self):
        self._wait(181)
        self._checkpoint("title_ready")
        self._wait(self.spec.title_wait, phase="title_wait")
        self._checkpoint("title_after_wait")
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
        self._wait(300)
        self._checkpoint("fixed_postgame_boot_complete")
        self.nes._backup()

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Specify opening timing in OpeningSpec, not reset options")
        gym.Env.reset(self, seed=seed)
        self.account.check()
        if self._ever_reset and not self._reuse_legacy_backup:
            self.nes.close()
            self.nes = NESEnv(str(ROOT / "battle_city.nes"))
        self._ever_reset = True
        self.checkpoints, self.wait_status = [], {}
        self.account.emulator_resets += 1
        self.nes.reset()
        if not self.nes._has_backup:
            self._boot()
        if self.spec.legacy_phase is not None:
            self._wait(self.spec.legacy_phase)
        self._tracker.reset(self.nes.ram)
        self._bot.reset()
        self._last_macro = -1
        self._deaths = self._shots = self._pickups = 0
        self._item_drops = 0
        self._item_opportunities = int(self.state.item is not None)
        self._combat_pickup_deadline = None
        self._last_progress_frame = 0
        self._boundary_waste_frames = self._boundary_waste_streak_frames = 0
        self._max_boundary_waste_frames = 0
        self._needs_reset = False
        return self._observation(), self._info("", {})

    def action_masks(self):
        return super().action_masks()[:10].copy()

    def step(self, action):
        self.account.before("agent")
        before = self.state.frame
        result = super().step(action)
        self.account.after("agent")
        self.account.agent_raw_frames += self.state.frame - before
        return result


def predeclared_specs():
    return [OpeningSpec(title, stage) for title in (0, 1, 31, 127) for stage in (0, 1, 63, 767)]


def conservative_plan_bound():
    # Original boot allows10 START/30 handshakes. Legacy phases reuse one backup.
    core = 181 + 10 * 31 + 1 + 300
    return core + sum(range(24)) + 25 * 32 + sum(
        2 * (core + spec.title_wait + spec.stage_wait + 32) for spec in predeclared_specs())


def semantic_payload(state):
    def tank(t):
        return dict(x=t.x, y=t.y, facing=t.facing, power=t.power,
                    armor=t.armor, drops_item=t.drops_item)
    enemies = sorted((tank(t) for t in state.enemies),
                     key=lambda t: (t["x"], t["y"], t["facing"], t["armor"], t["drops_item"]))
    bullets = sorted((dict(x=b.x, y=b.y, vx=b.vx, vy=b.vy,
                           owner_group="p1" if b.owner == 0 else "p2" if b.owner == 1 else "enemy")
                      for b in state.bullets),
                     key=lambda b: (b["owner_group"], b["x"], b["y"], b["vx"], b["vy"]))
    return dict(
        stage=state.stage, lives=state.lives, remaining=state.remaining_enemies,
        spawn_left=state.spawn_left, kills=state.kills, score=state.score,
        player=tank(state.player) if state.player else None, enemies=enemies, bullets=bullets,
        item=asdict(state.item) if state.item else None, base_destroyed=state.base_destroyed,
        game_over=state.game_over, shield=state.shield, freeze=state.freeze, fire_ready=state.fire_ready,
        terrain_sha256=_hash(state.terrain.tobytes()), brick_masks_sha256=_hash(state.brick_masks.tobytes()),
    )


def semantic_hash(state):
    return _hash(json.dumps(semantic_payload(state), sort_keys=True, separators=(",", ":")).encode())


def _ordinary_conditions(state, baseline=None):
    p = semantic_payload(state)
    valid = (state.stage == 1 and state.lives == 3 and state.remaining_enemies == 20
             and state.kills == 0 and state.score == 0 and not state.base_destroyed
             and not state.game_over and state.item is None)
    if baseline is not None:
        valid = valid and all(p[key] == baseline[key] for key in
                              ("player", "terrain_sha256", "brick_masks_sha256"))
    return bool(valid)


def _trial(env, path, *, role, repeat, baseline):
    before = env.account.snapshot()
    row = dict(role=role, repeat=repeat, spec=asdict(env.spec), status="running", error=None,
               stage=1, raw_initialization_attempted=0, agent_attempted=0)
    states, pixels = [], []
    try:
        env.reset(seed=0)
        ram, screen, initial = env.nes.ram.copy(), env.nes.screen.copy(), semantic_payload(env.state)
        row.update(initial_state=initial, initial_ram_sha256=_hash(ram.tobytes()),
                   initial_pixels_sha256=_hash(screen.tobytes()),
                   initial_semantic_sha256=semantic_hash(env.state),
                   ordinary_conditions_valid=_ordinary_conditions(env.state, baseline),
                   initial_raw_player_xy=[int(ram[0x90]), int(ram[0x98])])
        np.savez_compressed(path.with_suffix(".npz"), ram=ram, pixels=screen,
                            terrain=env.state.terrain, brick_masks=env.state.brick_masks)
        if not row["ordinary_conditions_valid"]:
            raise InvalidOpening("Initial map/player/lives/base/enemy-count conditions differ from ordinary start")
        states.append(initial)
        pixels.append(_hash(screen.tobytes()))
        for index in range(32):
            _, _, terminated, truncated, info = env.step(0)
            states.append(semantic_payload(env.state))
            pixels.append(_hash(env.nes.screen.copy().tobytes()))
            if terminated or truncated:
                row["early_outcome"] = info["outcome"]
                break
        row.update(status="valid", noop_decisions=len(states) - 1,
                   future_semantic_trace_sha256=_hash(json.dumps(states, sort_keys=True,
                                                                separators=(",", ":")).encode()),
                   future_pixels_trace_sha256=_hash(json.dumps(pixels).encode()))
    except InvalidOpening as error:
        row.update(status="invalid_opening", error=str(error))
    except Exception as error:
        row.update(status="aborted", error=f"{type(error).__name__}: {error}")
    finally:
        after = env.account.snapshot()
        row.update(checkpoints=env.checkpoints, wait_status=env.wait_status,
                   raw_initialization_attempted=after["raw_initialization_attempted"] - before["raw_initialization_attempted"],
                   raw_initialization_completed=after["raw_initialization_completed"] - before["raw_initialization_completed"],
                   agent_attempted=after["agent_attempted"] - before["agent_attempted"],
                   agent_completed=after["agent_completed"] - before["agent_completed"],
                   cumulative_accounting=after)
        _write(path.with_suffix(".json"), row)
        with path.with_name(path.name + "-future.jsonl").open("w", encoding="utf-8") as stream:
            for index, state in enumerate(states):
                stream.write(json.dumps(dict(decision=index, state=state), sort_keys=True) + "\n")
    return row


def run_opening_probe(output=NEW_ROOT / "opening-probe"):
    authorization = json.loads((NEW_ROOT / "authorization.json").read_text())
    if authorization.get("version") != "within-stage-authorization-v1" \
            or authorization.get("primary_stage") != 1:
        raise ValueError("New within-stage authorization is required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    epoch = min(LOCAL_DEADLINE, datetime.fromisoformat(authorization["deadline"]).timestamp())
    deadline = time.monotonic() + max(0., epoch - time.time())
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    account = ProbeAccounting(40000, deadline=deadline, monitor=monitor)
    specs = predeclared_specs()
    _write(output / "plan.json", dict(
        title_waits=[0, 1, 31, 127], stage_waits=[0, 1, 63, 767],
        new_menu_specs=[asdict(s) for s in specs], repeats_per_spec=2,
        known_legacy_reference_phases=list(range(24)), legacy_backup_reuse=True,
        legacy0_duplicate=True, noop_decisions=32, fixed_post_start_noop_frames=300,
        no_new_random_postboot_phase=True, conservative_upper_decisions=conservative_plan_bound(),
        deadline_epoch=epoch, authorization=authorization))
    if conservative_plan_bound() > 40000:
        raise ValueError("Predeclared plan exceeds reservation")
    report = dict(
        reservation="a7e9263f3fe442fbbe0b01879fdb5526", status="running", error=None,
        label="Opening feasibility only; not a frozen train/dev/test pool, no model learning",
        previous_results_unchanged=True, old_phases_role="already-exposed development references only",
        semantic_note="RAM/audio/cosmetic bytes and entity slot/generation IDs excluded; positions, physics-visible state and future canonical trajectories included.",
        stage=1, gpu=False, training=False, ram_writes=False, input_oracle=False,
        initialization_protocol="Original legal181-frame title boot and START/30 handshake; inserted NOOP menu waits; START then300 NOOPs; new specs add no postboot jitter.",
        sources={name: _hash((ROOT / name).read_bytes()) for name in
                 ("within_stage\\openings.py", "battle_city_env.py", "game_state.py", "battle_city.nes",
                  "runs\\20260930-within-stage\\authorization.json")},
        trials=[],
    )
    started = time.monotonic()
    baseline = None
    legacy = None
    try:
        monitor.check("admission", admission_slots=1, force=True)
        legacy = NaturalOpeningEnv(OpeningSpec(legacy_phase=0), account=account, reuse_legacy_backup=True)
        for phase in range(24):
            legacy.spec = OpeningSpec(legacy_phase=phase)
            row = _trial(legacy, output / f"legacy-{phase:02d}", role="legacy_reference",
                         repeat=0, baseline=None)
            report["trials"].append(row)
            if phase == 0 and row["status"] == "valid":
                baseline = row["initial_state"]
            if row["status"] != "valid":
                raise RuntimeError(f"Legacy reference phase{phase} failed: {row['error']}")
        legacy.spec = OpeningSpec(legacy_phase=0)
        report["trials"].append(_trial(legacy, output / "legacy-00-repeat",
                                      role="legacy_reference_repeat", repeat=1, baseline=None))
        legacy.close()
        legacy = None
        for index, spec in enumerate(specs):
            env = NaturalOpeningEnv(spec, account=account)
            try:
                for repeat in range(2):
                    row = _trial(env, output / f"menu-{index:02d}-repeat{repeat}",
                                 role="new_menu_probe", repeat=repeat, baseline=baseline)
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
        if legacy is not None:
            legacy.close()
        references = [r for r in report["trials"] if r["role"] == "legacy_reference" and r["status"] == "valid"]
        ref_traces = {r["future_semantic_trace_sha256"] for r in references}
        ref_initial = {r["initial_semantic_sha256"] for r in references}
        groups = defaultdict(list)
        for row in report["trials"]:
            if row["role"] == "new_menu_probe":
                groups[row["config_index"]].append(row)
        summaries = []
        for index, rows in sorted(groups.items()):
            valid = len(rows) == 2 and all(r["status"] == "valid" for r in rows)
            repeated = {}
            if valid:
                repeated = {key: rows[0][key] == rows[1][key] for key in (
                    "initial_ram_sha256", "initial_pixels_sha256", "initial_semantic_sha256",
                    "future_semantic_trace_sha256", "future_pixels_trace_sha256")}
            summaries.append(dict(
                config_index=index, spec=asdict(specs[index]), valid_both=valid,
                repeatability=repeated, statuses=[r["status"] for r in rows],
                initial_semantics_new=valid and rows[0]["initial_semantic_sha256"] not in ref_initial,
                future_semantics_new=valid and rows[0]["future_semantic_trace_sha256"] not in ref_traces,
                future_trace=rows[0].get("future_semantic_trace_sha256"),
            ))
        valid_new = [s for s in summaries if s["valid_both"]
                     and all(s["repeatability"].values()) and s["future_semantics_new"]]
        report.update(accounting=account.snapshot(), elapsed_seconds=time.monotonic() - started,
                      configuration_results=summaries, legacy_unique_initial_semantics=len(ref_initial),
                      legacy_unique_future_traces=len(ref_traces),
                      valid_reproducible_new_configs=len(valid_new),
                      distinct_future_traces_beyond_legacy=len({s["future_trace"] for s in valid_new}),
                      cpu_fixture_real_decisions=0, shared_ledger_written=False)
        _write(output / "report.json", report)
        _write(output / "progress.json", account.snapshot())
    monitor.check("complete", force=True)
    print(json.dumps({k: report[k] for k in (
        "status", "error", "accounting", "legacy_unique_initial_semantics",
        "legacy_unique_future_traces", "valid_reproducible_new_configs",
        "distinct_future_traces_beyond_legacy", "elapsed_seconds")}), flush=True)
    return report


if __name__ == "__main__":
    run_opening_probe()
