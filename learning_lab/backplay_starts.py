"""Real-ROM expert RAM pool for native-only reverse-curriculum initialization.

No ROM RAM is edited. NativeBatch receives the unmodified captured RAM once.
Its independent seeded PRNG, neutral input pipeline and zeroed native episode
counters are not an exact continuation of the NES machine. time_left restarts
at one using the native horizon. Formal ROM validation must use normal resets.

All expert snapshots, INCLUDING remaining=20 and teacher_frame=0, are labelled
curriculum starts. ``load_backplay_pool`` adds only a separately captured exact
``rust_env.stage_ram(1, seed=0)`` reference with full_start=True. Use that flag,
not remaining==20, for the final normal-initialization training stage. Loading
is side-effect-free: it never starts an emulator or native environment.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import game_state as gs
from battle_city_env import BattleCityEnv
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_env import NativeBatch, stage_ram, BACKEND_VERSION, RNG_VERSION
from rust_resources import ResourceMonitor


BUCKETS = (1, 2, 4, 8, 12, 16, 20)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "20260930-victory-backplay-starts"
TASK_DEADLINE = datetime.fromisoformat("2026-09-30T15:43:05+08:00").timestamp()


def _json(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _hash_ram(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _validate_ram(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != (2048,):
        raise ValueError("Expected uint8 RAM of shape (2048,)")


class BudgetStop(RuntimeError):
    pass


class DecisionBudget:
    def __init__(self, limit, *, deadline):
        if type(limit) is not int or not 1 <= limit <= 30000:
            raise ValueError("Combined decision allowance must be in 1..30000")
        self.limit, self.deadline = limit, deadline
        self.rom_decisions, self.native_decisions = 0, 0

    @property
    def total(self):
        return self.rom_decisions + self.native_decisions

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetStop("absolute_monotonic_deadline")
        if self.total >= self.limit:
            raise BudgetStop("combined_decision_budget")

    def take(self, domain):
        if domain not in ("rom", "native"):
            raise ValueError("Decision domain must be rom or native")
        self.check()
        if domain == "rom":
            self.rom_decisions += 1
        else:
            self.native_decisions += 1


class SnapshotSelector:
    def __init__(self, per_bucket=4, min_spacing_frames=64):
        if type(per_bucket) is not int or not 1 <= per_bucket <= 4 \
                or type(min_spacing_frames) is not int or min_spacing_frames < 1:
            raise ValueError("Expected 1..4 candidates per bucket and positive frame spacing")
        self.per_bucket, self.spacing = per_bucket, min_spacing_frames
        self.candidates, self.hashes = [], set()
        self.counts, self.opening_counts, self.rejections = Counter(), Counter(), Counter()
        self.last_frame = {}

    def consider(self, state, initial, *, seed, episode_id, initial_ram_sha256):
        if type(seed) is not int or seed not in (0, 1, 2):
            raise ValueError("Only training opening seeds0/1/2 are authorized")
        _validate_ram(initial)
        if not eligible_snapshot(state):
            self.rejections["ineligible_state"] += 1
            return False
        remaining = state.remaining_enemies
        if self.counts[remaining] >= self.per_bucket:
            self.rejections["bucket_full"] += 1
            return False
        opening = (initial_ram_sha256, remaining)
        if self.opening_counts[opening] >= max(1, (self.per_bucket + 1) // 2):
            self.rejections["opening_share_limit"] += 1
            return False
        key = opening
        if state.frame - self.last_frame.get(key, -1000000) < self.spacing:
            self.rejections["spacing"] += 1
            return False
        digest = _hash_ram(initial)
        if digest in self.hashes:
            self.rejections["duplicate_ram"] += 1
            return False
        candidate_id = len(self.candidates)
        self.candidates.append(dict(
            candidate_id=candidate_id, ram=initial.copy(), ram_sha256=digest,
            initial_ram_sha256=initial_ram_sha256, teacher_frame=state.frame,
            remaining=remaining, seed=seed, stage=1, episode_id=episode_id,
            native_rng_seed=271000 + candidate_id,
            source_lives=state.lives, source_power=state.player.power,
            source_shield=state.shield, source_freeze=state.freeze,
            active_enemies=len(state.enemies), source_time_left=max(0., 1 - state.frame / 12000),
        ))
        self.hashes.add(digest)
        self.counts[remaining] += 1
        self.opening_counts[opening] += 1
        self.last_frame[key] = state.frame
        return True


def eligible_snapshot(state):
    return bool(state.stage == 1 and state.remaining_enemies in BUCKETS
                and state.player is not None and not state.game_over
                and not state.base_destroyed and state.enemies and state.fire_ready
                and not any(bullet.owner == 0 for bullet in state.bullets))


def probe_noop(initial, *, native_seed, budget, monitor, max_frames=2000):
    """Import/reset one immutable candidate, then test only legal NOOP actions."""
    _validate_ram(initial)
    if type(max_frames) is not int or not 1 <= max_frames <= 2000:
        raise ValueError("NOOP horizon must be 1..2000 native frames")
    before, batch = budget.native_decisions, None
    result = dict(native_seed=int(native_seed), max_frames=max_frames, decisions=0,
                  frames=0, kills=0, deaths=0, shots=0, pickups=0, supported=False,
                  initial_ram_byte_exact=False, initial_time_left=None, outcome="not_started",
                  automatic_clear=False, automatic_progress=False, retained=False, error=None)
    try:
        budget.check()
        monitor.check("native_import")
        batch = NativeBatch(initial[None].copy(), max_frames, seeds=[int(native_seed)])
        observation = batch.reset(0, seed=int(native_seed)).copy()
        result["initial_ram_byte_exact"] = bool(np.array_equal(batch.ram(0), initial))
        result["initial_time_left"] = float(observation[36, 0, 0])
        result["support_flags_at_import"] = batch.support.tolist()
        if not result["initial_ram_byte_exact"] or not np.all(observation[36] == 1):
            raise RuntimeError("Native reset did not preserve RAM and reset time_left as expected")
        result["supported"] = True
        done = False
        while not done and result["frames"] < max_frames:
            monitor.check("native_noop")
            if not batch.masks()[0, 0]:
                raise RuntimeError("Native NOOP is unexpectedly mask-invalid")
            budget.take("native")
            _, _, terminated, stats = batch.step(np.zeros(1, np.int64))
            done = bool(terminated[0])
            for key, value in zip(("frames", "kills", "deaths", "shots", "pickups"), stats[0]):
                result[key] = int(value)
        terminal = batch.ram(0)
        if int(terminal[0x68]) != 0x80:
            outcome = "base_lost"
        elif int(terminal[0x51]) == 0 or (
            int(terminal[0x51]) <= 1 and 0x10 <= int(terminal[0xa0]) < 0x80
        ):
            outcome = "lives_exhausted"
        elif int(terminal[0x80]) == 0:
            outcome = "clear"
        else:
            outcome = "timeout"
        result.update(outcome=outcome, automatic_clear=outcome == "clear",
                      automatic_progress=result["kills"] > 0, retained=outcome != "clear",
                      terminal_remaining=int(terminal[0x80]),
                      support_flags_at_end=batch.support.tolist())
    except Exception as error:
        result.update(outcome="aborted", error=f"{type(error).__name__}: {error}", retained=False)
    finally:
        result["decisions"] = budget.native_decisions - before
        if batch is not None:
            batch.close()
    return result


def save_pool(path, candidates):
    """Write numeric/string-only arrays; no pickled objects and no dense observations."""
    candidates = list(candidates)
    arrays = {
        "ram": np.stack([r["ram"] for r in candidates]) if candidates else np.empty((0, 2048), np.uint8),
        "teacher_frames": np.array([r["teacher_frame"] for r in candidates], np.int64),
        "remaining": np.array([r["remaining"] for r in candidates], np.int64),
        "seeds": np.array([r["seed"] for r in candidates], np.int64),
        "stages": np.array([r["stage"] for r in candidates], np.int64),
        "episode_ids": np.array([r["episode_id"] for r in candidates], np.int64),
        "candidate_ids": np.array([r["candidate_id"] for r in candidates], np.int64),
        "native_rng_seeds": np.array([r["native_rng_seed"] for r in candidates], np.uint64),
        "ram_sha256": np.array([r["ram_sha256"] for r in candidates], dtype="<U64"),
        "initial_ram_sha256": np.array([r["initial_ram_sha256"] for r in candidates], dtype="<U64"),
        "full_start": np.zeros(len(candidates), bool),
    }
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_backplay_starts(path):
    """Return owned arrays from starts.npz; select a bucket with data['remaining']."""
    path = Path(path)
    if path.is_dir():
        path = path / "starts.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if "ram" not in data or data["ram"].ndim != 2 or data["ram"].shape[1:] != (2048,) \
            or data["ram"].dtype != np.uint8:
        raise ValueError("Invalid backplay RAM array")
    count = len(data["ram"])
    if "full_start" not in data:
        data["full_start"] = np.zeros(count, bool)
    if data["full_start"].shape != (count,) or data["full_start"].dtype != np.bool_ \
            or data["full_start"].any():
        raise ValueError("starts.npz must contain curriculum-only snapshots, never normal starts")
    for name, dtype in (
        ("teacher_frames", np.int64), ("remaining", np.int64), ("seeds", np.int64),
        ("stages", np.int64), ("episode_ids", np.int64), ("candidate_ids", np.int64),
        ("native_rng_seeds", np.uint64), ("ram_sha256", np.dtype("<U64")),
        ("initial_ram_sha256", np.dtype("<U64")),
    ):
        if name not in data or data[name].shape != (count,) or data[name].dtype != dtype:
            raise ValueError(f"Invalid backplay field {name}")
    for index, initial in enumerate(data["ram"]):
        if _hash_ram(initial) != data["ram_sha256"][index]:
            raise ValueError("Backplay RAM SHA256 mismatch")
        state = gs.parse(initial)
        if not eligible_snapshot(state) or state.remaining_enemies != data["remaining"][index] \
                or data["stages"][index] != 1 or data["seeds"][index] not in (0, 1, 2):
            raise ValueError("Backplay state violates the stage/opening/eligibility contract")
    return data


def create_normal_start_reference(output, *, budget, monitor):
    """Capture exact stage_ram(1,seed=0), charging its two direct NES NOOP frames."""
    output = Path(output)
    if (output / "normal_start.npz").exists() or (output / "normal_start.json").exists():
        raise FileExistsError("Normal-start reference already exists")
    budget.check()
    if budget.total + 2 > budget.limit:
        raise BudgetStop("Two raw ROM decisions are required for stage_ram")
    monitor.check("normal_reference")
    budget.take("rom")
    budget.take("rom")
    initial = stage_ram(1, seed=0)
    _validate_ram(initial)
    state = gs.parse(initial)
    if state.stage != 1 or state.remaining_enemies != 20 or state.base_destroyed or state.game_over:
        raise RuntimeError("Exact normal initializer produced an unexpected state")
    native = None
    try:
        native = NativeBatch(initial[None].copy(), 12000, seeds=[271999])
        observation = native.reset(0, seed=271999).copy()
        byte_exact = bool(np.array_equal(native.ram(0), initial))
        if not byte_exact or not np.all(observation[36] == 1):
            raise RuntimeError("Normal native reset altered RAM or time_left")
        flags = native.support.tolist()
    finally:
        if native is not None:
            native.close()
    with (output / "normal_start.npz").open("xb") as stream:
        np.savez_compressed(stream, ram=initial[None], full_start=np.ones(1, bool))
    metadata = dict(
        initialization_method="rust_env.stage_ram(1, seed=0)", stage=1, seed=0,
        full_start=True, curriculum_start=False, remaining=20, source_frame=-1,
        source_frame_note="No teacher trajectory; stage_ram adds two raw ROM NOOP frames after reset",
        ram_sha256=_hash_ram(initial), rom_raw_frame_decisions=2, native_decisions=0,
        native_import_byte_exact=byte_exact, native_support_flags=flags,
        noop_tested=False, native_time_left_at_reset=1.,
    )
    _json(output / "normal_start.json", metadata)
    return metadata


def load_backplay_pool(path):
    """Public training schema with explicit curriculum versus normal-start flags.

    rams:uint8[N,2048]; remaining/source_frames/seeds/stages/candidate_ids:int64[N];
    full_start/noop_tested:bool[N]. A source_frame of -1 denotes the normal
    reference, NOT a teacher frame. All arrays are owned. NOOP-clear and
    unsupported exclusions, missing buckets and the normal-init descriptor are
    returned as JSON-compatible metadata. Remaining20 alone NEVER means normal.
    """
    path = Path(path)
    root = path if path.is_dir() else path.parent
    curriculum = load_backplay_starts(root)
    report_path = root / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    probes = {p["candidate_id"]: p for p in report.get("native_probes", [])}
    automatic = [p["candidate_id"] for p in probes.values() if p.get("automatic_clear")]
    keep = ~np.isin(curriculum["candidate_ids"], automatic)
    ids = curriculum["candidate_ids"][keep]
    tested = np.array([probes.get(int(i), {}).get("outcome") in
                       ("clear", "timeout", "base_lost", "lives_exhausted") for i in ids], bool)
    result = dict(
        rams=curriculum["ram"][keep].copy(), remaining=curriculum["remaining"][keep].copy(),
        source_frames=curriculum["teacher_frames"][keep].copy(),
        seeds=curriculum["seeds"][keep].copy(), stages=curriculum["stages"][keep].copy(),
        candidate_ids=ids.copy(), full_start=np.zeros(int(keep.sum()), bool), noop_tested=tested,
        automatic_noop_clear_exclusions=automatic,
        unsupported_exclusions=[p["candidate_id"] for p in probes.values()
                                if p.get("error") and "Unsupported" in p["error"]],
        missing_buckets=report.get("missing_buckets", []),
        underfilled_buckets=report.get("underfilled_buckets", {}),
        curriculum_note="ALL expert snapshots are curriculum, even remaining20; select full_start=True for final normal initialization",
        normal_start=None,
    )
    reference_path = root / "normal_start.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        with np.load(root / "normal_start.npz", allow_pickle=False) as archive:
            initial = archive["ram"].copy()
            flags = archive["full_start"].copy()
        if initial.shape != (1, 2048) or initial.dtype != np.uint8 \
                or flags.shape != (1,) or flags.dtype != np.bool_ or not flags[0] \
                or _hash_ram(initial[0]) != reference.get("ram_sha256") \
                or reference.get("initialization_method") != "rust_env.stage_ram(1, seed=0)" \
                or reference.get("stage") != 1 or reference.get("seed") != 0 \
                or reference.get("full_start") is not True:
            raise ValueError("Invalid exact-normal-start reference")
        result["rams"] = np.concatenate((result["rams"], initial))
        for key, value in (("remaining", 20), ("source_frames", -1), ("seeds", 0),
                           ("stages", 1), ("candidate_ids", -1)):
            result[key] = np.append(result[key], np.int64(value))
        result["full_start"] = np.append(result["full_start"], True)
        result["noop_tested"] = np.append(result["noop_tested"], False)
        result["normal_start"] = reference
    return result


def collect_backplay_starts(output=OUTPUT, *, max_episodes=3, max_decisions=30000,
                            per_bucket=4, min_spacing_frames=64, noop_frames=2000, deadline=None):
    """Collect normal stage1 seeds0/1/2; return separate ROM/native invocation counts."""
    if type(max_episodes) is not int or not 1 <= max_episodes <= 6:
        raise ValueError("At most six expert ROM episodes are authorized")
    if type(noop_frames) is not int or not 1 <= noop_frames <= 2000:
        raise ValueError("Native NOOP horizons are limited to 2000 frames")
    selector = SnapshotSelector(per_bucket, min_spacing_frames)
    start = time.monotonic()
    hard_deadline = start + max(0., TASK_DEADLINE - time.time())
    deadline = hard_deadline if deadline is None else min(float(deadline), hard_deadline)
    budget = DecisionBudget(max_decisions, deadline=deadline)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    report = dict(
        format="tank-ai-backplay-starts-v1",
        label="Real-ROM teacher snapshot pool; native support probe only, NOT model validation or parity",
        reservation="9c24a7b79bd743c5963135a9822e462f",
        curriculum_order_remaining=list(BUCKETS), stage=1, training_seeds=[0, 1, 2],
        max_episodes=max_episodes, max_decisions=max_decisions, per_bucket=per_bucket,
        min_spacing_frames=min_spacing_frames, native_noop_max_frames=noop_frames,
        deadline_monotonic=deadline, backend_version=BACKEND_VERSION, rng_version=RNG_VERSION,
        native_rng_note="Independent seeded SplitMix PRNG, not ROM RNG replay; per-state seeds stored",
        native_reset_note="Captured RAM is copied unchanged; native frame/kills/deaths/shots/pickups and pending/last input reset to zero. time_left resets to1 using native max_frames, not source ROM elapsed time.",
        formal_rom_validation="Always use normal original-ROM starts, never these training RAM snapshots",
        ram_edits=False, oracle_rng=False, per_step_ram_corrections=False, gpu=False, training=False,
        all_teacher_snapshots_are_curriculum=True, normal_start=None,
        normal_reference_rom_decisions=0,
        episodes=[], candidates=[], native_probes=[], status="running", error=None,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "battle_city.nes", "battle_city_env.py", "game_state.py", "heuristic.py",
            "learning_lab\\teacher.py", "learning_lab\\teacher_v2.py", "rust_env.py",
            "native\\target\\release\\battle_mirror_probe.dll", "learning_lab\\backplay_starts.py")},
    )
    retained = []
    try:
        monitor.check("admission", admission_slots=1, force=True)
        before_reference = budget.rom_decisions
        try:
            report["normal_start"] = create_normal_start_reference(
                output, budget=budget, monitor=monitor)
        finally:
            report["normal_reference_rom_decisions"] = budget.rom_decisions - before_reference
        for episode_id in range(max_episodes):
            budget.check()
            seed = episode_id % 3
            env, info, error = None, dict(kills=0, deaths=0, frames=0, outcome=""), None
            before = budget.rom_decisions
            episode_started = time.monotonic()
            opening_hash = None
            try:
                env = BattleCityEnv(stage=1, objective="full", max_frames=12000, frame_skip=4)
                env.reset(seed=seed)
                opening_hash = _hash_ram(env.nes.ram.copy())
                teacher = TeacherControllerV2()
                teacher.reset()
                while not info["outcome"]:
                    budget.check()
                    monitor.check("rom_expert")
                    selector.consider(env.state, env.nes.ram.copy(), seed=seed, episode_id=episode_id,
                                      initial_ram_sha256=opening_hash)
                    masks = env.action_masks()[:10]
                    action = teacher.act(env.state, masks)
                    if not 0 <= action < 10 or not masks[action]:
                        raise ValueError("Teacher returned a mask-invalid action")
                    budget.take("rom")
                    _, _, _, _, info = env.step(action)
                    if budget.total % 128 == 0:
                        _json(output / "progress.json", dict(rom_decisions=budget.rom_decisions,
                              native_decisions=budget.native_decisions, total_decisions=budget.total))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                info["outcome"] = "aborted"
            finally:
                row = dict(episode_id=episode_id, seed=seed, stage=1,
                           initial_ram_sha256=opening_hash, outcome=info["outcome"],
                           teacher_clear=info["outcome"] == "clear", kills=info["kills"],
                           deaths=info["deaths"], frames=info["frames"],
                           decisions=budget.rom_decisions - before,
                           elapsed_seconds=time.monotonic() - episode_started, error=error)
                report["episodes"].append(row)
                with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if env is not None:
                    env.close()
            print(json.dumps(row), flush=True)
            if error:
                raise RuntimeError(error)
        for candidate in selector.candidates:
            budget.check()
            result = probe_noop(candidate["ram"], native_seed=candidate["native_rng_seed"],
                                budget=budget, monitor=monitor, max_frames=noop_frames)
            result.update(candidate_id=candidate["candidate_id"], remaining=candidate["remaining"])
            report["native_probes"].append(result)
            if result["retained"]:
                retained.append(candidate)
            with (output / "native_probes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            _json(output / "progress.json", dict(rom_decisions=budget.rom_decisions,
                  native_decisions=budget.native_decisions, total_decisions=budget.total))
            if result["error"] and ("BudgetStop" in result["error"]
                                    or "ResourcePressureError" in result["error"]):
                raise RuntimeError(result["error"])
        monitor.check("complete", force=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        save_pool(output / "candidates.npz", selector.candidates)
        save_pool(output / "starts.npz", retained)
        report["candidates"] = [{key: value for key, value in candidate.items() if key != "ram"}
                                for candidate in selector.candidates]
        counts = Counter(row["remaining"] for row in retained)
        report.update(
            retained_starts=len(retained), captured_candidates=len(selector.candidates),
            viable_reset_pool=bool(retained), retained_by_bucket={str(b): counts[b] for b in BUCKETS},
            missing_buckets=[b for b in BUCKETS if not counts[b]],
            underfilled_buckets={str(b): counts[b] for b in BUCKETS if counts[b] < per_bucket},
            selector_rejections=dict(selector.rejections),
            automatic_clear_rejections=sum(r["automatic_clear"] for r in report["native_probes"]),
            unprobed_candidates=len(selector.candidates) - len(report["native_probes"]),
            rom_decisions=budget.rom_decisions, native_decisions=budget.native_decisions,
            rom_controller_decisions=budget.rom_decisions - report["normal_reference_rom_decisions"],
            rom_direct_frame_decisions=report["normal_reference_rom_decisions"],
            total_decisions=budget.total, test_environment_decisions=0,
            rom_frames=sum(e["frames"] for e in report["episodes"]),
            native_frames=sum(p["frames"] for p in report["native_probes"]),
            decision_count_definition="ROM teacher env.step invocations plus direct NES NOOP frames in stage_ram; native singleton batch.step invocations, including a call that raises",
            elapsed_seconds=time.monotonic() - start,
            files={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                   for name in ("starts.npz", "candidates.npz", "normal_start.npz", "normal_start.json")
                   if (output / name).is_file()},
        )
        _json(output / "report.json", report)
    load_backplay_starts(output)
    print(json.dumps({key: report[key] for key in (
        "status", "retained_starts", "captured_candidates", "retained_by_bucket",
        "missing_buckets", "underfilled_buckets", "automatic_clear_rejections",
        "rom_decisions", "native_decisions", "total_decisions", "elapsed_seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=30000)
    parser.add_argument("--per-bucket", type=int, default=4)
    parser.add_argument("--min-spacing-frames", type=int, default=64)
    parser.add_argument("--noop-frames", type=int, default=2000)
    parser.add_argument("--deadline", type=float, help="Absolute time.monotonic() deadline")
    report = collect_backplay_starts(**vars(parser.parse_args()))
    return 0 if report["status"] == "completed" and report["viable_reset_pool"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
