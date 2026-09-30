"""Training-only conditional aiming scenes derived from SkillEnv shoot seed0.

Only copied initialization RAM is changed. Normal movement, turn snapping,
projectiles, collisions, freeze timers and terminal rules remain native engine
mechanics. No RAM is corrected during an episode. These are NOT full games or
the unchanged SkillEnv benchmark. Directions are gs.UP/DOWN/LEFT/RIGHT =0/1/2/3.
"""
import argparse
from collections import Counter
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import game_state as gs
from heuristic import shot_obstacle
from rust_env import NativeBatch, RustVectorEnv, episode_seed
from rust_resources import ResourceMonitor
from skills import SkillEnv

SCENE_VERSION = "native-aiming-four-directions-v1"
FRAME_CAP = 200
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "20260930-victory-aiming-task"
DEADLINE_EPOCH = datetime.fromisoformat("2026-09-30T17:43:36+08:00").timestamp()
_ROM_INITIALIZATION_DECISIONS = 0


@lru_cache(maxsize=1)
def _shoot_reset_ram():
    global _ROM_INITIALIZATION_DECISIONS
    env = SkillEnv("shoot", variant=0, max_frames=FRAME_CAP)
    try:
        # SkillEnv.reset performs exactly two explicit NES NOOP flush frames.
        _ROM_INITIALIZATION_DECISIONS += 2
        env.reset(seed=0)
        initial = env.nes.ram.copy()
        if not valid_scene(initial):
            raise RuntimeError("Existing SkillEnv shoot reset is not a valid aiming base")
        initial.setflags(write=False)
        return initial
    finally:
        env.close()


def _sha_ram(ram):
    return hashlib.sha256(ram.tobytes()).hexdigest()


def _direction(player, enemy):
    if player.x == enemy.x and player.y != enemy.y:
        return gs.DOWN if enemy.y > player.y else gs.UP
    if player.y == enemy.y and player.x != enemy.x:
        return gs.RIGHT if enemy.x > player.x else gs.LEFT
    return None


def generate_aiming_starts(count, seed):
    """Generate owned arrays; count must be a positive multiple of16, at most4096.

    Seed controls layouts ONLY; the source cartridge reset always uses seed0.
    Native import/support validation consumes no decision steps. Metadata reports
    the two raw ROM initialization frames on the first call in this process.
    """
    if type(count) is not int or not 16 <= count <= 4096 or count % 16:
        raise ValueError("count must be a multiple of16 in16..4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Generator seed must be a uint64 integer")
    before = _ROM_INITIALIZATION_DECISIONS
    base = _shoot_reset_ram().copy()
    grid = base[0x400:0x800].reshape(32, 32)[2:28, 2:28]
    for value in np.unique(grid):
        grid[grid == value] = gs.terrain_value(value)
    rng = np.random.default_rng(seed)
    records, seen = [], set()
    centers = np.arange(24, 217, 8)
    for direction in range(4):
        amount = count // 4
        facings = rng.permutation(np.tile(np.arange(4), amount // 4))
        enemy_facings = rng.permutation(np.tile(np.arange(4), amount // 4))
        offsets = rng.permutation(np.tile(np.array([0, 2, 4, 6]), amount // 4))
        dx, dy = gs.DXY[direction]
        for facing, enemy_facing, offset in zip(facings, enemy_facings, offsets):
            for attempt in range(10000):
                distance = int(rng.integers(6, 13)) * 8
                axis_delta = (dx or dy) * distance
                axis = int(rng.choice(centers[(centers + axis_delta >= 24)
                                              & (centers + axis_delta <= 216)]))
                perpendicular = int(rng.choice(centers[centers + offset <= 216])) + int(offset)
                px, py = (perpendicular, axis) if dy else (axis, perpendicular)
                ex, ey = px + dx * distance, py + dy * distance
                initial = base.copy()
                initial[0x90], initial[0x98] = px, py
                initial[0x92], initial[0x9a] = ex, ey
                initial[0xa0] = 0x80 | gs.ROM_FACING.index(int(facing))
                initial[0xa2] = 0xa0 | gs.ROM_FACING.index(int(enemy_facing))
                if not valid_scene(initial):
                    continue
                digest = _sha_ram(initial)
                if digest in seen:
                    continue
                seen.add(digest)
                records.append((initial, direction, (px, py), (ex, ey), int(facing),
                                int(enemy_facing), distance, int(offset), digest))
                break
            else:
                raise RuntimeError("Could not generate a unique legal aiming geometry")
    order = rng.permutation(count)
    records = [records[i] for i in order]
    data = dict(
        rams=np.stack([r[0] for r in records]), directions=np.array([r[1] for r in records], np.int64),
        player_positions=np.array([r[2] for r in records], np.int16),
        enemy_positions=np.array([r[3] for r in records], np.int16),
        player_facings=np.array([r[4] for r in records], np.int64),
        enemy_facings=np.array([r[5] for r in records], np.int64),
        distances=np.array([r[6] for r in records], np.int64),
        subcell_offsets=np.array([r[7] for r in records], np.int64),
        hashes=np.array([r[8] for r in records], dtype="<U64"),
    )
    batch = NativeBatch(data["rams"], FRAME_CAP,
                        seeds=[episode_seed(seed, 0, i) for i in range(count)])
    try:
        masks = batch.masks()
        if masks.shape != (count, 10) or masks.dtype != np.bool_ or not masks[:, 0].all():
            raise RuntimeError("Aiming scenes do not expose ten legal primitive masks")
        for index, initial in enumerate(data["rams"]):
            if not np.array_equal(batch.ram(index), initial):
                raise RuntimeError("Native import changed a generated aiming RAM")
        flags = batch.support.copy()
    finally:
        batch.close()
    data["metadata"] = dict(
        scene_version=SCENE_VERSION, full_game=False, training_only=True, generator_seed=seed,
        count=count, directions={"up": 0, "down": 1, "left": 2, "right": 3},
        frame_cap=FRAME_CAP, rom_boot_seed=0, source="SkillEnv('shoot',variant=0).reset(seed=0)",
        source_ram_sha256=_sha_ram(_shoot_reset_ram()),
        initialization_rom_decisions=_ROM_INITIALIZATION_DECISIONS - before,
        native_decisions=0, support_flags=flags.tolist(),
        modification_scope="Copied initial RAM positions/facings and cleaned occupancy overlays only",
        per_step_ram_writes=False, freeze_timer=int(base[0x100]),
        facing_design="Player AND enemy facings balanced independently within each relative direction",
        conditional_goal="Aim toward relative enemy position, not a fixed compass/action shortcut",
    )
    return data


def validate_split(train, test):
    if train["metadata"]["generator_seed"] == test["metadata"]["generator_seed"]:
        raise ValueError("Holdout generator seed must differ from training")
    overlap = set(train["hashes"]) & set(test["hashes"])
    if overlap:
        raise ValueError(f"Train/test full RAM identities overlap: {sorted(overlap)}")
    for dataset in (train, test):
        if len(set(dataset["hashes"])) != len(dataset["hashes"]):
            raise ValueError("Duplicate RAM identities within a split")
        if any(_sha_ram(ram) != digest for ram, digest in zip(dataset["rams"], dataset["hashes"])):
            raise ValueError("RAM identity hash mismatch")


def valid_scene(ram):
    if not isinstance(ram, np.ndarray) or ram.shape != (2048,) or ram.dtype != np.uint8:
        return False
    state = gs.parse(ram)
    if state.stage != 1 or state.player is None or len(state.enemies) != 1 \
            or state.base_destroyed or state.game_over or state.remaining_enemies != 1 \
            or state.spawn_left != 0 or state.bullets or not state.fire_ready \
            or (int(ram[0x100]) - 1) * 64 <= FRAME_CAP:
        return False
    player, enemy = state.player, state.enemies[0]
    if enemy.slot != 2 or enemy.armor != 1 or enemy.drops_item or int(ram[0xaa]) != 0x80:
        return False
    direction = _direction(player, enemy)
    if direction is None or not 48 <= abs(enemy.x - player.x) + abs(enemy.y - player.y) <= 96:
        return False
    snapped = (((player.x + 4) // 8 * 8, player.y) if direction in (gs.UP, gs.DOWN)
               else (player.x, (player.y + 4) // 8 * 8))
    for x, y in ((player.x, player.y), (enemy.x, enemy.y), snapped):
        if not 24 <= x <= 216 or not 24 <= y <= 216 \
                or not all(gs.can_drive(state.terrain[cell]) for cell in gs.tank_cells(x, y)):
            return False
    return all(shot_obstacle(state, x, y, direction, player.power) == "enemy"
               for x, y in ((player.x, player.y), snapped))


def oracle_action(state, direction, masks):
    desired = 1 + int(direction) if state.player and state.player.facing != direction else 5
    if masks[desired]:
        return desired
    return 0 if masks[0] else int(np.flatnonzero(masks)[0])


def make_aiming_env(starts, *, seed=0):
    """Existing fixed-template RustVectorEnv; parent may add sparse victory/history."""
    if starts["metadata"]["scene_version"] != SCENE_VERSION \
            or any(not valid_scene(ram) for ram in starts["rams"]):
        raise ValueError("Invalid aiming starts")
    return RustVectorEnv(starts["rams"].copy(), FRAME_CAP, seed=seed)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _save_split(output, name, data):
    with (output / f"{name}.npz").open("xb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    _write(output / f"{name}.json", data["metadata"])


def _subset(data, offsets):
    return [int(np.flatnonzero((data["directions"] == direction)
                              & (data["subcell_offsets"] == offset))[0])
            for direction in range(4) for offset in offsets]


def prototype(output=OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    monitor = ResourceMonitor(output / "resources.jsonl", device="cpu")
    deadline = time.monotonic() + max(0., DEADLINE_EPOCH - time.time())
    start = time.monotonic()
    before_boot = _ROM_INITIALIZATION_DECISIONS
    report = dict(
        reservation="fdae8520c2cd4d96a6ccb41bed6c832c", scene_version=SCENE_VERSION,
        label="Conditional aiming micro-task, NOT a full game or learned-policy result",
        status="running", rom_decisions=0, native_decisions=0, max_total_decisions=6000,
        frame_cap=FRAME_CAP, gpu=False, neural_training=False, episodes=[],
        oracle="Turn-only to correct facing, then FIRE when legal; no future lookahead",
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "learning_lab\\aiming_task.py", "skills.py", "game_state.py", "rust_env.py",
            "native\\target\\release\\battle_mirror_probe.dll", "battle_city.nes")},
    )
    try:
        monitor.check("admission", admission_slots=1, force=True)
        train, test = generate_aiming_starts(32, 311), generate_aiming_starts(128, 947)
        validate_split(train, test)
        report["rom_decisions"] = _ROM_INITIALIZATION_DECISIONS - before_boot
        _save_split(output, "train", train)
        _save_split(output, "test", test)
        report["train_metadata"], report["test_metadata"] = train["metadata"], test["metadata"]
        selections = {"train": _subset(train, (0, 4)), "test": _subset(test, (0, 2, 4, 6))}
        report["probe_indices"] = selections
        report["planned_upper_decisions"] = report["rom_decisions"] + 4 * 50 * sum(map(len, selections.values()))
        if report["planned_upper_decisions"] > 6000:
            raise ValueError("Aiming prototype exceeds the reserved cap")
        for split, data in (("train", train), ("test", test)):
            for index in selections[split]:
                for method in ("noop", "fire", "up_fire", "oracle"):
                    monitor.check("native_probe")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Aiming prototype deadline")
                    initial = data["rams"][index]
                    native_seed = episode_seed(711, int(split == "test"), index)
                    row = dict(split=split, index=index, method=method,
                               direction=int(data["directions"][index]),
                               subcell_offset=int(data["subcell_offsets"][index]),
                               player_position=data["player_positions"][index].tolist(),
                               enemy_position=data["enemy_positions"][index].tolist(),
                               player_facing=int(data["player_facings"][index]),
                               distance=int(data["distances"][index]), native_seed=native_seed,
                               decisions=0, frames=0, kills=0, deaths=0, outcome="running")
                    batch, error = None, None
                    try:
                        batch = NativeBatch(initial[None], FRAME_CAP, seeds=[native_seed])
                        current = batch.ram()
                        if not np.array_equal(current, initial):
                            raise RuntimeError("Native import changed aiming scene RAM")
                        state = gs.parse(current, frame=0, kills=0)
                        done = False
                        while not done:
                            monitor.check("native_step")
                            if time.monotonic() >= deadline or report["rom_decisions"] + report["native_decisions"] >= 6000:
                                raise TimeoutError("Aiming time/decision cap")
                            masks = batch.masks()[0]
                            if method == "oracle":
                                action = oracle_action(state, row["direction"], masks)
                            else:
                                requested = {"noop": 0, "fire": 5, "up_fire": 6}[method]
                                action = requested if masks[requested] else (0 if masks[0] else int(np.flatnonzero(masks)[0]))
                            if not masks[action]:
                                raise ValueError("Mask-invalid aiming action")
                            report["native_decisions"] += 1
                            row["decisions"] += 1
                            _, _, dones, stats = batch.step(np.array([action], np.int64))
                            done = bool(dones[0])
                            row.update(frames=int(stats[0, 0]), kills=int(stats[0, 1]),
                                       deaths=int(stats[0, 2]))
                            current = batch.ram()
                            state = gs.parse(current, frame=row["frames"], kills=row["kills"])
                        if state.base_destroyed:
                            row["outcome"] = "base_lost"
                        elif state.remaining_enemies == 0:
                            row["outcome"] = "clear"
                        elif state.lives == 0:
                            row["outcome"] = "lives_exhausted"
                        else:
                            row["outcome"] = "timeout"
                        row.update(clear=row["outcome"] == "clear", support_flags=batch.support.tolist(),
                                   freeze_at_end=int(current[0x100]),
                                   enemy_position_unchanged=bool(np.array_equal(
                                       current[[0x92, 0x9a]], initial[[0x92, 0x9a]])))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        row.update(outcome="aborted", clear=False, error=error)
                    finally:
                        if batch is not None:
                            batch.close()
                        report["episodes"].append(row)
                        with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row) + "\n")
                        _write(output / "progress.json", dict(
                            rom_decisions=report["rom_decisions"], native_decisions=report["native_decisions"],
                            total_decisions=report["rom_decisions"] + report["native_decisions"]))
                    if error:
                        raise RuntimeError(error)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="partial", error=f"{type(error).__name__}: {error}")
    finally:
        report["rom_decisions"] = _ROM_INITIALIZATION_DECISIONS - before_boot
        report["total_decisions"] = report["rom_decisions"] + report["native_decisions"]
        report["elapsed_seconds"] = time.monotonic() - start
        report["summary"] = {}
        for split in ("train", "test"):
            for method in ("noop", "fire", "up_fire", "oracle"):
                rows = [r for r in report["episodes"] if r["split"] == split and r["method"] == method]
                if rows:
                    report["summary"][f"{split}/{method}"] = dict(
                        cases=len(rows), clears=sum(r["clear"] for r in rows),
                        outcomes=dict(Counter(r["outcome"] for r in rows)),
                        decisions=sum(r["decisions"] for r in rows))
        report["oracle_failures"] = [r for r in report["episodes"]
                                     if r["method"] == "oracle" and not r["clear"]]
        _write(output / "report.json", report)
    monitor.check("complete", force=True)
    print(json.dumps({k: report[k] for k in (
        "status", "rom_decisions", "native_decisions", "total_decisions", "summary", "oracle_failures")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prototype(args.output)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
