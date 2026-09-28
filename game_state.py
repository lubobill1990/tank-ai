"""Verified Battle City RAM snapshots and frame-level combat events.

See docs/ram.md for cartridge-specific evidence. No pixel/13x13 state model.
"""
from dataclasses import dataclass
from functools import lru_cache
import warnings

import numpy as np

GRID, CELL, ARENA = 26, 8, 16
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DXY = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
ROM_FACING = (UP, LEFT, DOWN, RIGHT)
T_EMPTY, T_BRICK, T_STEEL = 0, 1, 2
T_WATER, T_ICE, T_BASE, T_UNKNOWN = 4, 8, 16, 32
ACTION_MASKS = (0, 16, 32, 64, 128, 1, 17, 33, 65, 129)
ACTION_NAMES = ('noop', 'up', 'down', 'left', 'right', 'fire',
                'up_fire', 'down_fire', 'left_fire', 'right_fire',
                'hunt', 'protect', 'dodge', 'counter', 'item')
ITEM_NAMES = ('shield', 'freeze', 'shovel', 'star', 'grenade', 'life')
_OCCUPIED_MATERIALS = frozenset((*range(16), 16, 18, 32, 33, 34))


def terrain_value(value):
    """Remove the ROM's transient tank-occupancy bit only for known materials.

    Base tiles 200..207 use bit7 as part of their identity, not an overlay.
    Unknown bytes are left intact so decoding still surfaces them explicitly.
    """
    value = int(value)
    return value & 0x7f if value & 0x80 and value & 0x7f in _OCCUPIED_MATERIALS else value


@lru_cache(maxsize=256)
def decode_material(value):
    value = terrain_value(value)
    if value in (0, 32, 34):
        return T_EMPTY
    if 1 <= value <= 15:
        return T_BRICK
    if value == 16:
        return T_STEEL
    if value == 18:
        return T_WATER
    if value == 33:
        return T_ICE
    if 200 <= value <= 207:
        return T_BASE
    warnings.warn(f'Unknown terrain byte 0x{value:02X}; treated as a wall',
                  RuntimeWarning, stacklevel=2)
    return T_UNKNOWN


def can_drive(material, *, break_brick=False):
    walls = T_STEEL | T_WATER | T_BASE | T_UNKNOWN
    return not (int(material) & (walls if break_brick else walls | T_BRICK))


def can_shoot_through(material, *, break_brick=False):
    walls = T_STEEL | T_BASE | T_UNKNOWN
    return not (int(material) & (walls if break_brick else walls | T_BRICK))


def is_destructible(material):
    return bool(int(material) & T_BRICK)


def tank_cells(x, y):
    """Cells intersecting [x-8,x+8) x [y-8,y+8), clipped to the arena."""
    r0, r1 = (int(y) - 8 - ARENA) // CELL, (int(y) + 7 - ARENA) // CELL
    c0, c1 = (int(x) - 8 - ARENA) // CELL, (int(x) + 7 - ARENA) // CELL
    return tuple((r, c) for r in range(max(0, r0), min(GRID - 1, r1) + 1)
                 for c in range(max(0, c0), min(GRID - 1, c1) + 1))


def _active(status):
    return 0x80 <= int(status) < 0xe0


def _live_bullet(status):
    return int(status) & 0xf0 == 0x40


@dataclass(frozen=True)
class Tank:
    slot: int
    x: int
    y: int
    facing: int
    power: int = 0
    armor: int = 1
    drops_item: bool = False

    @property
    def tile(self):
        return (self.y - ARENA) // CELL, (self.x - ARENA) // CELL


@dataclass(frozen=True)
class Bullet:
    slot: int
    generation: int
    owner: int
    x: int
    y: int
    vx: int
    vy: int


@dataclass(frozen=True)
class Item:
    x: int
    y: int
    kind: int


@dataclass(frozen=True)
class GameState:
    frame: int
    terrain: np.ndarray
    player: Tank | None
    enemies: tuple[Tank, ...]
    bullets: tuple[Bullet, ...]
    stage: int
    lives: int
    spawn_left: int
    kills: int
    score: int
    item: Item | None
    base_destroyed: bool
    game_over: bool
    shield: bool
    remaining_enemies: int = 20
    fire_ready: bool = False
    freeze: bool = False
    brick_masks: np.ndarray | None = None


@dataclass(frozen=True)
class Shot:
    x: int
    y: int
    facing: int
    power: int


@dataclass(frozen=True)
class FrameEvents:
    kills: int = 0
    deaths: int = 0
    shots: tuple[Shot, ...] = ()
    pickups: tuple[int, ...] = ()
    item_drops: int = 0


def _terrain(ram):
    raw = np.asarray(ram[0x400:0x800]).reshape(32, 32)[2:28, 2:28]
    terrain = np.zeros((GRID, GRID), dtype=np.uint8)
    brick_masks = np.zeros((GRID, GRID), dtype=np.uint8)
    for value in np.unique(raw):
        clean = terrain_value(value)
        cells = raw == value
        terrain[cells] = decode_material(clean)
        if 1 <= clean <= 15:
            brick_masks[cells] = clean
    terrain.setflags(write=False)
    brick_masks.setflags(write=False)
    return terrain, brick_masks


def parse(ram, *, frame=0, kills=None, generations=None, terrain=None, brick_masks=None):
    """Parse a stable emulator-frame boundary, never infer activity from X/Y."""
    if len(ram) != 2048:
        raise ValueError('Expected the complete 2048-byte NES RAM snapshot')
    if terrain is None or brick_masks is None:
        terrain, brick_masks = _terrain(ram)
    tanks = []
    for slot in range(8):
        if _active(ram[0xa0 + slot]):
            power = (int(ram[0x101 + slot]) >> 5) & 3 if slot < 2 else 0
            attribute = int(ram[0xa8 + slot])
            armor = 4 if attribute == 0xe4 else (attribute & 3) + 1
            tanks.append(Tank(slot, int(ram[0x90 + slot]), int(ram[0x98 + slot]),
                              ROM_FACING[int(ram[0xa0 + slot]) & 3], power,
                              armor if slot >= 2 else 1,
                              bool(slot >= 2 and attribute & 0x04)))
    player = next((t for t in tanks if t.slot == 0), None)
    bullets = []
    for slot in range(10):
        if _live_bullet(ram[0xcc + slot]):
            facing = ROM_FACING[int(ram[0xcc + slot]) & 3]
            dx, dy = DXY[facing]
            speed = 4 if int(ram[0xd6 + slot]) & 1 else 2
            bullets.append(Bullet(slot, generations[slot] if generations else 0,
                                  slot & 7, int(ram[0xb8 + slot]),
                                  int(ram[0xc2 + slot]), dx * speed, dy * speed))
    item = None
    if int(ram[0x86]) and not int(ram[0x62]) and int(ram[0x88]) < 6:
        item = Item(int(ram[0x86]), int(ram[0x87]), int(ram[0x88]))
    base_lost = int(ram[0x68]) != 0x80
    lives = int(ram[0x51])
    score = 0
    for digit in ram[0x15:0x1c]:
        score = score * 10 + int(digit)
    return GameState(
        frame, terrain, player,
        tuple(t for t in tanks if t.slot >= 2), tuple(bullets),
        int(ram[0x85]), lives, int(ram[0x7f]),
        sum(int(v) for v in ram[0x73:0x77]) if kills is None else kills,
        score, item, base_lost, base_lost or lives == 0,
        bool(player and int(ram[0x89])), int(ram[0x80]),
        bool(player and (int(ram[0xcc]) == 0
                         or (player.power >= 2 and int(ram[0xd4]) == 0))),
        bool(int(ram[0x100])), brick_masks,
    )


class StateTracker:
    """One update per NES frame. Reading the snapshot never advances history."""
    def reset(self, ram):
        self._previous = np.array(ram, dtype=np.uint8, copy=True)
        self._generations = [0] * 10
        self._next_generation = 1
        for slot in range(10):
            if _live_bullet(ram[0xcc + slot]):
                self._generations[slot] = self._allocate()
        self.state = parse(self._previous, generations=self._generations)
        return self.state

    def _allocate(self):
        generation = self._next_generation
        self._next_generation += 1
        return generation

    def advance(self, ram):
        current = np.array(ram, dtype=np.uint8, copy=True)
        previous = self._previous
        killed = sum(_active(previous[0xa0 + i])
                     and 0x10 <= int(current[0xa0 + i]) < 0x80
                     for i in range(2, 8))
        died = int(_active(previous[0xa0]) and 0x10 <= int(current[0xa0]) < 0x80)
        picked = ()
        if self.state.item is not None and int(current[0x62]) > 0 \
                and int(previous[0x62]) == 0 and int(current[0x49]) == 0:
            picked = (self.state.item.kind,)
        shot_pulse = int(current[0x30f]) == 1 and int(previous[0x30f]) != 1

        transferred = set()
        for primary in (0, 1):
            secondary = primary + 8
            if _live_bullet(previous[0xcc + primary]) \
                    and not _live_bullet(previous[0xcc + secondary]) \
                    and _live_bullet(current[0xcc + secondary]):
                old = next((b for b in self.state.bullets if b.slot == primary), None)
                if old is not None and abs(int(current[0xb8 + secondary]) - old.x) \
                        + abs(int(current[0xc2 + secondary]) - old.y) <= 4:
                    self._generations[secondary] = old.generation
                    transferred.add(primary)
        for slot in range(10):
            if _live_bullet(current[0xcc + slot]):
                if slot >= 8 and slot - 8 in transferred:
                    continue
                if not _live_bullet(previous[0xcc + slot]) or slot in transferred \
                        or (slot == 0 and shot_pulse):
                    self._generations[slot] = self._allocate()

        terrain = self.state.terrain
        brick_masks = self.state.brick_masks
        if not np.array_equal(previous[0x400:0x800], current[0x400:0x800]):
            terrain, brick_masks = _terrain(current)
        before = self.state
        self.state = parse(current, frame=before.frame + 1,
                           kills=before.kills + int(killed),
                           generations=self._generations, terrain=terrain,
                           brick_masks=brick_masks)
        carriers_after = {tank.slot for tank in self.state.enemies if tank.drops_item}
        drops = sum(tank.drops_item and tank.slot not in carriers_after
                    for tank in before.enemies) \
            if self.state.item is not None and self.state.stage == before.stage else 0
        shooter = self.state.player or before.player
        shots = (Shot(shooter.x, shooter.y, shooter.facing, shooter.power),) \
            if shot_pulse and shooter is not None else ()
        self._previous = current
        return self.state, FrameEvents(int(killed), died, shots, picked, int(drops))
