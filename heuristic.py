"""One snapshot-only controller for native 8px terrain and 16px tanks.

Positions are screen-pixel centers. Navigation searches 8px-spaced centers
with full swept footprints, adding the current/target axes for partial cells.
Brick edges are plans to clear a wall, not permission to drive through it.
Shots probe occupied 4px brick quarters at the center and center-1 on the
perpendicular axis. A missed residual invalidates tunnelling and forces a
reachable alignment or detour on the existing navigation graph.
Near a demolition turn, the ROM's nearest-8px snap can finish alignment without a
frame-skipped forward/backward oscillation; its swept footprint is checked.
Standalone item selection uses a value-dependent budget on that same A* cost,
including detours and demolition. Carriers receive a bounded hunt bonus, never
overriding the standalone controller's bullet response or base defense.
The only retained state is the next navigation waypoint, cleared by reset().
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from collections.abc import Iterator

import numpy as np

import game_state as gs


_HALF = 8
_BULLET_HALF = 2
_LOOKAHEAD = 24
_HIT_RADIUS = 9  # ROM tank/bullet collision requires both absolute deltas < 10.
_DODGE_DISTANCE = 16
_DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_NAMES = ("hunt", "protect", "dodge", "counter", "item")
_GUARD = (120, 184)
_Point = tuple[int, int]
_Rect = tuple[float, float, float, float]
_ANCHOR_X, _ANCHOR_Y = np.meshgrid(
    gs.ARENA + np.arange(gs.GRID) * gs.CELL,
    gs.ARENA + np.arange(gs.GRID) * gs.CELL,
)
_VALID_ANCHORS = (
    (_ANCHOR_X >= gs.ARENA + _HALF) & (_ANCHOR_Y >= gs.ARENA + _HALF)
    & (_ANCHOR_X <= gs.ARENA + gs.GRID * gs.CELL - _HALF)
    & (_ANCHOR_Y <= gs.ARENA + gs.GRID * gs.CELL - _HALF)
)


def _live(state: gs.GameState) -> bool:
    return state.player is not None and not state.game_over and not state.base_destroyed


def _box_time(x: float, y: float, vx: float, vy: float, box: _Rect) -> float:
    """First nonnegative ray/AABB intersection; infinity means no intersection."""
    enter, leave = 0.0, math.inf
    for position, speed, low, high in (
        (x, vx, box[0], box[2]), (y, vy, box[1], box[3]),
    ):
        if speed == 0:
            if position < low or position > high:
                return math.inf
        else:
            a, b = (low - position) / speed, (high - position) / speed
            enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
    return enter if enter <= leave else math.inf


def _cells(box: _Rect):
    x0, y0, x1, y1 = box
    for row in range(
        max(0, math.floor((y0 - gs.ARENA) / gs.CELL)),
        min(gs.GRID, math.ceil((y1 - gs.ARENA) / gs.CELL)),
    ):
        for col in range(
            max(0, math.floor((x0 - gs.ARENA) / gs.CELL)),
            min(gs.GRID, math.ceil((x1 - gs.ARENA) / gs.CELL)),
        ):
            yield row, col


def _cell_box(row: int, col: int, margin: int = 0) -> _Rect:
    x, y = gs.ARENA + col * gs.CELL, gs.ARENA + row * gs.CELL
    return x - margin, y - margin, x + gs.CELL + margin, y + gs.CELL + margin


def _cell_shot_time(
    state: gs.GameState, row: int, col: int,
    x: float, y: float, vx: float, vy: float,
) -> float:
    """ROM terrain probes are center/center-1, not the entity hitbox radius."""
    x0, y0, x1, y1 = _cell_box(row, col)
    mask = 15
    if int(state.terrain[row, col]) == gs.T_BRICK and state.brick_masks is not None:
        mask = int(state.brick_masks[row, col])
    boxes = [(x0, y0, x1, y1)] if mask == 15 else [
        (x0 + (bit % 2) * 4, y0 + (bit // 2) * 4,
         x0 + (bit % 2) * 4 + 4, y0 + (bit // 2) * 4 + 4)
        for bit in range(4) if mask & (1 << bit)
    ]
    return min((
        _box_time(x, y, vx, vy, (
            left, top, right - (1 if vx else 0), bottom - (1 if vy else 0),
        ))
        for left, top, right, bottom in boxes
    ), default=math.inf)


def _terrain_hits(
    state: gs.GameState, x: float, y: float, vx: float, vy: float,
    horizon: float, *, break_brick: bool = False,
) -> Iterator[tuple[float, int]]:
    end_x, end_y = x + vx * horizon, y + vy * horizon
    bounds = (
        min(x, end_x) - 1, min(y, end_y) - 1,
        max(x, end_x) + 1, max(y, end_y) + 1,
    )
    for row, col in _cells(bounds):
        material = int(state.terrain[row, col])
        if not gs.can_shoot_through(material, break_brick=break_brick):
            time = _cell_shot_time(state, row, col, x, y, vx, vy)
            if time <= horizon:
                yield time, material


def _obstacle_time(
    state: gs.GameState, x: float, y: float, vx: float, vy: float,
    horizon: float, *, break_brick: bool = False,
) -> float:
    return min((
        time for time, _ in _terrain_hits(
            state, x, y, vx, vy, horizon, break_brick=break_brick,
        )
    ), default=math.inf)


def _outward_boundary(x: int, y: int, facing: int) -> bool:
    low, high = gs.ARENA + _HALF, gs.ARENA + gs.GRID * gs.CELL - _HALF
    return ((facing == gs.LEFT and x <= low) or (facing == gs.RIGHT and x >= high)
            or (facing == gs.UP and y <= low) or (facing == gs.DOWN and y >= high))


def redundant_boundary_action(state: gs.GameState, primitive: int) -> bool:
    """Only reject outward edge commands when already facing that direction.

    A different direction may still turn and snap the perpendicular coordinate;
    those inputs remain legal, along with NOOP and all inward/tangential inputs.
    """
    if primitive not in range(10):
        raise ValueError(f"Invalid primitive action: {primitive!r}")
    player = state.player
    return bool(player is not None
                and _outward_boundary(player.x, player.y, player.facing)
                and primitive in (1 + player.facing, 5, 6 + player.facing))


def shot_obstacle(
    state: gs.GameState, x: int, y: int, facing: int, power: int = 0,
) -> str:
    """Classify the first physical hit of a shot from a tank-center position.

    The ray starts 8px ahead at the muzzle. Terrain uses the ROM's center and
    center-1 perpendicular probes against occupied 4px brick quarters; entity
    collisions retain their 2px bullet half-width.
    Returns boundary for an immediately outward muzzle at the arena edge.
    Otherwise returns enemy/brick/steel/base/unknown/none; an open infield ray
    remains none, not boundary. Terrain wins equal-distance
    ties with enemies. Water, ice and decoded forest are transparent.
    The controller's fire guard and reward use this same first-hit result.
    Steel is reported even at power >= 3; the caller decides its penalty.
    Coordinates come from the Shot event, independent of state.player.
    """
    if facing not in range(4):
        raise ValueError(f"Invalid shot heading: {facing!r}")
    if _outward_boundary(x, y, facing):
        return "boundary"
    dx, dy = _DXY[facing]
    x, y = x + dx * _HALF, y + dy * _HALF
    horizon = gs.GRID * gs.CELL
    closest, obstacle = math.inf, "none"
    for time, material in _terrain_hits(state, x, y, dx, dy, horizon):
        if time < closest:
            closest = time
            if material & gs.T_BASE:
                obstacle = "base"
            elif material & gs.T_UNKNOWN:
                obstacle = "unknown"
            elif material & gs.T_STEEL:
                obstacle = "steel"
            else:
                obstacle = "brick"
    half = _HIT_RADIUS
    for enemy in state.enemies:
        time = _box_time(x, y, dx, dy, (
            enemy.x - half, enemy.y - half, enemy.x + half, enemy.y + half,
        ))
        if time <= horizon and time < closest:
            closest, obstacle = time, "enemy"
    return obstacle


def _impact_time(
    state: gs.GameState, bullet: gs.Bullet, x: float, y: float,
    *, tank_vx: int = 0, tank_vy: int = 0, horizon: float = _LOOKAHEAD,
    offset: float = 0,
) -> float:
    if bullet.owner < 2 or (bullet.vx == 0 and bullet.vy == 0):
        return math.inf
    half = _HIT_RADIUS
    time = _box_time(
        bullet.x + bullet.vx * offset, bullet.y + bullet.vy * offset,
        bullet.vx - tank_vx, bullet.vy - tank_vy,
        (x - half, y - half, x + half, y + half),
    )
    if time > horizon:
        return math.inf
    blocked = _obstacle_time(
        state, bullet.x, bullet.y, bullet.vx, bullet.vy, offset + time,
    )
    return offset + time if blocked > offset + time else math.inf


def incoming_bullets(state: gs.GameState) -> tuple[gs.Bullet, ...]:
    """Enemy shots reaching the live player's hitbox within 24 emulator frames.

    Velocity is supplied by the snapshot, never inferred or advanced here.
    Friendly P1/P2 shots (owners 0/1) are excluded.
    Water passes shots; brick, steel and unknown materials occlude threats.
    """
    if not _live(state):
        return ()
    player = state.player
    assert player is not None
    threats = [
        (_impact_time(state, bullet, player.x, player.y), bullet)
        for bullet in state.bullets
    ]
    threats.sort(key=lambda pair: (pair[0], pair[1].slot, pair[1].generation))
    return tuple(bullet for time, bullet in threats if math.isfinite(time))


def bullet_danger(state: gs.GameState) -> np.ndarray:
    """24-frame urgency for stationary tank centers on the existing 8px anchors.

    Cell (r,c) represents center (16+8c,16+8r), not the center of a terrain
    square. The player's subcell channels retain its offset from that anchor.
    Only current enemy shots are forecast, stopping at the first real terrain
    hit; this is not a prediction of future firing, evasion or counterfire.
    """
    danger = np.zeros((gs.GRID, gs.GRID), dtype=np.float32)
    if state.game_over:
        return danger
    for bullet in state.bullets:
        if bullet.owner < 2 or (bullet.vx == 0 and bullet.vy == 0):
            continue
        enter = np.zeros_like(danger)
        leave = np.full_like(danger, np.inf)
        for position, velocity, centers in (
            (bullet.x, bullet.vx, _ANCHOR_X), (bullet.y, bullet.vy, _ANCHOR_Y),
        ):
            if velocity:
                a = (centers - _HIT_RADIUS - position) / velocity
                b = (centers + _HIT_RADIUS - position) / velocity
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
            else:
                enter = np.where(np.abs(centers - position) <= _HIT_RADIUS,
                                 enter, np.inf)
        blocked = _obstacle_time(state, bullet.x, bullet.y,
                                 bullet.vx, bullet.vy, _LOOKAHEAD)
        hit = (_VALID_ANCHORS & (enter <= leave)
               & (enter <= _LOOKAHEAD) & (enter < blocked))
        urgency = np.where(hit, (_LOOKAHEAD + 1 - enter) / (_LOOKAHEAD + 1), 0)
        np.maximum(danger, urgency, out=danger)
    return danger


def _safe_fire(state: gs.GameState, position: _Point, facing: int) -> bool:
    """A baseward shot is legal when something actually intercepts it first."""
    power = state.player.power if state.player is not None else 0
    return shot_obstacle(state, *position, facing, power) != "base"


def _fire_action(state: gs.GameState, facing: int) -> int:
    player = state.player
    assert player is not None
    return 5 if player.facing == facing else 6 + facing


def _segment_cost(
    state: gs.GameState, start: _Point, end: _Point, *, tunnel: bool,
) -> float:
    x, y = start
    tx, ty = end
    if x != tx and y != ty:
        return math.inf
    bounds = min(x, tx) - _HALF, min(y, ty) - _HALF, max(x, tx) + _HALF, max(y, ty) + _HALF
    limit = gs.ARENA + gs.GRID * gs.CELL
    if min(bounds[:2]) < gs.ARENA or max(bounds[2:]) > limit:
        return math.inf
    facing = (gs.RIGHT if tx > x else gs.LEFT) if tx != x else (
        gs.DOWN if ty > y else gs.UP
    )
    dx, dy = _DXY[facing]
    bricks = 0
    for row, col in _cells(bounds):
        if row >= 24 and 12 <= col < 14:
            return math.inf
        material = int(state.terrain[row, col])
        if not gs.can_drive(material, break_brick=tunnel):
            return math.inf
        if gs.is_destructible(material):
            # Hypothetical tunnel edges can start inside an already-cleared
            # cell, so include the rear footprint when checking their lane.
            if not math.isfinite(_cell_shot_time(
                state, row, col, x - dx * _HALF, y - dy * _HALF, dx, dy,
            )):
                return math.inf
            bricks += 1
    if bricks:
        if not _safe_fire(state, start, facing):
            return math.inf
    return abs(tx - x) + abs(ty - y) + bricks * gs.CELL * 4


@dataclass
class _Waypoint:
    key: tuple[int, int, int, int]
    start: _Point
    end: _Point

    def contains(self, position: _Point) -> bool:
        x, y = position
        sx, sy = self.start
        ex, ey = self.end
        return (
            self.start != self.end and position != self.end
            and min(sx, ex) <= x <= max(sx, ex)
            and min(sy, ey) <= y <= max(sy, ey)
        )


class HeuristicBot:
    """Standalone baseline and five primitive-action advisors over GameState."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._waypoints: dict[str, _Waypoint] = {}

    def advise(self, name: str, state: gs.GameState) -> int:
        if name not in _NAMES:
            raise ValueError(f"Unknown heuristic advisor: {name!r}")
        if not _live(state):
            return 0
        if name == "dodge":
            return self._dodge(state)
        if name == "counter":
            return self._counter(state)
        if name == "item":
            return 0 if state.item is None else self._navigate(
                name, state, (state.item.x, state.item.y),
            )
        if name == "hunt":
            return self._hunt(state)
        return self._protect(state)

    def act(self, state: gs.GameState) -> int:
        if not _live(state):
            return 0
        if incoming_bullets(state):
            return self._dodge(state) or self._counter(state)
        if any(abs(enemy.x - 120) + abs(enemy.y - 216) < 80 for enemy in state.enemies):
            return self._protect(state)
        if state.item is not None:
            player = state.player
            assert player is not None
            target = state.item.x, state.item.y
            step, cost = self._find_route(
                state, (player.x, player.y), target, max_cost=self._item_budget(state),
            )
            if math.isfinite(cost):
                return self._navigate("item", state, target, first_step=step)
        return self._hunt(state) if state.enemies else self._protect(state)

    @staticmethod
    def _item_budget(state: gs.GameState) -> int:
        """Maximum travel-plus-demolition cost worth paying for this pickup."""
        player, item = state.player, state.item
        assert player is not None and item is not None
        budgets = (
            24 if state.shield else 128,
            24 if state.freeze else 128,
            96,
            24 if player.power >= 3 else 192,
            128 + 16 * min(len(state.enemies), 4),
            256 if state.lives <= 1 else (160 if state.lives <= 3 else 96),
        )
        if item.kind not in range(len(budgets)):
            raise ValueError(f"Unknown item kind: {item.kind!r}")
        return budgets[item.kind]

    def _navigate(
        self, name: str, state: gs.GameState, target: _Point,
        *, first_step: _Point | None = None,
    ) -> int:
        player = state.player
        assert player is not None
        start = (player.x, player.y)
        if start == target:
            self._waypoints.pop(name, None)
            return 0
        key = (*target, state.stage, state.lives)
        waypoint = self._waypoints.get(name)
        if (
            waypoint is None or waypoint.key != key or not waypoint.contains(start)
            or not math.isfinite(_segment_cost(state, start, waypoint.end, tunnel=True))
        ):
            end = first_step if first_step is not None else self._find_step(state, start, target)
            if end is None:
                self._waypoints.pop(name, None)
                return 0
            waypoint = _Waypoint(key, start, end)
            self._waypoints[name] = waypoint
        tx, ty = waypoint.end
        vertical = player.facing in (gs.UP, gs.DOWN)
        aligned = (player.x, (player.y + 4) // 8 * 8) if vertical else (
            (player.x + 4) // 8 * 8, player.y,
        )
        if (
            aligned == waypoint.end and aligned != start
            and math.isfinite(_segment_cost(state, start, aligned, tunnel=False))
        ):
            following = self._find_step(state, aligned, target)
            if following is not None and not math.isfinite(
                _segment_cost(state, aligned, following, tunnel=False),
            ):
                nx, ny = following
                if vertical and ny == ty and nx != tx:
                    return 1 + (gs.RIGHT if nx > tx else gs.LEFT)
                if not vertical and nx == tx and ny != ty:
                    return 1 + (gs.DOWN if ny > ty else gs.UP)
        facing = (gs.RIGHT if tx > player.x else gs.LEFT) if tx != player.x else (
            gs.DOWN if ty > player.y else gs.UP
        )
        if math.isfinite(_segment_cost(state, start, waypoint.end, tunnel=False)):
            return 1 + facing
        return 6 + facing if _safe_fire(state, start, facing) else 0

    @staticmethod
    def _find_step(state: gs.GameState, start: _Point, target: _Point) -> _Point | None:
        return HeuristicBot._find_route(state, start, target)[0]

    @staticmethod
    def _find_route(
        state: gs.GameState, start: _Point, target: _Point, *, max_cost: float = math.inf,
    ) -> tuple[_Point | None, float]:
        """First waypoint and actual A* cost; infinity means no route in budget."""
        if abs(start[0] - target[0]) + abs(start[1] - target[1]) > max_cost:
            return None, math.inf
        centers = range(gs.ARENA + _HALF, gs.ARENA + gs.GRID * gs.CELL, gs.CELL)
        xs, ys = sorted(set(centers) | {start[0], target[0]}), sorted(set(centers) | {start[1], target[1]})
        source, goal = (xs.index(start[0]), ys.index(start[1])), (xs.index(target[0]), ys.index(target[1]))
        costs = {source: 0.0}
        first: dict[tuple[int, int], _Point] = {}
        queue = [(0.0, 0.0, source)]
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost != costs.get(node):
                continue
            if node == goal:
                return first.get(node), cost
            ix, iy = node
            position = xs[ix], ys[iy]
            for dx, dy in _DXY:
                nx, ny = ix + dx, iy + dy
                if not (0 <= nx < len(xs) and 0 <= ny < len(ys)):
                    continue
                end = xs[nx], ys[ny]
                edge = _segment_cost(state, position, end, tunnel=True)
                distance = cost + edge
                estimate = abs(end[0] - target[0]) + abs(end[1] - target[1])
                if distance + estimate <= max_cost and distance < costs.get((nx, ny), math.inf):
                    costs[nx, ny] = distance
                    first[nx, ny] = end if node == source else first[node]
                    heapq.heappush(queue, (distance + estimate, distance, (nx, ny)))
        return None, math.inf

    @staticmethod
    def _aim(state: gs.GameState, enemy: gs.Tank) -> int:
        player = state.player
        assert player is not None
        dx, dy = enemy.x - player.x, enemy.y - player.y
        candidates = []
        if abs(dx) < _HALF + _BULLET_HALF and dy:
            candidates.append(gs.DOWN if dy > 0 else gs.UP)
        if abs(dy) < _HALF + _BULLET_HALF and dx:
            candidates.append(gs.RIGHT if dx > 0 else gs.LEFT)
        for facing in candidates:
            if not _safe_fire(state, (player.x, player.y), facing):
                continue
            vx, vy = _DXY[facing]
            x, y = player.x + vx * _HALF, player.y + vy * _HALF
            half = _HALF + _BULLET_HALF
            hit = _box_time(x, y, vx, vy, (
                enemy.x - half, enemy.y - half, enemy.x + half, enemy.y + half,
            ))
            if _obstacle_time(state, x, y, vx, vy, hit, break_brick=True) > hit:
                return _fire_action(state, facing)
        return 0

    def _hunt(self, state: gs.GameState) -> int:
        player = state.player
        assert player is not None
        enemies = sorted(state.enemies, key=lambda enemy: (
            abs(enemy.x - player.x) + abs(enemy.y - player.y)
            - max(0, 80 - abs(enemy.x - 120) - abs(enemy.y - 216))
            - (24 if enemy.drops_item else 0),
            enemy.slot,
        ))
        for enemy in enemies:
            action = self._aim(state, enemy)
            if action:
                return action
        for enemy in enemies:
            action = self._navigate("hunt", state, (enemy.x, enemy.y))
            if action:
                return action
        return 0

    def _protect(self, state: gs.GameState) -> int:
        if state.enemies:
            enemy = min(state.enemies, key=lambda e: (
                abs(e.x - 120) + abs(e.y - 216), e.slot,
            ))
            action = self._aim(state, enemy)
            if action:
                return action
            # Get on the base side of the threat, but stay above the fortress.
            target = enemy.x, min(_GUARD[1], enemy.y + 24)
            action = self._navigate("protect", state, target)
            if action:
                return action
        return self._navigate("protect", state, _GUARD)

    @staticmethod
    def _counter(state: gs.GameState) -> int:
        player = state.player
        assert player is not None
        if not state.fire_ready:
            return 0
        for bullet in incoming_bullets(state):
            if bullet.vx and not bullet.vy:
                facing = gs.LEFT if bullet.vx > 0 else gs.RIGHT
                aligned = abs(bullet.y - player.y) <= 2 * _BULLET_HALF
                distance = abs(bullet.x - player.x)
            elif bullet.vy and not bullet.vx:
                facing = gs.UP if bullet.vy > 0 else gs.DOWN
                aligned = abs(bullet.x - player.x) <= 2 * _BULLET_HALF
                distance = abs(bullet.y - player.y)
            else:
                continue
            if aligned and distance > _HALF + _BULLET_HALF and _safe_fire(
                state, (player.x, player.y), facing,
            ):
                dx, dy = _DXY[facing]
                clear = _obstacle_time(
                    state, player.x + dx * _HALF, player.y + dy * _HALF,
                    dx, dy, distance - _HALF,
                )
                if clear > distance - _HALF:
                    return _fire_action(state, facing)
        return 0

    @staticmethod
    def _dodge(state: gs.GameState) -> int:
        threats = incoming_bullets(state)
        if not threats:
            return 0
        player = state.player
        assert player is not None
        stay = min(_impact_time(state, b, player.x, player.y) for b in threats)
        candidates = []
        for facing, (dx, dy) in enumerate(_DXY):
            end = player.x + dx * _DODGE_DISTANCE, player.y + dy * _DODGE_DISTANCE
            if not math.isfinite(_segment_cost(state, (player.x, player.y), end, tunnel=False)):
                continue
            hit = math.inf
            for bullet in state.bullets:
                hit = min(hit, _impact_time(
                    state, bullet, player.x, player.y, tank_vx=dx, tank_vy=dy,
                    horizon=_DODGE_DISTANCE,
                ), _impact_time(
                    state, bullet, *end, offset=_DODGE_DISTANCE,
                    horizon=_LOOKAHEAD - _DODGE_DISTANCE,
                ))
            if hit <= stay:
                continue
            perpendicular = dx * threats[0].vx + dy * threats[0].vy == 0
            base_distance = abs(end[0] - _GUARD[0]) + abs(end[1] - _GUARD[1])
            candidates.append((-hit, not perpendicular, base_distance, facing))
        return 1 + min(candidates)[3] if candidates else 0
