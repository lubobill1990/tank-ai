"""Flank-post base defence: deny the bottom corridor beside the fortress.

Diagnosis across the losing stages showed every base loss came from enemies
reaching the bottom row (y≈216) and shooting horizontally along it, or
driving down the centre column, while the guard point (120,184) sits above
the fortress and covers neither. BaseGuardBot replaces the protect advisor:
when an enemy enters the bottom band it moves to the flank post on that side
of the base at row level, where the tank's own body blocks the corridor and
the ordinary aim/counter logic shoots along it. Without bottom-band threats
the stock behaviour is kept.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import game_state as gs
from heuristic import HeuristicBot

BOTTOM_BAND = 168
BASE_X, BASE_Y = 120, 216
LEFT_POST = (96, 216)
RIGHT_POST = (144, 216)

# Per-process navigation guard: ice cells cost extra, and tunnelling through
# the base fortress bricks is forbidden (the fortress is not a shortcut).
_ICE_PENALTY = 0.0
_PATCHED = False
FORTRESS_BOX = (23, 26, 11, 15)  # rows, cols half-open: the base brick jacket only


def _fortress_brick_cells(state):
    r0, r1, c0, c1 = FORTRESS_BOX
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1)
            if int(state.terrain[r, c]) == gs.T_BRICK]


def set_ice_penalty(value):
    """Set the ice charge and install the navigation guard (once)."""
    global _ICE_PENALTY, _PATCHED
    _ICE_PENALTY = float(value)
    if _PATCHED:
        return
    import math as _math
    import heuristic as _heuristic
    original = _heuristic._segment_cost

    def guarded_segment_cost(state, start, end, *, tunnel):
        cost = original(state, start, end, tunnel=tunnel)
        if not _math.isfinite(cost):
            return cost
        x, y = start
        tx, ty = end
        bounds = (min(x, tx) - 8, min(y, ty) - 8, max(x, tx) + 8, max(y, ty) + 8)
        if tunnel:
            r0, r1, c0, c1 = FORTRESS_BOX
            for row, col in _heuristic._cells(bounds):
                if r0 <= row < r1 and c0 <= col < c1 \
                        and int(state.terrain[row, col]) == gs.T_BRICK:
                    return _math.inf  # never plan demolition through the fortress
        if _ICE_PENALTY:
            ice = sum(1 for r, c in _heuristic._cells(bounds)
                      if int(state.terrain[r, c]) == gs.T_ICE)
            cost += ice * _ICE_PENALTY
        return cost

    _heuristic._segment_cost = guarded_segment_cost
    # Methods captured the module global by name at call time, so rebinding
    # the module attribute is sufficient; keep a handle for debugging.
    _heuristic._segment_cost_original = original
    _PATCHED = True


def restrict_masks(state, masks):
    """Clear fire actions whose first hit would chew the own fortress bricks.

    A shot is allowed when an enemy sits on the ray closer than the terrain
    hit (that is what the fire is for); otherwise firing toward a fortress
    brick is pointless self-harm and is masked out at the source so neither
    the executor nor the MPC veto can pick it.
    """
    from heuristic import _HIT_RADIUS, _box_time, _obstacle_time, _DXY
    player = state.player
    if player is None:
        return masks
    masks = masks.copy()
    r0, r1, c0, c1 = FORTRESS_BOX
    for action in (5, 6, 7, 8, 9):
        if not masks[action]:
            continue
        facing = player.facing if action == 5 else action - 6
        dx, dy = _DXY[facing]
        x, y = player.x + dx * 8, player.y + dy * 8
        horizon = gs.GRID * gs.CELL
        hit_time = _obstacle_time(state, x, y, dx, dy, horizon)
        if not np.isfinite(hit_time):
            continue
        hx, hy = x + dx * hit_time, y + dy * hit_time
        row = int((hy - gs.ARENA) // gs.CELL)
        col = int((hx - gs.ARENA) // gs.CELL)
        if not (r0 <= row < r1 and c0 <= col < c1):
            continue
        enemy_first = any(
            _box_time(x, y, dx, dy, (e.x - _HIT_RADIUS, e.y - _HIT_RADIUS,
                                     e.x + _HIT_RADIUS, e.y + _HIT_RADIUS))
            < hit_time for e in state.enemies)
        if not enemy_first:
            masks[action] = False
    if not masks.any():
        masks[0] = True
    return masks


def bottom_threats(state):
    return [e for e in state.enemies if e.y >= BOTTOM_BAND]


GUARD_RACE_MARGIN = 48  # px of head start we must keep over an intruder

# Feature toggles for controlled A/B evaluation (set per process).
PREDICT_GUARD = False
INTERCEPT_FIRE = False


def set_features(predict_guard=None, intercept_fire=None):
    global PREDICT_GUARD, INTERCEPT_FIRE
    if predict_guard is not None:
        PREDICT_GUARD = bool(predict_guard)
    if intercept_fire is not None:
        INTERCEPT_FIRE = bool(intercept_fire)


def losing_guard_race(state):
    """Predictive base-defence trigger: are we about to lose the travel race?

    The reactive y>=168 trigger fires when an intruder already owns the
    bottom corridor - usually too late to walk back. Here the intruder's
    remaining path to the base is compared against our path to the matching
    flank post; the guard starts the moment waiting would lose the race.
    """
    player = state.player
    if player is None:
        return False
    for enemy in state.enemies:
        if enemy.y < 120:
            continue
        enemy_path = abs(enemy.x - BASE_X) + abs(enemy.y - BASE_Y)
        post = LEFT_POST if enemy.x < BASE_X else RIGHT_POST
        my_path = abs(player.x - post[0]) + abs(player.y - post[1])
        if enemy_path - my_path < GUARD_RACE_MARGIN:
            return True
    return False


def intercept_aim(bot, state):
    """Predictive fire: shoot where a moving enemy and my bullet will meet.

    _aim only fires at enemies already sitting on a firing lane. Tanks spend
    most of their time crossing lanes instead; assuming an enemy keeps
    moving along its facing at tank speed, fire early so the bullet and the
    enemy reach the lane crossing together. Terrain and own-base safety use
    the same guards as ordinary aimed fire.
    """
    from heuristic import _obstacle_time, _safe_fire, _fire_action
    player = state.player
    if player is None or not state.fire_ready:
        return 0
    bullet_speed = 4.0 if player.power >= 1 else 2.0
    tank_speed = 1.0  # conservative upper band; basic tanks ~0.75 px/frame
    for enemy in state.enemies:
        evx, evy = gs.DXY[enemy.facing]
        for facing, (dx, dy) in enumerate((gs.DXY[0], gs.DXY[1],
                                           gs.DXY[2], gs.DXY[3])):
            # Enemy must be moving ACROSS this firing lane, not along it.
            if dx and not evy:
                continue
            if dy and not evx:
                continue
            if dx:  # horizontal lane: enemy crosses y == player.y
                gap = (player.y - enemy.y)
                if gap * evy <= 0:
                    continue
                t_cross = (abs(gap) - 6) / tank_speed
                meet = enemy.x + evx * 0  # enemy x is constant while crossing
                if (meet - player.x) * dx <= 8:
                    continue
                t_bullet = (abs(meet - player.x) - 8) / bullet_speed
            else:  # vertical lane: enemy crosses x == player.x
                gap = (player.x - enemy.x)
                if gap * evx <= 0:
                    continue
                t_cross = (abs(gap) - 6) / tank_speed
                meet = enemy.y
                if (meet - player.y) * dy <= 8:
                    continue
                t_bullet = (abs(meet - player.y) - 8) / bullet_speed
            if t_cross > 40 or abs(t_cross - t_bullet) > 5:
                continue
            if not _safe_fire(state, (player.x, player.y), facing):
                continue
            x, y = player.x + dx * 8, player.y + dy * 8
            distance = abs(meet - (player.x if dx else player.y)) - 8
            if _obstacle_time(state, x, y, dx, dy, distance) <= distance:
                continue
            return _fire_action(state, facing)
    return 0


ITEM_URGENT_AGE = 400  # decisions (~27s) before an uncollected item is urgent
NEARBY_ITEM_COST = 56  # px of travel+demolition that makes any item a cheap grab


def item_reflex(bot, state):
    """Value-aware pickup policy; 0 means no pickup urge right now.

    Lives and shovels are always worth the trip; stars only while firepower
    is not maxed; the grenade's worth equals the enemies it would erase, so
    it is saved until the screen is crowded - unless it is about to be lost
    (age, or a carrier on field whose drop would replace it). Anything
    sitting within a cheap detour is grabbed on the way.
    """
    item, player = state.item, state.player
    if item is None or player is None:
        bot._item_key = None
        return 0
    key = (item.x, item.y, item.kind)
    bot._item_age = getattr(bot, '_item_age', 0) + 1 \
        if getattr(bot, '_item_key', None) == key else 0
    bot._item_key = key
    on_field = len(state.enemies)
    carrier_waiting = any(e.drops_item for e in state.enemies)
    expiring = bot._item_age >= ITEM_URGENT_AGE or carrier_waiting
    kind = item.kind
    if kind == 4:  # grenade
        want = on_field >= 4 or (expiring and on_field >= 1)
    elif kind == 3:  # star
        want = player.power < 3 or expiring
    elif kind in (2, 5):  # shovel, extra life
        want = True
    else:  # shield, freeze
        want = expiring
    import math
    if not want:
        if kind == 4 and on_field < 2:
            return 0  # a wasted grenade is worse than a waiting one
        _, cost = bot._find_route(state, (player.x, player.y),
                                  (item.x, item.y), max_cost=NEARBY_ITEM_COST)
        want = math.isfinite(cost)
    if want and kind != 2:
        # Someone is pushing the lower half: only a cheap detour is allowed,
        # the base outlives any power-up except the shovel.
        deepest = max((e.y for e in state.enemies), default=0)
        if deepest >= 150:
            _, cost = bot._find_route(state, (player.x, player.y),
                                      (item.x, item.y), max_cost=72)
            want = math.isfinite(cost)
    if want:
        return bot.advise('item', state)
    return 0


def unstick(bot, state):
    """Detect in-place dithering and clear the stale navigation waypoints."""
    from collections import deque
    player = state.player
    if player is None:
        return 0
    history = getattr(bot, '_pos_hist', None)
    if history is None:
        history = bot._pos_hist = deque(maxlen=24)
    history.append((player.x, player.y))
    if len(history) < 24 or not state.enemies \
            or any(b.owner >= 2 for b in state.bullets):
        return 0
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    if max(xs) - min(xs) <= 10 and max(ys) - min(ys) <= 10:
        bot.reset()  # drop cached waypoints that keep contradicting each other
        history.clear()
        return bot.advise('hunt', state)
    return 0


class BaseGuardBot(HeuristicBot):
    def _protect(self, state):
        threats = bottom_threats(state)
        if not threats:
            self._guard_slot = None
            return super()._protect(state)
        # Sticky threat selection: keep tracking the same enemy while it is
        # still in the bottom band; only retarget when it leaves/dies or a
        # clearly closer intruder appears. Without hysteresis the argmin
        # flips between two flanking enemies every few frames.
        def urgency(e):
            return abs(e.y - BASE_Y) * 2 + abs(e.x - BASE_X)
        enemy = None
        slot = getattr(self, '_guard_slot', None)
        if slot is not None:
            enemy = next((e for e in threats if e.slot == slot), None)
        best = min(threats, key=urgency)
        if enemy is None or urgency(best) + 48 < urgency(enemy):
            enemy = best
        self._guard_slot = enemy.slot
        action = self._aim(state, enemy)
        if action:
            return action
        # Sticky post side with a switching margin around the base centre.
        side = getattr(self, '_guard_side', None)
        if side is None or (enemy.x - BASE_X) * side < -16:
            side = -1 if enemy.x < BASE_X else 1
        self._guard_side = side
        post = LEFT_POST if side < 0 else RIGHT_POST
        player = state.player
        if abs(player.x - post[0]) + abs(player.y - post[1]) > 12:
            action = self._navigate('protect', state, post)
            if action:
                return action
        # Posted (or close enough): stand and trade shots, no micro-walking.
        action = self._counter(state)
        if action:
            return action
        facing = gs.LEFT if enemy.x < player.x else gs.RIGHT
        if player.facing != facing:
            return 1 + facing
        return 0
