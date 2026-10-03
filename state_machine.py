"""champion-v2: explicit mode state machine + sequence-commitment MPC.

Design A: behaviour is an explicit finite-state machine with per-mode
persistent targets, minimum dwell times and hysteretic transitions - the
six ad-hoc anti-dither patches of champion-v1, consolidated. Priorities:
DODGE > GUARD > ITEM > HUNT; higher priority always preempts, equal or
lower priority waits for the dwell to expire or the mode to invalidate.

Design C: the MPC evaluates candidates as HELD sequences (the first action
repeated for `hold` steps before the random tail) and the controller
commits to the chosen action for `commit` decisions, replanning early only
when danger or mode changes. Temporal consistency and a 2-3x planning
saving come from the same mechanism.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
import heuristic
from base_guard import (BASE_X, BASE_Y, BaseGuardBot, LEFT_POST, RIGHT_POST,
                        bottom_threats)
from mpc_teacher import MpcVetoTeacher

CONTROLLER_VERSION = 'fsm-v2'
PRIORITY = {'DODGE': 3, 'GUARD': 2, 'ITEM': 1, 'HUNT': 0}
MIN_DWELL = {'DODGE': 0, 'GUARD': 4, 'ITEM': 6, 'HUNT': 4}


class SequencePlanner(MpcVetoTeacher):
    """MPC whose candidates hold the first action for several steps."""

    VERSION = 'seqmpc-v2'

    def __init__(self, *, hold=3, **kwargs):
        super().__init__(**kwargs)
        self.hold = hold

    def plan(self, rams):
        from rust_env import NativeBatch
        L = len(rams)
        A, S = 10, self.plan_seeds
        plan_init = np.repeat(np.asarray(rams, np.uint8), A * S, axis=0)
        total = L * A * S
        seeds = [int(self.rng.integers(2**63)) for _ in range(total)]
        plan = NativeBatch(plan_init, 1 << 20, seeds=seeds)
        try:
            first = np.tile(np.repeat(np.arange(A), S), L)
            legal_first = plan.masks()[np.arange(total), first].copy()
            held = np.where(legal_first, first, 0)
            score = np.zeros(total, np.float32)
            failed = np.zeros(total, bool)
            alive = np.ones(total, bool)
            prev = np.zeros((total, 5), np.float32)
            factor = 1.0
            for d in range(self.depth):
                if d < self.hold:
                    # Hold the candidate while it stays legal; else idle.
                    acts = np.where(plan.masks()[np.arange(total), held],
                                    held, 0)
                else:
                    noise = self.rng.random((total, A)).astype(np.float32)
                    acts = np.argmax(noise * plan.masks(), axis=1)
                _, rewards, dones, stats = plan.step(acts)
                sp = stats.astype(np.float32)
                deaths = sp[:, 2] - prev[:, 2]
                score += factor * alive * rewards
                failed |= alive & ((deaths > 0) | (rewards <= -19.0))
                alive &= ~dones
                prev = sp
                factor *= self.discount
            from mpc_teacher import base_threat
            threat = np.zeros(total, np.float32)
            for j in np.flatnonzero(alive):
                threat[j] = base_threat(plan.ram(int(j)))
            score -= self.threat_weight * factor * threat
        finally:
            plan.close()
        score = np.where(legal_first, score, np.float32(-1e9))
        score = score.reshape(L, A, S).mean(axis=2)
        unsafe = (failed | ~legal_first).reshape(L, A, S).any(axis=2)
        return score, unsafe


class StatefulController:
    """Explicit-mode controller with sticky targets and dwell hysteresis."""

    VERSION = CONTROLLER_VERSION

    def __init__(self, planner=None, commit=1):
        self.bot = BaseGuardBot()
        self.planner = planner
        self.commit = commit
        self.mode = 'HUNT'
        self.mode_age = 0
        self.dodge_dir = None
        self.dodge_cooldown = 0
        self.hunt_slot = None
        self.item_key = None
        self.item_age = 0
        self.queue = deque()
        self.position_history = deque(maxlen=24)
        self.last_action = None

    # ------- desires -------
    def _want_item(self, state):
        item, player = state.item, state.player
        if item is None or player is None:
            self.item_age = 0
            return False
        key = (item.x, item.y, item.kind)
        self.item_age = self.item_age + 1 if self.item_key == key else 0
        self.item_key = key
        on_field = len(state.enemies)
        expiring = self.item_age >= 400 or any(e.drops_item for e in state.enemies)
        kind = item.kind
        if kind == 4:
            want = on_field >= 4 or (expiring and on_field >= 1)
        elif kind == 3:
            want = player.power < 3 or expiring
        elif kind in (2, 5):
            want = True
        else:
            want = expiring
        if not want:
            if kind == 4 and on_field < 2:
                return False
            _, cost = self.bot._find_route(state, (player.x, player.y),
                                           (item.x, item.y), max_cost=56)
            want = math.isfinite(cost)
        if want and kind != 2:
            deepest = max((e.y for e in state.enemies), default=0)
            if deepest >= 150:
                _, cost = self.bot._find_route(state, (player.x, player.y),
                                               (item.x, item.y), max_cost=72)
                want = math.isfinite(cost)
        return want

    def _desired_mode(self, state):
        if heuristic.incoming_bullets(state):
            self.dodge_cooldown = 2
            return 'DODGE'
        if self.dodge_cooldown > 0 and self.mode == 'DODGE':
            self.dodge_cooldown -= 1
            return 'DODGE'
        if bottom_threats(state):
            return 'GUARD'
        if self._want_item(state):
            return 'ITEM'
        return 'HUNT'

    def _transition(self, state):
        desired = self._desired_mode(state)
        if desired == self.mode:
            self.mode_age += 1
            return
        if PRIORITY[desired] > PRIORITY[self.mode] \
                or self.mode_age >= MIN_DWELL[self.mode] \
                or not self._mode_valid(state):
            if self.mode == 'DODGE':
                self.dodge_dir = None
            self.mode = desired
            self.mode_age = 0
            self.queue.clear()
        else:
            self.mode_age += 1

    def _mode_valid(self, state):
        if self.mode == 'GUARD':
            return bool(bottom_threats(state))
        if self.mode == 'ITEM':
            return state.item is not None
        if self.mode == 'DODGE':
            return bool(heuristic.incoming_bullets(state))
        return True

    # ------- mode actions -------
    def _act_dodge(self, state):
        player = state.player
        if self.dodge_dir is not None:
            dx, dy = gs.DXY[self.dodge_dir - 1]
            end = (player.x + dx * 16, player.y + dy * 16)
            if math.isfinite(heuristic._segment_cost(
                    state, (player.x, player.y), end, tunnel=False)):
                return self.dodge_dir
        action = self.bot.advise('dodge', state) or self.bot.advise('counter', state)
        self.dodge_dir = action if action in (1, 2, 3, 4) else None
        return action

    def _act_hunt(self, state):
        # Reuse the full original arbitration (base-proximity weighting,
        # opportunistic aim, item budgets) - v2.0's reinvented hunt lost to
        # it in the paired test.
        return self.bot.act(state)

    def _act(self, state):
        if self.mode == 'DODGE':
            return self._act_dodge(state)
        if self.mode == 'GUARD':
            return self.bot._protect(state)
        if self.mode == 'ITEM':
            return self.bot.advise('item', state)
        return self._act_hunt(state)

    def _unstick(self, state):
        player = state.player
        self.position_history.append((player.x, player.y))
        if len(self.position_history) < 24 or self.mode == 'GUARD' \
                or not state.enemies:
            return 0
        xs = [p[0] for p in self.position_history]
        ys = [p[1] for p in self.position_history]
        if max(xs) - min(xs) <= 10 and max(ys) - min(ys) <= 10:
            self.bot.reset()
            self.hunt_slot = None
            self.position_history.clear()
            self.queue.clear()
            return self.bot.advise('hunt', state)
        return 0

    # ------- public -------
    def decide(self, state, masks, ram):
        if state.player is None or state.game_over:
            self.queue.clear()
            return 0
        previous_mode = self.mode
        self._transition(state)
        danger = self.mode in ('DODGE', 'GUARD')
        # A queued (committed) action becomes the PROPOSAL only - the MPC
        # veto still screens every single decision. v2.0 skipped the veto on
        # committed ticks and paid for it in deaths.
        action = 0
        if self.queue and self.mode == previous_mode and not danger:
            queued = self.queue.popleft()
            if masks[queued]:
                action = queued
            else:
                self.queue.clear()
        if not action:
            self.queue.clear()
            action = self._unstick(state) if not danger else 0
            if not action:
                action = self._act(state)
            if not masks[action]:
                action = 0
            if not danger and action in (1, 2, 3, 4) and self.commit > 1:
                self.queue.extend([action] * (self.commit - 1))
        if self.planner is not None:
            try:
                scores, unsafe = self.planner.plan(ram[None])
                vetoed = int(self.planner.choose(
                    scores, unsafe, [masks], [action], [self.last_action])[0])
                if vetoed != action:
                    self.queue.clear()
                action = vetoed
            except Exception:
                pass
        self.last_action = action
        return action
