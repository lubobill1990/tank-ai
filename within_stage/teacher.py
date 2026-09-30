"""TRAIN-only teacher refinements, evaluated independently against frozen V2.

NON-LEARNED. No emulator, RNG oracle, RAM edits, map script, or future rollout.
Neither candidate improved the predeclared local8 TRAIN comparison. The public
WithinStageTeacher therefore RETAINS frozen V2 behavior, rather than deploying
a worse rule. The two experimental classes remain named for audit reproduction.
"""
import math

import numpy as np

import game_state as gs
from heuristic import HeuristicBot, _segment_cost
from learning_lab.teacher_v2 import TeacherControllerV2


TEACHER_VERSION = 'within-stage-v2-retained-after-train-refinement-v1'


class ProactiveDefenceTeacher(TeacherControllerV2):
    """reset once per episode; act accepts current canonical GameState and legal masks."""
    def reset(self):
        super().reset()
        self.override_count = 0

    @staticmethod
    def needs_early_defence(state):
        if state.player is None or state.game_over or not state.enemies:
            return False
        base_cells = np.argwhere(state.terrain & gs.T_BASE)
        if not len(base_cells):
            raise ValueError('Canonical terrain is missing the base landmark')
        by, bx = gs.ARENA + (base_cells.mean(axis=0) + .5) * gs.CELL
        guard_y = by - 4 * gs.CELL
        for enemy in state.enemies:
            enemy_distance = abs(enemy.x - bx) + abs(enemy.y - by)
            target = enemy.x, min(guard_y, enemy.y + 3 * gs.CELL)
            return_distance = abs(state.player.x - target[0]) + abs(state.player.y - target[1])
            if enemy_distance <= return_distance + 4 * gs.CELL:
                return True
        return False

    def act(self, state: gs.GameState, masks: np.ndarray) -> int:
        masks = np.asarray(masks)
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError('Expected a nonempty ten-primitive boolean mask')
        original = super().act(state, masks)
        if self.mode in ('counter', 'safety', 'stand_off') or not self.needs_early_defence(state):
            return original
        proposed = self.bot.advise('protect', state)
        if proposed == original or not masks[proposed] or self._base_fire(state, proposed):
            return original
        if not state.shield and self._danger(state, proposed) < math.inf:
            return original
        self.mode = 'proactive_defence'
        self.override_count += 1
        return proposed


class WaypointAlignedNavigator(HeuristicBot):
    def reset(self):
        super().reset()
        self.snap_turns = 0

    def _navigate(self, name, state, target, *, first_step=None):
        proposed = super()._navigate(name, state, target, first_step=first_step)
        player = state.player
        waypoint = self._waypoints.get(name)
        if player is None or waypoint is None:
            return proposed
        start = player.x, player.y
        vertical = player.facing in (gs.UP, gs.DOWN)
        aligned = (player.x, (player.y + 4) // 8 * 8) if vertical else (
            (player.x + 4) // 8 * 8, player.y)
        if aligned == start or aligned != waypoint.end \
                or not math.isfinite(_segment_cost(state, start, aligned, tunnel=False)):
            return proposed
        following = self._find_step(state, aligned, target)
        if following is None:
            return proposed
        x, y = aligned
        nx, ny = following
        direction = None
        if vertical and ny == y and nx != x:
            direction = gs.RIGHT if nx > x else gs.LEFT
        elif not vertical and nx == x and ny != y:
            direction = gs.DOWN if ny > y else gs.UP
        if direction is None:
            return proposed
        action = 1 + direction
        if action != proposed:
            self.snap_turns += 1
        return action


class WaypointAlignedTeacher(TeacherControllerV2):
    """Frozen V2 safety/target choices with one generic navigation-corner correction."""
    def __init__(self):
        super().__init__()
        self.bot = WaypointAlignedNavigator()

    def reset(self):
        super().reset()
        self.override_count = 0

    def act(self, state: gs.GameState, masks: np.ndarray) -> int:
        masks = np.asarray(masks)
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError('Expected a nonempty ten-primitive boolean mask')
        before = self.bot.snap_turns
        chosen = super().act(state, masks)
        if self.bot.snap_turns > before and self.mode == 'baseline':
            self.mode = 'waypoint_turn_snap'
            self.override_count += 1
        return chosen


class WithinStageTeacher(TeacherControllerV2):
    """Selected V2 fallback: no tested candidate improved local TRAIN clear rate."""
    def reset(self):
        super().reset()
        self.override_count = 0

    def act(self, state: gs.GameState, masks: np.ndarray) -> int:
        masks = np.asarray(masks)
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError('Expected a nonempty ten-primitive boolean mask')
        return super().act(state, masks)
