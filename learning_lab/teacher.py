"""NON-LEARNED development teacher, never a trained/generalization result.

Uses the production geometric navigator unchanged, with a separate primitive
safety filter. Only current canonical snapshots and retained controller state
are used; no emulator access, RAM writes, RNG oracle, or branch rollouts.
"""
import math
import numpy as np
import game_state as gs
from heuristic import HeuristicBot, _box_time, _impact_time, _segment_cost


class TeacherController:
    """Snapshot-only primitive controller; reset once per episode."""

    def __init__(self):
        self.bot = HeuristicBot()
        self.reset()

    def reset(self):
        self.bot.reset()
        self.mode = "baseline"

    @classmethod
    def _base_fire(cls, state, action):
        if action < 5 or state.player is None:
            return False
        player = state.player
        facing = player.facing if action == 5 else action - 6
        dx, dy = gs.DXY[facing]
        position, _ = cls._motion(state, action)
        # Never deliberately strip the fortress on a ray that reaches the base.
        for row, col in np.argwhere(state.terrain & gs.T_BASE):
            x, y = gs.ARENA + int(col) * 8, gs.ARENA + int(row) * 8
            if math.isfinite(_box_time(*position, dx, dy,
                                      (x - 2, y - 2, x + 10, y + 10))):
                return True
        return False

    @staticmethod
    def _motion(state, action):
        player = state.player
        start = (player.x, player.y)
        if action in (0, 5):
            return start, (0, 0)
        facing = action - 1 if action < 5 else action - 6
        dx, dy = gs.DXY[facing]
        if facing != player.facing:
            aligned = (((player.x + 4) // 8 * 8, player.y) if dy else
                       (player.x, (player.y + 4) // 8 * 8))
            if math.isfinite(_segment_cost(state, start, aligned, tunnel=False)):
                start = aligned
        end = start[0] + dx * 3, start[1] + dy * 3
        velocity = (dx * .75, dy * .75)
        if not math.isfinite(_segment_cost(state, start, end, tunnel=False)):
            velocity = (0, 0)
        return start, velocity

    @classmethod
    def _danger(cls, state, action):
        start, velocity = cls._motion(state, action)
        end = start[0] + velocity[0] * 4, start[1] + velocity[1] * 4
        hit = math.inf
        for bullet in state.bullets:
            hit = min(hit, _impact_time(state, bullet, *start,
                                       tank_vx=velocity[0], tank_vy=velocity[1],
                                       horizon=4),
                      _impact_time(state, bullet, *end, offset=4, horizon=20))
        return hit

    def act(self, state: gs.GameState, masks: np.ndarray) -> int:
        masks = np.asarray(masks, dtype=bool)
        if masks.shape != (10,) or not masks.any():
            raise ValueError("Expected a nonempty legal primitive mask of shape (10,)")
        legal = [int(a) for a in np.flatnonzero(masks)]
        if state.player is None or state.game_over:
            return 0 if masks[0] else legal[0]
        safe = [a for a in legal if not self._base_fire(state, a)]
        if not safe:
            return legal[0]
        action = self.bot.act(state)
        if action not in safe:
            action = 0 if 0 in safe else safe[0]
        counter = self.bot.advise("counter", state)
        if counter in safe and counter:
            self.mode = "counter"
            return counter
        danger = {a: self._danger(state, a) for a in safe}
        if not state.shield and danger[action] < math.inf:
            self.mode = "safety"
            return max(safe, key=lambda a: (danger[a], a == 0, a == action, -a))
        self.mode = "baseline"
        return action
