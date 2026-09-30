"""NON-LEARNED teacher v2: last-life close-range firing-lane stand-off.

V1 remains the planner. Only when firing is unavailable and no spare life
remains may a safe primitive move replace its action to leave an enemy's
current firing lane. This is a conservative geometric precaution, not a
prediction of enemy firing or RNG. No extra controller memory is introduced.
"""
import math

import numpy as np

import game_state as gs
from heuristic import _obstacle_time
from learning_lab.teacher import TeacherController as TeacherV1


class TeacherControllerV2(TeacherV1):
    """The frozen v1 controller plus one local firing-lane safety override."""

    @classmethod
    def _lane_risk(cls, state, action):
        position, velocity = cls._motion(state, action)
        x, y = position[0] + 4 * velocity[0], position[1] + 4 * velocity[1]
        fire_facing = (state.player.facing if action == 5 else action - 6) if action >= 5 else None
        risk = 0.
        for enemy in state.enemies:
            dx, dy = gs.DXY[enemy.facing]
            along = (x - enemy.x) * dx + (y - enemy.y) * dy
            across = abs((x - enemy.x) * dy - (y - enemy.y) * dx)
            if not 0 < along <= 48 or across >= 12:
                continue
            # A centered ready shot can intercept; shoulder-offset fire cannot.
            if across <= 4 and state.fire_ready and fire_facing is not None \
                    and gs.DXY[fire_facing] == (-dx, -dy):
                continue
            distance = max(0., along - 17)
            blocked = _obstacle_time(state, enemy.x + dx * 8, enemy.y + dy * 8,
                                     dx, dy, distance)
            if blocked <= distance:
                continue
            risk += (12 - across) * (2 - along / 48)
        return risk

    def act(self, state: gs.GameState, masks: np.ndarray) -> int:
        proposed = super().act(state, masks)
        if state.player is None or state.game_over or state.shield or state.freeze or state.fire_ready \
                or state.lives > 1 \
                or self.mode in ("counter", "safety"):
            return proposed
        proposed_risk = self._lane_risk(state, proposed)
        if proposed_risk <= 0:
            return proposed
        candidates = [proposed]
        for action in range(5):
            if masks[action] and self._danger(state, action) == math.inf:
                candidates.append(action)
        best = min(candidates, key=lambda action: (
            self._lane_risk(state, action), action != proposed, action != 0, action,
        ))
        if self._lane_risk(state, best) < proposed_risk:
            self.mode = "stand_off"
            return int(best)
        return proposed
