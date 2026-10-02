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

import game_state as gs
from heuristic import HeuristicBot

BOTTOM_BAND = 168
BASE_X, BASE_Y = 120, 216
LEFT_POST = (96, 216)
RIGHT_POST = (144, 216)


def bottom_threats(state):
    return [e for e in state.enemies if e.y >= BOTTOM_BAND]


class BaseGuardBot(HeuristicBot):
    def _protect(self, state):
        threats = bottom_threats(state)
        if not threats:
            return super()._protect(state)
        # Most urgent: closest to the base along the bottom corridor.
        enemy = min(threats, key=lambda e: abs(e.y - BASE_Y) * 2 + abs(e.x - BASE_X))
        action = self._aim(state, enemy)
        if action:
            return action
        post = LEFT_POST if enemy.x < BASE_X else RIGHT_POST
        player = state.player
        if (player.x, player.y) != post:
            action = self._navigate('protect', state, post)
            if action:
                return action
        # Posted (or stuck): face the corridor and trade shots.
        action = self._counter(state)
        if action:
            return action
        facing = gs.LEFT if enemy.x < player.x else gs.RIGHT
        if player.facing != facing:
            return 1 + facing
        return 0
