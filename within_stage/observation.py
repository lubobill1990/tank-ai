"""Versioned current-geometry offsets; no teacher state or future information."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import game_state as gs
from battle_city_env import CHANNEL_NAMES

OBSERVATION_VERSION = 'within-stage-grid26-precision-v1'
PRECISION_CHANNEL_NAMES = CHANNEL_NAMES + (
    'enemy_offset_x', 'enemy_offset_y',
    'player_bullet_offset_x', 'player_bullet_offset_y',
    'enemy_bullet_offset_x', 'enemy_bullet_offset_y',
    'item_offset_x', 'item_offset_y',
)


def encode_precision(state, legacy):
    if (not isinstance(legacy, np.ndarray) or legacy.shape != (47, 26, 26)
            or legacy.dtype != np.float32 or not np.isfinite(legacy).all()):
        raise ValueError('Precision encoding requires finite canonical float32(47,26,26)')
    extra = np.zeros((8, gs.GRID, gs.GRID), dtype=np.float32)
    counts = np.zeros((4, gs.GRID, gs.GRID), dtype=np.int32)

    def put(group, x, y, cells):
        offset = ((x - gs.ARENA) % gs.CELL / (gs.CELL - 1),
                  (y - gs.ARENA) % gs.CELL / (gs.CELL - 1))
        for row, column in cells:
            if 0 <= row < gs.GRID and 0 <= column < gs.GRID:
                extra[group * 2:group * 2 + 2, row, column] += offset
                counts[group, row, column] += 1

    for enemy in state.enemies:
        put(0, enemy.x, enemy.y, gs.tank_cells(enemy.x, enemy.y))
    for bullet in state.bullets:
        if bullet.owner != 0 and bullet.owner < 2:
            continue
        group = 1 if bullet.owner == 0 else 2
        cell = ((bullet.y - gs.ARENA) // gs.CELL, (bullet.x - gs.ARENA) // gs.CELL)
        put(group, bullet.x, bullet.y, (cell,))
    if state.item is not None:
        item = state.item
        cell = ((item.y - gs.ARENA) // gs.CELL, (item.x - gs.ARENA) // gs.CELL)
        put(3, item.x, item.y, (cell,))
    # Overlapping entities retain the same anonymous aggregation convention as
    # the legacy planes. This improves precision, not full slot-level observability.
    for group in range(4):
        extra[group * 2:group * 2 + 2] /= np.maximum(counts[group], 1)
    return np.concatenate((legacy, extra), axis=0)


class PrecisionObservation(gym.ObservationWrapper):
    def __init__(self, env):
        if env.observation_space.shape != (47, 26, 26) or env.observation_space.dtype != np.float32:
            raise ValueError('Wrap the canonical single-frame environment before stacking')
        super().__init__(env)
        self.observation_space = spaces.Box(-1, 1, (55, 26, 26), np.float32)
        self.observation_version = OBSERVATION_VERSION

    def observation(self, observation):
        return encode_precision(self.unwrapped.state, observation)
