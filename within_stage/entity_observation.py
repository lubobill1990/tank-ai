"""Current-geometry entity tokens plus unchanged two-frame precision raster.

Tokens deliberately omit RAM bytes, slot/generation IDs, RNG state, clocks,
teacher state/actions, counters, trajectories and collision recommendations.
The legacy precision grid remains unchanged (including its existing scalars).
This is a relational representation candidate, not a full-observability claim.
"""
from copy import deepcopy
from numbers import Real

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import game_state as gs
from within_stage.observation import encode_precision, OBSERVATION_VERSION, PRECISION_CHANNEL_NAMES


ENTITY_VERSION = 'within-stage-current-entities-v1'
ENTITY_FIELDS = (
    'valid', 'kind', 'x_over_255', 'y_over_255',
    'relative_player_x_over_255', 'relative_player_y_over_255',
    'relative_base_x_over_255', 'relative_base_y_over_255',
    'direction_x', 'direction_y', 'bullet_vx_over_4', 'bullet_vy_over_4',
    'armor_over_4', 'power_over_3', 'item_kind_plus_one_over_6',
    'carrier_or_bullet_team',
)
TOKEN_GROUPS = (('player', 0, 1), ('enemies', 1, 7), ('bullets', 7, 17), ('item', 17, 18))
BASE_CENTER = (gs.ARENA + 13 * gs.CELL, gs.ARENA + 25 * gs.CELL)


def entity_schema():
    """Return an owned JSON contract; immutable field/group constants define v1."""
    return {
        'version': ENTITY_VERSION, 'grid_observation_version': OBSERVATION_VERSION,
        'grid_shape': [110, 26, 26], 'entities_shape': [18, 16], 'dtype': 'float32',
        'grid_frame_channels': list(PRECISION_CHANNEL_NAMES), 'history_order': ['previous55', 'current55'],
        'fields': list(ENTITY_FIELDS), 'groups': [list(group) for group in TOKEN_GROUPS],
        'kind_codes': {'player': .25, 'enemy': .5, 'bullet': .75, 'item': 1.},
        'position_normalization': 'screen pixel /255; relative differences /255, no rounding or clipping',
        'base_center_pixels': list(BASE_CENTER),
        'base_frame': 'canonical visible grid base landmark center, including when base is destroyed',
        'direction': 'tank facing unit vector; bullet velocity sign; item zero',
        'velocity': 'actual present bullet vx/vy divided by4; tanks/items zero, never inferred future motion',
        'armor': 'current tank armor /4, non-tanks zero',
        'power': 'current player power /3, other token types zero',
        'item_kind': '(current item kind+1)/6, other token types zero',
        'carrier_or_team': 'enemy carrier0/1; bullets P1=-1,P2=-0.5,enemy=1; other types zero',
        'missing_player': 'player token all zero; all relative-player coordinates zero',
        'padding': 'all-zero tokens with valid0; valid1 otherwise; overflow raises',
        'order': 'fixed groups; enemy/bullet tokens sorted lexicographically by encoded fields, '
                 'never by slot/generation; attention has no slot-position embedding',
        'raw_ram_role': 'offline current-geometry conversion only; raw bytes never enter model observations',
    }


def entity_space():
    return spaces.Dict({
        'grid': spaces.Box(-1, 1, (110, 26, 26), np.float32),
        'entities': spaces.Box(-1, 1, (18, 16), np.float32)})


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
            or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in[{low},{high}]')
    return float(value)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in[{low},{high}]')
    return int(value)


def _token(entity, kind, player):
    x, y = _number(entity.x, 'entity x', 0, 255), _number(entity.y, 'entity y', 0, 255)
    result = np.zeros(16, np.float32)
    result[:4] = (1., kind, x / 255, y / 255)
    if player is not None:
        px, py = _number(player.x, 'player x', 0, 255), _number(player.y, 'player y', 0, 255)
        result[4:6] = ((x - px) / 255, (y - py) / 255)
    result[6:8] = ((x - BASE_CENTER[0]) / 255, (y - BASE_CENTER[1]) / 255)
    return result


def _tank_token(tank, player, is_player):
    token = _token(tank, .25 if is_player else .5, player)
    token[8:10] = gs.DXY[_integer(tank.facing, 'tank facing', 0, 3)]
    token[12] = _integer(tank.armor, 'tank armor', 1, 4) / 4
    if is_player:
        token[13] = _integer(tank.power, 'player power', 0, 3) / 3
    else:
        token[15] = bool(tank.drops_item)
    return token


def encode_entities(state):
    """Encode current GameState into owned float32(18,16), without dynamics queries."""
    if len(state.enemies) > 6 or len(state.bullets) > 10:
        raise ValueError('At most six enemies and ten current bullets are supported')
    result = np.zeros((18, 16), np.float32)
    if state.player is not None:
        result[0] = _tank_token(state.player, state.player, True)
    enemies = [_tank_token(tank, state.player, False) for tank in state.enemies]
    for index, token in enumerate(sorted(enemies, key=lambda row: tuple(row))):
        result[1 + index] = token
    bullets = []
    for bullet in state.bullets:
        token = _token(bullet, .75, state.player)
        vx, vy = _number(bullet.vx, 'bullet vx', -4, 4), _number(bullet.vy, 'bullet vy', -4, 4)
        owner = _integer(bullet.owner, 'bullet owner', 0, 7)
        token[8:10] = (np.sign(vx), np.sign(vy))
        token[10:12] = (vx / 4, vy / 4)
        token[15] = -1 if owner == 0 else -.5 if owner == 1 else 1
        bullets.append(token)
    for index, token in enumerate(sorted(bullets, key=lambda row: tuple(row))):
        result[7 + index] = token
    if state.item is not None:
        token = _token(state.item, 1., state.player)
        token[14] = (_integer(state.item.kind, 'item kind', 0, 5) + 1) / 6
        result[17] = token
    return result


def entities_from_ram(ram):
    """Offline audit conversion using only StateTracker.reset's current snapshot."""
    if not isinstance(ram, np.ndarray) or ram.shape != (2048,) or ram.dtype != np.uint8:
        raise ValueError('Expected one complete uint8(2048,) audit RAM snapshot')
    tracker = gs.StateTracker()
    return encode_entities(tracker.reset(ram))


def entity_batch_from_ram(grids, rams):
    """Convert existing precision dataset grids + audit snapshots; no RAM in result."""
    if not isinstance(grids, np.ndarray) or grids.dtype != np.float32 or grids.ndim != 4 \
            or grids.shape[1:] != (110, 26, 26) or not np.isfinite(grids).all() \
            or not isinstance(rams, np.ndarray) or rams.dtype != np.uint8 \
            or rams.shape != (len(grids), 2048):
        raise ValueError('Expected finite precision grids(B,110,26,26) and uint8 RAM(B,2048)')
    tokens = np.empty((len(grids), 18, 16), np.float32)
    for index, ram in enumerate(rams):
        tokens[index] = entities_from_ram(ram)
    return {'grid': grids.copy(), 'entities': tokens}


class EntityObservationEnv(gym.Wrapper):
    """Wrap RAW47 once; this wrapper owns precision encoding AND two-frame history.

    Do not add PrecisionObservation/VecFrameStack outside this wrapper. Reset
    forwards the caller's opening/seed unchanged and zeroes only history, never
    RAM. Every return owns its arrays and info. Actions remain ten primitives.
    """
    def __init__(self, env):
        if env.observation_space.shape != (47, 26, 26) or env.observation_space.dtype != np.float32 \
                or not isinstance(env.action_space, spaces.Discrete) \
                or env.action_space.n != 10 or env.action_space.start != 0:
            raise ValueError('EntityObservationEnv requires raw47 float32 and Discrete(10)')
        super().__init__(env)
        self.observation_space = entity_space()
        self.observation_version = ENTITY_VERSION
        self._previous = None

    def _encode(self, observation):
        state = self.unwrapped.state
        frame = encode_precision(state, observation)
        previous = np.zeros_like(frame) if self._previous is None else self._previous
        result = {'grid': np.concatenate((previous, frame), axis=0), 'entities': encode_entities(state)}
        self._previous = frame.copy()
        return result

    def reset(self, *, seed=None, options=None):
        self._previous = None
        observation, info = self.env.reset(seed=seed, options=options)
        return self._encode(observation), deepcopy(info)

    def step(self, action):
        if self._previous is None:
            raise RuntimeError('Reset before stepping entity observations')
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(observation), reward, terminated, truncated, deepcopy(info)

    def action_masks(self):
        masks = np.asarray(self.env.get_wrapper_attr('action_masks')())
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError('Expected nonempty ten-primitive boolean masks')
        return masks.copy()
