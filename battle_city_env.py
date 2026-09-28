"""The single 26x26 Battle City environment for learning and heuristics."""
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from nes_py import NESEnv
import numpy as np

import game_state as gs
from heuristic import (HeuristicBot, bullet_danger, incoming_bullets,
                       redundant_boundary_action, shot_obstacle)

ENV_VERSION = 'grid26-events-v7-clean-terrain'
CHANNEL_NAMES = (
    'brick', 'steel', 'water', 'ice', 'base',
    'player', 'player_fx', 'player_fy', 'enemy', 'enemy_fx', 'enemy_fy', 'armor',
    'pbullet', 'pbullet_vx', 'pbullet_vy', 'ebullet', 'ebullet_vx', 'ebullet_vy',
    'remaining', 'spawn_left', 'lives', 'power', 'shield', 'freeze', 'fire_ready',
    'item_shield', 'item_freeze', 'item_shovel', 'item_star', 'item_grenade', 'item_life',
    'last_hunt', 'last_protect', 'last_dodge', 'last_counter', 'last_item',
    'time_left', 'objective_combat', 'objective_defense',
    'brick_tl', 'brick_tr', 'brick_bl', 'brick_br', 'player_offset_x', 'player_offset_y',
    'bullet_danger',
    'enemy_drops_item',
)
OBS_CHANNELS = len(CHANNEL_NAMES)
MACROS = ('hunt', 'protect', 'dodge', 'counter', 'item')
OBJECTIVES = ('combat', 'defense', 'full')
DEFENSE_FRAMES = 900
COMBAT_PICKUP_FRAMES = 600


def _danger(state):
    if not state.enemies:
        return 0.0
    distance = min(abs(t.x - 120) + abs(t.y - 216) for t in state.enemies)
    return max(0.0, 1 - distance / 96)


class BattleCityEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 60}

    def __init__(self, stage=1, objective='full', max_frames=12000,
                 render_mode='rgb_array', frame_skip=4, rom_path=None,
                 reward_profile='shaped'):
        super().__init__()
        if not isinstance(stage, int) or not 1 <= stage <= 35:
            raise ValueError('stage must be an integer in 1..35')
        if objective not in OBJECTIVES:
            raise ValueError(f'objective must be one of {OBJECTIVES}')
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError('max_frames must be a positive integer')
        if not isinstance(frame_skip, int) or frame_skip <= 0:
            raise ValueError('frame_skip must be a positive integer')
        if render_mode not in self.metadata['render_modes']:
            raise ValueError('Unsupported render mode')
        if reward_profile not in ('shaped', 'sparse'):
            raise ValueError('reward_profile must be shaped or sparse')
        rom = Path(rom_path) if rom_path else Path(__file__).with_name('battle_city.nes')
        if not rom.is_file():
            raise FileNotFoundError(rom)
        self.nes = NESEnv(str(rom))
        self.stage, self.objective = stage, objective
        self.max_frames, self.frame_skip = max_frames, frame_skip
        self.render_mode = render_mode
        self.reward_profile = reward_profile
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            (OBS_CHANNELS, gs.GRID, gs.GRID),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(15)
        self._tracker = gs.StateTracker()
        self._bot = HeuristicBot()
        self._needs_reset = True

    @property
    def state(self):
        return self._tracker.state

    def _boot(self):
        for _ in range(181):
            self.nes.step(0)
        for _ in range(10):
            if int(self.nes.ram[0x85]) == 1:
                break
            self.nes.step(8)
            for _ in range(30):
                self.nes.step(0)
        if int(self.nes.ram[0x85]) != 1:
            raise RuntimeError('Failed to reach the stage-selection screen')
        for _ in range(self.stage - 1):
            for _ in range(3):
                self.nes.step(1)
            for _ in range(10):
                self.nes.step(0)
        if int(self.nes.ram[0x85]) != self.stage:
            raise RuntimeError('Stage selection failed')
        self.nes.step(8)
        for _ in range(300):
            self.nes.step(0)
        self.nes._backup()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Reset options are not supported; configure the environment explicitly')
        self.nes.reset()
        if not self.nes._has_backup:
            self._boot()
        for _ in range(int(self.np_random.integers(0, 24))):
            self.nes.step(0)
        self._tracker.reset(self.nes.ram)
        self._bot.reset()
        self._last_macro = -1
        self._deaths = self._shots = self._pickups = 0
        self._item_drops = 0
        self._item_opportunities = int(self.state.item is not None)
        self._combat_pickup_deadline = None
        self._last_progress_frame = 0
        self._boundary_waste_frames = self._boundary_waste_streak_frames = 0
        self._max_boundary_waste_frames = 0
        self._needs_reset = False
        return self._observation(), self._info('', {})

    def action_masks(self):
        if self._needs_reset:
            raise gym.error.ResetNeeded('Reset before querying actions')
        state = self.state
        masks = np.ones(15, dtype=bool)
        masks[5] = state.fire_ready
        masks[10] = state.player is not None and bool(state.enemies)
        masks[11] = state.player is not None
        danger = bool(incoming_bullets(state))
        masks[12] = state.player is not None and danger
        masks[13] = bool(danger and self._bot.advise('counter', state))
        masks[14] = state.player is not None and state.item is not None
        for action in range(1, 10):
            if redundant_boundary_action(state, action):
                masks[action] = False
        return masks

    def heuristic_action(self):
        if self._needs_reset:
            raise gym.error.ResetNeeded('Reset before querying actions')
        return self._bot.act(self.state)

    def _outcome(self, deaths):
        state = self.state
        if state.base_destroyed:
            return 'base_lost'
        if state.lives == 0 or (deaths and state.lives <= 1):
            return 'lives_exhausted'
        if state.remaining_enemies == 0 or state.stage != self.stage:
            return 'clear' if self.objective == 'full' else 'objective'
        if self.objective == 'combat' and state.kills >= 1:
            if self._combat_pickup_deadline is None:
                self._combat_pickup_deadline = state.frame + COMBAT_PICKUP_FRAMES
            if state.item is None or state.frame >= self._combat_pickup_deadline:
                return 'objective'
        if self.objective == 'defense' and state.frame >= DEFENSE_FRAMES \
                and state.kills >= 2:
            return 'objective'
        if state.frame >= self.max_frames:
            return 'timeout'
        return ''

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded('Episode ended; reset before stepping')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action!r}')
        action = int(action)
        before = self.state
        threatened = {b.generation for b in incoming_bullets(before)}
        primitive = action if action < 10 else self._bot.advise(MACROS[action - 10], before)
        if not 0 <= primitive < 10:
            raise RuntimeError(f'Heuristic returned invalid primitive {primitive}')
        mask = gs.ACTION_MASKS[primitive]
        redundant_boundary = redundant_boundary_action(before, primitive)
        changed_pose = False
        parts = dict(time=0.0, kill=0.0, death=0.0, clear=0.0, base_loss=0.0,
                     danger=0.0, guard=0.0, evasion=0.0, pickup=0.0,
                     base_fire=0.0, steel_fire=0.0, boundary_fire=0.0)
        deaths = 0
        outcome = ''
        for frame in range(self.frame_skip):
            pre = self.state
            self.nes.step(mask if frame < max(1, self.frame_skip // 2) else mask & ~1)
            post, events = self._tracker.advance(self.nes.ram)
            parts['time'] -= 0.002 / 4
            parts['kill'] += events.kills
            parts['death'] -= 3 * events.deaths
            parts['guard'] += events.kills * _danger(pre)
            deaths += events.deaths
            self._deaths += events.deaths
            self._shots += len(events.shots)
            self._pickups += len(events.pickups)
            self._item_drops += events.item_drops
            self._item_opportunities += events.item_drops
            for kind in events.pickups:
                parts['pickup'] += (0.7, 0.5, 0.5,
                                    1.0 if pre.player and pre.player.power < 3 else 0.5,
                                    0.5, 2.0)[kind]
            for shot in events.shots:
                obstacle = shot_obstacle(pre, shot.x, shot.y, shot.facing, shot.power)
                if obstacle == 'base':
                    parts['base_fire'] -= 0.5
                elif obstacle == 'steel' and shot.power < 3:
                    parts['steel_fire'] -= 0.05
                elif obstacle == 'boundary':
                    parts['boundary_fire'] -= 0.05
            moved = pre.player is not None and post.player is not None \
                and (pre.player.x, pre.player.y) != (post.player.x, post.player.y)
            turned = pre.player is not None and post.player is not None \
                and pre.player.facing != post.player.facing
            changed_pose = changed_pose or moved or turned
            if moved or events.kills or events.pickups:
                self._last_progress_frame = post.frame
            outcome = self._outcome(deaths)
            if outcome:
                break
        parts['danger'] = -0.05 * _danger(self.state)
        if outcome == 'base_lost':
            parts['base_loss'] = -20.0
        elif outcome == 'clear':
            parts['clear'] = 5.0
        elif outcome == 'objective':
            parts['clear'] = 1.0
        if threatened and not deaths and self.state.player is not None:
            remaining = {b.generation for b in incoming_bullets(self.state)}
            if threatened.isdisjoint(remaining):
                parts['evasion'] = 0.05
        if redundant_boundary and not changed_pose:
            elapsed = self.state.frame - before.frame
            self._boundary_waste_frames += elapsed
            self._boundary_waste_streak_frames += elapsed
            self._max_boundary_waste_frames = max(
                self._max_boundary_waste_frames, self._boundary_waste_streak_frames)
        else:
            self._boundary_waste_streak_frames = 0
        self._last_macro = action - 10 if action >= 10 else -1
        self._needs_reset = bool(outcome)
        if self.reward_profile == 'sparse':
            for key in parts.keys() - {'time', 'kill', 'death', 'clear', 'base_loss'}:
                parts[key] = 0.0
        return (self._observation(), float(sum(parts.values())),
                bool(outcome and outcome != 'timeout'), outcome == 'timeout',
                self._info(outcome, parts))

    def _info(self, outcome, components):
        state = self.state
        return dict(outcome=outcome, objective=self.objective, stage=self.stage,
                    kills=state.kills, lives=state.lives, frames=state.frame,
                    shots=self._shots, deaths=self._deaths, pickups=self._pickups,
                    item_drops=self._item_drops, item_opportunities=self._item_opportunities,
                    boundary_waste_frames=self._boundary_waste_frames,
                    boundary_waste_streak_frames=self._boundary_waste_streak_frames,
                    max_boundary_waste_frames=self._max_boundary_waste_frames,
                    stagnation_frames=state.frame - self._last_progress_frame,
                    reward_components=components, score=state.score)

    def _observation(self):
        state = self.state
        grid = np.zeros((OBS_CHANNELS, gs.GRID, gs.GRID), dtype=np.float32)
        for channel, material in enumerate((gs.T_BRICK, gs.T_STEEL | gs.T_UNKNOWN,
                                             gs.T_WATER, gs.T_ICE)):
            grid[channel] = (state.terrain & material) != 0
        grid[4, 24:26, 12:14] = 1
        if state.player:
            dx, dy = gs.DXY[state.player.facing]
            for r, c in gs.tank_cells(state.player.x, state.player.y):
                grid[5:8, r, c] = (1, dx, dy)
                grid[43:45, r, c] = ((state.player.x - gs.ARENA) % gs.CELL / 7,
                                     (state.player.y - gs.ARENA) % gs.CELL / 7)
        for enemy in state.enemies:
            dx, dy = gs.DXY[enemy.facing]
            for r, c in gs.tank_cells(enemy.x, enemy.y):
                grid[8, r, c] += 1 / 6
                grid[9, r, c] += dx
                grid[10, r, c] += dy
                grid[11, r, c] = max(grid[11, r, c], enemy.armor / 4)
                if enemy.drops_item:
                    grid[46, r, c] = 1
        for bullet in state.bullets:
            r, c = (bullet.y - gs.ARENA) // gs.CELL, (bullet.x - gs.ARENA) // gs.CELL
            if not (0 <= r < gs.GRID and 0 <= c < gs.GRID):
                continue
            if bullet.owner == 0:
                channel, count = 12, 0.5
            elif bullet.owner >= 2:
                channel, count = 15, 1 / 6
            else:
                continue
            grid[channel:channel + 3, r, c] += (count, bullet.vx / 4, bullet.vy / 4)
        scalars = (state.remaining_enemies / 20, state.spawn_left / 20,
                   state.lives / 10, state.player.power / 3 if state.player else 0,
                   state.shield, state.freeze, state.fire_ready)
        for channel, value in enumerate(scalars, 18):
            grid[channel].fill(value)
        if state.item:
            r, c = (state.item.y - gs.ARENA) // gs.CELL, (state.item.x - gs.ARENA) // gs.CELL
            if 0 <= r < gs.GRID and 0 <= c < gs.GRID:
                grid[25 + state.item.kind, r, c] = 1
        if self._last_macro >= 0:
            grid[31 + self._last_macro].fill(1)
        grid[36].fill(max(0, 1 - state.frame / self.max_frames))
        grid[37].fill(self.objective == 'combat')
        grid[38].fill(self.objective == 'defense')
        if state.brick_masks is None:
            raise RuntimeError('Canonical observations require residual brick masks')
        for channel, bit in enumerate((1, 2, 4, 8), 39):
            grid[channel] = (state.brick_masks & bit) != 0
        grid[45] = bullet_danger(state)
        np.clip(grid, -1, 1, out=grid)
        return grid

    def render(self):
        if self.render_mode == 'human':
            return self.nes.render(mode='human')
        return self.nes.screen.copy()

    def close(self):
        self.nes.close()
