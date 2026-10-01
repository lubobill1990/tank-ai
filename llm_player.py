"""LLM commander plays Battle City in the Rust engine.

The language model issues strategic orders (hunt / protect / item / goto /
hold) from a textual battlefield rendering every few in-game seconds; the
geometric HeuristicBot executes them between calls, bullet dodge/counter
overrides run automatically, and the MPC lookahead veto rejects fatal
primitives. Costs are metered from API usage and the episode aborts at the
configured budget. Rust outcomes are not original-ROM wins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
from heuristic import HeuristicBot
from mpc_teacher import MpcVetoTeacher, CachedParser, outcome_of
from rust_env import NativeBatch, episode_seed, stage_ram

ANTHROPIC_VERSION = '2023-06-01'
# USD per million tokens; cache reads billed at a tenth of input.
PRICES = {
    'claude-sonnet-5': (3.0, 15.0),
    'claude-haiku-4-5': (1.0, 5.0),
    'claude-opus-5': (15.0, 75.0),
}
TERRAIN_CHARS = {gs.T_EMPTY: '.', gs.T_BRICK: '#', gs.T_STEEL: '@',
                 gs.T_WATER: '~', gs.T_ICE: ',', gs.T_BASE: 'X',
                 gs.T_UNKNOWN: '?'}
FACING_NAMES = {gs.UP: 'up', gs.DOWN: 'down', gs.LEFT: 'left', gs.RIGHT: 'right'}

SYSTEM_PROMPT = """\
You command the player tank in NES Battle City (stage view is a 26x26 grid).
Destroy all 20 enemy tanks before losing all lives, and NEVER let any enemy
shot or tank reach the base X at the bottom center. Losing the base ends the
game immediately.

Map legend: '.' open  '#' brick wall  '@' steel (impassable)  '~' water
(blocks tanks, not bullets)  ',' ice  'X' your base  'P' you  'E' enemy tank
'A' armored enemy (needs several hits)  '*' enemy bullet  'o' your bullet
'$' power-up item. Row 0 is the top; enemies spawn at the top and push down
toward the base. Coordinates are given as (row,col).

You give one strategic order; a competent low-level controller executes it,
automatically dodging bullets and shooting enemies in its line of fire. Orders:
- {"mode":"hunt"}            chase and destroy the nearest enemies
- {"mode":"protect"}         fall back and guard the base approaches
- {"mode":"item"}            grab the power-up item
- {"mode":"goto","target":[row,col]}  move there, then HOLD that cell as a
  shield, shooting whatever lines up. Use this to body-block the base
  approaches: the base sits at rows 24-25, cols 12-13, so standing at
  (23,12), (24,10) or (24,15) physically blocks the usual attack lanes.
- {"mode":"hold"}            stay put, dodge and shoot anything lined up

Key judgement you are trusted with: when any enemy is closer to the base than
you are, or several enemies are pushing the bottom half, switch to protect
early - the controller's historical losses are almost all late base defence.
Hunt aggressively while spawns are far from the base. Items are valuable but
never worth losing the base.

Reply with ONLY the JSON order, optionally followed by one short sentence of
reasoning on a new line."""


def render_map(state):
    grid = [[TERRAIN_CHARS[int(v)] for v in row] for row in state.terrain]
    for bullet in state.bullets:
        r, c = (bullet.y - gs.ARENA) // gs.CELL, (bullet.x - gs.ARENA) // gs.CELL
        if 0 <= r < 26 and 0 <= c < 26:
            grid[r][c] = '*' if bullet.owner >= 2 else 'o'
    if state.item is not None:
        r, c = (state.item.y - gs.ARENA) // gs.CELL, (state.item.x - gs.ARENA) // gs.CELL
        if 0 <= r < 26 and 0 <= c < 26:
            grid[r][c] = '$'
    for enemy in state.enemies:
        r, c = enemy.tile
        if 0 <= r < 26 and 0 <= c < 26:
            grid[r][c] = 'A' if enemy.armor > 1 else 'E'
    if state.player is not None:
        r, c = state.player.tile
        if 0 <= r < 26 and 0 <= c < 26:
            grid[r][c] = 'P'
    header = '    ' + ''.join(str(c % 10) for c in range(26))
    rows = [f'{r:>3} ' + ''.join(grid[r]) for r in range(26)]
    return header + '\n' + '\n'.join(rows)


def render_state(state, stage, frame, events, current_mode):
    lines = [f'Stage {stage}, frame {frame} (~{frame // 60}s).']
    player = state.player
    if player is not None:
        pr, pc = player.tile
        lines.append(f'You: cell ({pr},{pc}), facing {FACING_NAMES[player.facing]}, '
                     f'power {player.power}, lives left {state.lives}.')
    else:
        lines.append(f'You are respawning. Lives left {state.lives}.')
    on_field = len(state.enemies)
    lines.append(f'Enemies: {state.remaining_enemies} total remain '
                 f'({on_field} on screen, {state.spawn_left} waiting to spawn).')
    for enemy in state.enemies:
        r, c = enemy.tile
        base_d = abs(r - 25) + abs(c - 12)
        tags = []
        if enemy.armor > 1:
            tags.append(f'armor {enemy.armor}')
        if enemy.drops_item:
            tags.append('carries item')
        tag = f" ({', '.join(tags)})" if tags else ''
        lines.append(f'  enemy at ({r},{c}) facing {FACING_NAMES[enemy.facing]}, '
                     f'{base_d} cells from base{tag}')
    threats = [b for b in state.bullets if b.owner >= 2]
    if threats:
        lines.append(f'{len(threats)} enemy bullet(s) in flight.')
    if state.item is not None:
        r = (state.item.y - gs.ARENA) // gs.CELL
        c = (state.item.x - gs.ARENA) // gs.CELL
        lines.append(f'Item on field at ({r},{c}): {gs.ITEM_NAMES[state.item.kind]}.')
    if state.freeze:
        lines.append('Enemies are FROZEN right now - free kills or repositioning.')
    if state.shield:
        lines.append('You are temporarily invincible.')
    lines.append(f'Current order: {current_mode}.')
    if events:
        lines.append('Since your last order: ' + '; '.join(events) + '.')
    lines.append('')
    lines.append(render_map(state))
    lines.append('')
    lines.append('Give the next order (JSON).')
    return '\n'.join(lines)


class Commander:
    def __init__(self, base_url, token, model, budget_usd, log_path):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.model = model
        self.budget = budget_usd
        self.cost = 0.0
        self.calls = 0
        self.log = open(log_path, 'a', encoding='utf-8')

    def order(self, state_text, state=None):
        payload = {
            'model': self.model, 'max_tokens': 300, 'temperature': 0.2,
            'system': [{'type': 'text', 'text': SYSTEM_PROMPT,
                        'cache_control': {'type': 'ephemeral'}}],
            'messages': [{'role': 'user', 'content': state_text}],
        }
        data = json.dumps(payload).encode()
        headers = {'content-type': 'application/json',
                   'anthropic-version': ANTHROPIC_VERSION,
                   'authorization': f'Bearer {self.token}',
                   'x-api-key': self.token}
        url = self.base_url + '/v1/messages'
        last_error = None
        body = None
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                break
            except urllib.error.HTTPError as error:
                last_error = error
                # devtunnel occasionally answers POSTs with a redirect blip;
                # follow an absolute Location once, otherwise back off and retry.
                location = error.headers.get('Location') if error.headers else None
                if error.code in (301, 302, 307, 308) and location \
                        and location.startswith('https://'):
                    url = location
                time.sleep(8 * (attempt + 1))
            except Exception as error:  # noqa: BLE001 - retry then surface
                last_error = error
                time.sleep(8 * (attempt + 1))
        if body is None:
            raise RuntimeError(f'LLM call failed after retries: {last_error}')
        usage = body.get('usage', {})
        prices = PRICES.get(self.model, (3.0, 15.0))
        self.cost += (usage.get('input_tokens', 0) * prices[0]
                      + usage.get('cache_read_input_tokens', 0) * prices[0] * 0.1
                      + usage.get('cache_creation_input_tokens', 0) * prices[0] * 1.25
                      + usage.get('output_tokens', 0) * prices[1]) / 1e6
        self.calls += 1
        text = ''.join(block.get('text', '') for block in body.get('content', []))
        order = self._parse(text)
        self.log.write(json.dumps({'call': self.calls, 'cost_usd': round(self.cost, 4),
                                   'prompt': state_text, 'reply': text,
                                   'order': order, 'usage': usage},
                                  ensure_ascii=False) + '\n')
        self.log.flush()
        return order, text

    @staticmethod
    def _parse(text):
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            return {'mode': 'protect'}
        try:
            order = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {'mode': 'protect'}
        mode = order.get('mode')
        if mode not in ('hunt', 'protect', 'item', 'goto', 'hold'):
            return {'mode': 'protect'}
        if mode == 'goto':
            target = order.get('target')
            if (not isinstance(target, (list, tuple)) or len(target) != 2
                    or not all(isinstance(v, (int, float)) for v in target)):
                return {'mode': 'protect'}
            order['target'] = [int(target[0]), int(target[1])]
        return order


def aim_without_moving(bot, state):
    """Fire at any lined-up enemy from the current position; never chase."""
    player = state.player
    if player is None:
        return 0
    for enemy in sorted(state.enemies,
                        key=lambda e: abs(e.x - player.x) + abs(e.y - player.y)):
        action = bot._aim(state, enemy)
        if action:
            return action
    return 0


def execute_order(bot, state, order):
    """Primitive action for the current strategic order; 0 when satisfied."""
    if state.player is None or state.game_over:
        return 0
    if any(b.owner >= 2 for b in state.bullets):
        action = bot.advise('dodge', state) or bot.advise('counter', state)
        if action:
            return action
    mode = order['mode']
    if mode == 'hold':
        return bot.advise('counter', state) or aim_without_moving(bot, state)
    if mode == 'item' and state.item is not None:
        return bot.advise('item', state)
    if mode == 'goto':
        row, col = order['target']
        x = min(max(gs.ARENA + col * gs.CELL + 4, gs.ARENA + 8),
                gs.ARENA + gs.GRID * gs.CELL - 8)
        y = min(max(gs.ARENA + row * gs.CELL + 4, gs.ARENA + 8),
                gs.ARENA + gs.GRID * gs.CELL - 8)
        action = bot._navigate('item', state, (x, y))
        if action:
            return action
        # Arrived: hold the position as a shield, shooting what approaches.
        return bot.advise('counter', state) or aim_without_moving(bot, state)
    if mode == 'protect':
        return bot.advise('protect', state)
    return bot.advise('hunt', state)


def play_episode(stage, *, seed, commander, llm_every=30, max_calls=80,
                 max_frames=12000, depth=12, plan_seeds=3, use_veto=True):
    ram0 = stage_ram(stage, seed=0)
    batch = NativeBatch(ram0[None], max_frames, seeds=[seed])
    batch.observe()
    bot = HeuristicBot()
    parser = CachedParser()
    planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                             plan_rng_seed=seed & 0xffffffff) if use_veto else None
    order = {'mode': 'hunt'}
    events = []
    prev = {'kills': 0, 'lives': None, 'item': None}
    outcome = 'incomplete'
    stats_final = {}
    decisions_since_call = llm_every  # consult immediately at frame 0
    try:
        for step in range(max_frames // 4 + 8):
            ram = batch.ram(0)
            state = parser.parse(ram)
            if prev['lives'] is None:
                prev['lives'] = state.lives
            if state.kills > prev['kills']:
                events.append(f"destroyed {state.kills - prev['kills']} enemy tank(s)")
                prev['kills'] = state.kills
            if state.lives < prev['lives']:
                events.append('you were destroyed and respawned (lost a life)')
                prev['lives'] = state.lives
                decisions_since_call = llm_every
            item_now = None if state.item is None else (state.item.x, state.item.y)
            if item_now != prev['item'] and item_now is not None:
                events.append('a power-up item appeared')
                decisions_since_call = llm_every
            prev['item'] = item_now
            if decisions_since_call >= llm_every and commander.calls < max_calls \
                    and commander.cost < commander.budget and state.player is not None:
                text = render_state(state, stage, step * 4, events, order['mode'])
                order, _ = commander.order(text)
                events = []
                decisions_since_call = 0
            decisions_since_call += 1
            action = execute_order(bot, state, order)
            masks = batch.masks()[0].copy()
            if not masks[action]:
                action = 0
            if planner is not None:
                scores, unsafe = planner.plan(ram[None])
                action = int(planner.choose(scores, unsafe, [masks], [action])[0])
            _, _, dones, stats = batch.step(np.array([action]))
            if dones[0]:
                final_ram = batch.ram(0)
                outcome = outcome_of(final_ram)
                frame, kills, deaths, shots, pickups = map(int, stats[0])
                stats_final = {'frames': frame, 'kills': kills, 'deaths': deaths,
                               'shots': shots, 'pickups': pickups}
                break
    finally:
        batch.close()
    return {'stage': stage, 'seed': seed, 'outcome': outcome, **stats_final,
            'llm_calls': commander.calls, 'cost_usd': round(commander.cost, 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--base-seed', type=int, default=90000)
    parser.add_argument('--model', default='claude-sonnet-5')
    parser.add_argument('--base-url',
                        default='https://pkzzhbqc-23461.jpe1.devtunnels.ms')
    parser.add_argument('--token', default='local-dev-key')
    parser.add_argument('--budget-usd', type=float, default=20.0)
    parser.add_argument('--llm-every', type=int, default=30,
                        help='decisions (4 frames each) between scheduled calls')
    parser.add_argument('--max-calls', type=int, default=80)
    parser.add_argument('--no-veto', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for episode in range(args.episodes):
        commander = Commander(args.base_url, args.token, args.model,
                              args.budget_usd, args.output / 'transcript.jsonl')
        result = play_episode(
            args.stage, seed=episode_seed(args.base_seed, args.stage, episode),
            commander=commander, llm_every=args.llm_every,
            max_calls=args.max_calls, use_veto=not args.no_veto)
        results.append(result)
        print(json.dumps(result), flush=True)
        commander.log.close()
    summary = {
        'model': args.model, 'stage': args.stage,
        'clears': sum(r['outcome'] == 'clear' for r in results),
        'episodes': len(results),
        'total_cost_usd': round(sum(r['cost_usd'] for r in results), 4),
        'results': results,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2),
                                              encoding='utf-8')
    print(json.dumps({k: summary[k] for k in
                      ('model', 'stage', 'clears', 'episodes', 'total_cost_usd')}))


if __name__ == '__main__':
    main()
