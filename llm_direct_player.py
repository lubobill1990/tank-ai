"""Pure LLM plays Battle City: the model IS the controller.

Unlike llm_player.py (where the LLM issues strategic orders and a geometric
HeuristicBot + MPC veto do the actual driving), here the language model picks a
primitive joypad action (0-9) directly from a textual board every query. There
is no geometric executor and, by default, no safety veto - the tank does
exactly and only what the model says. This measures whether Claude can play the
game itself, not steer a hand-coded pilot.

The model may attach a "hold" count to repeat its action for a few decisions,
which lets it self-pace and bounds API cost/latency (a stage-1 clear is ~1000
decisions at 4 frames each). An optional --safety-veto adds the MPC fatal-move
filter back, purely to quantify how much a safety net would buy.

Rust engine outcomes are not original-ROM wins, but stage-1 geometry/enemy
behaviour matches the ROM closely enough for a feasibility read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
from llm_player import Commander, render_map, FACING_NAMES, TERRAIN_CHARS
from mpc_teacher import MpcVetoTeacher, CachedParser, outcome_of
from rust_env import NativeBatch, episode_seed, stage_ram

# action id -> human label; matches the Rust NativeBatch 10-action space.
ACTION_LABELS = {
    0: 'wait (no move, no fire)',
    1: 'move/face UP',
    2: 'move/face DOWN',
    3: 'move/face LEFT',
    4: 'move/face RIGHT',
    5: 'FIRE in the direction you currently face',
    6: 'turn to face UP and FIRE',
    7: 'turn to face DOWN and FIRE',
    8: 'turn to face LEFT and FIRE',
    9: 'turn to face RIGHT and FIRE',
}

DIRECT_SYSTEM = """\
You are directly driving the player tank in NES Battle City, one joypad action
at a time. The arena is a 26x26 grid of cells shown to you as text each turn.
You win by destroying all 20 enemy tanks before you run out of lives. You lose
instantly if an enemy tank or an enemy bullet reaches your base (marked X at
the bottom centre, rows 24-25 cols 12-13), so defending it always outranks
grabbing items or chasing kills.

You output ONE primitive action per turn. The ten actions are:
  0 = wait (do nothing)
  1 = move/face UP        2 = move/face DOWN
  3 = move/face LEFT      4 = move/face RIGHT
  5 = FIRE straight ahead (in the direction you currently face)
  6 = face UP and fire    7 = face DOWN and fire
  8 = face LEFT and fire  9 = face RIGHT and fire

Movement rules: actions 1-4 move you one step AND turn you to face that way; a
tank only moves in the direction it faces. To shoot an enemy you must face it
(its row or column must line up with yours, nothing solid between) and fire -
use 6/7/8/9 to aim-and-shoot in one action, or 5 if you already face the
target. Only ONE of your bullets exists at a time: if the live state says your
cannon is RELOADING, firing does nothing - move instead. Enemy bullets kill you
in one hit - if one shares your row (moving left/right) or your column (moving
up/down), step sideways off that line immediately. '#' brick is destructible by
shots and blocks movement; '@' steel and '~' water block movement; ',' ice
makes you slide.

HOW YOU SEE THE WORLD each turn:
- A STATIC STAGE MAP (terrain only) is given once and stays fixed - use it to
  understand the overall layout, lanes to the base, and where walls are.
- A LIVE STATE block gives the fresh, authoritative positions every turn: your
  cell and facing, each enemy's position described RELATIVE to you (e.g. "3 UP
  and 2 LEFT"), incoming bullets, the item, and whether your cannon is ready.
- A 9x9 LOCAL VIEW centred on you (P) shows your immediate surroundings live;
  '%' marks the arena edge. Trust the LIVE STATE and LOCAL VIEW over the static
  map when they disagree (walls you shot away, tanks that moved).
- "Your last moves" lists what you just did - if you have been repeating a move
  (e.g. left,right,left,right) without progress, you are stuck oscillating:
  change plan, pick a concrete target enemy and drive/align to it.

Priorities, in order: (1) survive - dodge any bullet on your line; (2) keep the
base safe - intercept any enemy in the bottom rows or heading for the base;
(3) hunt - align with an enemy and fire to kill all 20. Do not wander.

Answer FORMAT - strict, exactly these five lines in this order and nothing else:
PLAN: <your MACRO goal; a durable multi-turn intention, e.g. "hold the base
  column 12 until enemies thin out, then hunt top enemies". KEEP the same PLAN
  across turns unless the battlefield clearly changed - do not flip-flop.>
SUBGOAL: <your MICRO goal right now; a concrete next step, e.g. "reach column 9
  then turn up to line up the enemy at row 5">
ASSESS: <one-line read of the battlefield: biggest threat, is the base safe>
ACTION: <n>  HOLD: <h>   (n is 0-9; h is 1-6 turns to repeat before re-looking:
  larger h for long straight travel in open space, h=1 when enemies/bullets are
  near and you must react every step)
WHY: <one short clause: why THIS action serves the SUBGOAL/PLAN now>

Keep every line to one sentence. Your PLAN and SUBGOAL from last turn are shown
back to you - stay consistent with them unless you justify a change in ASSESS."""


def _live_grid(state):
    """Full 26x26 char grid with live terrain, tanks, bullets and item."""
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
    return grid


def render_terrain_static(state):
    """Terrain-only global map, fixed for the whole stage -> cached once.

    Only brick ('#') is destructible; steel/water/base/ice never change. Live
    walls near the player are always shown accurately in the egocentric view,
    so minor brick staleness here (a wall the player already shot away) does not
    mislead local decisions while keeping this block byte-identical for caching.
    """
    grid = [[TERRAIN_CHARS[int(v)] for v in row] for row in state.terrain]
    header = '      ' + ''.join(str(c % 10) for c in range(26))
    rows = [f'row {r:>2} ' + ''.join(grid[r]) for r in range(26)]
    legend = ("Legend: '.'=open  '#'=brick(destructible, blocks)  '@'=steel"
              "(blocks)  '~'=water(blocks tanks)  ','=ice(slippery)  'X'=base")
    return ('STATIC STAGE MAP (terrain only; row 0 top, col 0 left; your base X '
            'is at rows 24-25, cols 12-13):\n' + header + '\n'
            + '\n'.join(rows) + '\n' + legend)


def _rel(dr, dc):
    """Human egocentric phrase for a (drow, dcol) offset from the player."""
    parts = []
    if dr < 0:
        parts.append(f'{-dr} UP')
    elif dr > 0:
        parts.append(f'{dr} DOWN')
    if dc < 0:
        parts.append(f'{-dc} LEFT')
    elif dc > 0:
        parts.append(f'{dc} RIGHT')
    return ' and '.join(parts) if parts else 'same cell'


def render_ego(state, radius=4):
    """Local egocentric grid centred on the player (live entities + walls)."""
    if state.player is None:
        return '(respawning - no local view)'
    grid = _live_grid(state)
    pr, pc = state.player.tile
    rows = []
    for r in range(pr - radius, pr + radius + 1):
        cells = []
        for c in range(pc - radius, pc + radius + 1):
            if 0 <= r < 26 and 0 <= c < 26:
                cells.append(grid[r][c])
            else:
                cells.append('%')  # outside the arena wall
        rows.append(''.join(cells))
    return '\n'.join(rows)


def render_dynamic(state, stage, frame, history, kills, plan='', subgoal=''):
    """Small, fast-changing block: live positions, threats, your recent moves.

    Everything static (rules, terrain) lives in cached blocks; only this delta
    is fresh input each turn, so prompt caching covers the bulk of the tokens.
    The standing PLAN/SUBGOAL from last turn are echoed back for consistency.
    """
    lines = [f'=== LIVE STATE  stage {stage}, frame {frame} (~{frame//60}s) ===']
    if plan:
        lines.append(f'Your standing PLAN (from last turn): {plan}')
    if subgoal:
        lines.append(f'Your standing SUBGOAL (from last turn): {subgoal}')
    player = state.player
    if player is not None:
        pr, pc = player.tile
        lines.append(f'You (P) at (row {pr}, col {pc}) facing '
                     f'{FACING_NAMES[player.facing].upper()}. Lives {state.lives}, '
                     f'power {player.power}. Enemies killed {kills}/20.')
    else:
        lines.append(f'You are respawning. Lives {state.lives}. Killed {kills}/20.')
    own = [b for b in state.bullets if b.owner < 2]
    if own:
        b = own[0]
        r, c = (b.y - gs.ARENA) // gs.CELL, (b.x - gs.ARENA) // gs.CELL
        lines.append(f'Cannon RELOADING: your bullet (o) still flying at '
                     f'(row {r}, col {c}). Firing now is WASTED - move or reposition.')
    elif state.fire_ready:
        lines.append('Cannon READY to fire.')
    lines.append(f'{state.remaining_enemies} enemies left to kill '
                 f'({len(state.enemies)} on screen, {state.spawn_left} not spawned yet).')
    if player is not None:
        for enemy in state.enemies:
            r, c = enemy.tile
            dr, dc = r - player.tile[0], c - player.tile[1]
            aligned = ''
            if dr == 0:
                aligned = ' [SAME ROW as you -> face LEFT/RIGHT and fire]'
            elif dc == 0:
                aligned = ' [SAME COLUMN as you -> face UP/DOWN and fire]'
            tag = ' ARMORED' if enemy.armor > 1 else ''
            carry = ' carries-item' if enemy.drops_item else ''
            base_d = abs(r - 25) + abs(c - 12)
            lines.append(f'  enemy{tag}{carry} at (row {r},col {c}) = {_rel(dr, dc)} '
                         f'from you, facing {FACING_NAMES[enemy.facing]}, '
                         f'{base_d} from base{aligned}.')
    threats = [b for b in state.bullets if b.owner >= 2]
    for b in threats:
        r, c = (b.y - gs.ARENA) // gs.CELL, (b.x - gs.ARENA) // gs.CELL
        moving = ('right' if b.vx > 0 else 'left') if abs(b.vx) >= abs(b.vy) \
            else ('down' if b.vy > 0 else 'up')
        danger = ''
        if player is not None:
            dr, dc = r - player.tile[0], c - player.tile[1]
            if (dr == 0 and moving in ('left', 'right')) or \
               (dc == 0 and moving in ('up', 'down')):
                danger = ' [ON YOUR LINE - DODGE sideways NOW]'
            lines.append(f'  !! enemy bullet at (row {r},col {c}) = {_rel(dr, dc)}, '
                         f'moving {moving}{danger}.')
        else:
            lines.append(f'  !! enemy bullet at (row {r},col {c}) moving {moving}.')
    if state.item is not None:
        r = (state.item.y - gs.ARENA) // gs.CELL
        c = (state.item.x - gs.ARENA) // gs.CELL
        lines.append(f'Power-up $ at (row {r},col {c}): {gs.ITEM_NAMES[state.item.kind]}.')
    if state.freeze:
        lines.append('Enemies are FROZEN now - free hits.')
    if state.shield:
        lines.append('You are briefly invincible.')
    if history:
        lines.append('Your last moves (oldest->newest): ' + ', '.join(history) + '.')
    lines.append('Local view, 9x9 centred on you (P). E/A=enemy  *=enemy bullet  '
                 "o=your bullet  $=item  #brick @steel ~water ,ice X=base  .=open  "
                 '%=arena edge:')
    lines.append(render_ego(state, radius=4))
    lines.append('Pick ONE action now. Reply MUST start with: ACTION: <0-9>  HOLD: <1-6>')
    return '\n'.join(lines)


class DirectPilot(Commander):
    """Reuses Commander's HTTP/retry/cost metering; direct-action prompt+parse.

    Representation v2: the big static context (rules + stage terrain map) is sent
    as cached prefix blocks, so only the small live-state delta is fresh input
    each turn; the pilot also carries the player's recent action history so the
    model is aware of its own sequence and can break oscillation loops.
    """

    def start_episode(self, state):
        self._static_terrain = render_terrain_static(state)
        from collections import deque
        self._history = deque(maxlen=10)
        self._plan = ''
        self._subgoal = ''
        self.cache_read = 0
        self.cache_write = 0
        self.fresh_in = 0

    def remember(self, action, hold):
        label = {0: 'wait', 1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'fire',
                 6: 'fire-up', 7: 'fire-down', 8: 'fire-left',
                 9: 'fire-right'}.get(action, str(action))
        if hold > 1:
            label += f'x{hold}'
        self._history.append(label)

    def act(self, dynamic_text, include_terrain=True):
        # Blocks carry cache_control (system + terrain) in case the gateway ever
        # honours prompt caching; it currently bills every token fresh, so the
        # global terrain map is only re-sent periodically (the live ego view in
        # the dynamic block carries current local walls every turn). Prefill
        # "PLAN:" so the reply is the structured goal/assessment/action block
        # (macro+micro goals, reason) in a fixed, parseable order.
        content = []
        if include_terrain:
            content.append({'type': 'text',
                            'text': getattr(self, '_static_terrain', ''),
                            'cache_control': {'type': 'ephemeral'}})
        content.append({'type': 'text', 'text': dynamic_text})
        payload = {
            'model': self.model, 'max_tokens': 170,
            'system': [{'type': 'text', 'text': DIRECT_SYSTEM,
                        'cache_control': {'type': 'ephemeral'}}],
            'messages': [
                {'role': 'user', 'content': content},
                {'role': 'assistant', 'content': 'PLAN:'},
            ],
        }
        if 'opus' not in self.model:
            payload['temperature'] = 0.2
        reply = 'PLAN:' + self._post(payload, dynamic_text)
        action, hold = self._parse_action(reply)
        self._plan = self._extract(reply, 'PLAN') or getattr(self, '_plan', '')
        self._subgoal = self._extract(reply, 'SUBGOAL') or getattr(self, '_subgoal', '')
        return (action, hold), reply

    @staticmethod
    def _extract(text, field):
        m = re.search(rf'{field}:\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else ''

    def _post(self, payload, state_text):
        import urllib.request
        import urllib.error
        from llm_player import ANTHROPIC_VERSION
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
        prices = __import__('llm_player').PRICES.get(self.model, (3.0, 15.0))
        self.cost += (usage.get('input_tokens', 0) * prices[0]
                      + usage.get('cache_read_input_tokens', 0) * prices[0] * 0.1
                      + usage.get('cache_creation_input_tokens', 0) * prices[0] * 1.25
                      + usage.get('output_tokens', 0) * prices[1]) / 1e6
        self.cache_read = getattr(self, 'cache_read', 0) + usage.get('cache_read_input_tokens', 0)
        self.cache_write = getattr(self, 'cache_write', 0) + usage.get('cache_creation_input_tokens', 0)
        self.fresh_in = getattr(self, 'fresh_in', 0) + usage.get('input_tokens', 0)
        self.calls += 1
        text = ''.join(block.get('text', '') for block in body.get('content', []))
        return text

    @staticmethod
    def _parse_action(text):
        """Return (action 0-9, hold 1-6). Tolerant of format drift."""
        action, hold = 0, 1
        m = re.search(r'ACTION:\s*([0-9])', text, re.IGNORECASE)
        if m:
            action = int(m.group(1))
        else:  # fall back to the last bare digit 0-9 in the reply
            digits = re.findall(r'\b([0-9])\b', text)
            if digits:
                action = int(digits[-1])
        h = re.search(r'HOLD:\s*([1-9])', text, re.IGNORECASE)
        if h:
            hold = max(1, min(6, int(h.group(1))))
        return action, hold


def play_direct_episode(stage, *, seed, pilot, max_hold=6, max_calls=4000,
                        max_frames=12000, use_veto=False, depth=12, plan_seeds=3,
                        verbose=False):
    ram0 = stage_ram(stage, seed=0)
    batch = NativeBatch(ram0[None], max_frames, seeds=[seed])
    batch.observe()
    parser = CachedParser()
    planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                             plan_rng_seed=seed & 0xffffffff) if use_veto else None
    outcome = 'incomplete'
    stats_final = {}
    vetoes = 0
    action, hold_left = 0, 0
    pilot.start_episode(parser.parse(batch.ram(0)))
    try:
        for step in range(max_frames // 4 + 8):
            ram = batch.ram(0)
            state = parser.parse(ram)
            kills = int(state.kills)
            # Re-query the model when the hold expires, or immediately if an enemy
            # bullet now shares the player's row/column (a new lethal threat the
            # held action may not be dodging), as long as we have budget left.
            urgent = state.player is not None and any(
                b.owner >= 2 and ((b.y - gs.ARENA) // gs.CELL == state.player.tile[0]
                                  or (b.x - gs.ARENA) // gs.CELL == state.player.tile[1])
                for b in state.bullets)
            need = hold_left <= 0 or urgent
            if need and pilot.calls < max_calls and pilot.cost < pilot.budget \
                    and state.player is not None:
                text = render_dynamic(state, stage, step * 4,
                                      list(getattr(pilot, '_history', [])), kills,
                                      plan=getattr(pilot, '_plan', ''),
                                      subgoal=getattr(pilot, '_subgoal', ''))
                (action, hold), reply = pilot.act(
                    text, include_terrain=(pilot.calls % 25 == 0))
                pilot.remember(action, hold)
                hold_left = min(hold, max_hold)
                pilot.log.write(json.dumps(
                    {'call': pilot.calls, 'step': step, 'cost_usd': round(pilot.cost, 4),
                     'kills': kills, 'action': action, 'hold': hold_left,
                     'cache_read': pilot.cache_read, 'fresh_in': pilot.fresh_in,
                     'reply': reply}, ensure_ascii=False) + '\n')
                pilot.log.flush()
                if verbose:
                    print(f'  step {step}: kills={kills} action={action} '
                          f'hold={hold_left} cost=${pilot.cost:.3f} '
                          f'cache_read={pilot.cache_read} fresh={pilot.fresh_in}',
                          flush=True)
            hold_left -= 1
            masks = batch.masks()[0].copy()
            chosen = action if masks[action] else 0
            if planner is not None:
                scores, unsafe = planner.plan(ram[None])
                safe = int(planner.choose(scores, unsafe, [masks], [chosen])[0])
                if safe != chosen:
                    vetoes += 1
                chosen = safe
            _, _, dones, stats = batch.step(np.array([chosen]))
            if dones[0]:
                final_ram = batch.ram(0)
                outcome = outcome_of(final_ram)
                frame, kills, deaths, shots, pickups = map(int, stats[0])
                stats_final = {'frames': frame, 'kills': kills, 'deaths': deaths,
                               'shots': shots, 'pickups': pickups}
                break
            if pilot.cost >= pilot.budget:
                outcome = 'budget_exhausted'
                break
    finally:
        batch.close()
    return {'stage': stage, 'seed': seed, 'outcome': outcome, **stats_final,
            'llm_calls': pilot.calls, 'vetoes': vetoes,
            'cost_usd': round(pilot.cost, 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--base-seed', type=int, default=90000)
    parser.add_argument('--model', default='claude-haiku-4-5')
    parser.add_argument('--base-url',
                        default='https://pkzzhbqc-23461.jpe1.devtunnels.ms')
    parser.add_argument('--token', default='local-dev-key')
    parser.add_argument('--budget-usd', type=float, default=20.0)
    parser.add_argument('--max-hold', type=int, default=6)
    parser.add_argument('--max-calls', type=int, default=4000)
    parser.add_argument('--safety-veto', action='store_true',
                        help='add the MPC fatal-move filter back (not pure LLM)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for episode in range(args.episodes):
        pilot = DirectPilot(args.base_url, args.token, args.model,
                            args.budget_usd, args.output / 'transcript.jsonl')
        result = play_direct_episode(
            args.stage, seed=episode_seed(args.base_seed, args.stage, episode),
            pilot=pilot, max_hold=args.max_hold, max_calls=args.max_calls,
            use_veto=args.safety_veto, verbose=args.verbose)
        results.append(result)
        print(json.dumps(result), flush=True)
        pilot.log.close()
    summary = {
        'model': args.model, 'stage': args.stage, 'pure_llm': not args.safety_veto,
        'clears': sum(r['outcome'] == 'clear' for r in results),
        'episodes': len(results),
        'total_cost_usd': round(sum(r['cost_usd'] for r in results), 4),
        'results': results,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2),
                                              encoding='utf-8')
    print(json.dumps({k: summary[k] for k in
                      ('model', 'stage', 'pure_llm', 'clears', 'episodes',
                       'total_cost_usd')}))


if __name__ == '__main__':
    main()
