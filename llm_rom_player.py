"""LLM commander plays Battle City on the original ROM, with MP4 capture.

Same strategic-order loop as llm_player, but the environment is the real
emulator (ordinary resets, 12,000-frame allowance, no RAM writes) and every
episode is recorded to its own silent MP4 at 15 fps (one frame per 4-frame
decision; nes-py exposes no audio). The MPC veto imports each ROM RAM
snapshot into the Rust engine for lookahead scoring and falls back to the
raw order when a state cannot be represented natively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
from heuristic import HeuristicBot
from learning_lab.demonstrations import DemonstrationWriter
from llm_player import Commander, execute_order, render_state
from mpc_teacher import CachedParser, MpcVetoTeacher

BASE_X, BASE_Y = 120, 216


class Writer55(DemonstrationWriter):
    """DemonstrationWriter variant storing 55-channel precision frames."""

    def __init__(self, path):
        super().__init__(path)
        from native_codec import NativeRunCodec
        self.codec = NativeRunCodec((55, 26, 26))

    def finish(self, episode_reports, metadata):
        result = super().finish(episode_reports, metadata)
        meta_path = self.path / 'metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['frame_shape'] = [55, 26, 26]
        meta['format'] = 'native-run-demonstrations-55-v1'
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return result


def strategy_features(state):
    """Compact numeric summary of a decision state for strategist training."""
    player = state.player
    features = {
        'player': None if player is None else
            [*player.tile, int(player.facing), int(player.power)],
        'lives': int(state.lives), 'remaining': int(state.remaining_enemies),
        'spawn_left': int(state.spawn_left), 'kills': int(state.kills),
        'fire_ready': bool(state.fire_ready), 'freeze': bool(state.freeze),
        'shield': bool(state.shield),
        'enemy_bullets': sum(b.owner >= 2 for b in state.bullets),
        'item': None if state.item is None else
            [int((state.item.y - gs.ARENA) // gs.CELL),
             int((state.item.x - gs.ARENA) // gs.CELL), int(state.item.kind)],
        'enemies': [[*e.tile, int(e.facing), int(e.armor), int(e.drops_item)]
                    for e in state.enemies],
    }
    if player is not None:
        features['player_base_dist'] = abs(player.x - BASE_X) + abs(player.y - BASE_Y)
    if state.enemies:
        features['enemy_base_dist'] = min(
            abs(e.x - BASE_X) + abs(e.y - BASE_Y) for e in state.enemies)
    return features


class EpisodeVideo:
    def __init__(self, path, fps=15, scale=2):
        self.path = Path(path)
        size = f'{256 * scale}:{240 * scale}'
        self.process = subprocess.Popen(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '256x240',
             '-r', str(fps), '-i', '-',
             '-vf', f'scale={size}:flags=neighbor',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path)],
            stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=120)


class ScheduleCommander:
    """Per-stage scripted strategy compiled from LLM gameplay: no network, no NN.

    phase 1: the given opening order until remaining enemies <= switch_at;
    phase 2: hunt. Dodge/counter overrides and the MPC veto still apply.
    """

    def __init__(self, opening, switch_at, target=None):
        self.opening = opening
        self.switch_at = switch_at
        self.target = target
        self.calls = 0
        self.cost = 0.0
        self.budget = float('inf')
        self.model = f'schedule:{opening}@{switch_at}'

    def order(self, state_text, state=None):
        self.calls += 1
        if state is not None and state.remaining_enemies <= self.switch_at:
            return {'mode': 'hunt'}, 'phase2'
        if self.opening == 'anchor':
            return {'mode': 'anchor'}, 'phase1'
        order = {'mode': self.opening}
        if self.opening == 'goto' and self.target:
            order['target'] = list(self.target)
        return order, 'phase1'


def play_rom_episode(stage, *, seed, commander, output_dir, episode_tag,
                     llm_every=30, max_calls=80, max_frames=12000,
                     depth=12, plan_seeds=3, use_veto=True,
                     initial_mode='hunt', record_demos=False, record_video=True,
                     guard_bot=False, demo_channels='47', controller='reflex-v1'):
    from stable_baselines3.common.utils import set_random_seed
    from rust_train import ROMPrimitiveEnv
    fsm = None
    if controller == 'fsm-v2':
        from state_machine import SequencePlanner, StatefulController
        planner = SequencePlanner(depth=depth, plan_seeds=plan_seeds,
                                  plan_rng_seed=seed & 0xffffffff) if use_veto else None
        fsm = StatefulController(planner)
    elif controller == 'reflex-seq':
        from state_machine import SequencePlanner
        planner = SequencePlanner(depth=depth, plan_seeds=plan_seeds,
                                  plan_rng_seed=seed & 0xffffffff) if use_veto else None
    else:
        planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                                 plan_rng_seed=seed & 0xffffffff) if use_veto else None
    env = ROMPrimitiveEnv(stage, max_frames)
    raw = env.env
    if guard_bot or fsm is not None or controller == 'reflex-seq':
        from base_guard import BaseGuardBot, set_ice_penalty
        bot = BaseGuardBot()
    else:
        bot = HeuristicBot()
        set_ice_penalty = None
    parser = CachedParser()
    order = {'mode': initial_mode}
    events = []
    prev = {'kills': 0, 'lives': None, 'item': None}
    decisions_since_call = llm_every
    outcome, info = 'unfinished', {}
    partial = Path(output_dir) / f'.partial-{episode_tag}.mp4'
    video = EpisodeVideo(partial) if record_video else None
    writer = writer55 = None
    if record_demos:
        if demo_channels in ('47', 'both'):
            writer = DemonstrationWriter(Path(output_dir) / 'demos' / episode_tag)
        if demo_channels in ('55', 'both'):
            from within_stage.observation import encode_precision
            writer55 = Writer55(Path(output_dir) / 'demos55' / episode_tag)
    strategy_rows = []
    veto_failures = 0
    started = time.perf_counter()
    try:
        set_random_seed(seed)
        observation, _ = env.reset(seed=seed)
        if set_ice_penalty is not None:
            ice_cells = int((raw.state.terrain == gs.T_ICE).sum())
            set_ice_penalty(8.0 if ice_cells > 80 else 0.0)
        if video is not None:
            video.write(raw.nes.screen)
        episode_start = True
        last_action = None
        for step in range(max_frames // 4 + 4):
            ram = np.array(raw.nes.ram, dtype=np.uint8, copy=True)
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
                order, _ = commander.order(text, state=state)
                strategy_rows.append({'frame': step * 4, 'stage': stage,
                                      'features': strategy_features(state),
                                      'order': order})
                events = []
                decisions_since_call = 0
            decisions_since_call += 1
            masks = np.asarray(raw.action_masks()[:10], dtype=bool)
            if guard_bot or fsm is not None or controller == 'reflex-seq':
                from base_guard import restrict_masks
                masks = restrict_masks(state, masks)
            if fsm is not None:
                action = fsm.decide(state, masks, ram)
            else:
                action = execute_order(bot, state, order)
                if not masks[action]:
                    action = 0
                if planner is not None:
                    try:
                        scores, unsafe = planner.plan(ram[None])
                        action = int(planner.choose(scores, unsafe, [masks],
                                                    [action], [last_action])[0])
                    except Exception:
                        veto_failures += 1
                last_action = action
            if writer is not None or writer55 is not None:
                frame47 = np.asarray(observation, np.float32)
                if writer is not None:
                    writer.add(frame47, int(action), masks,
                               episode_start=episode_start,
                               executed_action=int(action))
                if writer55 is not None:
                    writer55.add(encode_precision(state, frame47), int(action),
                                 masks, episode_start=episode_start,
                                 executed_action=int(action))
                episode_start = False
            observation, _, terminated, truncated, step_info = env.step(int(action))
            if video is not None:
                video.write(raw.nes.screen)
            if terminated or truncated:
                info = step_info
                outcome = step_info.get('outcome', 'unknown')
                break
    finally:
        if video is not None:
            video.close()
    demo_report = [{'episode_id': 0, 'outcome': outcome, 'stage': stage,
                    'kills': info.get('kills'), 'deaths': info.get('deaths'),
                    'frames': info.get('frames'), 'seed': seed}]
    demo_meta = {'collector': 'llm-commander-rom-v1', 'stage': stage,
                 'stream': 0, 'base_seed': seed,
                 'model': getattr(commander, 'model', 'schedule'),
                 'initial_mode': initial_mode, 'llm_every': llm_every,
                 'noise': 0.0, 'beta': 1.0,
                 'backend': 'original-rom-llm-commander'}
    if writer is not None:
        writer.finish(demo_report, demo_meta)
    if writer55 is not None:
        writer55.finish(demo_report, dict(demo_meta, observation='precision55'))
    if strategy_rows:
        with (Path(output_dir) / 'strategy.jsonl').open('a', encoding='utf-8') as f:
            for row in strategy_rows:
                row['outcome'] = outcome
                f.write(json.dumps(row) + '\n')
    if video is not None:
        final = Path(output_dir) / f'{episode_tag}-{outcome}.mp4'
        partial.replace(final)
    else:
        final = None
    env.close()
    return {'stage': stage, 'seed': seed, 'outcome': outcome,
            'controller': controller,
            'kills': info.get('kills'), 'deaths': info.get('deaths'),
            'frames': info.get('frames'), 'lives': info.get('lives'),
            'llm_calls': commander.calls, 'cost_usd': round(commander.cost, 4),
            'veto_fallbacks': veto_failures,
            'wall_seconds': round(time.perf_counter() - started, 1),
            'video': None if final is None else str(final)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--base-seed', type=int, default=95000)
    parser.add_argument('--model', default='claude-sonnet-5')
    parser.add_argument('--base-url',
                        default='https://pkzzhbqc-23461.jpe1.devtunnels.ms')
    parser.add_argument('--token', default='local-dev-key')
    parser.add_argument('--budget-usd', type=float, default=20.0)
    parser.add_argument('--llm-every', type=int, default=30)
    parser.add_argument('--max-calls', type=int, default=80)
    parser.add_argument('--no-veto', action='store_true')
    parser.add_argument('--initial-mode', default='hunt',
                        choices=('hunt', 'protect', 'hold'))
    parser.add_argument('--record-demos', action='store_true')
    parser.add_argument('--demo-channels', choices=('47', '55', 'both'), default='47')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--strategist', type=Path, default=None,
                        help='local MLP commander checkpoint instead of the LLM')
    parser.add_argument('--schedule', default=None,
                        help='scripted commander "opening,switch_at[,row,col]"')
    parser.add_argument('--guard-bot', action='store_true',
                        help='flank-post base defence executor')
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--plan-seeds', type=int, default=3)
    parser.add_argument('--controller',
                        choices=('reflex-v1', 'fsm-v2', 'reflex-seq'),
                        default='reflex-v1')
    parser.add_argument('--tag-suffix', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    stages = []
    for part in args.stages.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / 'results.jsonl'
    for stage in stages:
        for episode in range(args.episodes):
            if args.schedule is not None:
                parts = args.schedule.split(',')
                target = ([int(parts[2]), int(parts[3])] if len(parts) > 3 else None)
                commander = ScheduleCommander(parts[0], int(parts[1]), target)
            elif args.strategist is not None:
                from strategist import StrategistCommander
                commander = StrategistCommander(args.strategist)
            else:
                commander = Commander(args.base_url, args.token, args.model,
                                      args.budget_usd,
                                      args.output / 'transcript.jsonl')
            result = play_rom_episode(
                stage, seed=args.base_seed + stage * 100 + episode,
                commander=commander, output_dir=args.output,
                episode_tag=f'stage{stage:02d}-ep{episode}-s{args.base_seed}{args.tag_suffix}',
                llm_every=args.llm_every, max_calls=args.max_calls,
                use_veto=not args.no_veto, initial_mode=args.initial_mode,
                record_demos=args.record_demos, record_video=not args.no_video,
                guard_bot=args.guard_bot, demo_channels=args.demo_channels,
                depth=args.depth, plan_seeds=args.plan_seeds,
                controller=args.controller)
            if hasattr(commander, 'log'):
                commander.log.close()
            with results_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
