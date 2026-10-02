"""Save-state campaign: retry each stage from its start snapshot until cleared.

At every stage start the emulator state is snapshotted (nes-py backup). A
failed attempt restores the snapshot, desynchronises the game RNG with a
random number of idle frames, and tries again - exactly the human save-state
grind, with lives and star power rolling back to the stage-start save. Only
the successful attempt of each stage (plus the following interlude) is kept
on video; segments are concatenated into one continuous full-game recording.
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
from base_guard import BaseGuardBot, set_ice_penalty
from battle_city_env import BattleCityEnv
from campaign import INTERLUDE_LIMIT, soft_advance
from llm_player import execute_order
from llm_rom_player import EpisodeVideo, ScheduleCommander
from mpc_teacher import CachedParser, MpcVetoTeacher


def play_stage(env, stage, planner_seed, max_frames, video):
    planner = MpcVetoTeacher(depth=12, plan_seeds=3,
                             plan_rng_seed=planner_seed & 0xffffffff)
    bot = BaseGuardBot()
    parser = CachedParser()
    commander = ScheduleCommander('hunt', 21)
    ice_cells = int((env.state.terrain == gs.T_ICE).sum())
    set_ice_penalty(8.0 if ice_cells > 80 else 0.0)
    for _ in range(max_frames // 4 + 8):
        ram = np.array(env.nes.ram, dtype=np.uint8, copy=True)
        state = parser.parse(ram)
        masks = np.asarray(env.action_masks()[:10], dtype=bool)
        order, _ = commander.order(None, state=state)
        action = execute_order(bot, state, order)
        if not masks[action]:
            action = 0
        try:
            scores, unsafe = planner.plan(ram[None])
            action = int(planner.choose(scores, unsafe, [masks], [action])[0])
        except Exception:
            pass
        _, _, terminated, truncated, info = env.step(int(action))
        if video is not None:
            video.write(env.nes.screen)
        if terminated or truncated:
            return info.get('outcome'), info
    return 'unfinished', {}


def run(seed, output_dir, *, tag, max_attempts=30, max_frames=12000,
        record_video=True, exit_floors=None, stage_attempts=None):
    exit_floors = exit_floors or {}
    stage_attempts = stage_attempts or {}
    output_dir = Path(output_dir)
    segments_dir = output_dir / f'{tag}-segments'
    segments_dir.mkdir(parents=True, exist_ok=True)
    env = BattleCityEnv(stage=1, max_frames=max_frames, reward_profile='sparse')
    rng = np.random.default_rng(seed)
    log = []
    segments = []
    stage = 1
    final = 'unknown'
    started = time.perf_counter()
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(seed)
        env.reset(seed=seed)
        while stage <= 35:
            env.nes._backup()
            cleared = False
            attempts = 0
            floor = exit_floors.get(stage, 0)
            budget = stage_attempts.get(stage, max_attempts)
            for attempt in range(budget):
                attempts = attempt + 1
                if attempt > 0:
                    env.nes._restore()
                    for _ in range(int(rng.integers(1, 48))):
                        env.nes.step(0)
                    soft_advance(env, stage)
                segment = segments_dir / f'stage{stage:02d}.mp4'
                video = EpisodeVideo(segment) if record_video else None
                outcome, info = 'error', {}
                try:
                    outcome, info = play_stage(
                        env, stage, seed * 1000 + stage * 37 + attempt,
                        max_frames, video)
                finally:
                    done_clear = outcome == 'clear'
                    if video is not None and done_clear and stage < 35:
                        # Keep rolling through the interlude on this segment.
                        pass
                if outcome == 'clear' and (info.get('lives') or 0) >= floor:
                    cleared = True
                    break
                if video is not None:
                    video.close()
                    segment.unlink(missing_ok=True)
            log.append({'stage': stage, 'attempts': attempts,
                        'cleared': cleared, 'kills': info.get('kills'),
                        'deaths': info.get('deaths'), 'frames': info.get('frames'),
                        'lives_left': info.get('lives')})
            print(json.dumps(log[-1]), flush=True)
            if not cleared:
                final = f'gave_up_stage_{stage}'
                if record_video:
                    video.close()
                break
            if stage == 35:
                final = 'campaign_complete'
                if record_video:
                    video.close()
                    segments.append(segments_dir / f'stage{stage:02d}.mp4')
                break
            transitioned = False
            for frame in range(INTERLUDE_LIMIT):
                env.nes.step(0)
                if record_video and frame % 4 == 0:
                    video.write(env.nes.screen)
                ram = env.nes.ram
                if int(ram[0x85]) == stage + 1 and 0x80 <= int(ram[0xa0]) < 0xe0 \
                        and int(ram[0x80]) == 20:
                    transitioned = True
                    break
            for frame in range(16):
                env.nes.step(0)
                if record_video and frame % 4 == 0:
                    video.write(env.nes.screen)
            if record_video:
                video.close()
                segments.append(segments_dir / f'stage{stage:02d}.mp4')
            if not transitioned:
                final = 'transition_failed'
                break
            stage += 1
            soft_advance(env, stage)
    finally:
        env.close()
    result = {'seed': seed, 'final': final,
              'stages_cleared': sum(entry['cleared'] for entry in log),
              'total_attempts': sum(entry['attempts'] for entry in log),
              'stages': log,
              'wall_seconds': round(time.perf_counter() - started, 1)}
    if record_video and segments:
        list_path = segments_dir / 'concat.txt'
        list_path.write_text(
            ''.join(f"file '{p.resolve().as_posix()}'\n" for p in segments),
            encoding='utf-8')
        full = output_dir / f'{tag}-full.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-f', 'concat', '-safe', '0', '-i', str(list_path),
                        '-c', 'copy', str(full)], check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        result['video'] = str(full)
    (output_dir / f'{tag}.json').write_text(json.dumps(result, indent=2),
                                            encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('seed', 'final', 'stages_cleared', 'total_attempts',
                       'wall_seconds')} | {'video': result.get('video')}),
          flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--exit-floors', default='',
                        help='e.g. "33:2,30:2" - a clear only counts with lives >= N')
    parser.add_argument('--stage-attempts', default='',
                        help='e.g. "34:120" - per-stage attempt budget override')
    args = parser.parse_args(argv)

    def parse_map(text):
        return {int(k): int(v) for k, v in
                (pair.split(':') for pair in text.split(',') if pair)}

    run(args.seed, args.output, tag=args.tag or f'saverun-seed{args.seed}',
        max_attempts=args.max_attempts, exit_floors=parse_map(args.exit_floors),
        stage_attempts=parse_map(args.stage_attempts))


if __name__ == '__main__':
    main()
