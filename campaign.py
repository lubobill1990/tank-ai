"""Continuous campaign run on the original ROM with carry-over, recorded.

The champion stack plays from stage 1 onward in ONE emulator session: after a
clear the runner steps the NES frame by frame through the score/curtain
interlude until the next stage's gameplay begins, then soft-resets only the
Python-side trackers. Star power, lives and score therefore carry over exactly
as in the real game. The whole campaign is captured to a single silent MP4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
from base_guard import BaseGuardBot, set_ice_penalty
from battle_city_env import BattleCityEnv
from llm_player import execute_order
from llm_rom_player import EpisodeVideo, ScheduleCommander
from mpc_teacher import CachedParser, MpcVetoTeacher

INTERLUDE_LIMIT = 2400  # frames allowed for score screen + stage curtain


def soft_advance(env, next_stage):
    """Reset Python-side bookkeeping around the already-running emulator."""
    env.stage = next_stage
    env._tracker.reset(env.nes.ram)
    env._bot.reset()
    env._deaths = env._shots = env._pickups = 0
    env._item_drops = 0
    env._item_opportunities = int(env.state.item is not None)
    env._combat_pickup_deadline = None
    env._last_progress_frame = 0
    env._boundary_waste_frames = env._boundary_waste_streak_frames = 0
    env._max_boundary_waste_frames = 0
    env._needs_reset = False


def run_campaign(seed, output_dir, *, tag, max_frames=12000, depth=12,
                 plan_seeds=3, record_video=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = BattleCityEnv(stage=1, max_frames=max_frames, reward_profile='sparse')
    partial = output_dir / f'.partial-{tag}.mp4'
    video = EpisodeVideo(partial) if record_video else None
    stages_log = []
    stage = 1
    final = 'unknown'
    started = time.perf_counter()
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(seed)
        env.reset(seed=seed)
        if video is not None:
            video.write(env.nes.screen)
        while stage <= 35:
            planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                                     plan_rng_seed=(seed * 97 + stage) & 0xffffffff)
            bot = BaseGuardBot()
            parser = CachedParser()
            commander = ScheduleCommander('hunt', 21)
            ice_cells = int((env.state.terrain == gs.T_ICE).sum())
            set_ice_penalty(8.0 if ice_cells > 80 else 0.0)
            entry_power = 0 if env.state.player is None else env.state.player.power
            entry_lives = env.state.lives
            outcome, info = 'unfinished', {}
            from base_guard import restrict_masks
            last_action = None
            for _ in range(max_frames // 4 + 8):
                ram = np.array(env.nes.ram, dtype=np.uint8, copy=True)
                state = parser.parse(ram)
                masks = restrict_masks(state,
                                       np.asarray(env.action_masks()[:10], dtype=bool))
                order, _ = commander.order(None, state=state)
                action = execute_order(bot, state, order)
                if not masks[action]:
                    action = 0
                try:
                    scores, unsafe = planner.plan(ram[None])
                    action = int(planner.choose(scores, unsafe, [masks], [action],
                                                [last_action])[0])
                except Exception:
                    pass
                last_action = action
                _, _, terminated, truncated, step_info = env.step(int(action))
                if video is not None:
                    video.write(env.nes.screen)
                if terminated or truncated:
                    outcome, info = step_info.get('outcome'), step_info
                    break
            stages_log.append({
                'stage': stage, 'outcome': outcome,
                'kills': info.get('kills'), 'deaths': info.get('deaths'),
                'frames': info.get('frames'), 'lives_left': info.get('lives'),
                'entry_power': entry_power, 'entry_lives': entry_lives})
            print(json.dumps(stages_log[-1]), flush=True)
            if outcome != 'clear':
                final = outcome
                break
            if stage == 35:
                final = 'campaign_complete'
                break
            # Interlude: score tally and next-stage curtain run on their own.
            for frame in range(INTERLUDE_LIMIT):
                env.nes.step(0)
                if video is not None and frame % 4 == 0:
                    video.write(env.nes.screen)
                ram = env.nes.ram
                if int(ram[0x85]) == stage + 1 and 0x80 <= int(ram[0xa0]) < 0xe0 \
                        and int(ram[0x80]) == 20:
                    break
            else:
                final = 'transition_failed'
                break
            # Let the spawn settle for a few frames before taking control.
            for frame in range(16):
                env.nes.step(0)
                if video is not None and frame % 4 == 0:
                    video.write(env.nes.screen)
            stage += 1
            soft_advance(env, stage)
    finally:
        if video is not None:
            video.close()
        env.close()
    stages_cleared = sum(entry['outcome'] == 'clear' for entry in stages_log)
    result = {'seed': seed, 'final': final, 'stages_cleared': stages_cleared,
              'reached_stage': stage, 'stages': stages_log,
              'wall_seconds': round(time.perf_counter() - started, 1)}
    video_path = None
    if record_video:
        video_path = output_dir / f'{tag}-cleared{stages_cleared}.mp4'
        partial.replace(video_path)
        result['video'] = str(video_path)
    (output_dir / f'{tag}.json').write_text(json.dumps(result, indent=2),
                                            encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('seed', 'final', 'stages_cleared', 'reached_stage',
                       'wall_seconds')} | {'video': result.get('video')}),
          flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args(argv)
    for seed in (int(s) for s in args.seeds.split(',')):
        run_campaign(seed, args.output, tag=f'campaign-seed{seed}',
                     max_frames=args.max_frames,
                     record_video=not args.no_video)


if __name__ == '__main__':
    main()
