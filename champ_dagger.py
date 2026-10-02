"""DAgger with the champion system as the expert labeller.

The student checkpoint plays on the original ROM (sampled actions); at every
decision the champion stack - per-stage schedule, BaseGuardBot executor with
flank-guard and item reflexes, MPC lookahead veto - computes the action it
would have taken, which is recorded as the label. Collection follows the
standard DemonstrationWriter format.
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

from base_guard import BaseGuardBot
from learning_lab.demonstrations import DemonstrationWriter
from llm_player import execute_order
from llm_rom_player import ScheduleCommander
from mpc_teacher import CachedParser, MpcVetoTeacher, load_student

STAGE_SCHEDULES = {2: ('protect', 16), 13: ('protect', 16), 21: ('protect', 16),
                   25: ('protect', 12), 33: ('protect', 12)}


def make_commander(stage):
    opening, switch_at = STAGE_SCHEDULES.get(stage, ('hunt', 21))
    return ScheduleCommander(opening, switch_at)


def collect_stage(stage, *, episodes, output, base_seed, student, student_device,
                  beta, max_frames, depth, plan_seeds, deadline=None):
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from rust_train import ROMPrimitiveEnv
    planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                             plan_rng_seed=base_seed * 1000003 + stage)
    policy = load_student(student, student_device)
    rng = np.random.default_rng(base_seed * 7919 + stage)
    output = Path(output)
    writer = DemonstrationWriter(output)
    reports = []
    vec = VecFrameStack(DummyVecEnv([lambda: ROMPrimitiveEnv(stage, max_frames)]),
                        n_stack=2, channels_order='first')
    raw = vec.venv.envs[0].env
    planned = fallback = 0
    try:
        for episode in range(episodes):
            if deadline is not None and time.perf_counter() > deadline:
                break
            seed = base_seed + episode
            set_random_seed(seed)
            vec.seed(seed)
            obs = vec.reset()
            bot = BaseGuardBot()
            commander = make_commander(stage)
            parser = CachedParser()
            episode_start = True
            decisions = 0
            for _ in range((max_frames + 3) // 4 + 4):
                ram = np.array(raw.nes.ram, dtype=np.uint8, copy=True)
                state = parser.parse(ram)
                masks = np.asarray(raw.action_masks()[:10], dtype=bool)
                order, _ = commander.order(None, state=state)
                label = execute_order(bot, state, order)
                if not masks[label]:
                    label = 0
                try:
                    scores, unsafe = planner.plan(ram[None])
                    label = int(planner.choose(scores, unsafe, [masks], [label])[0])
                    planned += 1
                except Exception:
                    fallback += 1
                student_action, _ = policy.predict(obs, deterministic=False,
                                                   action_masks=masks[None])
                run = label if rng.random() < beta else int(student_action[0])
                writer.add(obs[0, 47:].copy(), label, masks,
                           episode_start=episode_start, executed_action=run)
                episode_start = False
                decisions += 1
                obs, _, done, infos = vec.step(np.array([run]))
                if done[0]:
                    info = infos[0]
                    reports.append({'episode_id': len(reports),
                                    'outcome': info['outcome'], 'stage': stage,
                                    'kills': info['kills'], 'deaths': info['deaths'],
                                    'frames': info['frames'],
                                    'decisions': decisions, 'seed': seed})
                    break
            else:
                reports.append({'episode_id': len(reports), 'outcome': 'unfinished',
                                'stage': stage, 'decisions': decisions, 'seed': seed})
    finally:
        vec.close()
        while len(reports) < sum(writer.starts):
            reports.append({'episode_id': len(reports), 'outcome': 'incomplete',
                            'stage': stage})
        writer.finish(reports, {
            'collector': 'champion-dagger-v1', 'stage': stage, 'stream': 0,
            'base_seed': base_seed, 'beta': beta, 'noise': 0.0,
            'depth': depth, 'plan_seeds': plan_seeds, 'max_frames': max_frames,
            'student': str(student), 'planned_labels': planned,
            'fallback_labels': fallback,
            'backend': 'original-rom-champion-labels'})
    clears = sum(r['outcome'] == 'clear' for r in reports)
    summary = {'stage': stage, 'episodes': len(reports), 'clears': clears,
               'planned': planned, 'fallback': fallback}
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=310000)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--student-device', default='cpu')
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--plan-seeds', type=int, default=3)
    args = parser.parse_args(argv)
    stages = []
    for part in args.stages.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    args.output.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        stage_dir = args.output / f'stage-{stage:02d}'
        if stage_dir.exists():
            continue
        summary = collect_stage(
            stage, episodes=args.episodes, output=stage_dir,
            base_seed=args.base_seed + stage * 100, student=args.student,
            student_device=args.student_device, beta=args.beta,
            max_frames=args.max_frames, depth=args.depth,
            plan_seeds=args.plan_seeds)
        with (args.output / 'summaries.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(summary) + '\n')


if __name__ == '__main__':
    main()
