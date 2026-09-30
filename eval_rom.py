"""Evaluate a 10-action MaskablePPO checkpoint on the original ROM, full stages.

This is the authoritative acceptance surface: 12,000-frame allowance, ordinary
resets, no teacher, no RAM writes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_stages(text):
    stages = []
    for part in text.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from rust_train import ROMPrimitiveEnv
    model = MaskablePPO.load(
        args.model, device=args.device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer,
                        'rollout_buffer_kwargs': {}, 'n_steps': 2, 'n_envs': 1,
                        'batch_size': 2})
    stages = parse_stages(args.stages)
    seeds = [int(s) for s in args.seeds.split(',')]
    episodes = []
    start = time.perf_counter()
    for stage in stages:
        vec = VecFrameStack(DummyVecEnv([lambda stage=stage: ROMPrimitiveEnv(stage, args.max_frames)]),
                            n_stack=2, channels_order='first')
        try:
            for seed in seeds:
                set_random_seed(seed)
                vec.seed(seed)
                obs = vec.reset()
                for _ in range((args.max_frames + 3) // 4 + 4):
                    masks = np.stack(vec.env_method('action_masks'))
                    action, _ = model.predict(obs, deterministic=not args.sample,
                                              action_masks=masks)
                    obs, _, done, infos = vec.step(action)
                    if done[0]:
                        info = infos[0]
                        entry = {key: info[key] for key in
                                 ('outcome', 'frames', 'kills', 'deaths', 'lives')}
                        entry.update(stage=stage, seed=seed)
                        episodes.append(entry)
                        break
                else:
                    episodes.append({'outcome': 'unfinished', 'stage': stage, 'seed': seed})
        finally:
            vec.close()
        clears = sum(e['outcome'] == 'clear' for e in episodes if e['stage'] == stage)
        print(f'stage {stage:2d}: {clears}/{len(seeds)}', flush=True)
    report = {
        'surface': 'original-rom-full-stage', 'model': str(args.model),
        'deterministic': not args.sample, 'max_frames': args.max_frames,
        'episodes': episodes,
        'wins': sum(e['outcome'] == 'clear' for e in episodes),
        'total': len(episodes),
        'per_stage': {str(stage): sum(e['outcome'] == 'clear' and e['stage'] == stage
                                      for e in episodes) for stage in stages},
        'elapsed_seconds': time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'wins': report['wins'], 'total': report['total']}))


if __name__ == '__main__':
    main()
