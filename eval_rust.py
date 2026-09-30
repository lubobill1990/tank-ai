"""Evaluate a 10-action MaskablePPO checkpoint inside the Rust engine.

Fast in-loop validation surface; Rust outcomes are not original-ROM wins.
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


def evaluate(model, stage, *, episodes, seed, deterministic, max_frames=12000):
    from stable_baselines3.common.vec_env import VecFrameStack
    from rust_env import RustVectorEnv, stage_ram
    ram0 = stage_ram(stage, seed=0)
    initial = np.stack([ram0] * episodes)
    vec = VecFrameStack(RustVectorEnv(initial, max_frames, seed=seed),
                        n_stack=2, channels_order='first')
    results = [None] * episodes
    try:
        obs = vec.reset()
        for _ in range(max_frames // 4 + 8):
            masks = np.stack(vec.env_method('action_masks'))
            actions, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
            obs, rewards, dones, infos = vec.step(actions)
            for i in np.flatnonzero(dones):
                if results[int(i)] is None:
                    info = infos[int(i)]
                    results[int(i)] = {key: info[key] for key in
                                       ('outcome', 'frames', 'kills', 'deaths', 'lives')}
            if all(r is not None for r in results):
                break
    finally:
        vec.close()
    return [r if r is not None else {'outcome': 'unfinished'} for r in results]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=9000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample', action='store_true',
                        help='sampled actions instead of deterministic')
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    model = MaskablePPO.load(
        args.model, device=args.device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer,
                        'rollout_buffer_kwargs': {}, 'n_steps': 2, 'n_envs': 1,
                        'batch_size': 2})
    stages = parse_stages(args.stages)
    report = {'model': str(args.model), 'seed': args.seed,
              'deterministic': not args.sample, 'stages': {}}
    total_clears = total_eps = 0
    start = time.perf_counter()
    for stage in stages:
        episodes = evaluate(model, stage, episodes=args.episodes,
                            seed=args.seed + stage, deterministic=not args.sample,
                            max_frames=args.max_frames)
        clears = sum(e['outcome'] == 'clear' for e in episodes)
        total_clears += clears
        total_eps += len(episodes)
        report['stages'][str(stage)] = {
            'clears': clears, 'episodes': len(episodes),
            'outcomes': {k: sum(e['outcome'] == k for e in episodes)
                         for k in set(e['outcome'] for e in episodes)},
            'kills_mean': float(np.mean([e.get('kills', 0) for e in episodes])),
        }
        print(f"stage {stage:2d}: {clears}/{len(episodes)} "
              f"{report['stages'][str(stage)]['outcomes']}", flush=True)
    report['total'] = {'clears': total_clears, 'episodes': total_eps,
                       'clear_rate': total_clears / max(total_eps, 1),
                       'elapsed_seconds': time.perf_counter() - start}
    print(json.dumps(report['total']))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
