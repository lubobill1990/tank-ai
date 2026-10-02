"""Receding-horizon ROM evaluation of an action-chunking checkpoint.

Pure network execution: no executor, no MPC veto, no LLM. Every `stride`
decisions the model predicts a K-action chunk from the stacked observation;
queued actions that are illegal under the live mask fall back to the best
legal action of that step's stored logits.
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

from chunk_bc import build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from stable_baselines3.common.utils import set_random_seed
    from rust_train import ROMPrimitiveEnv
    payload = torch.load(args.model, map_location=args.device, weights_only=False)
    chunk = payload['chunk']
    channels = payload.get('channels', 47)
    model = build_model(chunk, channels).to(args.device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    stride = min(args.stride, chunk)
    if channels == 55:
        from within_stage.observation import encode_precision

    stages = []
    for part in args.stages.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    seeds = [int(s) for s in args.seeds.split(',')]
    episodes = []
    start = time.perf_counter()
    for stage in stages:
        for seed in seeds:
            env = ROMPrimitiveEnv(stage, args.max_frames)
            raw = env.env
            set_random_seed(seed)
            observation, _ = env.reset(seed=seed)
            if channels == 55:
                observation = encode_precision(raw.state, np.asarray(observation, np.float32))
            previous = np.zeros((channels, 26, 26), np.float32)
            queue_logits = None
            queue_pos = 0
            outcome, info = 'unfinished', {}
            for step in range(args.max_frames // 4 + 4):
                if queue_logits is None or queue_pos >= stride:
                    stacked = np.concatenate([previous, observation])[None]
                    with torch.no_grad():
                        logits = model(torch.as_tensor(stacked, device=args.device))
                    queue_logits = logits[0].float().cpu().numpy()
                    queue_pos = 0
                masks = np.asarray(raw.action_masks()[:10], dtype=bool)
                step_logits = np.where(masks, queue_logits[queue_pos], -1e9)
                action = int(step_logits.argmax())
                queue_pos += 1
                previous = np.asarray(observation, np.float32)
                observation, _, terminated, truncated, step_info = env.step(action)
                if channels == 55:
                    observation = encode_precision(raw.state,
                                                   np.asarray(observation, np.float32))
                if terminated or truncated:
                    outcome = step_info.get('outcome', 'unknown')
                    info = step_info
                    break
            env.close()
            episodes.append({'stage': stage, 'seed': seed, 'outcome': outcome,
                             'kills': info.get('kills'), 'frames': info.get('frames')})
        clears = sum(e['outcome'] == 'clear' for e in episodes if e['stage'] == stage)
        print(f'stage {stage:2d}: {clears}/{len(seeds)}', flush=True)
    report = {
        'model': str(args.model), 'stride': stride, 'chunk': chunk,
        'episodes': episodes,
        'wins': sum(e['outcome'] == 'clear' for e in episodes),
        'total': len(episodes),
        'per_stage': {str(s): sum(e['outcome'] == 'clear' and e['stage'] == s
                                  for e in episodes) for s in stages},
        'elapsed_seconds': time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'wins': report['wins'], 'total': report['total']}))


if __name__ == '__main__':
    main()
