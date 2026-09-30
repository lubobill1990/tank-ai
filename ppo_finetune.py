"""PPO fine-tune of a BC checkpoint inside the Rust engine, value-warmup first.

Phase 1 freezes everything except the value head so PPO's critic catches up
with the cloned actor; phase 2 unfreezes (optionally keeping the features
extractor frozen) and continues with a conservative learning rate and KL cap.
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

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def parse_stages(text):
    stages = []
    for part in text.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    return stages


class OutcomeLog:
    def __init__(self):
        self.window = []

    def push(self, info):
        self.window.append(info['outcome'])
        if len(self.window) > 400:
            self.window.pop(0)

    def rates(self):
        if not self.window:
            return {}
        return {k: self.window.count(k) / len(self.window)
                for k in set(self.window)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--stage-weights', default=None,
                        help='JSON mapping stage->weight for env allocation')
    parser.add_argument('--envs', type=int, default=70)
    parser.add_argument('--warmup-steps', type=int, default=71680)
    parser.add_argument('--steps', type=int, default=430080)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup-lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--target-kl', type=float, default=0.02)
    parser.add_argument('--ent-coef', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--freeze-extractor', action='store_true')
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--checkpoint-every', type=int, default=71680)
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(4)
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecFrameStack
    from native_rollout import NativeGPURunRolloutBuffer
    from rust_env import RustVectorEnv, stage_ram

    args.output.mkdir(parents=True, exist_ok=False)
    stages = parse_stages(args.stages)
    if args.stage_weights:
        weights = {int(k): float(v) for k, v in json.loads(args.stage_weights).items()}
    else:
        weights = {}
    pool = []
    for stage in stages:
        pool.extend([stage] * max(1, round(weights.get(stage, 1.0))))
    assignment = [pool[i % len(pool)] for i in range(args.envs)]
    rams = {stage: stage_ram(stage, seed=0) for stage in sorted(set(assignment))}
    initial = np.stack([rams[stage] for stage in assignment])
    vec = VecFrameStack(RustVectorEnv(initial, args.max_frames, seed=args.seed),
                        n_stack=2, channels_order='first')

    buffer_kwargs = {'channels_per_frame': 47, 'max_storage_bytes': 2 << 30}
    model = MaskablePPO.load(
        args.checkpoint, env=vec, device=args.device,
        custom_objects={
            'n_steps': args.n_steps, 'batch_size': args.batch_size,
            'n_epochs': args.epochs, 'learning_rate': args.warmup_lr,
            'gamma': args.gamma, 'gae_lambda': 0.95,
            'clip_range': args.clip, 'target_kl': args.target_kl,
            'ent_coef': args.ent_coef, 'seed': args.seed,
            'rollout_buffer_class': (NativeGPURunRolloutBuffer
                                     if args.device == 'cuda' else MaskableRolloutBuffer),
            'rollout_buffer_kwargs': (buffer_kwargs if args.device == 'cuda' else {}),
        })

    log = OutcomeLog()
    state = {'next_checkpoint': args.checkpoint_every}

    class Monitor(BaseCallback):
        def _on_step(self):
            if not np.isfinite(self.locals['rewards']).all():
                raise FloatingPointError('Nonfinite native reward')
            for info in self.locals['infos']:
                if 'outcome' in info:
                    log.push(info)
            if self.num_timesteps >= state['next_checkpoint']:
                state['next_checkpoint'] += args.checkpoint_every
                path = args.output / f'step-{self.num_timesteps:08d}.zip'
                self.model.save(path)
                print(json.dumps({'step': self.num_timesteps, 'outcomes': log.rates()}),
                      flush=True)
            return True

    def set_trainable(actor, extractor):
        policy = model.policy
        for parameter in policy.parameters():
            parameter.requires_grad = True
        if not extractor:
            for name in ('features_extractor', 'pi_features_extractor',
                         'vf_features_extractor'):
                module = getattr(policy, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        if not actor:
            for module in (policy.mlp_extractor.policy_net, policy.action_net):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    start = time.perf_counter()
    if args.warmup_steps:
        set_trainable(actor=False, extractor=False)
        model.learn(total_timesteps=args.warmup_steps, callback=Monitor(),
                    reset_num_timesteps=True, progress_bar=False)
        model.save(args.output / 'after-warmup.zip')
        print(json.dumps({'phase': 'warmup_done', 'outcomes': log.rates(),
                          'seconds': time.perf_counter() - start}), flush=True)

    set_trainable(actor=True, extractor=not args.freeze_extractor)
    model.learning_rate = args.lr
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group['lr'] = args.lr
    model.learn(total_timesteps=args.steps, callback=Monitor(),
                reset_num_timesteps=False, progress_bar=False)
    model.save(args.output / 'final.zip')
    report = {'checkpoint': str(args.checkpoint), 'stages': stages,
              'assignment_counts': {str(s): assignment.count(s) for s in sorted(set(assignment))},
              'steps': int(model.num_timesteps), 'outcomes_recent': log.rates(),
              'elapsed_seconds': time.perf_counter() - start,
              'config': {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2),
                                             encoding='utf-8')
    print(json.dumps(report['outcomes_recent']))
    vec.close()


if __name__ == '__main__':
    main()
