"""RNG-backend ablation: splitmix vs cartridge vs cartridge+scratch.

Trains one policy per (backend, seed) under identical PPO budgets, then
evaluates every checkpoint on two surfaces:
  - cross-backend Rust episodes (same engine, different randomness), measuring
    sensitivity to the training distribution;
  - original-ROM transfer via ``rust_train.py eval`` (the only acceptance
    surface; full per-episode frame allowance from the training manifest).

Descriptive comparison only: two seeds per arm bound no significance claims.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).parent
ARMS = {
    'splitmix': {'rng_backend': 'splitmix', 'scratch': False},
    'cartridge': {'rng_backend': 'cartridge', 'scratch': False},
    'cartridge-scratch': {'rng_backend': 'cartridge', 'scratch': True},
    # Gold standard: every training transition from the local ROM (rom_train.py,
    # affinity-limited to all-but-two cores). No randomness or simulation gap.
    'rom': {'rng_backend': None, 'scratch': False},
}


def run(command):
    print('>>', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def train_arm(args, arm, spec, seed, output):
    if arm == 'rom':
        run([sys.executable, ROOT / 'rom_train.py',
             '--output', output, '--device', args.device,
             '--steps', args.steps, '--envs', args.envs, '--n-steps', args.n_steps,
             '--batch-size', args.batch_size, '--epochs', args.epochs,
             '--max-frames', args.max_frames, '--seed', seed,
             '--stages', ','.join(map(str, args.stages))])
        return
    command = [sys.executable, ROOT / 'rust_train.py', 'train',
               '--output', output, '--codec', args.codec, '--device', args.device,
               '--steps', args.steps, '--envs', args.envs, '--n-steps', args.n_steps,
               '--batch-size', args.batch_size, '--epochs', args.epochs,
               '--max-frames', args.max_frames, '--seed', seed,
               '--stages', ','.join(map(str, args.stages)),
               '--rng-backend', spec['rng_backend']]
    if spec['scratch']:
        command += ['--scratch-trace', args.scratch_trace]
    run(command)


def rom_eval(args, model_path, output_path):
    """Inline ROM transfer episodes for arms whose manifest is not a rust_train one."""
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from rust_train import ROMPrimitiveEnv

    model = MaskablePPO.load(
        model_path, device=args.device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer,
                        'rollout_buffer_kwargs': {}, 'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    episodes = []
    for stage in args.stages:
        vec = VecFrameStack(DummyVecEnv([lambda stage=stage: ROMPrimitiveEnv(stage, args.max_frames)]),
                            n_stack=2, channels_order='first')
        try:
            for seed in [int(s) for s in str(args.rom_eval_seeds).split(',')]:
                for deterministic in (True, False):
                    set_random_seed(seed)
                    vec.seed(seed)
                    obs = vec.reset()
                    total = 0.0
                    for _ in range((args.max_frames + 3) // 4):
                        masks = np.stack(vec.env_method('action_masks'))
                        with torch.no_grad():
                            action, _ = model.predict(
                                obs, deterministic=deterministic, action_masks=masks)
                        obs, reward, done, infos = vec.step(action)
                        total += float(reward[0])
                        if done[0]:
                            info = infos[0]
                            episodes.append({key: info[key] for key in
                                             ('outcome', 'frames', 'kills', 'deaths',
                                              'shots', 'pickups', 'lives')})
                            episodes[-1].update(stage=stage, seed=seed,
                                                deterministic=deterministic, reward=total)
                            break
                    else:
                        raise RuntimeError('ROM transfer episode exceeded its frame limit')
        finally:
            vec.close()
    report = {'backend': 'original-rom-transfer', 'episodes': episodes,
              'wins': sum(case['outcome'] == 'clear' for case in episodes),
              'note': 'Inline transfer episodes for the direct-ROM arm.'}
    output_path.write_text(json.dumps(report, indent=2), encoding='ascii')
    return report


def cross_eval(args, model_path, manifest):
    """Deterministic+sampled Rust episodes under every backend's randomness."""
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecFrameStack
    from rust_env import RustVectorEnv, ScratchTrace, stage_ram

    model = MaskablePPO.load(
        model_path, device=args.device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer,
                        'rollout_buffer_kwargs': {}, 'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    trace = ScratchTrace.load(args.scratch_trace)
    snapshots = {stage: stage_ram(stage, seed=manifest['training']['seed'])
                 for stage in args.stages}
    episodes = []
    for surface, spec in ARMS.items():
        if spec['rng_backend'] is None:
            continue  # 'rom' is a training arm, not a Rust evaluation surface
        scratch = trace if spec['scratch'] else None
        for stage in args.stages:
            vec = VecFrameStack(
                RustVectorEnv(snapshots[stage][None, :], args.max_frames, seed=0,
                              rng_backend=spec['rng_backend'], scratch=scratch),
                n_stack=2, channels_order='first')
            try:
                for seed in args.eval_seeds:
                    for deterministic in (True, False):
                        set_random_seed(seed)
                        vec.seed(seed)
                        obs = vec.reset()
                        total = 0.0
                        for _ in range((args.max_frames + 3) // 4):
                            masks = np.stack(vec.env_method('action_masks'))
                            with torch.no_grad():
                                action, _ = model.predict(
                                    obs, deterministic=deterministic, action_masks=masks)
                            obs, reward, done, infos = vec.step(action)
                            total += float(reward[0])
                            if done[0]:
                                info = infos[0]
                                episodes.append({key: info[key] for key in
                                                 ('outcome', 'frames', 'kills', 'deaths',
                                                  'shots', 'pickups', 'lives')})
                                episodes[-1].update(surface=surface, stage=stage, seed=seed,
                                                    deterministic=deterministic, reward=total)
                                break
                        else:
                            raise RuntimeError('Rust episode exceeded its frame limit')
            finally:
                vec.close()
    return episodes


def summarize(episodes, keys):
    groups = {}
    for episode in episodes:
        groups.setdefault(tuple(episode[key] for key in keys), []).append(episode)
    result = {}
    for group, cases in sorted(groups.items()):
        result['/'.join(map(str, group))] = {
            'episodes': len(cases),
            'clears': sum(c['outcome'] == 'clear' for c in cases),
            'base_lost': sum(c['outcome'] == 'base_lost' for c in cases),
            'lives_exhausted': sum(c['outcome'] == 'lives_exhausted' for c in cases),
            'timeouts': sum(c['outcome'] == 'timeout' for c in cases),
            'mean_kills': round(float(np.mean([c['kills'] for c in cases])), 3),
            'mean_deaths': round(float(np.mean([c['deaths'] for c in cases])), 3),
            'mean_frames': round(float(np.mean([c['frames'] for c in cases])), 1),
            'mean_reward': round(float(np.mean([c['reward'] for c in cases])), 3),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--scratch-trace', type=Path,
                        default=ROOT / 'runs' / 'scratch-trace-v1.npz')
    parser.add_argument('--seeds', default='37,1037')
    parser.add_argument('--stages', default='1,2,3')
    parser.add_argument('--steps', type=int, default=1048576)
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--codec', default='cuda')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rom-eval-seeds', default='101,202')
    parser.add_argument('--eval-seeds', default='1001,2002,3003,4004')
    args = parser.parse_args(argv)
    args.stages = [int(s) for s in str(args.stages).split(',')]
    args.eval_seeds = [int(s) for s in str(args.eval_seeds).split(',')]
    seeds = [int(s) for s in str(args.seeds).split(',')]
    arms = [arm.strip() for arm in str(args.arms).split(',') if arm.strip()]
    if any(arm not in ARMS for arm in arms):
        raise SystemExit(f'arms must be among {sorted(ARMS)}')
    if not args.scratch_trace.is_file():
        raise SystemExit(f'missing scratch trace: {args.scratch_trace}')
    args.output.mkdir(parents=True, exist_ok=False)

    results = {}
    for arm in arms:
        spec = ARMS[arm]
        for seed in seeds:
            name = f'{arm}-s{seed}'
            output = args.output / name
            started = time.perf_counter()
            train_arm(args, arm, spec, seed, output)
            if arm == 'rom':
                rom_eval(args, output / 'model.zip', output / 'rom-transfer.json')
            else:
                run([sys.executable, ROOT / 'rust_train.py', 'eval',
                     '--model', output / 'model.zip',
                     '--output', output / 'rom-transfer.json',
                     '--stages', ','.join(map(str, args.stages)),
                     '--seeds', args.rom_eval_seeds, '--device', args.device])
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            cross = cross_eval(args, output / 'model.zip', manifest)
            (output / 'rust-cross-eval.json').write_text(
                json.dumps({'episodes': cross}, indent=2), encoding='ascii')
            rom = json.loads((output / 'rom-transfer.json').read_text(encoding='utf-8'))
            results[name] = {
                'arm': arm, 'seed': seed,
                'minutes': round((time.perf_counter() - started) / 60, 1),
                'train_outcomes': json.loads(
                    (output / 'result.json').read_text(encoding='utf-8'))
                    ['completed_episode_outcomes'],
                'rust_cross_eval': summarize(cross, ('surface',)),
                'rom_transfer': summarize(rom['episodes'], ()),
                'rom_wins': rom['wins'],
                'scratch_trace_sha256': manifest.get('scratch_trace_sha256'),
            }
            (args.output / 'summary.json').write_text(
                json.dumps({'config': {k: str(v) for k, v in vars(args).items()},
                            'results': results,
                            'note': 'Two seeds per arm; descriptive comparison, '
                                    'not a significance claim. ROM remains the '
                                    'only acceptance surface.'},
                           indent=2), encoding='ascii')
            print(f'== {name} done:', json.dumps(results[name]['rust_cross_eval'],
                                                 sort_keys=True), flush=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
