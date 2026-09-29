"""Experimental Rust backend: train/compare/eval acceptance commands.

Use ``study`` for bounded CNN/spatial/history learning with periodic original-ROM
validation, or ``learn`` for an explicitly prepared single-trial configuration.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from battle_city_env import BattleCityEnv, CHANNEL_NAMES, ENV_VERSION
from experiments import integer_list, sha256, write_json
from policy import Grid26CNN
from rust_env import (BACKEND_VERSION, RNG_VERSION, SEED_VERSION, RustVectorEnv,
                      native_library, stage_ram)

ROOT = Path(__file__).parent
CODECS = ('dense', 'zlib', 'native', 'cuda')


def contract():
    return {
        'backend': BACKEND_VERSION, 'rng': RNG_VERSION, 'seed_derivation': SEED_VERSION,
        'actions': 10, 'observation_shape': [94, 26, 26], 'channels': list(CHANNEL_NAMES),
        'n_stack': 2, 'frame_skip': 4, 'reward': 'sparse-kill1-death-3-base-20-clear5-frame-0.0005',
        'termination': 'full-stage-multi-life-v1',
    }


def validate_manifest(manifest):
    if manifest.get('contract') != contract():
        raise ValueError('Incompatible native model manifest (backend/RNG/actions/observations/episode rules)')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'compare'):
        sub = commands.add_parser(command)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--codec', choices=CODECS, default='native')
        sub.add_argument('--steps', type=int, default=4096)
        sub.add_argument('--envs', type=int, default=8)
        sub.add_argument('--n-steps', type=int, default=64)
        sub.add_argument('--batch-size', type=int, default=64)
        sub.add_argument('--epochs', type=int, default=2)
        sub.add_argument('--max-frames', type=int, default=12000)
        sub.add_argument('--stages', type=integer_list, default=(1,))
        sub.add_argument('--seed', type=int, default=37)
        sub.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        sub.add_argument('--max-storage-mib', type=int, default=2048)
        sub.add_argument('--plain-batch', action='store_true',
                         help='Diagnostic baseline: separate native status/mask calls')
    sub = commands.add_parser('eval')
    sub.add_argument('--model', type=Path, required=True)
    sub.add_argument('--output', type=Path, required=True)
    sub.add_argument('--stages', type=integer_list, default=(1, 5, 10, 20, 35))
    sub.add_argument('--seeds', type=integer_list, default=(0, 22))
    sub.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    if any(not 1 <= stage <= 35 for stage in args.stages):
        parser.error('stages must be in 1..35')
    if args.command != 'eval':
        for name in ('steps', 'envs', 'n_steps', 'batch_size', 'epochs', 'max_frames', 'max_storage_mib'):
            if getattr(args, name) <= 0:
                parser.error(f'{name} must be positive')
        if args.envs > 4096 or args.max_frames >= 2**32 or not 0 <= args.seed < 2**32:
            parser.error('envs must be <=4096; max_frames and seed must fit uint32')
        rollout = args.envs * args.n_steps
        if rollout < 2 or args.batch_size < 2 or args.batch_size > rollout or rollout % args.batch_size:
            parser.error('rollout size must be divisible by batch_size, both at least 2')
        if args.steps % rollout:
            parser.error('steps must be a multiple of envs*n_steps; no budget overshoot is allowed')
        if args.envs < len(args.stages):
            parser.error('envs must cover every requested stage')
        if args.codec == 'cuda' and args.device != 'cuda':
            parser.error('cuda decoding requires --device cuda')
        if args.command == 'compare' and args.codec == 'dense':
            parser.error('compare requires a compact codec')
    elif any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error('seeds must fit uint32')
    return args


def _configure(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')


def _buffer_kwargs(codec, max_bytes):
    if codec == 'dense':
        return {}
    from compact_rollout import CompactMaskableRolloutBuffer
    if codec == 'zlib':
        cls = CompactMaskableRolloutBuffer
    else:
        from native_rollout import NativeRunRolloutBuffer, NativeGPURunRolloutBuffer
        cls = NativeGPURunRolloutBuffer if codec == 'cuda' else NativeRunRolloutBuffer
    return {'rollout_buffer_class': cls,
            'rollout_buffer_kwargs': {'channels_per_frame': 47, 'max_storage_bytes': max_bytes}}


def _policy_hash(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _peak_rss():
    class Counters(ct.Structure):
        _fields_ = [('cb', ct.c_uint32), ('faults', ct.c_uint32)] + [
            (name, ct.c_size_t) for name in (
                'peak_working_set', 'working_set', 'peak_paged', 'paged',
                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
    counters = Counters()
    counters.cb = ct.sizeof(counters)
    function = ct.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    function.argtypes = [ct.c_void_p, ct.POINTER(Counters), ct.c_uint32]
    function.restype = ct.c_int
    if not function(ct.c_void_p(-1), ct.byref(counters), counters.cb):
        raise ct.WinError(ct.get_last_error())
    return int(counters.peak_working_set)


class EpisodeResults(BaseCallback):
    def __init__(self):
        super().__init__()
        self.outcomes = {}
        self.kills = self.deaths = self.pickups = 0

    def _on_step(self):
        if not np.isfinite(self.locals['rewards']).all():
            raise FloatingPointError('Nonfinite native reward')
        for info in self.locals['infos']:
            if 'outcome' in info:
                key = info['outcome']
                self.outcomes[key] = self.outcomes.get(key, 0) + 1
                self.kills += info['kills']
                self.deaths += info['deaths']
                self.pickups += info['pickups']
        return True


def train(args):
    _configure(args.device)
    lib = native_library()
    limit = args.max_storage_mib * 1024**2
    dense_bytes = args.envs * args.n_steps * (94 * 26 * 26 * 4 + 68)
    if args.codec == 'dense' and dense_bytes > limit:
        raise MemoryError(f'Dense rollout needs at least {dense_bytes} bytes; exceeds storage limit {limit}')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in
               sorted((ROOT / 'native' / 'src').glob('*.rs'))}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {
        'contract': contract(), 'training': config, 'native_source_sha256': sources,
        'native_library_sha256': sha256(lib._name), 'rom_sha256': sha256(ROOT / 'battle_city.nes'),
        'rom_observer_version': ENV_VERSION, 'torch': torch.__version__, 'numpy': np.__version__,
        'resume_supported': False, 'independent_rom_parity': False,
        'seed_schedule': 'SHA256(tank-ai-episode-v1:seed+env_index:env_index:episode), first 8 bytes LE',
    }
    write_json(args.output / 'config.json', manifest)
    snapshots = [stage_ram(stage, seed=args.seed) for stage in args.stages]
    initial = np.stack([snapshots[i % len(snapshots)] for i in range(args.envs)])
    manifest['initial_ram_sha256'] = [hashlib.sha256(row.tobytes()).hexdigest() for row in initial]
    vec = VecFrameStack(RustVectorEnv(initial, args.max_frames, seed=args.seed,
                                    fast=not args.plain_batch), n_stack=2, channels_order='first')
    try:
        model = MaskablePPO(
            'CnnPolicy', vec, seed=args.seed, device=args.device, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=args.epochs, learning_rate=2.5e-4,
            gamma=0.99, gae_lambda=0.95, verbose=0,
            policy_kwargs={'features_extractor_class': Grid26CNN, 'normalize_images': False},
            **_buffer_kwargs(args.codec, limit))
        callback = EpisodeResults()
        if args.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if model.num_timesteps != args.steps:
            raise RuntimeError('PPO did not honor the exact requested timestep budget')
        if not all(torch.isfinite(value).all().item() for value in model.policy.state_dict().values()):
            raise FloatingPointError('Nonfinite policy parameters')
        model_path = args.output / 'model.zip'
        model.save(model_path)
        manifest['checkpoint_sha256'] = sha256(model_path)
        write_json(args.output / 'manifest.json', manifest)
        buffer = model.rollout_buffer
        storage = (buffer.total_storage_nbytes if args.codec != 'dense' else
                   sum(value.nbytes for value in vars(buffer).values() if isinstance(value, np.ndarray)))
        report = {
            'backend': BACKEND_VERSION, 'codec': args.codec, 'device': args.device,
            'steps': model.num_timesteps, 'training_seconds': elapsed,
            'transitions_per_second': model.num_timesteps / elapsed,
            'policy_sha256': _policy_hash(model.policy), 'rollout_storage_bytes': storage,
            'dense_observation_bytes': args.envs * args.n_steps * 94 * 26 * 26 * 4,
            'peak_process_rss_bytes': _peak_rss(),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
            'gpu_decoded_minibatches': getattr(buffer, 'gpu_decoded_minibatches', 0),
            'completed_episode_outcomes': callback.outcomes, 'completed_episode_kills': callback.kills,
            'completed_episode_deaths': callback.deaths, 'completed_episode_pickups': callback.pickups,
            'note': 'Storage/throughput acceptance only; not a trained-quality or ROM-parity claim.',
        }
        write_json(args.output / 'result.json', report)
        print(json.dumps(report, sort_keys=True))
        return report
    finally:
        vec.close()


class ROMPrimitiveEnv(gym.Wrapper):
    def __init__(self, stage, max_frames):
        super().__init__(BattleCityEnv(stage=stage, max_frames=max_frames, reward_profile='sparse'))
        self.action_space = gym.spaces.Discrete(10)

    def action_masks(self):
        return self.env.action_masks()[:10]

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Native transfer accepts only primitive actions 0..9')
        return self.env.step(action)


def evaluate_rom(args):
    _configure(args.device)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.model.with_name('manifest.json').read_text(encoding='utf-8'))
    validate_manifest(manifest)
    if sha256(args.model) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint hash does not match its manifest')
    if sha256(ROOT / 'battle_city.nes') != manifest['rom_sha256']:
        raise ValueError('ROM differs from training stage-initialization reference')
    model = MaskablePPO.load(
        args.model, device=args.device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer, 'rollout_buffer_kwargs': {},
                        'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    if model.action_space.n != 10 or model.observation_space.shape != (94, 26, 26):
        raise ValueError('Checkpoint spaces do not match the primitive native contract')
    episodes = []
    max_frames = manifest['training']['max_frames']
    for stage in args.stages:
        vec = VecFrameStack(DummyVecEnv([lambda: ROMPrimitiveEnv(stage, max_frames)]),
                            n_stack=2, channels_order='first')
        try:
            for seed in args.seeds:
                for deterministic in (True, False):
                    set_random_seed(seed)
                    vec.seed(seed)
                    obs = vec.reset()
                    total = 0.0
                    for _ in range((max_frames + 3) // 4):
                        masks = np.stack(vec.env_method('action_masks'))
                        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
                        obs, reward, done, infos = vec.step(action)
                        total += float(reward[0])
                        if done[0]:
                            info = infos[0]
                            episodes.append({
                                key: info[key] for key in
                                ('outcome', 'frames', 'kills', 'deaths', 'shots', 'pickups', 'lives')})
                            episodes[-1].update(stage=stage, seed=seed,
                                                deterministic=deterministic, reward=total)
                            break
                    else:
                        raise RuntimeError('ROM transfer episode exceeded its configured frame limit')
        finally:
            vec.close()
    report = {
        'backend': 'original-rom-transfer', 'checkpoint_sha256': manifest['checkpoint_sha256'],
        'rom_observer_version': ENV_VERSION, 'episodes': episodes,
        'wins': sum(case['outcome'] == 'clear' for case in episodes),
        'note': 'Transfer outcomes, not paired trajectory parity. ROM shot/kill observer limitations remain.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return report


def compare(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for codec in ('dense', args.codec):
        output = args.output / codec
        command = [sys.executable, str(Path(__file__).resolve()), 'train',
                   '--output', str(output), '--codec', codec, '--device', args.device,
                   '--stages', ','.join(map(str, args.stages))]
        for option in ('steps', 'envs', 'n_steps', 'batch_size', 'epochs',
                       'max_frames', 'seed', 'max_storage_mib'):
            command.extend(['--' + option.replace('_', '-'), str(getattr(args, option))])
        if args.plain_batch:
            command.append('--plain-batch')
        subprocess.run(command, cwd=ROOT, check=True)
        reports[codec] = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    dense, compact = reports['dense'], reports[args.codec]
    report = {
        'results': reports, 'same_final_parameters': dense['policy_sha256'] == compact['policy_sha256'],
        'throughput_ratio': compact['transitions_per_second'] / dense['transitions_per_second'],
        'rollout_storage_ratio': dense['rollout_storage_bytes'] / compact['rollout_storage_bytes'],
        'note': 'Fresh processes, same latest engine/config/seed; only rollout storage differs. Short-run result.',
    }
    write_json(args.output / 'comparison.json', report)
    if not report['same_final_parameters']:
        raise RuntimeError('Dense/compact PPO parameter hashes differ; comparison rejected')
    print(json.dumps(report, sort_keys=True))
    return report


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ('study', 'learn'):
        from rust_learning import main as learning_main
        return learning_main(arguments)
    args = parse_args(arguments)
    return {'train': train, 'eval': evaluate_rom, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
