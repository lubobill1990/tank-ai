"""Direct original-ROM PPO training under the ten-primitive-action contract.

Gold-standard arm for the RNG ablation: the same MaskablePPO budget and
hyperparameters as ``rust_train.py train``, but every transition comes from the
local ROM via nes_py. No Rust mirror is involved, so there is no simulation or
randomness gap; the costs are wall-clock, CPU and no per-episode seed replay.

The process restricts itself (and its spawned environment workers, which
inherit both settings on Windows) to all-but-two CPU cores at below-normal
priority, keeping the host responsive.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).parent
BACKEND = 'original-rom-direct-v1'
RESERVED_CORES = 2
BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def limit_host_footprint():
    cores = os.cpu_count() or 1
    usable = max(1, cores - RESERVED_CORES)
    kernel32 = ct.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ct.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ct.c_void_p, ct.c_size_t]
    kernel32.SetPriorityClass.argtypes = [ct.c_void_p, ct.c_uint32]
    handle = kernel32.GetCurrentProcess()
    if not kernel32.SetProcessAffinityMask(handle, (1 << usable) - 1):
        raise ct.WinError(ct.get_last_error())
    if not kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS):
        raise ct.WinError(ct.get_last_error())
    return usable


def make_factory(stage, max_frames):
    def factory():
        # Keep workers light: battle_city_env only, never torch/sb3 (a 32-way
        # concurrent torch import storm can leave a worker's numpy partially
        # initialized under host memory pressure).
        import gymnasium as gym
        from battle_city_env import BattleCityEnv

        class PrimitiveEnv(gym.Wrapper):
            """Worker-side copy of rust_train.ROMPrimitiveEnv."""
            def __init__(self):
                super().__init__(BattleCityEnv(stage=stage, max_frames=max_frames,
                                               reward_profile='sparse'))
                self.action_space = gym.spaces.Discrete(10)

            def action_masks(self):
                return self.env.action_masks()[:10]

            def step(self, action):
                if not self.action_space.contains(action):
                    raise ValueError('Direct ROM training accepts only primitive actions 0..9')
                return self.env.step(action)

        return PrimitiveEnv()
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1048576)
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--stages', default='1,2,3')
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    args.stages = [int(s) for s in str(args.stages).split(',')]
    rollout = args.envs * args.n_steps
    if args.steps % rollout or rollout % args.batch_size:
        raise SystemExit('steps must be a multiple of envs*n_steps, divisible by batch_size')
    usable = limit_host_footprint()

    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

    from battle_city_env import ENV_VERSION
    from experiments import sha256, write_json
    from policy import Grid26CNN
    from rust_train import EpisodeResults, _configure, _policy_hash

    class RomEpisodeResults(EpisodeResults):
        """The ROM env reports outcome='' on every non-terminal step."""
        def _on_step(self):
            if not np.isfinite(self.locals['rewards']).all():
                raise FloatingPointError('Nonfinite ROM reward')
            for info in self.locals['infos']:
                if info.get('outcome'):
                    self.outcomes[info['outcome']] = self.outcomes.get(info['outcome'], 0) + 1
                    self.kills += info['kills']
                    self.deaths += info['deaths']
                    self.pickups += info['pickups']
            return True

    _configure(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'backend': BACKEND, 'rom_observer_version': ENV_VERSION,
        'training': {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        'contract_shape': {'actions': 10, 'observation_shape': [94, 26, 26], 'n_stack': 2,
                           'frame_skip': 4, 'reward': 'sparse', 'termination': 'rom-env-v1'},
        'rom_sha256': sha256(ROOT / 'battle_city.nes'),
        'torch': torch.__version__, 'numpy': np.__version__,
        'usable_cores': usable, 'resume_supported': False,
        'note': 'Direct original-ROM training; episode randomness is the ROM boot-phase '
                'jitter and is not replayable per episode seed.',
    }
    write_json(args.output / 'config.json', manifest)
    factories = [make_factory(args.stages[i % len(args.stages)], args.max_frames)
                 for i in range(args.envs)]
    vec = VecFrameStack(SubprocVecEnv(factories, start_method='spawn'),
                        n_stack=2, channels_order='first')
    try:
        model = MaskablePPO(
            'CnnPolicy', vec, seed=args.seed, device=args.device, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=args.epochs, learning_rate=2.5e-4,
            gamma=0.99, gae_lambda=0.95, verbose=0,
            policy_kwargs={'features_extractor_class': Grid26CNN, 'normalize_images': False})
        callback = RomEpisodeResults()
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        elapsed = time.perf_counter() - start
        if model.num_timesteps != args.steps:
            raise RuntimeError('PPO did not honor the exact requested timestep budget')
        if not all(torch.isfinite(value).all().item()
                   for value in model.policy.state_dict().values()):
            raise FloatingPointError('Nonfinite policy parameters')
        model_path = args.output / 'model.zip'
        model.save(model_path)
        manifest['checkpoint_sha256'] = sha256(model_path)
        write_json(args.output / 'manifest.json', manifest)
        report = {
            'backend': BACKEND, 'device': args.device, 'steps': model.num_timesteps,
            'training_seconds': elapsed,
            'transitions_per_second': model.num_timesteps / elapsed,
            'policy_sha256': _policy_hash(model.policy),
            'completed_episode_outcomes': callback.outcomes,
            'completed_episode_kills': callback.kills,
            'completed_episode_deaths': callback.deaths,
            'completed_episode_pickups': callback.pickups,
            'usable_cores': usable,
            'note': 'Shared-host timing; not an isolated speed benchmark.',
        }
        write_json(args.output / 'result.json', report)
        print(json.dumps(report, sort_keys=True))
    finally:
        vec.close()


if __name__ == '__main__':
    main()
