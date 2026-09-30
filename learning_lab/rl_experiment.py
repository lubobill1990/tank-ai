"""Small causal reward comparisons; not a replacement for the frozen study."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

from learning_lab.budget import ResearchBudget
from learning_lab.evaluation import evaluate_candidate
from learning_lab.rewards import RewardProfileVecEnv
from rust_env import stage_ram
from rust_learning import (
    PPO_SETTINGS, StatusWriter, TrainingRuntime, TrialConfig, _hash, _write_new, runtime_signature,
)
from rust_resources import ResourceMonitor
from rust_train import _policy_hash

VERSION = 'lab-reward-comparison-v1'


def source_signature():
    return {
        'runtime': runtime_signature(),
        'lab_sources': {name: _hash(Path(__file__).parent / name) for name in
                        ('rl_experiment.py', 'rewards.py', 'budget.py', 'evaluation.py')},
    }


def load_checkpoint(path, *, device='cpu'):
    path = Path(path)
    manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
    if (manifest.get('version') != VERSION or manifest.get('source') != source_signature()
            or manifest.get('checkpoint_sha256') != _hash(path)):
        raise ValueError('Incompatible or changed laboratory checkpoint/provenance')
    model = MaskablePPO.load(
        path, device=device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer,
                        'rollout_buffer_kwargs': {}, 'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    if (model.action_space.n != 10 or model.observation_space.shape != (94, 26, 26)
            or model.num_timesteps != manifest['optimized_steps']):
        raise ValueError('Laboratory checkpoint spaces or optimized step mismatch')
    return model


def run_trial(*, output, budget, profile, steps=65536, seed=0, envs=32, n_steps=256,
              batch_size=256, epochs=4, checkpoint_every=16384, device='cuda', evaluate=True):
    if profile not in ('baseline', 'loss_balanced_v1'):
        raise ValueError('Unknown comparison reward profile')
    config = TrialConfig(
        output=str(output), model='cnn', seed=seed, steps=steps, envs=envs, n_steps=n_steps,
        batch_size=batch_size, epochs=epochs, checkpoint_every=checkpoint_every,
        stages=(1,), eval_stages=(2, 3), device=device,
        codec='cuda' if device == 'cuda' else 'native').validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    token = budget.reserve('ppo-' + hashlib.sha256(str(output).encode()).hexdigest()[:16], steps)
    deadline = time.monotonic() + min(1800, budget.snapshot()['remaining_seconds'])
    resources = ResourceMonitor(output / 'resources.jsonl', device=device)
    writer = StatusWriter(output)
    report = {
        'version': VERSION, 'status': 'initializing', 'profile': profile, 'seed': seed,
        'config': config.to_dict(), 'ppo': PPO_SETTINGS, 'reservation': token,
        'optimized_steps': 0, 'collected_steps': 0, 'checkpoints': [],
        'learning_effect_valid': False, 'role': 'development-reward-ablation',
        'evaluation_completed': False,
        'initialization': 'Same explicit post-wrapper vector reset for both profiles; no control steps precede it.',
    }
    runtime = None
    settled = False
    started = time.monotonic()

    def save():
        if source_signature() != report['source']:
            raise ValueError('Laboratory sources changed during training')
        resources.check('checkpoint', force=True)
        path = output / f'checkpoint-{runtime.optimized_steps:09d}.zip'
        actual_counter = runtime.model.num_timesteps
        try:
            runtime.model.num_timesteps = runtime.optimized_steps
            runtime.model.save(path)
        finally:
            runtime.model.num_timesteps = actual_counter
        manifest = {
            'version': VERSION, 'source': report['source'], 'profile': profile,
            'config': config.to_dict(), 'ppo': PPO_SETTINGS,
            'optimized_steps': runtime.optimized_steps,
            'checkpoint_sha256': _hash(path), 'load_purpose': 'laboratory-evaluation-only',
        }
        _write_new(str(path) + '.manifest.json', manifest)
        record = {'path': str(path), 'step': runtime.optimized_steps, 'sha256': manifest['checkpoint_sha256']}
        report['checkpoints'].append(record)
        report['final_checkpoint'] = record

    try:
        resources.check('initializing', admission_slots=1, force=True)
        report['source'] = source_signature()
        initial = np.stack([stage_ram(1, seed=0)] * envs)
        report['initial_ram_sha256'] = hashlib.sha256(initial[0].tobytes()).hexdigest()
        runtime = TrainingRuntime(config, initial, deadline, resources=resources)
        # Both profiles take this same reset, avoiding stale return accumulators
        # when installing a laboratory-only wrapper on the existing runtime.
        runtime.env = RewardProfileVecEnv(runtime.env, profile=profile)
        runtime.model.set_env(runtime.env, force_reset=True)
        runtime.model._last_obs = runtime.env.reset()
        runtime.model._last_episode_starts = np.ones(envs, dtype=bool)
        report['initial_policy_sha256'] = _policy_hash(runtime.model.policy)
        save()
        while runtime.optimized_steps < steps:
            if time.monotonic() >= deadline:
                break
            report.update(status='training', optimized_steps=runtime.optimized_steps,
                          collected_steps=runtime.collected_steps)
            writer.write(report)
            if not runtime.update(deadline):
                break
            if runtime.optimized_steps > steps:
                raise RuntimeError('Laboratory training exceeded reserved interactions')
            if runtime.optimized_steps % checkpoint_every == 0:
                save()
        report.update(optimized_steps=runtime.optimized_steps, collected_steps=runtime.collected_steps,
                      diagnostics=runtime.diagnostics(), final_policy_sha256=_policy_hash(runtime.model.policy))
        budget.settle(token, runtime.collected_steps)
        settled = True
        if report['final_checkpoint']['step'] != runtime.optimized_steps:
            save()
        report['status'] = 'completed' if runtime.optimized_steps == steps else 'incomplete'
        if evaluate and report['status'] == 'completed':
            writer.write({**report, 'status': 'evaluating'})
            report['evaluation'] = evaluate_candidate(
                runtime.model, model_name='cnn', output=output / 'development.json',
                budget=budget, stages=(1,), seeds=(0, 1))
            report['evaluation_completed'] = True
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if runtime is not None:
            report.update(optimized_steps=runtime.optimized_steps, collected_steps=runtime.collected_steps)
            runtime.close()
    report.update(wall_seconds=time.monotonic() - started, exposure_exact=settled,
                  reservation_note='Settled actual decisions.' if settled else
                  'Failed work remains charged at its full reserved upper bound.')
    _write_new(output / 'result.json', report)
    writer.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--profiles', default='baseline,loss_balanced_v1')
    args = parser.parse_args(argv)
    budget = ResearchBudget(args.budget)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for profile in args.profiles.split(','):
        result = run_trial(output=args.output / profile, budget=budget,
                           profile=profile, steps=args.steps, seed=args.seed)
        reports.append(result)
        if result['status'] != 'completed':
            return 1
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    if len(reports) > 1 and len({r['initial_policy_sha256'] for r in reports}) != 1:
        raise RuntimeError('Paired reward comparison did not start from identical policy weights')
    _write_new(args.output / 'comparison.json', {'trials': reports, 'budget': budget.snapshot()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
