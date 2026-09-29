"""Bounded fresh Rust learning with held-out ROM selection, not exact resume.

``study`` owns at most two child process trees. ``learn --config`` consumes one
hash-bound record from that study; it is not a historical-checkpoint warm start.
Immutable monitor snapshots are authoritative; display pointers are best effort.
"""
from dataclasses import asdict, dataclass, fields
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import warnings

from rust_resources import ResourceMonitor, ResourcePressureError

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MAX_STEPS = 262144
MAX_TOTAL_STEPS = 1572864
FORMAL_MAX_FRAMES = 12000
PPO_SETTINGS = dict(learning_rate=5e-5, target_kl=0.01, gamma=0.99, gae_lambda=0.95,
                    ent_coef=0.01, clip_range=0.1, vf_coef=0.5)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class StatusUnavailableError(FileNotFoundError):
    """No immutable snapshot or compatibility pointer has been published yet."""


class StatusWriter:
    def __init__(self, directory, name='status'):
        self.directory, self.name = Path(directory), name
        self.previous = None
        sequences = [p.name[len(name) + 1:].split('-', 1)[0]
                     for p in self.directory.glob(f'{name}-*.json')]
        self.sequence = max((int(value) for value in sequences if value.isdigit()), default=0)
        self.warned = set()

    def _warn_once(self, category, message):
        if category not in self.warned:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            self.warned.add(category)

    def _cleanup(self, path, snapshot):
        try:
            path.unlink(missing_ok=True)
        except PermissionError as error:
            self._warn_once(
                'cleanup-' + path.suffix,
                f'Monitor cleanup sharing lock on {path}: {error}; authoritative snapshot path: {snapshot}')

    def write(self, status):
        self.sequence = max(time.time_ns(), self.sequence + 1)
        stem = f'{self.name}-{self.sequence:020d}-{uuid.uuid4().hex}'
        snapshot = self.directory / (stem + '.json')
        temporary = self.directory / (stem + '.tmp')
        try:
            _write_new(temporary, status)
            digest = _hash(temporary)
            if digest == self.previous:
                return
            # Publish a new immutable name only after the complete payload is durable.
            os.rename(temporary, snapshot)
        finally:
            self._cleanup(temporary, snapshot)
        self.previous = digest
        pending = self.directory / (stem + '.pending')
        try:
            with snapshot.open('rb') as source, pending.open('xb') as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, self.directory / (self.name + '.json'))
        except PermissionError as error:
            self._warn_once('pointer', f'Monitor display pointer sharing lock: {error}; '
                            f'authoritative snapshot: {snapshot}')
        finally:
            self._cleanup(pending, snapshot)
        return snapshot


def read_status(directory, name='status'):
    """Read the newest published snapshot; corruption/access failures are fatal."""
    snapshots = sorted(Path(directory).glob(f'{name}-*.json'), reverse=True)
    if snapshots:
        return _read(snapshots[0])
    try:
        return _read(Path(directory) / (name + '.json'))
    except FileNotFoundError as error:
        raise StatusUnavailableError(f'No published {name} status in {directory}') from error


@dataclass(frozen=True)
class TrialConfig:
    output: str
    model: str = 'cnn'
    seed: int = 0
    steps: int = MAX_STEPS
    eval_every: int = 65536
    checkpoint_every: int = 16384
    envs: int = 32
    n_steps: int = 256
    batch_size: int = 256
    epochs: int = 4
    device: str = 'cuda'
    codec: str = 'cuda'
    stages: tuple = (1, 2, 3)
    eval_stages: tuple = (4, 5)
    eval_seeds: tuple = (101, 202)
    confirm_stages: tuple = (6, 7)
    confirm_seeds: tuple = (1001, 2002)
    max_wall_seconds: float = 3600
    patience: int = 3
    max_frames: int = FORMAL_MAX_FRAMES
    max_storage_mib: int = 2048
    initial_ram_file: str = ''

    def to_dict(self):
        return asdict(self)

    def validate(self):
        from rust_models import MODEL_NAMES
        if self.model not in MODEL_NAMES:
            raise ValueError('Unknown model')
        for name in ('steps', 'envs', 'n_steps', 'batch_size', 'epochs', 'eval_every',
                     'checkpoint_every', 'patience', 'max_frames', 'max_storage_mib'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.max_wall_seconds) or not 0 < self.max_wall_seconds <= 3600:
            raise ValueError('max_wall_seconds must be in (0, 3600]')
        if self.steps > MAX_STEPS or self.max_storage_mib > 2048:
            raise ValueError('Approved per-trial step/storage cap exceeded')
        if self.patience > 3:
            raise ValueError('Approved patience cap is three unimproved completed checks')
        if self.envs > 4096 or self.max_frames >= 2**32:
            raise ValueError('Invalid native environment/frame bounds')
        rollout = self.envs * self.n_steps
        if self.batch_size < 2 or rollout % self.batch_size or self.batch_size > rollout:
            raise ValueError('rollout must be divisible by batch_size >= 2')
        for name in ('steps', 'eval_every', 'checkpoint_every'):
            if getattr(self, name) % rollout:
                raise ValueError(f'{name} must be divisible by envs*n_steps; no overshoot')
        if self.device not in ('cpu', 'cuda') or self.codec not in ('dense', 'zlib', 'native', 'cuda'):
            raise ValueError('Unknown device/codec')
        if self.codec == 'cuda' and self.device != 'cuda':
            raise ValueError('CUDA codec requires CUDA device')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must fit uint32')
        for name in ('stages', 'eval_stages', 'confirm_stages', 'eval_seeds', 'confirm_seeds'):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f'{name} must be nonempty and unique')
            if any(type(v) is not int or not (1 <= v <= 35 if 'stages' in name else 0 <= v < 2**32)
                   for v in values):
                raise ValueError(f'Invalid {name}')
        stage_sets = [set(self.stages), set(self.eval_stages), set(self.confirm_stages)]
        if any(stage_sets[i] & stage_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Training, selection and confirmation stages must be disjoint')
        if set(self.eval_seeds) & set(self.confirm_seeds):
            raise ValueError('Selection and confirmation seeds must be disjoint')
        if self.envs < len(self.stages):
            raise ValueError('envs must cover all training stages')
        return self


def _integers(text):
    try:
        result = tuple(int(v) for v in text.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected comma-separated integers') from error
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    learn_parser = commands.add_parser('learn')
    learn_parser.add_argument('--config', type=Path, required=True)
    study_parser = commands.add_parser('study')
    study_parser.add_argument('--output', type=Path, required=True)
    study_parser.add_argument('--models', default='cnn,spatial,history')
    study_parser.add_argument('--seeds', type=_integers, default=(0, 1))
    study_parser.add_argument('--parallel', type=int, choices=(1, 2), default=2)
    defaults = TrialConfig(output='')
    for name in ('steps', 'eval_every', 'checkpoint_every', 'envs', 'n_steps',
                 'batch_size', 'epochs', 'patience', 'max_frames', 'max_storage_mib'):
        study_parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(defaults, name))
    for name in ('stages', 'eval_stages', 'eval_seeds', 'confirm_stages', 'confirm_seeds'):
        study_parser.add_argument('--' + name.replace('_', '-'), type=_integers, default=getattr(defaults, name))
    study_parser.add_argument('--max-wall-seconds', type=float, default=3600)
    study_parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    study_parser.add_argument('--codec', choices=('dense', 'zlib', 'native', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'study':
        try:
            trial_configs(args)
        except ValueError as error:
            parser.error(str(error))
    return args


def trial_configs(args):
    models = args.models.split(',')
    if len(set(models)) != len(models) or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Models and training seeds must be unique')
    if len(models) * len(args.seeds) > 6 or len(models) * len(args.seeds) * args.steps > MAX_TOTAL_STEPS:
        raise ValueError('Approved six-trial/total transition cap exceeded')
    common = {f.name: getattr(args, f.name) for f in fields(TrialConfig)
              if hasattr(args, f.name) and f.name != 'output'}
    return [TrialConfig(**common, output=str(args.output.resolve() / f'{model}-seed{seed}'),
                        model=model, seed=seed).validate() for model in models for seed in args.seeds]


def memory_preflight(config):
    from rust_models import model_spec
    shape = [47 * model_spec(config.model)['n_stack'], 26, 26]
    observation_bytes = config.envs * config.n_steps * math.prod(shape) * 4
    limit = config.max_storage_mib * 1024**2
    dense_bytes = observation_bytes + config.envs * config.n_steps * 68
    if config.codec == 'dense' and dense_bytes > limit:
        raise MemoryError(f'Dense rollout needs at least {dense_bytes} bytes, cap is {limit}')
    return {'observation_shape': shape, 'channels_per_frame': 47,
            'dense_observation_bytes': observation_bytes, 'dense_lower_bound_bytes': dense_bytes,
            'max_storage_bytes': limit, 'decoded_minibatch_bytes': config.batch_size * math.prod(shape) * 4,
            'storage_cap_scope': 'persistent rollout storage, not model/optimizer/CUDA decode workspace'}


def learning_contract(model_name):
    from battle_city_env import CHANNEL_NAMES
    from rust_env import BACKEND_VERSION, RNG_VERSION, SEED_VERSION
    from rust_models import model_spec
    spec = model_spec(model_name)
    return {'backend': BACKEND_VERSION, 'rng': RNG_VERSION, 'seed_derivation': SEED_VERSION,
            'actions': 10, 'channels': list(CHANNEL_NAMES), 'channels_per_frame': 47,
            'n_stack': spec['n_stack'], 'observation_shape': [47 * spec['n_stack'], 26, 26],
            'frame_skip': 4, 'initial_stage_ram_seed': 0,
            'reward': 'sparse-kill1-death-3-base-20-clear5-frame-0.0005',
            'termination': 'full-stage-multi-life-v1',
            'history': 'raw-oldest-to-newest-zero-padded-reset-on-episode-v1',
            'model_spec': spec}


def runtime_signature():
    from battle_city_env import ENV_VERSION
    from rust_env import native_library
    import numpy as np
    import torch
    import stable_baselines3
    import sb3_contrib
    library = native_library()
    python_sources = {p.name: _hash(p) for p in sorted(ROOT.glob('*.py'))}
    native_sources = {}
    for directory, children, names in os.walk(ROOT / 'native'):
        children[:] = sorted(name for name in children if name not in ('target', '.git'))
        for name in sorted(names):
            if name.endswith('.rs'):
                path = Path(directory) / name
                native_sources[str(path.relative_to(ROOT))] = _hash(path)
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs'):
        path = ROOT / 'native' / name
        if path.exists():
            native_sources[str(path.relative_to(ROOT))] = _hash(path)
    commit = subprocess.run(['git', '--no-pager', 'rev-parse', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {'python_sources': python_sources, 'native_sources': native_sources,
            'native_library_sha256': _hash(library._name), 'rom_sha256': _hash(ROOT / 'battle_city.nes'),
            'source_commit': commit, 'rom_observer_version': ENV_VERSION,
            'torch': torch.__version__, 'numpy': np.__version__,
            'stable_baselines3': stable_baselines3.__version__, 'sb3_contrib': sb3_contrib.__version__}


def verify_frozen(provenance):
    if provenance.get('runtime') != runtime_signature():
        raise ValueError('Frozen Python/native/library/ROM/runtime signature changed')
    ram_path = Path(provenance['initial_ram_file'])
    if _hash(ram_path) != provenance['initial_ram_file_sha256']:
        raise ValueError('Frozen stage RAM file changed')
    import numpy as np
    initial = np.load(ram_path, allow_pickle=False)
    if initial.dtype != np.uint8 or initial.shape != (len(provenance['stages']), 2048):
        raise ValueError('Invalid frozen stage RAM shape/dtype')
    if [hashlib.sha256(row.tobytes()).hexdigest() for row in initial] != provenance['initial_ram_sha256']:
        raise ValueError('Frozen stage RAM row hash mismatch')
    return initial


def checkpoint_manifest(config, *, provenance, checkpoint_sha256, step, optimizer_updates):
    return {'rust_learning_schema_version': SCHEMA_VERSION,
            'contract': learning_contract(config.model), 'config': config.to_dict(),
            'ppo': PPO_SETTINGS, 'actor_critic_heads': {'pi': [64, 64], 'vf': [64, 64]},
            'provenance': provenance, 'checkpoint_sha256': checkpoint_sha256,
            'optimized_steps': step, 'optimizer_updates': optimizer_updates,
            'fresh_initialization': True, 'initial_training_seed': config.seed,
            'resume_supported': False, 'load_purpose': 'evaluation-only',
            'memory': memory_preflight(config)}


def validate_learning_manifest(manifest, *, verify_runtime=True):
    if manifest.get('rust_learning_schema_version') != SCHEMA_VERSION:
        raise ValueError('Not a compatible Rust learning checkpoint')
    try:
        config = TrialConfig(**manifest['config']).validate()
        expected = learning_contract(config.model)
        digest = manifest['checkpoint_sha256']
        valid = (manifest['contract'] == expected and manifest['ppo'] == PPO_SETTINGS
                 and manifest['actor_critic_heads'] == {'pi': [64, 64], 'vf': [64, 64]}
                 and manifest['fresh_initialization'] is True
                 and manifest['initial_training_seed'] == config.seed
                 and manifest['resume_supported'] is False
                 and manifest['load_purpose'] == 'evaluation-only'
                 and manifest['memory'] == memory_preflight(config)
                 and type(manifest['optimized_steps']) is int
                 and 0 <= manifest['optimized_steps'] <= config.steps
                 and manifest['optimized_steps'] % (config.envs * config.n_steps) == 0
                 and type(manifest['optimizer_updates']) is int
                 and manifest['optimizer_updates'] >= 0
                 and isinstance(digest, str) and len(digest) == 64
                 and all(c in '0123456789abcdef' for c in digest))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed Rust learning manifest') from error
    if not valid:
        raise ValueError('Incompatible/forged action, model, history or training manifest')
    if verify_runtime:
        verify_frozen(manifest['provenance'])
    return config


def load_learning_checkpoint(path, device='cpu'):
    """Strict local evaluation loader; never allocates the saved training buffer."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    from rust_models import extractor_class
    path = Path(path)
    manifest = _read(str(path) + '.manifest.json')
    config = validate_learning_manifest(manifest)
    if _hash(path) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    model = MaskablePPO.load(
        path, device=device,
        custom_objects={'rollout_buffer_class': MaskableRolloutBuffer, 'rollout_buffer_kwargs': {},
                        'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    if (model.action_space.n != 10
            or tuple(model.observation_space.shape) != tuple(manifest['contract']['observation_shape'])
            or not isinstance(model.policy.features_extractor, extractor_class(config.model))
            or model.num_timesteps != manifest['optimized_steps']
            or model._n_updates != manifest['optimizer_updates']):
        raise ValueError('Serialized checkpoint does not match action/model/history/step manifest')
    return model


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def _training_metrics(model):
    import numpy as np
    import torch

    def check(value, label):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Nonfinite {label} tensor')
        elif isinstance(value, dict):
            for child in value.values():
                check(child, label)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child, label)
        elif isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
            raise FloatingPointError(f'Nonfinite {label} scalar')

    check(model.policy.state_dict(), 'policy')
    check(model.policy.optimizer.state_dict(), 'optimizer')
    values = model.logger.name_to_value
    for key in ('train/loss', 'train/value_loss', 'train/entropy_loss', 'train/approx_kl',
                'train/policy_gradient_loss', 'train/clip_fraction'):
        if key in values and not math.isfinite(float(values[key])):
            raise FloatingPointError(f'Nonfinite PPO optimizer metric: {key}')
    metrics = {key: float(value) for key, value in values.items()
               if key.startswith('train/') and isinstance(value, (int, float, np.number))}
    undefined = {}
    for key, value in metrics.items():
        if not math.isfinite(value):
            metrics[key] = None
            undefined[key] = (
                'Explained variance can be undefined for constant returns; nonfinite diagnostic serialized as null.'
                if key == 'train/explained_variance'
                else 'Nonfinite diagnostic serialized as null; policy, optimizer and loss checks passed.')
    return metrics, undefined


def selection_score(metrics):
    if not _finite(metrics):
        raise FloatingPointError('Nonfinite validation metrics')
    return (int(metrics['rom']['modes']['deterministic']['summary']['wins']),
            float(metrics['skills']['modes']['deterministic']['summary']['mean_success_rate']))


class StopGate:
    def __init__(self, patience):
        self.patience = patience
        self.best_score = None
        self.unimproved_checks = 0
        self.completed_checks = 0

    @property
    def stopped(self):
        return self.unimproved_checks >= self.patience

    def observe(self, metrics, *, completed=True):
        if not completed:
            return False
        score = selection_score(metrics)
        improved = self.best_score is None or score > self.best_score
        self.completed_checks += 1
        if improved:
            self.best_score, self.unimproved_checks = score, 0
        else:
            self.unimproved_checks += 1
        return improved


def run_training_loop(runtime, config, *, evaluate, save, publish, clock=time.monotonic,
                      started=None):
    """After-update cadence; partial collection/validation never masquerades as completed."""
    started = clock() if started is None else started
    deadline = started + config.max_wall_seconds
    gate = StopGate(config.patience)
    report = {'evaluations': [], 'checkpoints': [], 'best': None, 'confirmation': None,
              'stop_reason': None, 'learning_effect_valid': False, 'status': 'running'}

    def snapshot(phase):
        report.update(phase=phase, optimized_steps=runtime.optimized_steps,
                      collected_steps=runtime.collected_steps, optimizer_updates=runtime.optimizer_updates,
                      completed_checks=gate.completed_checks, unimproved_checks=gate.unimproved_checks,
                      wall_seconds=clock() - started, diagnostics=runtime.diagnostics())
        publish(report)

    def checkpoint(role):
        record = save(role, runtime.optimized_steps)
        report['checkpoints'].append({'role': role, **record})
        report[role] = record
        return record

    def validation():
        record = {'step': runtime.optimized_steps, 'status': 'running'}
        report['evaluations'].append(record)
        snapshot('validation')
        try:
            metrics = evaluate(runtime.optimized_steps)
            if clock() >= deadline:
                raise TimeoutError('Validation finished after whole-trial deadline')
            improved = gate.observe(metrics)
            record.update(status='completed', metrics=metrics, selection_score=list(selection_score(metrics)))
            if improved:
                checkpoint('best')
        except (TimeoutError, FloatingPointError, ResourcePressureError) as error:
            record.update(status='incomplete', error=str(error), traceback=traceback.format_exc())
            if hasattr(error, 'partial_results'):
                record['partial_results'] = error.partial_results
            raise
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}',
                          traceback=traceback.format_exc())
            raise
        snapshot('validated')

    try:
        if clock() >= deadline:
            raise TimeoutError('Deadline exhausted during initialization')
        checkpoint('latest')
        validation()
        while runtime.optimized_steps < config.steps:
            if gate.stopped:
                report['stop_reason'] = 'patience'
                break
            if clock() >= deadline:
                raise TimeoutError('Whole-trial deadline exhausted')
            snapshot('training')
            if not runtime.update(deadline):
                raise TimeoutError('Whole-trial deadline exhausted during partial rollout')
            if runtime.optimized_steps > config.steps:
                raise RuntimeError('Optimizer exceeded the requested transition cap')
            if runtime.optimized_steps % config.checkpoint_every == 0:
                checkpoint('latest')
            if clock() >= deadline:
                raise TimeoutError('Whole-trial deadline exhausted after optimizer update')
            if runtime.optimized_steps % config.eval_every == 0 or runtime.optimized_steps == config.steps:
                validation()
        if report['stop_reason'] is None:
            report['stop_reason'] = 'step_cap'
    except TimeoutError as error:
        report.update(stop_reason='wall_cap', error=str(error), traceback=traceback.format_exc())
    except ResourcePressureError as error:
        report.update(stop_reason='resource_pressure', error=str(error), traceback=traceback.format_exc())
    except FloatingPointError as error:
        report.update(stop_reason='nonfinite', error=str(error), traceback=traceback.format_exc())
    except Exception as error:
        report.update(stop_reason='error', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    # A nonfinite optimizer must never become a valid final checkpoint.
    if report['stop_reason'] not in ('nonfinite', 'error', 'resource_pressure'):
        checkpoint('final')
        if clock() >= deadline:
            report['stop_reason'] = 'wall_cap'
    if report['best'] and report['stop_reason'] in ('step_cap', 'patience') and clock() < deadline:
        record = {'step': report['best']['step'], 'status': 'running',
                  'checkpoint': report['best'], 'selection_use': False}
        report['confirmation'] = record
        snapshot('confirmation')
        try:
            metrics = evaluate(report['best']['step'], confirmation=True)
            if clock() >= deadline:
                raise TimeoutError('Confirmation finished after whole-trial deadline')
            selection_score(metrics)
            record.update(status='completed', metrics=metrics)
        except Exception as error:
            record.update(status='incomplete', error=f'{type(error).__name__}: {error}',
                          traceback=traceback.format_exc())
            report.update(error=record['error'], traceback=record['traceback'])
            if isinstance(error, TimeoutError):
                report['stop_reason'] = 'wall_cap'
            elif isinstance(error, ResourcePressureError):
                report['stop_reason'] = 'resource_pressure'
            elif isinstance(error, FloatingPointError):
                report['stop_reason'] = 'nonfinite'
            else:
                report['stop_reason'] = 'error'
    report['learning_effect_valid'] = (
        report['stop_reason'] in ('step_cap', 'patience') and gate.completed_checks >= 2
        and report['confirmation'] is not None and report['confirmation']['status'] == 'completed')
    report['status'] = ('failed' if report['stop_reason'] in ('error', 'nonfinite')
                        else 'completed' if report['learning_effect_valid'] else 'incomplete')
    snapshot('finished')
    return report


class TrainingRuntime:
    def __init__(self, config, initial, deadline, resources=None):
        import numpy as np
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import VecFrameStack
        from rust_env import RustVectorEnv
        from rust_models import extractor_class, model_spec
        from rust_train import _buffer_kwargs, _configure
        _configure(config.device)
        self.resources = resources
        self.config, self.optimized_steps, self.collected_steps = config, 0, 0
        self.episodes, self.updates, self.peak_storage = [], [], 0
        self.undefined_diagnostic_counts = {}
        self.env = VecFrameStack(RustVectorEnv(initial, config.max_frames, seed=config.seed),
                                 n_stack=model_spec(config.model)['n_stack'], channels_order='first')
        try:
            self.model = MaskablePPO(
                'CnnPolicy', self.env, seed=config.seed, device=config.device, n_steps=config.n_steps,
                batch_size=config.batch_size, n_epochs=config.epochs, verbose=0, **PPO_SETTINGS,
                policy_kwargs={'features_extractor_class': extractor_class(config.model),
                               'normalize_images': False,
                               'net_arch': {'pi': [64, 64], 'vf': [64, 64]}},
                **_buffer_kwargs(config.codec, config.max_storage_mib * 1024**2))
            runtime = self

            class BudgetCallback(BaseCallback):
                def _on_step(self):
                    runtime.collected_steps = self.model.num_timesteps
                    if runtime.resources is not None:
                        runtime.resources.check('rollout')
                    if not np.isfinite(self.locals['rewards']).all():
                        raise FloatingPointError('Nonfinite native training reward')
                    for index, info in enumerate(self.locals['infos']):
                        if 'outcome' in info:
                            runtime.episodes.append({
                                key: value for key, value in info.items()
                                if key not in ('terminal_observation',)})
                            runtime.episodes[-1].update(
                                env_index=index, stage=config.stages[index % len(config.stages)],
                                collected_step=self.model.num_timesteps)
                    return time.monotonic() < runtime.deadline

            self.deadline = deadline
            self.total, self.callback = self.model._setup_learn(config.steps, callback=BudgetCallback())
            self.callback.on_training_start(locals(), globals())
            if config.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
        except BaseException:
            self.env.close()
            raise

    @property
    def optimizer_updates(self):
        return self.model._n_updates

    def update(self, deadline):
        self.deadline = deadline
        collected = self.model.collect_rollouts(
            self.env, callback=self.callback, rollout_buffer=self.model.rollout_buffer,
            n_rollout_steps=self.config.n_steps)
        self.collected_steps = self.model.num_timesteps
        if not collected or time.monotonic() >= deadline:
            return False
        if self.resources is not None:
            self.resources.check('optimizer', force=True)
        self.model._update_current_progress_remaining(self.model.num_timesteps, self.total)
        self.model.train()
        metrics, undefined = _training_metrics(self.model)
        for key in undefined:
            self.undefined_diagnostic_counts[key] = self.undefined_diagnostic_counts.get(key, 0) + 1
        self.optimized_steps = self.model.num_timesteps
        self.updates.append({'step': self.optimized_steps, **metrics, 'undefined_diagnostics': undefined})
        self.peak_storage = max(self.peak_storage, self._storage())
        return True

    def _storage(self):
        import numpy as np
        buffer = self.model.rollout_buffer
        return int(getattr(buffer, 'total_storage_nbytes',
                           sum(v.nbytes for v in vars(buffer).values() if isinstance(v, np.ndarray))))

    def diagnostics(self):
        import torch
        return {'completed_episodes': self.episodes, 'optimizer_metrics': self.updates,
                'resources': self.resources.latest if self.resources is not None else None,
                'undefined_diagnostic_counts': self.undefined_diagnostic_counts,
                'rollout_storage_bytes': self._storage(), 'peak_rollout_storage_bytes': self.peak_storage,
                'gpu_decoded_minibatches': getattr(self.model.rollout_buffer, 'gpu_decoded_minibatches', 0),
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()
                if self.config.device == 'cuda' else 0,
                'native_returns_are_diagnostic_only': True}

    def close(self):
        self.callback.on_training_end()
        self.env.close()


def _save_checkpoint(runtime, config, provenance, role, step):
    verify_frozen(provenance)
    path = Path(config.output) / f'{role}-{step:09d}.zip'
    if path.exists() or Path(str(path) + '.manifest.json').exists():
        raise FileExistsError(path)
    collected = runtime.model.num_timesteps
    try:
        # A interrupted rollout changes SB3's counter, not the optimized weights.
        runtime.model.num_timesteps = step
        runtime.model.save(path)
    finally:
        runtime.model.num_timesteps = collected
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    manifest = checkpoint_manifest(config, provenance=provenance, checkpoint_sha256=_hash(path),
                                   step=step, optimizer_updates=runtime.optimizer_updates)
    _write_new(str(path) + '.manifest.json', manifest)
    return {'path': str(path), 'step': step, 'sha256': manifest['checkpoint_sha256'],
            'manifest_sha256': _hash(str(path) + '.manifest.json')}


def _progress(report):
    completed = [r for r in report.get('evaluations', []) if r['status'] == 'completed']
    if len(completed) < 2:
        return {'interpretation': 'Insufficient completed checkpoints to assess learning.'}
    first, last = completed[0]['metrics'], completed[-1]['metrics']
    result = {}
    for backend in ('rust', 'rom'):
        initial = first[backend]['modes']['deterministic']['summary']
        final = last[backend]['modes']['deterministic']['summary']
        result[backend] = {'initial': initial, 'last': final,
                           'win_rate_delta': final['win_rate'] - initial['win_rate']}
    result['skills'] = {'initial': first['skills'], 'last': last['skills']}
    rust_up, rom_up = result['rust']['win_rate_delta'] > 0, result['rom']['win_rate_delta'] > 0
    result['interpretation'] = (
        'Both improve; any additional budget needs separate approval.' if rust_up and rom_up else
        'Rust improves but ROM does not; inspect transfer/distribution differences.' if rust_up else
        'ROM improves without Rust win-rate improvement; inspect raw outcomes and skills.' if rom_up else
        'Neither win rate improves; inspect learning/task design and skill results.')
    result['caution'] = 'Descriptive within-trial metrics, not significance or a cross-model winner.'
    return result


def run_trial(config, provenance):
    started = time.monotonic()
    config.validate()
    directory = Path(config.output)
    directory.mkdir(parents=False, exist_ok=False)
    writer = StatusWriter(directory)
    runtime = None
    report = None
    last_report = None
    checkpoints = {}
    resources = ResourceMonitor(directory / 'resources.jsonl', device=config.device)

    def publish(status):
        nonlocal last_report
        last_report = status
        writer.write(status)

    try:
        resources.check('initializing', admission_slots=1, force=True)
        initial_stages = verify_frozen(provenance)
        if list(config.stages) != provenance['stages']:
            raise ValueError('Trial stage mapping differs from frozen RAM')
        memory = memory_preflight(config)
        import numpy as np
        initial = np.stack([initial_stages[i % len(initial_stages)] for i in range(config.envs)])
        _write_new(directory / 'config.json', {'config': config.to_dict(), 'provenance': provenance,
                                              'contract': learning_contract(config.model), 'memory': memory,
                                              'initial_env_ram_sha256': [
                                                  hashlib.sha256(row.tobytes()).hexdigest() for row in initial]})
        publish({'phase': 'initializing', 'status': 'running', 'optimized_steps': 0})
        deadline = started + config.max_wall_seconds
        runtime = TrainingRuntime(config, initial, deadline, resources=resources)

        def save(role, step):
            resources.check('checkpoint', force=True)
            record = _save_checkpoint(runtime, config, provenance, role, step)
            checkpoints[role] = record
            return record

        def evaluate(step, confirmation=False):
            from rust_validation import evaluate_suite
            verify_frozen(provenance)
            resources.check('confirmation' if confirmation else 'validation', force=True)
            model = (load_learning_checkpoint(checkpoints['best']['path'], device=config.device)
                     if confirmation else runtime.model)
            return evaluate_suite(
                model, model_name=config.model,
                stages=config.confirm_stages if confirmation else config.eval_stages,
                seeds=config.confirm_seeds if confirmation else config.eval_seeds,
                max_frames=FORMAL_MAX_FRAMES, deadline=deadline,
                resource_check=lambda: resources.check('validation'))

        report = run_training_loop(runtime, config, evaluate=evaluate, save=save,
                                   publish=publish, started=started)
    except Exception as error:
        report = last_report if last_report is not None else {'checkpoints': list(checkpoints.values())}
        pressure = isinstance(error, ResourcePressureError)
        report.update(status='incomplete' if pressure else 'failed',
                      stop_reason='resource_pressure' if pressure else 'error', learning_effect_valid=False,
                      error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(),
                      optimized_steps=runtime.optimized_steps if runtime else 0,
                      collected_steps=runtime.collected_steps if runtime else 0,
                      wall_seconds=time.monotonic() - started)
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as error:
                if report is None:
                    raise
                report.update(status='failed', stop_reason='error', learning_effect_valid=False,
                              error=f"{report.get('error', '')} Cleanup {type(error).__name__}: {error}".strip(),
                              cleanup_traceback=traceback.format_exc())
    if report.get('error'):
        failure = {key: report.get(key) for key in (
            'status', 'stop_reason', 'optimized_steps', 'collected_steps',
            'error', 'traceback', 'cleanup_traceback', 'latest', 'best')}
        failure.update(model=config.model, seed=config.seed, resources=resources.latest,
                       learning_effect_valid=False, automatic_retry=False)
        _write_new(directory / 'failure.json', failure)
    report.update(model=config.model, seed=config.seed, config=config.to_dict(), provenance=provenance,
                  exposure_exact=True,
                  progress=_progress(report), wall_seconds=time.monotonic() - started,
                  timing='shared-resource timing; not uncontended architecture throughput',
                  checkpoint_retention='All immutable latest/best/final checkpoints retained; no deletion.',
                  checkpoint_storage_bytes=sum(p.stat().st_size for p in directory.glob('*.zip')),
                  total_output_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()))
    # Critical result/report failures are not swallowed or turned into a successful exit.
    _write_new(directory / 'result.json', report)
    write_results(directory / 'RESULTS.md', [report])
    _write_new(directory / 'result.manifest.json',
               {'rust_learning_schema_version': SCHEMA_VERSION, 'result_sha256': _hash(directory / 'result.json'),
                'report_sha256': _hash(directory / 'RESULTS.md'), 'learning_effect_valid': report['learning_effect_valid'],
                'status': report['status'], 'provenance': provenance})
    writer.write(report)
    return report


def write_results(path, reports):
    lines = ['# Bounded Rust learning results', '',
             'Fresh 10-action models; ROM determines selection. Skills are not full-game wins.',
             'Formal validation allows 12,000 frames. Timing may share a GPU.',
             'No trajectory-parity, statistical-significance or automatic-winner claim.', '',
             '| Model | Seed | State | Stop | Optimized / collected | Seconds | Checkpoint bytes |',
             '|---|---:|---|---|---:|---:|---:|']
    for report in reports:
        lines.append(f"| {report.get('model', '?')} | {report.get('seed', '?')} | {report['status']} | "
                     f"{report.get('stop_reason', '?')} | {report.get('optimized_steps', 0)} / "
                     f"{report.get('collected_steps', 0)} | {report.get('wall_seconds', 0):.1f} | "
                     f"{report.get('checkpoint_storage_bytes', 0)} |")
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
        for report in reports:
            stream.write(f"\n## {report.get('model', '?')} seed {report.get('seed', '?')}\n"
                         f"Learning-effect evaluation valid: {report.get('learning_effect_valid', False)}.\n"
                         'All latest/best/final checkpoints are retained. See result.json for every raw case.\n'
                         '\n```json\n')
            json.dump(report.get('progress', {}), stream, indent=2, allow_nan=False)
            stream.write('\n```\n')
            if report.get('error'):
                stream.write('Error: ' + report['error'] + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_study(args):
    from dataclasses import replace
    import numpy as np
    from rust_env import stage_ram
    configs = trial_configs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_signature()
    initial = np.stack([stage_ram(stage, seed=0) for stage in args.stages])
    ram_path = output / 'initial-ram.npy'
    with ram_path.open('xb') as stream:
        np.save(stream, initial, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    provenance = {'runtime': runtime, 'stages': list(args.stages), 'initial_ram_seed': 0,
                  'initial_ram_file': str(ram_path), 'initial_ram_file_sha256': _hash(ram_path),
                  'initial_ram_sha256': [hashlib.sha256(row.tobytes()).hexdigest() for row in initial]}
    records = []
    record_dir = output / 'trial-configs'
    record_dir.mkdir()
    for config in configs:
        config = replace(config, initial_ram_file=str(ram_path))
        path = record_dir / f'{config.model}-seed{config.seed}.json'
        _write_new(path, {'config': config.to_dict(), 'provenance': provenance})
        records.append({'path': str(path), 'sha256': _hash(path), 'output': config.output,
                        'model': config.model, 'seed': config.seed})
    manifest = {'rust_learning_schema_version': SCHEMA_VERSION, 'provenance': provenance,
                'trials': records, 'total_step_cap': sum(c.steps for c in configs),
                'parallel': args.parallel, 'automatic_retry': False}
    _write_new(output / 'study.manifest.json', manifest)
    verify_frozen(provenance)
    return manifest


def load_trial_record(path):
    path = Path(path).resolve()
    manifest = _read(path.parent.parent / 'study.manifest.json')
    if manifest.get('rust_learning_schema_version') != SCHEMA_VERSION:
        raise ValueError('Unknown study manifest')
    matches = [record for record in manifest['trials'] if Path(record['path']).resolve() == path]
    if len(matches) != 1 or _hash(path) != matches[0]['sha256']:
        raise ValueError('Trial config is not hash-bound to this study')
    record = _read(path)
    if record['provenance'] != manifest['provenance']:
        raise ValueError('Trial provenance differs from study')
    config = TrialConfig(**record['config']).validate()
    if (config.output != matches[0]['output'] or config.model != matches[0]['model']
            or config.seed != matches[0]['seed']
            or config.initial_ram_file != record['provenance']['initial_ram_file']):
        raise ValueError('Trial identity differs from study record')
    verify_frozen(record['provenance'])
    return config, record['provenance']


def terminate_owned_process(process):
    """Terminate exactly the owned PID tree, including Windows venv launchers."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       check=False, capture_output=True)
    else:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def study(args):
    manifest = prepare_study(args)
    output = args.output.resolve()
    writer = StatusWriter(output, 'batch')
    pending = list(manifest['trials'])
    active, results, states = {}, [], {}
    resources = ResourceMonitor(output / 'resources.jsonl')
    try:
        while pending or active:
            resources.check('supervisor')
            while pending and len(active) < args.parallel:
                resources.check('admission', admission_slots=len(active) + 1, force=True)
                record = pending.pop(0)
                verify_frozen(manifest['provenance'])
                log = (output / f"{record['model']}-seed{record['seed']}.log").open('xb')
                command = [sys.executable, str(ROOT / 'rust_train.py'), 'learn', '--config', record['path']]
                try:
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                        start_new_session=os.name != 'nt')
                except BaseException:
                    log.close()
                    raise
                active[process.pid] = (process, record, log, time.monotonic())
                states[record['output']] = {'status': 'running', 'pid': process.pid}
            for pid, (process, record, log, started) in list(active.items()):
                result_path = Path(record['output']) / 'result.json'
                try:
                    states[record['output']]['monitor'] = read_status(record['output'])
                except StatusUnavailableError:
                    pass
                if time.monotonic() - started >= args.max_wall_seconds and process.poll() is None:
                    terminate_owned_process(process)
                    states[record['output']]['forced_wall_cleanup'] = True
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[pid]
                forced_wall = states[record['output']].get('forced_wall_cleanup', False)
                recovery_required = forced_wall or not result_path.exists()
                if forced_wall:
                    report = dict(states[record['output']].get('monitor', {}))
                    report.update(model=record['model'], seed=record['seed'], status='incomplete',
                                  stop_reason='wall_cap', learning_effect_valid=False,
                                  wall_seconds=time.monotonic() - started,
                                  error='Supervisor terminated owned PID tree at whole-trial wall cap.',
                                  exposure_exact=False,
                                  exposure_note='Step counts are last observed lower bounds after forced termination.')
                elif result_path.exists():
                    try:
                        report = _read(result_path)
                        result_manifest = _read(Path(record['output']) / 'result.manifest.json')
                        if (_hash(result_path) != result_manifest['result_sha256']
                                or _hash(Path(record['output']) / 'RESULTS.md') != result_manifest['report_sha256']
                                or report.get('provenance', {}) != manifest['provenance']
                                or report['model'] != record['model'] or report['seed'] != record['seed']):
                            raise ValueError('Child result signature/identity mismatch')
                        if code != 0 and report['status'] != 'failed':
                            report.update(status='failed', stop_reason='child_exit',
                                          error=f'Child exited {code}', learning_effect_valid=False)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        report = dict(states[record['output']].get('monitor', {}))
                        report.update(model=record['model'], seed=record['seed'], status='failed',
                                      stop_reason='result_error', error=f'{type(error).__name__}: {error}',
                                      learning_effect_valid=False, exposure_exact=False)
                        recovery_required = True
                else:
                    report = dict(states[record['output']].get('monitor', {}))
                    report.update(model=record['model'], seed=record['seed'], status='failed',
                                  stop_reason='child_exit', error=f'Child exited {code} without durable result',
                                  learning_effect_valid=False, exposure_exact=False)
                if recovery_required:
                    recovery = Path(record['output'])
                    recovery.mkdir(exist_ok=True)
                    _write_new(recovery / 'supervisor-result.json', report)
                results.append(report)
                states[record['output']].update(status=report['status'], exit_code=code)
            writer.write({'status': 'running' if pending or active else 'finished', 'trials': states,
                          'pending': len(pending), 'active': len(active), 'completed': len(results)})
            if active:
                time.sleep(0.5)
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        traceback.print_exc()
        for _, record, _, _ in active.values():
            states[record['output']]['status'] = 'interrupted'
        for record in pending:
            states[record['output']] = {'status': 'not_started'}
        failure = {
            'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'trials': {path: {key: value for key, value in state.items() if key != 'monitor'}
                       for path, state in states.items()},
            'pending': len(pending), 'active': 0, 'completed': len(results),
            'error': f'{type(error).__name__}: {error}', 'traceback': failure_traceback,
            'resources': resources.latest, 'automatic_retry': False, 'exposure_exact': False,
            'note': 'Supervisor aborted; use signed per-trial results where present, otherwise last checkpoints.',
        }
        _write_new(output / 'supervisor-failure.json', failure)
        writer.write(failure)
        raise
    finally:
        for process, _, log, _ in active.values():
            terminate_owned_process(process)
            log.close()
    result = {'rust_learning_schema_version': SCHEMA_VERSION, 'trials': results,
              'total_step_cap': manifest['total_step_cap'],
              'optimized_steps': sum(r.get('optimized_steps', 0) for r in results),
              'collected_steps': sum(r.get('collected_steps', 0) for r in results),
              'exposure_exact': all(r.get('exposure_exact', True) for r in results),
              'all_learning_effect_evaluations_valid': all(r['learning_effect_valid'] for r in results),
              'automatic_winner': None, 'automatic_retry': False, 'provenance': manifest['provenance']}
    _write_new(output / 'result.json', result)
    write_results(output / 'RESULTS.md', results)
    _write_new(output / 'result.manifest.json',
               {'result_sha256': _hash(output / 'result.json'), 'report_sha256': _hash(output / 'RESULTS.md'),
                'all_learning_effect_evaluations_valid': result['all_learning_effect_evaluations_valid']})
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'study':
        result = study(args)
        return 1 if any(r['status'] == 'failed' for r in result['trials']) else 0
    config, provenance = load_trial_record(args.config)
    result = run_trial(config, provenance)
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
