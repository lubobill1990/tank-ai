"""Offline masked behavior cloning; development accuracy is not full-game success.

Run ``python -m learning_lab.imitation --datasets CORPUS ... --output NEW_DIR``.
Deadlines are absolute ``time.monotonic()`` values. No environment is stepped.
Checkpoints are trusted-local SB3 archives with separate lab-only manifests; this
module never calls, weakens, or substitutes for the production strict loaders.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F


_FORMAT = 'tank-ai-imitation-checkpoint-v1'
_ROOT = Path(__file__).resolve().parent.parent
_WARNING = ('Episode-held-out development accuracy is not full-game success or '
            'a learned-model ROM clear. Warm-start splits apply to this run only.')
_CONTRACT = {
    'algorithm': 'MaskablePPO', 'policy': 'CnnPolicy', 'extractor': 'cnn',
    'observation_shape': [94, 26, 26], 'observation_dtype': 'float32',
    'n_stack': 2, 'channels_order': 'first', 'actions': 10,
    'normalize_images': False, 'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
    'objective': 'masked_teacher_cross_entropy', 'value_function_supervision': False,
}


class _DeadlineReached(Exception):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise _DeadlineReached('Absolute monotonic deadline reached')


def _sha256(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            _check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _source_hashes():
    return {name: _sha256(_ROOT.joinpath(*name.split('/'))) for name in (
        'learning_lab/imitation.py', 'rust_models.py', 'policy.py')}


def _json_write(path, data):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
                       encoding='utf-8')
    pending.replace(path)


def _dataset_identity(path, namespace, metadata, deadline):
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'Dataset has no files: {path}')
    entries = [{'path': p.name if path.is_file() else p.relative_to(path).as_posix(),
                'bytes': p.stat().st_size, 'sha256': _sha256(p, deadline)} for p in files]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode('utf-8')).hexdigest()
    return {'path': str(path), 'namespace': namespace, 'sha256': digest, 'files': entries,
            'metadata': json.loads(json.dumps(metadata, allow_nan=False))}


def _split_episodes(episode_ids, seed):
    episodes = np.unique(episode_ids)
    if len(episodes) < 2:
        raise ValueError('At least two whole episodes are required for train/validation isolation')
    shuffled = np.random.default_rng(seed).permutation(episodes)
    count = max(1, math.ceil(len(episodes) * 0.2))
    validation = np.isin(episode_ids, shuffled[:count])
    train_indices, validation_indices = np.flatnonzero(~validation), np.flatnonzero(validation)
    return train_indices, validation_indices, {
        'method': 'seeded-whole-episode-80-20', 'seed': int(seed),
        'train_episode_ids': sorted(shuffled[count:].tolist()),
        'validation_episode_ids': sorted(shuffled[:count].tolist()),
        'train_samples': len(train_indices), 'validation_samples': len(validation_indices),
    }


def _validate_batch(batch, expected_episodes):
    count = len(expected_episodes)
    specs = {'observations': ((count, 94, 26, 26), np.float32),
             'actions': ((count,), np.int64), 'action_masks': ((count, 10), np.bool_),
             'episode_ids': ((count,), np.int64)}
    for name, (shape, dtype) in specs.items():
        value = batch.get(name)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f'Dataset {name} must have shape {shape} and dtype {np.dtype(dtype)}')
    if not np.array_equal(batch['episode_ids'], expected_episodes):
        raise ValueError('Dataset batch episode IDs disagree with its split index')
    if not np.isfinite(batch['observations']).all():
        raise ValueError('Dataset observations must be finite')
    actions, masks = batch['actions'], batch['action_masks']
    if np.any(actions < 0) or np.any(actions >= 10) or not masks.any(axis=1).all() \
            or not masks[np.arange(count), actions].all():
        raise ValueError('Every teacher action must be legal under its nonempty action mask')


class _Corpus:
    def __init__(self, paths, output, deadline):
        from learning_lab.demonstrations import DemonstrationDataset
        self.datasets, self.identities, self.offsets = [], [], [0]
        keys = []
        try:
            for index, path in enumerate(paths):
                _check_deadline(deadline)
                if not path.exists() or output.is_relative_to(path):
                    raise ValueError('Dataset must exist and must not contain the output directory')
                dataset = DemonstrationDataset(path)
                self.datasets.append(dataset)
                ids = np.asarray(dataset.episode_ids)
                if ids.shape != (len(dataset),) or ids.dtype != np.int64 or len(ids) == 0:
                    raise ValueError('Dataset requires nonempty int64 episode_ids, one per sample')
                namespace = f'dataset{index}'
                keys.extend(f'{namespace}:{int(episode)}' for episode in ids)
                self.offsets.append(self.offsets[-1] + len(ids))
                self.identities.append(_dataset_identity(
                    path, namespace, dataset.metadata, deadline))
            self.episode_ids = np.asarray(keys)
        except BaseException:
            self.close()
            raise

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        count = len(indices)
        result = {'observations': np.empty((count, 94, 26, 26), np.float32),
                  'actions': np.empty(count, np.int64),
                  'action_masks': np.empty((count, 10), bool)}
        owners = np.searchsorted(self.offsets[1:], indices, side='right')
        for owner in np.unique(owners):
            positions = np.flatnonzero(owners == owner)
            local = indices[positions] - self.offsets[owner]
            dataset = self.datasets[owner]
            batch = dataset.batch(local)
            _validate_batch(batch, dataset.episode_ids[local])
            for name in result:
                result[name][positions] = batch[name]
        return result

    def close(self):
        errors = []
        for dataset in self.datasets:
            if callable(getattr(dataset, 'close', None)):
                try:
                    dataset.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise RuntimeError('; '.join(str(error) for error in errors))


class _OfflineSpaces(gym.Env):
    """Spaces-only construction boundary: accidentally starting rollouts is an error."""
    observation_space = gym.spaces.Box(-1, 1, (94, 26, 26), dtype=np.float32)
    action_space = gym.spaces.Discrete(10)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError('Offline imitation must not reset an environment')

    def step(self, action):
        raise RuntimeError('Offline imitation must not step an environment')


def _new_model(learning_rate, seed, device):
    from sb3_contrib import MaskablePPO
    from rust_models import extractor_class
    return MaskablePPO(
        'CnnPolicy', _OfflineSpaces(), learning_rate=learning_rate, seed=seed, device=device,
        n_steps=2, batch_size=2, n_epochs=1, verbose=0,
        policy_kwargs={'features_extractor_class': extractor_class('cnn'),
                       'normalize_images': False,
                       'net_arch': {'pi': [64, 64], 'vf': [64, 64]}})


def _masked_cross_entropy(logits, actions, masks):
    if logits.ndim != 2 or masks.shape != logits.shape or masks.dtype != torch.bool \
            or actions.shape != (len(logits),) or actions.dtype != torch.long \
            or len(logits) == 0:
        raise ValueError('Invalid masked categorical batch')
    if not torch.isfinite(logits).all() or not masks.any(dim=1).all() \
            or (actions < 0).any() or (actions >= logits.shape[1]).any() \
            or not masks.gather(1, actions[:, None]).all():
        raise ValueError('Teacher actions must be legal and logits finite')
    return F.cross_entropy(logits.masked_fill(~masks, float('-inf')), actions)


def _batch_loss(model, batch):
    observations = torch.as_tensor(batch['observations'], device=model.device)
    actions = torch.as_tensor(batch['actions'], device=model.device)
    masks = torch.as_tensor(batch['action_masks'], device=model.device)
    logits = model.policy.get_distribution(observations).distribution.logits
    loss = _masked_cross_entropy(logits, actions, masks)
    correct = (logits.masked_fill(~masks, float('-inf')).argmax(dim=1) == actions).sum().item()
    return loss, int(correct)


def _evaluate(model, corpus, indices, batch_size, deadline, monitor, report):
    model.policy.set_training_mode(False)
    total, correct, count = 0., 0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            _check_deadline(deadline)
            if monitor is not None:
                monitor.check('imitation_validation')
            selection = indices[start:start + batch_size]
            batch = corpus.batch(selection)
            _check_deadline(deadline)
            loss, matches = _batch_loss(model, batch)
            total += float(loss.item()) * len(selection)
            correct += matches
            count += len(selection)
            report['validation_samples'] += len(selection)
    return {'ce': total / count, 'accuracy': correct / count, 'samples': count}


def _save_checkpoint(model, path, report, role):
    path = Path(path)
    pending = path.with_name(path.stem + '.pending.zip')
    model.save(pending)
    pending.replace(path)
    digest = _sha256(path)
    manifest = {
        'format': _FORMAT, 'checkpoint_file': path.name, 'checkpoint_sha256': digest,
        'contract': deepcopy(_CONTRACT), 'role': role, 'warning': _WARNING,
        'sources': report['sources'], 'versions': report['versions'],
        'datasets': report['datasets'], 'split': report['split'],
        'training_config': report['training_config'],
        'actual_device': report['actual_device'],
        'parent_checkpoint': report['parent_checkpoint'], 'exact_simulator_resume': False,
        'progress': {key: report[key] for key in (
            'status', 'stop_reason', 'epochs_completed', 'current_epoch',
            'supervised_samples', 'validation_samples', 'optimizer_updates',
            'environment_interactions')},
    }
    _json_write(path.with_suffix('.manifest.json'), manifest)
    return {'path': str(path), 'sha256': digest, 'epoch': report['epochs_completed']}


def _verify_model(model):
    from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
    from rust_models import extractor_class
    space = model.observation_space
    if not isinstance(space, gym.spaces.Box) or space.shape != (94, 26, 26) \
            or space.dtype != np.float32 or not np.all(space.low == -1) \
            or not np.all(space.high == 1) \
            or not isinstance(model.action_space, gym.spaces.Discrete) \
            or model.action_space.n != 10 or model.action_space.start != 0 \
            or type(model.policy) is not MaskableActorCriticCnnPolicy \
            or type(model.policy.features_extractor) is not extractor_class('cnn') \
            or model.policy.normalize_images \
            or model.policy.net_arch != _CONTRACT['net_arch']:
        raise ValueError('Loaded policy does not match the imitation shape/action/architecture contract')
    if not isinstance(model.policy.optimizer, torch.optim.Adam):
        raise ValueError('Checkpoint requires the full policy Adam optimizer')


def _verify_provenance(manifest):
    try:
        datasets, split = manifest['datasets'], manifest['split']
        namespaces = []
        for dataset in datasets:
            namespace, files = dataset['namespace'], dataset['files']
            if not isinstance(namespace, str) or not isinstance(files, list) or not files \
                    or not isinstance(dataset['metadata'], dict) or not dataset['path']:
                raise ValueError('Invalid dataset identity')
            for entry in files:
                digest = entry['sha256']
                if not isinstance(entry['path'], str) or not entry['path'] \
                        or type(entry['bytes']) is not int or entry['bytes'] < 0 \
                        or not isinstance(digest, str) or len(digest) != 64 \
                        or any(char not in '0123456789abcdef' for char in digest):
                    raise ValueError('Invalid dataset file identity')
            digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()
            if digest != dataset['sha256']:
                raise ValueError('Dataset identity SHA256 mismatch')
            namespaces.append(namespace)
        if namespaces != [f'dataset{i}' for i in range(len(datasets))]:
            raise ValueError('Invalid dataset namespaces')
        train, validation = split['train_episode_ids'], split['validation_episode_ids']
        if not isinstance(train, list) or not isinstance(validation, list) or not train \
                or not validation or not all(isinstance(item, str) for item in train + validation):
            raise ValueError('Invalid episode split')
        if len(set(train + validation)) != len(train) + len(validation) \
                or any(item.rsplit(':', 1)[0] not in namespaces for item in train + validation) \
                or any(type(split[name]) is not int or split[name] <= 0
                       for name in ('train_samples', 'validation_samples')):
            raise ValueError('Invalid or overlapping whole-episode split')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed imitation provenance') from error


def load_imitation_checkpoint(path, device='cpu'):
    """Verify a trusted-local lab zip/sidecar, then restore policy AND optimizer.

    A directory selects ``best.zip`` if available, otherwise ``final.zip``.
    Dataset content identities are retained, not re-read at inference time.
    Source identity must still match this implementation and its CNN definition.
    SB3 zip deserialization is not a sandbox; never load untrusted archives.
    """
    from sb3_contrib import MaskablePPO
    path = Path(path).resolve()
    if path.is_dir():
        path = path / ('best.zip' if (path / 'best.zip').is_file() else 'final.zip')
    sidecar = path.with_suffix('.manifest.json')
    try:
        manifest = json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'Missing or invalid imitation manifest: {sidecar}') from error
    if not isinstance(manifest, dict) or manifest.get('format') != _FORMAT \
            or manifest.get('contract') != _CONTRACT \
            or manifest.get('checkpoint_file') != path.name \
            or manifest.get('sources') != _source_hashes() \
            or not manifest.get('datasets') or not isinstance(manifest.get('split'), dict) \
            or not isinstance(manifest.get('training_config'), dict) \
            or manifest.get('exact_simulator_resume') is not False:
        raise ValueError('Invalid imitation manifest contract or source identity')
    _verify_provenance(manifest)
    if not path.is_file() or _sha256(path) != manifest.get('checkpoint_sha256'):
        raise ValueError('Checkpoint SHA256 mismatch')
    model = MaskablePPO.load(path, device=device)
    _verify_model(model)
    return model


def _start_monitor(output, device, report):
    try:
        from rust_resources import ResourceMonitor
        monitor = ResourceMonitor(output / 'resources.jsonl', device=str(device))
        report['resource_monitor'] = {'status': 'enabled', 'path': str(monitor.path)}
        monitor.check('imitation_start', force=True)
        return monitor
    except (ImportError, OSError) as error:
        report['resource_monitor'] = {'status': 'unavailable', 'reason': str(error)}
        return None


def train_imitation(dataset_paths, output, *, epochs=10, batch_size=256,
                    learning_rate=3e-4, seed=0, device='cpu', deadline=None,
                    initial_checkpoint=None):
    """Clone legal teacher actions; return and persist explicit run accounting.

    ``output`` must not exist. Splits hold out ceil(20%) of namespaced whole
    episodes (minimum two episodes). Completed epochs get independent archives;
    best validation CE and the latest/partial model are saved separately.
    ``supervised_samples`` counts examples in successful optimizer updates,
    not unique frames or environment interactions. Warm starts retain Adam
    state but use this call's learning rate, seed, split, and epoch count.
    Deadlines stop new minibatches; checkpoint finalization may outlast them.
    Configuration/fresh-output errors raise; runtime/data errors return
    ``status='failed'`` with an error, and deadline/resource stops are partial.
    """
    for name, value, minimum in (('epochs', epochs, 1), ('batch_size', batch_size, 1),
                                 ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError('deadline must be an absolute finite monotonic time')
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        'format': 'tank-ai-imitation-report-v1', 'status': 'running', 'stop_reason': None,
        'warning': _WARNING, 'training_config': {
            'epochs': int(epochs), 'batch_size': int(batch_size), 'learning_rate': learning_rate,
            'seed': int(seed), 'device': str(device), 'deadline_monotonic': deadline},
        'epochs_completed': 0, 'current_epoch': 0, 'supervised_samples': 0,
        'validation_samples': 0, 'optimizer_updates': 0, 'environment_interactions': 0,
        'datasets': [], 'split': None, 'parent_checkpoint': None,
        'actual_device': None,
        'initial_validation_ce': None, 'best_validation_ce': None, 'history': [],
        'checkpoints': {'epochs': [], 'best': None, 'final': None},
        'sources': {}, 'versions': {}, 'resource_monitor': {'status': 'not_started'},
    }
    model, corpus = None, None
    try:
        _check_deadline(deadline)
        monitor = _start_monitor(output, device, report)
        import sb3_contrib
        import stable_baselines3
        report['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                              'torch': str(torch.__version__),
                              'sb3_contrib': sb3_contrib.__version__,
                              'stable_baselines3': stable_baselines3.__version__}
        report['sources'] = _source_hashes()
        paths = [Path(path).resolve() for path in paths]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Provide one or more datasets without duplicate paths')
        corpus = _Corpus(paths, output, deadline)
        report['datasets'] = corpus.identities
        train_indices, validation_indices, report['split'] = _split_episodes(
            corpus.episode_ids, seed)
        _check_deadline(deadline)
        if initial_checkpoint is None:
            model = _new_model(learning_rate, seed, device)
        else:
            parent = Path(initial_checkpoint).resolve()
            if parent.is_dir():
                parent /= 'best.zip' if (parent / 'best.zip').is_file() else 'final.zip'
            model = load_imitation_checkpoint(parent, device=device)
            report['parent_checkpoint'] = {'path': str(parent), 'sha256': _sha256(parent)}
            model.seed, model.learning_rate = int(seed), learning_rate
            model.set_random_seed(int(seed))
            model._setup_lr_schedule()
            for group in model.policy.optimizer.param_groups:
                group['lr'] = learning_rate
        report['actual_device'] = str(model.device)
        initial = _evaluate(model, corpus, validation_indices, batch_size, deadline, monitor, report)
        report['initial_validation_ce'] = report['best_validation_ce'] = initial['ce']
        report['checkpoints']['best'] = _save_checkpoint(model, output / 'best.zip', report, 'best')
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            report['current_epoch'] = epoch
            order = rng.permutation(train_indices)
            model.policy.set_training_mode(True)
            total, seen = 0., 0
            for start in range(0, len(order), batch_size):
                _check_deadline(deadline)
                if monitor is not None:
                    monitor.check('imitation_minibatch')
                selection = order[start:start + batch_size]
                batch = corpus.batch(selection)
                _check_deadline(deadline)
                model.policy.optimizer.zero_grad(set_to_none=True)
                loss, _ = _batch_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm,
                                               error_if_nonfinite=True)
                _check_deadline(deadline)
                model.policy.optimizer.step()
                report['optimizer_updates'] += 1
                report['supervised_samples'] += len(selection)
                total += float(loss.item()) * len(selection)
                seen += len(selection)
            validation = _evaluate(model, corpus, validation_indices, batch_size,
                                   deadline, monitor, report)
            report['epochs_completed'] = epoch
            report['history'].append({'epoch': epoch, 'training_ce': total / seen,
                                      'validation_ce': validation['ce'],
                                      'validation_accuracy': validation['accuracy']})
            checkpoint = _save_checkpoint(
                model, output / f'epoch-{epoch:04d}.zip', report, 'completed_epoch')
            report['checkpoints']['epochs'].append(checkpoint)
            if validation['ce'] < report['best_validation_ce']:
                report['best_validation_ce'] = validation['ce']
                report['checkpoints']['best'] = _save_checkpoint(
                    model, output / 'best.zip', report, 'best')
            _json_write(output / 'report.json', report)
        report['status'], report['stop_reason'] = 'completed', 'epochs_completed'
    except (_DeadlineReached, KeyboardInterrupt) as error:
        report['status'] = 'partial'
        report['stop_reason'] = 'deadline' if isinstance(error, _DeadlineReached) else 'interrupted'
    except Exception as error:
        resource_pressure = type(error).__name__ == 'ResourcePressureError'
        report['status'] = 'partial' if resource_pressure else 'failed'
        report['stop_reason'] = 'resource_pressure' if resource_pressure else 'error'
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if corpus is not None:
            try:
                corpus.close()
            except Exception as error:
                report['status'], report['stop_reason'] = 'failed', 'cleanup_error'
                report['cleanup_error'] = f'{type(error).__name__}: {error}'
        if model is not None:
            try:
                report['checkpoints']['final'] = _save_checkpoint(
                    model, output / 'final.zip', report, 'final')
            except Exception as error:
                report['status'], report['stop_reason'] = 'failed', 'checkpoint_error'
                report['checkpoint_error'] = f'{type(error).__name__}: {error}'
        report['elapsed_seconds'] = max(0., time.monotonic() - started)
        _json_write(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--deadline', type=float, help='Absolute time.monotonic() deadline')
    parser.add_argument('--initial-checkpoint')
    args = vars(parser.parse_args(argv))
    args['dataset_paths'] = args.pop('datasets')
    report = train_imitation(**args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return {'completed': 0, 'failed': 1, 'partial': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
