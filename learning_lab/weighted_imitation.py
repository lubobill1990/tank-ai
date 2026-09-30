"""Train-only class-balanced cloning, separate from frozen unweighted artifacts.

Weights are min(4, scale / sqrt(training_count)); scale is chosen so their
training-example-weighted mean is one even after clipping. Absent classes get
zero weight. Minibatch loss is mean(weight[label] * masked CE), not PyTorch's
per-minibatch sum-of-weights normalization. Validation is always unweighted.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from learning_lab import imitation as base


_FORMAT = 'tank-ai-weighted-imitation-checkpoint-v1'
_WEIGHTING = {
    'rule': 'inverse_sqrt_training_class_counts_v1', 'cap': 4.0,
    'normalization': 'capped_training_example_mean_one', 'zero_count_weight': 0.0,
    'loss_reduction': 'mean_of_weighted_examples', 'validation': 'unweighted',
}
_CONTRACT = {
    **deepcopy(base._CONTRACT), 'objective': 'weighted_masked_teacher_cross_entropy',
    'weighting': deepcopy(_WEIGHTING),
}


def _source_hashes():
    return {**base._source_hashes(),
            'learning_lab/weighted_imitation.py': base._sha256(Path(__file__))}


def _class_weights(counts):
    counts = np.asarray(counts)
    if counts.shape != (10,) or counts.dtype.kind not in 'iu' or np.any(counts < 0) \
            or not np.any(counts > 0):
        raise ValueError('Class counts must be ten nonnegative integers with positive total')
    counts = counts.astype(np.float64)
    raw = np.zeros(10, np.float64)
    present = counts > 0
    raw[present] = 1 / np.sqrt(counts[present])
    capped = np.zeros(10, bool)
    for _ in range(10):
        free = present & ~capped
        scale = (counts.sum() - 4 * counts[capped].sum()) / np.dot(counts[free], raw[free])
        weights = np.minimum(4., scale * raw)
        newly_capped = free & (scale * raw > 4)
        if not newly_capped.any():
            return weights
        capped |= newly_capped
    raise ValueError('Could not normalize capped class weights')


def _weighted_masked_cross_entropy(logits, actions, masks, weights):
    base._masked_cross_entropy(logits, actions, masks)
    if weights.shape != (logits.shape[1],) or not weights.is_floating_point() \
            or not torch.isfinite(weights).all() or (weights < 0).any() \
            or (weights > 4).any() or (weights[actions] <= 0).any():
        raise ValueError('Training labels require finite positive class weights, capped at four')
    losses = F.cross_entropy(logits.masked_fill(~masks, float('-inf')), actions, reduction='none')
    return (losses * weights[actions]).mean()


def _count_classes(corpus, indices, batch_size, deadline, monitor, report):
    counts = np.zeros(10, np.int64)
    for start in range(0, len(indices), batch_size):
        base._check_deadline(deadline)
        if monitor is not None:
            monitor.check('weighted_imitation_class_counts')
        selection = indices[start:start + batch_size]
        batch = corpus.batch(selection)
        counts += np.bincount(batch['actions'], minlength=10)
        report['class_count_samples'] += len(selection)
        report['train_class_counts'] = counts.tolist()
        base._check_deadline(deadline)
    return counts


def _train_epoch(model, corpus, order, weights, batch_size, deadline, monitor, report,
                 *, deadline_check=base._check_deadline):
    model.policy.set_training_mode(True)
    total, seen = 0., 0
    for start in range(0, len(order), batch_size):
        deadline_check(deadline)
        if monitor is not None:
            monitor.check('weighted_imitation_minibatch')
        selection = order[start:start + batch_size]
        batch = corpus.batch(selection)
        deadline_check(deadline)
        observations = torch.as_tensor(batch['observations'], device=model.device)
        actions = torch.as_tensor(batch['actions'], device=model.device)
        masks = torch.as_tensor(batch['action_masks'], device=model.device)
        model.policy.optimizer.zero_grad(set_to_none=True)
        logits = model.policy.get_distribution(observations).distribution.logits
        loss = _weighted_masked_cross_entropy(logits, actions, masks, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm,
                                       error_if_nonfinite=True)
        deadline_check(deadline)
        model.policy.optimizer.step()
        report['optimizer_updates'] += 1
        report['supervised_samples'] += len(selection)
        total += float(loss.item()) * len(selection)
        seen += len(selection)
    return total / seen


def _save_checkpoint(model, path, report, role):
    path = Path(path)
    pending = path.with_name(path.stem + '.pending.zip')
    model.save(pending)
    pending.replace(path)
    digest = base._sha256(path)
    manifest = {
        'format': _FORMAT, 'checkpoint_file': path.name, 'checkpoint_sha256': digest,
        'contract': deepcopy(_CONTRACT), 'role': role, 'warning': base._WARNING,
        'exact_simulator_resume': False,
        **{name: deepcopy(report[name]) for name in (
            'sources', 'versions', 'datasets', 'split', 'training_config', 'actual_device',
            'parent_checkpoint', 'train_class_counts', 'class_weights', 'class_weighting')},
        'progress': {name: report[name] for name in (
            'status', 'stop_reason', 'epochs_completed', 'current_epoch',
            'class_count_samples', 'supervised_samples', 'validation_samples',
            'optimizer_updates', 'environment_interactions')},
    }
    base._json_write(path.with_suffix('.manifest.json'), manifest)
    return {'path': str(path), 'sha256': digest, 'epoch': report['epochs_completed']}


def _checkpoint_path(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'best.zip' if (path / 'best.zip').is_file() else 'final.zip'
    return path


def _verify_weighting(manifest):
    try:
        counts = np.asarray(manifest['train_class_counts'])
        expected = _class_weights(counts)
        actual = np.asarray(manifest['class_weights'], dtype=np.float64)
        parent = manifest['parent_checkpoint']
        if actual.shape != (10,) or not np.isfinite(actual).all() \
                or not np.allclose(actual, expected, rtol=0, atol=1e-12) \
                or int(counts.sum()) != manifest['split']['train_samples'] \
                or manifest['progress']['class_count_samples'] != int(counts.sum()) \
                or manifest['progress']['environment_interactions'] != 0 \
                or manifest['class_weighting'] != _WEIGHTING \
                or parent['format'] != base._FORMAT or not isinstance(parent['path'], str) \
                or not parent['path']:
            raise ValueError('Invalid weighted objective, train-only class counts, or parent provenance')
        for name in ('sha256', 'manifest_sha256'):
            digest = parent[name]
            if not isinstance(digest, str) or len(digest) != 64 \
                    or any(char not in '0123456789abcdef' for char in digest):
                raise ValueError('Invalid parent checkpoint SHA256')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed weighted checkpoint provenance') from error


def load_weighted_checkpoint(path, device='cpu'):
    """Verify a trusted-local weighted archive and restore full policy/Adam state.

    This accepts only the weighted format, not base imitation checkpoints.
    The original loader remains unchanged and independently verifies parents.
    Parent/dataset identities are retained without requiring those files at
    inference time. SB3 archives must never come from untrusted sources.
    """
    from sb3_contrib import MaskablePPO
    path = _checkpoint_path(path)
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Missing or invalid weighted imitation manifest') from error
    if not isinstance(manifest, dict) or manifest.get('format') != _FORMAT \
            or manifest.get('contract') != _CONTRACT or manifest.get('checkpoint_file') != path.name \
            or manifest.get('sources') != _source_hashes() or not manifest.get('datasets') \
            or not isinstance(manifest.get('split'), dict) \
            or not isinstance(manifest.get('training_config'), dict) \
            or manifest.get('exact_simulator_resume') is not False:
        raise ValueError('Invalid weighted imitation manifest contract or source identity')
    base._verify_provenance(manifest)
    _verify_weighting(manifest)
    if not path.is_file() or base._sha256(path) != manifest.get('checkpoint_sha256'):
        raise ValueError('Weighted checkpoint SHA256 mismatch')
    model = MaskablePPO.load(path, device=device)
    base._verify_model(model)
    return model


def train_weighted_imitation(dataset_paths, output, *, initial_checkpoint, epochs=6,
                             batch_size=256, learning_rate=1e-4, seed=0, device='cpu',
                             deadline=None):
    """Fine-tune a verified BASE imitation checkpoint using train-only weighting.

    A fresh output directory is required. With identical ordered datasets and
    seed, the split and shuffled minibatch order match the unweighted control.
    Parent Adam state is retained, with this call's learning rate and seed.
    Counts are a separate offline pass; neither it nor supervised samples count
    as environment interactions. Validation/best selection use unweighted CE.
    Absolute monotonic deadlines guard counting, training, and validation;
    final checkpoint serialization may outlast a deadline. Errors/partial runs
    are persisted explicitly, with best, completed-epoch, and final archives.
    """
    for name, value, minimum in (('epochs', epochs, 1), ('batch_size', batch_size, 1), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if initial_checkpoint is None:
        raise ValueError('initial_checkpoint must be a verified base imitation checkpoint')
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError('deadline must be a finite absolute monotonic time')
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        'format': 'tank-ai-weighted-imitation-report-v1', 'status': 'running', 'stop_reason': None,
        'warning': base._WARNING, 'training_config': {
            'epochs': int(epochs), 'batch_size': int(batch_size), 'learning_rate': learning_rate,
            'seed': int(seed), 'device': str(device), 'deadline_monotonic': deadline},
        'epochs_completed': 0, 'current_epoch': 0, 'supervised_samples': 0,
        'validation_samples': 0, 'optimizer_updates': 0, 'environment_interactions': 0,
        'class_count_samples': 0, 'train_class_counts': [0] * 10, 'class_weights': None,
        'class_weighting': deepcopy(_WEIGHTING),
        'datasets': [], 'split': None, 'parent_checkpoint': None, 'actual_device': None,
        'initial_validation_ce': None, 'best_validation_ce': None, 'history': [],
        'checkpoints': {'epochs': [], 'best': None, 'final': None},
        'sources': {}, 'versions': {}, 'resource_monitor': {'status': 'not_started'},
    }
    model, corpus = None, None
    try:
        base._check_deadline(deadline)
        monitor = base._start_monitor(output, device, report)
        import sb3_contrib
        import stable_baselines3
        report['versions'] = {
            'python': platform.python_version(), 'numpy': np.__version__,
            'torch': str(torch.__version__), 'sb3_contrib': sb3_contrib.__version__,
            'stable_baselines3': stable_baselines3.__version__}
        report['sources'] = _source_hashes()
        paths = [Path(path).resolve() for path in paths]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Provide one or more datasets without duplicate paths')
        corpus = base._Corpus(paths, output, deadline)
        report['datasets'] = corpus.identities
        train_indices, validation_indices, report['split'] = base._split_episodes(corpus.episode_ids, seed)
        counts = _count_classes(corpus, train_indices, batch_size, deadline, monitor, report)
        report['class_weights'] = _class_weights(counts).tolist()
        base._check_deadline(deadline)
        parent = _checkpoint_path(initial_checkpoint)
        model = base.load_imitation_checkpoint(parent, device=device)
        report['parent_checkpoint'] = {
            'path': str(parent), 'sha256': base._sha256(parent), 'format': base._FORMAT,
            'manifest_sha256': base._sha256(parent.with_suffix('.manifest.json'))}
        model.seed, model.learning_rate = int(seed), learning_rate
        model.set_random_seed(int(seed))
        model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group['lr'] = learning_rate
        report['actual_device'] = str(model.device)
        weights = torch.tensor(report['class_weights'], dtype=torch.float32, device=model.device)
        initial = base._evaluate(model, corpus, validation_indices, batch_size, deadline, monitor, report)
        report['initial_validation_ce'] = report['best_validation_ce'] = initial['ce']
        report['checkpoints']['best'] = _save_checkpoint(model, output / 'best.zip', report, 'best')
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            report['current_epoch'] = epoch
            training_ce = _train_epoch(model, corpus, rng.permutation(train_indices), weights,
                                       batch_size, deadline, monitor, report)
            validation = base._evaluate(
                model, corpus, validation_indices, batch_size, deadline, monitor, report)
            report['epochs_completed'] = epoch
            report['history'].append({
                'epoch': epoch, 'training_weighted_ce': training_ce,
                'validation_ce': validation['ce'], 'validation_accuracy': validation['accuracy']})
            report['checkpoints']['epochs'].append(_save_checkpoint(
                model, output / f'epoch-{epoch:04d}.zip', report, 'completed_epoch'))
            if validation['ce'] < report['best_validation_ce']:
                report['best_validation_ce'] = validation['ce']
                report['checkpoints']['best'] = _save_checkpoint(model, output / 'best.zip', report, 'best')
            base._json_write(output / 'report.json', report)
        report['status'], report['stop_reason'] = 'completed', 'epochs_completed'
    except (base._DeadlineReached, KeyboardInterrupt) as error:
        report['status'] = 'partial'
        report['stop_reason'] = 'deadline' if isinstance(error, base._DeadlineReached) else 'interrupted'
    except Exception as error:
        pressure = type(error).__name__ == 'ResourcePressureError'
        report['status'] = 'partial' if pressure else 'failed'
        report['stop_reason'] = 'resource_pressure' if pressure else 'error'
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
                report['checkpoints']['final'] = _save_checkpoint(model, output / 'final.zip', report, 'final')
            except Exception as error:
                report['status'], report['stop_reason'] = 'failed', 'checkpoint_error'
                report['checkpoint_error'] = f'{type(error).__name__}: {error}'
        report['elapsed_seconds'] = max(0., time.monotonic() - started)
        base._json_write(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--deadline', type=float, help='Absolute time.monotonic() deadline')
    args = vars(parser.parse_args(argv))
    args['dataset_paths'] = args.pop('datasets')
    report = train_weighted_imitation(**args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return {'completed': 0, 'failed': 1, 'partial': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
