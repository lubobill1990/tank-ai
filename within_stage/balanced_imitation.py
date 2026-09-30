"""Precision-CNN actor BC: only training CE is inverse-sqrt class balanced.

Unchanged TRAIN authentication/split, sampling, architecture and unweighted
validation are reused. No oversampling/offset ablation/teacher or environment.
The strict loader recomputes counts from authenticated source corpora; keep
those datasets and the immediate parent available for verification.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from learning_lab.weighted_imitation import _class_weights as class_weights
from learning_lab.weighted_imitation import _weighted_masked_cross_entropy as weighted_ce, _WEIGHTING
from within_stage import imitation as base
from within_stage.precision_policy import (
    actor_logits, precision_schema, save_precision_checkpoint, load_precision_checkpoint, file_sha256)

OBJECTIVE = 'actor_only_train_count_balanced_precision_ce_v1'
ROOT = Path(__file__).resolve().parents[1]


def _sources():
    return {**base._sources(), 'within_stage\\balanced_imitation.py': file_sha256(Path(__file__)),
            'learning_lab\\weighted_imitation.py': file_sha256(ROOT / 'learning_lab' / 'weighted_imitation.py')}


def training_counts(corpus, indices):
    """Use only authenticated TRAIN-split label rows, without decoding dense frames."""
    counts = np.zeros(10, np.int64)
    for index, dataset in enumerate(corpus.datasets):
        start, stop = corpus.offsets[index:index + 2]
        local = indices[(indices >= start) & (indices < stop)] - start
        actions = np.load(dataset.path / 'teacher_actions.npy', mmap_mode='r', allow_pickle=False)
        try:
            counts += np.bincount(actions[local], minlength=10)
        finally:
            actions._mmap.close()
    return counts


def atomic_actor_update(model, batch, weights, deadline):
    """Restore policy+Adam if a minibatch fails or expires; counters commit outside."""
    snapshot = deepcopy((model.policy.state_dict(), model.policy.optimizer.state_dict()))
    try:
        model.policy.optimizer.zero_grad(set_to_none=True)
        observations = torch.as_tensor(batch['observations'], device=model.device)
        actions = torch.as_tensor(batch['actions'], device=model.device)
        masks = torch.as_tensor(batch['action_masks'], device=model.device)
        loss = weighted_ce(actor_logits(model, observations), actions, masks, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5, error_if_nonfinite=True)
        base._check(deadline)
        model.policy.optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
            raise FloatingPointError('Nonfinite balanced-BC parameters')
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
               for state in model.policy.optimizer.state.values() for v in state.values()):
            raise FloatingPointError('Nonfinite balanced-BC Adam state')
        return float(loss.item())
    except BaseException:
        model.policy.load_state_dict(snapshot[0], strict=True)
        model.policy.optimizer.load_state_dict(snapshot[1])
        model.policy.optimizer.zero_grad(set_to_none=True)
        raise


def _save(model, path, report, role):
    if _sources() != report['sources'] or base.critic_fingerprint(model) != report['initial_critic']:
        raise ValueError('Balanced BC source/critic invariants changed')
    membership = report['family_membership']
    if file_sha256(membership['frozen_split_path']) != membership['frozen_split_sha256'] \
            or file_sha256(membership['pool_path']) != membership['pool_sha256']:
        raise ValueError('Frozen TRAIN split/pool changed')
    config = {key: deepcopy(report[key]) for key in (
        'datasets', 'split', 'family_membership', 'sources', 'parent_checkpoint', 'training_config',
        'train_class_counts', 'class_weights', 'initial_critic')}
    config.update(schema=precision_schema(), objective=OBJECTIVE, weighting=deepcopy(_WEIGHTING),
                  validation='unweighted', offsets_enabled=True, role=role,
                  evaluation_adapter={'required': False, 'zero_channel_ranges': []},
                  progress={key: report[key] for key in ('status', 'epochs_completed',
                    'optimizer_updates', 'supervised_samples', 'validation_samples', 'environment_interactions')},
                  warning='Class balancing is a hypothesis; internal CE does not select deployment.')
    return save_precision_checkpoint(model, path, migration_record=model.precision_migration_record,
                                     dataset_config=config)


def load_balanced_precision_checkpoint(path, device='cpu'):
    """Strict precision loader PLUS authenticated TRAIN-count/weight/parent verification."""
    model = load_precision_checkpoint(path, device=device)
    config = model.precision_dataset_config
    if config.get('objective') != OBJECTIVE or config.get('sources') != _sources() \
            or config.get('weighting') != _WEIGHTING or config.get('validation') != 'unweighted' \
            or config.get('offsets_enabled') is not True:
        raise ValueError('Invalid balanced precision objective/source contract')
    try:
        parent = config['parent_checkpoint']
        if file_sha256(parent['path']) != parent['sha256'] or file_sha256(
                Path(parent['path']).with_suffix('.manifest.json')) != parent['manifest_sha256']:
            raise ValueError('Balanced precision parent identity mismatch')
        contract = base._training_contract()
        corpus = base.TrainingCorpus([Path(row['path']) for row in config['datasets']], contract, None)
        try:
            train, _, split = base._split_episodes(corpus.episode_ids, config['training_config']['seed'])
            counts = training_counts(corpus, train)
            weights = np.asarray(config['class_weights'])
            if corpus.identities != config['datasets'] or split != config['split'] \
                    or not all(type(c) is int for c in config['train_class_counts']) \
                    or config['train_class_counts'] != counts.tolist() \
                    or weights.shape != (10,) or weights.dtype.kind not in 'fi' \
                    or not np.allclose(weights, class_weights(counts), rtol=0, atol=1e-12):
                raise ValueError('Balanced counts/weights do not match actual authenticated TRAIN rows')
        finally:
            corpus.close()
        if base.critic_fingerprint(model) != config['initial_critic']:
            raise ValueError('Balanced precision critic was modified')
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('Missing/malformed balanced provenance or source datasets') from error
    return model


def train_balanced_precision(dataset_paths, initial_checkpoint, output, *, epochs=20,
                              batch=128, lr=1e-4, seed=0, device='cpu', deadline=None):
    """One loss-only comparison. Entire copied training driver stays below250 lines.

    Use the same datasets/initial checkpoint/seed as unweighted precision BC.
    Counts are recomputed from its identical whole-episode TRAIN split. Weights
    are cap4 inverse-sqrt counts, example-mean1, absent0; no minibatch renormalizing.
    Checkpoints use the frozen precision format plus strict balanced config.
    Writes initial/epoch/final archives, best.json and report.json, never overwrites.
    """
    if any(type(v) is not int or v < 1 for v in (epochs, batch)) or type(seed) is not int \
            or not 0 <= seed < 2**32 or isinstance(lr, bool) or not isinstance(lr, (float, int)) \
            or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid epochs/batch/lr/seed')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (float, int))
                                 or not math.isfinite(deadline)):
        raise ValueError('deadline must be finite absolute monotonic time')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Select explicit CPU/CUDA device')
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    paths, output = [Path(p).resolve() for p in paths], Path(output).resolve()
    if not paths or len(set(paths)) != len(paths) or any(output.is_relative_to(p) for p in paths):
        raise ValueError('Distinct datasets and a separate fresh output are required')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format='within-stage-balanced-precision-imitation-v1',
        status='running', stop_reason=None, objective=OBJECTIVE, offsets_enabled=True,
        training_config=dict(epochs=epochs, batch_size=batch, learning_rate=lr, seed=seed,
                             device=str(device), deadline_monotonic=deadline),
        sources={}, datasets=[], split=None, family_membership=None, parent_checkpoint=None,
        initial_critic=None, critic_unchanged=None, train_class_counts=None, class_weights=None,
        epochs_completed=0, optimizer_updates=0, supervised_samples=0, validation_samples=0,
        environment_interactions=0, initial_validation_ce=None, best_validation_ce=None, history=[],
        checkpoints={'initial': None, 'epochs': [], 'best': None, 'final': None},
        deployment_selection_performed=False, resource_monitor={'status': 'not_started'})
    started, model, corpus, monitor = time.monotonic(), None, None, None
    try:
        base._check(deadline)
        monitor = base.ResourceMonitor(output / 'resources.jsonl', device=str(device).split(':')[0])
        report['resource_monitor'] = {'status': 'enabled', 'path': str(monitor.path)}
        monitor.check('balanced_bc_admission', force=True)
        contract = base._training_contract()
        report['sources'] = _sources()
        corpus = base.TrainingCorpus(paths, contract, deadline)
        train, valid, report['split'] = base._split_episodes(corpus.episode_ids, seed)
        report['datasets'] = corpus.identities
        report['family_membership'] = {k: v for k, v in contract.items() if k != 'allowed_configs'}
        report['train_class_counts'] = training_counts(corpus, train).tolist()
        report['class_weights'] = class_weights(np.asarray(report['train_class_counts'])).tolist()
        base._check(deadline)
        parent = Path(initial_checkpoint).resolve()
        model = load_precision_checkpoint(parent, device=device)
        if model.precision_dataset_config.get('objective') == OBJECTIVE:
            model = load_balanced_precision_checkpoint(parent, device=device)
        report['parent_checkpoint'] = {'path': str(parent), 'sha256': file_sha256(parent),
            'manifest_sha256': file_sha256(parent.with_suffix('.manifest.json'))}
        report['initial_critic'] = base.critic_fingerprint(model)
        if str(device).startswith('cuda') and model.device.type != 'cuda':
            raise RuntimeError('Requested CUDA cannot silently fall back')
        model.learning_rate, model.seed = lr, seed
        model.set_random_seed(seed); model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group['lr'] = lr
        weights = torch.tensor(report['class_weights'], dtype=torch.float32, device=model.device)
        evaluation = base._evaluate(model, corpus, valid, batch, True, deadline, monitor, report)
        report['initial_validation_ce'] = report['best_validation_ce'] = evaluation['ce']
        initial = _save(model, output / 'initial.zip', report, 'initial')
        report['checkpoints']['initial'] = report['checkpoints']['best'] = initial
        base._write(output / 'report.json', report)
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            model.policy.set_training_mode(True)
            order, total, count = rng.permutation(train), 0., 0
            for first in range(0, len(order), batch):
                base._check(deadline); monitor.check('balanced_bc_minibatch')
                ids = order[first:first + batch]
                values = corpus.batch(ids)
                base._check(deadline)
                loss = atomic_actor_update(model, values, weights, deadline)
                report['optimizer_updates'] += 1; report['supervised_samples'] += len(ids)
                total += loss * len(ids); count += len(ids)
            evaluation = base._evaluate(model, corpus, valid, batch, True, deadline, monitor, report)
            report['epochs_completed'] = epoch
            report['history'].append(dict(epoch=epoch, training_weighted_ce=total / count,
                validation_ce=evaluation['ce'], validation_accuracy=evaluation['accuracy']))
            saved = _save(model, output / f'epoch-{epoch:04d}.zip', report, 'completed_epoch')
            report['checkpoints']['epochs'].append(saved)
            if evaluation['ce'] < report['best_validation_ce']:
                report['best_validation_ce'], report['checkpoints']['best'] = evaluation['ce'], saved
            base._write(output / 'report.json', report)
        report.update(status='completed', stop_reason='epochs_completed')
    except (base.DeadlineReached, base.ResourcePressureError, KeyboardInterrupt) as error:
        report.update(status='partial', stop_reason=type(error).__name__, error=str(error))
    except Exception as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if corpus is not None:
            try:
                corpus.close()
            except Exception as error:
                report.update(status='failed', cleanup_error=str(error))
        if model is not None and report['initial_critic'] is not None:
            report['critic_unchanged'] = base.critic_fingerprint(model) == report['initial_critic']
            try:
                report['checkpoints']['final'] = _save(model, output / 'final.zip', report, 'final_or_partial')
            except Exception as error:
                report.update(status='failed', checkpoint_error=f'{type(error).__name__}: {error}')
        report['elapsed_seconds'] = time.monotonic() - started
        base._write(output / 'best.json', report['checkpoints']['best'])
        base._write(output / 'report.json', report)
    return report
