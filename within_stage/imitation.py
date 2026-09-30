"""Actor-only precision55 BC on authenticated TRAIN opening families.

Held-out episodes here are an internal 80/20 DEVELOPMENT accuracy split of
TRAIN-family demonstrations, never the opening protocol's DEV or FINAL set.
No environment, audit RAM, reward training, or critic update is used.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from learning_lab.imitation import _masked_cross_entropy as masked_ce, _split_episodes
from rust_resources import ResourceMonitor, ResourcePressureError
from within_stage.demonstrations import PrecisionDataset, SPLIT_SHA256
from within_stage.observation import OBSERVATION_VERSION
from within_stage.precision_policy import (
    actor_logits, file_sha256, state_sha256, precision_schema,
    load_precision_checkpoint, save_precision_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT_PATH = ROOT / 'runs' / '20260930-within-stage' / 'frozen-split.json'
FROZEN_SPLIT_SHA256 = SPLIT_SHA256
WARNING = ('Whole-episode holdout accuracy is DEVELOPMENT ONLY within TRAIN families; '
           'it is not a full-game clear, DEV-opening evaluation, FINAL test, or checkpoint deployment decision.')


class DeadlineReached(RuntimeError):
    pass


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineReached('Precision BC absolute monotonic deadline reached')


def _write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\imitation.py', 'within_stage\\demonstrations.py',
        'within_stage\\precision_policy.py', 'within_stage\\observation.py',
        'learning_lab\\imitation.py', 'native_codec.py', 'policy.py', 'rust_models.py')}


def adapt_observations(observations, *, offsets_enabled):
    """Owned input; offsets-off must also be used by the later evaluation adapter."""
    if type(offsets_enabled) is not bool or not isinstance(observations, np.ndarray) \
            or observations.dtype != np.float32 or observations.ndim != 4 \
            or observations.shape[1:] != (110, 26, 26) or not np.isfinite(observations).all():
        raise ValueError('Expected finite batched precision float32 observations and boolean offsets_enabled')
    result = observations.copy()
    if not offsets_enabled:
        result[:, 47:55] = 0
        result[:, 102:110] = 0
    return result


def _training_contract():
    path = Path(FROZEN_SPLIT_PATH).resolve()
    if file_sha256(path) != FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen TRAIN split SHA256 mismatch')
    split = json.loads(path.read_text(encoding='utf-8'))
    expected = {'observation_version': OBSERVATION_VERSION, 'channels_per_frame': 55,
                'actions': 10, 'n_stack': 2, 'stage': 1, 'post_start_frames': 150}
    if any(split.get(key) != value for key, value in expected.items()):
        raise ValueError('Frozen TRAIN precision schema differs')
    train = split['train']
    families, configs = train['family_ids'], train['all_config_indices']
    if len(families) != 40 or len(set(families)) != 40 \
            or not configs or len(configs) != len(set(configs)):
        raise ValueError('Frozen TRAIN partition must contain40 unique families and unique configs')
    pool_path = Path(split['pool_path']).resolve()
    if file_sha256(pool_path) != split['pool_sha256']:
        raise ValueError('Frozen opening-pool SHA256 mismatch')
    pool = json.loads(pool_path.read_text(encoding='utf-8'))
    allowed = {row['config_index']: deepcopy(row) for row in pool['specs']
               if row['config_index'] in set(configs)}
    if len(allowed) != len(configs) or set(row['family_id'] for row in allowed.values()) != set(families):
        raise ValueError('Frozen TRAIN family/config mapping mismatch')
    return {'partition': 'train', 'frozen_split_path': str(path),
            'frozen_split_sha256': FROZEN_SPLIT_SHA256,
            'pool_path': str(pool_path), 'pool_sha256': split['pool_sha256'],
            'train_family_ids': list(families), 'allowed_configs': allowed}


def _validate_batch(batch, expected_ids):
    count = len(expected_ids)
    for key, shape, dtype in (
        ('observations', (count, 110, 26, 26), np.float32), ('actions', (count,), np.int64),
        ('action_masks', (count, 10), np.bool_), ('episode_ids', (count,), np.int64),
    ):
        value = batch.get(key)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f'Invalid precision batch {key}')
    if not np.array_equal(batch['episode_ids'], expected_ids):
        raise ValueError('Precision batch episode IDs disagree with split membership')
    if not np.isfinite(batch['observations']).all():
        raise ValueError('Precision observations must be finite')
    actions, masks = batch['actions'], batch['action_masks']
    if np.any(actions < 0) or np.any(actions >= 10) or not masks.any(axis=1).all() \
            or not masks[np.arange(count), actions].all():
        raise ValueError('Teacher action must be legal under its nonempty mask')


class TrainingCorpus:
    """Only batch four learner fields; never expose PrecisionDataset.audit RAM."""
    def __init__(self, paths, contract, deadline):
        self.datasets, self.identities, self.offsets = [], [], [0]
        self.membership = []
        ids = []
        try:
            for dataset_index, path in enumerate(paths):
                _check(deadline)
                dataset = PrecisionDataset(path)
                self.datasets.append(dataset)
                metadata = dataset.metadata
                if metadata.get('selected_partition') != 'train' \
                        or metadata.get('split_sha256') != contract['frozen_split_sha256']:
                    raise ValueError('Dataset is not bound to the frozen TRAIN partition')
                plan = metadata.get('plan')
                if not isinstance(plan, list) or not plan:
                    raise ValueError('TRAIN corpus must contain its origin opening plan')
                by_episode = {}
                for row in plan:
                    episode, config = row['episode_id'], row['config_index']
                    allowed = contract['allowed_configs'].get(config)
                    if type(episode) is not int or episode < 0 or episode in by_episode \
                            or allowed is None or row['family_id'] != allowed['family_id'] \
                            or row['opening_spec'] != allowed['spec']:
                        raise ValueError('Dataset origin family/config/spec is outside frozen TRAIN or malformed')
                    by_episode[episode] = row
                reports = json.loads((path / 'episode_reports.json').read_text(encoding='utf-8'))
                reports_by_episode = {}
                for row in reports:
                    episode = row['episode_id']
                    origin = by_episode.get(episode)
                    if episode in reports_by_episode or origin is None \
                            or any(row.get(key) != origin[key] for key in (
                                'config_index', 'family_id', 'opening_spec')) \
                            or row.get('stage') != 1 or not row.get('outcome'):
                        raise ValueError('Episode report violates authenticated TRAIN origin')
                    reports_by_episode[episode] = row
                episode_ids = dataset.episode_ids
                if episode_ids.dtype != np.int64 or episode_ids.shape != (len(dataset),) or not len(dataset):
                    raise ValueError('Precision corpus must have nonempty int64 episode IDs')
                namespace = f'dataset{dataset_index}'
                for episode in np.unique(episode_ids):
                    row = reports_by_episode.get(int(episode))
                    count = int(np.count_nonzero(episode_ids == episode))
                    if row is None or row.get('decisions') != count:
                        raise ValueError('Stored episode lacks a complete TRAIN identity/length report')
                    self.membership.append({
                        'episode_id': f'{namespace}:{int(episode)}', 'samples': count,
                        'family_id': row['family_id'], 'config_index': row['config_index'],
                        'outcome': row['outcome'], 'episode_finished': row.get('episode_finished', False)})
                ids.extend(f'{namespace}:{int(episode)}' for episode in episode_ids)
                self.offsets.append(self.offsets[-1] + len(dataset))
                integrity = json.loads((path / 'integrity.json').read_text(encoding='utf-8'))
                self.identities.append({
                    'path': str(path), 'sha256': dataset.fingerprint, 'namespace': namespace,
                    'integrity_sha256': file_sha256(path / 'integrity.json'),
                    'files': integrity['files'], 'metadata': metadata})
            self.episode_ids = np.asarray(ids)
        except BaseException:
            self.close()
            raise

    def batch(self, indices):
        indices = np.asarray(indices, np.int64)
        result = {
            'observations': np.empty((len(indices), 110, 26, 26), np.float32),
            'actions': np.empty(len(indices), np.int64),
            'action_masks': np.empty((len(indices), 10), bool)}
        owners = np.searchsorted(self.offsets[1:], indices, side='right')
        for owner in np.unique(owners):
            rows = np.flatnonzero(owners == owner)
            local = indices[rows] - self.offsets[owner]
            dataset = self.datasets[owner]
            values = dataset.batch(local)
            _validate_batch(values, dataset.episode_ids[local])
            for name in result:
                result[name][rows] = values[name]
        return result

    def close(self):
        for dataset in self.datasets:
            dataset.close()


def critic_fingerprint(model):
    """Hash critic-only weights AND per-parameter Adam buffers, excluding shared LR."""
    policy = model.policy
    modules = {'encoder': policy.vf_features_extractor, 'mlp': policy.mlp_extractor.value_net,
               'head': policy.value_net}
    weights = {name: module.state_dict() for name, module in modules.items()}
    optimizer = {f'{name}.{key}': policy.optimizer.state.get(parameter, {})
                 for name, module in modules.items() for key, parameter in module.named_parameters()}
    return {'weights_sha256': state_sha256(weights), 'adam_state_sha256': state_sha256(optimizer)}


def _loss(model, batch, offsets_enabled):
    observation = adapt_observations(batch['observations'], offsets_enabled=offsets_enabled)
    logits = actor_logits(model, torch.as_tensor(observation, device=model.device))
    actions = torch.as_tensor(batch['actions'], device=model.device)
    masks = torch.as_tensor(batch['action_masks'], device=model.device)
    loss = masked_ce(logits, actions, masks)
    correct = int((logits.detach().masked_fill(~masks, float('-inf')).argmax(1) == actions).sum().item())
    return loss, correct


def _evaluate(model, corpus, indices, batch_size, offsets_enabled, deadline, monitor, report):
    model.policy.set_training_mode(False)
    total, correct, count = 0., 0, 0
    with torch.no_grad():
        for first in range(0, len(indices), batch_size):
            _check(deadline)
            if monitor is not None:
                monitor.check('precision_bc_validation')
            batch_ids = indices[first:first + batch_size]
            batch = corpus.batch(batch_ids)
            _check(deadline)
            loss, matches = _loss(model, batch, offsets_enabled)
            total += loss.item() * len(batch_ids)
            correct += matches; count += len(batch_ids)
            report['validation_samples'] += len(batch_ids)
    return {'ce': total / count, 'accuracy': correct / count}


def _save(model, path, report, role):
    if _sources() != report['sources']:
        raise ValueError('Frozen precision BC sources changed during this run')
    membership = report['family_membership']
    if file_sha256(membership['frozen_split_path']) != membership['frozen_split_sha256'] \
            or file_sha256(membership['pool_path']) != membership['pool_sha256']:
        raise ValueError('Frozen TRAIN split/pool changed during this run')
    if critic_fingerprint(model) != report['initial_critic']:
        raise ValueError('Actor-only precision BC changed critic weights or Adam state')
    config = {
        'schema': precision_schema(), 'datasets': report['datasets'], 'split': report['split'],
        'family_membership': report['family_membership'], 'sources': report['sources'],
        'objective': 'masked_teacher_cross_entropy_actor_only',
        'validation': 'unweighted whole-episode DEVELOPMENT within TRAIN families',
        'offsets_enabled': report['offsets_enabled'],
        'evaluation_adapter': report['evaluation_adapter'], 'warning': WARNING,
        'parent_checkpoint': report['parent_checkpoint'], 'training_config': report['training_config'],
        'role': role, 'critic_unchanged': True, 'initial_critic': report['initial_critic'],
        'progress': {key: report[key] for key in (
            'status', 'stop_reason', 'epochs_completed', 'current_epoch',
            'supervised_samples', 'optimizer_updates', 'validation_samples', 'environment_interactions')},
    }
    return save_precision_checkpoint(model, path, migration_record=model.precision_migration_record,
                                     dataset_config=config)


def train_precision_imitation(dataset_paths, initial_checkpoint, output, *, epochs=20,
                               batch_size=128, learning_rate=1e-4, seed=0, device='cpu',
                               deadline=None, offsets_enabled=True):
    """Train only the actor on precision demonstrations from frozen TRAIN families.

    The initial checkpoint MUST use the precision loader/manifest. Its precision
    Adam is retained with the newly requested LR; migration lineage is unchanged.
    Critic gradients remain None, including when prior critic Adam momentum exists.
    Best uses validation CE only, explicitly NOT final deployment selection.
    Files: initial.zip, epoch-NNNN.zip, final.zip, best.json, report.json, resources.jsonl.
    best.json/report.checkpoints.best points to an existing saved checkpoint;
    checkpoint archives/manifests are never overwritten.
    """
    for name, value, minimum in (('epochs', epochs, 1), ('batch_size', batch_size, 1), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if seed >= 2**32 or type(offsets_enabled) is not bool:
        raise ValueError('seed must be uint32 and offsets_enabled must be boolean')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError('deadline must be a finite absolute monotonic time')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Use an explicit CPU/CUDA device')
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        'format': 'within-stage-precision-imitation-v1', 'status': 'running', 'stop_reason': None,
        'warning': WARNING, 'offsets_enabled': offsets_enabled,
        'evaluation_adapter': {
            'name': 'precision-offsets-identity' if offsets_enabled else 'zero-precision-offsets-each-frame',
            'required': not offsets_enabled,
            'zero_channel_ranges': [] if offsets_enabled else [[47, 55], [102, 110]]},
        'training_config': {'epochs': int(epochs), 'batch_size': int(batch_size),
                            'learning_rate': float(learning_rate), 'seed': int(seed),
                            'device': str(device), 'deadline_monotonic': deadline},
        'epochs_completed': 0, 'current_epoch': 0, 'supervised_samples': 0, 'optimizer_updates': 0,
        'validation_samples': 0, 'environment_interactions': 0,
        'datasets': [], 'split': None, 'family_membership': None, 'sources': {},
        'parent_checkpoint': None, 'critic_unchanged': None, 'initial_critic': None,
        'history': [], 'initial_validation_ce': None, 'best_validation_ce': None,
        'resource_monitor': {'status': 'not_started'},
        'checkpoints': {'epochs': [], 'best': None, 'initial': None, 'final': None}}
    model, corpus = None, None
    try:
        _check(deadline)
        monitor = None
        try:
            monitor = ResourceMonitor(output / 'resources.jsonl', device=str(device).split(':')[0])
            report['resource_monitor'] = {'status': 'enabled', 'path': str(monitor.path)}
            monitor.check('precision_bc_admission', force=True)
        except OSError as error:
            report['resource_monitor'] = {'status': 'unavailable', 'error': str(error)}
            monitor = None
        report['sources'] = _sources()
        membership = _training_contract()
        paths = [Path(path).resolve() for path in paths]
        if not paths or len(set(paths)) != len(paths) or any(output.is_relative_to(path) for path in paths):
            raise ValueError('Use distinct datasets outside the fresh output directory')
        corpus = TrainingCorpus(paths, membership, deadline)
        report['datasets'] = corpus.identities
        train_ids, valid_ids, report['split'] = _split_episodes(corpus.episode_ids, seed)
        report['family_membership'] = {
            key: value for key, value in membership.items() if key != 'allowed_configs'}
        report['family_membership']['episode_origins'] = corpus.membership
        _check(deadline)
        parent = Path(initial_checkpoint).resolve()
        model = load_precision_checkpoint(parent, device=device)
        report['parent_checkpoint'] = {
            'path': str(parent), 'sha256': file_sha256(parent),
            'manifest_sha256': file_sha256(parent.with_suffix('.manifest.json'))}
        report['initial_critic'] = critic_fingerprint(model)
        report['critic_unchanged'] = True
        if str(device).startswith('cuda') and model.device.type != 'cuda':
            raise RuntimeError('Requested CUDA cannot silently fall back to CPU')
        report['actual_device'] = str(model.device)
        model.seed, model.learning_rate = int(seed), learning_rate
        model.set_random_seed(int(seed)); model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group['lr'] = learning_rate
        initial = _evaluate(model, corpus, valid_ids, batch_size, offsets_enabled, deadline, monitor, report)
        report['initial_validation_ce'] = report['best_validation_ce'] = initial['ce']
        saved = _save(model, output / 'initial.zip', report, 'initial_validation_reference')
        report['checkpoints']['initial'] = report['checkpoints']['best'] = saved
        _write(output / 'best.json', saved)
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            report['current_epoch'] = epoch
            model.policy.set_training_mode(True)
            order = rng.permutation(train_ids)
            total, count = 0., 0
            for first in range(0, len(order), batch_size):
                _check(deadline)
                if monitor is not None:
                    monitor.check('precision_bc_actor_minibatch')
                indices = order[first:first + batch_size]
                batch = corpus.batch(indices)
                _check(deadline)
                model.policy.optimizer.zero_grad(set_to_none=True)
                loss, _ = _loss(model, batch, offsets_enabled)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5, error_if_nonfinite=True)
                _check(deadline)
                model.policy.optimizer.step()
                report['optimizer_updates'] += 1
                report['supervised_samples'] += len(indices)
                total += loss.item() * len(indices); count += len(indices)
            valid = _evaluate(model, corpus, valid_ids, batch_size, offsets_enabled, deadline, monitor, report)
            report['epochs_completed'] = epoch
            report['history'].append({'epoch': epoch, 'training_ce': total / count,
                                       'validation_ce': valid['ce'], 'validation_accuracy': valid['accuracy']})
            saved = _save(model, output / f'epoch-{epoch:04d}.zip', report, 'completed_epoch')
            report['checkpoints']['epochs'].append(saved)
            if valid['ce'] < report['best_validation_ce']:
                report['best_validation_ce'] = valid['ce']
                report['checkpoints']['best'] = saved
                _write(output / 'best.json', saved)
            _write(output / 'report.json', report)
        report['status'], report['stop_reason'] = 'completed', 'epochs_completed'
    except (DeadlineReached, ResourcePressureError, KeyboardInterrupt) as error:
        report.update(status='partial', stop_reason=type(error).__name__, error=str(error))
    except Exception as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if corpus is not None:
            try:
                corpus.close()
            except Exception as error:
                report.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
        if model is not None and report['initial_critic'] is not None:
            report['critic_unchanged'] = critic_fingerprint(model) == report['initial_critic']
            try:
                report['checkpoints']['final'] = _save(model, output / 'final.zip', report, 'final_or_partial')
            except Exception as error:
                report.update(status='failed', checkpoint_error=f'{type(error).__name__}: {error}')
        report['elapsed_seconds'] = max(0., time.monotonic() - started)
        if not (output / 'best.json').exists():
            _write(output / 'best.json', report['checkpoints']['best'])
        _write(output / 'report.json', report)
    return report
