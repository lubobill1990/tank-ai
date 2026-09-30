"""Fresh actor-only entity BC using the frozen precision TRAIN corpus/split.

Only current audit RAM for the selected sample indices is converted into
geometric tokens. The model receives grid/entities, never raw RAM or future
snapshots. Epoch CE selection is provisional DEVELOPMENT accuracy; deployment
selection still requires real DEV episodes, not offline CE alone.
"""
import math
from pathlib import Path
import time

import numpy as np
import torch

from within_stage import imitation as precision_bc
from within_stage.entity_observation import entity_batch_from_ram, entity_schema
from within_stage.entity_policy import make_entity_model, save_entity_checkpoint
from within_stage.precision_policy import file_sha256
from rust_resources import ResourceMonitor, ResourcePressureError


ROOT = Path(__file__).resolve().parents[1]
WARNING = ('Actor-only entity BC; internal episode holdout is DEVELOPMENT within TRAIN families. '
           'Best CE is not a deployed-candidate decision: real DEV evaluation is required. '
           'No raw RAM/future snapshot/model action recommendation is a learner input.')


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\entity_imitation.py', 'within_stage\\entity_observation.py',
        'within_stage\\entity_policy.py', 'within_stage\\imitation.py',
        'within_stage\\demonstrations.py', 'within_stage\\observation.py',
        'within_stage\\precision_policy.py', 'game_state.py')}


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise precision_bc.DeadlineReached('Entity BC deadline reached')


class EntityTrainingCorpus(precision_bc.TrainingCorpus):
    """Same membership/indices/grid/actions as precision BC, plus current tokens."""
    def batch(self, indices):
        values = super().batch(indices)
        indices = np.asarray(indices, dtype=np.int64)
        rams = np.empty((len(indices), 2048), np.uint8)
        owners = np.searchsorted(self.offsets[1:], indices, side='right')
        for owner in np.unique(owners):
            positions = np.flatnonzero(owners == owner)
            local = indices[positions] - self.offsets[owner]
            dataset = self.datasets[owner]
            audit = dataset.audit(local)
            if audit['ram'].dtype != np.uint8 or audit['ram'].shape != (len(local), 2048) \
                    or not np.array_equal(audit['episode_ids'], dataset.episode_ids[local]):
                raise ValueError('Current audit RAM/episode indices disagree with learner batch')
            rams[positions] = audit['ram']
        return {'observations': entity_batch_from_ram(values['observations'], rams),
                'actions': values['actions'], 'action_masks': values['action_masks']}


def _loss(model, batch):
    observation = {key: torch.as_tensor(value, device=model.device)
                   for key, value in batch['observations'].items()}
    actions = torch.as_tensor(batch['actions'], device=model.device)
    masks = torch.as_tensor(batch['action_masks'], device=model.device)
    logits = model.policy.get_distribution(observation).distribution.logits
    loss = precision_bc.masked_ce(logits, actions, masks)
    correct = int((logits.detach().masked_fill(~masks, float('-inf')).argmax(1) == actions).sum().item())
    return loss, correct


def _evaluate(model, corpus, indices, batch_size, deadline, monitor, report):
    model.policy.set_training_mode(False)
    total, count, correct = 0., 0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            _check(deadline)
            if monitor is not None:
                monitor.check('entity_bc_validation')
            selection = indices[start:start + batch_size]
            batch = corpus.batch(selection)
            _check(deadline)
            loss, matches = _loss(model, batch)
            total += loss.item() * len(selection)
            correct += matches; count += len(selection)
            report['validation_samples'] += len(selection)
    return {'ce': total / count, 'accuracy': correct / count}


def _save(model, path, report, role):
    if _sources() != report['sources']:
        raise ValueError('Entity BC source identity changed')
    membership = report['family_membership']
    if file_sha256(membership['frozen_split_path']) != membership['frozen_split_sha256'] \
            or file_sha256(membership['pool_path']) != membership['pool_sha256']:
        raise ValueError('Frozen TRAIN split/pool identity changed')
    if precision_bc.critic_fingerprint(model) != report['initial_critic']:
        raise ValueError('Actor-only entity BC modified critic weights or Adam state')
    config = {
        'schema': entity_schema(), 'datasets': report['datasets'], 'split': report['split'],
        'family_membership': membership, 'sources': report['sources'],
        'objective': 'actor_only_masked_entity_bc', 'parent_checkpoint': None,
        'initialization': 'fresh-independent-actor-and-critic', 'role': role,
        'deployment_selection_note': 'Offline CE best is not deployment selection; run real DEV evaluation.',
        'audit_conversion': 'entity_batch_from_ram on current rows only; raw RAM excluded from model dict',
        'training_config': report['training_config'], 'initial_critic': report['initial_critic'],
        'progress': {key: report[key] for key in (
            'status', 'stop_reason', 'epochs_completed', 'current_epoch',
            'optimizer_updates', 'supervised_samples', 'validation_samples', 'environment_interactions')},
    }
    return save_entity_checkpoint(model, path, dataset_config=config)


def train_entity_imitation(dataset_paths, output, *, epochs=20, batch_size=128,
                            learning_rate=1e-4, seed=0, device='cpu', deadline=None):
    """Fresh masked actor BC. No old CNN checkpoint argument or environment calls.

    Writes initial.zip, epoch-NNNN.zip, final.zip, best.json and report.json.
    best.json points to a saved epoch/initial archive, not an extra overwritten
    model. Counts exclude validation from supervised updates and remain0 for
    environment_interactions. Partial runs save current completed update state.
    """
    for name, value, low in (('epochs', epochs, 1), ('batch_size', batch_size, 1), ('seed', seed, 0)):
        if type(value) is not int or value < low:
            raise ValueError(f'{name} must be an integer >= {low}')
    if seed >= 2**32 or isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) \
            or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Use uint32 seed and positive finite learning_rate')
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not math.isfinite(deadline)):
        raise ValueError('deadline must be finite absolute monotonic time')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Select explicit CPU/CUDA device')
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        'format': 'within-stage-entity-imitation-v1', 'status': 'running', 'stop_reason': None,
        'warning': WARNING, 'parent_checkpoint': None,
        'deployment_selection_performed': False,
        'best_criterion': 'within-TRAIN-episode-validation-CE-candidate-only',
        'training_config': {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                            'seed': seed, 'device': str(device), 'deadline_monotonic': deadline},
        'epochs_completed': 0, 'current_epoch': 0, 'supervised_samples': 0, 'optimizer_updates': 0,
        'validation_samples': 0, 'environment_interactions': 0, 'history': [],
        'initial_validation_ce': None, 'best_validation_ce': None, 'initial_critic': None,
        'critic_unchanged': None, 'sources': {}, 'datasets': [], 'split': None, 'family_membership': None,
        'resource_monitor': {'status': 'not_started'},
        'checkpoints': {'initial': None, 'epochs': [], 'best': None, 'final': None}}
    corpus = model = monitor = None
    try:
        _check(deadline)
        try:
            monitor = ResourceMonitor(output / 'resources.jsonl', device=str(device).split(':')[0])
            report['resource_monitor'] = {'status': 'enabled', 'path': str(monitor.path)}
            monitor.check('entity_bc_admission', force=True)
        except OSError as error:
            report['resource_monitor'] = {'status': 'unavailable', 'error': str(error)}
            monitor = None
        report['sources'] = _sources()
        contract = precision_bc._training_contract()
        paths = [Path(path).resolve() for path in paths]
        if not paths or len(set(paths)) != len(paths) or any(output.is_relative_to(path) for path in paths):
            raise ValueError('Distinct datasets and a separate fresh output are required')
        corpus = EntityTrainingCorpus(paths, contract, deadline)
        report['datasets'] = corpus.identities
        train, valid, report['split'] = precision_bc._split_episodes(corpus.episode_ids, seed)
        report['family_membership'] = {k: v for k, v in contract.items() if k != 'allowed_configs'}
        report['family_membership']['episode_origins'] = corpus.membership
        _check(deadline)
        model = make_entity_model(device=device, seed=seed, lr=learning_rate)
        report['actual_device'] = str(model.device)
        report['initial_critic'] = precision_bc.critic_fingerprint(model)
        report['critic_unchanged'] = True
        initial = _evaluate(model, corpus, valid, batch_size, deadline, monitor, report)
        report['initial_validation_ce'] = report['best_validation_ce'] = initial['ce']
        checkpoint = _save(model, output / 'initial.zip', report, 'initial_validation_reference')
        report['checkpoints']['initial'] = report['checkpoints']['best'] = checkpoint
        precision_bc._write(output / 'best.json', checkpoint)
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            report['current_epoch'] = epoch
            model.policy.set_training_mode(True)
            order = rng.permutation(train)
            total, count = 0., 0
            for start_index in range(0, len(order), batch_size):
                _check(deadline)
                if monitor is not None:
                    monitor.check('entity_bc_actor_update')
                selection = order[start_index:start_index + batch_size]
                batch = corpus.batch(selection)
                _check(deadline)
                model.policy.optimizer.zero_grad(set_to_none=True)
                loss, _ = _loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5, error_if_nonfinite=True)
                _check(deadline)
                model.policy.optimizer.step()
                report['optimizer_updates'] += 1
                report['supervised_samples'] += len(selection)
                total += loss.item() * len(selection); count += len(selection)
            evaluation = _evaluate(model, corpus, valid, batch_size, deadline, monitor, report)
            report['epochs_completed'] = epoch
            report['history'].append({'epoch': epoch, 'training_ce': total / count,
                                      'validation_ce': evaluation['ce'],
                                      'validation_accuracy': evaluation['accuracy']})
            checkpoint = _save(model, output / f'epoch-{epoch:04d}.zip', report, 'completed_epoch')
            report['checkpoints']['epochs'].append(checkpoint)
            if evaluation['ce'] < report['best_validation_ce']:
                report['best_validation_ce'] = evaluation['ce']
                report['checkpoints']['best'] = checkpoint
                precision_bc._write(output / 'best.json', checkpoint)
            precision_bc._write(output / 'report.json', report)
        report.update(status='completed', stop_reason='epochs_completed')
    except (precision_bc.DeadlineReached, ResourcePressureError, KeyboardInterrupt) as error:
        report.update(status='partial', stop_reason=type(error).__name__, error=str(error))
    except Exception as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if corpus is not None:
            try:
                corpus.close()
            except Exception as error:
                report.update(status='failed', cleanup_error=str(error))
        if model is not None:
            report['critic_unchanged'] = precision_bc.critic_fingerprint(model) == report['initial_critic']
            try:
                report['checkpoints']['final'] = _save(model, output / 'final.zip', report, 'final_or_partial')
            except Exception as error:
                report.update(status='failed', checkpoint_error=f'{type(error).__name__}: {error}')
        report['elapsed_seconds'] = max(0., time.monotonic() - start)
        if not (output / 'best.json').exists():
            precision_bc._write(output / 'best.json', report['checkpoints']['best'])
        precision_bc._write(output / 'report.json', report)
    return report
