"""Full-trainable ROM PPO plus separate post-update expert-anchor rehearsal.

Rehearsal is NOT on-policy data: four 256-example masked-CE minibatches use
only clean winning ROM teacher-V2 episodes0/1. PPO and replay commit atomically.
No actuator noise, feature freezing, new environment steps, or teacher control
is introduced by replay. Original ROM fine-tune evaluation/retention is reused.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from learning_lab import imitation, rom_finetune as control
from learning_lab.demonstrations import DemonstrationDataset
from learning_lab.collect_teacher_v2 import verify_integrity


FORMAT = 'tank-ai-rom-rehearsal-checkpoint-v1'
REPLAY = {
    'selected_episode_ids': [0, 1], 'selected_episode_counts': [1159, 1206],
    'anchor_samples': 2365, 'minibatches': 4, 'batch_size': 256,
    'loss_weight': 1.0, 'learning_rate': 5e-5, 'max_grad_norm': .5,
    'loss': 'masked_teacher_cross_entropy', 'sampling': 'uniform-with-replacement',
    'phase': 'after-successful-PPO-before-transaction-commit',
    'on_policy': False, 'new_environment_decisions': 0,
}


def _contract():
    return {**control._contract(), 'rehearsal': deepcopy(REPLAY),
            'auxiliary_bc': True, 'feature_freezing': False}


def _sources():
    return {**control._sources(),
            'learning_lab\\rom_rehearsal.py': control.machinery._hash(Path(__file__)),
            'learning_lab\\demonstrations.py': control.machinery._hash(
                control.ROOT / 'learning_lab' / 'demonstrations.py'),
            'learning_lab\\collect_teacher_v2.py': control.machinery._hash(
                control.ROOT / 'learning_lab' / 'collect_teacher_v2.py')}


def _check_episode(row, episode, count):
    if row.get('episode_id') != episode or row.get('stage') != 1 or row.get('seed') != episode \
            or row.get('outcome') != 'clear' or row.get('clear') is not True \
            or row.get('episode_finished') is not True or row.get('error') is not None \
            or row.get('perturbation_probability') != 0 or row.get('changed_actions') != 0 \
            or any(row.get(key) != count for key in (
                'decisions', 'attempted_decisions', 'completed_decisions')) \
            or row.get('terminal_remaining_enemies') != 0 \
            or row.get('terminal_base_destroyed') is not False:
        raise ValueError('Anchors must be complete clean stage1 teacher-V2 wins on source seeds0/1')


def _verify_anchor_identity(identity):
    try:
        if identity['selected_episode_ids'] != [0, 1] \
                or identity['selected_episode_counts'] != [1159, 1206] \
                or identity['selected_samples'] != 2365 or identity['new_environment_decisions'] != 0:
            raise ValueError('Incorrect selected anchor episodes/counts')
        dataset = identity['dataset']
        metadata, files = dataset['metadata'], dataset['files']
        if metadata.get('teacher_class') != 'learning_lab.teacher_v2.TeacherControllerV2' \
                or metadata.get('learned_model') is not False \
                or not metadata.get('provenance', '').startswith('Normal original-ROM reset') \
                or str(metadata.get('backend', '')).startswith('NATIVE') or not files:
            raise ValueError('Anchor identity is not the ROM teacher-V2 corpus')
        for entry in files:
            if not isinstance(entry['path'], str) or type(entry['bytes']) is not int or entry['bytes'] < 0 \
                    or not isinstance(entry['sha256'], str) or len(entry['sha256']) != 64 \
                    or any(c not in '0123456789abcdef' for c in entry['sha256']):
                raise ValueError('Malformed anchor file identity')
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if digest != dataset['sha256']:
            raise ValueError('Anchor dataset SHA256 mismatch')
        sealed_files = [entry for entry in files if entry['path'] != 'integrity.json']
        seals = [entry for entry in files if entry['path'] == 'integrity.json']
        if len(seals) != 1 or seals[0]['sha256'] != identity['integrity_sha256'] \
                or hashlib.sha256(json.dumps(sealed_files, sort_keys=True).encode()).hexdigest() \
                != identity['sealed_fingerprint_sha256']:
            raise ValueError('Anchor seal identity mismatch')
        if len(identity['selected_episode_reports']) != 2:
            raise ValueError('Missing selected episode provenance')
        for episode, count in enumerate((1159, 1206)):
            _check_episode(identity['selected_episode_reports'][episode], episode, count)
        if identity['selected_indices_sha256'] != hashlib.sha256(
                np.arange(2365, dtype=np.int64).tobytes()).hexdigest():
            raise ValueError('Anchor row selection mismatch')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed anchor identity') from error


class AnchorDataset:
    """Owned index selection over a read-only corpus; unselected frames are never batched."""
    def __init__(self, path, deadline=None):
        self.path = Path(path).resolve()
        fingerprint = verify_integrity(
            self.path, None if deadline is None else time.time() + max(0., deadline - time.monotonic()))
        self.dataset = DemonstrationDataset(self.path)
        try:
            rows = json.loads((self.path / 'episode_reports.json').read_text())
            counts = [int(np.count_nonzero(self.dataset.episode_ids == episode)) for episode in (0, 1)]
            self.indices = np.flatnonzero(np.isin(self.dataset.episode_ids, [0, 1]))
            self.indices.setflags(write=False)
            self.identity = {
                'dataset': imitation._dataset_identity(self.path, 'anchors', self.dataset.metadata, deadline),
                'selected_episode_ids': [0, 1], 'selected_episode_counts': counts,
                'selected_samples': len(self.indices), 'selected_episode_reports': rows[:2],
                'selected_indices_sha256': hashlib.sha256(self.indices.tobytes()).hexdigest(),
                'new_environment_decisions': 0,
                'sealed_fingerprint_sha256': fingerprint,
                'integrity_sha256': control.machinery._hash(self.path / 'integrity.json'),
            }
            _verify_anchor_identity(self.identity)
            labels = np.load(self.path / 'teacher_actions.npy', allow_pickle=False)
            executed = np.load(self.path / 'executed_actions.npy', allow_pickle=False)
            if not np.array_equal(labels[self.indices], executed[self.indices]):
                raise ValueError('Selected clean anchors contain perturbed executed actions')
        except BaseException:
            self.dataset.close()
            raise

    def sample_indices(self, rng, size):
        return rng.choice(self.indices, size=size, replace=True)

    def batch(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' \
                or not np.isin(indices, self.indices).all():
            raise ValueError('Replay requested a row outside selected anchor episodes0/1')
        batch = self.dataset.batch(indices)
        imitation._validate_batch(batch, self.dataset.episode_ids[indices])
        return batch

    def close(self):
        self.dataset.close()


def rehearse(model, anchors, rng, guard, progress, *, batch_size=256, minibatches=4):
    """A separate supervised phase; never reads or writes the on-policy rollout buffer."""
    if not all(parameter.requires_grad for parameter in model.policy.parameters()):
        raise ValueError('Rehearsal variant requires fully trainable features and heads')
    model.policy.set_training_mode(True)
    for group in model.policy.optimizer.param_groups:
        group['lr'] = REPLAY['learning_rate']
    for _ in range(minibatches):
        guard.check('rehearsal_minibatch')
        indices = anchors.sample_indices(rng, batch_size)
        batch = anchors.batch(indices)
        guard.check('rehearsal_forward')
        model.policy.optimizer.zero_grad(set_to_none=True)
        loss, correct = imitation._batch_loss(model, batch)
        loss = loss * REPLAY['loss_weight']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite rehearsal loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), REPLAY['max_grad_norm'],
                                       error_if_nonfinite=True)
        guard.check('rehearsal_adam')
        model.policy.optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
            raise FloatingPointError('Nonfinite policy after rehearsal')
        progress['bc_updates'] += 1
        progress['supervised_samples'] += len(indices)
        progress['losses'].append({'ce': float(loss.item()), 'accuracy': correct / len(indices)})
    return progress


class RehearsalRuntime(control._Runtime):
    def __init__(self, *args, anchors, replay_batch_size=256, **kwargs):
        self.anchors = anchors
        self.replay_batch_size = replay_batch_size
        self.rehearsal_source_pin = _sources()
        self.replay_rng = np.random.default_rng(kwargs['seed'])
        self.supervised_samples = self.bc_updates = self.ppo_adam_steps = 0
        self.pending_replay = None
        self.discarded_replay = None
        super().__init__(*args, **kwargs)
        if not all(parameter.requires_grad for parameter in self.model.policy.parameters()):
            self.close()
            raise ValueError('Rehearsal must not inherit frozen parameters')

    def optimize(self):
        self.guard.check('combined_update')
        before = self._last_good
        previous_count = self._step_count['count']
        previous_rng = deepcopy(self.replay_rng.bit_generator.state)
        previous_progress = self.model._current_progress_remaining
        self.pending_replay = {'bc_updates': 0, 'supervised_samples': 0, 'losses': []}
        self.state_verified = False
        try:
            self.model._update_current_progress_remaining(self.model.num_timesteps, self.total)
            self.model.train()
            ppo_updates = self._step_count['count'] - previous_count
            rehearse(self.model, self.anchors, self.replay_rng, self.guard, self.pending_replay,
                     batch_size=self.replay_batch_size, minibatches=REPLAY['minibatches'])
            self.guard.check('combined_commit')
            committed = self.export_warm()
        except BaseException:
            self.discarded_replay = deepcopy(self.pending_replay)
            self.model.policy.load_state_dict(before['policy'], strict=True)
            self.model.policy.optimizer.load_state_dict(before['optimizer'])
            self.model.policy.optimizer.zero_grad(set_to_none=True)
            self.model._n_updates = before['ppo_epoch_updates']
            self.model._current_progress_remaining = previous_progress
            self._step_count['count'] = previous_count
            self.replay_rng.bit_generator.state = previous_rng
            self.state_verified = True
            raise
        self.optimized_steps += self.rollout_size
        self.ppo_adam_steps += ppo_updates
        self.bc_updates += self.pending_replay['bc_updates']
        self.supervised_samples += self.pending_replay['supervised_samples']
        self.adam_steps = self._step_count['count']
        self._last_good, self.state_verified = committed, True
        self.model.logger.record('rehearsal/ce', float(np.mean([r['ce'] for r in self.pending_replay['losses']])))
        self.model.logger.record('rehearsal/accuracy', float(np.mean([r['accuracy'] for r in self.pending_replay['losses']])))
        self.model.logger.record('rehearsal/supervised_samples', self.supervised_samples)
        self.model.logger.record('rehearsal/bc_updates', self.bc_updates)
        self.model.logger.record('train/ppo_adam_steps', self.ppo_adam_steps)


def _save_checkpoint(runtime, path, parent, role):
    if not runtime.state_verified or runtime.optimized_steps % control.ROLLOUT:
        raise ValueError('Only initial or jointly committed PPO+replay updates may be saved')
    transactions = runtime.optimized_steps // control.ROLLOUT
    if runtime.bc_updates != transactions * 4 or runtime.supervised_samples != runtime.bc_updates * 256 \
            or runtime.adam_steps != runtime.ppo_adam_steps + runtime.bc_updates:
        raise ValueError('Checkpoint does not have complete four-minibatch rehearsal transactions')
    if runtime.rehearsal_source_pin != _sources():
        raise ValueError('Rehearsal source identity changed')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    collected = runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps = runtime.optimized_steps
        runtime.model.save(path, exclude=['_last_obs', '_last_episode_starts', '_last_original_obs'])
    finally:
        runtime.model.num_timesteps = collected
    manifest = {
        'format': FORMAT, 'contract': _contract(), 'checkpoint_file': path.name,
        'checkpoint_sha256': control.machinery._hash(path), 'sources': runtime.rehearsal_source_pin,
        'parent': parent, 'anchors': deepcopy(runtime.anchors.identity), 'role': role,
        'optimized_steps': runtime.optimized_steps, 'adam_steps': runtime.adam_steps,
        'ppo_adam_steps': runtime.ppo_adam_steps, 'bc_updates': runtime.bc_updates,
        'supervised_samples': runtime.supervised_samples, 'replay_environment_decisions': 0,
        'ppo_epoch_updates': int(runtime.model._n_updates), 'exact_simulator_resume': False,
        'policy_sha256': control._state_hash(runtime.model.policy.state_dict()),
        'adam_sha256': control._state_hash(runtime.model.policy.optimizer.state_dict()),
        'start_policy_sha256': runtime.start_policy_sha256, 'start_adam_sha256': runtime.start_adam_sha256,
        'warning': 'Rehearsal teacher labels are auxiliary supervised data, never on-policy transitions. '
                   'Development retention is not unseen-map generalization.',
    }
    control.machinery._write(path.with_suffix('.manifest.json'), manifest)
    return {'path': str(path), 'sha256': manifest['checkpoint_sha256'], 'optimized_steps': runtime.optimized_steps}


def load_rehearsal_checkpoint(path, device='cpu'):
    """Strict trusted-local model+Adam loader; no source dataset or environment is opened."""
    path = Path(path).resolve()
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        parent = manifest['parent']
        if manifest['format'] != FORMAT or manifest['contract'] != _contract() \
                or manifest['sources'] != _sources() or manifest['checkpoint_file'] != path.name \
                or manifest['exact_simulator_resume'] is not False \
                or isinstance(manifest['contract']['ppo']['gamma'], bool) \
                or type(manifest['optimized_steps']) is not int or not 0 <= manifest['optimized_steps'] <= 65536 \
                or manifest['optimized_steps'] % control.ROLLOUT \
                or parent['manifest']['format'] != imitation._FORMAT \
                or parent['manifest']['contract'] != imitation._CONTRACT \
                or parent['manifest']['sources'] != imitation._source_hashes():
            raise ValueError('Invalid rehearsal contract/source/parent')
        for key in ('sha256', 'manifest_sha256'):
            digest = parent[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid parent SHA256')
        if parent['sha256'] != parent['manifest']['checkpoint_sha256']:
            raise ValueError('Parent identity mismatch')
        imitation._verify_provenance(parent['manifest'])
        _verify_anchor_identity(manifest['anchors'])
        transactions = manifest['optimized_steps'] // control.ROLLOUT
        if manifest['bc_updates'] != transactions * 4 \
                or manifest['supervised_samples'] != manifest['bc_updates'] * 256 \
                or manifest['adam_steps'] != manifest['ppo_adam_steps'] + manifest['bc_updates'] \
                or manifest['replay_environment_decisions'] != 0:
            raise ValueError('Invalid committed PPO/rehearsal counters')
        if control.machinery._hash(path) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA256 mismatch')
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('Missing or malformed rehearsal manifest') from error
    model = MaskablePPO.load(path, device=device, custom_objects={
        'rollout_buffer_class': control.NativeGPURunRolloutBuffer,
        'rollout_buffer_kwargs': {'channels_per_frame': 47, 'max_storage_bytes': 2 * 1024**3},
        'n_steps': 2, 'n_envs': 1, 'batch_size': 2})
    imitation._verify_model(model)
    if isinstance(model.gamma, bool) or model.gamma != 1. or model.gae_lambda != .95 \
            or model.learning_rate != 5e-5 or model.ent_coef != .001 or model.target_kl != .01 \
            or model.n_epochs != 4 or model.clip_range(1.) != .1 \
            or model.num_timesteps != manifest['optimized_steps'] \
            or model._n_updates != manifest['ppo_epoch_updates'] \
            or control._state_hash(model.policy.state_dict()) != manifest['policy_sha256'] \
            or control._state_hash(model.policy.optimizer.state_dict()) != manifest['adam_sha256'] \
            or not all(parameter.requires_grad for parameter in model.policy.parameters()):
        raise ValueError('Serialized full-trainable PPO/Adam contract mismatch')
    return model


strict_load_rehearsal_checkpoint = load_rehearsal_checkpoint


def run_rehearsal_finetune(initial_checkpoint, anchor_dataset_path, output, budget, *,
                           steps=65536, seed=0, device='cpu'):
    """Original full-trainable ROM protocol plus atomic post-PPO rehearsal only."""
    if type(steps) is not int or not control.ROLLOUT <= steps <= 65536 or steps % control.ROLLOUT:
        raise ValueError('steps must be a multiple of1024 in1024..65536')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    if str(device) == 'auto' or torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('Choose an explicit CPU/CUDA device')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format='tank-ai-rom-rehearsal-run-v1', status='running', stop_reason=None,
                  contract=_contract(), requested_steps=steps, seed=seed, device=str(device),
                  rollouts=[], evaluations=[], checkpoints=[], optimized_steps=0, adam_steps=0,
                  ppo_adam_steps=0, bc_updates=0, supervised_samples=0, replay_environment_decisions=0,
                  bootstrap_reservations=[], unresolved_reservations=[], train_completed_decisions=0,
                  exact_simulator_resume=False, no_teacher_control=True, best_checkpoint=None)
    runtime, bootstrap, parent, anchors = None, None, None, None
    prefix = 'rom-rehearsal-' + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    control.machinery._write(output / 'report.json', report)
    try:
        state = budget.snapshot()
        if not 0 < state['max_interactions'] <= 4000000:
            raise ValueError('Research ledger cap must not exceed4M')
        seconds = max(0., min(state['remaining_seconds'], state['deadline_epoch'] - time.time(),
                              control.GLOBAL_DEADLINE - time.time()))
        monitor = control.ResourceMonitor(output / 'resources.jsonl', device=str(device).split(':')[0])
        guard = control.machinery._Guard(time.monotonic() + seconds,
                                         min(state['deadline_epoch'], control.GLOBAL_DEADLINE), monitor)
        guard.check('admission'); monitor.check('admission', admission_slots=1, force=True)
        anchors = AnchorDataset(anchor_dataset_path, guard.deadline)
        report['anchors'] = anchors.identity
        warm = imitation.load_imitation_checkpoint(initial_checkpoint, device='cpu')
        parent = control._parent_identity(initial_checkpoint)
        report['parent'], report['sources'] = parent, _sources()
        bootstrap = control.BootstrapMeter(budget, prefix, guard)
        report['bootstrap_reservations'] = bootstrap.rows
        bootstrap.begin_initial()
        control.machinery._write(output / 'report.json', report)
        runtime = RehearsalRuntime(warm, output, guard, bootstrap, anchors=anchors,
                                   seed=seed, device=device, max_steps=steps)
        del warm
        bootstrap.flush_success()
        initial = _save_checkpoint(runtime, output / 'initial.zip', parent, 'initial')
        report['checkpoints'].append(initial); report['best_checkpoint'] = initial
        control.machinery._write(output / 'report.json', report)
        baseline = control._evaluate(runtime.model, output / 'development-initial', budget, guard, initial, 0)
        report['evaluations'].append(baseline)
        if baseline['status'] != 'completed':
            report.update(status='partial', stop_reason='incomplete_initial_development')
            return report
        selection = control.DevelopmentSelection(baseline)
        if baseline['train_wins'] < 2:
            report.update(status='partial', stop_reason='initial_two_train_wins_not_reproduced')
            return report
        while runtime.optimized_steps < steps:
            guard.check('rollout_start')
            row = control._collect_reserved(runtime, budget, bootstrap,
                                             f'{prefix}-rollout-{len(report["rollouts"])}')
            report['rollouts'].append(row)
            report['train_completed_decisions'] += row['completed_decisions']
            if not row['complete']:
                report.update(status='partial', stop_reason='unresolved_or_partial_rollout', error=row['error'])
                break
            runtime.optimize()
            for name in ('optimized_steps', 'adam_steps', 'ppo_adam_steps', 'bc_updates', 'supervised_samples'):
                report[name] = getattr(runtime, name)
            metrics = control._json_safe(dict(runtime.model.logger.name_to_value))
            with (output / 'optimizer-metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'optimized_steps': runtime.optimized_steps, **metrics},
                                         allow_nan=False) + '\n')
            runtime.model.logger.dump(runtime.optimized_steps)
            checkpoint = None
            if runtime.optimized_steps % 8192 == 0:
                checkpoint = _save_checkpoint(runtime, output / f'optimized-{runtime.optimized_steps:06d}.zip',
                                               parent, 'periodic')
                report['checkpoints'].append(checkpoint)
            if runtime.optimized_steps % 16384 == 0:
                evaluation = control._evaluate(
                    runtime.model, output / f'development-{runtime.optimized_steps:06d}',
                    budget, guard, checkpoint, len(report['evaluations']))
                report['evaluations'].append(evaluation)
                if evaluation['status'] != 'completed':
                    report.update(status='partial', stop_reason='incomplete_development')
                    break
                better, stop = selection.consider(evaluation)
                if better:
                    report['best_checkpoint'] = checkpoint
                report['below_initial_completed_checks'] = selection.below_initial_checks
                if stop:
                    report.update(status='completed', stop_reason='two_completed_train_win_regressions')
                    break
            control.machinery._write(output / 'report.json', report)
        if report['status'] == 'running':
            report.update(status='completed', stop_reason='step_cap')
    except (control.DeadlineReached, control.BudgetExhausted, control.ResourcePressureError,
            TimeoutError, imitation._DeadlineReached) as error:
        report.update(status='partial', stop_reason=type(error).__name__, error=str(error))
    except BaseException as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if runtime is not None:
            try:
                for name in ('optimized_steps', 'adam_steps', 'ppo_adam_steps', 'bc_updates', 'supervised_samples'):
                    report[name] = getattr(runtime, name)
                report['discarded_replay'] = runtime.discarded_replay
                report['checkpoints'].append(_save_checkpoint(
                    runtime, output / 'final.zip', parent, 'final_jointly_committed'))
            except Exception as error:
                report['checkpoint_error'] = f'{type(error).__name__}: {error}'
            try:
                runtime.close()
            except Exception as error:
                report['cleanup_error'] = f'{type(error).__name__}: {error}'
        if anchors is not None:
            anchors.close()
        rows = list(report['rollouts']) + list(report['bootstrap_reservations'])
        for evaluation in report['evaluations']:
            rows.extend(evaluation['reservations']); rows.extend(evaluation['bootstrap_reservations'])
        report['unresolved_reservations'] = [row['reservation'] for row in rows if not row['settled']]
        report['budget_snapshot'] = budget.snapshot()
        control.machinery._write(output / 'report.json', report)
        control.machinery._write(output / 'best.json', report['best_checkpoint'])
    return report
