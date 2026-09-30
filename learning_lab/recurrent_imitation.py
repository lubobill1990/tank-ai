"""Causal current-frame CNN/LSTM cloning with chronological whole-episode TBPTT.

This is a separate .pt policy format, not a MaskablePPO archive. Training and
validation use unweighted masked teacher CE; no value/reward learning occurs.
The ROM adapter consumes only the current last 47 planes of a two-frame stack.
Memory resets at an all-zero previous-frame prefix or explicit reset(), never
from life/death fields. Checkpoints do not contain live recurrent state.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from learning_lab import imitation as base
from policy import Grid26CNN


_FORMAT = 'tank-ai-recurrent-imitation-checkpoint-v1'
_CONTRACT = {
    'architecture': 'Grid26CNN-256-LSTM-128-actor-10-v1',
    'wrapper_observation_shape': [94, 26, 26],
    'recurrent_input_shape': [47, 26, 26], 'input_slice': 'current-last-47-only',
    'observation_dtype': 'float32', 'features_dim': 256, 'hidden_size': 128,
    'num_layers': 1, 'bidirectional': False, 'dropout': 0.0, 'actions': 10,
    'objective': 'unweighted_masked_teacher_cross_entropy', 'class_weighting': None,
    'training_order': 'shuffle-whole-episodes-only; chronological-chunks-within-episode',
    'hidden_training': 'zero-at-episode-start; carry-and-detach-after-each-chunk',
    'inference_reset': 'explicit-reset-or-all-zero-previous-47-planes',
    'reset_on_death': False, 'future_context': False, 'value_function_supervision': False,
}
_DECODER = {
    'name': 'causal-lstm-masked-categorical-v1',
    'input': 'current-last-47-of-canonical-94-plane-stack',
    'memory': 'one-stream-internal-lstm-state',
    'reset': 'explicit-reset-or-all-zero-previous-47-planes',
    'reset_on_death': False, 'deterministic': 'masked-argmax',
    'sampled': 'masked-categorical', 'uses_teacher_or_game_state_rules': False,
}


def _source_hashes():
    return {**base._source_hashes(),
            'learning_lab/recurrent_imitation.py': base._sha256(Path(__file__))}


def detach_state(state):
    return None if state is None else tuple(value.detach() for value in state)


class RecurrentActor(nn.Module):
    """One unidirectional stream: (time, 47, 26, 26) -> (time, 10), (h, c)."""
    def __init__(self):
        super().__init__()
        self.extractor = Grid26CNN(spaces.Box(-1, 1, (47, 26, 26), np.float32),
                                   features_dim=256)
        self.lstm = nn.LSTM(256, 128, num_layers=1, batch_first=True,
                            dropout=0, bidirectional=False)
        self.actor = nn.Linear(128, 10)

    def forward(self, frames, state=None):
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (47, 26, 26) \
                or frames.dtype != torch.float32 or len(frames) == 0:
            raise ValueError('Expected nonempty float32 current-frame sequence (T, 47, 26, 26)')
        features = self.extractor(frames)
        output, hidden = self.lstm(features.unsqueeze(0), state)
        return self.actor(output.squeeze(0)), hidden


class RecurrentPolicy:
    """Single-stream ROM-player adapter; predict returns (action, None).

    Accepts (94,26,26), or a single leading batch dimension. Multiple concurrent
    environments require independent wrappers; mixing streams is rejected.
    Calling reset() or seeing a canonical zero previous-frame prefix clears h/c.
    No death, score, teacher state, waypoint, or future frame is inspected.
    """
    learned_model = True
    observation_space = spaces.Box(-1, 1, (94, 26, 26), np.float32)
    action_space = spaces.Discrete(10)

    def __init__(self, policy, optimizer=None):
        self.policy = policy
        self.optimizer = optimizer
        self.device = next(policy.parameters()).device
        self.decoder_config = deepcopy(_DECODER)
        self._hidden = None

    def reset(self):
        self._hidden = None

    def predict(self, observation, *, deterministic=True, action_masks):
        observation, masks = np.asarray(observation), np.asarray(action_masks)
        vectorized = observation.shape == (1, 94, 26, 26)
        if vectorized:
            observation = observation[0]
        if observation.shape != (94, 26, 26):
            raise ValueError('Recurrent predict requires a single canonical two-frame observation')
        if observation.dtype != np.float32 or not np.isfinite(observation).all():
            raise ValueError('Expected a finite float32 observation')
        if masks.shape == (1, 10):
            masks = masks[0]
        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
            raise ValueError('Expected a nonempty ten-action boolean mask')
        state = None if not observation[:47].any() else self._hidden
        self.policy.eval()
        with torch.no_grad():
            frames = torch.as_tensor(observation[47:][None], device=self.device)
            logits, hidden = self.policy(frames, state)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite recurrent logits')
            legal = torch.as_tensor(masks[None], device=self.device)
            logits = logits.masked_fill(~legal, float('-inf'))
            action = logits.argmax(dim=1) if deterministic else torch.distributions.Categorical(
                logits=logits).sample()
            self._hidden = detach_state(hidden)
        actions = action.cpu().numpy().astype(np.int64, copy=False)
        return (actions if vectorized else actions[0]), None


def _episode_indices(corpus, episode_ids):
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Expected distinct nonempty whole-episode IDs')
    result = []
    for episode_id in episode_ids:
        indices = np.flatnonzero(corpus.episode_ids == episode_id)
        if not len(indices) or (len(indices) > 1 and not np.all(np.diff(indices) == 1)):
            raise ValueError(f'Episode must exist as one chronological contiguous block: {episode_id}')
        result.append((episode_id, indices))
    return result


def _run_episodes(policy, corpus, episodes, sequence_length, deadline, monitor, report,
                  *, optimizer=None, deadline_check=base._check_deadline):
    training = optimizer is not None
    policy.train(training)
    device = next(policy.parameters()).device
    total, correct, samples = 0., 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for episode_id, indices in episodes:
            hidden = None
            report['current_episode_id'] = episode_id
            for start in range(0, len(indices), sequence_length):
                deadline_check(deadline)
                if monitor is not None:
                    monitor.check('recurrent_training_chunk' if training else 'recurrent_validation_chunk')
                selection = indices[start:start + sequence_length]
                report['current_chunk_start'] = start
                batch = corpus.batch(selection)
                deadline_check(deadline)
                frames = torch.as_tensor(batch['observations'][:, 47:], device=device)
                actions = torch.as_tensor(batch['actions'], device=device)
                masks = torch.as_tensor(batch['action_masks'], device=device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits, next_hidden = policy(frames, hidden)
                loss = base._masked_cross_entropy(logits, actions, masks)
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
                    deadline_check(deadline)
                    optimizer.step()
                    report['optimizer_updates'] += 1
                    report['supervised_samples'] += len(selection)
                else:
                    report['validation_samples'] += len(selection)
                hidden = detach_state(next_hidden)
                total += float(loss.item()) * len(selection)
                correct += int((logits.detach().masked_fill(~masks, float('-inf')).argmax(dim=1)
                                == actions).sum().item())
                samples += len(selection)
            name = 'training_episodes_completed' if training else 'validation_episodes_completed'
            report[name] += 1
    return {'ce': total / samples, 'accuracy': correct / samples, 'samples': samples}


def _weights_hash(state_dict):
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Policy state must contain named tensors only')
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(str(value.dtype).encode('ascii') + b'\0')
        digest.update(json.dumps(list(value.shape)).encode('ascii') + b'\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(policy, optimizer, path, report, role):
    path = Path(path)
    state = policy.state_dict()
    weights_sha = _weights_hash(state)
    progress = {key: report[key] for key in (
        'status', 'stop_reason', 'epochs_completed', 'current_epoch', 'current_episode_id',
        'current_chunk_start', 'supervised_samples', 'validation_samples', 'optimizer_updates',
        'training_episodes_completed', 'validation_episodes_completed', 'environment_interactions')}
    payload = {
        'format': _FORMAT, 'contract': deepcopy(_CONTRACT), 'policy_state_dict': state,
        'optimizer_state_dict': optimizer.state_dict(), 'policy_sha256': weights_sha,
        'training_config': deepcopy(report['training_config']), 'progress': progress,
    }
    pending = path.with_name(path.stem + '.pending.pt')
    torch.save(payload, pending)
    pending.replace(path)
    digest = base._sha256(path)
    manifest = {
        'format': _FORMAT, 'checkpoint_file': path.name, 'checkpoint_sha256': digest,
        'policy_sha256': weights_sha, 'contract': deepcopy(_CONTRACT),
        'decoder_config': deepcopy(_DECODER), 'role': role, 'warning': base._WARNING,
        'hidden_state_saved': False, 'exact_simulator_resume': False,
        'optimizer': 'Adam', 'progress': progress,
        **{key: deepcopy(report[key]) for key in (
            'sources', 'versions', 'datasets', 'split', 'training_config', 'actual_device')},
    }
    base._json_write(path.with_suffix('.manifest.json'), manifest)
    return {'path': str(path), 'sha256': digest, 'policy_sha256': weights_sha,
            'epoch': report['epochs_completed']}


def _checkpoint_path(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'best.pt' if (path / 'best.pt').is_file() else 'final.pt'
    return path


def load_recurrent_checkpoint(path, device='cpu'):
    """Load this separate strict .pt format, with zero initial inference memory.

    Verifies source, archive, parameter, architecture and dataset identities;
    torch.load uses weights_only=True. Dataset/parent files are not needed at
    inference. Saved optimizer state is available on wrapper.optimizer, but
    neither simulator nor mid-episode training resume is claimed.
    """
    path = _checkpoint_path(path)
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Missing or invalid recurrent checkpoint manifest') from error
    if not isinstance(manifest, dict) or manifest.get('format') != _FORMAT \
            or manifest.get('contract') != _CONTRACT or manifest.get('decoder_config') != _DECODER \
            or manifest.get('sources') != _source_hashes() or manifest.get('checkpoint_file') != path.name \
            or not manifest.get('datasets') or not isinstance(manifest.get('split'), dict) \
            or not isinstance(manifest.get('training_config'), dict) \
            or manifest.get('hidden_state_saved') is not False \
            or manifest.get('exact_simulator_resume') is not False \
            or manifest.get('optimizer') != 'Adam':
        raise ValueError('Invalid recurrent architecture, reset, or source manifest contract')
    base._verify_provenance(manifest)
    if not path.is_file() or base._sha256(path) != manifest.get('checkpoint_sha256'):
        raise ValueError('Recurrent checkpoint SHA256 mismatch')
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload['format'] != _FORMAT or payload['contract'] != _CONTRACT \
                or payload['training_config'] != manifest['training_config'] \
                or payload['progress'] != manifest['progress'] \
                or payload['progress']['environment_interactions'] != 0:
            raise ValueError('Checkpoint payload disagrees with recurrent manifest')
        weights_sha = _weights_hash(payload['policy_state_dict'])
        if weights_sha != manifest.get('policy_sha256') or weights_sha != payload['policy_sha256']:
            raise ValueError('Policy weights SHA256 mismatch')
        policy = RecurrentActor().to(device)
        policy.load_state_dict(payload['policy_state_dict'], strict=True)
        if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
            raise ValueError('Checkpoint policy weights must be finite')
        optimizer = torch.optim.Adam(policy.parameters(), lr=manifest['training_config']['learning_rate'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    except (KeyError, TypeError, RuntimeError, AttributeError) as error:
        raise ValueError('Malformed or incompatible recurrent checkpoint payload') from error
    policy.eval()
    return RecurrentPolicy(policy, optimizer)


def train_recurrent_imitation(dataset_paths, output, *, epochs=10, sequence_length=64,
                              learning_rate=1e-4, seed=0, device='cpu', deadline=None):
    """Train whole episodes with current-only causal memory and chronological TBPTT.

    Only episode order is shuffled. Hidden h/c survives chunks, is detached after
    each optimizer update, and is zeroed for each training/validation episode.
    Class weighting is intentionally disabled for this architecture ablation.
    A fresh output directory is mandatory. All supervised and validation counts
    are offline; environment_interactions is always zero. Absolute monotonic
    deadlines are checked at every chunk and before optimizer updates; final
    checkpoint serialization can outlast the deadline. Partial/error reports
    preserve completed-update counts, not a promise of exact training resume.
    """
    for name, value, minimum in (('epochs', epochs, 1), ('sequence_length', sequence_length, 1),
                                 ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
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
        'format': 'tank-ai-recurrent-imitation-report-v1', 'status': 'running', 'stop_reason': None,
        'warning': base._WARNING, 'training_config': {
            'epochs': int(epochs), 'sequence_length': int(sequence_length), 'learning_rate': learning_rate,
            'seed': int(seed), 'device': str(device), 'deadline_monotonic': deadline,
            'class_weighting': None},
        'epochs_completed': 0, 'current_epoch': 0, 'current_episode_id': None, 'current_chunk_start': None,
        'supervised_samples': 0, 'validation_samples': 0, 'optimizer_updates': 0,
        'training_episodes_completed': 0, 'validation_episodes_completed': 0,
        'environment_interactions': 0, 'datasets': [], 'split': None, 'actual_device': None,
        'initial_validation_ce': None, 'best_validation_ce': None, 'history': [],
        'checkpoints': {'epochs': [], 'best': None, 'final': None},
        'sources': {}, 'versions': {}, 'resource_monitor': {'status': 'not_started'},
    }
    policy, optimizer, corpus = None, None, None
    try:
        base._check_deadline(deadline)
        monitor = base._start_monitor(output, device, report)
        report['sources'] = _source_hashes()
        report['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                              'torch': str(torch.__version__)}
        paths = [Path(path).resolve() for path in paths]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Provide one or more datasets without duplicate paths')
        corpus = base._Corpus(paths, output, deadline)
        report['datasets'] = corpus.identities
        _, _, report['split'] = base._split_episodes(corpus.episode_ids, seed)
        training = _episode_indices(corpus, report['split']['train_episode_ids'])
        validation = _episode_indices(corpus, report['split']['validation_episode_ids'])
        base._check_deadline(deadline)
        torch.manual_seed(int(seed))
        policy = RecurrentActor().to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        report['actual_device'] = str(next(policy.parameters()).device)
        initial = _run_episodes(policy, corpus, validation, sequence_length, deadline, monitor, report)
        report['initial_validation_ce'] = report['best_validation_ce'] = initial['ce']
        report['checkpoints']['best'] = _save_checkpoint(policy, optimizer, output / 'best.pt', report, 'best')
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            report['current_epoch'] = epoch
            ordered = [training[index] for index in rng.permutation(len(training))]
            train_result = _run_episodes(policy, corpus, ordered, sequence_length, deadline,
                                         monitor, report, optimizer=optimizer)
            valid_result = _run_episodes(policy, corpus, validation, sequence_length,
                                         deadline, monitor, report)
            report['epochs_completed'] = epoch
            report['history'].append({
                'epoch': epoch, 'training_ce': train_result['ce'],
                'validation_ce': valid_result['ce'], 'validation_accuracy': valid_result['accuracy'],
                'train_episode_order': [episode_id for episode_id, _ in ordered]})
            report['checkpoints']['epochs'].append(_save_checkpoint(
                policy, optimizer, output / f'epoch-{epoch:04d}.pt', report, 'completed_epoch'))
            if valid_result['ce'] < report['best_validation_ce']:
                report['best_validation_ce'] = valid_result['ce']
                report['checkpoints']['best'] = _save_checkpoint(
                    policy, optimizer, output / 'best.pt', report, 'best')
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
        if policy is not None and optimizer is not None:
            try:
                report['checkpoints']['final'] = _save_checkpoint(
                    policy, optimizer, output / 'final.pt', report, 'final')
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--deadline', type=float, help='Absolute time.monotonic() deadline')
    args = vars(parser.parse_args(argv))
    args['dataset_paths'] = args.pop('datasets')
    report = train_recurrent_imitation(**args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return {'completed': 0, 'failed': 1, 'partial': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
