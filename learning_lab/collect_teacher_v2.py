"""Frozen NON-LEARNED teacher-v2 stage-1 development collection and success copy.

Only seeds 0/1/2 are used. No model training, GPU work, held-out openings, or
budget-ledger writes occur. Attempts count before teacher.act; completed counts
require a returned env.step. A failed step's pre-action label is retained and
explicitly marked by its aborted episode, never included in the success subset.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from learning_lab.demonstrations import DemonstrationDataset, DemonstrationWriter, _action
from rust_resources import ResourceMonitor


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = 'd21dd493e908469f94ba6c3b2124c021'
GLOBAL_DEADLINE = datetime.fromisoformat('2026-09-30T21:01:44+08:00').timestamp()
WARNING = ('NON-LEARNED teacher; stage-1 development-only seeds 0/1/2. '
           'Repeated clean episodes are duplicates, not independent generalization.')


def _write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError('Collector/copy wall deadline reached')


def _sha(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            _check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _sources():
    names = ('battle_city.nes', 'battle_city_env.py', 'game_state.py', 'heuristic.py',
             'native_codec.py', 'rust_models.py', 'policy.py',
             'learning_lab\\teacher.py', 'learning_lab\\teacher_v2.py',
             'learning_lab\\demonstrations.py', 'learning_lab\\imitation.py',
             'learning_lab\\weighted_imitation.py', 'learning_lab\\recurrent_imitation.py',
             'learning_lab\\collect_teacher_v2.py')
    return {name: _sha(ROOT / name) for name in names}


def episode_config(episode):
    if type(episode) is not int or not 0 <= episode < 24:
        raise ValueError('Episode index must be in 0..23')
    return {'episode_id': episode, 'stage': 1, 'seed': episode % 3,
            'perturbation_probability': 0. if episode < 6 else .02,
            'perturbation_rng_seed': 601 + episode}


def _seal(path, deadline=None):
    entries = [{'path': item.name, 'bytes': item.stat().st_size, 'sha256': _sha(item, deadline)}
               for item in sorted(path.iterdir()) if item.is_file() and item.name != 'integrity.json']
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    _write(path / 'integrity.json', {'format': 'teacher-v2-corpus-integrity-v1',
                                    'files': entries, 'fingerprint_sha256': fingerprint})
    return fingerprint


def verify_integrity(path, deadline=None):
    path = Path(path)
    manifest = json.loads((path / 'integrity.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    actual_names = sorted(p.name for p in path.iterdir() if p.is_file() and p.name != 'integrity.json')
    if manifest.get('format') != 'teacher-v2-corpus-integrity-v1' \
            or sorted(entry['path'] for entry in entries) != actual_names:
        raise ValueError('Corpus integrity file list mismatch')
    for entry in entries:
        if Path(entry['path']).name != entry['path']:
            raise ValueError('Invalid corpus integrity path')
        item = path / entry['path']
        if item.stat().st_size != entry['bytes'] or _sha(item, deadline) != entry['sha256']:
            raise ValueError(f'Corpus SHA256 mismatch: {entry["path"]}')
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest['fingerprint_sha256']:
        raise ValueError('Corpus fingerprint mismatch')
    return fingerprint


def _verify_dataset(path, deadline=None, monitor=None):
    dataset = DemonstrationDataset(path)
    try:
        digest = hashlib.sha256()
        for first in range(0, len(dataset), 32):
            _check_deadline(deadline)
            if monitor is not None:
                monitor.check('dataset_verification')
            indices = np.arange(first, min(first + 32, len(dataset)))
            batch = dataset.batch(indices)
            digest.update(batch['observations'][:, 47:].tobytes(order='C'))
        if digest.hexdigest() != dataset.metadata['raw_frame_sha256']:
            raise ValueError('Decoded raw frames do not match writer SHA256')
        result = {'samples': len(dataset), 'episodes': dataset.metadata['episodes'],
                  'raw_frame_sha256': digest.hexdigest(), 'all_frames_verified': True,
                  'teacher_and_executed_actions_mask_valid': True,
                  'new_environment_decisions': 0}
        _write(Path(path) / 'verification.json', result)
        return result
    finally:
        dataset.close()


def collect(path, *, max_episodes=24, max_decisions=48000, deadline,
            env_factory=None, teacher_factory=None):
    """Collect every scheduled outcome; optional factories are synthetic test boundaries."""
    if type(max_episodes) is not int or not 1 <= max_episodes <= 24 \
            or type(max_decisions) is not int or not 1 <= max_decisions <= 48000:
        raise ValueError('Bounds are 1..24 episodes and 1..48000 attempted decisions')
    if (env_factory is None) != (teacher_factory is None):
        raise ValueError('Supply both synthetic factories or neither')
    synthetic = env_factory is not None
    started, started_monotonic = time.time(), time.monotonic()
    deadline = min(float(deadline), GLOBAL_DEADLINE, started + 45 * 60)
    # Reserve a little of the bounded wall window for finishing/verification.
    collection_deadline = deadline - min(60., max(0., deadline - started) * .05)
    monotonic_deadline = started_monotonic + max(0., collection_deadline - started)
    writer = DemonstrationWriter(Path(path).resolve())
    monitor = ResourceMonitor(writer.path / 'resources.jsonl', device='cpu')
    sources = _sources()
    metadata = {
        'label': 'NON-LEARNED TeacherControllerV2 demonstrations; no model trained',
        'teacher_class': 'learning_lab.teacher_v2.TeacherControllerV2',
        'learned_model': False, 'development_only': True, 'warning': WARNING,
        'reservation': None if synthetic else RESERVATION, 'synthetic_fixture': synthetic,
        'stage': 1, 'opening_seeds': [0, 1, 2], 'clean_episodes_planned': min(6, max_episodes),
        'max_episodes': max_episodes, 'max_decisions': max_decisions,
        'max_frames_per_episode': 12000, 'frame_skip': 4,
        'perturbation_probability_after_six': .02,
        'perturbation_rng': 'numpy.default_rng(601 + episode_id)',
        'perturbation_distribution': 'uniform over every legal action, including teacher action',
        'label_timing': 'Owned observation and teacher label BEFORE executed env.step',
        'attempt_definition': 'Increment before every teacher.act, including failed attempts',
        'completed_definition': 'Increment only after env.step returns successfully',
        'failed_step_note': 'Last stored action of an aborted episode may not have completed execution',
        'provenance': 'Normal original-ROM reset; no RAM edits, rollouts, or RNG oracle',
        'source_sha256': sources, 'started_at': started, 'deadline': deadline,
        'collection_deadline': collection_deadline, 'native_environment_decisions': 0,
        'real_test_probe_decisions': 0,
    }
    _write(writer.path / 'collection_config.json', metadata)
    reports, attempted, completed, frames = [], 0, 0, 0
    env, failure, stop_reason = None, None, 'episode_cap'
    try:
        if time.time() >= collection_deadline:
            stop_reason = 'deadline'
        else:
            monitor.check('admission', admission_slots=1, force=True)
            if not synthetic:
                from battle_city_env import BattleCityEnv
                from learning_lab.teacher_v2 import TeacherControllerV2
                env_factory = lambda: BattleCityEnv(stage=1, objective='full',
                                                    max_frames=12000, frame_skip=4)
                teacher_factory = TeacherControllerV2
            env, teacher = env_factory(), teacher_factory()
            for episode in range(max_episodes):
                if attempted >= max_decisions:
                    stop_reason = 'decision_budget'
                    break
                if time.time() >= collection_deadline or time.monotonic() >= monotonic_deadline:
                    stop_reason = 'deadline'
                    break
                config = episode_config(episode)
                rng = np.random.default_rng(config['perturbation_rng_seed'])
                ep_attempts = ep_completed = stored = perturbations = changed = 0
                before_frames, episode_started = 0, time.monotonic()
                info = {'frames': 0, 'kills': 0, 'deaths': 0, 'outcome': ''}
                outcome, episode_error, finished = '', None, False
                terminal_remaining, terminal_base = None, None
                try:
                    teacher.reset()
                    observation, info = env.reset(seed=config['seed'])
                    while not outcome:
                        monitor.check('collection')
                        if attempted >= max_decisions:
                            outcome = stop_reason = 'decision_budget'
                            break
                        if time.time() >= collection_deadline or time.monotonic() >= monotonic_deadline:
                            outcome = stop_reason = 'deadline'
                            break
                        masks = np.asarray(env.action_masks()[:10]).copy()
                        raw = np.array(observation, copy=True)
                        attempted += 1
                        ep_attempts += 1
                        label = _action(teacher.act(env.state, masks))
                        executed = label
                        if config['perturbation_probability'] and rng.random() < .02:
                            perturbations += 1
                            executed = int(rng.choice(np.flatnonzero(masks)))
                        changed += int(executed != label)
                        writer.add(raw, label, masks, episode_start=stored == 0, executed_action=executed)
                        stored += 1
                        observation, _, terminated, truncated, info = env.step(executed)
                        completed += 1
                        ep_completed += 1
                        frames += int(info['frames']) - before_frames
                        before_frames = int(info['frames'])
                        outcome = info['outcome']
                        if bool(terminated or truncated) != bool(outcome) \
                                or outcome not in ('', 'clear', 'base_lost', 'lives_exhausted', 'timeout'):
                            raise ValueError('Unexpected full-game terminal contract')
                        if outcome:
                            finished = True
                            terminal_remaining = int(env.state.remaining_enemies)
                            terminal_base = bool(env.state.base_destroyed)
                            if outcome == 'clear' and (terminal_remaining != 0 or terminal_base):
                                raise ValueError('Clear outcome lacks terminal ROM-state confirmation')
                        if attempted % 128 == 0:
                            _write(writer.path / 'progress.json', {
                                'attempted_decisions': attempted, 'completed_decisions': completed,
                                'episode_id': episode, 'stored_samples': len(writer.actions)})
                except Exception as error:
                    episode_error = f'{type(error).__name__}: {error}'
                    outcome, finished, stop_reason = 'aborted', False, 'error'
                row = {
                    **config, 'outcome': outcome, 'clear': outcome == 'clear',
                    'episode_finished': finished, 'attempted_decisions': ep_attempts,
                    'completed_decisions': ep_completed, 'decisions': stored,
                    'kills': int(info.get('kills', 0)), 'deaths': int(info.get('deaths', 0)),
                    'frames': int(info.get('frames', 0)), 'perturbation_attempts': perturbations,
                    'changed_actions': changed, 'error': episode_error,
                    'failure_cause': episode_error or (None if outcome == 'clear' else outcome),
                    'terminal_remaining_enemies': terminal_remaining,
                    'terminal_base_destroyed': terminal_base,
                    'elapsed_seconds': time.monotonic() - episode_started,
                    'cumulative_attempted_decisions': attempted, 'cumulative_completed_decisions': completed,
                }
                reports.append(row)
                with (writer.path / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                if episode_error:
                    failure = episode_error
                    break
                if outcome in ('decision_budget', 'deadline'):
                    break
    except Exception as error:
        failure, stop_reason = f'{type(error).__name__}: {error}', 'error'
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                failure, stop_reason = f'Environment close failed: {error}', 'error'
        if _sources() != sources:
            failure, stop_reason = 'Frozen source hashes changed during collection', 'error'
        summary = {
            'status': 'failed' if failure else 'completed' if stop_reason == 'episode_cap' else 'partial',
            'stop_reason': stop_reason, 'error': failure,
            'attempted_decisions': attempted, 'completed_decisions': completed,
            'stored_samples': len(writer.actions), 'reported_episodes': len(reports),
            'clear_episodes': sum(row['clear'] for row in reports),
            'outcomes': dict(Counter(row['outcome'] for row in reports)),
            'gameplay_frames': frames, 'elapsed_seconds': time.monotonic() - started_monotonic,
            'rom_environment_decisions': 0 if synthetic else attempted,
            'native_environment_decisions': 0, 'real_test_probe_decisions': 0,
            'synthetic_controller_attempts': attempted if synthetic else 0,
            'changed_actions': sum(row['changed_actions'] for row in reports),
            'source_hashes_unchanged': _sources() == sources,
        }
        metadata.update(summary)
        writer.finish(reports, metadata)
        _write(writer.path / 'summary.json', summary)
        _write(writer.path / 'progress.json', {**summary, 'finished': True})
    # Expired-before-start collections still publish a valid, sealed empty corpus.
    verification_deadline = deadline if len(writer.actions) else None
    _verify_dataset(writer.path, verification_deadline, monitor)
    monitor.check('collection_complete', force=True)
    fingerprint = _seal(writer.path, verification_deadline)
    return {**summary, 'path': str(writer.path), 'fingerprint_sha256': fingerprint}


def copy_success_episodes(source, output, *, deadline):
    """Copy complete verified clears, without any new environment/controller calls."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source):
        raise ValueError('Success copy must be outside the frozen source corpus')
    fingerprint = verify_integrity(source, deadline)
    reports = json.loads((source / 'episode_reports.json').read_text(encoding='utf-8'))
    dataset = DemonstrationDataset(source)
    writer = None
    try:
        selected = []
        for row in reports:
            if row['outcome'] != 'clear':
                continue
            indices = np.flatnonzero(dataset.episode_ids == row['episode_id'])
            if not row['clear'] or not row['episode_finished'] or row['error'] is not None \
                    or not len(indices) or len(indices) != row['decisions'] \
                    or row['completed_decisions'] != len(indices) \
                    or row['attempted_decisions'] != len(indices) \
                    or row['terminal_remaining_enemies'] != 0 or row['terminal_base_destroyed'] is not False:
                raise ValueError('Only complete terminal-state-confirmed clear episodes may be copied')
            selected.append((row, indices))
        writer = DemonstrationWriter(output)
        monitor = ResourceMonitor(output / 'resources.jsonl', device='cpu')
        monitor.check('copy_start', force=True)
        executed = np.load(source / 'executed_actions.npy', allow_pickle=False)
        origins, mapping, copied_reports = [], [], []
        for new_id, (row, indices) in enumerate(selected):
            copied = deepcopy(row)
            copied['origin_episode_id'], copied['episode_id'] = row['episode_id'], new_id
            copied_reports.append(copied)
            mapping.append({'destination_episode_id': new_id, 'origin_episode_id': row['episode_id'],
                            'source_start': int(indices[0]), 'source_stop': int(indices[-1]) + 1,
                            'samples': len(indices)})
            for first in range(0, len(indices), 32):
                _check_deadline(deadline)
                monitor.check('copy')
                selection = indices[first:first + 32]
                batch = dataset.batch(selection)
                for position, source_index in enumerate(selection):
                    writer.add(batch['observations'][position, 47:], int(batch['actions'][position]),
                               batch['action_masks'][position],
                               episode_start=first == 0 and position == 0,
                               executed_action=int(executed[source_index]))
                origins.extend(map(int, selection))
        metadata = {
            'label': 'SUCCESS-ONLY WHOLE-EPISODE NON-LEARNED TeacherControllerV2 demonstrations',
            'learned_model': False, 'development_only': True, 'success_filtered': True,
            'warning': 'Success-filtered development-only data; not independent generalization. '
                       'All failures remain in the original full corpus.',
            'source_dataset': str(source), 'source_fingerprint_sha256': fingerprint,
            'source_integrity_sha256': _sha(source / 'integrity.json'),
            'source_metadata': dataset.metadata, 'source_sha256': dataset.metadata['source_sha256'],
            'source_attempted_decisions': dataset.metadata['attempted_decisions'],
            'source_completed_decisions': dataset.metadata['completed_decisions'],
            'new_environment_decisions': 0, 'origin_mapping': mapping,
        }
        writer.finish(copied_reports, metadata)
        np.save(output / 'origin_indices.npy', np.asarray(origins, np.int64), allow_pickle=False)
        _write(output / 'origin_mapping.json', mapping)
        verification = _verify_dataset(output, deadline, monitor)
        target = DemonstrationDataset(output)
        try:
            target_executed = np.load(output / 'executed_actions.npy', allow_pickle=False)
            for first in range(0, len(target), 32):
                _check_deadline(deadline)
                monitor.check('copy_bit_exact_verification')
                positions = np.arange(first, min(first + 32, len(target)))
                source_indices = np.asarray(origins, np.int64)[positions]
                original, copied = dataset.batch(source_indices), target.batch(positions)
                for name in ('observations', 'actions', 'action_masks'):
                    if original[name].tobytes() != copied[name].tobytes():
                        raise ValueError(f'Success copy changed {name}')
                if not np.array_equal(executed[source_indices], target_executed[positions]):
                    raise ValueError('Success copy changed executed actions')
        finally:
            target.close()
        verification['complete_history_labels_masks_executed_actions_bit_exact'] = True
        _write(output / 'verification.json', verification)
        summary = {'copied_episodes': len(selected), 'copied_samples': len(origins),
                   'new_environment_decisions': 0, 'source_fingerprint_sha256': fingerprint,
                   'success_filtered': True, 'development_only': True, 'warning': metadata['warning']}
        _write(output / 'summary.json', summary)
        monitor.check('copy_complete', force=True)
        copied_fingerprint = _seal(output, deadline)
        return {**summary, 'path': str(output), 'fingerprint_sha256': copied_fingerprint}
    finally:
        dataset.close()
        if writer is not None:
            writer.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / '20260930-victory-demonstrations-v2')
    parser.add_argument('--success-output', type=Path,
                        default=ROOT / 'runs' / '20260930-victory-demonstrations-v2-success')
    parser.add_argument('--max-episodes', type=int, default=24)
    parser.add_argument('--max-decisions', type=int, default=48000)
    parser.add_argument('--deadline', default='2026-09-30T21:01:44+08:00')
    args = parser.parse_args(argv)
    if args.output.exists() or args.success_output.exists():
        raise FileExistsError('Both collection output directories must be fresh')
    deadline = min(datetime.fromisoformat(args.deadline).timestamp(),
                   GLOBAL_DEADLINE, time.time() + 45 * 60)
    full = collect(args.output, max_episodes=args.max_episodes,
                   max_decisions=args.max_decisions, deadline=deadline)
    success = copy_success_episodes(args.output, args.success_output, deadline=deadline)
    print(json.dumps({'reservation': RESERVATION, 'ledger_settled': False,
                      'full': full, 'success': success}), flush=True)
    return 1 if full['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
