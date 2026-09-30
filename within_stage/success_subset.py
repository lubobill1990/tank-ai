"""Streaming success-filtered TRAIN-only precision55 copies, with full provenance.

Only published PrecisionDataset corpora are accepted. No source file is edited;
negative outcomes and partial tails remain at their original paths. Filtering
does not make data representative and is not learned-policy success evidence.
This module never collects data, starts an environment, or reads DEV/FINAL cases.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from within_stage import imitation
from within_stage.demonstrations import PrecisionWriter, PrecisionDataset, verify_integrity
from within_stage.precision_policy import file_sha256


ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 128
WARNING = ('Success-filtered TRAIN-only demonstrations; not representative of all visited states. '
           'All source failures/partial tails remain archived unchanged. '
           'Copied teacher/mixed-control clears are not learned-model wins or generalization evidence.')


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\success_subset.py', 'within_stage\\demonstrations.py',
        'within_stage\\imitation.py', 'within_stage\\observation.py', 'native_codec.py')}


def _skip_reason(row, count, completed_steps=None):
    if not count:
        return 'empty_episode'
    if row.get('outcome') != 'clear':
        return 'nonclear_outcome'
    if row.get('episode_finished') is not True:
        return 'incomplete_episode'
    if row.get('truncated', False) or row.get('TimeLimit.truncated', False):
        return 'truncated_episode'
    if row.get('error') is not None or row.get('status') in ('failed', 'partial', 'running'):
        return 'failed_episode'
    if row.get('clear') is not True or row.get('decisions') != count \
            or row.get('completed_decisions') != count:
        return 'unverified_completion'
    if row.get('terminal_remaining') != 0 or row.get('terminal_base_destroyed') is not False:
        return 'unverified_terminal_clear'
    if completed_steps is not None and not completed_steps.all():
        return 'failed_step_in_episode'
    return None


def _verify_copy(source, target, mapping, source_executed, target_executed):
    for offset in range(0, mapping['samples'], BATCH_SIZE):
        size = min(BATCH_SIZE, mapping['samples'] - offset)
        old = np.arange(mapping['source_start'] + offset, mapping['source_start'] + offset + size)
        new = np.arange(mapping['destination_start'] + offset, mapping['destination_start'] + offset + size)
        before, after = source.batch(old), target.batch(new)
        for key in ('observations', 'actions', 'action_masks'):
            if before[key].dtype != after[key].dtype or before[key].tobytes() != after[key].tobytes():
                raise ValueError(f'Copied {key} differs from source')
        old_audit, new_audit = source.audit(old), target.audit(new)
        for key in ('ram', 'frames'):
            if old_audit[key].tobytes() != new_audit[key].tobytes():
                raise ValueError(f'Copied audit {key} differs from source')
        if source_executed[old].tobytes() != target_executed[new].tobytes():
            raise ValueError('Copied executed actions differ from source')
        if not np.all(after['episode_ids'] == mapping['destination_episode_id']):
            raise ValueError('Copied episode boundaries are inconsistent')


def copy_success_episodes(dataset_paths, output, split_path):
    """Copy complete clears only and return verified counts/origins.

    output must not exist. split_path must match the frozen TRAIN split used by
    the existing trainer. Every input plan/report is authenticated using that
    trainer's TrainingCorpus. Read/decode batches never exceed128 samples.
    Raises before output creation if no eligible clear exists, a source is
    unfinished, or any input provenance is outside TRAIN. No empty dataset is
    fabricated. Later copy errors leave an explicit failed report, never success.
    """
    paths = [dataset_paths] if isinstance(dataset_paths, (str, Path)) else list(dataset_paths)
    paths = [Path(path).resolve() for path in paths]
    output, split_path = Path(output).resolve(), Path(split_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not paths or len(paths) != len(set(paths)) or any(output.is_relative_to(path) for path in paths):
        raise ValueError('Use distinct source datasets and a separate fresh output')
    contract = imitation._training_contract()
    if file_sha256(split_path) != contract['frozen_split_sha256']:
        raise ValueError('Supplied split_path differs from the frozen TRAIN split')
    sources = _sources()
    corpus = imitation.TrainingCorpus(paths, contract, None)
    writer = target = None
    report = {'status': 'running', 'role': 'success-filtered TRAIN-only',
              'warning': WARNING, 'environment_interactions': 0, 'source_files_modified': False,
              'source_samples': sum(len(data) for data in corpus.datasets),
              'source_reported_episodes': 0, 'copied_episodes': 0, 'copied_samples': 0,
              'skipped_episode_reasons': {}, 'sources': sources}
    try:
        source_catalog, selected, skipped = [], [], []
        for source_id, (path, dataset) in enumerate(zip(paths, corpus.datasets)):
            if dataset.metadata.get('status') in ('running', 'collecting'):
                raise ValueError('Cannot copy a corpus whose collector is still running')
            identity = deepcopy(corpus.identities[source_id])
            identity.update(source_dataset_id=source_id,
                            metadata_sha256=file_sha256(path / 'metadata.json'),
                            episode_reports_sha256=file_sha256(path / 'episode_reports.json'))
            source_catalog.append(identity)
            reports = json.loads((path / 'episode_reports.json').read_text(encoding='utf-8'))
            report['source_reported_episodes'] += len(reports)
            flags_path = path / 'step_completed.npy'
            flags = np.load(flags_path, mmap_mode='r', allow_pickle=False) if flags_path.is_file() else None
            try:
                if flags is not None:
                    if flags.shape != (len(dataset),) or flags.dtype != np.bool_:
                        raise ValueError('Source step-completion audit has invalid shape/dtype')
                    identity['step_completed_sha256'] = file_sha256(flags_path)
                for row in reports:
                    indices = np.flatnonzero(dataset.episode_ids == row['episode_id'])
                    reason = _skip_reason(row, len(indices), flags[indices] if flags is not None else None)
                    origin = {'source_dataset_id': source_id, 'source_dataset_path': str(path),
                              'source_episode_id': row['episode_id'], 'family_id': row['family_id'],
                              'config_index': row['config_index'], 'stage': row['stage'],
                              'source_sha256': dataset.fingerprint, 'source_samples': len(indices),
                              'outcome': row['outcome']}
                    if reason is not None:
                        skipped.append({**origin, 'reason': reason})
                    else:
                        if not np.all(np.diff(indices) == 1):
                            raise ValueError('A source episode must be contiguous')
                        selected.append((source_id, deepcopy(row), int(indices[0]), len(indices), origin))
            finally:
                if flags is not None:
                    flags._mmap.close()
        if not selected:
            raise ValueError('No complete clear episodes; no empty dataset was created')
        plans, reports, mappings = [], [], []
        destination_start = 0
        for new_id, (source_id, original, start, count, origin) in enumerate(selected):
            origin = {**origin, 'destination_episode_id': new_id,
                      'source_start': start, 'source_stop': start + count,
                      'destination_start': destination_start, 'destination_stop': destination_start + count,
                      'samples': count, 'source_integrity_sha256': source_catalog[source_id]['integrity_sha256'],
                      'source_metadata_sha256': source_catalog[source_id]['metadata_sha256'],
                      'source_episode_reports_sha256': source_catalog[source_id]['episode_reports_sha256']}
            mappings.append(origin)
            plans.append({'episode_id': new_id, 'family_id': original['family_id'],
                          'config_index': original['config_index'],
                          'opening_spec': deepcopy(original['opening_spec'])})
            row = deepcopy(original)
            row.update(episode_id=new_id, origin=origin, learned_model_win=False,
                       success_filtered=True, source_episode_id=original['episode_id'])
            reports.append(row)
            destination_start += count
        metadata = {
            'role': 'success-filtered TRAIN-only precision55 demonstrations',
            'selected_partition': 'train', 'split_sha256': contract['frozen_split_sha256'],
            'plan': plans, 'success_filtered': True, 'representative': False, 'warning': WARNING,
            'source_datasets': source_catalog, 'source_samples': report['source_samples'],
            'subset_utility_sources': sources, 'learned_model_wins': False,
            'new_environment_interactions': 0, 'origin_mapping_file': 'origin_mapping.json',
            'skipped_episodes_file': 'skipped_episodes.json',
            'selection_rule': 'nonempty whole complete clear; error-free, not truncated, '
                              'all steps completed, terminal remaining0 and base intact',
            'status': 'completed',
        }
        writer = PrecisionWriter(output)
        for source_id, dataset in enumerate(corpus.datasets):
            executed = np.load(paths[source_id] / 'executed_actions.npy', mmap_mode='r', allow_pickle=False)
            try:
                for mapping in (item for item in mappings if item['source_dataset_id'] == source_id):
                    for offset in range(0, mapping['samples'], BATCH_SIZE):
                        size = min(BATCH_SIZE, mapping['samples'] - offset)
                        indices = np.arange(mapping['source_start'] + offset,
                                            mapping['source_start'] + offset + size)
                        batch, audit = dataset.batch(indices), dataset.audit(indices)
                        for i, source_index in enumerate(indices):
                            writer.add(batch['observations'][i, 55:], int(batch['actions'][i]),
                                       batch['action_masks'][i], ram=audit['ram'][i],
                                       frame=int(audit['frames'][i]), episode_start=offset == 0 and i == 0,
                                       executed_action=int(executed[source_index]))
            finally:
                executed._mmap.close()
        writer.finish(reports, metadata)
        imitation._write(output / 'origin_mapping.json', mappings)
        imitation._write(output / 'skipped_episodes.json', skipped)
        target = PrecisionDataset(output)
        target_executed = np.load(output / 'executed_actions.npy', mmap_mode='r', allow_pickle=False)
        try:
            for source_id, source in enumerate(corpus.datasets):
                executed = np.load(paths[source_id] / 'executed_actions.npy', mmap_mode='r', allow_pickle=False)
                try:
                    for mapping in (item for item in mappings if item['source_dataset_id'] == source_id):
                        _verify_copy(source, target, mapping, executed, target_executed)
                finally:
                    executed._mmap.close()
        finally:
            target_executed._mmap.close()
        if len(target) != destination_start or target.metadata['episodes'] != len(selected):
            raise ValueError('Copied sample/episode totals differ')
        for source_id, identity in enumerate(source_catalog):
            if verify_integrity(paths[source_id]) != identity['sha256'] \
                    or file_sha256(paths[source_id] / 'integrity.json') != identity['integrity_sha256']:
                raise ValueError('Source dataset changed during copying')
            if 'step_completed_sha256' in identity and file_sha256(paths[source_id] / 'step_completed.npy') \
                    != identity['step_completed_sha256']:
                raise ValueError('Source completion audit changed during copying')
        if _sources() != sources or file_sha256(split_path) != contract['frozen_split_sha256']:
            raise ValueError('Utility sources or frozen split changed during copying')
        report.update(status='completed', copied_episodes=len(selected), copied_samples=len(target),
                      copied_family_count=len({row['family_id'] for row in plans}),
                      skipped_episode_reasons=dict(Counter(row['reason'] for row in skipped)),
                      dataset_path=str(output), fingerprint_sha256=target.fingerprint,
                      bit_exact_frames_history_ram_frame_numbers_masks_actions_executed=True,
                      all_copied_actions_mask_legal=True)
        imitation._write(output / 'report.json', report)
        receipts = [{'path': name, 'sha256': file_sha256(output / name)} for name in (
            'integrity.json', 'origin_mapping.json', 'skipped_episodes.json', 'report.json')]
        imitation._write(output / 'subset-integrity.json', {
            'format': 'within-stage-success-subset-v1', 'files': receipts,
            'sha256': hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()})
        return report
    except Exception as error:
        if output.exists():
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            imitation._write(output / 'report.json', report)
        raise
    finally:
        if target is not None:
            target.close()
        if writer is not None:
            writer.close()
        corpus.close()
