"""Offline SYNTHETIC paired-player geometry queries, not natural trajectories.

Only P1 x/y bytes0x90/0x98 may change. Native one-step tests are separate
feasibility evidence, never global reachability, exact transition, or win proof.
No query is integrated into PrecisionDataset/training by this utility.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

import game_state as gs
from battle_city_env import BattleCityEnv
from heuristic import redundant_boundary_action
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_resources import ResourceMonitor
from within_stage import imitation
from within_stage.observation import encode_precision
from within_stage.precision_policy import file_sha256


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'within-stage-offline-relocation-queries-v1'
EDITABLE_BYTES = (0x90, 0x98)


class RejectedGeometry(ValueError):
    pass


def _sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def current_state(ram, frame):
    if not isinstance(ram, np.ndarray) or ram.dtype != np.uint8 or ram.shape != (2048,) \
            or isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or frame < 0:
        raise ValueError('Expected uint8(2048,) RAM and nonnegative observed frame')
    return replace(gs.StateTracker().reset(ram), frame=int(frame))


def precision_frame(ram, frame):
    state = current_state(ram, frame)
    holder = SimpleNamespace(state=state, max_frames=12000, objective='full', _last_macro=-1)
    return encode_precision(state, BattleCityEnv._observation(holder))


def primitive_masks(state):
    mask = np.ones(10, bool)
    mask[5] = state.fire_ready
    for action in range(1, 10):
        if redundant_boundary_action(state, action):
            mask[action] = False
    return mask


def _body_safe(ram, x, y):
    if not 24 <= x <= 216 or not 24 <= y <= 216:
        raise RejectedGeometry('outside_arena')
    state = gs.parse(ram)
    if not all(gs.can_drive(state.terrain[row, col]) for row, col in gs.tank_cells(x, y)):
        raise RejectedGeometry('blocked_body')
    for slot in range(1, 8):
        if gs._active(ram[0xA0 + slot]) and abs(x - int(ram[0x90 + slot])) < 16 \
                and abs(y - int(ram[0x98 + slot])) < 16:
            raise RejectedGeometry('live_tank_overlap')


def relocate_pair(previous, current, translation):
    """Return owned RAM copies after conservative two-snapshot geometry checks."""
    if not isinstance(translation, (tuple, list)) or len(translation) != 2 \
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or int(v) % 8 for v in translation):
        raise ValueError('Translation must contain two integer multiples of8 pixels')
    states = [current_state(ram, 0) for ram in (previous, current)]
    for ram, state in zip((previous, current), states):
        if state.player is None or state.game_over or state.stage != 1:
            raise RejectedGeometry('inactive_player_or_nonlive_stage1')
        if any(bullet.owner == 0 for bullet in state.bullets):
            raise RejectedGeometry('active_p1_bullet')
    if not np.array_equal(previous[0x400:0x800], current[0x400:0x800]):
        raise RejectedGeometry('changed_terrain')
    result = []
    dx, dy = map(int, translation)
    keep = np.ones(2048, bool); keep[list(EDITABLE_BYTES)] = False
    for ram in (previous, current):
        x, y = int(ram[0x90]) + dx, int(ram[0x98]) + dy
        _body_safe(ram, x, y)
        modified = ram.copy()
        modified[0x90], modified[0x98] = x, y
        if not np.array_equal(modified[keep], ram[keep]):
            raise ValueError('Unsafe field edit outside P1 x/y')
        result.append(modified)
    for address in EDITABLE_BYTES:
        if int(result[1][address]) - int(result[0][address]) != int(current[address]) - int(previous[address]):
            raise ValueError('Translation changed observed player displacement')
    return tuple(result)


def make_query(previous, current, frames, translation):
    if len(frames) != 2 or int(frames[1]) - int(frames[0]) != 4:
        raise RejectedGeometry('nonadjacent_decision_frames')
    translated = relocate_pair(previous, current, translation)
    observations = np.concatenate([precision_frame(ram, frame) for ram, frame in zip(translated, frames)])
    state = current_state(translated[1], frames[1])
    masks = primitive_masks(state)
    teacher = TeacherControllerV2()
    teacher.reset()
    action = int(teacher.act(state, masks.copy()))
    if not 0 <= action < 10 or not masks[action]:
        raise ValueError('Fresh synthetic teacher label violates its legal mask')
    return {'observations': observations, 'action': action, 'action_masks': masks,
            'ram_pair': np.stack(translated), 'translation': tuple(map(int, translation)),
            'source_ram_sha256': [_sha(previous), _sha(current)],
            'synthetic_ram_sha256': [_sha(ram) for ram in translated]}


def generate_pilot(dataset_path, output, *, queries=256, seed=1701, deadline):
    """Generate a bounded standalone npz/JSON artifact, using TRAIN current geometry only."""
    if type(queries) is not int or not 1 <= queries <= 256:
        raise ValueError('Offline feasibility pilot is limited to1..256 queries')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(dataset_path).resolve()
    sources = {name: file_sha256(ROOT / name) for name in (
        'within_stage\\relocation_queries.py', 'within_stage\\observation.py',
        'within_stage\\demonstrations.py', 'within_stage\\imitation.py',
        'learning_lab\\teacher.py', 'learning_lab\\teacher_v2.py',
        'game_state.py', 'heuristic.py', 'battle_city_env.py')}
    report = {
        'format': VERSION, 'role': 'OFFLINE-SYNTHETIC-GEOMETRY-NOT-TRAJECTORY',
        'status': 'running', 'seed': seed, 'requested_queries': queries,
        'teacher_label_calls': 0, 'offline_environment_calls': 0, 'sources': sources,
        'native_validation': None, 'parent_acceptance_required_before_training': True,
        'warning': 'These are synthetic geometry pairs, not normal starts, real trajectories, '
                   'full-game wins, or globally reachable/transition-exact states.',
        'rules': {'editable_bytes': list(EDITABLE_BYTES), 'translation_quantum_pixels': 8,
                  'both_frames_required': True, 'no_active_p1_bullets': True,
                  'raw_terrain_unchanged': True, 'body_size': [16, 16],
                  'passability': 'conservative whole-cell solidity check', 'no_live_tank_overlap': True}}
    imitation._write(output / 'plan.json', report)
    corpus = None
    accepted, origins, rejections = [], [], Counter()
    try:
        monitor = ResourceMonitor(output / 'resources.jsonl', device='cpu')
        monitor.check('offline_geometry_admission', force=True)
        contract = imitation._training_contract()
        corpus = imitation.TrainingCorpus([source], contract, deadline)
        data = corpus.datasets[0]
        report['source_dataset'] = corpus.identities[0]
        report['split_sha256'] = contract['frozen_split_sha256']
        origins_by_episode = {int(row['episode_id'].split(':')[1]): row for row in corpus.membership}
        candidates = np.flatnonzero(data.episode_ids[1:] == data.episode_ids[:-1]) + 1
        rng = np.random.default_rng(seed)
        unique = set()
        for current_index in rng.permutation(candidates):
            if time.monotonic() >= deadline:
                raise TimeoutError('Offline relocation deadline reached')
            monitor.check('offline_geometry_query')
            indices = np.array([int(current_index) - 1, int(current_index)])
            audit = data.audit(indices)
            previous, current = audit['ram']
            frames = audit['frames']
            try:
                relocate_pair(previous, current, (0, 0))
                if int(frames[1]) - int(frames[0]) != 4:
                    raise RejectedGeometry('nonadjacent_decision_frames')
            except RejectedGeometry as error:
                rejections[str(error)] += 1
                continue
            translations = [(dx, dy) for dx in range(-192, 193, 8) for dy in range(-192, 193, 8)
                            if (dx or dy) and all(24 <= int(ram[0x90]) + dx <= 216
                                                and 24 <= int(ram[0x98]) + dy <= 216
                                                for ram in (previous, current))]
            chosen = None
            for translation_index in rng.permutation(len(translations)):
                translation = translations[int(translation_index)]
                try:
                    translated = relocate_pair(previous, current, translation)
                except RejectedGeometry as error:
                    rejections[str(error)] += 1
                    continue
                identity = tuple(_sha(ram) for ram in translated)
                if identity in unique:
                    continue
                chosen = translation
                unique.add(identity)
                break
            if chosen is None:
                rejections['no_supported_translation'] += 1
                continue
            zero = make_query(previous, current, frames, (0, 0)); report['teacher_label_calls'] += 1
            state = current_state(current, frames[1])
            teacher = TeacherControllerV2(); teacher.reset()
            source_label = int(teacher.act(state, primitive_masks(state)))
            report['teacher_label_calls'] += 1
            if zero['action'] != source_label:
                raise ValueError('Translation-zero label differs from independent reset-teacher replay')
            stored = data.batch([int(current_index)])
            if zero['observations'].tobytes() != stored['observations'][0].tobytes():
                raise ValueError('Translation-zero recomputed observation differs from stored precision history')
            query = make_query(previous, current, frames, chosen)
            report['teacher_label_calls'] += 1
            if query['observations'].tobytes() == zero['observations'].tobytes():
                raise ValueError('Nonzero relocation failed to add different geometric input')
            family = origins_by_episode[int(data.episode_ids[current_index])]
            origins.append({
                'query_id': len(accepted), 'source_dataset_path': str(source),
                'source_dataset_sha256': data.fingerprint, 'source_indices': indices.tolist(),
                'source_episode_id': int(data.episode_ids[current_index]),
                'family_id': family['family_id'], 'config_index': family['config_index'],
                'frames': frames.tolist(), 'translation': list(chosen),
                'source_ram_sha256': query['source_ram_sha256'],
                'synthetic_ram_sha256': query['synthetic_ram_sha256'],
                'stored_teacher_action': int(stored['actions'][0]), 'reset_source_teacher_action': source_label,
                'synthetic_teacher_action': query['action'], 'zero_translation_label_replay_exact': True,
                'zero_translation_observation_bit_exact': True, 'field_edits_only_p1_xy': True})
            accepted.append(query)
            if len(accepted) == queries:
                break
        if len(accepted) != queries:
            raise ValueError('Insufficient conservatively valid source pairs/translations; constraints not relaxed')
        np.savez_compressed(output / 'queries.npz',
            observations=np.stack([q['observations'] for q in accepted]),
            actions=np.array([q['action'] for q in accepted], np.int64),
            action_masks=np.stack([q['action_masks'] for q in accepted]),
            synthetic_ram_pairs=np.stack([q['ram_pair'] for q in accepted]),
            source_indices=np.array([row['source_indices'] for row in origins], np.int64),
            source_frames=np.array([row['frames'] for row in origins], np.int64),
            translations=np.array([row['translation'] for row in origins], np.int64))
        report.update(status='completed', queries=len(accepted),
                      label_counts=np.bincount([q['action'] for q in accepted], minlength=10).tolist(),
                      changed_from_reset_source=sum(row['synthetic_teacher_action'] != row['reset_source_teacher_action'] for row in origins),
                      changed_from_stored_source=sum(row['synthetic_teacher_action'] != row['stored_teacher_action'] for row in origins),
                      zero_replay_differs_from_stored_memory_label=sum(row['reset_source_teacher_action'] != row['stored_teacher_action'] for row in origins),
                      distinct_geometry_queries=len(unique), query_file_sha256=file_sha256(output / 'queries.npz'))
    except Exception as error:
        report.update(status='negative-or-incomplete', error=f'{type(error).__name__}: {error}', queries=len(accepted))
    finally:
        if corpus is not None:
            corpus.close()
        report['rejection_counts'] = dict(rejections)
        report['sources_unchanged'] = sources == {name: file_sha256(ROOT / name) for name in sources}
        imitation._write(output / 'origins.json', origins)
        imitation._write(output / 'report.json', report)
    return report


def native_validity_probe(output, *, count=128, deadline):
    """At most128 initializations + one decision each; stop on any invalid support/field."""
    from rust_env import NativeBatch
    output = Path(output)
    if not 1 <= count <= 128:
        raise ValueError('Native validity is limited to128 query states')
    if (output / 'native-validity.json').exists():
        raise FileExistsError('No automatic repetition of native validity probes')
    report = {'status': 'running', 'native_initialization_attempted': 0, 'native_initialization_completed': 0,
              'native_control_attempted': 0, 'native_control_completed': 0, 'rom_calls': 0,
              'claims_reachability': False, 'claims_transition_equivalence': False, 'claims_wins': False}
    batch = None
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError('Native validity deadline')
        with np.load(output / 'queries.npz', allow_pickle=False) as arrays:
            n = min(count, len(arrays['actions']))
            initial = arrays['synthetic_ram_pairs'][:n, 1].copy()
            actions = arrays['actions'][:n].copy()
            expected_masks = arrays['action_masks'][:n].copy()
        report['native_initialization_attempted'] = n
        imitation._write(output / 'native-validity.json', report)
        batch = NativeBatch(initial, max_frames=12000, seeds=[470000 + index for index in range(n)])
        report['native_initialization_completed'] = n
        report['initial_support_codes'] = batch.support.tolist()
        for index in range(n):
            if not np.array_equal(batch.ram(index), initial[index]):
                raise ValueError('Native import changed synthetic RAM fields')
        if not np.array_equal(batch.masks(), expected_masks):
            raise ValueError('Native/canonical legal masks disagree; gate failed')
        if time.monotonic() >= deadline:
            raise TimeoutError('Native validity deadline before step')
        report['native_control_attempted'] = n
        imitation._write(output / 'native-validity.json', report)
        observation, _, done, stats = batch.step(actions)
        report['native_control_completed'] = n
        report['post_step_support_codes'] = batch.support.tolist()
        if not np.isfinite(observation).all() or np.any(batch.support):
            raise ValueError('Native support/nonfinite validity gate failed')
        report.update(status='passed', initial_ram_byte_exact=True, masks_exact=True,
                      finite_observations=True, terminal_after_one_step=int(done.sum()),
                      native_gameplay_frames=int(stats[:, 0].sum()))
    except Exception as error:
        report.update(status='negative', error=f'{type(error).__name__}: {error}')
    finally:
        if batch is not None:
            batch.close()
        report['total_attempted_calls'] = report['native_initialization_attempted'] + report['native_control_attempted']
        report['total_completed_calls'] = report['native_initialization_completed'] + report['native_control_completed']
        report['ledger_settled'] = False
        imitation._write(output / 'native-validity.json', report)
    return report
