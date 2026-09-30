"""TRAIN-only mixed-control correction data, never on-policy learner victories.

First40 episodes cover all40 TRAIN families in seeded shuffled order using
their representative aliases. Later40-episode cycles reshuffle families and
choose a uniform alias within each family. The schedule is fixed before play.
Teacher labels are always collected BEFORE stepping. Teacher/student selection
and optional movement-only actuator noise use separate episode-local RNGs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from learning_lab.teacher_v2 import TeacherControllerV2
from rust_resources import ResourceMonitor, ResourcePressureError
from rust_train import _policy_hash
from rust_validation import _evaluation_state, _primitive
from within_stage.demonstrations import PrecisionWriter, PrecisionDataset, SPLIT_SHA256
from within_stage.imitation import adapt_observations
from within_stage.observation import OBSERVATION_VERSION, PrecisionObservation
from within_stage.opening_pool import make_opening_env
from within_stage.openings import ProbeStop
from within_stage.precision_policy import file_sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT_SHA256 = SPLIT_SHA256
VERSION = 'within-stage-recovery-data-v1'
EXTRA_ARRAYS = ('student_proposals', 'selected_actions', 'selected_sources', 'step_completed')


def _write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _sources():
    return {name: file_sha256(ROOT / name) for name in (
        'within_stage\\recovery_data.py', 'within_stage\\demonstrations.py',
        'within_stage\\observation.py', 'within_stage\\precision_policy.py',
        'within_stage\\imitation.py', 'within_stage\\opening_pool.py',
        'within_stage\\openings.py', 'within_stage\\early_openings.py',
        'learning_lab\\teacher.py', 'learning_lab\\teacher_v2.py',
        'battle_city_env.py', 'game_state.py', 'heuristic.py', 'battle_city.nes', 'native_codec.py')}


def _seed(seed, domain, episode):
    text = f'{VERSION}:{seed}:{domain}:{episode}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little')


def training_plan(split_path, *, episodes=40, seed=0):
    """Authenticate TRAIN membership; never use other partitions or outcomes."""
    if type(episodes) is not int or not 1 <= episodes <= 1000 \
            or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Use1..1000 episodes and a uint64 schedule seed')
    split_path = Path(split_path).resolve()
    if file_sha256(split_path) != FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen TRAIN split SHA256 mismatch')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    if any(split.get(key) != value for key, value in (
        ('stage', 1), ('post_start_frames', 150), ('channels_per_frame', 55),
        ('actions', 10), ('n_stack', 2), ('observation_version', OBSERVATION_VERSION))):
        raise ValueError('Frozen TRAIN precision55/handoff contract differs')
    pool_path = Path(split['pool_path']).resolve()
    if file_sha256(pool_path) != split['pool_sha256']:
        raise ValueError('Frozen opening pool hash mismatch')
    pool = json.loads(pool_path.read_text(encoding='utf-8'))
    train = split['train']
    families, representatives = train['family_ids'], train['representative_config_indices']
    configs = train['all_config_indices']
    if len(families) != 40 or len(set(families)) != 40 or len(representatives) != 40 \
            or len(set(representatives)) != 40 or len(set(configs)) != len(configs):
        raise ValueError('TRAIN must contain40 distinct families/representatives')
    allowed = {row['config_index']: row for row in pool['specs'] if row['config_index'] in configs}
    if set(allowed) != set(configs) or {row['family_id'] for row in allowed.values()} != set(families):
        raise ValueError('TRAIN alias/config membership is inconsistent')
    representative = dict(zip(families, representatives))
    for family, config in representative.items():
        if config not in allowed or allowed[config]['family_id'] != family:
            raise ValueError('TRAIN representative does not belong to its declared family')
    aliases = {family: sorted(config for config, row in allowed.items() if row['family_id'] == family)
               for family in families}
    plan = []
    for cycle in range((episodes + 39) // 40):
        ordering = np.random.default_rng(_seed(seed, 'family-order', cycle)).permutation(40)
        for position in ordering:
            if len(plan) == episodes:
                break
            episode = len(plan)
            family = families[int(position)]
            config = (representative[family] if cycle == 0 else int(np.random.default_rng(
                _seed(seed, 'alias', episode)).choice(aliases[family])))
            plan.append({
                'episode_id': episode, 'config_index': config, 'family_id': family,
                'opening_spec': deepcopy(allowed[config]['spec']),
                'alias_policy': 'representative-first-cycle' if cycle == 0 else 'uniform-family-alias',
                'beta_rng_seed': _seed(seed, 'teacher-selection', episode),
                'noise_rng_seed': _seed(seed, 'noise-request', episode),
                'replacement_rng_seed': _seed(seed, 'noise-action', episode)})
    return pool, plan


def _make_precision_env(pool, config_index, account, deadline):
    return PrecisionObservation(make_opening_env(pool, config_index, account=account, max_frames=12000))


def _guard(account, deadline, monitor=None):
    if time.monotonic() >= deadline:
        raise ProbeStop('Recovery collection deadline reached')
    account.check()
    if account.attempted >= account.limit:
        raise ProbeStop('Recovery collection account exhausted')
    if monitor is not None:
        monitor.check('recovery_collection')


class _DeadlineAccount:
    """Forward accounting while bounding every initializer substep locally."""
    def __init__(self, account, deadline):
        object.__setattr__(self, '_account', account)
        object.__setattr__(self, '_deadline', deadline)

    def __getattr__(self, name):
        return getattr(self._account, name)

    def __setattr__(self, name, value):
        setattr(self._account, name, value)

    def check(self):
        if time.monotonic() >= self._deadline:
            raise ProbeStop('Recovery collection deadline reached during initialization/step')
        self._account.check()

    def before(self, kind):
        self.check()
        self._account.before(kind)

    def after(self, kind):
        self._account.after(kind)


def _student_record(model):
    if model is None:
        return {'present': False, 'policy_sha256': None, 'offsets_enabled': True}
    if model.observation_space.shape != (110, 26, 26) or model.action_space.n != 10 \
            or not isinstance(model.policy, torch.nn.Module) or not callable(model.predict):
        raise ValueError('Student must be a precision110 ten-primitive torch policy')
    if torch.device(model.device).type != 'cpu':
        raise ValueError('Recovery collection requires a CPU-loaded student; no GPU work is launched here')
    config = getattr(model, 'precision_dataset_config', {})
    offsets = getattr(model, 'offsets_enabled', config.get('offsets_enabled', True))
    if type(offsets) is not bool:
        raise ValueError('Student offsets_enabled must be explicit boolean')
    return {'present': True, 'policy_sha256': _policy_hash(model.policy),
            'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
            'offsets_enabled': offsets, 'deterministic': True,
            'precision_migration_record': deepcopy(getattr(model, 'precision_migration_record', None)),
            'precision_dataset_config': deepcopy(config)}


def _owned_observation(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 \
            or value.shape != (55, 26, 26) or not np.isfinite(value).all():
        raise ValueError('Collector requires finite float32 precision55 observations')
    return value.copy()


def _seal_extras(path):
    names = ['integrity.json', 'decisions.jsonl', 'collection_config.json', 'report.json',
             *(name + '.npy' for name in EXTRA_ARRAYS)]
    rows = [{'path': name, 'bytes': (path / name).stat().st_size, 'sha256': file_sha256(path / name)}
            for name in sorted(names)]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    _write(path / 'recovery-integrity.json', {'format': VERSION, 'files': rows, 'sha256': digest})


def verify_recovery_integrity(path):
    path = Path(path)
    from within_stage.demonstrations import verify_integrity
    core = verify_integrity(path)
    record = json.loads((path / 'recovery-integrity.json').read_text(encoding='utf-8'))
    if record['format'] != VERSION or record['sha256'] != hashlib.sha256(
            json.dumps(record['files'], sort_keys=True).encode()).hexdigest():
        raise ValueError('Recovery audit manifest mismatch')
    for row in record['files']:
        if Path(row['path']).name != row['path'] or (path / row['path']).stat().st_size != row['bytes'] \
                or file_sha256(path / row['path']) != row['sha256']:
            raise ValueError('Recovery execution-audit file mismatch')
    return {'precision_fingerprint': core, 'execution_audit_sha256': record['sha256']}


def collect_recovery_data(model, output, split_path, *, account, episodes=40,
                           teacher_probability=.8, noise_probability=.005, seed=0, deadline):
    """Collect corrections on mixed-policy states without training or settling budgets.

    account is an existing ProbeAccounting-compatible allowance. The opening
    environment owns its before/after(raw/agent) counters; this function never
    double-charges or extends that allowance. Teacher/student inference calls
    are logged separately from actual initialization/step interactions.
    All model predictions are deterministic CPU inference. model=None explicitly
    selects the teacher every time. Noise uniformly chooses a DIFFERENT legal
    movement/no-op action0..4, after beta-selection, without retries.
    """
    for name, value in (('teacher_probability', teacher_probability), ('noise_probability', noise_probability)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) \
                or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite in[0,1]')
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not np.isfinite(deadline):
        raise ValueError('deadline must be a finite absolute monotonic time')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {
        'format': VERSION, 'status': 'running', 'stop_reason': None, 'error': None,
        'learner_wins': 0, 'mixed_trajectory_clears': 0, 'teacher_only_trajectory_clears': 0,
        'student_present': model is not None, 'teacher_attempts': 0, 'student_attempts': 0,
        'samples': 0, 'episode_reports': [], 'plan': [],
        'teacher_probability': teacher_probability, 'noise_probability': noise_probability,
        'seed': seed, 'account_before': account.snapshot(), 'ledger_settled': False,
        'warning': 'TRAIN-only teacher corrections. Mixed/noisy trajectory clears are NOT learned-model wins. '
                   'Teacher labels are supervised corrections, not on-policy learner actions.'}
    writer = monitor = None
    student, pinned_sources = None, None
    audit_arrays = {key: [] for key in EXTRA_ARRAYS}
    try:
        _guard(account, deadline)
        pool, plan = training_plan(split_path, episodes=episodes, seed=seed)
        student = _student_record(model)
        pinned_sources = _sources()
        report.update(plan=plan, student=student, sources=pinned_sources,
                      split_sha256=file_sha256(split_path),
                      split_path=str(Path(split_path).resolve()))
        writer = PrecisionWriter(output)
        metadata = {
            'role': 'TRAIN-only mixed-control TeacherControllerV2 precision corrections',
            'selected_partition': 'train', 'split_sha256': report['split_sha256'], 'plan': plan,
            'representative_order': 'shuffled-family coverage; first40 representatives, later uniform aliases',
            'teacher_class': 'learning_lab.teacher_v2.TeacherControllerV2', 'student': student,
            'teacher_probability': teacher_probability, 'noise_probability': noise_probability,
            'effective_teacher_probability': 1. if model is None else teacher_probability,
            'movement_noise_actions': [0, 1, 2, 3, 4], 'collector_sources': pinned_sources,
            'ram_writes': False, 'learned_model_wins': False, 'max_frames': 12000, 'frame_skip': 4,
            'development_or_final_evaluated': False, 'final_future_traces_read': False,
            'warning': report['warning'], 'label_timing': 'All observations/RAM/teacher/student labels BEFORE env.step',
            'rng_domains': ['family-order', 'alias', 'teacher-selection', 'noise-request', 'noise-action']}
        _write(output / 'collection_config.json', metadata)
        (output / 'decisions.jsonl').touch(exist_ok=False)
        monitor = ResourceMonitor(output / 'resources.jsonl', device='cpu')
        monitor.check('recovery_admission', force=True)
        bounded_account = _DeadlineAccount(account, deadline)
        with (output / 'decisions.jsonl').open('a', encoding='utf-8') as trace, _evaluation_state(model):
            for selected in plan:
                _guard(account, deadline, monitor)
                env = None
                before = account.snapshot()
                stored = completed = requested = applied = teacher_selected = student_selected = 0
                info = {'frames': 0, 'kills': 0, 'deaths': 0, 'outcome': ''}
                finished, error, stop = False, None, False
                beta_rng = np.random.default_rng(selected['beta_rng_seed'])
                noise_rng = np.random.default_rng(selected['noise_rng_seed'])
                action_rng = np.random.default_rng(selected['replacement_rng_seed'])
                try:
                    env = _make_precision_env(pool, selected['config_index'], bounded_account, deadline)
                    observation, _ = env.reset(seed=0)
                    observation = _owned_observation(observation)
                    history = np.zeros((110, 26, 26), np.float32)
                    history[55:] = observation
                    teacher = TeacherControllerV2()
                    teacher.reset()
                    if model is not None and callable(getattr(model, 'reset', None)):
                        model.reset()
                    while not finished:
                        _guard(account, deadline, monitor)
                        masks = np.asarray(env.get_wrapper_attr('action_masks')()).copy()
                        if masks.shape != (10,) or masks.dtype != np.bool_ or not masks.any():
                            raise ValueError('Expected nonempty ten-primitive boolean masks')
                        raw_ram = env.unwrapped.nes.ram.copy()
                        frame = int(env.unwrapped.state.frame)
                        report['teacher_attempts'] += 1
                        label = _primitive(teacher.act(env.unwrapped.state, masks.copy()))
                        if not masks[label]:
                            raise ValueError('Teacher selected a mask-invalid label')
                        student_action = -1
                        if model is not None:
                            report['student_attempts'] += 1
                            student_obs = adapt_observations(
                                history[None], offsets_enabled=student['offsets_enabled'])[0]
                            student_action = _primitive(model.predict(
                                student_obs, deterministic=True, action_masks=masks.copy())[0])
                            if not masks[student_action]:
                                raise ValueError('Student predicted an illegal primitive')
                        beta_draw, noise_draw = float(beta_rng.random()), float(noise_rng.random())
                        use_teacher = model is None or beta_draw < teacher_probability
                        chosen = label if use_teacher else student_action
                        source = 'teacher' if use_teacher else 'student'
                        alternatives = np.flatnonzero(masks[:5] & (np.arange(5) != chosen))
                        noise_requested = noise_draw < noise_probability
                        executed = int(action_rng.choice(alternatives)) if noise_requested and len(alternatives) else chosen
                        changed = executed != chosen
                        event = {
                            'episode_id': selected['episode_id'], 'decision_index': stored, 'frame': frame,
                            'teacher_action': label, 'student_proposed_action': student_action,
                            'selected_action': chosen, 'selected_source': source, 'executed_action': executed,
                            'beta_draw': beta_draw, 'noise_draw': noise_draw, 'noise_requested': noise_requested,
                            'noise_applied': False, 'noise_alternate_chosen': changed,
                            'noise_miss_reason': 'no_legal_movement_alternative' if noise_requested and not changed else None,
                            'legal_mask': masks.tolist(), 'legal_noise_alternatives': alternatives.tolist(),
                            'step_completed': False, 'immediate_terminal_after_noise': False}
                        _guard(account, deadline, monitor)
                        writer.add(observation, label, masks, ram=raw_ram, frame=frame,
                                   episode_start=stored == 0, executed_action=executed)
                        stored += 1
                        requested += int(noise_requested)
                        teacher_selected += int(use_teacher); student_selected += int(not use_teacher)
                        try:
                            next_observation, _, terminated, truncated, info = env.step(executed)
                            completed += 1
                            event['step_completed'] = True
                            event['noise_applied'] = changed
                            applied += int(changed)
                            finished = bool(terminated or truncated)
                            event['immediate_terminal_after_noise'] = finished and changed
                            event['outcome'] = info['outcome']
                            observation = _owned_observation(next_observation)
                            history[:55] = history[55:].copy()
                            history[55:] = observation
                        finally:
                            trace.write(json.dumps(event, allow_nan=False) + '\n')
                            audit_arrays['student_proposals'].append(student_action)
                            audit_arrays['selected_actions'].append(chosen)
                            audit_arrays['selected_sources'].append(source)
                            audit_arrays['step_completed'].append(event['step_completed'])
                        if stored % 128 == 0:
                            trace.flush()
                            _write(output / 'progress.json', {'episode_id': selected['episode_id'],
                                                              'samples': len(writer.actions),
                                                              'accounting': account.snapshot()})
                except (ProbeStop, ResourcePressureError) as failure:
                    info['outcome'], error, stop = 'budget_or_deadline', str(failure), True
                    report.update(status='partial', stop_reason='budget_or_deadline', error=error)
                except Exception as failure:
                    info['outcome'], error, stop = 'aborted', f'{type(failure).__name__}: {failure}', True
                    report.update(status='failed', stop_reason='error', error=error)
                finally:
                    terminal_remaining = getattr(env.unwrapped.state, 'remaining_enemies', None) if env else None
                    terminal_base = getattr(env.unwrapped.state, 'base_destroyed', None) if env else None
                    clear = bool(finished and info['outcome'] == 'clear')
                    if clear and (terminal_remaining != 0 or terminal_base):
                        clear = False; stop = True
                        report.update(status='failed', stop_reason='unverified_clear', error='Clear lacks terminal-state proof')
                    row = {
                        **selected, 'stage': 1, 'outcome': info['outcome'], 'clear': clear,
                        'episode_finished': finished, 'decisions': stored, 'completed_decisions': completed,
                        'frames': int(info['frames']), 'kills': int(info['kills']), 'deaths': int(info['deaths']),
                        'error': error, 'teacher_selected': teacher_selected, 'student_selected': student_selected,
                        'noise_requested': requested, 'noise_applied': applied, 'learned_model_win': False,
                        'trajectory_kind': 'teacher-only' if model is None and applied == 0 else 'mixed-or-noisy',
                        'terminal_remaining': terminal_remaining, 'terminal_base_destroyed': terminal_base,
                        'initialization_calls': account.raw_initialization_attempted - before['raw_initialization_attempted'],
                        'control_calls': account.agent_attempted - before['agent_attempted']}
                    report['episode_reports'].append(row)
                    if clear:
                        key = 'teacher_only_trajectory_clears' if model is None and applied == 0 else 'mixed_trajectory_clears'
                        report[key] += 1
                    if env is not None:
                        try:
                            env.close()
                        except Exception as failure:
                            error = f'Environment cleanup failed: {failure}'
                            row['error'] = error
                            report.update(status='failed', stop_reason='cleanup_error', error=error)
                            stop = True
                if model is not None and _policy_hash(model.policy) != student['policy_sha256']:
                    report.update(status='failed', stop_reason='parameter_change', error='Student parameters changed')
                    stop = True
                if stop:
                    break
            else:
                report.update(status='completed', stop_reason='episode_plan_complete')
    except (ProbeStop, ResourcePressureError) as error:
        report.update(status='partial', stop_reason='budget_or_deadline', error=str(error))
    except Exception as error:
        report.update(status='failed', stop_reason='error', error=f'{type(error).__name__}: {error}')
    finally:
        if student is not None and model is not None:
            report['policy_sha256_after'] = _policy_hash(model.policy)
            report['student_parameters_unchanged'] = report['policy_sha256_after'] == student['policy_sha256']
            if not report['student_parameters_unchanged']:
                report.update(status='failed', stop_reason='parameter_change', error='Student parameters changed')
        if writer is None:
            output.mkdir(parents=True, exist_ok=False)
        else:
            report['samples'] = len(writer.actions)
            report['sources_unchanged'] = pinned_sources == _sources()
            if not report['sources_unchanged']:
                report.update(status='failed', error='Collection sources changed')
            metadata.update(status=report['status'], stop_reason=report['stop_reason'],
                            collection_error=report['error'], account_after=account.snapshot())
            try:
                writer.finish(report['episode_reports'], metadata)
            finally:
                writer.close()
            for name, values in audit_arrays.items():
                dtype = '<U7' if name == 'selected_sources' else np.bool_ if name == 'step_completed' else np.int64
                np.save(output / f'{name}.npy', np.asarray(values, dtype=dtype), allow_pickle=False)
        report['account_after'] = account.snapshot()
        report['actual_attempted_interactions'] = account.attempted - report['account_before']['attempted']
        report['actual_completed_interactions'] = account.completed - report['account_before']['completed']
        _write(output / 'report.json', report)
        if writer is not None:
            _seal_extras(output)
    return report
