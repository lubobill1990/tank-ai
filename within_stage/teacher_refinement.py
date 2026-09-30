"""Bounded TRAIN-only paired teacher runner; no DEV/FINAL lookup, no ledger settlement."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from learning_lab.teacher_v2 import TeacherControllerV2
from rust_resources import ResourceMonitor, ResourcePressureError
from rust_validation import _primitive
from within_stage.opening_pool import make_opening_env
from within_stage.openings import ProbeAccounting, ProbeStop


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs' / '20260930-within-stage' / 'teacher-refinement-v1'


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def run_cases(label, indices, *, teacher_factory, description, output=OUTPUT):
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    report = json.loads((output / 'report.json').read_text())
    if label in {row['label'] for row in report['runs']}:
        raise ValueError('Refinement run labels cannot be reused')
    if not indices or any(type(i) is not int or not 0 <= i < 16 for i in indices):
        raise ValueError('Only predeclared first16 TRAIN indices are available')
    if any(i >= 8 for i in indices) and not (output / 'selected-controller.json').is_file():
        raise ValueError('Freeze chosen controller before TRAIN confirmation')
    if report['episodes_started'] + len(indices) > plan['max_full_episodes']:
        raise ValueError('Episode cap would be exceeded')
    directory = output / label
    directory.mkdir(exist_ok=False)
    monitor = ResourceMonitor(directory / 'resources.jsonl', device='cpu')
    deadline = time.monotonic() + max(0., plan['deadline_epoch'] - time.time())
    account = ProbeAccounting(plan['max_calls'], deadline=deadline, monitor=monitor)
    for name in ('raw_initialization_attempted', 'raw_initialization_completed', 'agent_attempted',
                 'agent_completed', 'agent_raw_frames', 'emulator_resets'):
        setattr(account, name, report['accounting'][name])
    source_paths = [
        ROOT / 'learning_lab' / 'teacher.py', ROOT / 'learning_lab' / 'teacher_v2.py',
        ROOT / 'within_stage' / 'teacher.py', ROOT / 'within_stage' / 'teacher_refinement.py',
        ROOT / 'within_stage' / 'opening_pool.py', ROOT / 'within_stage' / 'early_openings.py',
        ROOT / 'within_stage' / 'openings.py', ROOT / 'game_state.py',
        ROOT / 'heuristic.py', ROOT / 'battle_city_env.py', ROOT / 'battle_city.nes']
    sources = {str(path.relative_to(ROOT)): _hash(path) for path in source_paths}
    (directory / 'teacher-source.txt').write_bytes((ROOT / 'within_stage' / 'teacher.py').read_bytes())
    trial = {'label': label, 'description': description, 'indices': list(indices),
             'status': 'running', 'cases': [], 'sources': sources,
             'teacher_class': f'{teacher_factory.__module__}.{teacher_factory.__qualname__}',
             'account_before': account.snapshot(), 'learned_model': False}
    report['runs'].append(trial)
    _write(directory / 'config.json', trial)
    _write(output / 'report.json', report)
    pool_path = Path(plan['pool_path'])
    if _hash(pool_path) != plan['pool_sha256']:
        raise ValueError('Frozen opening pool hash mismatch')
    original = json.loads(pool_path.read_text())
    wanted = {plan['cases'][i]['config_index'] for i in indices}
    pool = {key: original[key] for key in ('format', 'stage', 'handoff', 'frozen', 'partition', 'reference')}
    pool['specs'] = [{key: row[key] for key in ('config_index', 'family_id', 'spec', 'initial_ram_sha256')}
                     for row in original['specs'] if row['config_index'] in wanted]
    pool['families'] = [{'family_id': row['family_id'], 'config_indices': [row['config_index']]}
                        for row in pool['specs']]
    del original
    try:
        for index in indices:
            account.check()
            if account.attempted >= account.limit:
                raise ProbeStop('100000 initialization+controller calls exhausted')
            case = plan['cases'][index]
            before = account.snapshot()
            row = {**case, 'status': 'running', 'outcome': None, 'kills': 0, 'deaths': 0,
                   'frames': 0, 'decisions': 0, 'teacher_attempts': 0, 'action_counts': [0] * 10,
                   'overrides': 0, 'teacher_clear': False, 'error': None}
            trial['cases'].append(row)
            report['episodes_started'] += 1
            _write(output / 'report.json', report)
            raw = None
            try:
                raw = make_opening_env(pool, case['config_index'], account=account, max_frames=12000)
                raw.reset(seed=0)
                row['initial_ram_sha256_actual'] = hashlib.sha256(raw.nes.ram.tobytes()).hexdigest()
                if row['initial_ram_sha256_actual'] != case['initial_ram_sha256']:
                    raise ValueError('Frozen initial state mismatch')
                controller = teacher_factory()
                controller.reset()
                with (directory / f'case-{index:02d}-actions.jsonl').open('x', encoding='utf-8') as log:
                    while True:
                        account.check()
                        if account.attempted >= account.limit:
                            raise ProbeStop('100000-call allowance exhausted')
                        state = raw.state
                        masks = raw.action_masks()
                        row['teacher_attempts'] += 1
                        action = _primitive(controller.act(state, masks))
                        if not masks[action]:
                            raise ValueError('Teacher selected a masked action')
                        trace = {'frame': state.frame, 'action': action,
                                 'mode': getattr(controller, 'mode', 'baseline'),
                                 'player': asdict(state.player) if state.player else None,
                                 'enemies': [asdict(e) for e in state.enemies],
                                 'bullets': [asdict(b) for b in state.bullets],
                                 'item': asdict(state.item) if state.item else None,
                                 'lives': state.lives, 'fire_ready': state.fire_ready,
                                 'kills': state.kills, 'remaining_enemies': state.remaining_enemies}
                        log.write(json.dumps(trace) + '\n')
                        _, _, terminated, truncated, info = raw.step(action)
                        row['decisions'] += 1
                        row['action_counts'][action] += 1
                        row.update(outcome=info['outcome'], kills=info['kills'],
                                   deaths=info['deaths'], frames=info['frames'],
                                   overrides=getattr(controller, 'override_count', 0))
                        if row['decisions'] % 128 == 0:
                            report['accounting'] = account.snapshot()
                            _write(output / 'report.json', report)
                            log.flush()
                        if terminated or truncated:
                            clear = info['outcome'] == 'clear'
                            if clear and (raw.state.remaining_enemies or raw.state.base_destroyed):
                                raise ValueError('Clear lacks terminal-state proof')
                            row.update(status='completed', teacher_clear=clear,
                                       terminal_remaining=raw.state.remaining_enemies,
                                       terminal_base_destroyed=raw.state.base_destroyed)
                            break
            except (ProbeStop, ResourcePressureError) as error:
                row.update(status='partial', error=str(error))
                raise
            except Exception as error:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                if raw is not None:
                    raw.close()
                row['attempted_calls'] = account.attempted - before['attempted']
                row['completed_calls'] = account.completed - before['completed']
                report['accounting'] = account.snapshot()
                _write(directory / f'case-{index:02d}.json', row)
                _write(output / 'report.json', report)
                print(json.dumps({key: row[key] for key in ('index', 'config_index', 'status', 'outcome',
                                                            'kills', 'deaths', 'frames', 'overrides',
                                                            'attempted_calls')}), flush=True)
        trial['status'] = 'completed'
    except (ProbeStop, ResourcePressureError) as error:
        trial.update(status='partial', error=str(error))
    except Exception as error:
        trial.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        rows = [row for row in trial['cases'] if row['status'] == 'completed']
        trial['clears'] = sum(row['teacher_clear'] for row in rows)
        trial['completed_cases'] = len(rows)
        trial['mean_kills'] = sum(row['kills'] for row in rows) / len(rows) if rows else None
        trial['sources_unchanged'] = sources == {str(path.relative_to(ROOT)): _hash(path) for path in source_paths}
        trial['account_after'] = account.snapshot()
        report['accounting'] = account.snapshot()
        report['status'] = 'in_progress' if trial['status'] == 'completed' else trial['status']
        _write(directory / 'summary.json', trial)
        _write(output / 'report.json', report)
    return trial
