"""One hash-bound heldout-final-once measurement; never candidate selection.

Importing this module does not read opening pools or create environments.
Only run_final_confirmation unseals the supplied pool, after a candidate is
predeclared. The same research ledger permanently claims one final attempt.
No teacher, curriculum, RAM writes, training, alternate-model retry or video
frame accumulation occurs. Optional video is intentionally not implemented.
"""
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from battle_city_env import BattleCityEnv
from learning_lab.budget import BudgetExhausted
from learning_lab.evaluation import _CountedPolicy
from learning_lab.rom_finetune import GLOBAL_DEADLINE
from rust_resources import ResourceMonitor, ResourcePressureError
from rust_train import _policy_hash
from rust_validation import PrimitiveActions, play_episode, _evaluation_state
from learning_lab.collect_teacher_v2 import _sha as _hash, _write


ROOT = Path(__file__).resolve().parents[1]
ROLE = 'heldout-final-once'
STAGES = (1, 8, 18, 28)
STAGE_CHOICES = (STAGES, (3, 8, 18, 28))
CASE_RESERVATION = 5120  # <=3000 four-frame actions + <=2048 boot/reset substeps.
WARNING = ('Final/test measurements are never used for gradients, model/decoder selection, '
           'hyperparameter tuning or a retry with another candidate. Failure is a valid final outcome.')


class IdentityError(RuntimeError):
    pass


def _type_name(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + '.' + cls.__qualname__


def model_architecture(model):
    policy = getattr(model, 'policy', None)
    if not isinstance(policy, torch.nn.Module) or not callable(getattr(model, 'predict', None)):
        raise ValueError('A learned torch policy with predict is required')
    shape = tuple(model.observation_space.shape)
    if not isinstance(model.observation_space, gym.spaces.Box) \
            or model.observation_space.dtype != np.float32 \
            or len(shape) != 3 or shape[1:] != (26, 26) or shape[0] not in (47, 94, 752) \
            or not isinstance(model.action_space, gym.spaces.Discrete) \
            or model.action_space.n != 10 or model.action_space.start != 0:
        raise ValueError('Expected canonical47/94/752-plane observations and ten primitives')
    from sb3_contrib import MaskablePPO
    if not isinstance(model, MaskablePPO) and getattr(model, 'learned_model', False) is not True:
        raise ValueError('Teacher/baseline controllers are not final learned candidates')
    extractor = getattr(policy, 'features_extractor', getattr(policy, 'extractor', None))
    return {
        'model_class': _type_name(model), 'policy_class': _type_name(policy),
        'observation_shape': list(shape), 'observation_dtype': str(model.observation_space.dtype),
        'actions': 10, 'extractor_class': _type_name(extractor) if extractor is not None else None,
        'net_arch': deepcopy(getattr(policy, 'net_arch', None)),
        'normalize_images': getattr(policy, 'normalize_images', None),
        'activation': _type_name(policy.activation_fn) if hasattr(policy, 'activation_fn') else None,
    }


def _loader_identity(loader):
    if isinstance(loader, str):
        module, name = loader.rsplit('.', 1)
        if not module.startswith('learning_lab.'):
            raise ValueError('Use a repository strict loader under learning_lab')
        loader = getattr(importlib.import_module(module), name)
    if not callable(loader) or '<locals>' in loader.__qualname__ \
            or not loader.__module__.startswith('learning_lab.'):
        raise ValueError('A module-level repository strict loader is required')
    path = Path(inspect.getsourcefile(loader)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Loader source must be in the repository')
    return {'name': loader.__module__ + '.' + loader.__qualname__,
            'source_file': str(path), 'source_sha256': _hash(path)}


def make_selection_record(model, checkpoint, *, loader, development_gate_passed, rationale,
                          primary_stage=1):
    """Declare candidate AND primary map1/3 without reading held-out seeds."""
    if type(primary_stage) is not int or primary_stage not in (1, 3):
        raise ValueError('Predeclare primary_stage as1 or3')
    if type(development_gate_passed) is not bool or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('Declare qualification (true OR false) and a nonempty selection rationale')
    checkpoint = Path(checkpoint).resolve()
    sidecar = checkpoint.with_suffix('.manifest.json')
    return {
        'role': ROLE, 'checkpoint': str(checkpoint), 'checkpoint_sha256': _hash(checkpoint),
        'checkpoint_manifest_sha256': _hash(sidecar) if sidecar.is_file() else None,
        'policy_sha256': _policy_hash(model.policy), 'architecture': model_architecture(model),
        'loader': _loader_identity(loader),
        'deployment_decoder': deepcopy(getattr(model, 'decoder_config', {'name': 'ordinary-policy-predict'})),
        'deterministic': True, 'development_gate_passed': development_gate_passed, 'rationale': rationale,
        'primary_stage': primary_stage,
        'warning': WARNING,
    }


def _source_hashes(model, loader):
    paths = [ROOT / name for name in (
        'learning_lab\\final_confirmation.py', 'battle_city_env.py', 'game_state.py',
        'heuristic.py', 'rust_validation.py', 'rust_train.py', 'learning_lab\\evaluation.py',
        'learning_lab\\budget.py', 'rust_resources.py', 'battle_city.nes')]
    paths += [Path(loader['source_file'])]
    for obj in (type(model), type(model.policy)):
        source = inspect.getsourcefile(obj)
        if source:
            paths.append(Path(source).resolve())
    from nes_py import nes_env
    paths.append(Path(nes_env.__file__).resolve())
    paths.extend(Path(nes_env.__file__).parent.glob('lib_nes_env*.pyd'))
    return {str(path): _hash(path) for path in sorted(set(paths))}


def _validate_selection(model, checkpoint, selection):
    if not isinstance(selection, dict) or selection.get('role') != ROLE \
            or selection.get('deterministic') is not True \
            or type(selection.get('primary_stage')) is not int or selection['primary_stage'] not in (1, 3) \
            or type(selection.get('development_gate_passed')) is not bool \
            or not isinstance(selection.get('rationale'), str) or not selection['rationale'].strip():
        raise IdentityError('Invalid predeclared deterministic final selection')
    if selection.get('checkpoint') != str(checkpoint) or _hash(checkpoint) != selection.get('checkpoint_sha256'):
        raise IdentityError('Selected checkpoint SHA256/path mismatch')
    sidecar = checkpoint.with_suffix('.manifest.json')
    if (_hash(sidecar) if sidecar.is_file() else None) != selection.get('checkpoint_manifest_sha256'):
        raise IdentityError('Selected checkpoint manifest identity changed')
    if _policy_hash(model.policy) != selection.get('policy_sha256') \
            or model_architecture(model) != selection.get('architecture'):
        raise IdentityError('Selected policy parameters/architecture changed')
    decoder = getattr(model, 'decoder_config', {'name': 'ordinary-policy-predict'})
    if decoder != selection.get('deployment_decoder'):
        raise IdentityError('Selected deployment decoder changed')
    if _loader_identity(selection['loader']['name']) != selection['loader']:
        raise IdentityError('Selected strict-loader source changed')


def _pool(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    rows = value['stable_stage_and_generalization_test']
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError('Exactly20 declared final opening seeds are required')
    seeds, delays = [], []
    for row in rows:
        seed, delay = row['seed'], row['reset_delay_frames']
        if type(seed) is not int or not 0 <= seed < 2**32 or type(delay) is not int \
                or not 0 <= delay < 24 or int(np.random.default_rng(seed).integers(0, 24)) != delay:
            raise ValueError('Declared final seed/reset-delay contract is invalid')
        seeds.append(seed); delays.append(delay)
    development = value['development_rom_seeds'] + value['training_rom_seeds']
    used_delays = {int(np.random.default_rng(seed).integers(0, 24)) for seed in development}
    if len(set(seeds)) != 20 or len(set(delays)) != 20 or set(seeds) & set(development) \
            or set(delays) & used_delays or set(value['sealed_stages']) != {8, 18, 28}:
        raise ValueError('Final seed pool overlaps development or contains duplicate openings')
    reset_hashes = {_hash(ROOT / 'battle_city_env.py'),
                    hashlib.sha256(inspect.getsource(BattleCityEnv.reset).encode()).hexdigest()}
    if value['reset_delay_source_sha256'] not in reset_hashes:
        raise ValueError('Opening pool reset-source hash mismatch')
    return seeds, rows


class _Guard:
    def __init__(self, budget, monitor):
        state = budget.snapshot()
        if not 0 < state['max_interactions'] <= 4000000:
            raise ValueError('Invalid global research budget')
        self.epoch = min(state['deadline_epoch'], GLOBAL_DEADLINE)
        self.deadline = time.monotonic() + max(0., min(
            state['remaining_seconds'], self.epoch - time.time()))
        self.monitor = monitor

    def check(self, phase):
        if time.monotonic() >= self.deadline or time.time() >= self.epoch:
            raise BudgetExhausted('Final confirmation global/wall budget expired')
        self.monitor.check(phase)


def _make_environment(stage):
    return BattleCityEnv(stage=stage, objective='full', max_frames=12000, frame_skip=4,
                         reward_profile='sparse')


class _ObservedROM(PrimitiveActions):
    def __init__(self, raw, *, guard, evidence, row, hashes, predictor):
        super().__init__(raw, deadline=guard.deadline)
        self.guard, self.evidence, self.row = guard, evidence, row
        self.hashes, self.predictor, self.resetting = hashes, predictor, False
        self.bootstrap_attempted = self.bootstrap_completed = self.gameplay_completed = 0
        original = raw.nes.step
        def checked_frame(*args, **kwargs):
            guard.check('final_nes_frame')
            if self.resetting:
                if self.bootstrap_attempted >= 2048 \
                        or self.bootstrap_attempted + predictor.attempts >= CASE_RESERVATION:
                    raise BudgetExhausted('Final reset substep cap')
                self.bootstrap_attempted += 1
            result = original(*args, **kwargs)
            if self.resetting:
                self.bootstrap_completed += 1
            return result
        raw.nes.step = checked_frame

    def reset(self, *, seed=None, options=None):
        self.resetting = True
        try:
            observation, info = super().reset(seed=seed, options=options)
        finally:
            self.resetting = False
        ram, pixels = self.env.nes.ram.copy(), self.env.nes.screen.copy()
        digest = hashlib.sha256(ram.tobytes()).hexdigest()
        self.row['initial_ram_sha256'] = digest
        self.row['initial_pixels_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
        np.save(self.evidence / 'initial-ram.npy', ram, allow_pickle=False)
        np.save(self.evidence / 'initial-pixels.npy', pixels, allow_pickle=False)
        if digest in self.hashes:
            raise ValueError('Duplicate actual initial RAM identity within the same map')
        self.hashes.add(digest)
        self.row['initial_identity_unique'] = True
        return observation, info

    def step(self, action):
        self.guard.check('final_primitive_step')
        if not self.action_masks()[action]:
            raise ValueError('Final policy selected a masked primitive')
        self.row['action_counts'][int(action)] += 1
        result = super().step(action)
        self.gameplay_completed += 1
        info = result[4]
        trace = {'decision': self.gameplay_completed, 'action': int(action), 'reward': float(result[1]),
                 'terminated': bool(result[2]), 'truncated': bool(result[3]), 'info': info}
        with (self.evidence / 'actions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(trace, allow_nan=False) + '\n')
        return result


def _summarize(report, seen):
    per_stage = {}
    for stage in report['stages']:
        rows = [row for row in report['cases'] if row['stage'] == stage and row['status'] == 'completed']
        threshold = 16 if stage == report['primary_stage'] else 6
        wins = sum(row['verified_full_clear'] for row in rows)
        complete = len(rows) == 20
        per_stage[str(stage)] = {
            'planned': 20, 'completed': len(rows), 'wins': wins, 'threshold': threshold,
            'complete': complete, 'passed': complete and wins >= threshold,
            'win_rate_on_completed': wins / len(rows) if rows else None,
            'initial_ram_hashes': sorted(seen[stage]),
            'outcomes': {outcome: sum(row['episode']['outcome'] == outcome for row in rows)
                         for outcome in sorted({row['episode']['outcome'] for row in rows})}}
    report['per_stage'] = per_stage
    report['completed_cases'] = sum(row['completed'] for row in per_stage.values())
    report['overall_passed'] = report['completed_cases'] == 80 and all(
        row['passed'] for row in per_stage.values())
    report['actual_decisions'] = sum(row.get('attempted_decisions', 0) for row in report['cases'])
    report['unresolved_reservations'] = [row['reservation'] for row in report['cases']
                                         if row.get('reservation') and not row.get('settled', False)]


def _receipt_path(budget):
    path = getattr(budget, 'path', None)
    if path is None:
        raise ValueError('Final confirmation requires budget.path for the durable one-use receipt')
    return Path(path).resolve().parent / 'final-test-used.json'


def _create_final_receipt(budget, receipt):
    """Exclusive, durable creation; even a partial/crashed receipt is never replaced."""
    path = _receipt_path(budget)
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def run_final_confirmation(model, checkpoint, output, budget, *, selection_record,
                           opening_pool_path, stages=STAGES):
    """Perform the single predeclared final measurement, including unqualified candidates.

    Use make_selection_record BEFORE this call. It does not read held-out seeds.
    Predeclare primary_stage=1 or3 there, then pass matching stages=(primary,8,18,28).
    This function persists selection and SOURCE.SIG before even invoking the
    declared CPU strict loader for policy/checkpoint binding. An exclusive
    budget.path.parent/final-test-used.json receipt and zero-interaction ledger
    claim prevent retargeting, including after partial failure. The parent must
    stop all training before invoking this final measurement.
    Failed/incomplete cases retain reservations; no replacement seed/model is tried.
    """
    output, checkpoint, opening_pool_path = map(lambda p: Path(p).resolve(),
                                                (output, checkpoint, opening_pool_path))
    output.mkdir(parents=True, exist_ok=False)
    stages = tuple(stages)
    valid_scope = stages in STAGE_CHOICES and all(type(stage) is int for stage in stages)
    report = {'format': 'tank-ai-final-confirmation-v1', 'role': ROLE, 'status': 'running',
              'planned_cases': 80, 'completed_cases': 0, 'overall_passed': False,
              'stages': list(stages) if valid_scope else [],
              'primary_stage': stages[0] if valid_scope else None,
              'warning': WARNING, 'selection': deepcopy(selection_record), 'cases': [],
              'video_recording': False, 'unresolved_reservations': [], 'actual_decisions': 0}
    seen = {stage: set() for stage in report['stages']}
    previous_case_sha = None
    try:
        if not valid_scope:
            raise ValueError('Final scope must be exactly (1,8,18,28) or (3,8,18,28)')
        receipt_path = _receipt_path(budget)
        if receipt_path.exists():
            raise FileExistsError(f'Final test already claimed: {receipt_path}')
        _validate_selection(model, checkpoint, selection_record)
        if selection_record['primary_stage'] != stages[0]:
            raise ValueError('Requested stages must match the predeclared primary_stage')
        seeds, pool_rows = _pool(opening_pool_path)
        pool_sha = _hash(opening_pool_path)
        sources = _source_hashes(model, selection_record['loader'])
        selection = {**deepcopy(selection_record), 'opening_pool': str(opening_pool_path),
                     'opening_pool_sha256': pool_sha, 'declared_openings': pool_rows,
                     'stages': list(stages), 'planned_cases': 80, 'max_frames': 12000,
                     'frame_skip': 4, 'sources': sources,
                     'final_test_never_used_for_gradients_or_selection': True}
        _write(output / 'selection.json', selection)
        selection_sha = _hash(output / 'selection.json')
        signature = {'format': 'tank-ai-final-source-sig-v1', 'role': ROLE,
                     'selection_sha256': selection_sha, 'checkpoint_sha256': selection['checkpoint_sha256'],
                     'policy_sha256': selection['policy_sha256'], 'opening_pool_sha256': pool_sha,
                     'primary_stage': stages[0], 'stages': list(stages),
                     'sources': sources, 'warning': WARNING}
        _write(output / 'SOURCE.SIG.json', signature)
        signature_sha = _hash(output / 'SOURCE.SIG.json')
        report.update(selection=selection, selection_sha256=selection_sha, source_sig_sha256=signature_sha)
        _write(output / 'report.json', report)
        monitor = ResourceMonitor(output / 'resources.jsonl', device=str(getattr(model, 'device', 'cpu')).split(':')[0])
        guard = _Guard(budget, monitor)
        guard.check('final_admission')
        receipt_path = _create_final_receipt(budget, {
            'format': 'tank-ai-final-test-use-receipt-v1', 'role': ROLE,
            'selection_sha256': selection_sha, 'opening_pool_sha256': pool_sha,
            'checkpoint_sha256': selection['checkpoint_sha256'],
            'policy_sha256': selection['policy_sha256'], 'source_sig_sha256': signature_sha,
            'primary_stage': stages[0], 'stages': list(stages),
            'selection_file': str(output / 'selection.json'), 'output': str(output),
            'budget_path': str(Path(budget.path).resolve()), 'created_epoch': time.time(),
            'warning': WARNING, 'required_parent_protocol': 'Stop all training before this receipt; '
                       'never use final results for another update or candidate selection.',
        })
        receipt_sha = _hash(receipt_path)
        report['final_test_receipt'] = {'path': str(receipt_path), 'sha256': receipt_sha}
        _write(output / 'report.json', report)
        module, name = selection['loader']['name'].rsplit('.', 1)
        loader = getattr(importlib.import_module(module), name)
        verified = loader(checkpoint, device='cpu')
        if _policy_hash(verified.policy) != selection['policy_sha256']:
            raise IdentityError('Passed policy is not the selected checkpoint policy')
        del verified
        claim = budget.reserve(ROLE, 1)
        budget.settle(claim, 0)
        report['final_once_claim'] = {'reservation': claim, 'actual_environment_decisions': 0}

        def identities_unchanged():
            _validate_selection(model, checkpoint, selection)
            if _hash(opening_pool_path) != pool_sha or _source_hashes(model, selection['loader']) != sources \
                    or _hash(output / 'selection.json') != selection_sha \
                    or _hash(output / 'SOURCE.SIG.json') != signature_sha \
                    or _hash(receipt_path) != receipt_sha:
                raise IdentityError('Final pool/source/selection identity changed')

        for stage in stages:
            for position, seed in enumerate(seeds):
                guard.check('final_case')
                identities_unchanged()
                evidence = output / f'stage-{stage:02d}-case-{position:02d}'
                evidence.mkdir()
                token = budget.reserve(f'{ROLE}-stage-{stage}-case-{position}', CASE_RESERVATION)
                row = {'stage': stage, 'seed': seed, 'opening_index': position, 'status': 'running',
                       'reservation': token, 'reserved': CASE_RESERVATION, 'settled': False,
                       'selection_sha256': selection_sha, 'source_sig_sha256': signature_sha,
                       'final_test_receipt_sha256': receipt_sha,
                       'checkpoint_sha256': selection['checkpoint_sha256'],
                       'policy_sha256_before': selection['policy_sha256'],
                       'previous_case_sha256': previous_case_sha, 'action_counts': [0] * 10,
                       'episode': None, 'verified_full_clear': False}
                report['cases'].append(row)
                _write(evidence / 'case.json', row)
                _write(output / 'report.json', report)
                raw = None
                predictor = _CountedPolicy(model, 3000, guard.deadline)
                try:
                    if callable(getattr(model, 'reset', None)):
                        model.reset()
                    identities_unchanged()
                    with _evaluation_state(model):
                        raw = _ObservedROM(_make_environment(stage), guard=guard, evidence=evidence,
                                           row=row, hashes=seen[stage], predictor=predictor)
                        episode = play_episode(raw, model=predictor,
                                               nstack=model.observation_space.shape[0] // 47,
                                               deterministic=True, seed=seed, deadline=guard.deadline,
                                               resource_check=lambda: guard.check('final_episode'))
                    row['episode'] = episode
                    if not 0 < episode['frames'] <= 12000 or not 0 < episode['decisions'] <= 3000:
                        raise ValueError('Final episode exceeded its declared frame/decision cap')
                    row['terminal_remaining_enemies'] = int(raw.state.remaining_enemies)
                    row['terminal_base_destroyed'] = bool(raw.state.base_destroyed)
                    terminal_ram, terminal_pixels = raw.env.nes.ram.copy(), raw.env.nes.screen.copy()
                    np.save(evidence / 'terminal-ram.npy', terminal_ram, allow_pickle=False)
                    np.save(evidence / 'terminal-pixels.npy', terminal_pixels, allow_pickle=False)
                    row['terminal_ram_sha256'] = hashlib.sha256(terminal_ram.tobytes()).hexdigest()
                    row['terminal_pixels_sha256'] = hashlib.sha256(terminal_pixels.tobytes()).hexdigest()
                    identities_unchanged()
                    clear = episode['outcome'] == 'clear'
                    if clear and (raw.state.remaining_enemies != 0 or raw.state.base_destroyed):
                        raise ValueError('Clear outcome lacks terminal ROM-state confirmation')
                    row['verified_full_clear'] = clear
                    row['policy_sha256_after'] = _policy_hash(model.policy)
                    row['status'] = 'completed'
                except (BudgetExhausted, TimeoutError, ResourcePressureError) as error:
                    row.update(status='partial', error=f'{type(error).__name__}: {error}')
                    raise
                except BaseException as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
                    raise
                finally:
                    row['gameplay_attempted'] = predictor.attempts
                    row['gameplay_completed'] = raw.gameplay_completed if raw is not None else 0
                    row['initialization_attempted'] = raw.bootstrap_attempted if raw is not None else 0
                    row['initialization_completed'] = raw.bootstrap_completed if raw is not None else 0
                    row['attempted_decisions'] = row['gameplay_attempted'] + row['initialization_attempted']
                    row['completed_decisions'] = row['gameplay_completed'] + row['initialization_completed']
                    if raw is not None:
                        try:
                            raw.close()
                        except Exception as error:
                            row.update(status='failed', error=f'Environment cleanup failed: {error}')
                    if row['status'] == 'completed':
                        budget.settle(token, row['attempted_decisions'])
                        row['settled'] = True
                    _write(evidence / 'case.json', row)
                    previous_case_sha = _hash(evidence / 'case.json')
                    row['evidence_sha256'] = previous_case_sha
                    _summarize(report, seen)
                    _write(output / 'report.json', report)
                    if row['status'] == 'failed' and 'Environment cleanup failed:' in row.get('error', ''):
                        raise RuntimeError(row['error'])
        report['status'] = 'completed'
    except (BudgetExhausted, TimeoutError, ResourcePressureError) as error:
        report.update(status='partial', error=f'{type(error).__name__}: {error}')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        _summarize(report, seen)
        if report['status'] != 'completed':
            report['overall_passed'] = False
        report['budget_snapshot'] = budget.snapshot()
        _write(output / 'report.json', report)
        (output / 'report.sha256').write_text(_hash(output / 'report.json') + '\n', encoding='ascii')
    return report
