"""CPU-only seeded NATIVE teacher pilot; never ROM/model-win evidence.

The existing per-frame ABI feeds StateTracker once per actual native frame.
Action repeat and terminal behavior are checked against NativeBatch.step before
collection. No RAM is written except through the backend's normal seeded reset.
Every ROM initialization NES.step is conservatively charged to the same cap.
"""
import argparse
import ctypes as ct
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

import game_state as gs
from battle_city_env import BattleCityEnv
from heuristic import redundant_boundary_action
from learning_lab.collect_teacher_v2 import _seal, _sha, _sources, _verify_dataset, _write
from learning_lab.demonstrations import DemonstrationWriter, _action
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_env import NativeBatch, episode_seed, stage_ram, BACKEND_VERSION, RNG_VERSION
from rust_resources import ResourceMonitor


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = 'aedfcab218f04aabb0404eb31edbdcbd'
GLOBAL_DEADLINE = datetime.fromisoformat('2026-09-30T21:01:44+08:00').timestamp()
WARNING = ('NATIVE seeded experimental non-learned teacher trajectories, NOT ROM or model wins. '
           'Independent native PRNG streams are not independent ROM timing phases. '
           'Native mechanics/RNG distribution and terminal conventions differ from the ROM.')


class StopBudget(Exception):
    pass


class BridgeError(RuntimeError):
    pass


class DecisionBudget:
    def __init__(self, maximum, deadline):
        if type(maximum) is not int or not 1 <= maximum <= 65536:
            raise ValueError('Total decision cap must be in 1..65536')
        self.maximum, self.deadline = maximum, float(deadline)
        self.counts = {f'{phase}_{kind}': 0 for phase in (
            'rom_initialization', 'native_probe', 'native_collection')
            for kind in ('attempted', 'completed')}
        self.native_physics_frames = 0

    @property
    def attempted(self):
        return sum(value for key, value in self.counts.items() if key.endswith('_attempted'))

    @property
    def completed(self):
        return sum(value for key, value in self.counts.items() if key.endswith('_completed'))

    def check(self):
        if time.time() >= self.deadline:
            raise StopBudget('deadline')
        if self.attempted >= self.maximum:
            raise StopBudget('decision_budget')

    def charge(self, phase):
        self.check()
        self.counts[phase + '_attempted'] += 1

    def complete(self, phase):
        self.counts[phase + '_completed'] += 1

    def report(self):
        return {**self.counts, 'total_attempted_decisions': self.attempted,
                'total_completed_decisions': self.completed,
                'native_physics_frames': self.native_physics_frames,
                'maximum_decisions': self.maximum, 'deadline': self.deadline,
                'reservation': RESERVATION, 'ledger_settled': False}


def canonical_observation(state, max_frames):
    proxy = SimpleNamespace(state=state, max_frames=max_frames, _last_macro=-1, objective='full')
    return BattleCityEnv._observation(proxy)


def canonical_masks(state):
    result = np.ones(10, bool)
    result[5] = state.fire_ready
    for action in range(1, 10):
        if redundant_boundary_action(state, action):
            result[action] = False
    return result


class NativeBackend:
    """One native environment; typed private ABI handles do not patch old modules."""
    def __init__(self, initial, max_frames, seed, budget):
        self.batch = NativeBatch(initial[None].copy(), max_frames, seeds=[seed])
        self.budget = budget
        self._frame = ct.CFUNCTYPE(ct.c_int32, ct.c_void_p, ct.c_size_t, ct.c_uint8)(
            ('mirror_frame', self.batch.lib))
        self._counters = ct.CFUNCTYPE(ct.c_int32, ct.c_void_p, ct.c_size_t,
                                     ct.POINTER(ct.c_uint32))(('mirror_counters', self.batch.lib))
        self._counter_buffer = np.empty(5, np.uint32)

    def reset(self, seed):
        self.batch.reset(0, seed=seed)

    def ram(self):
        return self.batch.ram(0)

    def counters(self):
        self.batch._check(self._counters(self.batch.handle, 0,
                                        self._counter_buffer.ctypes.data_as(ct.POINTER(ct.c_uint32))))
        return self._counter_buffer.copy()

    def frame_step(self, control):
        if time.time() >= self.budget.deadline:
            raise StopBudget('deadline')
        self.batch._check(self._frame(self.batch.handle, 0, control))
        self.budget.native_physics_frames += 1
        self.batch._masks_fresh = False

    def observe(self):
        return self.batch.observe()[0].copy()

    def masks(self):
        return self.batch.masks()[0].copy()

    def support(self):
        return int(self.batch.lib.mirror_support(self.batch.handle, 0))

    def close(self):
        self.batch.close()


class FrameBridge:
    """Actual frame-level tracker with explicit capture-before-autoreset semantics."""
    def __init__(self, backend, *, stage, stream_id, master_seed, max_frames=12000):
        if stage not in (1, 3):
            raise ValueError('Only native training stages 1 and 3 are allowed')
        self.backend, self.stage, self.stream_id = backend, stage, stream_id
        self.master_seed, self.max_frames = master_seed, max_frames
        self.episode_id = 0
        self.tracker = gs.StateTracker()
        self.support_codes = set()
        self.observer_counter_differences = []
        self._last_kill_difference = 0
        self.alignment_checks, self.max_observation_error = 0, 0.
        try:
            self._reset()
        except BaseException:
            backend.close()
            raise

    @property
    def state(self):
        return self.tracker.state

    def _context(self):
        return f'stage={self.stage} stream={self.stream_id} episode={self.episode_id} frame={self.state.frame}'

    def _check_counters(self):
        support = self.backend.support()
        self.support_codes.add(support)
        if support:
            raise BridgeError(f'Native support code {support}; {self._context()}')
        counters = list(map(int, self.backend.counters()))
        self.stats = dict(zip(('frames', 'kills', 'deaths', 'shots', 'pickups'), counters))
        if self.state.frame != counters[0]:
            raise BridgeError(f'StateTracker/native frame mismatch: tracker '
                              f'frame={self.state.frame}, native={counters}; '
                              f'{self._context()}')
        # The frozen frame observer cannot see a spawn and kill within one frame.
        # Keep its canonical state intact; terminal statistics use native counters.
        difference = counters[1] - self.state.kills
        if difference != self._last_kill_difference:
            self.observer_counter_differences.append({
                'episode_id': self.episode_id, 'frame': self.state.frame,
                'tracker_kills': self.state.kills, 'native_kills': counters[1],
                'difference': difference,
            })
            self._last_kill_difference = difference
        if self.state.stage != self.stage:
            raise BridgeError(f'Unexpected live-RAM stage {self.state.stage}; {self._context()}')

    def _align(self):
        self._check_counters()
        observed = self.backend.observe()
        expected = canonical_observation(self.state, self.max_frames)
        error = float(np.max(np.abs(observed - expected)))
        self.max_observation_error = max(self.max_observation_error, error)
        if observed.shape != (47, 26, 26) or not np.isfinite(observed).all() \
                or not np.allclose(observed, expected, rtol=0, atol=1e-6):
            channels = np.flatnonzero(np.max(np.abs(observed - expected), axis=(1, 2)) > 1e-6).tolist()
            raise BridgeError(f'Canonical/native observation mismatch max={error} channels={channels}; '
                              f'{self._context()}')
        masks = self.backend.masks()
        if not np.array_equal(masks, canonical_masks(self.state)):
            raise BridgeError(f'Canonical/native mask mismatch; {self._context()}')
        self.observation, self.action_masks = observed.copy(), masks.copy()
        self.alignment_checks += 1

    def _reset(self):
        self.seed = episode_seed(self.master_seed, self.stream_id, self.episode_id)
        self._last_kill_difference = 0
        self.backend.reset(self.seed)
        self.tracker.reset(self.backend.ram())
        self._align()

    def step(self, action):
        action = _action(action)
        if not self.action_masks[action]:
            raise ValueError('Teacher selected an illegal native action')
        episode, seed = self.episode_id, self.seed
        deaths_before = self.stats['deaths']
        outcome = ''
        for index in range(4):
            control = gs.ACTION_MASKS[action]
            self.backend.frame_step(control if index < 2 else control & ~1)
            ram = self.backend.ram()
            self.tracker.advance(ram)
            self._check_counters()
            if ram[0x68] != 0x80:
                outcome = 'base_lost'
            elif ram[0x51] == 0 or (self.stats['deaths'] > deaths_before and ram[0x51] <= 1):
                outcome = 'lives_exhausted'
            elif ram[0x80] == 0:
                outcome = 'clear'
            elif self.stats['frames'] >= self.max_frames:
                outcome = 'timeout'
            if outcome:
                break
        self._align()
        result = {
            'done': bool(outcome), 'outcome': outcome, 'stats': dict(self.stats),
            'state': self.state, 'observation': self.observation.copy(), 'ram': ram.copy(),
            'episode_id': episode, 'episode_seed': seed,
        }
        if outcome:
            self.episode_id += 1
            self._reset()
        return result

    def close(self):
        self.backend.close()


def training_ram(stage, budget):
    """Call frozen stage_ram(seed=0), counting every NES.step via read-only profiling."""
    from nes_py import NESEnv
    if stage not in (1, 3):
        raise ValueError('Only stages 1 and 3 are allowed')
    code, previous = NESEnv.step.__code__, sys.getprofile()
    if previous is not None:
        raise RuntimeError('Cannot safely replace an existing profiler for initialization accounting')

    def profile(frame, event, value):
        if frame.f_code is code:
            if event == 'call':
                budget.charge('rom_initialization')
            elif event == 'return' and value is not None:
                budget.complete('rom_initialization')

    try:
        sys.setprofile(profile)
        return stage_ram(stage, seed=0)
    finally:
        sys.setprofile(previous)


def validate_native_bridge(initials, budget, *, master_seed):
    """Paired controls against frozen NativeBatch.step, including repeated timeouts."""
    records = []
    for stage, initial in initials.items():
        stream_id, max_frames = 100 + stage, 48
        seed = episode_seed(master_seed, stream_id, 0)
        bridge = FrameBridge(NativeBackend(initial, max_frames, seed, budget),
                             stage=stage, stream_id=stream_id, master_seed=master_seed, max_frames=max_frames)
        reference = NativeBatch(initial[None].copy(), max_frames, seeds=[seed])
        timeouts, before_frame = 0, 0
        try:
            np.testing.assert_array_equal(bridge.observation, reference.observe()[0])
            for index in range(32):
                masks = reference.masks()[0].copy()
                np.testing.assert_array_equal(masks, bridge.action_masks)
                proposal = (5, 1, 6, 3, 8, 4, 9, 2, 7, 0)[index % 10]
                action = proposal if masks[proposal] else 0
                budget.charge('native_probe')
                actual = bridge.step(action)
                budget.complete('native_probe')
                budget.charge('native_probe')
                obs, _, done, stats = reference.step(np.array([action]))
                budget.complete('native_probe')
                budget.native_physics_frames += int(stats[0, 0]) - before_frame
                np.testing.assert_array_equal(actual['ram'], reference.ram(0))
                np.testing.assert_array_equal(actual['observation'], obs[0])
                np.testing.assert_array_equal(list(actual['stats'].values()), stats[0])
                if bool(done[0]) != actual['done']:
                    raise BridgeError('NativeBatch terminal parity failed')
                before_frame = int(stats[0, 0])
                if actual['done']:
                    timeouts += actual['outcome'] == 'timeout'
                    reference.reset(0, seed=bridge.seed)
                    before_frame = 0
                    np.testing.assert_array_equal(bridge.observation, reference.obs[0])
            records.append({'stage': stage, 'paired_decisions': 32, 'native_decisions': 64,
                            'autoreset_timeouts': timeouts, 'alignment_checks': bridge.alignment_checks,
                            'max_observation_error': bridge.max_observation_error,
                            'support_codes': sorted(bridge.support_codes), 'status': 'passed'})
        finally:
            reference.close()
            bridge.close()
    return {'status': 'passed', 'cases': records,
            'contract': 'Live-RAM StateTracker.advance per actual frame; exact native RAM/stats/obs/mask parity'}


def collect_streams(output, streams, budget, *, teacher_factory=TeacherControllerV2,
                    monitor=None, max_complete=64, sources=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not 1 <= len(streams) <= 16 or not 16 <= max_complete <= 64:
        raise ValueError('Use 1..16 streams and a completion cap in 16..64')
    writers = [DemonstrationWriter(output / f'stream-{stream.stream_id:02d}-stage-{stream.stage}')
               for stream in streams]
    teachers = [teacher_factory() for _ in streams]
    reports = [[] for _ in streams]
    steps, attempts = [0] * len(streams), [0] * len(streams)
    records, public_paths = [], []
    gate, error, reason = None, None, 'episode_cap'
    try:
        for teacher in teachers:
            teacher.reset()
        while len(records) < max_complete:
            for index, stream in enumerate(streams):
                budget.charge('native_collection')
                attempts[index] += 1
                if monitor is not None:
                    monitor.check('native_teacher_collection')
                action = _action(teachers[index].act(stream.state, stream.action_masks.copy()))
                writers[index].add(stream.observation, action, stream.action_masks,
                                    episode_start=steps[index] == 0, executed_action=action)
                steps[index] += 1
                result = stream.step(action)
                budget.complete('native_collection')
                if result['done']:
                    row = {
                        'episode_id': result['episode_id'], 'stream_id': stream.stream_id,
                        'stage': stream.stage, 'episode_seed': result['episode_seed'],
                        'outcome': result['outcome'], 'decisions': steps[index],
                        'attempted_decisions': attempts[index], 'episode_finished': True,
                        **result['stats'], 'native_clear': result['outcome'] == 'clear',
                        'terminal_remaining_enemies': result['state'].remaining_enemies,
                        'terminal_base_destroyed': result['state'].base_destroyed,
                        'learned_model': False, 'rom_win': False,
                    }
                    reports[index].append(row)
                    records.append(row)
                    with (output / 'episodes.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
                    steps[index] = attempts[index] = 0
                    teachers[index].reset()
                    if len(records) == 16:
                        gate = {'completed_episodes': 16,
                                'clear_episodes': sum(row['native_clear'] for row in records),
                                'mean_kills': float(np.mean([row['kills'] for row in records]))}
                        gate['stop'] = gate['clear_episodes'] == 0 and gate['mean_kills'] < 10
                        if gate['stop']:
                            reason = 'pilot_quality_gate'
                            raise StopBudget(reason)
                    if len(records) >= max_complete:
                        break
    except StopBudget as stopped:
        reason = str(stopped)
    except Exception as failure:
        reason = 'bridge_failure' if isinstance(failure, BridgeError) else 'error'
        error = f'{type(failure).__name__}: {failure}'
    finally:
        for index, (stream, writer) in enumerate(zip(streams, writers)):
            if steps[index] or attempts[index]:
                counters = dict(zip(('frames', 'kills', 'deaths', 'shots', 'pickups'),
                                    map(int, stream.backend.counters())))
                reports[index].append({
                    'episode_id': len(reports[index]), 'stream_id': stream.stream_id, 'stage': stream.stage,
                    'episode_seed': stream.seed, 'outcome': reason, 'decisions': steps[index],
                    'attempted_decisions': attempts[index], 'episode_finished': False,
                    **counters, 'native_clear': False, 'rom_win': False, 'learned_model': False})
            metadata = {
                'backend': 'NATIVE-seeded-experimental', 'backend_version': BACKEND_VERSION,
                'rng_version': RNG_VERSION, 'stage': stream.stage, 'stream_id': stream.stream_id,
                'master_seed': stream.master_seed, 'initial_rom_seed': 0, 'learned_model': False,
                'rom_wins': False, 'warning': WARNING, 'source_sha256': sources or {},
                'support_codes': sorted(stream.support_codes), 'alignment_checks': stream.alignment_checks,
                'max_observation_error': stream.max_observation_error,
                'observer_counter_differences': stream.observer_counter_differences,
                'counter_semantics': 'Teacher uses canonical StateTracker observer counts; '
                                     'terminal stats are authoritative native counts. Same-frame '
                                     'spawn/kill events may be absent from observer counts.',
                'frame_clock': 'StateTracker advanced once per actual native frame, checked against mirror_counters',
                'history': 'One streaming writer per environment; starts reset only at native episode boundaries',
                'label_timing': 'Canonical live-RAM teacher state and aligned observation before primitive step',
                'collection_stop_reason': reason, 'collection_error': error,
                'initial_ram_policy': 'Frozen stage_ram(stage,seed=0); no post-reset RAM writes',
                'native_decisions': sum(row['attempted_decisions'] for row in reports[index]),
                'rom_controller_decisions': 0,
            }
            writer.finish(reports[index], metadata)
            _verify_dataset(writer.path)
            _seal(writer.path)
            if len(writer.actions) and error is None:
                public_paths.append(str(writer.path))
    by_stage = {}
    for stage in (1, 3):
        rows = [row for row in records if row['stage'] == stage]
        by_stage[str(stage)] = {
            'complete_episodes': len(rows), 'native_clears': sum(row['native_clear'] for row in rows),
            'mean_kills': float(np.mean([row['kills'] for row in rows])) if rows else None,
            'outcomes': {name: sum(row['outcome'] == name for row in rows)
                         for name in sorted({row['outcome'] for row in rows})}}
    return {
        'stop_reason': reason, 'error': error, 'completed_episodes': len(records),
        'clear_episodes': sum(row['native_clear'] for row in records),
        'clear_fraction': sum(row['native_clear'] for row in records) / len(records) if records else None,
        'pilot_gate': gate, 'per_stage': by_stage, 'dataset_paths': public_paths,
        'incomplete_episode_count': sum(not row['episode_finished'] for rows in reports for row in rows),
        'support_codes': sorted({code for stream in streams for code in stream.support_codes}),
        'alignment_checks': sum(stream.alignment_checks for stream in streams),
        'max_observation_error': max(stream.max_observation_error for stream in streams),
        'observer_counter_differences': [
            {'stream_id': stream.stream_id, 'stage': stream.stage, **event}
            for stream in streams for event in stream.observer_counter_differences],
    }


def _source_hashes():
    values = _sources()
    values['learning_lab\\native_teacher_data.py'] = _sha(Path(__file__))
    values['rust_env.py'] = _sha(ROOT / 'rust_env.py')
    values['native\\target\\release\\battle_mirror_probe.dll'] = _sha(
        ROOT / 'native' / 'target' / 'release' / 'battle_mirror_probe.dll')
    for path in sorted((ROOT / 'native' / 'src').glob('*.rs')):
        values[str(path.relative_to(ROOT))] = _sha(path)
    return values


def run(output, *, envs=8, max_decisions=65536, max_complete=64, master_seed=20260930, deadline):
    if type(envs) is not int or not 1 <= envs <= 16:
        raise ValueError('Use at most sixteen CPU environments')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    budget = DecisionBudget(max_decisions, min(deadline, GLOBAL_DEADLINE, started + 45 * 60))
    monitor = ResourceMonitor(output / 'resources.jsonl', device='cpu')
    report = {'status': 'running', 'warning': WARNING, 'reservation': RESERVATION,
              'stages': [1, 3], 'initial_rom_seed': 0, 'streams': envs, 'master_seed': master_seed,
              'real_unit_test_decisions': 0, 'rom_teacher_gameplay_decisions': 0,
              'rom_initialization_accounting': 'Every NES.step inside frozen stage_ram is charged conservatively',
              'source_sha256': _source_hashes(), 'bridge_validation': None, 'collection': None}
    streams = []
    try:
        monitor.check('admission', admission_slots=1, force=True)
        initials = {}
        for stage in (1, 3):
            budget.check()
            initials[stage] = training_ram(stage, budget)
            np.save(output / f'initial-stage-{stage}-seed-0.npy', initials[stage], allow_pickle=False)
            report['budget'] = budget.report()
            _write(output / 'report.json', report)
        report['bridge_validation'] = validate_native_bridge(initials, budget, master_seed=master_seed)
        _write(output / 'bridge-validation.json', report['bridge_validation'])
        report['budget'] = budget.report()
        _write(output / 'report.json', report)
        for index in range(envs):
            stage = (1, 3)[index % 2]
            native_seed = episode_seed(master_seed, index, 0)
            streams.append(FrameBridge(NativeBackend(initials[stage], 12000, native_seed, budget),
                                       stage=stage, stream_id=index, master_seed=master_seed))
        report['collection'] = collect_streams(output / 'datasets', streams, budget,
                                               monitor=monitor, max_complete=max_complete,
                                               sources=report['source_sha256'])
        report['status'] = 'failed' if report['collection']['error'] else 'completed'
    except StopBudget as stopped:
        report.update(status='partial', stop_reason=str(stopped))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        for stream in streams:
            stream.close()
        report['budget'] = budget.report()
        report['elapsed_seconds'] = time.time() - started
        report['source_hashes_unchanged'] = report['source_sha256'] == _source_hashes()
        _write(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / '20260930-victory-native-teacher-pilot')
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--max-decisions', type=int, default=65536)
    parser.add_argument('--max-complete', type=int, default=64)
    parser.add_argument('--master-seed', type=int, default=20260930)
    parser.add_argument('--deadline', default='2026-09-30T21:01:44+08:00')
    args = vars(parser.parse_args(argv))
    args['deadline'] = datetime.fromisoformat(args['deadline']).timestamp()
    report = run(**args)
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'budget': report['budget'], 'collection': report['collection'],
                      'source_hashes_unchanged': report['source_hashes_unchanged']}), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
