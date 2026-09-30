"""MPC-veto teacher: TeacherControllerV2 proposes, batched Rust lookahead vetoes.

For each live collector environment and each of the ten primitive actions, the
planner clones the current RAM into a fresh seeded planning batch, executes the
candidate first, continues with masked random actions, and scores the rollout by
the engine's own discounted sparse reward plus an end-state base-threat penalty.
Actions whose rollouts ever die or lose the base are vetoed when a safe
alternative exists; among safe actions the teacher's proposal is preferred.

Planning rollouts use independent SplitMix64 streams re-seeded per decision;
they approximate, never reproduce, the collector environment's future. Clears
recorded here are Rust-engine outcomes, not original-ROM wins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_state as gs
from learning_lab.demonstrations import DemonstrationWriter
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_env import NativeBatch, episode_seed, stage_ram

BASE_X, BASE_Y = 120, 216
MAX_BASE_DIST = 272.0


class GuardedTeacher(TeacherControllerV2):
    """V2 with earlier base defense: guard when threats outpace the player.

    The stock HeuristicBot only switches to protect inside an 80px Manhattan
    radius of the base. When the closest enemy is nearer to the base than the
    player is (plus a margin), that switch comes too late to travel back, so
    this proposer asks the protect advisor first in that situation. All V2
    safety filters still apply to the substituted proposal.
    """

    GUARD_RADIUS = 136
    MARGIN = 24

    def act(self, state, masks):
        proposal = super().act(state, masks)
        player = state.player
        if player is None or state.game_over or not state.enemies:
            return proposal
        threat = min(abs(e.x - BASE_X) + abs(e.y - BASE_Y) for e in state.enemies)
        mine = abs(player.x - BASE_X) + abs(player.y - BASE_Y)
        if threat < self.GUARD_RADIUS and threat < mine + self.MARGIN:
            guard = self.bot.advise('protect', state)
            if guard and masks[guard] and not self._base_fire(state, guard):
                if self.mode in ('counter', 'safety', 'stand_off'):
                    return proposal
                self.mode = 'guard'
                return int(guard)
        return proposal


def outcome_of(ram):
    if ram[0x68] != 0x80:
        return 'base_lost'
    if ram[0x51] == 0 or (ram[0x51] <= 1 and 0x10 <= ram[0xa0] < 0x80):
        return 'lives_exhausted'
    if ram[0x80] == 0:
        return 'clear'
    return 'timeout'


def base_threat(ram):
    """Closest active enemy's normalized proximity to the base, 0 when none."""
    threat = 0.0
    for slot in range(2, 8):
        if 0x80 <= ram[0xa0 + slot] < 0xe0:
            dist = abs(int(ram[0x90 + slot]) - BASE_X) + abs(int(ram[0x98 + slot]) - BASE_Y)
            threat = max(threat, 1.0 - dist / MAX_BASE_DIST)
    return threat


class CachedParser:
    """Reuse decoded terrain while the terrain bytes are unchanged."""

    def __init__(self):
        self._bytes = None
        self._terrain = None
        self._masks = None

    def parse(self, ram):
        terrain_bytes = ram[0x400:0x800].tobytes()
        if terrain_bytes != self._bytes:
            self._bytes = terrain_bytes
            self._terrain = self._masks = None
        state = gs.parse(ram, terrain=self._terrain, brick_masks=self._masks)
        self._terrain, self._masks = state.terrain, state.brick_masks
        return state


class MpcVetoTeacher:
    """Score all ten first actions with seeded lookahead rollouts in one batch."""

    def __init__(self, *, depth=12, plan_seeds=3, discount=0.96,
                 threat_weight=6.0, safety_margin=0.5, plan_rng_seed=0):
        self.depth = depth
        self.plan_seeds = plan_seeds
        self.discount = discount
        self.threat_weight = threat_weight
        self.safety_margin = safety_margin
        self.rng = np.random.default_rng(plan_rng_seed)

    def plan(self, rams):
        """rams: (L, 2048) uint8 live snapshots -> (scores (L,10), unsafe (L,10))."""
        L = len(rams)
        A, S = 10, self.plan_seeds
        plan_init = np.repeat(np.asarray(rams, np.uint8), A * S, axis=0)
        total = L * A * S
        seeds = [int(self.rng.integers(2**63)) for _ in range(total)]
        plan = NativeBatch(plan_init, 1 << 20, seeds=seeds)
        try:
            first = np.tile(np.repeat(np.arange(A), S), L)
            legal_first = plan.masks()[np.arange(total), first].copy()
            acts0 = np.where(legal_first, first, 0)
            score = np.zeros(total, np.float32)
            failed = np.zeros(total, bool)
            alive = np.ones(total, bool)
            prev = np.zeros((total, 5), np.float32)
            factor = 1.0
            for d in range(self.depth):
                if d == 0:
                    acts = acts0
                else:
                    noise = self.rng.random((total, A)).astype(np.float32)
                    acts = np.argmax(noise * plan.masks(), axis=1)
                _, rewards, dones, stats = plan.step(acts)
                sp = stats.astype(np.float32)
                deaths = sp[:, 2] - prev[:, 2]
                score += factor * alive * rewards
                failed |= alive & ((deaths > 0) | (rewards <= -19.0))
                alive &= ~dones
                prev = sp
                factor *= self.discount
            threat = np.zeros(total, np.float32)
            for j in np.flatnonzero(alive):
                threat[j] = base_threat(plan.ram(int(j)))
            score -= self.threat_weight * factor * threat
        finally:
            plan.close()
        score = np.where(legal_first, score, np.float32(-1e9))
        score = score.reshape(L, A, S).mean(axis=2)
        unsafe = (failed | ~legal_first).reshape(L, A, S).any(axis=2)
        return score, unsafe

    def choose(self, scores, unsafe, masks, proposals):
        """Prefer the proposal among safe near-best actions; veto deadly ones."""
        actions = np.zeros(len(scores), np.int64)
        for i, (score, bad, mask, proposal) in enumerate(zip(scores, unsafe, masks, proposals)):
            legal_scores = np.where(mask, score, -1e9)
            best = float(legal_scores.max())
            safe = mask & ~bad & (score >= best - self.safety_margin)
            if safe.any():
                actions[i] = proposal if safe[proposal] else int(np.argmax(np.where(safe, score, -1e9)))
            else:
                actions[i] = int(np.argmax(legal_scores))
        return actions


def load_student(path, device):
    """Student policy for DAgger-style execution; teacher still labels."""
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(path, device=device,
                            custom_objects={'n_steps': 2, 'n_envs': 1, 'batch_size': 2})


def collect(stage, *, episodes, envs, output, base_seed, depth, plan_seeds,
            noise, max_frames, deadline=None, threat_weight=6.0, quiet=False,
            guard=False, student=None, student_device='cpu', beta=0.0):
    """Run the MPC teacher on one stage; each env appends to its own stream."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ram0 = stage_ram(stage, seed=0)
    initial = np.stack([ram0] * envs)
    batch = NativeBatch(initial, max_frames,
                        seeds=[episode_seed(base_seed, i, 0) for i in range(envs)])
    batch.observe()  # np.empty buffer holds garbage until the first native fill
    planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                             threat_weight=threat_weight,
                             plan_rng_seed=base_seed * 1000003 + stage)
    noise_rng = np.random.default_rng(base_seed * 9176 + stage)
    teacher_class = GuardedTeacher if guard else TeacherControllerV2
    teachers = [teacher_class() for _ in range(envs)]
    policy = load_student(student, student_device) if student else None
    previous_obs = np.zeros((envs, 47, 26, 26), np.float32) if policy is not None else None
    parsers = [CachedParser() for _ in range(envs)]
    writers, reports, counters = [], [], []
    for i in range(envs):
        writers.append(DemonstrationWriter(output / f'stream-{i:02d}'))
        reports.append([])
        counters.append(0)
    episode_index = [0] * envs
    episode_active = [True] * envs
    pending_start = [True] * envs
    finished_episodes = 0
    t0 = time.perf_counter()
    try:
        while finished_episodes < episodes:
            if deadline is not None and time.perf_counter() > deadline:
                break
            live = [i for i in range(envs) if episode_active[i]]
            if not live:
                break
            rams = [batch.ram(i) for i in live]
            states = [parsers[i].parse(rams[k]) for k, i in enumerate(live)]
            col_masks = batch.masks().copy()
            proposals = [teachers[i].act(states[k], col_masks[i]) for k, i in enumerate(live)]
            scores, unsafe = planner.plan(np.stack(rams))
            chosen = planner.choose(scores, unsafe, [col_masks[i] for i in live], proposals)
            obs_now = batch.obs.copy()
            student_actions = None
            if policy is not None:
                stacked = np.concatenate([previous_obs, obs_now], axis=1)
                sampled, _ = policy.predict(stacked, deterministic=False,
                                            action_masks=col_masks)
                student_actions = sampled
                previous_obs[list(live)] = obs_now[list(live)]
            actions = np.zeros(envs, np.int64)
            executed = {}
            for k, i in enumerate(live):
                act = int(chosen[k])
                run = act
                if student_actions is not None and noise_rng.random() >= beta:
                    run = int(student_actions[i])
                elif noise > 0 and noise_rng.random() < noise:
                    options = np.flatnonzero(col_masks[i] & ~unsafe[k])
                    if len(options) > 1:
                        run = int(noise_rng.choice(options))
                actions[i] = run
                executed[i] = (act, run)
            obs_before = obs_now
            obs, rewards, dones, stats = batch.step(actions)
            for k, i in enumerate(live):
                teach, run = executed[i]
                writers[i].add(obs_before[i], teach, col_masks[i],
                               episode_start=pending_start[i], executed_action=run)
                pending_start[i] = False
                counters[i] += 1
                if dones[i]:
                    ram = batch.ram(i)
                    frame, kills, deaths, shots, pickups = map(int, stats[i])
                    reports[i].append({
                        'episode_id': len(reports[i]), 'outcome': outcome_of(ram),
                        'stage': stage, 'frames': frame, 'kills': kills,
                        'deaths': deaths, 'shots': shots, 'pickups': pickups,
                        'decisions': counters[i],
                        'episode_seed': episode_seed(base_seed, i, episode_index[i]),
                    })
                    counters[i] = 0
                    finished_episodes += 1
                    episode_index[i] += 1
                    remaining = episodes - finished_episodes
                    if remaining >= sum(episode_active):
                        batch.reset(i, seed=episode_seed(base_seed, i, episode_index[i]))
                        teachers[i].reset()
                        if previous_obs is not None:
                            previous_obs[i] = 0
                        pending_start[i] = True
                    else:
                        episode_active[i] = False
            if not quiet and finished_episodes and finished_episodes % 8 == 0:
                pass
    finally:
        batch.close()
        summaries = []
        for i in range(envs):
            if counters[i] > 0:
                reports[i].append({'episode_id': len(reports[i]), 'outcome': 'incomplete',
                                   'stage': stage, 'decisions': counters[i]})
            meta = writers[i].finish(reports[i], {
                'collector': 'mpc-veto-teacher-v1', 'stage': stage, 'stream': i,
                'guard': guard,
                'base_seed': base_seed, 'depth': depth, 'plan_seeds': plan_seeds,
                'noise': noise, 'max_frames': max_frames,
                'threat_weight': threat_weight,
                'backend': 'rust-seeded-experimental-v1',
                'note': 'Rust-engine outcomes; not original-ROM wins.',
            })
            summaries.append({'stream': i, 'samples': meta['samples'],
                              'episodes': meta['episodes'],
                              'outcomes': [r['outcome'] for r in reports[i]]})
        elapsed = time.perf_counter() - t0
        all_reports = [r for stream in reports for r in stream]
        clears = sum(r['outcome'] == 'clear' for r in all_reports)
        total = sum(r['outcome'] != 'incomplete' for r in all_reports)
        summary = {
            'stage': stage, 'episodes': total, 'clears': clears,
            'clear_rate': clears / total if total else 0.0,
            'elapsed_seconds': elapsed,
            'outcome_counts': {k: sum(r['outcome'] == k for r in all_reports)
                               for k in set(r['outcome'] for r in all_reports)},
            'kills_mean': float(np.mean([r['kills'] for r in all_reports
                                         if 'kills' in r])) if total else 0.0,
            'streams': summaries,
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2),
                                             encoding='utf-8')
        if not quiet:
            print(json.dumps({k: summary[k] for k in
                              ('stage', 'episodes', 'clears', 'clear_rate',
                               'elapsed_seconds', 'outcome_counts', 'kills_mean')}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--plan-seeds', type=int, default=3)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--threat-weight', type=float, default=6.0)
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--student', type=Path, default=None)
    parser.add_argument('--student-device', default='cpu')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='probability of executing the teacher action in DAgger mode')
    args = parser.parse_args(argv)
    deadline = time.perf_counter() + args.max_seconds if args.max_seconds else None
    collect(args.stage, episodes=args.episodes, envs=args.envs, output=args.output,
            base_seed=args.base_seed, depth=args.depth, plan_seeds=args.plan_seeds,
            noise=args.noise, max_frames=args.max_frames, deadline=deadline,
            threat_weight=args.threat_weight, guard=args.guard, student=args.student,
            student_device=args.student_device, beta=args.beta)


if __name__ == '__main__':
    main()
