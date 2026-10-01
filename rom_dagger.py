"""DAgger on the original ROM: student acts, Rust-planning teacher labels.

Each decision snapshots the ROM RAM, imports it into a seeded NativeBatch for
MPC lookahead scoring (TeacherControllerV2/GuardedTeacher proposal + veto), and
records the teacher label while executing the student's action on the ROM.
States whose RAM the native engine cannot represent are skipped for planning
and labeled by the rule teacher alone. Clears here are student outcomes under
ordinary ROM resets; no RAM is written.
"""
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
from mpc_teacher import CachedParser, GuardedTeacher, MpcVetoTeacher, load_student
from learning_lab.teacher_v2 import TeacherControllerV2


def collect_stage(stage, *, episodes, output, base_seed, depth, plan_seeds,
                  guard, student, student_device, beta, max_frames, deadline=None):
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from rust_train import ROMPrimitiveEnv
    planner = MpcVetoTeacher(depth=depth, plan_seeds=plan_seeds,
                             plan_rng_seed=base_seed * 1000003 + stage)
    teacher_class = GuardedTeacher if guard else TeacherControllerV2
    policy = load_student(student, student_device)
    rng = np.random.default_rng(base_seed * 7919 + stage)
    writer = DemonstrationWriter(Path(output))
    reports = []
    vec = VecFrameStack(DummyVecEnv([lambda: ROMPrimitiveEnv(stage, max_frames)]),
                        n_stack=2, channels_order='first')
    raw = vec.venv.envs[0].env  # BattleCityEnv
    planned = skipped = 0
    try:
        for episode in range(episodes):
            if deadline is not None and time.perf_counter() > deadline:
                break
            seed = base_seed + episode
            set_random_seed(seed)
            vec.seed(seed)
            obs = vec.reset()
            teacher = teacher_class()
            parser = CachedParser()
            episode_start = True
            decisions = 0
            for _ in range((max_frames + 3) // 4 + 4):
                ram = np.array(raw.nes.ram, dtype=np.uint8, copy=True)
                state = parser.parse(ram)
                masks = np.asarray(raw.action_masks()[:10], dtype=bool)
                proposal = teacher.act(state, masks)
                label = proposal
                try:
                    scores, unsafe = planner.plan(ram[None])
                    label = int(planner.choose(scores, unsafe, [masks], [proposal])[0])
                    planned += 1
                except Exception:
                    skipped += 1
                student_action, _ = policy.predict(
                    obs, deterministic=False,
                    action_masks=masks[None])
                run = label if rng.random() < beta else int(student_action[0])
                writer.add(obs[0, 47:].copy(), label, masks,
                           episode_start=episode_start, executed_action=run)
                episode_start = False
                decisions += 1
                obs, _, done, infos = vec.step(np.array([run]))
                if done[0]:
                    info = infos[0]
                    reports.append({'episode_id': len(reports),
                                    'outcome': info['outcome'], 'stage': stage,
                                    'frames': info['frames'], 'kills': info['kills'],
                                    'deaths': info['deaths'], 'decisions': decisions,
                                    'seed': seed})
                    break
            else:
                reports.append({'episode_id': len(reports), 'outcome': 'unfinished',
                                'stage': stage, 'decisions': decisions, 'seed': seed})
    finally:
        vec.close()
        while len(reports) < sum(writer.starts):
            reports.append({'episode_id': len(reports), 'outcome': 'incomplete',
                            'stage': stage})
        writer.finish(reports, {
            'collector': 'rom-dagger-v1', 'stage': stage, 'stream': 0,
            'base_seed': base_seed, 'depth': depth, 'plan_seeds': plan_seeds,
            'guard': guard, 'noise': 0.0, 'beta': beta, 'max_frames': max_frames,
            'student': str(student), 'planned_labels': planned,
            'rule_only_labels': skipped,
            'backend': 'original-rom-states-rust-planned-labels',
        })
    clears = sum(r['outcome'] == 'clear' for r in reports)
    summary = {'stage': stage, 'episodes': len(reports), 'clears': clears,
               'planned': planned, 'rule_only': skipped,
               'outcomes': {k: sum(r['outcome'] == k for r in reports)
                            for k in set(r['outcome'] for r in reports)}}
    (Path(output) / 'summary.json').write_text(json.dumps(summary, indent=2),
                                               encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--plan-seeds', type=int, default=3)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--student-device', default='cpu')
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--max-frames', type=int, default=12000)
    parser.add_argument('--max-seconds', type=float, default=None)
    args = parser.parse_args(argv)
    deadline = time.perf_counter() + args.max_seconds if args.max_seconds else None
    collect_stage(args.stage, episodes=args.episodes, output=args.output,
                  base_seed=args.base_seed, depth=args.depth,
                  plan_seeds=args.plan_seeds, guard=args.guard,
                  student=args.student, student_device=args.student_device,
                  beta=args.beta, max_frames=args.max_frames, deadline=deadline)


if __name__ == '__main__':
    main()
