"""Run mpc_teacher collection across stages in parallel worker processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def run_stage(stage, args):
    output = Path(args.output) / f'stage-{stage:02d}'
    if (output / 'summary.json').exists():
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        print(f'stage {stage:02d}: cached clear_rate={summary["clear_rate"]:.2f}', flush=True)
        return stage, summary
    command = [sys.executable, str(ROOT / 'mpc_teacher.py'),
               '--stage', str(stage), '--episodes', str(args.episodes),
               '--envs', str(args.envs), '--output', str(output),
               '--base-seed', str(args.base_seed + stage),
               '--depth', str(args.depth), '--plan-seeds', str(args.plan_seeds),
               '--noise', str(args.noise), '--threat-weight', str(args.threat_weight)]
    if args.max_seconds:
        command += ['--max-seconds', str(args.max_seconds)]
    if args.guard:
        command.append('--guard')
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        print(f'stage {stage:02d}: FAILED\n{proc.stderr[-2000:]}', flush=True)
        return stage, None
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    print(f'stage {stage:02d}: clear={summary["clears"]}/{summary["episodes"]} '
          f'kills={summary["kills_mean"]:.1f} {time.perf_counter()-start:.0f}s '
          f'{summary["outcome_counts"]}', flush=True)
    return stage, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=100)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--plan-seeds', type=int, default=3)
    parser.add_argument('--noise', type=float, default=0.02)
    parser.add_argument('--threat-weight', type=float, default=6.0)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--guard', action='store_true')
    args = parser.parse_args()
    stages = []
    for part in args.stages.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for stage, summary in pool.map(lambda s: run_stage(s, args), stages):
            results[stage] = summary
    overall = {
        'stages': {str(s): (results[s] if results[s] is None else
                            {k: results[s][k] for k in ('episodes', 'clears', 'clear_rate',
                                                        'kills_mean', 'outcome_counts')})
                   for s in stages},
        'total_clears': sum(r['clears'] for r in results.values() if r),
        'total_episodes': sum(r['episodes'] for r in results.values() if r),
    }
    (args.output / 'sweep.json').write_text(json.dumps(overall, indent=2), encoding='utf-8')
    print(json.dumps({'total': f"{overall['total_clears']}/{overall['total_episodes']}"}))


if __name__ == '__main__':
    main()
