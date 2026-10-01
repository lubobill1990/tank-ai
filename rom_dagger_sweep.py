"""Run rom_dagger across stages in parallel worker processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_stage(stage, args):
    output = Path(args.output) / f'stage-{stage:02d}'
    if (output / 'summary.json').exists():
        return stage, json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    command = [sys.executable, str(ROOT / 'rom_dagger.py'),
               '--stage', str(stage), '--episodes', str(args.episodes),
               '--output', str(output), '--base-seed', str(args.base_seed + stage * 100),
               '--depth', str(args.depth), '--plan-seeds', str(args.plan_seeds),
               '--student', str(args.student), '--beta', str(args.beta)]
    if args.guard:
        command.append('--guard')
    if args.max_seconds:
        command += ['--max-seconds', str(args.max_seconds)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        print(f'stage {stage:02d}: FAILED\n{proc.stderr[-1500:]}', flush=True)
        return stage, None
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    print(f"stage {stage:02d}: clear={summary['clears']}/{summary['episodes']} "
          f"planned={summary['planned']} rule_only={summary['rule_only']}", flush=True)
    return stage, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=60000)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--plan-seeds', type=int, default=3)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=None)
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
    clears = sum(r['clears'] for r in results.values() if r)
    total = sum(r['episodes'] for r in results.values() if r)
    print(json.dumps({'total': f'{clears}/{total}'}))


if __name__ == '__main__':
    main()
