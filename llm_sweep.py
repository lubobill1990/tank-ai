"""Run llm_player across stages in parallel worker processes."""
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
    command = [sys.executable, str(ROOT / 'llm_player.py'),
               '--stage', str(stage), '--episodes', str(args.episodes),
               '--output', str(output), '--base-seed', str(args.base_seed + stage),
               '--model', args.model, '--budget-usd', str(args.budget_usd),
               '--llm-every', str(args.llm_every), '--max-calls', str(args.max_calls)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        print(f'stage {stage:02d}: FAILED\n{proc.stderr[-1500:]}', flush=True)
        return stage, None
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    print(f"stage {stage:02d}: clear={summary['clears']}/{summary['episodes']} "
          f"cost=${summary['total_cost_usd']}", flush=True)
    return stage, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=91000)
    parser.add_argument('--model', default='claude-sonnet-5')
    parser.add_argument('--budget-usd', type=float, default=20.0)
    parser.add_argument('--llm-every', type=int, default=30)
    parser.add_argument('--max-calls', type=int, default=80)
    parser.add_argument('--workers', type=int, default=7)
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
    done = {s: r for s, r in results.items() if r}
    overall = {
        'model': args.model,
        'clears': sum(r['clears'] for r in done.values()),
        'episodes': sum(r['episodes'] for r in done.values()),
        'total_cost_usd': round(sum(r['total_cost_usd'] for r in done.values()), 2),
        'per_stage': {str(s): f"{r['clears']}/{r['episodes']}" for s, r in sorted(done.items())},
    }
    (args.output / 'sweep.json').write_text(json.dumps(overall, indent=2), encoding='utf-8')
    print(json.dumps(overall))


if __name__ == '__main__':
    main()
