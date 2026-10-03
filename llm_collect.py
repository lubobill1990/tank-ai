"""Parallel LLM-commander ROM demonstration collection across stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_stage(stage, args):
    output = Path(args.output) / f'stage-{stage:02d}'
    marker = output / 'done.json'
    if marker.exists():
        return stage, json.loads(marker.read_text(encoding='utf-8'))
    command = [sys.executable, str(ROOT / 'llm_rom_player.py'),
               '--stages', str(stage), '--episodes', str(args.episodes),
               '--output', str(output), '--base-seed', str(args.base_seed),
               '--model', args.model, '--budget-usd', str(args.budget_usd),
               '--llm-every', str(args.llm_every), '--max-calls', str(args.max_calls),
               '--initial-mode', args.initial_mode, '--record-demos']
    if args.no_video:
        command.append('--no-video')
    if args.guard_bot:
        command.append('--guard-bot')
    if args.controller:
        command += ['--controller', args.controller]
    if args.predict_guard:
        command.append('--predict-guard')
    if args.intercept_fire:
        command.append('--intercept-fire')
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        print(f'stage {stage:02d}: FAILED\n{proc.stderr[-1200:]}', flush=True)
        return stage, None
    results = [json.loads(line) for line in
               (output / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    summary = {'stage': stage,
               'clears': sum(r['outcome'] == 'clear' for r in results),
               'episodes': len(results),
               'cost_usd': round(sum(r['cost_usd'] for r in results), 3)}
    marker.write_text(json.dumps(summary), encoding='utf-8')
    print(f"stage {stage:02d}: clear={summary['clears']}/{summary['episodes']} "
          f"cost=${summary['cost_usd']}", flush=True)
    return stage, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', default='1-35')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-seed', type=int, default=200000)
    parser.add_argument('--model', default='claude-sonnet-5')
    parser.add_argument('--budget-usd', type=float, default=20.0)
    parser.add_argument('--llm-every', type=int, default=20)
    parser.add_argument('--max-calls', type=int, default=100)
    parser.add_argument('--initial-mode', default='protect')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--guard-bot', action='store_true')
    parser.add_argument('--controller', default='')
    parser.add_argument('--predict-guard', action='store_true')
    parser.add_argument('--intercept-fire', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
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
    overall = {'model': args.model,
               'clears': sum(r['clears'] for r in done.values()),
               'episodes': sum(r['episodes'] for r in done.values()),
               'cost_usd': round(sum(r['cost_usd'] for r in done.values()), 2),
               'per_stage': {str(s): f"{r['clears']}/{r['episodes']}"
                             for s, r in sorted(done.items())}}
    (args.output / 'sweep.json').write_text(json.dumps(overall, indent=2),
                                            encoding='utf-8')
    print(json.dumps(overall))


if __name__ == '__main__':
    main()
