"""Drive learning_lab.imitation.train_imitation over globbed demo stream dirs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True,
                        help='directories scanned recursively for stream datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--initial-checkpoint', type=Path, default=None)
    parser.add_argument('--max-seconds', type=float, default=None)
    args = parser.parse_args()
    datasets = []
    for root in args.roots:
        for metadata in sorted(Path(root).rglob('metadata.json')):
            datasets.append(metadata.parent)
    if not datasets:
        raise SystemExit('No datasets found')
    print(f'{len(datasets)} stream datasets', flush=True)
    import time
    from learning_lab.imitation import train_imitation
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    report = train_imitation(
        datasets, args.output, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, device=args.device,
        deadline=deadline, initial_checkpoint=args.initial_checkpoint)
    print(json.dumps({k: report[k] for k in
                      ('status', 'stop_reason', 'epochs_completed',
                       'supervised_samples', 'best_validation_ce')}, default=str))
    history = report.get('history', [])
    if history:
        print(json.dumps(history[-1]))


if __name__ == '__main__':
    main()
