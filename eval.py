"""Fixed full-stage deterministic AND sampled evaluation, written as JSON."""
import argparse
from pathlib import Path

from experiments import (
    DEFAULT_BENCHMARK_SEEDS, DEFAULT_BENCHMARK_STAGES, checkpoint_manifest,
    evaluate, integer_list, load_checkpoint, make_env, runtime_signature,
    validate_stages_seeds, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument("--model", type=Path, help="explicit compatible .zip checkpoint")
    agent.add_argument("--heuristic", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; refuses overwrite")
    parser.add_argument("--stages", type=integer_list, help="default: manifest held-out benchmark stages")
    parser.add_argument("--seeds", type=integer_list, help="default: manifest held-out benchmark seeds")
    parser.add_argument("--max-frames", type=int, help="must match the checkpoint's episode rules")
    parser.add_argument("--nstack", type=int, help="must match the checkpoint; heuristic default 2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        validate_stages_seeds(args.stages or DEFAULT_BENCHMARK_STAGES, args.seeds or DEFAULT_BENCHMARK_SEEDS)
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("max-frames must be positive")
        if args.nstack is not None and args.nstack <= 0:
            raise ValueError("nstack must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    manifest = checkpoint_manifest(args.model) if args.model else None
    stages = args.stages or (tuple(manifest["benchmark"]["stages"]) if manifest else DEFAULT_BENCHMARK_STAGES)
    seeds = args.seeds or (tuple(manifest["benchmark"]["seeds"]) if manifest else DEFAULT_BENCHMARK_SEEDS)
    nstack = args.nstack if args.nstack is not None else (manifest["training"]["nstack"] if manifest else 2)
    max_frames = args.max_frames if args.max_frames is not None else (manifest["training"]["max_frames"] if manifest else 12000)
    if manifest and max_frames != manifest["training"]["max_frames"]:
        raise ValueError("evaluation max-frames must match the checkpoint episode rules")
    raw = make_env(stage=stages[0], max_frames=max_frames)
    try:
        signature = runtime_signature(raw, nstack=nstack)
    finally:
        raw.close()
    model = None
    if manifest:
        model, manifest = load_checkpoint(args.model, signature, device=args.device)
    report = evaluate(model=model, stages=stages, seeds=seeds, nstack=nstack, max_frames=max_frames)
    configured = (tuple(manifest["benchmark"]["stages"]), tuple(manifest["benchmark"]["seeds"])) if manifest else (
        DEFAULT_BENCHMARK_STAGES, DEFAULT_BENCHMARK_SEEDS,
    )
    report.update(
        role="held_out_benchmark" if (stages, seeds) == configured else "custom_full_stage_evaluation",
        runtime=signature, checkpoint=str(args.model.resolve()) if args.model else None,
        checkpoint_sha256=manifest["checkpoint_sha256"] if manifest else None,
    )
    write_json(args.output, report)
    for mode, result in report["modes"].items():
        print(f"{mode}: {result['summary']}")
    print(f"evaluation written: {args.output}")


if __name__ == "__main__":
    main()
