"""Watch a fixed checkpoint, one run's latest policy, or the heuristic."""
import argparse
from pathlib import Path
import warnings

import numpy as np

from experiments import (
    EpisodePlayer, checkpoint_manifest, latest_checkpoint, load_checkpoint, make_env,
    runtime_signature, validate_outcome, validate_stages_seeds,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument("--model", type=Path)
    agent.add_argument("--run-dir", type=Path, help="follow this run at episode boundaries")
    agent.add_argument("--heuristic", action="store_true")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nstack", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampled", action="store_true", help="sample instead of deterministic model actions")
    parser.add_argument("--fps", type=int, default=15, help="policy decisions per second")
    parser.add_argument("--scale", type=int, default=3)
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--record-dir", type=Path, help="best ten MP4s; default: run/recordings")
    recording.add_argument("--no-record", action="store_true", help="disable episode recording")
    args = parser.parse_args(argv)
    try:
        validate_stages_seeds((args.stage,), (args.seed,))
        if args.fps <= 0 or args.scale <= 0:
            raise ValueError("fps and scale must be positive")
        if args.nstack is not None and args.nstack <= 0:
            raise ValueError("nstack must be positive")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("max-frames must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    return args


class RunFollower:
    def __init__(self, run, signature, checkpoint, *, device, manifest=None):
        self.run, self.signature = Path(run).resolve(), signature
        self.checkpoint, self.device = Path(checkpoint).resolve(), device
        self.manifest = manifest

    def refresh(self, player):
        if not player.done:
            return False
        checkpoint = latest_checkpoint(self.run)
        if checkpoint == self.checkpoint:
            return False
        try:
            model, manifest = load_checkpoint(checkpoint, self.signature, device=self.device)
        except (FileNotFoundError, ValueError):
            missing = not checkpoint.is_file() or not checkpoint.with_suffix(".zip.manifest.json").is_file()
            if not missing or latest_checkpoint(self.run) == checkpoint:
                raise
            warnings.warn("Latest checkpoint rotated during load; retrying next episode", RuntimeWarning)
            return False
        player.model = model
        self.checkpoint = checkpoint
        self.manifest = manifest
        print(f"watch switched: {checkpoint.name} steps={model.num_timesteps}", flush=True)
        return True


def main(argv=None):
    args = parse_args(argv)
    checkpoint = latest_checkpoint(args.run_dir) if args.run_dir else args.model
    manifest = checkpoint_manifest(checkpoint) if checkpoint else None
    nstack = args.nstack if args.nstack is not None else (manifest["training"]["nstack"] if manifest else 2)
    max_frames = args.max_frames if args.max_frames is not None else (manifest["training"]["max_frames"] if manifest else 12000)
    if manifest and max_frames != manifest["training"]["max_frames"]:
        raise ValueError("watch max-frames must match the checkpoint episode rules")
    raw = make_env(stage=args.stage, objective="full", max_frames=max_frames)
    recorder = None
    try:
        model = None
        signature = runtime_signature(raw, nstack=nstack)
        if manifest:
            model, _ = load_checkpoint(checkpoint, signature, device=args.device)
        import pyglet
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(args.seed)
        player = EpisodePlayer(raw, nstack=nstack, model=model, deterministic=not args.sampled)
        follower = RunFollower(args.run_dir, signature, checkpoint,
                               device=args.device, manifest=manifest) \
            if args.run_dir else None
        player.reset(args.seed)
        frame = raw.render()
        height, width = frame.shape[:2]
        episode = 0
        if not args.no_record:
            from recordings import BestRecordings
            default_run = args.run_dir or (checkpoint.parent if checkpoint else Path("runs") / "heuristic")
            if not args.run_dir and checkpoint and checkpoint.parent.name == "checkpoints":
                default_run = checkpoint.parent.parent
            record_dir = args.record_dir or default_run / "recordings"
            recorder = BestRecordings(record_dir, fps=60 / raw.frame_skip)
            print(f"recording best 10 episodes: {Path(record_dir).resolve()}", flush=True)

        def recording_metadata():
            current = follower.manifest if follower else manifest
            path = follower.checkpoint if follower else checkpoint
            return dict(stage=args.stage, seed=(args.seed + episode) % 2**32,
                        agent="model" if player.model is not None else "heuristic",
                        model_steps=int(player.model.num_timesteps) if player.model is not None else 0,
                        checkpoint=str(path.resolve()) if path else None,
                        checkpoint_sha256=current["checkpoint_sha256"] if current else None,
                        sampled=args.sampled, runtime=signature)

        if recorder:
            recorder.begin(frame, recording_metadata())

        def model_label():
            if player.model is None:
                return "heuristic"
            name = args.run_dir.name if args.run_dir else checkpoint.name
            return f"{name} | steps {player.model.num_timesteps}"

        window = pyglet.window.Window(width * args.scale, height * args.scale,
                                      caption="Tank AI - " + model_label())
        label = pyglet.text.Label("", x=8, y=window.height - 8, anchor_y="top",
                                  multiline=True, width=window.width - 16,
                                  color=(255, 255, 255, 255))
        print(f"watch started: {model_label()} stage={args.stage}", flush=True)
        terminal_pause = 0

        @window.event
        def on_draw():
            window.clear()
            image = pyglet.image.ImageData(
                width, height, "RGB", np.ascontiguousarray(frame).tobytes(), pitch=-width * 3,
            )
            image.blit(0, 0, width=window.width, height=window.height)
            label.draw()

        def update(_dt):
            nonlocal frame, episode, terminal_pause
            if player.done:
                if terminal_pause:
                    terminal_pause -= 1
                    return
                if follower and follower.refresh(player):
                    window.set_caption("Tank AI - " + model_label())
                episode += 1
                set_random_seed((args.seed + episode) % 2**32)
                player.reset((args.seed + episode) % 2**32)
                if recorder:
                    recorder.begin(raw.render(), recording_metadata())
            _, done, info = player.step()
            frame = raw.render()
            if recorder:
                recorder.write(frame)
            label.text = (f"{model_label()}\nstage {args.stage} episode {episode + 1} kills {info['kills']} "
                          f"lives {info['lives']} {info['outcome']}")
            if done:
                validate_outcome(info)
                if recorder:
                    saved = recorder.finish(info)
                    if saved:
                        print(f"recording retained: {saved}", flush=True)
                print({key: info[key] for key in (
                    "stage", "objective", "outcome", "kills", "shots", "deaths", "pickups", "frames",
                )}, flush=True)
                terminal_pause = args.fps

        pyglet.clock.schedule_interval(update, 1.0 / args.fps)
        try:
            pyglet.app.run()
        finally:
            pyglet.clock.unschedule(update)
            window.close()
    finally:
        try:
            if recorder:
                recorder.close()
        finally:
            raw.close()


if __name__ == "__main__":
    main()
