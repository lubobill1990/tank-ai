"""Lossless supervised NON-LEARNED teacher corpus and two-frame minibatches.

Collect: ``uv run --no-sync python -m learning_lab.demonstrations``
Load: ``data = DemonstrationDataset(path); batch = data.batch([0, 1])``.
History channels are [previous47, current47], with zero previous at every reset.
Split by episode, never by frames; repeated clean openings are not independent.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from native_codec import NativeRunCodec


FRAME_SHAPE = (47, 26, 26)
FORMAT = "native-run-demonstrations-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "20260930-victory-demonstrations-v1"


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _action(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or not 0 <= value < 10:
        raise ValueError("Expected an integer primitive action in 0..9")
    return int(value)


class DemonstrationWriter:
    """Append independently encoded frames; finish publishes a loadable corpus."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.codec = NativeRunCodec(FRAME_SHAPE)
        self.stream = (self.path / "frames.bin").open("xb")
        self.offsets = [0]
        self.actions, self.executed, self.masks, self.starts, self.ids = [], [], [], [], []
        self.digest = hashlib.sha256()
        self.finished = False

    def add(self, observation47, teacher_action, action_masks, *, episode_start,
            executed_action):
        if self.finished or self.stream.closed:
            raise RuntimeError("Writer is closed or already finished")
        teacher_action, executed_action = _action(teacher_action), _action(executed_action)
        if not isinstance(action_masks, np.ndarray) or action_masks.dtype != np.bool_ \
                or action_masks.shape != (10,) or not action_masks.any():
            raise ValueError("Expected a nonempty boolean action mask of shape (10,)")
        if not action_masks[teacher_action] or not action_masks[executed_action]:
            raise ValueError("Both teacher and executed actions must be mask-valid")
        if not isinstance(episode_start, (bool, np.bool_)):
            raise ValueError("episode_start must be boolean")
        if not self.actions and not episode_start:
            raise ValueError("First observation must start an episode")
        payload = self.codec.encode(observation47)
        self.stream.write(payload)
        self.offsets.append(self.offsets[-1] + len(payload))
        self.actions.append(teacher_action)
        self.executed.append(executed_action)
        self.masks.append(action_masks.copy())
        self.starts.append(bool(episode_start))
        self.ids.append((self.ids[-1] if self.ids else -1) + int(episode_start))
        self.digest.update(observation47.tobytes(order="C"))
        if len(self.actions) % 128 == 0:
            self.stream.flush()

    def finish(self, episode_reports, metadata):
        if self.finished:
            raise RuntimeError("Writer is already finished")
        reports = deepcopy(list(episode_reports))
        episode_count = int(sum(self.starts))
        if len(reports) < episode_count:
            raise ValueError("Every stored episode needs an outcome report")
        for episode, report in enumerate(reports):
            if report.get("episode_id", episode) != episode or not report.get("outcome"):
                raise ValueError("Reports need ordered episode ids and nonempty outcomes")
            if len(str(report["outcome"])) > 64:
                raise ValueError("Outcome exceeds the 64-character storage limit")
            report["episode_id"] = episode
        self.close()
        arrays = {
            "offsets": np.asarray(self.offsets, np.uint64),
            "teacher_actions": np.asarray(self.actions, np.int64),
            "executed_actions": np.asarray(self.executed, np.int64),
            "action_masks": np.asarray(self.masks, bool).reshape(-1, 10),
            "episode_starts": np.asarray(self.starts, bool),
            "episode_ids": np.asarray(self.ids, np.int64),
            "outcomes": np.asarray([r["outcome"] for r in reports], dtype="<U64"),
        }
        for name, array in arrays.items():
            np.save(self.path / f"{name}.npy", array, allow_pickle=False)
        _json(self.path / "episode_reports.json", reports)
        final = deepcopy(metadata)
        final.update(format=FORMAT, frame_shape=list(FRAME_SHAPE), frame_dtype="float32",
                     history_order=["previous", "current"], samples=len(self.actions),
                     episodes=episode_count, reported_episodes=len(reports),
                     codec="NativeRunCodec bit-exact independent frame payloads",
                     raw_frame_sha256=self.digest.hexdigest(),
                     payload_bytes=self.offsets[-1])
        _json(self.path / "metadata.json", final)
        self.finished = True
        return deepcopy(final)

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()


class DemonstrationDataset:
    """Read-only compressed backing; batch returns fresh, caller-owned arrays."""

    def __init__(self, path):
        self.path = Path(path)
        self._metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        if self._metadata.get("format") != FORMAT \
                or self._metadata.get("frame_shape") != list(FRAME_SHAPE) \
                or self._metadata.get("history_order") != ["previous", "current"]:
            raise ValueError("Unsupported demonstration format or history layout")
        self._arrays = {}
        for name in ("offsets", "teacher_actions", "executed_actions", "action_masks",
                     "episode_starts", "episode_ids", "outcomes"):
            array = np.load(self.path / f"{name}.npy", allow_pickle=False)
            array.setflags(write=False)
            self._arrays[name] = array
        self._validate()
        size = (self.path / "frames.bin").stat().st_size
        self._payload = np.memmap(self.path / "frames.bin", mode="r", dtype=np.uint8) if size else None
        self.codec = NativeRunCodec(FRAME_SHAPE)
        self.closed = False

    def _validate(self):
        n = self._metadata["samples"]
        for name, shape, dtype in (
            ("offsets", (n + 1,), np.uint64), ("teacher_actions", (n,), np.int64),
            ("executed_actions", (n,), np.int64), ("action_masks", (n, 10), np.bool_),
            ("episode_starts", (n,), np.bool_), ("episode_ids", (n,), np.int64),
        ):
            array = self._arrays[name]
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Invalid {name} shape or dtype")
        offsets = self._arrays["offsets"]
        if offsets[0] != 0 or (n and np.any(offsets[1:] <= offsets[:-1])) \
                or int(offsets[-1]) != (self.path / "frames.bin").stat().st_size:
            raise ValueError("Payload offsets do not match the stream")
        starts, ids = self._arrays["episode_starts"], self._arrays["episode_ids"]
        if n and not starts[0]:
            raise ValueError("First sample must start an episode")
        if not np.array_equal(ids, np.cumsum(starts, dtype=np.int64) - 1):
            raise ValueError("Episode ids disagree with reset boundaries")
        if int(starts.sum()) != self._metadata["episodes"] \
                or len(self._arrays["outcomes"]) < int(starts.sum()):
            raise ValueError("Episode outcomes or count are incomplete")
        masks = self._arrays["action_masks"]
        for name in ("teacher_actions", "executed_actions"):
            actions = self._arrays[name]
            if np.any((actions < 0) | (actions >= 10)) \
                    or not np.all(masks[np.arange(n), actions]):
                raise ValueError(f"{name} contains mask-invalid labels")

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def episode_ids(self):
        return self._arrays["episode_ids"].view()

    def __len__(self):
        return int(self._metadata["samples"])

    def _frame(self, index):
        offsets = self._arrays["offsets"]
        return self.codec.decode(self._payload[int(offsets[index]):int(offsets[index + 1])].tobytes())

    def batch(self, indices):
        if self.closed:
            raise RuntimeError("Dataset is closed")
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("Batch indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Demonstration index is out of range")
        indices = indices.astype(np.int64, copy=False)
        observations = np.zeros((len(indices), 94, 26, 26), np.float32)
        for row, index in enumerate(indices):
            observations[row, 47:] = self._frame(index)
            if not self._arrays["episode_starts"][index]:
                observations[row, :47] = self._frame(index - 1)
        return {
            "observations": observations,
            "actions": self._arrays["teacher_actions"][indices].copy(),
            "action_masks": self._arrays["action_masks"][indices].copy(),
            "episode_ids": self._arrays["episode_ids"][indices].copy(),
        }

    def close(self):
        if not self.closed and self._payload is not None:
            self._payload._mmap.close()
        self.closed = True


def collect(path, *, max_episodes=20, max_decisions=32000, deadline):
    """Six clean episodes followed by legal-action perturbations; CPU/ROM only."""
    from battle_city_env import BattleCityEnv
    from learning_lab.teacher import TeacherController
    from rust_resources import ResourceMonitor

    if not 6 <= max_episodes <= 20 or not 1 <= max_decisions <= 32000:
        raise ValueError("Collection bounds are 6..20 episodes and 1..32000 decisions")
    writer = DemonstrationWriter(path)
    monitor = ResourceMonitor(writer.path / "resources.jsonl", device="cpu")
    reports, decisions, frames = [], 0, 0
    started = time.time()
    deadline = min(float(deadline), started + 45 * 60)
    metadata = {
        "label": "NON-LEARNED teacher demonstrations; no model trained",
        "reservation": "6dd8aee73bdd4311a0190dd7de6009d6",
        "stage": 1, "opening_seeds": [0, 1, 2], "clean_episodes": 6,
        "max_episodes": max_episodes, "max_decisions": max_decisions,
        "max_frames_per_episode": 12000, "frame_skip": 4,
        "perturbation_probability": .05, "perturbation_rng": "numpy.default_rng(173 + episode_id)",
        "perturbation_distribution": "uniform over legal primitive actions, including teacher action",
        "label_timing": "Teacher label and observation captured BEFORE the executed action",
        "provenance": "BattleCityEnv normal ROM reset; current GameState; no RAM edits or RNG oracle",
        "test_environment_decisions": 0, "native_environment_decisions": 0,
        "split_warning": "Episode-level development splits only; repeated clean openings duplicate trajectories. No generalization claim.",
        "started_at": started, "deadline": deadline,
    }
    files = ["battle_city.nes", "heuristic.py", "game_state.py", "battle_city_env.py",
             "native_codec.py", "learning_lab\\teacher.py", "learning_lab\\demonstrations.py"]
    metadata["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in files}
    _json(writer.path / "collection_config.json", metadata)
    env = None
    failure = None
    try:
        monitor.check("admission", admission_slots=1, force=True)
        env = BattleCityEnv(stage=1, objective="full", max_frames=12000, frame_skip=4)
        teacher = TeacherController()
        for episode in range(max_episodes):
            if decisions >= max_decisions or time.time() >= deadline:
                break
            episode_started = time.time()
            seed, perturbation = episode % 3, 0.0 if episode < 6 else .05
            rng = np.random.default_rng(173 + episode)
            count, attempts, changed, previous_frame = 0, 0, 0, 0
            info = dict(kills=0, deaths=0, frames=0, outcome="")
            error = None
            teacher.reset()
            try:
                observation, info = env.reset(seed=seed)
                while not info["outcome"]:
                    monitor.check("collection")
                    if decisions >= max_decisions or time.time() >= deadline:
                        info["outcome"] = "decision_budget" if decisions >= max_decisions else "deadline"
                        break
                    masks = env.action_masks()[:10].copy()
                    teacher_action = teacher.act(env.state, masks)
                    executed_action = teacher_action
                    if perturbation and rng.random() < perturbation:
                        attempts += 1
                        executed_action = int(rng.choice(np.flatnonzero(masks)))
                    changed += int(executed_action != teacher_action)
                    writer.add(observation, teacher_action, masks, episode_start=count == 0,
                               executed_action=executed_action)
                    count += 1
                    decisions += 1
                    observation, _, _, _, info = env.step(executed_action)
                    frames += info["frames"] - previous_frame
                    previous_frame = info["frames"]
                    if decisions % 128 == 0:
                        _json(writer.path / "progress.json",
                              dict(decisions=decisions, frames=frames, episode_id=episode,
                                   current_episode_decisions=count))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                info["outcome"] = "aborted"
            report = dict(episode_id=episode, seed=seed, stage=1, outcome=info["outcome"],
                          clear=info["outcome"] == "clear", kills=info["kills"],
                          deaths=info["deaths"], frames=info["frames"], decisions=count,
                          perturbation_probability=perturbation, perturbation_rng_seed=173 + episode,
                          perturbation_attempts=attempts, changed_actions=changed,
                          time_seconds=time.time() - episode_started,
                          cumulative_decisions=decisions, cumulative_frames=frames,
                          failure_cause=error or (None if info["outcome"] == "clear" else info["outcome"]))
            reports.append(report)
            with (writer.path / "episodes.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            if error:
                raise RuntimeError(error)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
        metadata.update(actual_decisions=decisions, gameplay_frames=frames,
                        clear_episodes=sum(r["clear"] for r in reports),
                        failed_episodes=sum(not r["clear"] for r in reports),
                        elapsed_seconds=time.time() - started, error=failure,
                        changed_actions=sum(r["changed_actions"] for r in reports))
        writer.finish(reports, metadata)
        _json(writer.path / "progress.json",
              dict(decisions=decisions, frames=frames, episodes=len(reports), finished=True))
    if failure:
        raise RuntimeError(failure)
    monitor.check("verification", force=True)
    dataset = DemonstrationDataset(writer.path)
    try:
        digest = hashlib.sha256()
        for first in range(0, len(dataset), 32):
            monitor.check("verification")
            if time.time() >= deadline:
                raise TimeoutError("Corpus verification deadline")
            batch = dataset.batch(np.arange(first, min(first + 32, len(dataset))))
            digest.update(batch["observations"][:, 47:].tobytes(order="C"))
        if digest.hexdigest() != dataset.metadata["raw_frame_sha256"]:
            raise AssertionError("Decoded corpus differs from source float32 frames")
        verification = dict(samples=len(dataset), decoded_all_frames_bit_exact=True,
                            all_teacher_and_executed_actions_mask_valid=True,
                            environment_decisions=decisions, native_environment_decisions=0,
                            synthetic_test_environment_decisions=0,
                            clear_episodes=metadata["clear_episodes"],
                            failed_episodes=metadata["failed_episodes"],
                            storage_bytes=sum(p.stat().st_size for p in writer.path.iterdir() if p.is_file()))
        _json(writer.path / "verification.json", verification)
        print(json.dumps(verification), flush=True)
    finally:
        dataset.close()
    monitor.check("complete", force=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=32000)
    parser.add_argument("--deadline", default="2026-09-30T14:09:12+08:00")
    args = parser.parse_args()
    deadline = min(datetime.fromisoformat(args.deadline).timestamp(),
                   datetime.fromisoformat("2026-09-30T21:01:44+08:00").timestamp())
    collect(args.output, max_episodes=args.max_episodes,
            max_decisions=args.max_decisions, deadline=deadline)


if __name__ == "__main__":
    main()
