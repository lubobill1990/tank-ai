"""Versioned lossless55-plane demonstrations plus audit-only raw RAM/frame streams.

PrecisionDataset.batch returns ONLY observations/actions/masks/episode_ids.
PrecisionDataset.audit separately exposes owned RAM/frame copies for observation
audits, never learner inputs or reset-state initialization. No old47 format changes.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from native_codec import NativeRunCodec
from learning_lab.demonstrations import _action
from learning_lab.teacher_v2 import TeacherControllerV2
from rust_resources import ResourceMonitor
from within_stage.observation import (
    PrecisionObservation, OBSERVATION_VERSION, PRECISION_CHANNEL_NAMES,
)
from within_stage.openings import ProbeAccounting, ProbeStop
from within_stage.opening_pool import load_pool, make_opening_env

ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = ROOT / "runs" / "20260930-within-stage"
FORMAT = "within-stage-precision-demonstrations-v1"
FRAME_SHAPE = (55, 26, 26)
SPLIT_SHA256 = "a94de5882737587e06930d6d782d512228f3248958df26f188fc2b494816eebe"
LOCAL_DEADLINE = datetime.fromisoformat("2026-09-30T22:44:12+08:00").timestamp()
ARRAYS = {
    "offsets": np.uint64, "teacher_actions": np.int64, "executed_actions": np.int64,
    "action_masks": np.bool_, "episode_starts": np.bool_, "episode_ids": np.int64,
    "frame_numbers": np.int64,
}
DATA_FILES = ("observations.bin", "ram.bin", "metadata.json", "episode_reports.json",
              *(f"{key}.npy" for key in ARRAYS))


def schema():
    return dict(observation_version=OBSERVATION_VERSION, channels=list(PRECISION_CHANNEL_NAMES),
                frame_shape=[55, 26, 26], frame_dtype="float32", batch_shape=[110, 26, 26],
                history_order=["previous", "current"], zero_prefix_at_episode_start=True,
                actions=10, ram_shape=[2048], ram_dtype="uint8", frame_number_dtype="int64")


def _schema_hash():
    return hashlib.sha256(json.dumps(schema(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def _sources():
    return {name: _sha(ROOT / name) for name in (
        "within_stage\\demonstrations.py", "within_stage\\observation.py",
        "native_codec.py", "game_state.py", "battle_city_env.py",
        "native\\target\\release\\battle_mirror_probe.dll")}


def verify_integrity(path):
    path = Path(path)
    manifest = json.loads((path / "integrity.json").read_text())
    entries = manifest.get("files", [])
    if manifest.get("format") != FORMAT or {r["path"] for r in entries} != set(DATA_FILES):
        raise ValueError("Precision corpus integrity/schema file list mismatch")
    for row in entries:
        if Path(row["path"]).name != row["path"]:
            raise ValueError("Invalid precision corpus integrity path")
        file = path / row["path"]
        if file.stat().st_size != row["bytes"] or _sha(file) != row["sha256"]:
            raise ValueError(f"Precision corpus hash mismatch: {row['path']}")
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("fingerprint_sha256"):
        raise ValueError("Precision corpus fingerprint mismatch")
    return digest


class PrecisionWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.codec = NativeRunCodec(FRAME_SHAPE)
        self.observations = (self.path / "observations.bin").open("xb")
        self.ram_stream = (self.path / "ram.bin").open("xb")
        self.offsets = [0]
        self.actions, self.executed, self.masks = [], [], []
        self.starts, self.ids, self.frames = [], [], []
        self.observation_digest, self.ram_digest = hashlib.sha256(), hashlib.sha256()
        self.finished = False

    def add(self, observation55, teacher_action, masks, *, ram, frame, episode_start, executed_action):
        if self.finished or self.observations.closed or self.ram_stream.closed:
            raise RuntimeError("Precision writer is closed/finished")
        teacher_action, executed_action = _action(teacher_action), _action(executed_action)
        if not isinstance(masks, np.ndarray) or masks.shape != (10,) or masks.dtype != np.bool_ \
                or not masks.any() or not masks[teacher_action] or not masks[executed_action]:
            raise ValueError("Teacher/executed action must be legal under ten boolean masks")
        if not isinstance(ram, np.ndarray) or ram.shape != (2048,) or ram.dtype != np.uint8:
            raise ValueError("Audit RAM must be uint8(2048,)")
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)) or frame < 0:
            raise ValueError("Audit frame must be a nonnegative integer")
        if not isinstance(episode_start, (bool, np.bool_)) or (not self.actions and not episode_start):
            raise ValueError("First sample must have a boolean episode-start boundary")
        if self.frames and not episode_start and frame <= self.frames[-1]:
            raise ValueError("Frame numbers must increase within an episode")
        payload = self.codec.encode(observation55)
        raw_ram = ram.tobytes(order="C")
        self.observations.write(payload)
        self.ram_stream.write(raw_ram)
        self.offsets.append(self.offsets[-1] + len(payload))
        self.actions.append(teacher_action)
        self.executed.append(executed_action)
        self.masks.append(masks.copy())
        self.starts.append(bool(episode_start))
        self.ids.append((self.ids[-1] if self.ids else -1) + int(episode_start))
        self.frames.append(int(frame))
        self.observation_digest.update(observation55.tobytes(order="C"))
        self.ram_digest.update(raw_ram)
        if len(self.actions) % 128 == 0:
            self.observations.flush()
            self.ram_stream.flush()

    def finish(self, reports, metadata):
        if self.finished:
            raise RuntimeError("Precision corpus already published")
        reports = deepcopy(list(reports))
        if len(reports) < sum(self.starts):
            raise ValueError("Every stored episode requires an outcome report")
        counts = Counter(self.ids)
        for index, row in enumerate(reports):
            if row.get("episode_id", index) != index or not row.get("outcome"):
                raise ValueError("Reports must contain ordered episode IDs and outcomes")
            if "decisions" in row and row["decisions"] != counts[index]:
                raise ValueError("Episode report length differs from stored samples")
            row["episode_id"] = index
        self.close()
        arrays = dict(
            offsets=np.asarray(self.offsets, np.uint64),
            teacher_actions=np.asarray(self.actions, np.int64),
            executed_actions=np.asarray(self.executed, np.int64),
            action_masks=np.asarray(self.masks, bool).reshape(-1, 10),
            episode_starts=np.asarray(self.starts, bool), episode_ids=np.asarray(self.ids, np.int64),
            frame_numbers=np.asarray(self.frames, np.int64))
        for name, values in arrays.items():
            np.save(self.path / f"{name}.npy", values, allow_pickle=False)
        _write(self.path / "episode_reports.json", reports)
        final = deepcopy(metadata)
        final.update(
            format=FORMAT, schema=schema(), schema_sha256=_schema_hash(), sources=_sources(),
            observation_version=OBSERVATION_VERSION, frame_shape=list(FRAME_SHAPE),
            channels_per_frame=55, n_stack=2, history_order=["previous", "current"],
            samples=len(self.actions), episodes=int(sum(self.starts)), reported_episodes=len(reports),
            codec="NativeRunCodec independent55-plane float32 bit-exact payloads",
            ram_role="audit_only_not_model_input_or_reset", ram_row_bytes=2048,
            raw_observation_sha256=self.observation_digest.hexdigest(),
            raw_ram_sha256=self.ram_digest.hexdigest(), payload_bytes=self.offsets[-1])
        _write(self.path / "metadata.json", final)
        entries = [dict(path=name, bytes=(self.path / name).stat().st_size,
                        sha256=_sha(self.path / name)) for name in sorted(DATA_FILES)]
        fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        _write(self.path / "integrity.json", dict(format=FORMAT, files=entries, fingerprint_sha256=fingerprint))
        self.finished = True
        return dict(final, fingerprint_sha256=fingerprint)

    def close(self):
        for stream in (self.observations, self.ram_stream):
            if not stream.closed:
                stream.flush()
                stream.close()


class PrecisionDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.fingerprint = verify_integrity(self.path)
        self._metadata = json.loads((self.path / "metadata.json").read_text())
        if self._metadata.get("format") != FORMAT or self._metadata.get("schema") != schema() \
                or self._metadata.get("schema_sha256") != _schema_hash() \
                or self._metadata.get("ram_role") != "audit_only_not_model_input_or_reset":
            raise ValueError("Unsupported precision55 corpus/schema")
        self._arrays = {key: np.load(self.path / f"{key}.npy", allow_pickle=False) for key in ARRAYS}
        self._validate()
        for array in self._arrays.values():
            array.setflags(write=False)
        n = len(self)
        self._payload = np.memmap(self.path / "observations.bin", mode="r", dtype=np.uint8) if n else None
        self._ram = np.memmap(self.path / "ram.bin", mode="r", dtype=np.uint8, shape=(n, 2048)) if n else None
        self.codec = NativeRunCodec(FRAME_SHAPE)
        self.closed = False

    def _validate(self):
        n = self._metadata["samples"]
        for key, dtype in ARRAYS.items():
            shape = (n + 1,) if key == "offsets" else (n, 10) if key == "action_masks" else (n,)
            if self._arrays[key].shape != shape or self._arrays[key].dtype != dtype:
                raise ValueError(f"Invalid precision array: {key}")
        offsets = self._arrays["offsets"]
        if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]) \
                or int(offsets[-1]) != (self.path / "observations.bin").stat().st_size \
                or (self.path / "ram.bin").stat().st_size != n * 2048:
            raise ValueError("Precision stream lengths/offsets disagree")
        starts, ids = self._arrays["episode_starts"], self._arrays["episode_ids"]
        if (n and not starts[0]) or not np.array_equal(ids, np.cumsum(starts, dtype=np.int64) - 1):
            raise ValueError("Precision episode IDs/boundaries disagree")
        if int(starts.sum()) != self._metadata["episodes"]:
            raise ValueError("Precision episode count mismatch")
        frames = self._arrays["frame_numbers"]
        if np.any(frames < 0) or np.any((~starts[1:]) & (frames[1:] <= frames[:-1])):
            raise ValueError("Invalid precision RAM frame ordering")
        masks = self._arrays["action_masks"]
        for name in ("teacher_actions", "executed_actions"):
            actions = self._arrays[name]
            if np.any((actions < 0) | (actions >= 10)) or not masks[np.arange(n), actions].all():
                raise ValueError(f"Mask-invalid precision {name}")

    @property
    def episode_ids(self):
        return self._arrays["episode_ids"].view()

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def __len__(self):
        return int(self._metadata["samples"])

    def _indices(self, indices):
        if self.closed:
            raise RuntimeError("Precision dataset closed")
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("Expected one-dimensional integer sample indices")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Precision sample index outside dataset")
        return indices.astype(np.int64, copy=False)

    def _frame(self, index):
        offsets = self._arrays["offsets"]
        return self.codec.decode(self._payload[int(offsets[index]):int(offsets[index + 1])].tobytes())

    def batch(self, indices):
        indices = self._indices(indices)
        observations = np.zeros((len(indices), 110, 26, 26), np.float32)
        for row, index in enumerate(indices):
            observations[row, 55:] = self._frame(index)
            if not self._arrays["episode_starts"][index]:
                observations[row, :55] = self._frame(index - 1)
        return dict(observations=observations, actions=self._arrays["teacher_actions"][indices].copy(),
                    action_masks=self._arrays["action_masks"][indices].copy(),
                    episode_ids=self._arrays["episode_ids"][indices].copy())

    def audit(self, indices):
        indices = self._indices(indices)
        return dict(ram=self._ram[indices].copy() if len(indices) else np.empty((0, 2048), np.uint8),
                    frames=self._arrays["frame_numbers"][indices].copy(),
                    episode_ids=self._arrays["episode_ids"][indices].copy())

    def close(self):
        if not self.closed:
            for array in (self._payload, self._ram):
                if array is not None:
                    array._mmap.close()
        self.closed = True


def train_collection_plan():
    split_path = NEW_ROOT / "frozen-split.json"
    if _sha(split_path) != SPLIT_SHA256:
        raise ValueError("Frozen split identity mismatch")
    split = json.loads(split_path.read_text())
    if split.get("observation_version") != OBSERVATION_VERSION or split.get("channels_per_frame") != 55 \
            or split.get("actions") != 10 or split.get("n_stack") != 2 \
            or split.get("stage") != 1 or split.get("post_start_frames") != 150:
        raise ValueError("Frozen TRAIN precision contract differs")
    pool_path = Path(split["pool_path"])
    if _sha(pool_path) != split["pool_sha256"]:
        raise ValueError("Frozen opening pool identity mismatch")
    pool = load_pool(pool_path)
    by_config = {row["config_index"]: row for row in pool["specs"]}
    configs, families = split["train"]["representative_config_indices"], split["train"]["family_ids"]
    if len(configs) != 40 or len(families) != 40 or len(set(configs)) != 40 or len(set(families)) != 40:
        raise ValueError("Expected40 distinct frozen TRAIN representatives/families")
    plan = []
    for index, (config, family) in enumerate(zip(configs[:24], families[:24])):
        if by_config[config]["family_id"] != family:
            raise ValueError("TRAIN representative/family mapping mismatch")
        plan.append(dict(episode_id=index, config_index=config, family_id=family,
                         opening_spec=by_config[config]["spec"]))
    return pool, plan


def collect_precision(path=NEW_ROOT / "precision-teacher-v1"):
    pool, plan = train_collection_plan()
    authorization = json.loads((NEW_ROOT / "authorization.json").read_text())
    end_epoch = min(LOCAL_DEADLINE, datetime.fromisoformat(authorization["deadline"]).timestamp())
    started = time.monotonic()
    deadline = started + max(0., end_epoch - time.time())
    collection_deadline = deadline - 90
    writer = PrecisionWriter(path)
    monitor = ResourceMonitor(writer.path / "resources.jsonl", device="cpu")
    account = ProbeAccounting(80000, deadline=collection_deadline, monitor=monitor)
    sources = _sources()
    extras = {name: _sha(ROOT / name) for name in (
        "within_stage\\opening_pool.py", "within_stage\\early_openings.py", "within_stage\\openings.py",
        "learning_lab\\teacher.py", "learning_lab\\teacher_v2.py")}
    reports, teacher_attempts, failure, stop_reason = [], 0, None, "plan_complete"
    metadata = dict(
        role="TRAIN-only NON-LEARNED TeacherControllerV2 precision55 corpus",
        split_sha256=SPLIT_SHA256, selected_partition="train",
        representative_order="first24 in frozen TRAIN order, not outcome-selected",
        plan=plan, collector_sources=extras, reservation="7af7da48c84d4030a68c640bad03fd44",
        perturbation_probability=0., ram_writes=False, ram_role="audit_only_not_model_input_or_reset",
        max_frames=12000, frame_skip=4, max_initialization_plus_control_calls=80000,
        label_timing="Precision observation55, RAM, frame and teacher label all captured BEFORE env.step",
        development_or_final_evaluated=False, final_future_traces_read=False,
        warning="Non-learned demonstrations, not learner wins. Only TRAIN families used.",
    )
    _write(writer.path / "collection_config.json", metadata)
    try:
        monitor.check("admission", admission_slots=1, force=True)
        for selected in plan:
            account.check()
            if account.attempted >= account.limit:
                stop_reason = "decision_cap"
                break
            raw = env = None
            before = account.snapshot()
            stored, completed, attempts = 0, 0, 0
            info = dict(outcome="", frames=0, kills=0, deaths=0)
            finished, error = False, None
            begin = time.monotonic()
            try:
                raw = make_opening_env(pool, selected["config_index"], account=account, max_frames=12000)
                env = PrecisionObservation(raw)
                observation, info = env.reset(seed=0)
                teacher = TeacherControllerV2()
                teacher.reset()
                while not finished:
                    account.check()
                    if account.attempted >= account.limit:
                        stop_reason = "decision_cap"
                        info["outcome"] = "decision_budget"
                        break
                    masks = raw.action_masks()
                    teacher_attempts += 1
                    attempts += 1
                    action = _action(teacher.act(raw.state, masks))
                    if not masks[action]:
                        raise ValueError("Precision teacher label violates primitive mask")
                    writer.add(observation, action, masks, ram=raw.nes.ram.copy(),
                               frame=raw.state.frame, episode_start=stored == 0, executed_action=action)
                    stored += 1
                    observation, _, terminated, truncated, info = env.step(action)
                    completed += 1
                    finished = bool(terminated or truncated)
                    if account.attempted % 128 == 0:
                        _write(writer.path / "progress.json", dict(account.snapshot(), stored_samples=len(writer.actions)))
            except ProbeStop as exc:
                stop_reason = "cap_or_deadline"
                info["outcome"] = "aborted_budget_or_deadline"
                error = str(exc)
            except Exception as exc:
                failure = error = f"{type(exc).__name__}: {exc}"
                stop_reason, info["outcome"] = "error", "aborted"
            finally:
                row = dict(
                    selected, stage=1, outcome=info["outcome"], clear=finished and info["outcome"] == "clear",
                    episode_finished=finished, teacher_attempts=attempts, completed_decisions=completed,
                    decisions=stored, frames=int(info["frames"]), kills=int(info["kills"]),
                    deaths=int(info["deaths"]), error=error, elapsed_seconds=time.monotonic() - begin,
                    initialization_calls=account.raw_initialization_attempted - before["raw_initialization_attempted"],
                    control_calls=account.agent_attempted - before["agent_attempted"])
                if raw is not None and finished:
                    row.update(terminal_remaining=raw.state.remaining_enemies,
                               terminal_base_destroyed=raw.state.base_destroyed)
                reports.append(row)
                with (writer.path / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if env is not None:
                    env.close()
                elif raw is not None:
                    raw.close()
                print(json.dumps(row), flush=True)
            if not finished:
                break
    except Exception as exc:
        failure, stop_reason = f"{type(exc).__name__}: {exc}", "error"
    finally:
        unchanged = _sources() == sources and all(_sha(ROOT / name) == digest for name, digest in extras.items())
        if not unchanged:
            failure, stop_reason = "Frozen collector/observation source changed", "source_change"
        summary = dict(
            status="failed" if failure else "completed" if len(reports) == 24 and stop_reason == "plan_complete" else "partial",
            stop_reason=stop_reason, error=failure, accounting=account.snapshot(),
            teacher_attempts=teacher_attempts, samples=len(writer.actions),
            episodes=len(reports), clears=sum(r["clear"] for r in reports),
            outcomes=dict(Counter(r["outcome"] for r in reports)),
            frozen_sources_unchanged=unchanged, native_environment_decisions=0,
            elapsed_seconds=time.monotonic() - started)
        metadata.update(summary)
        writer.finish(reports, metadata)
        _write(writer.path / "summary.json", summary)
    data = PrecisionDataset(writer.path)
    try:
        observation_digest, ram_digest = hashlib.sha256(), hashlib.sha256()
        for first in range(0, len(data), 32):
            if time.monotonic() >= deadline:
                raise TimeoutError("Precision corpus verification deadline")
            monitor.check("precision_verification")
            indices = np.arange(first, min(first + 32, len(data)))
            observations = data.batch(indices)["observations"][:, 55:]
            audit = data.audit(indices)
            observation_digest.update(observations.tobytes(order="C"))
            ram_digest.update(audit["ram"].tobytes(order="C"))
        if observation_digest.hexdigest() != data.metadata["raw_observation_sha256"] \
                or ram_digest.hexdigest() != data.metadata["raw_ram_sha256"]:
            raise ValueError("Precision observation/RAM stream round-trip changed bytes")
        _write(writer.path / "verification.json", dict(
            all_frames_and_ram_bit_exact=True, samples=len(data), schema_sha256=_schema_hash(),
            dataset_fingerprint=data.fingerprint, extra_environment_decisions=0))
    finally:
        data.close()
    monitor.check("complete", force=True)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    collect_precision()
