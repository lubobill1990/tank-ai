"""Streaming, single-writer best-episode MP4 retention.

BestRecordings(directory, *, fps=15, scale=3, limit=10) acquires an OS file
lock until close(). begin(rgb_uint8, metadata) writes the initial frame;
write(rgb_uint8) appends one frame. finish(info) accepts terminal full-stage
episodes and returns the retained Path or None. close() aborts a partial round.
Abort sends EOF and waits for encoder exit before discarding the partial MP4,
including when ffmpeg on PATH is a shim that waits for an encoder child.
Call close() in a finally block. All failures are raised to the caller.

entries is a defensive copy of the ranked list persisted in rankings.json.
Each entry has file, sequence, metadata (the caller's JSON-serializable dict),
objective/outcome/kills/deaths/frames/lives/shots/pickups/score, optional stage,
recording_fps and recording_frames. Game frames rank duration; recording_frames
counts submitted images, including begin's image. Clear episodes precede
failures, then kills descending, deaths ascending, and game frames ascending
for clears / descending for failures. Equal ranks retain the older sequence.
Combat/defense training episodes are discarded, not ranked as stage clears.

ffmpeg must be on PATH; Windows encoder processes have no console window.
Encoding is silent CPU libx264, one thread, yuv420p,
faststart, nearest-neighbor integer scaling; odd output dimensions are padded
on the right/bottom to even dimensions. Frames and stderr are never accumulated
in memory: stderr goes to a temporary file, with only its tail read on error.

The directory reserves rankings.json, .tank-ai-recordings.lock, and UUID names
tank-ai-<32 lowercase hex>.mp4, .tank-ai-<hex>.partial.mp4 and
.tank-ai-<hex>.index.tmp. Startup validates the entire index before recovering
unindexed files in that exact namespace. Links/reparse points and directories
are rejected, never followed or recursively removed. Do not place unrelated
files in this reserved namespace. The lock file intentionally remains on disk;
its open-file lock, not its existence, determines ownership.

Promotion precedes atomic index replacement; displaced files are removed only
after the replacement succeeds. Encode/index failures preserve the old top
list. A deletion failure after commit is raised, but the committed list remains
authoritative; the next startup retries orphan cleanup.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import BinaryIO
from uuid import uuid4

import numpy as np


_VIDEO = re.compile(r"tank-ai-[0-9a-f]{32}\.mp4")
_TEMP = re.compile(r"\.tank-ai-[0-9a-f]{32}\.(?:partial\.mp4|index\.tmp)")
_COUNTS = ("kills", "deaths", "frames", "lives", "shots", "pickups", "score")
_OUTCOMES = ("clear", "base_lost", "lives_exhausted", "timeout")
_WAIT_SECONDS = 60


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _fps(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("fps must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be a finite positive number")
    return float(value)


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON-serializable dict")

    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("metadata keys must be strings")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)

    try:
        check_keys(value)
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"metadata must contain finite JSON values: {exc}") from exc


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate index key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite index value: {value}")


def _regular_file(path, *, missing_ok=False):
    try:
        details = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    if (not stat.S_ISREG(details.st_mode) or details.st_nlink != 1
            or getattr(details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise ValueError(f"Expected an owned regular file, not a link/directory: {path}")
    return True


def _rank(entry):
    clear = entry["outcome"] == "clear"
    return (not clear, -entry["kills"], entry["deaths"],
            entry["frames"] if clear else -entry["frames"], entry["sequence"])


def _episode_info(info):
    if not isinstance(info, dict) or info.get("objective") != "full":
        raise ValueError("A ranked episode must have objective='full'")
    if info.get("outcome") not in _OUTCOMES:
        raise ValueError("A full episode must have a terminal full-stage outcome")
    result = {"objective": "full", "outcome": info["outcome"]}
    for key in _COUNTS:
        result[key] = _integer(info.get(key), key)
    if "stage" in info:
        result["stage"] = _integer(info["stage"], "stage", 1)
    return result


@dataclass
class _Round:
    path: Path
    shape: tuple[int, ...]
    metadata: dict
    process: subprocess.Popen | None = None
    stderr: BinaryIO | None = None
    frame_count: int = 0


class BestRecordings:
    def __init__(self, directory: Path, *, fps: float = 15, scale: int = 3,
                 limit: int = 10):
        self.fps = _fps(fps)
        self.scale = _integer(scale, "scale", 1)
        self.limit = _integer(limit, "limit", 1)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory = self.directory.resolve(strict=True)
        self._round: _Round | None = None
        self._entries: list[dict] = []
        self._next_sequence = 0
        self._closed = False
        lock_path = self.directory / ".tank-ai-recordings.lock"
        _regular_file(lock_path, missing_ok=True)
        self._lock = lock_path.open("a+b")
        ready = False
        try:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._lock.seek(0)
                    msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Cannot acquire recordings writer lock: {lock_path}") from exc
            self._load_index()
            self._recover()
            if len(self._entries) > self.limit:
                kept, displaced = self._entries[:self.limit], self._entries[self.limit:]
                self._commit_index(kept, self._next_sequence)
                self._entries = kept
                for entry in displaced:
                    self._remove_owned(self.directory / entry["file"])
            ready = True
        finally:
            if not ready:
                self._closed = True
                self._lock.close()

    @property
    def entries(self) -> list[dict]:
        return deepcopy(self._entries)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Recordings writer is closed")

    def _active(self):
        self._ensure_open()
        if self._round is None:
            raise RuntimeError("No recording round is active")
        return self._round

    def _load_index(self):
        path = self.directory / "rankings.json"
        if not _regular_file(path, missing_ok=True):
            return
        with path.open("r", encoding="utf-8") as stream:
            index = json.load(stream, object_pairs_hook=_object,
                              parse_constant=_invalid_constant)
        if (not isinstance(index, dict)
                or set(index) != {"version", "next_sequence", "entries"}
                or type(index["version"]) is not int or index["version"] != 1
                or not isinstance(index["entries"], list)):
            raise ValueError("Invalid recordings index schema/version")
        next_sequence = _integer(index["next_sequence"], "next_sequence")
        files, sequences = set(), set()
        for entry in index["entries"]:
            if not isinstance(entry, dict):
                raise ValueError("Index entries must be objects")
            name = entry.get("file")
            if not isinstance(name, str) or _VIDEO.fullmatch(name) is None:
                raise ValueError(f"Invalid owned recording basename: {name!r}")
            sequence = _integer(entry.get("sequence"), "sequence")
            if name in files or sequence in sequences or sequence >= next_sequence:
                raise ValueError("Duplicate recording or invalid sequence in index")
            files.add(name)
            sequences.add(sequence)
            _episode_info(entry)
            _metadata(entry.get("metadata"))
            _fps(entry.get("recording_fps"))
            _integer(entry.get("recording_frames"), "recording_frames", 1)
            _regular_file(self.directory / name)
            if (self.directory / name).stat().st_size == 0:
                raise ValueError(f"Empty indexed recording: {name}")
        if index["entries"] != sorted(index["entries"], key=_rank):
            raise ValueError("Recordings index is not in rank order")
        self._entries = index["entries"]
        self._next_sequence = next_sequence

    def _remove_owned(self, path):
        if path.parent != self.directory or not (
                _VIDEO.fullmatch(path.name) or _TEMP.fullmatch(path.name)):
            raise ValueError(f"Refusing to remove unowned recording path: {path}")
        if _regular_file(path, missing_ok=True):
            path.unlink()

    def _recover(self):
        indexed = {entry["file"] for entry in self._entries}
        orphans = [
            path for path in self.directory.iterdir()
            if _TEMP.fullmatch(path.name)
            or (_VIDEO.fullmatch(path.name) and path.name not in indexed)
        ]
        for path in orphans:
            _regular_file(path)
        for path in orphans:
            self._remove_owned(path)

    def _commit_index(self, entries, next_sequence):
        path = self.directory / f".tank-ai-{uuid4().hex}.index.tmp"
        index_path = self.directory / "rankings.json"
        _regular_file(index_path, missing_ok=True)
        stream = path.open("x", encoding="utf-8", newline="\n")
        try:
            with stream:
                json.dump({"version": 1, "next_sequence": next_sequence, "entries": entries},
                          stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path, index_path)
        finally:
            self._remove_owned(path)

    @staticmethod
    def _frame(frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or frame.shape[0] == 0 or frame.shape[1] == 0):
            raise ValueError("Frames must be nonempty RGB uint8 arrays of shape (H, W, 3)")
        return frame

    def begin(self, frame: np.ndarray, metadata: dict):
        self._ensure_open()
        if self._round is not None:
            raise RuntimeError("A recording round is already active")
        frame = self._frame(frame)
        metadata = _metadata(metadata)
        path = self.directory / f".tank-ai-{uuid4().hex}.partial.mp4"
        with path.open("xb"):
            pass
        round_ = _Round(path, frame.shape, metadata)
        self._round = round_
        started = False
        try:
            round_.stderr = tempfile.TemporaryFile(mode="w+b")
            height, width = frame.shape[:2]
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
                "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{width}x{height}", "-framerate", str(self.fps),
                "-i", "pipe:0", "-an", "-filter_threads", "1",
                "-vf", f"scale=iw*{self.scale}:ih*{self.scale}:flags=neighbor,"
                       "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-threads", "1",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(path),
            ]
            round_.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=round_.stderr, bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.write(frame)
            started = True
        finally:
            if not started:
                self._abort_round()

    @staticmethod
    def _encoder_error(round_, message):
        detail = ""
        if round_.stderr is not None:
            round_.stderr.seek(0, os.SEEK_END)
            size = round_.stderr.tell()
            round_.stderr.seek(max(0, size - 8192))
            detail = round_.stderr.read().decode("utf-8", errors="replace").strip()
        return RuntimeError(f"ffmpeg {message}" + (f": {detail}" if detail else ""))

    def write(self, frame: np.ndarray):
        round_ = self._active()
        written = False
        try:
            frame = self._frame(frame)
            if frame.shape != round_.shape:
                raise ValueError("Frame shape changed during an episode")
            process = round_.process
            if process is None or process.stdin is None:
                raise RuntimeError("Recording encoder has not started")
            if process.poll() is not None:
                raise self._encoder_error(round_, f"exited with status {process.returncode}")
            data = memoryview(np.ascontiguousarray(frame)).cast("B")
            while data:
                try:
                    size = process.stdin.write(data)
                except OSError as exc:
                    raise self._encoder_error(round_, f"stdin write failed ({exc})") from exc
                if size is None or size <= 0:
                    raise self._encoder_error(round_, "stdin write made no progress")
                data = data[size:]
            round_.frame_count += 1
            written = True
        finally:
            if not written:
                self._abort_round()

    def finish(self, info: dict) -> Path | None:
        round_ = self._active()
        promoted = None
        committed = False
        try:
            if isinstance(info, dict) and info.get("objective") in ("combat", "defense"):
                return None
            entry = _episode_info(info)
            process = round_.process
            if process is None or process.stdin is None:
                raise RuntimeError("Recording encoder has not started")
            process.stdin.close()
            try:
                returncode = process.wait(timeout=_WAIT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                raise self._encoder_error(round_, "timed out finalizing MP4") from exc
            if returncode:
                raise self._encoder_error(round_, f"exited with status {returncode}")
            _regular_file(round_.path)
            if round_.path.stat().st_size == 0:
                raise self._encoder_error(round_, "produced an empty MP4")
            entry.update(sequence=self._next_sequence, metadata=round_.metadata,
                         recording_fps=self.fps, recording_frames=round_.frame_count,
                         file=f"tank-ai-{uuid4().hex}.mp4")
            ranked = sorted([*self._entries, entry], key=_rank)
            kept, displaced = ranked[:self.limit], ranked[self.limit:]
            if entry not in kept:
                return None
            destination = self.directory / entry["file"]
            if destination.exists():
                raise FileExistsError(f"Recording destination already exists: {destination}")
            with round_.path.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.rename(round_.path, destination)
            promoted = destination
            self._commit_index(kept, self._next_sequence + 1)
            committed = True
            self._entries = kept
            self._next_sequence += 1
            for old in displaced:
                self._remove_owned(self.directory / old["file"])
            return promoted
        finally:
            try:
                if promoted is not None and not committed:
                    self._remove_owned(promoted)
            finally:
                self._abort_round()

    def _abort_round(self):
        round_ = self._round
        self._round = None
        if round_ is None:
            return
        try:
            process = round_.process
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                finally:
                    try:
                        process.wait(timeout=_WAIT_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=_WAIT_SECONDS)
                        raise
        finally:
            try:
                if round_.stderr is not None:
                    round_.stderr.close()
            finally:
                self._remove_owned(round_.path)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._abort_round()
        finally:
            self._lock.close()
