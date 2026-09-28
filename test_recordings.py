import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

from recordings import BestRecordings


FRAME = np.zeros((4, 6, 3), dtype=np.uint8)


def episode(**changes):
    info = dict(objective="full", outcome="timeout", kills=3, deaths=1,
                frames=120, lives=2, shots=10, pickups=1, score=300)
    info.update(changes)
    return info


def windows_encoder_processes(path):
    quoted = str(path).replace("'", "''")
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "ConvertTo-Json -Compress -InputObject @("
        "Get-CimInstance Win32_Process -Filter \"Name = 'ffmpeg.exe'\" | "
        f"Where-Object {{ $_.CommandLine -and $_.CommandLine.Contains('{quoted}') }} | "
        "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine)"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, check=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return json.loads(result.stdout)


class FixtureProcess:
    """Replace only encoding; exercise real files, ranking, locks and commits."""

    def __init__(self, command, *, stdin, stdout, stderr, creationflags=0, **kwargs):
        self.path = Path(command[-1])
        self.stdin = io.BytesIO()
        self.stderr = stderr
        self.returncode = None
        self.creationflags = creationflags

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.path.write_bytes(b"deterministic encoded episode")
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.processes = []

        def encode(*args, **kwargs):
            process = FixtureProcess(*args, **kwargs)
            self.processes.append(process)
            return process

        self.encoder = patch("recordings.subprocess.Popen", side_effect=encode)
        self.encoder.start()
        self.addCleanup(self.encoder.stop)

    def recorder(self, **kwargs):
        recorder = BestRecordings(self.directory, **kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def record(self, recorder, *, metadata=None, **info):
        recorder.begin(FRAME, metadata or {"model": "heuristic", "seed": 7})
        recorder.write(FRAME)
        return recorder.finish(episode(**info))

    def test_encoder_uses_no_console_window_on_windows(self):
        recorder = self.recorder()
        self.record(recorder)
        expected = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.assertEqual(self.processes[-1].creationflags, expected)

    def test_ranking_clear_kills_deaths_then_outcome_specific_time(self):
        recorder = self.recorder()
        cases = [
            ("failed_short", dict(frames=100)),
            ("failed_long", dict(frames=200)),
            ("clear_slow", dict(outcome="clear", frames=200)),
            ("clear_fast", dict(outcome="clear", frames=100)),
            ("failed_fewer_deaths", dict(deaths=0)),
            ("failed_more_kills", dict(kills=4, deaths=3)),
            ("clear_more_kills", dict(outcome="clear", kills=4, deaths=3)),
            ("clear_fewer_deaths", dict(outcome="clear", deaths=0)),
        ]
        for label, info in cases:
            self.record(recorder, metadata={"label": label}, **info)
        self.assertEqual([e["metadata"]["label"] for e in recorder.entries], [
            "clear_more_kills", "clear_fewer_deaths", "clear_fast", "clear_slow",
            "failed_more_kills", "failed_fewer_deaths", "failed_long", "failed_short",
        ])

    def test_top_ten_cap_rejection_displacement_and_persistence(self):
        recorder = self.recorder()
        paths = [self.record(recorder, kills=k) for k in range(12)]
        self.assertEqual([e["kills"] for e in recorder.entries], list(range(11, 1, -1)))
        self.assertFalse(paths[0].exists())
        self.assertFalse(paths[1].exists())
        self.assertEqual(len(list(self.directory.glob("*.mp4"))), 10)
        previous = (self.directory / "rankings.json").read_bytes()
        self.assertIsNone(self.record(recorder, kills=0))
        self.assertEqual((self.directory / "rankings.json").read_bytes(), previous)
        recorder.close()
        reopened = self.recorder()
        self.assertEqual([e["kills"] for e in reopened.entries], list(range(11, 1, -1)))
        self.assertEqual(len(list(self.directory.glob("*.mp4"))), 10)

    def test_exact_ties_keep_older_entries_across_restarts(self):
        recorder = self.recorder(limit=2)
        first = self.record(recorder, metadata={"seed": 1})
        second = self.record(recorder, metadata={"seed": 2})
        recorder.close()
        reopened = self.recorder(limit=2)
        self.assertIsNone(self.record(reopened, metadata={"seed": 3}))
        self.assertEqual([e["metadata"]["seed"] for e in reopened.entries], [1, 2])
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_metadata_snapshot_counts_and_entries_are_not_mutable_aliases(self):
        recorder = self.recorder(fps=12.5)
        metadata = {"model": "ppo", "checkpoint": "model.zip", "seed": 0, "nested": [1]}
        recorder.begin(FRAME, metadata)
        metadata["nested"].append(2)
        recorder.write(FRAME[:, ::-1, :])
        path = recorder.finish(episode(outcome="clear"))
        entry = recorder.entries[0]
        self.assertEqual(entry["file"], path.name)
        self.assertEqual(entry["metadata"], {
            "model": "ppo", "checkpoint": "model.zip", "seed": 0, "nested": [1],
        })
        self.assertEqual(entry["recording_fps"], 12.5)
        self.assertEqual(entry["recording_frames"], 2)
        for key, value in episode(outcome="clear").items():
            self.assertEqual(entry[key], value)
        entry["metadata"]["seed"] = 99
        entry["kills"] = 99
        self.assertEqual(recorder.entries[0]["metadata"]["seed"], 0)
        self.assertEqual(recorder.entries[0]["kills"], 3)

    def test_training_objectives_are_never_retained(self):
        recorder = self.recorder()
        for objective, outcome in [("combat", "objective"), ("defense", "timeout")]:
            self.assertIsNone(self.record(recorder, objective=objective, outcome=outcome))
        self.assertEqual(recorder.entries, [])
        self.assertEqual(list(self.directory.glob("*.mp4")), [])

    def test_invalid_full_episode_aborts_and_does_not_change_completed_entries(self):
        recorder = self.recorder()
        saved = self.record(recorder)
        for changes in [dict(outcome="objective"), dict(outcome=""),
                        dict(kills=-1), dict(deaths=True), dict(frames=1.5),
                        dict(score=float("nan")), dict(objective=None)]:
            with self.subTest(changes=changes):
                recorder.begin(FRAME, {})
                with self.assertRaises(ValueError):
                    recorder.finish(episode(**changes))
                self.assertTrue(saved.exists())
                self.assertEqual(len(recorder.entries), 1)
        self.assertEqual(len(list(self.directory.glob("*.mp4"))), 1)

    def test_close_aborts_partial_and_is_idempotent(self):
        recorder = self.recorder()
        saved = self.record(recorder)
        recorder.begin(FRAME, {})
        self.assertEqual(len(recorder.entries), 1)
        recorder.close()
        recorder.close()
        self.assertEqual(list(self.directory.glob("*.mp4")), [saved])
        self.assertIsNotNone(self.processes[-1].returncode)
        self.assertTrue(self.processes[-1].stdin.closed)
        with self.assertRaises(RuntimeError):
            recorder.begin(FRAME, {})
        self.assertEqual(len(self.recorder().entries), 1)

    def test_invalid_frames_metadata_and_call_order_are_explicit(self):
        recorder = self.recorder()
        for action in [lambda: recorder.write(FRAME), lambda: recorder.finish(episode())]:
            with self.assertRaises(RuntimeError):
                action()
        for frame in [FRAME.astype(float), FRAME[:, :, 0], np.zeros((0, 2, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                recorder.begin(frame, {})
        with self.assertRaises(ValueError):
            recorder.begin(FRAME, {"bad": float("nan")})
        recorder.begin(FRAME, {})
        with self.assertRaises(RuntimeError):
            recorder.begin(FRAME, {})
        with self.assertRaises(ValueError):
            recorder.write(np.zeros((2, 6, 3), np.uint8))
        self.assertEqual(list(self.directory.glob("*.mp4")), [])

    def test_encoder_exit_failure_preserves_existing_top_recording(self):
        recorder = self.recorder(limit=1)
        saved = self.record(recorder)
        old_index = (self.directory / "rankings.json").read_bytes()
        recorder.begin(FRAME, {})
        process = self.processes[-1]
        process.stderr.write(b"encoder disk full\n")
        process.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "encoder disk full"):
            recorder.finish(episode(outcome="clear"))
        self.assertTrue(saved.exists())
        self.assertEqual((self.directory / "rankings.json").read_bytes(), old_index)
        self.assertEqual(list(self.directory.glob("*.mp4")), [saved])

    def test_broken_stdin_aborts_without_promoting_and_allows_next_round(self):
        recorder = self.recorder()
        recorder.begin(FRAME, {})
        process = self.processes[-1]
        with patch.object(process.stdin, "write", side_effect=BrokenPipeError("broken encoder")):
            with self.assertRaises((OSError, RuntimeError)):
                recorder.write(FRAME)
        self.assertIsNotNone(process.returncode)
        self.assertTrue(process.stdin.closed)
        self.assertEqual(list(self.directory.glob("*.mp4")), [])
        self.assertIsNotNone(self.record(recorder))

    def test_spawn_failure_does_not_leave_partial_or_hold_round_open(self):
        recorder = self.recorder()
        with patch("recordings.subprocess.Popen", side_effect=OSError("cannot start encoder")):
            with self.assertRaisesRegex(OSError, "cannot start encoder"):
                recorder.begin(FRAME, {})
        self.assertEqual(list(self.directory.glob("*.mp4")), [])
        self.assertIsNotNone(self.record(recorder))

    def test_failed_index_commit_keeps_old_index_files_and_memory(self):
        recorder = self.recorder(limit=1)
        saved = self.record(recorder)
        previous = (self.directory / "rankings.json").read_bytes()
        previous_entries = recorder.entries
        replace = os.replace

        def fail_index(source, destination):
            if Path(destination).name == "rankings.json":
                raise OSError("index disk full")
            return replace(source, destination)

        with patch("recordings.os.replace", side_effect=fail_index):
            with self.assertRaisesRegex(OSError, "index disk full"):
                self.record(recorder, outcome="clear")
        self.assertEqual(recorder.entries, previous_entries)
        self.assertEqual((self.directory / "rankings.json").read_bytes(), previous)
        self.assertEqual(list(self.directory.glob("*.mp4")), [saved])
        recorder.close()
        self.assertEqual(self.recorder(limit=1).entries, previous_entries)

    def test_collision_with_existing_recording_does_not_delete_it(self):
        recorder = self.recorder(limit=1)
        saved = self.record(recorder)
        previous = (self.directory / "rankings.json").read_bytes()
        recorder.begin(FRAME, {})
        with patch("recordings.uuid4", return_value=UUID(saved.stem.removeprefix("tank-ai-"))):
            with self.assertRaises(FileExistsError):
                recorder.finish(episode(outcome="clear"))
        self.assertTrue(saved.exists())
        self.assertEqual((self.directory / "rankings.json").read_bytes(), previous)

    def test_collision_with_index_temporary_does_not_delete_existing_file(self):
        recorder = self.recorder(limit=1)
        saved = self.record(recorder)
        temporary = self.directory / ".tank-ai-00000000000000000000000000000002.index.tmp"
        temporary.write_bytes(b"existing file")
        recorder.begin(FRAME, {})
        with patch("recordings.uuid4", side_effect=[UUID(int=1), UUID(int=2)]):
            with self.assertRaises(FileExistsError):
                recorder.finish(episode(outcome="clear"))
        self.assertTrue(temporary.exists())
        self.assertEqual(temporary.read_bytes(), b"existing file")
        self.assertEqual(list(self.directory.glob("*.mp4")), [saved])

    def test_fsync_failure_preserves_completed_top_entry(self):
        recorder = self.recorder(limit=1)
        saved = self.record(recorder)
        previous = (self.directory / "rankings.json").read_bytes()
        with patch("recordings.os.fsync", side_effect=OSError("cannot sync disk")):
            with self.assertRaisesRegex(OSError, "cannot sync disk"):
                self.record(recorder, outcome="clear")
        self.assertEqual((self.directory / "rankings.json").read_bytes(), previous)
        self.assertEqual(list(self.directory.glob("*.mp4")), [saved])

    def test_encoder_timeout_is_raised_and_encoder_is_reaped(self):
        recorder = self.recorder()
        recorder.begin(FRAME, {})
        process = self.processes[-1]
        wait = process.wait
        timed_out = False

        def timeout_then_exit(timeout=None):
            nonlocal timed_out
            if not timed_out:
                timed_out = True
                raise subprocess.TimeoutExpired("ffmpeg", 60)
            return wait(timeout)

        with patch.object(process, "wait", side_effect=timeout_then_exit):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                recorder.finish(episode())
        self.assertIsNotNone(process.returncode)
        self.assertTrue(process.stdin.closed)
        self.assertEqual(list(self.directory.glob("*.mp4")), [])

    def test_close_timeout_is_explicit_and_releases_writer_lock(self):
        recorder = self.recorder()
        recorder.begin(FRAME, {})
        process = self.processes[-1]
        with patch.object(process, "wait", side_effect=[
            subprocess.TimeoutExpired("ffmpeg", 60), -9,
        ]):
            with self.assertRaises(subprocess.TimeoutExpired):
                recorder.close()
        self.assertEqual(process.returncode, -9)
        self.assertTrue(process.stdin.closed)
        self.assertEqual(list(self.directory.glob("*.mp4")), [])
        recorder.close()
        self.assertEqual(self.recorder().entries, [])

    def test_encoder_zero_exit_with_empty_output_is_not_success(self):
        recorder = self.recorder()
        recorder.begin(FRAME, {})
        self.processes[-1].returncode = 0
        with self.assertRaisesRegex(RuntimeError, "empty"):
            recorder.finish(episode())
        self.assertEqual(recorder.entries, [])
        self.assertEqual(list(self.directory.glob("*.mp4")), [])

    def test_cleanup_failure_is_explicit_and_recovered_after_commit(self):
        recorder = self.recorder(limit=1)
        old = self.record(recorder)
        unlink = Path.unlink

        def fail_old(path, *args, **kwargs):
            if path == old:
                raise PermissionError("file temporarily busy")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_old):
            with self.assertRaisesRegex(PermissionError, "file temporarily busy"):
                self.record(recorder, outcome="clear")
        self.assertEqual(recorder.entries[0]["outcome"], "clear")
        self.assertTrue(old.exists())
        recorder.close()
        reopened = self.recorder(limit=1)
        self.assertEqual(reopened.entries[0]["outcome"], "clear")
        self.assertFalse(old.exists())
        self.assertEqual(len(list(self.directory.glob("*.mp4"))), 1)

    def test_smaller_limit_on_restart_commits_before_pruning(self):
        recorder = self.recorder()
        for kills in range(4):
            self.record(recorder, kills=kills)
        recorder.close()
        smaller = self.recorder(limit=2)
        self.assertEqual([e["kills"] for e in smaller.entries], [3, 2])
        self.assertEqual(len(list(self.directory.glob("*.mp4"))), 2)
        persisted = json.loads((self.directory / "rankings.json").read_text())
        self.assertEqual([e["kills"] for e in persisted["entries"]], [3, 2])

    def test_index_is_committed_before_displaced_file_is_removed(self):
        recorder = self.recorder(limit=1)
        old = self.record(recorder)
        unlink = Path.unlink

        def verify_unlink(path, *args, **kwargs):
            if path == old:
                persisted = json.loads((self.directory / "rankings.json").read_text())
                self.assertEqual(persisted["entries"][0]["outcome"], "clear")
                self.assertNotEqual(persisted["entries"][0]["file"], old.name)
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", verify_unlink):
            new = self.record(recorder, outcome="clear")
        self.assertTrue(new.exists())
        self.assertFalse(old.exists())

    def test_unsafe_index_basename_is_rejected_without_deleting_anything(self):
        recorder = self.recorder()
        saved = self.record(recorder)
        recorder.close()
        index = self.directory / "rankings.json"
        original = json.loads(index.read_text())
        unrelated = self.directory / "holiday.mp4"
        unrelated.write_bytes(b"not owned")
        for name in ["../holiday.mp4", "..\\holiday.mp4", str(unrelated),
                     "holiday.mp4", "tank-ai-bogus.mp4", saved.name + ":stream"]:
            with self.subTest(name=name):
                changed = json.loads(json.dumps(original))
                changed["entries"][0]["file"] = name
                index.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.recorder()
                self.assertTrue(saved.exists())
                self.assertEqual(unrelated.read_bytes(), b"not owned")
        index.write_text(json.dumps(original))
        self.assertEqual(len(self.recorder().entries), 1)

    def test_index_rejects_corruption_missing_files_and_duplicate_entries(self):
        recorder = self.recorder()
        saved = self.record(recorder)
        recorder.close()
        index = self.directory / "rankings.json"
        original = json.loads(index.read_text())
        invalid = [
            "{broken", '{"version":1,"version":2}',
            json.dumps({**original, "version": 99}),
            json.dumps({**original, "entries": original["entries"] * 2}),
            json.dumps({**original, "next_sequence": -1}),
        ]
        for contents in invalid:
            index.write_text(contents)
            with self.assertRaises(ValueError):
                self.recorder()
            self.assertTrue(saved.exists())
        index.write_text(json.dumps(original))
        saved.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.recorder()

    def test_recovery_removes_only_exact_owned_orphan_files(self):
        recorder = self.recorder()
        saved = self.record(recorder)
        recorder.close()
        partial = self.directory / ".tank-ai-11111111111111111111111111111111.partial.mp4"
        index_temp = self.directory / ".tank-ai-22222222222222222222222222222222.index.tmp"
        promoted = self.directory / "tank-ai-33333333333333333333333333333333.mp4"
        unrelated = self.directory / "holiday.mp4"
        almost = self.directory / ".tank-ai-not-a-uuid.partial.mp4"
        for path in [partial, index_temp, promoted, unrelated, almost]:
            path.write_bytes(b"fixture")
        reopened = self.recorder()
        self.assertEqual(len(reopened.entries), 1)
        self.assertTrue(saved.exists())
        for path in [partial, index_temp, promoted]:
            self.assertFalse(path.exists())
        self.assertTrue(unrelated.exists())
        self.assertTrue(almost.exists())

    def test_owned_looking_directory_is_not_removed(self):
        owned_directory = self.directory / ".tank-ai-11111111111111111111111111111111.partial.mp4"
        owned_directory.mkdir()
        with self.assertRaises(ValueError):
            self.recorder()
        self.assertTrue(owned_directory.is_dir())

    def test_owned_looking_hardlink_is_not_followed_or_deleted(self):
        original = self.directory / "unrelated.mp4"
        original.write_bytes(b"do not delete")
        linked = self.directory / ".tank-ai-11111111111111111111111111111111.partial.mp4"
        os.link(original, linked)
        with self.assertRaises(ValueError):
            self.recorder()
        self.assertEqual(original.read_bytes(), b"do not delete")
        self.assertTrue(linked.exists())

    def test_corrupt_index_prevents_orphan_cleanup(self):
        partial = self.directory / ".tank-ai-11111111111111111111111111111111.partial.mp4"
        partial.write_bytes(b"must remain while index is invalid")
        (self.directory / "rankings.json").write_text("{incomplete")
        with self.assertRaises(ValueError):
            self.recorder()
        self.assertTrue(partial.exists())

    def test_second_writer_is_rejected_and_lock_is_reusable_after_close(self):
        recorder = self.recorder()
        with self.assertRaisesRegex(RuntimeError, "lock|writer"):
            self.recorder()
        recorder.close()
        self.assertEqual(self.recorder().entries, [])

    def test_invalid_constructor_settings(self):
        for kwargs in [dict(fps=0), dict(fps=float("nan")), dict(fps=float("inf")),
                       dict(scale=0), dict(scale=1.5), dict(limit=0), dict(limit=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.recorder(**kwargs)


class RealEncoderTests(unittest.TestCase):
    def test_live_path_encoder_close_waits_for_child_and_releases_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recorder = BestRecordings(directory)
            try:
                frame = np.zeros((240, 256, 3), np.uint8)
                recorder.begin(frame, {"seed": 0})
                for _ in range(8):
                    recorder.write(frame)
                partial, = directory.glob("*.partial.mp4")
                process = recorder._round.process
                before = windows_encoder_processes(partial) if os.name == "nt" else []
                if os.name == "nt":
                    self.assertTrue(before, "The real encoder must be running before abort")
                with self.subTest(process_tree=before):
                    try:
                        recorder.close()
                    finally:
                        if os.name == "nt":
                            after = windows_encoder_processes(partial)
                    recorder.close()
                    self.assertEqual(process.returncode, 0)
                    if os.name == "nt":
                        self.assertEqual(after, [])
                    self.assertEqual(list(directory.glob("*.mp4")), [])
                    self.assertEqual(recorder.entries, [])
                    # Recreating the exact name requires the encoder's file handle to be gone.
                    with partial.open("xb") as stream:
                        stream.write(b"file is available")
                    partial.unlink()
            finally:
                recorder.close()

    def test_real_mp4_codec_size_fps_count_silence_and_faststart(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            recorder = BestRecordings(directory)
            try:
                frame = np.zeros((5, 7, 3), np.uint8)
                frame[:, :, 0] = 240
                recorder.begin(frame, {"checkpoint": "tiny.zip", "seed": 42})
                recorder.write(frame)
                recorder.write(frame)
                saved = recorder.finish(episode(outcome="clear"))
                self.assertIsNotNone(saved)
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-count_frames", "-show_streams",
                    "-show_format", "-of", "json", str(saved),
                ], capture_output=True, text=True, check=True, timeout=30)
                probe = json.loads(result.stdout)
                self.assertEqual(len(probe["streams"]), 1)
                video = probe["streams"][0]
                self.assertEqual(video["codec_type"], "video")
                self.assertEqual(video["codec_name"], "h264")
                self.assertEqual(video["pix_fmt"], "yuv420p")
                self.assertEqual((video["width"], video["height"]), (22, 16))
                self.assertEqual(video["r_frame_rate"], "15/1")
                self.assertEqual(int(video["nb_read_frames"]), 3)
                self.assertAlmostEqual(float(probe["format"]["duration"]), 0.2, places=2)
                content = saved.read_bytes()
                self.assertLess(content.index(b"moov"), content.index(b"mdat"))
                self.assertEqual(recorder.entries[0]["recording_frames"], 3)
                recorder.begin(frame, {})
            finally:
                recorder.close()
            self.assertEqual(list(directory.glob("*.mp4")), [saved])

    def test_file_lock_is_released_even_after_abrupt_process_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([
                sys.executable, "-c",
                "import os, sys; from pathlib import Path; "
                "from recordings import BestRecordings; "
                "recorder = BestRecordings(Path(sys.argv[1])); os._exit(0)",
                temp,
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorder = BestRecordings(Path(temp))
            recorder.close()

    def test_active_writer_is_also_exclusive_across_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = BestRecordings(Path(temp))
            try:
                result = subprocess.run([
                    sys.executable, "-c",
                    "import sys\nfrom pathlib import Path\n"
                    "from recordings import BestRecordings\n"
                    "try:\n    recorder = BestRecordings(Path(sys.argv[1]))\n"
                    "except RuntimeError:\n    sys.exit(7)\n"
                    "recorder.close()\nsys.exit(0)\n",
                    temp,
                ], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 7, result.stderr)
            finally:
                recorder.close()

    def test_large_encoder_stderr_cannot_deadlock_or_flood_error_message(self):
        popen = subprocess.Popen

        def noisy_encoder(command, **kwargs):
            return popen([
                sys.executable, "-c",
                "import sys; sys.stderr.buffer.write(b'x' * 1048576 + b'\\nencoder failed'); "
                "sys.stderr.flush(); sys.stdin.buffer.read(); sys.exit(7)",
            ], **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            recorder = BestRecordings(Path(temp))
            try:
                with patch("recordings.subprocess.Popen", side_effect=noisy_encoder):
                    recorder.begin(FRAME, {})
                    with self.assertRaisesRegex(RuntimeError, "encoder failed") as raised:
                        recorder.finish(episode())
                self.assertLess(len(str(raised.exception)), 9000)
                self.assertEqual(recorder.entries, [])
                self.assertEqual(list(Path(temp).glob("*.mp4")), [])
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
