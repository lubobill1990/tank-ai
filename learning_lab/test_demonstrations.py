"""Synthetic codec/dataset tests; these consume zero environment decisions."""
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from learning_lab.demonstrations import DemonstrationDataset, DemonstrationWriter


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("runs") / ("demonstrations-test-" + uuid.uuid4().hex)
        self.path = self.root / "corpus"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def make_corpus(self):
        frames = np.zeros((4, 47, 26, 26), np.float32)
        frames[:, 4].fill(1)
        frames[0].view(np.uint32)[0, 0, :4] = [0x80000000, 0x7fc00001, 0x7fa12345, 0xff800000]
        frames[1, 7, 8, 9] = .1234567
        frames[2, 8, 6, 4] = -.25
        frames[3, 40] = .8
        masks = np.ones((4, 10), bool)
        masks[1, 9] = False
        writer = DemonstrationWriter(self.path)
        self.addCleanup(writer.close)
        for index, frame in enumerate(frames):
            writer.add(frame, index, masks[index], episode_start=index in (0, 2),
                       executed_action=(index + 1) % 10)
        reports = [{"episode_id": 0, "outcome": "clear", "decisions": 2},
                   {"episode_id": 1, "outcome": "lives_exhausted", "decisions": 2}]
        writer.finish(reports, {"label": "synthetic", "nested": {"owned": True}})
        return frames, masks

    def open_dataset(self):
        dataset = DemonstrationDataset(self.path)
        self.addCleanup(dataset.close)
        return dataset

    def test_bit_exact_history_reset_and_unordered_duplicate_batch(self):
        frames, masks = self.make_corpus()
        dataset = self.open_dataset()
        self.assertEqual(len(dataset), 4)
        batch = dataset.batch([3, 0, 2, 1, 3])
        self.assertEqual(batch["observations"].shape, (5, 94, 26, 26))
        self.assertEqual(batch["observations"].dtype, np.float32)
        self.assertEqual(batch["actions"].dtype, np.int64)
        self.assertEqual(batch["action_masks"].dtype, np.bool_)
        self.assertEqual(batch["episode_ids"].dtype, np.int64)
        for row, index in enumerate((3, 0, 2, 1, 3)):
            expected = np.concatenate((np.zeros_like(frames[0]) if index in (0, 2)
                                       else frames[index - 1], frames[index]))
            self.assertEqual(batch["observations"][row].tobytes(), expected.tobytes())
        np.testing.assert_array_equal(batch["actions"], [3, 0, 2, 1, 3])
        np.testing.assert_array_equal(batch["action_masks"], masks[[3, 0, 2, 1, 3]])
        np.testing.assert_array_equal(dataset.episode_ids, [0, 0, 1, 1])
        np.testing.assert_array_equal(np.load(self.path / "executed_actions.npy"), [1, 2, 3, 4])
        np.testing.assert_array_equal(np.load(self.path / "outcomes.npy"), ["clear", "lives_exhausted"])

    def test_writer_and_batch_own_data(self):
        writer = DemonstrationWriter(self.path)
        self.addCleanup(writer.close)
        frame = np.ones((47, 26, 26), np.float32)
        masks = np.ones(10, bool)
        writer.add(frame, 1, masks, episode_start=True, executed_action=2)
        frame.fill(9)
        masks.fill(False)
        writer.finish([{"outcome": "clear"}], {"nested": {"owned": True}})
        dataset = self.open_dataset()
        first = dataset.batch([0])
        first["observations"].fill(99)
        first["actions"].fill(9)
        first["action_masks"].fill(False)
        first["episode_ids"].fill(99)
        metadata = dataset.metadata
        metadata["nested"]["owned"] = False
        second = dataset.batch([0])
        self.assertTrue(np.all(second["observations"][:, :47] == 0))
        self.assertTrue(np.all(second["observations"][:, 47:] == 1))
        self.assertEqual(second["actions"].tolist(), [1])
        self.assertTrue(second["action_masks"].all())
        self.assertEqual(second["episode_ids"].tolist(), [0])
        self.assertTrue(dataset.metadata["nested"]["owned"])
        with self.assertRaises(ValueError):
            dataset.episode_ids[0] = 4

    def test_writer_rejects_invalid_labels_and_does_not_append_them(self):
        writer = DemonstrationWriter(self.path)
        self.addCleanup(writer.close)
        frame = np.zeros((47, 26, 26), np.float32)
        masks = np.ones(10, bool)
        masks[4] = False
        for teacher, executed in ((4, 0), (0, 4), (10, 0), (-1, 0), (True, 0)):
            with self.assertRaises(ValueError):
                writer.add(frame, teacher, masks, episode_start=True, executed_action=executed)
        with self.assertRaises(ValueError):
            writer.add(frame, 0, masks, episode_start=False, executed_action=0)
        with self.assertRaises(ValueError):
            writer.add(frame, 0, np.ones(15, bool), episode_start=True, executed_action=0)
        with self.assertRaises(ValueError):
            writer.add(frame.astype(np.float64), 0, masks, episode_start=True, executed_action=0)
        writer.add(frame, 0, masks, episode_start=True, executed_action=1)
        writer.finish([{"outcome": "timeout"}], {})
        self.assertEqual(len(self.open_dataset()), 1)

    def test_dataset_rejects_invalid_indices_and_owns_empty_batch(self):
        self.make_corpus()
        dataset = self.open_dataset()
        for indices in ([-1], [4], [0.5], [True], [[0]]):
            with self.assertRaises((ValueError, IndexError)):
                dataset.batch(indices)
        self.assertEqual(dataset.batch([])["observations"].shape, (0, 94, 26, 26))

    def test_fresh_output_and_finish_are_not_overwritable(self):
        self.make_corpus()
        with self.assertRaises(FileExistsError):
            DemonstrationWriter(self.path)

    def test_loader_rejects_corrupted_label_mask(self):
        self.make_corpus()
        masks = np.load(self.path / "action_masks.npy")
        masks[0, 0] = False
        np.save(self.path / "action_masks.npy", masks)
        with self.assertRaises(ValueError):
            DemonstrationDataset(self.path)

    def test_loader_rejects_truncated_payload(self):
        self.make_corpus()
        payload = self.path / "frames.bin"
        payload.write_bytes(payload.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            DemonstrationDataset(self.path)

    def test_stream_is_compact_and_metadata_is_explicit(self):
        self.make_corpus()
        dataset = self.open_dataset()
        self.assertLess((self.path / "frames.bin").stat().st_size, 4 * 47 * 26 * 26 * 4 // 10)
        self.assertEqual(dataset.metadata["frame_shape"], [47, 26, 26])
        self.assertEqual(dataset.metadata["history_order"], ["previous", "current"])
        self.assertEqual(dataset.metadata["samples"], 4)
        self.assertEqual(dataset.metadata["format"], "native-run-demonstrations-v1")
        self.assertEqual(len(json.loads((self.path / "episode_reports.json").read_text())), 2)


if __name__ == "__main__":
    unittest.main()
