"""Precision corpus tests: synthetic only, no learning data or environment calls."""
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from within_stage.demonstrations import PrecisionDataset, PrecisionWriter


class PrecisionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("runs") / ("precision-data-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.path = self.root / "corpus"
        self.addCleanup(shutil.rmtree, self.root)

    def write_fixture(self):
        frames = np.zeros((4, 55, 26, 26), np.float32)
        frames[0].view(np.uint32)[0, 0, :4] = [0x80000000, 0x7fc00001, 0x3f800000, 0xff800000]
        frames[1, 47, 2:5, 3:6] = 2 / 7
        frames[2, 54, 4, 9] = 6 / 7
        frames[3, 52] = .75
        rams = np.arange(4 * 2048, dtype=np.uint16).astype(np.uint8).reshape(4, 2048)
        rams[:, 0] = np.arange(4)
        masks = np.ones((4, 10), bool)
        masks[1, 9] = False
        writer = PrecisionWriter(self.path)
        self.addCleanup(writer.close)
        for i in range(4):
            writer.add(frames[i], i, masks[i], ram=rams[i], frame=(i % 2) * 4,
                       episode_start=i in (0, 2), executed_action=i + 1)
        writer.finish([{"episode_id": 0, "outcome": "clear", "decisions": 2},
                       {"episode_id": 1, "outcome": "base_lost", "decisions": 2}],
                      {"role": "synthetic", "nested": {"owned": True}})
        return frames, rams, masks

    def open_dataset(self):
        data = PrecisionDataset(self.path)
        self.addCleanup(data.close)
        return data

    def test_exact_codec_two_stack_order_reset_and_ram_audit(self):
        frames, rams, masks = self.write_fixture()
        data = self.open_dataset()
        self.assertEqual(len(data), 4)
        indices = [3, 0, 2, 1, 3]
        batch = data.batch(indices)
        self.assertEqual(set(batch), {"observations", "actions", "action_masks", "episode_ids"})
        self.assertEqual(batch["observations"].shape, (5, 110, 26, 26))
        self.assertEqual(batch["observations"].dtype, np.float32)
        self.assertEqual(batch["actions"].dtype, np.int64)
        self.assertEqual(batch["action_masks"].dtype, np.bool_)
        self.assertEqual(batch["episode_ids"].dtype, np.int64)
        for row, index in enumerate(indices):
            expected = np.concatenate((np.zeros_like(frames[0]) if index in (0, 2)
                                       else frames[index - 1], frames[index]))
            self.assertEqual(batch["observations"][row].tobytes(), expected.tobytes())
        audit = data.audit(indices)
        self.assertEqual(audit["ram"].dtype, np.uint8)
        np.testing.assert_array_equal(audit["ram"], rams[indices])
        np.testing.assert_array_equal(audit["frames"], [4, 0, 0, 4, 4])
        np.testing.assert_array_equal(batch["action_masks"], masks[indices])
        np.testing.assert_array_equal(data.episode_ids, [0, 0, 1, 1])

    def test_writer_and_reader_arrays_are_owned(self):
        writer = PrecisionWriter(self.path)
        self.addCleanup(writer.close)
        frame = np.ones((55, 26, 26), np.float32)
        ram = np.ones(2048, np.uint8)
        masks = np.ones(10, bool)
        writer.add(frame, 1, masks, ram=ram, frame=0, episode_start=True, executed_action=2)
        frame.fill(9); ram.fill(9); masks.fill(False)
        writer.finish([{"outcome": "timeout"}], {"nested": {"owned": True}})
        data = self.open_dataset()
        batch, audit = data.batch([0]), data.audit([0])
        batch["observations"].fill(99)
        batch["actions"].fill(9)
        batch["action_masks"].fill(False)
        audit["ram"].fill(8)
        data.metadata["nested"]["owned"] = False
        self.assertTrue(np.all(data.batch([0])["observations"][:, 55:] == 1))
        self.assertTrue(np.all(data.audit([0])["ram"] == 1))
        self.assertTrue(data.metadata["nested"]["owned"])
        with self.assertRaises(ValueError):
            data.episode_ids[0] = 8

    def test_bad_masks_ram_shape_frames_and_first_boundary_are_rejected(self):
        writer = PrecisionWriter(self.path)
        self.addCleanup(writer.close)
        obs, ram, masks = np.zeros((55, 26, 26), np.float32), np.zeros(2048, np.uint8), np.ones(10, bool)
        masks[3] = False
        for action in (-1, 3, 10, True):
            with self.assertRaises(ValueError):
                writer.add(obs, action, masks, ram=ram, frame=0, episode_start=True, executed_action=0)
        for extra in ({"ram": np.zeros(2047, np.uint8)}, {"frame": -1}, {"episode_start": False}):
            options = dict(ram=ram, frame=0, episode_start=True, executed_action=0)
            options.update(extra)
            with self.assertRaises(ValueError):
                writer.add(obs, 0, masks, **options)
        with self.assertRaises(ValueError):
            writer.add(np.zeros((47, 26, 26), np.float32), 0, masks,
                       ram=ram, frame=0, episode_start=True, executed_action=0)

    def test_hash_integrity_rejects_ram_or_payload_mutation(self):
        self.write_fixture()
        path = self.path / "ram.bin"
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaises(ValueError):
            PrecisionDataset(self.path)
        path.write_bytes(original)
        path = self.path / "observations.bin"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaises(ValueError):
            PrecisionDataset(self.path)

    def test_schema_and_source_hashes_are_explicit_and_old_format_rejected(self):
        self.write_fixture()
        data = self.open_dataset()
        self.assertEqual(data.metadata["schema"]["frame_shape"], [55, 26, 26])
        self.assertEqual(data.metadata["schema"]["batch_shape"], [110, 26, 26])
        self.assertEqual(len(data.metadata["schema_sha256"]), 64)
        self.assertTrue(data.metadata["sources"]["within_stage\\observation.py"])
        self.assertEqual(data.metadata["ram_role"], "audit_only_not_model_input_or_reset")
        from learning_lab.demonstrations import DemonstrationDataset
        with self.assertRaises(ValueError):
            DemonstrationDataset(self.path)

    def test_batch_invalid_indices_and_empty_shape(self):
        self.write_fixture()
        data = self.open_dataset()
        for indices in ([-1], [4], [0.5], [True], [[0]]):
            with self.assertRaises((ValueError, IndexError)):
                data.batch(indices)
        self.assertEqual(data.batch([])["observations"].shape, (0, 110, 26, 26))


if __name__ == "__main__":
    unittest.main()
