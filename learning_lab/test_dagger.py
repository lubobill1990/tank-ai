"""Fake-environment DAgger checks; zero ROM/native environment decisions or GPU work."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from learning_lab import dagger


class FakeEnv:
    instances = []

    def __init__(self, *, stage, objective, max_frames, frame_skip):
        self.stage, self.actions, self.closed = stage, [], False
        self.frame_skip = frame_skip
        self.buffer = np.zeros((47, 26, 26), np.float32)
        self.instances.append(self)

    def reset(self, *, seed):
        self.seed = seed
        self.state = SimpleNamespace(frame=0, marker=10 + seed)
        self.buffer.fill(self.state.marker)
        return self.buffer, self.info("")

    def action_masks(self):
        masks = np.zeros(15, bool)
        masks[1:4] = True
        return masks

    def info(self, outcome):
        return dict(outcome=outcome, frames=self.state.frame, kills=len(self.actions),
                    deaths=0, stage=self.stage)

    def step(self, action):
        assert self.action_masks()[action]
        self.actions.append(action)
        self.state.frame += 4
        self.state.marker += action
        self.buffer.fill(self.state.marker)
        done = len(self.actions) >= 3
        return self.buffer, 0., done, False, self.info("clear" if done else "")

    def close(self):
        self.closed = True


class FakeTeacher:
    instances = []

    def __init__(self):
        self.calls, self.resets = [], 0
        self.mode = "fake-current-state"
        self.instances.append(self)

    def reset(self):
        self.resets += 1

    def act(self, state, masks):
        self.calls.append((state.frame, state.marker))
        return 1 + state.marker % 2


class FakePolicy:
    device = "cpu"

    def __init__(self, action=3, on_predict=None):
        self.action, self.on_predict = action, on_predict
        self.calls = []

    def predict(self, observation, *, deterministic, action_masks):
        assert deterministic is True
        assert observation.shape == (94, 26, 26)
        assert action_masks.shape == (10,)
        self.calls.append((observation.copy(), action_masks.copy()))
        if self.on_predict:
            self.on_predict()
        return np.asarray(self.action), None


class FakeWriter:
    instances = []

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows, self.reports, self.metadata = [], None, None
        self.instances.append(self)

    def add(self, observation47, teacher_action, action_masks, *, episode_start,
            executed_action):
        assert action_masks[teacher_action] and action_masks[executed_action]
        self.rows.append(dict(observation=observation47.copy(), teacher=teacher_action,
                              masks=action_masks.copy(), start=episode_start,
                              executed=executed_action))

    def finish(self, episode_reports, metadata):
        self.reports, self.metadata = episode_reports, metadata
        return metadata

    def close(self):
        pass


class FakeMonitor:
    def __init__(self, path, *, device):
        assert device == "cpu"
        Path(path).touch(exist_ok=False)

    def check(self, phase, **kwargs):
        pass


class DaggerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("runs") / ("dagger-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.output = self.root / "corpus"
        FakeEnv.instances, FakeTeacher.instances, FakeWriter.instances = [], [], []
        self.clock = [100.]
        self.patches = [
            patch.object(dagger, "BattleCityEnv", FakeEnv),
            patch.object(dagger, "TeacherController", FakeTeacher),
            patch.object(dagger, "DemonstrationWriter", FakeWriter),
            patch.object(dagger, "ResourceMonitor", FakeMonitor),
            patch.object(dagger, "_GLOBAL_DEADLINE_EPOCH", 1e20),
            patch.object(dagger.time, "monotonic", side_effect=lambda: self.clock[0]),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def collect(self, model=None, **kwargs):
        defaults = dict(max_episodes=2, max_decisions=6, deadline=1000.,
                        teacher_probability=0.)
        defaults.update(kwargs)
        return dagger.collect_dagger(model or FakePolicy(), self.output, **defaults)

    def test_pre_action_teacher_labels_and_exact_zero_reset_history(self):
        model = FakePolicy()
        report = self.collect(model)
        rows = FakeWriter.instances[-1].rows
        self.assertEqual(report["actual_decisions"], 6)
        self.assertEqual(report["student_control_decisions"], 6)
        self.assertEqual(report["teacher_control_decisions"], 0)
        self.assertEqual(len(rows), 6)
        for index, row in enumerate(rows):
            stack = model.calls[index][0]
            self.assertEqual(stack[47:].tobytes(), row["observation"].tobytes())
            expected_previous = np.zeros((47, 26, 26), np.float32) if index % 3 == 0 \
                else rows[index - 1]["observation"]
            self.assertEqual(stack[:47].tobytes(), expected_previous.tobytes())
            marker = int(row["observation"][0, 0, 0])
            self.assertEqual(row["teacher"], 1 + marker % 2)
            self.assertEqual(row["executed"], 3)
            self.assertEqual(row["start"], index % 3 == 0)
        self.assertEqual(sum(len(t.calls) for t in FakeTeacher.instances), 6)
        self.assertEqual(sum(t.resets for t in FakeTeacher.instances), 2)
        self.assertTrue(all(env.closed for env in FakeEnv.instances))

    def test_teacher_mixture_controls_are_distinct_from_model_wins(self):
        report = self.collect(teacher_probability=1.)
        self.assertEqual(report["teacher_control_decisions"], 6)
        self.assertEqual(report["student_control_decisions"], 0)
        self.assertEqual(report["overridden_student_decisions"], 6)
        self.assertEqual(report["training_trajectory_clears"], 2)
        self.assertEqual(report["verified_model_wins"], 0)
        self.assertFalse(report["formal_model_validation"])
        self.assertIn("not model win", report["label"].lower())
        for row in FakeWriter.instances[-1].rows:
            self.assertEqual(row["teacher"], row["executed"])
        np.testing.assert_array_equal(np.load(self.output / "student_actions.npy"), [3] * 6)
        self.assertTrue(np.load(self.output / "teacher_interventions.npy").all())

    def test_pure_student_clears_remain_collection_not_formal_validation(self):
        report = self.collect()
        self.assertEqual(report["training_trajectory_clears"], 2)
        self.assertEqual(report["verified_model_wins"], 0)
        self.assertFalse(report["formal_model_validation"])
        self.assertFalse(np.load(self.output / "teacher_interventions.npy").any())

    def test_beta_rng_and_contexts_reproduce_control_sources(self):
        report = self.collect(teacher_probability=.5, rng_seed=271)
        expected = np.random.default_rng(271).random(6) < .5
        np.testing.assert_array_equal(np.load(self.output / "teacher_interventions.npy"), expected)
        contexts = [json.loads(line) for line in (self.output / "contexts.jsonl").read_text().splitlines()]
        self.assertEqual(len(contexts), 6)
        for flag, row in zip(expected, contexts):
            self.assertEqual(row["control_source"], "teacher" if flag else "student")
            self.assertEqual(row["executed_action"],
                             row["teacher_action"] if flag else row["student_action"])
            self.assertTrue(row["step_completed"])
        self.assertEqual(report["teacher_control_decisions"], int(expected.sum()))

    def test_invalid_student_action_fails_without_rom_step(self):
        report = self.collect(FakePolicy(action=9))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["actual_decisions"], 0)
        self.assertEqual(sum(len(e.actions) for e in FakeEnv.instances), 0)
        self.assertIn("mask", report["error"].lower())

    def test_invalid_teacher_action_fails_before_student_or_step(self):
        model = FakePolicy()
        with patch.object(FakeTeacher, "act", return_value=9):
            report = self.collect(model)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["actual_decisions"], 0)
        self.assertFalse(model.calls)
        self.assertFalse(FakeWriter.instances[-1].rows)

    def test_teacher_selection_is_counted_even_when_student_agrees(self):
        report = self.collect(FakePolicy(action=1), teacher_probability=1., max_decisions=1)
        self.assertEqual(report["actual_decisions"], 1)
        self.assertEqual(report["teacher_control_decisions"], 1)
        self.assertEqual(report["overridden_student_decisions"], 0)
        self.assertTrue(np.load(self.output / "teacher_interventions.npy")[0])

    def test_resource_pressure_aborts_before_any_step(self):
        def check(_, phase, **kwargs):
            if phase == "collection":
                raise dagger.ResourcePressureError("synthetic pressure")
        with patch.object(FakeMonitor, "check", check):
            report = self.collect()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stop_reason"], "resource_pressure")
        self.assertEqual(report["actual_decisions"], 0)
        self.assertTrue(all(env.closed for env in FakeEnv.instances))

    def test_step_exception_keeps_attempt_and_completion_counts_distinct(self):
        with patch.object(FakeEnv, "step", side_effect=RuntimeError("synthetic step failure")):
            report = self.collect()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["actual_decisions"], 1)
        self.assertEqual(report["completed_decisions"], 0)
        contexts = [json.loads(line) for line in (self.output / "contexts.jsonl").read_text().splitlines()]
        self.assertFalse(contexts[0]["step_completed"])

    def test_decision_cap_produces_partial_episode_without_extra_step(self):
        report = self.collect(max_decisions=2)
        self.assertEqual(report["actual_decisions"], 2)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stop_reason"], "decision_budget")
        self.assertEqual(report["episodes"][0]["outcome"], "decision_budget")

    def test_deadline_after_prediction_prevents_env_step(self):
        model = FakePolicy(on_predict=lambda: self.clock.__setitem__(0, 1001.))
        report = self.collect(model)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stop_reason"], "deadline")
        self.assertEqual(report["actual_decisions"], 0)
        self.assertFalse(FakeWriter.instances[-1].rows)

    def test_sealed_stages_and_new_openings_rejected_before_creation(self):
        for bad in (dict(stages=(8,)), dict(stages=(18,)), dict(stages=(28,)),
                    dict(stages=(4,)), dict(seeds=(3,)), dict(seeds=(23,)),
                    dict(seeds=(True,)), dict(stages=()), dict(seeds=(0, 0)),
                    dict(teacher_probability=-.1), dict(teacher_probability=1.1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.collect(**bad)
        self.assertFalse(self.output.exists())
        self.assertFalse(FakeEnv.instances)

    def test_global_deadline_caps_absent_local_deadline_without_reset(self):
        with patch.object(dagger, "_GLOBAL_DEADLINE_EPOCH", 10.), \
                patch.object(dagger.time, "time", return_value=11.):
            report = self.collect(deadline=None)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stop_reason"], "deadline")
        self.assertFalse(FakeEnv.instances)

    def test_cli_calls_verified_imitation_loader_on_cpu(self):
        model_path = self.root / "lab-bc.zip"
        model_path.write_bytes(b"fake-checkpoint-never-deserialized")
        model_path.with_suffix(".manifest.json").write_text("{}")
        model = FakePolicy()
        with patch("learning_lab.imitation.load_imitation_checkpoint", return_value=model) as loader, \
                redirect_stdout(io.StringIO()):
            code = dagger.main([
                "--model", str(model_path), "--output", str(self.output),
                "--max-episodes", "1", "--max-decisions", "3",
                "--teacher-probability", "0", "--deadline", "1000",
            ])
        self.assertEqual(code, 0)
        loader.assert_called_once_with(model_path, device="cpu")
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["checkpoint_provenance"]["loader"],
                         "learning_lab.imitation.load_imitation_checkpoint")
        self.assertEqual(report["actual_decisions"], 3)

    def test_cli_rejects_sealed_stage_before_loading_model(self):
        with patch("learning_lab.imitation.load_imitation_checkpoint") as loader:
            with self.assertRaises(ValueError):
                dagger.main(["--model", "unused.zip", "--output", str(self.output),
                             "--stages", "8"])
        loader.assert_not_called()

    def test_real_writer_format_loads_with_existing_dataset(self):
        from learning_lab.demonstrations import DemonstrationDataset, DemonstrationWriter
        with patch.object(dagger, "DemonstrationWriter", DemonstrationWriter):
            report = self.collect()
        dataset = DemonstrationDataset(self.output)
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), report["actual_decisions"])
        batch = dataset.batch([0, 3])
        self.assertTrue(np.all(batch["observations"][:, :47] == 0))
        self.assertEqual(dataset.episode_ids.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertFalse(dataset.metadata["formal_model_validation"])


if __name__ == "__main__":
    unittest.main()
