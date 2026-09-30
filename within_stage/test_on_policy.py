"""Fake/CPU-only on-policy contracts; no real ROM/native/GPU work."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from within_stage import on_policy as online
from policy import Grid26CNN
from within_stage.test_training_env import FakeEarly


class FakeBudget:
    def __init__(self):
        self.rows = []

    def reserve(self, label, amount):
        self.rows.append(dict(label=label, reserved=amount, actual=None))
        return str(len(self.rows) - 1)

    def settle(self, token, actual):
        self.rows[int(token)]["actual"] = actual

    def snapshot(self):
        return dict(remaining_seconds=1000., max_interactions=4000000,
                    remaining_interactions=4000000, deadline_epoch=time.time()+1000)


class OnPolicyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("runs") / ("onpolicy-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        torch.set_num_threads(1)

    def test_chunk_adapter_charges_boot_and_control_once_and_nested_eval_settlement(self):
        shared = FakeBudget()
        allowance = online.TrialAllowance(shared, "trial", 150000, time.monotonic()+1000)
        account = online.TrainingAccount(allowance)
        for kind in ("raw", "raw", "agent"):
            account.before(kind); account.after(kind)
        self.assertEqual(allowance.used, 3)
        token = allowance.reserve("dev", 1000)
        self.assertEqual(allowance.available, 148997)
        allowance.settle(token, 90)
        self.assertEqual(allowance.used, 93)
        allowance.finish(certain=True)
        self.assertEqual(shared.rows[0]["actual"], 93)

    def test_unknown_failure_keeps_parent_reservation_fully_charged(self):
        shared = FakeBudget()
        allowance = online.TrialAllowance(shared, "failed", 150000, time.monotonic()+1000)
        account = online.TrainingAccount(allowance)
        account.before("raw")
        allowance.finish(certain=False)
        self.assertIsNone(shared.rows[0]["actual"])

    def test_tight_allowance_stops_before_boot_not_just_after_control(self):
        allowance = online.TrialAllowance(FakeBudget(), "small", 2, time.monotonic()+1000)
        account = online.TrainingAccount(allowance)
        account.before("raw"); account.after("raw")
        account.before("raw"); account.after("raw")
        with self.assertRaises(online.AllowanceStop):
            account.before("agent")
        self.assertEqual(allowance.used, 2)
        self.assertEqual(account.raw_initialization_completed, 2)
        self.assertEqual(account.agent_attempted, 0)

    def test_profile_is_precision55_separate_critic_and_small_ppo_update(self):
        profile = online.contract()
        self.assertEqual(profile["channels_per_frame"], 55)
        self.assertEqual(profile["observation_shape"], [110, 26, 26])
        self.assertFalse(profile["share_features_extractor"])
        self.assertEqual(profile["ppo"]["learning_rate"], 5e-5)
        self.assertEqual(profile["ppo"]["clip_range"], .1)
        self.assertEqual(profile["ppo"]["n_steps"], 256)

    def test_dev_schedule_uses_only_first6_representatives_and_movementstress(self):
        split = {"development": {"representative_config_indices": list(range(20,32))}}
        plan = online.development_plan(split)
        self.assertEqual(len(plan), 12)
        self.assertEqual({r["config_index"] for r in plan}, set(range(20,26)))
        for i in range(6):
            rows = [r for r in plan if r["config_index"] == 20+i]
            self.assertEqual([r["intervention_decision"] for r in rows], [None,150])
            self.assertEqual(rows[1]["noise_seed"], 801+i)

    def test_wall_stop_precedes_any_counter_charge(self):
        allowance = online.TrialAllowance(FakeBudget(), "past", 10, time.monotonic()+1000)
        allowance.deadline = 0
        with self.assertRaises(online.AllowanceStop):
            online.TrainingAccount(allowance).before("raw")
        self.assertEqual(allowance.used, 0)

    def test_development_helper_receives_local_budget_not_parent_and_exact_plan(self):
        split = self.root / "split.json"
        split.write_text('{"development":{"representative_config_indices":[1,2,3,4,5,6,7]}}')
        allowance = online.TrialAllowance(FakeBudget(), "evals", 150000, time.monotonic()+1000)
        seen = []
        def evaluate(model, **kwargs):
            seen.append(kwargs)
            token = kwargs["budget"].reserve(str(len(seen)), 1000)
            kwargs["budget"].settle(token, 100)
            return dict(intervention_decision=kwargs["intervention_decision"],
                        ordinary_model_clear=False, recovery_success=False, episode={"kills":1})
        with patch.object(online,"file_sha256",return_value=online.FROZEN_SPLIT_SHA256), \
                patch.object(online,"evaluate_development_case",side_effect=evaluate):
            result = online._evaluate(object(),self.root/"parent.zip",split,self.root/"dev",allowance)
        self.assertEqual(allowance.used,1200)
        self.assertEqual(len(seen),12)
        self.assertTrue(all(x["budget"] is allowance for x in seen))
        self.assertEqual(result["normal_cases"],6)
        self.assertEqual(result["stress_cases"],6)
        self.assertEqual({x["config_index"] for x in seen},set(range(1,7)))

    def test_legacy_or_bad_initial_checkpoint_rejected_before_reserving(self):
        checkpoint=self.root/"legacy.zip"
        checkpoint.write_bytes(b"not a precision checkpoint")
        checkpoint.with_suffix(".manifest.json").write_text('{"format":"legacy94"}')
        budget=FakeBudget()
        with self.assertRaises(ValueError):
            online.run_on_policy(checkpoint,self.root/"output",
                Path("runs")/"20260930-within-stage"/"frozen-split.json",budget,steps=1024)
        self.assertFalse(budget.rows)
        self.assertFalse((self.root/"output").exists())

    def test_pending_development_failure_keeps_conservative_outer_charge(self):
        shared=FakeBudget()
        allowance=online.TrialAllowance(shared,"uncertain-dev",150000,time.monotonic()+1000)
        allowance.reserve("failed-dev",15000)
        allowance.finish(certain=True)
        self.assertIsNone(shared.rows[0]["actual"])
        self.assertEqual(allowance.available,135000)


class PrecisionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from learning_lab.demonstrations import DemonstrationWriter
        from learning_lab.imitation import train_imitation
        from within_stage.precision_policy import make_precision_model, save_precision_checkpoint, precision_schema
        torch.set_num_threads(1)
        cls.root=Path("runs")/("online-precision-fixture-"+uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(shutil.rmtree,cls.root)
        writer=DemonstrationWriter(cls.root/"legacy-data")
        try:
            for episode in range(3):
                for step in range(2):
                    writer.add(np.full((47,26,26),.1,np.float32),2,np.ones(10,bool),
                               episode_start=step==0,executed_action=2)
            writer.finish([{"outcome":"timeout"} for _ in range(3)],{"fixture":True})
        finally:writer.close()
        previous=train_imitation([cls.root/"legacy-data"],cls.root/"legacy",epochs=1,batch_size=2)
        if previous["status"]!="completed":raise AssertionError(previous)
        model,record=make_precision_model(previous["checkpoints"]["final"]["path"],device="cpu")
        model.policy.optimizer.zero_grad(set_to_none=True)
        loss=model.policy.predict_values(torch.full((1,110,26,26),.1)).square().mean()
        loss.backward()
        model.policy.optimizer.step()
        cls.initial=cls.root/"initial.zip"
        save_precision_checkpoint(model,cls.initial,migration_record=record,
                                  dataset_config={"schema":precision_schema(),"datasets":[],"split":None})

    def setUp(self):
        self.output=self.root/uuid.uuid4().hex
        self.output.mkdir()
        self.allowance=online.TrialAllowance(FakeBudget(),"cpu",150000,time.monotonic()+1000)
        self.account=online.TrainingAccount(self.allowance)

    def runtime(self):
        parent=online.load_precision_checkpoint(self.initial,device="cpu")
        from within_stage import training_env
        with patch.object(training_env,"make_opening_env",
                          side_effect=lambda pool,index,*,account,max_frames:FakeEarly(index,account)):
            result=online.OnlineRuntime(parent,Path("runs")/"20260930-within-stage"/"frozen-split.json",
                self.output,self.account,steps=8,seed=0,noise_probability=1.,device="cpu",
                n_steps=2,batch_size=4)
            # Future automatic resets must also stay synthetic throughout the test.
        self.patcher=patch.object(training_env,"make_opening_env",
                          side_effect=lambda pool,index,*,account,max_frames:FakeEarly(index,account))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(result.close)
        return parent,result

    def test_tiny_precision_rollout_preserves_adam_55codec_and_intended_actions(self):
        parent,runtime=self.runtime()
        self.assertEqual(runtime.initial_precision_optimizer_sha256,
                         online.state_sha256(parent.policy.optimizer.state_dict()))
        self.assertEqual(runtime.model.rollout_buffer.channels_per_frame,55)
        self.assertEqual(runtime.model.rollout_buffer.frame_shape,(55,26,26))
        self.assertFalse(runtime.model.policy.share_features_extractor)
        self.assertTrue(all(p.requires_grad for p in runtime.model.policy.parameters()))
        self.assertEqual(self.account.raw_initialization_completed,8)
        self.assertTrue(runtime.collect())
        self.assertEqual(self.account.agent_completed,8)
        self.assertEqual(runtime.optimized_steps,0)
        runtime.optimize()
        self.assertEqual(runtime.optimized_steps,8)
        self.assertEqual(sum(runtime.callback.intended_counts.values()),8)
        self.assertEqual(runtime.callback.interventions,8)
        self.assertFalse(torch.cuda.is_initialized())

    def test_strict_precision_save_reload_and_partial_optimizer_rollback(self):
        parent,runtime=self.runtime()
        runtime.collect(); runtime.optimize()
        # Synthetic progress tags only; no real environment interaction occurred.
        runtime.optimized_steps=1024
        provenance=dict(sources=online.source_hashes(),parent=online._parent_record(self.initial,parent))
        record=online.save_online(runtime,self.output/"fixture.zip",provenance,"cpu_fixture")
        loaded=online.load_on_policy_checkpoint(record["path"])
        self.assertEqual(loaded.gamma,1.)
        self.assertEqual(loaded.rollout_buffer.channels_per_frame,55)
        self.assertEqual(loaded.learning_rate,5e-5)
        for name,value in runtime.model.policy.state_dict().items():
            torch.testing.assert_close(value,loaded.policy.state_dict()[name],rtol=0,atol=0)
        before=online.state_sha256(runtime.model.policy.state_dict())
        def partial():
            with torch.no_grad():next(runtime.model.policy.parameters()).add_(2)
            raise online.AllowanceStop("partial update")
        with patch.object(runtime.model,"train",side_effect=partial):
            with self.assertRaises(online.AllowanceStop):runtime.optimize()
        self.assertEqual(before,online.state_sha256(runtime.model.policy.state_dict()))
        self.assertEqual(runtime.optimized_steps,1024)

    def test_deadline_blocks_real_reset_work_in_fake_runtime(self):
        _,runtime=self.runtime()
        self.allowance.deadline=0
        before=self.account.attempted
        with self.assertRaises(online.AllowanceStop):
            runtime.collect()
        self.assertEqual(self.account.attempted,before)
        self.assertEqual(runtime.optimized_steps,0)

if __name__ == "__main__":
    unittest.main()
