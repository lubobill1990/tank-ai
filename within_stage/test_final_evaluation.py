"""Synthetic one-shot FINAL evaluator tests; never use real FINAL receipt or ROM."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
import torch

from within_stage import final_evaluation as final


class FakeModel:
    def __init__(self, kind="precision"):
        self.policy = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = (gym.spaces.Box(-1, 1, (110,26,26), np.float32) if kind == "precision"
                                  else final.entity_space())
        self.seen = []

    def predict(self, observation, *, deterministic, action_masks):
        assert deterministic and action_masks[0]
        self.seen.append(deepcopy(observation))
        return np.asarray(0), None


class FakeRaw(gym.Env):
    observation_space = gym.spaces.Box(-1,1,(47,26,26),np.float32)
    action_space = gym.spaces.Discrete(10)
    max_frames = 12000
    instances = []
    mutate = False
    bad_clear = False
    long = False
    fail_step = False

    def __init__(self, account, index, model):
        self.account,self.index,self.model=account,index,model
        self.nes=SimpleNamespace(ram=np.full(2048,index,np.uint8))
        self.closed=False
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        self.account.before("raw");self.account.after("raw")
        self.tick=0
        self.state=SimpleNamespace(remaining_enemies=20,base_destroyed=False,lives=3,frame=0,
                                  game_over=False)
        return np.zeros((47,26,26),np.float32),{}

    def action_masks(self):
        return np.ones(10,bool)

    def step(self, action):
        self.account.before("agent")
        if self.fail_step:
            raise RuntimeError("synthetic uncertain step failure")
        self.tick+=1
        done=self.tick>=(102 if self.long else 1)
        self.state=SimpleNamespace(remaining_enemies=(1 if self.bad_clear else 0) if done else 20,
                                  base_destroyed=False,lives=3,frame=self.tick*4,game_over=False)
        if self.mutate:
            with torch.no_grad():self.model.policy.weight.add_(1)
        self.account.after("agent");self.account.agent_raw_frames+=4
        return np.zeros((47,26,26),np.float32),0.,done,False,dict(
            outcome="clear" if done else "",objective="full",frames=self.tick*4,
            kills=20 if done else 0,deaths=0,lives=3)

    def render(self):
        return np.zeros((4,4,3),np.uint8)

    def close(self):
        self.closed=True


class FakeObservation(gym.Wrapper):
    def __init__(self, raw, kind):
        super().__init__(raw);self.kind=kind

    def action_masks(self):
        return self.env.action_masks()

    def obs(self):
        if self.kind=="precision":return np.zeros((55,26,26),np.float32)
        return {"grid":np.zeros((110,26,26),np.float32),"entities":np.zeros((18,16),np.float32)}

    def reset(self, **kwargs):
        _,info=self.env.reset(**kwargs);return self.obs(),info

    def step(self, action):
        _,reward,term,trunc,info=self.env.step(action)
        return self.obs(),reward,term,trunc,info


class FakeBudget:
    def __init__(self, seconds=1000):
        self.rows=[];self.seconds=seconds
    def snapshot(self):
        return {"remaining_seconds":self.seconds,"deadline_epoch":time.time()+self.seconds,
                "max_interactions":4000000,"remaining_interactions":4000000}
    def reserve(self,label,amount):
        self.rows.append({"amount":amount,"actual":None});return str(len(self.rows)-1)
    def settle(self,token,amount):
        self.rows[int(token)]["actual"]=amount


class FakeMonitor:
    def __init__(self,path,**kwargs):Path(path).touch(exist_ok=False)
    def check(self,*args,**kwargs):pass


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root=Path("runs")/("synthetic-final-test-"+uuid.uuid4().hex)
        self.root.mkdir(parents=True);self.addCleanup(shutil.rmtree,self.root)
        self.pool={"stage":1,"handoff":150,"specs":[],"families":[]}
        for i in range(20):
            self.pool["specs"].append({"config_index":i,"family_id":i,
                "spec":{"title_wait":0,"stage_wait":0,"post_start_frames":150,"legacy_phase":None},
                "initial_ram_sha256":hashlib.sha256(np.full(2048,i,np.uint8).tobytes()).hexdigest()})
            self.pool["families"].append({"family_id":i,"config_indices":[i]})
        self.pool_path=self.root/"pool.json"
        self.pool_path.write_text(json.dumps(self.pool))
        self.split=self.root/"split.json"
        value={"version":"within-stage-family-split-v1","stage":1,"max_frames":12000,
               "post_start_frames":150,"channels_per_frame":55,"actions":10,"n_stack":2,
               "observation_version":final.OBSERVATION_VERSION,
               "pool_path":str(self.pool_path.resolve()),"pool_sha256":final._sha(self.pool_path),
               "final":{"family_ids":list(range(20)),"representative_config_indices":list(range(20)),
                        "all_config_indices":list(range(20))},
               "train":{"family_ids":[30],"all_config_indices":[30]},
               "development":{"family_ids":[31],"all_config_indices":[31]},
               "recovery_final_schedule":[{"config_index":i,"family_id":i,"decision_index":100 if i%2==0 else 200,
                                          "noise_seed":20261001000+i,"allowed_actions":[0,1,2,3,4]} for i in range(20)]}
        self.split.write_text(json.dumps(value))
        self.checkpoint=self.root/"model.zip";self.checkpoint.write_bytes(b"trusted-fake-model")
        self.checkpoint.with_suffix(".manifest.json").write_text("{}")
        self.model=FakeModel()
        FakeRaw.instances=[];FakeRaw.mutate=FakeRaw.bad_clear=FakeRaw.long=FakeRaw.fail_step=False
        patches=[
            patch.object(final,"RECEIPT_PATH",self.root/"within-stage-final-used.json"),
            patch.object(final,"FROZEN_SPLIT_SHA256",final._sha(self.split)),
            patch.object(final,"load_pool",return_value=deepcopy(self.pool)),
            patch.object(final,"_trusted_load",side_effect=lambda kind,path:deepcopy(self.model)),
            patch.object(final,"_source_hashes",return_value={"synthetic":"a"*64}),
            patch.object(final,"make_opening_env",side_effect=lambda pool,index,*,account,max_frames:
                         FakeRaw(account,index,self.model)),
            patch.object(final,"_wrap_observation",side_effect=lambda raw,kind,deadline:FakeObservation(raw,kind)),
            patch.object(final,"ResourceMonitor",FakeMonitor),
            patch.object(final,"GLOBAL_DEADLINE_EPOCH",1e20),
        ]
        for p in patches:p.start();self.addCleanup(p.stop)

    def selection(self, kind="precision"):
        if kind=="entity":self.model=FakeModel("entity")
        return final.make_final_selection(self.model,self.checkpoint,self.split,loader=kind,
                                           offsets_enabled=True if kind=="precision" else None)

    def test_plan40cases_and_missed_stress_never_inflates_qualification(self):
        selection=self.selection()
        self.assertFalse(final.RECEIPT_PATH.exists())
        self.assertEqual(len(selection["case_plan"]),40)
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"result",
                                         FakeBudget(),selection,self.split)
        self.assertEqual(report["status"],"completed")
        self.assertEqual(report["summary"]["normal_clears"],20)
        self.assertEqual(report["summary"]["recovery_successes"],0)
        self.assertEqual(report["summary"]["recovery_planned"],20)
        self.assertEqual(report["summary"]["recovery_applied"],0)
        self.assertFalse(report["summary"]["qualified"])
        self.assertEqual(report["accounting"]["attempted"],80)
        self.assertTrue(final.RECEIPT_PATH.exists())

    def test_existing_receipt_blocks_all_other_outputs_or_candidates(self):
        selection=self.selection()
        final.RECEIPT_PATH.write_text('{"status":"partial"}')
        with self.assertRaises(FileExistsError):
            final.run_final_evaluation(self.model,self.checkpoint,self.root/"other",
                                       FakeBudget(),selection,self.split)
        self.assertFalse(FakeRaw.instances)

    def test_bad_hash_or_final_partition_refused_before_receipt(self):
        selection=self.selection()
        changed=deepcopy(selection);changed["checkpoint_sha256"]="0"*64
        with self.assertRaises(ValueError):
            final.run_final_evaluation(self.model,self.checkpoint,self.root/"bad",
                                       FakeBudget(),changed,self.split)
        self.assertFalse(final.RECEIPT_PATH.exists())
        self.pool["specs"][0]["family_id"]=30
        with patch.object(final,"load_pool",return_value=self.pool),self.assertRaises(ValueError):
            self.selection()

    def test_forged_terminal_clear_is_failure_not_a_success(self):
        selection=self.selection();FakeRaw.bad_clear=True
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"badclear",
                                         FakeBudget(),selection,self.split)
        self.assertNotEqual(report["status"],"completed")
        self.assertFalse(report["summary"]["qualified"])
        self.assertEqual(report["summary"]["normal_clears"],0)
        self.assertEqual(len(report["cases"]),1)

    def test_model_mutation_invalidates_case_and_stops_all_followups(self):
        selection=self.selection();FakeRaw.mutate=True
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"mutation",
                                         FakeBudget(),selection,self.split)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(len(report["cases"]),1)
        self.assertFalse(report["summary"]["qualified"])

    def test_expired_deadline_creates_receipt_but_no_complete_claim_or_env(self):
        selection=self.selection()
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"expired",
                                         FakeBudget(seconds=0),selection,self.split)
        self.assertEqual(report["status"],"partial")
        self.assertTrue(final.RECEIPT_PATH.exists())
        self.assertFalse(FakeRaw.instances)
        self.assertFalse(report["summary"]["qualified"])

    def test_entity_dict_owns_history_precision_only_is_stacked(self):
        selection=self.selection("entity")
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"entity",
                                         FakeBudget(),selection,self.split)
        self.assertEqual(report["status"],"completed")
        self.assertTrue(all(isinstance(obs,dict) and obs["grid"].shape==(110,26,26)
                            for obs in self.model.seen))
        self.assertTrue(all(obs["entities"].shape==(18,16) for obs in self.model.seen))

    def test_partial_or_unapplied_cases_cannot_pass_planned_denominator(self):
        cases=[{"status":"completed","mode":"normal","ordinary_model_clear":True} for _ in range(20)]
        cases += [{"status":"completed","mode":"recovery","recovery_success":True,"intervention_applied":True} for _ in range(15)]
        summary=final.summarize(cases)
        self.assertEqual(summary["recovery_conditional_rate"],1.)
        self.assertEqual(summary["recovery_planned_rate"],.75)
        self.assertFalse(summary["qualified"])

    def test_actual_movement_stress_is_logged_but_unreached200_schedule_stays_failure(self):
        selection=self.selection();FakeRaw.long=True
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"applied",
                                         FakeBudget(),selection,self.split)
        self.assertEqual(report["status"],"completed")
        self.assertEqual(report["summary"]["normal_clears"],20)
        self.assertEqual(report["summary"]["recovery_applied"],10)
        self.assertEqual(report["summary"]["recovery_successes"],10)
        self.assertEqual(report["summary"]["recovery_conditional_rate"],1.)
        self.assertEqual(report["summary"]["recovery_planned_rate"],.5)
        self.assertFalse(report["summary"]["qualified"])
        cases=[r for r in report["cases"] if r["mode"]=="recovery" and r["intervention_applied"]]
        self.assertTrue(all(c["executed_actions"][100] in (1,2,3,4) for c in cases))
        self.assertTrue(all(c["actions"][100]==0 for c in cases))

    def test_precision_history_has_one_zero_prefix_not_extra_frame_stack(self):
        selection=self.selection()
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"precision",
                                         FakeBudget(),selection,self.split)
        self.assertEqual(report["status"],"completed")
        self.assertTrue(all(obs.shape==(110,26,26) for obs in self.model.seen))
        self.assertTrue(all(np.all(obs[:55]==0) for obs in self.model.seen))

    def test_source_mutation_after_selection_fails_before_receipt(self):
        selection=self.selection()
        with patch.object(final,"_source_hashes",return_value={"changed":"b"*64}):
            with self.assertRaises(ValueError):
                final.run_final_evaluation(self.model,self.checkpoint,self.root/"sources",
                                           FakeBudget(),selection,self.split)
        self.assertFalse(final.RECEIPT_PATH.exists())
        self.assertFalse(FakeRaw.instances)

    def test_unapproved_prediction_adapter_is_not_a_selected_plain_model(self):
        self.model.predict=lambda observation,**kwargs:(np.asarray(0),None)
        with self.assertRaises(ValueError):
            self.selection()
        self.assertFalse(final.RECEIPT_PATH.exists())

    def test_precision_offsets_must_be_explicit_and_entity_cannot_double_adapt(self):
        with self.assertRaises(ValueError):
            final.make_final_selection(self.model,self.checkpoint,self.split,loader="precision")
        self.model=FakeModel("entity")
        with self.assertRaises(ValueError):
            final.make_final_selection(self.model,self.checkpoint,self.split,
                                       loader="entity",offsets_enabled=False)
        self.assertFalse(final.RECEIPT_PATH.exists())

    def test_failed_inflight_step_keeps_reservation_and_blocks_repeat(self):
        selection=self.selection();FakeRaw.fail_step=True;budget=FakeBudget()
        report=final.run_final_evaluation(self.model,self.checkpoint,self.root/"failed-step",
                                         budget,selection,self.split)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["accounting"]["attempted"],2)
        self.assertEqual(report["accounting"]["completed"],1)
        self.assertIsNone(budget.rows[0]["actual"])
        self.assertEqual(len(report["unsettled_reservations"]),1)
        self.assertTrue((self.root/"failed-step"/"case-00"/"last-observed.npz").is_file())
        with self.assertRaises(FileExistsError):
            final.run_final_evaluation(self.model,self.checkpoint,self.root/"retry",
                                       budget,selection,self.split)

if __name__=="__main__":
    unittest.main()
