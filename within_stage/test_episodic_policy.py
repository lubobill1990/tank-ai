"""CPU synthetic long-credit tests; zero real environment or GPU calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
import torch
from native_rollout import NativeGPURunRolloutBuffer

from within_stage import episodic_policy as episodic
from within_stage import on_policy as short
from within_stage.test_on_policy import FakeBudget
from within_stage import test_on_policy as short_tests
from within_stage.test_training_env import FakeEarly


class EpisodicTests(unittest.TestCase):
    def test_only_trace_horizon_changes_ppo_profile(self):
        a,b=short.contract(),episodic.contract()
        changed={k for k in a["ppo"] if a["ppo"][k]!=b["ppo"][k]}
        self.assertEqual(changed,{"n_steps","gae_lambda"})
        self.assertEqual(b["ppo"]["n_steps"],2048)
        self.assertEqual(b["ppo"]["gae_lambda"],1.)
        self.assertEqual(b["rollout_controls"],8192)
        self.assertEqual(b["dev_every"],32768)
        self.assertEqual(b["whole_trial_cap"],200000)

    def test_terminal_reward_reaches_all_previous_actions_hand_computed(self):
        buffer=NativeGPURunRolloutBuffer(
            3,gym.spaces.Box(-1,1,(110,26,26),np.float32),gym.spaces.Discrete(10),
            device="cpu",gamma=1.,gae_lambda=1.,n_envs=1,channels_per_frame=55)
        for step,value in enumerate((.4,.2,.8)):
            buffer.add(np.zeros((1,110,26,26),np.float32),np.array([0]),
                       np.array([1. if step==2 else 0.],np.float32),np.array([step==0]),
                       torch.tensor([value]),torch.tensor([0.]),action_masks=np.ones((1,10),bool))
        buffer.compute_returns_and_advantage(torch.tensor([999.]),np.array([True]))
        np.testing.assert_allclose(buffer.returns[:,0],[1.,1.,1.],atol=1e-6)
        np.testing.assert_allclose(buffer.advantages[:,0],[.6,.8,.2],atol=1e-6)
        self.assertEqual(buffer.frame_shape,(55,26,26))

    def test_long_buffer_allocation_keeps55planes_not47(self):
        buffer=NativeGPURunRolloutBuffer(
            2048,gym.spaces.Box(-1,1,(110,26,26),np.float32),gym.spaces.Discrete(10),
            device="cpu",gamma=1.,gae_lambda=1.,n_envs=4,channels_per_frame=55)
        self.assertEqual(buffer.frame_shape,(55,26,26))
        self.assertEqual(buffer.buffer_size,2048)
        self.assertEqual(buffer.n_envs,4)
        self.assertFalse(hasattr(buffer,"observations"))

    def test_long_allowance_caps_bootstraps_and_keeps_unknown_failure_reserved(self):
        parent=FakeBudget()
        allowance=episodic.EpisodicAllowance(parent,"long",200000,time.monotonic()+100)
        account=short.TrainingAccount(allowance)
        account.before("raw");account.after("raw")
        self.assertEqual(allowance.used,1)
        allowance.finish(certain=False)
        self.assertIsNone(parent.rows[0]["actual"])
        self.assertEqual(parent.rows[0]["reserved"],200000)
        with self.assertRaises(ValueError):
            episodic.EpisodicAllowance(parent,"overspend",200001,time.monotonic()+100)

    def test_reset_boundary_does_not_send_later_episode_reward_backward(self):
        buffer=NativeGPURunRolloutBuffer(
            3,gym.spaces.Box(-1,1,(110,26,26),np.float32),gym.spaces.Discrete(10),
            device="cpu",gamma=1.,gae_lambda=1.,n_envs=1,channels_per_frame=55)
        for step,(reward,start) in enumerate(((0.,True),(-1.,False),(1.,True))):
            buffer.add(np.zeros((1,110,26,26),np.float32),np.array([0]),np.array([reward]),
                       np.array([start]),torch.tensor([0.]),torch.tensor([0.]),
                       action_masks=np.ones((1,10),bool))
        buffer.compute_returns_and_advantage(torch.tensor([99.]),np.array([True]))
        np.testing.assert_allclose(buffer.returns[:,0],[-1.,-1.,1.])


class LongRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the tiny authenticated CPU precision checkpoint fixture only.
        short_tests.PrecisionRuntimeTests.setUpClass.__func__(cls)

    def setUp(self):
        self.output=self.root/uuid.uuid4().hex
        self.output.mkdir()
        self.allowance=episodic.EpisodicAllowance(FakeBudget(),"long-fixture",200000,
                                                 time.monotonic()+1000)
        self.account=short.TrainingAccount(self.allowance)
        from within_stage import training_env
        p=patch.object(training_env,"make_opening_env",
                       side_effect=lambda pool,index,*,account,max_frames:FakeEarly(index,account))
        p.start();self.addCleanup(p.stop)
        self.parent=short.load_precision_checkpoint(self.initial,device="cpu")
        self.runtime=episodic.EpisodicRuntime(
            self.parent,Path("runs")/"20260930-within-stage"/"frozen-split.json",
            self.output,self.account,steps=8192,seed=0,noise_probability=.001,device="cpu")
        self.addCleanup(self.runtime.close)

    def test_initial_policy_adam_long_shape_and_clean_partial_collection(self):
        r=self.runtime
        self.assertEqual(short.state_sha256(self.parent.policy.state_dict()),r.initial_policy_sha256)
        self.assertEqual(short.state_sha256(self.parent.policy.optimizer.state_dict()),r.initial_adam_sha256)
        self.assertEqual(r.model.n_steps,2048)
        self.assertEqual(r.model.gae_lambda,1.)
        self.assertEqual(r.model.rollout_buffer.gae_lambda,1.)
        self.assertEqual(r.model.rollout_buffer.channels_per_frame,55)
        self.assertEqual(r.rollout_size,8192)
        with patch.object(r.callback,"_on_step",return_value=False):
            self.assertFalse(r.collect())
        self.assertEqual(self.account.agent_completed,4)
        self.assertEqual(r.optimized_steps,0)

    def test_strict_long_profile_checkpoint_reload_and_atomic_rollback(self):
        r=self.runtime
        r.model.policy.optimizer.zero_grad(set_to_none=True)
        r.model.policy.predict_values(torch.full((1,110,26,26),.1)).square().mean().backward()
        r.model.policy.optimizer.step()
        r.optimized_steps=8192  # Synthetic checkpoint-accounting fixture; no real collection.
        r._last_good=r._snapshot()
        provenance=dict(sources=episodic.sources(),parent=short._parent_record(self.initial,self.parent))
        record=episodic.save_checkpoint(r,self.output/"long.zip",provenance,"cpu_fixture")
        loaded=episodic.load_episodic_checkpoint(record["path"])
        self.assertEqual(loaded.gae_lambda,1.)
        self.assertEqual(loaded.rollout_buffer.channels_per_frame,55)
        self.assertEqual(loaded.num_timesteps,8192)
        for name,value in r.model.policy.state_dict().items():
            torch.testing.assert_close(value,loaded.policy.state_dict()[name],rtol=0,atol=0)
        sidecar=Path(record["path"]).with_suffix(".episodic.json")
        text=sidecar.read_text()
        manifest=json.loads(text)
        manifest["contract"]["ppo"]["gae_lambda"]=.95
        sidecar.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            episodic.load_episodic_checkpoint(record["path"])
        sidecar.write_text(text)
        previous=short.state_sha256(r.model.policy.state_dict())
        def interrupted():
            with torch.no_grad():next(r.model.policy.parameters()).add_(1)
            raise short.AllowanceStop("fixture deadline")
        with patch.object(r.model,"train",side_effect=interrupted):
            with self.assertRaises(short.AllowanceStop):
                r.optimize()
        self.assertEqual(previous,short.state_sha256(r.model.policy.state_dict()))
        self.assertEqual(r.optimized_steps,8192)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":
    unittest.main()
