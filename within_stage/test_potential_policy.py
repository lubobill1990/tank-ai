"""CPU-only telescoping-credit tests; no real environment/GPU interactions."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import uuid

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from within_stage import potential_policy as potential
from within_stage import on_policy as short
from within_stage import test_on_policy as short_tests
from within_stage.test_training_env import FakeEarly


class CounterVec(VecEnv):
    objective_version="finite-victory-sparse-v1"
    def __init__(self, outcomes):
        self.script=outcomes;self.index=0;self.kills=0
        super().__init__(1,gym.spaces.Box(-1,1,(110,26,26),np.float32),gym.spaces.Discrete(10))
    def reset(self):
        self.index=0;self.kills=0
        return np.zeros((1,110,26,26),np.float32)
    def initial_remaining(self):return np.array([20],np.int64)
    def current_kills(self):return np.array([self.kills],np.int64)
    def step_async(self,actions):pass
    def step_wait(self):
        self.kills,outcome=self.script[self.index];self.index+=1
        done=bool(outcome);r=1. if outcome=="clear" else -1. if done else 0.
        info={"reward_profile":self.objective_version,"outcome":outcome}
        if done:info.update(episode={"r":r,"l":self.index},**{"TimeLimit.truncated":False})
        return np.zeros((1,110,26,26),np.float32),np.array([r],np.float32),np.array([done]),[info]
    def close(self):pass
    def get_attr(self,name,indices=None):return [None]
    def set_attr(self,*args,**kwargs):pass
    def env_method(self,*args,**kwargs):return [np.ones(10,bool)]
    def env_is_wrapped(self,*args,**kwargs):return [False]


class PotentialTests(unittest.TestCase):
    def test_twenty_kill_clear_and_early_losses_sum_to_terminal_objective(self):
        for outcome in ("clear","base_lost","lives_exhausted","timeout"):
            for kills in (0,1,7,19):
                script=[(k,"") for k in range(1,kills+1)]+[(20 if outcome=="clear" else kills,outcome)]
                env=potential.ProgressPotentialVecEnv(CounterVec(script),gamma=1.)
                try:
                    env.reset();total=0.
                    for _ in script:
                        _,rewards,done,infos=env.step(np.array([0]));total+=float(rewards[0])
                    self.assertEqual(total,1. if outcome=="clear" else -1.)
                    self.assertEqual(infos[0]["episode"]["r"],total)
                    self.assertEqual(infos[0]["potential_next"],0.)
                    self.assertFalse(infos[0]["TimeLimit.truncated"])
                finally:env.close()

    def test_no_progress_loop_pays_nothing_and_autoreset_zeroes_potential(self):
        raw=CounterVec([(0,""),(0,""),(1,""),(0,"timeout"),(1,""),(1,"base_lost")])
        env=potential.ProgressPotentialVecEnv(raw,gamma=1.)
        self.addCleanup(env.close);env.reset()
        rewards=[float(env.step(np.array([0]))[1][0]) for _ in range(6)]
        np.testing.assert_allclose(rewards,[0,0,.05,-1.05,.05,-1.05],atol=1e-15)
        self.assertEqual(sum(rewards[:4]),-1.)
        self.assertEqual(sum(rewards[4:]),-1.)
        env.reset()
        self.assertEqual(float(env.step(np.array([0]))[1][0]),0.)

    def test_rejects_wrong_gamma_or_decreasing_nonterminal_counter(self):
        with self.assertRaises(ValueError):
            potential.ProgressPotentialVecEnv(CounterVec([]),gamma=.99)
        env=potential.ProgressPotentialVecEnv(CounterVec([(2,""),(1,"")]),gamma=1.)
        self.addCleanup(env.close);env.reset();env.step(np.array([0]))
        with self.assertRaises(ValueError):env.step(np.array([0]))

    def test_ppo_profile_unchanged_only_reward_differs(self):
        self.assertEqual(potential.contract()["ppo"],short.contract()["ppo"])
        self.assertEqual(potential.contract()["max_controls"],32768)
        self.assertEqual(potential.contract()["potential"],"current_kills/20; terminal0; reset0")

    def test_all_monotone_kill_jumps_preserve_terminal_total_without_new_loop_bonus(self):
        rng=np.random.default_rng(57)
        for _ in range(100):
            path=sorted(rng.integers(0,21,size=20).tolist())
            outcome=rng.choice(["clear","base_lost","lives_exhausted","timeout"])
            env=potential.ProgressPotentialVecEnv(CounterVec([(k,"") for k in path]+[(0,outcome)]),gamma=1.)
            try:
                env.reset()
                rewards=[float(env.step(np.array([0]))[1][0]) for _ in range(21)]
                self.assertAlmostEqual(sum(rewards),1. if outcome=="clear" else -1.,places=14)
                for i in range(1,len(path)):
                    if path[i]==path[i-1]:self.assertEqual(rewards[i],0.)
            finally:env.close()


class PotentialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        short_tests.PrecisionRuntimeTests.setUpClass.__func__(cls)

    def test_runtime_wrap_does_not_reset_again_and_strict_save_load(self):
        output=self.root/uuid.uuid4().hex;output.mkdir()
        allowance=short.TrialAllowance(short_tests.FakeBudget(),"potential",150000,time.monotonic()+1000)
        account=short.TrainingAccount(allowance)
        from within_stage import training_env
        with patch.object(training_env,"make_opening_env",
                          side_effect=lambda pool,index,*,account,max_frames:FakeEarly(index,account)):
            parent=short.load_precision_checkpoint(self.initial,device="cpu")
            runtime=potential.PotentialRuntime(parent,Path("runs")/"20260930-within-stage"/"frozen-split.json",
                output,account,steps=8,seed=0,noise_probability=.001,device="cpu",n_steps=2,batch_size=4)
            try:
                self.assertEqual(account.raw_initialization_completed,8) # exactly one reset per4envs
                self.assertEqual(runtime.model.gae_lambda,.95)
                self.assertEqual(runtime.model.rollout_buffer.channels_per_frame,55)
                self.assertTrue(runtime.collect())
                runtime.optimize()
                self.assertEqual(runtime.optimized_steps,8)
                runtime.optimized_steps=1024 # synthetic archive tag only
                provenance={"sources":potential.sources(),"parent":short._parent_record(self.initial,parent)}
                record=potential.save_checkpoint(runtime,output/"fixture.zip",provenance,"cpu_fixture")
                loaded=potential.load_potential_checkpoint(record["path"])
                self.assertEqual(loaded.gamma,1.)
                self.assertEqual(loaded.gae_lambda,.95)
                self.assertEqual(loaded.rollout_buffer.channels_per_frame,55)
                for name,value in runtime.model.policy.state_dict().items():
                    torch.testing.assert_close(value,loaded.policy.state_dict()[name],rtol=0,atol=0)
                sidecar=Path(record["path"]).with_suffix(".potential.json")
                saved=sidecar.read_text()
                manifest=json.loads(saved)
                manifest["contract"]["potential"]="loop-bonus"
                sidecar.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    potential.load_potential_checkpoint(record["path"])
                sidecar.write_text(saved)
            finally:runtime.close()
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":
    unittest.main()
