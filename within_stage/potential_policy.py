"""Gamma1 potential shaping over the unchanged sparse TRAIN environment.

Only rewards change: r'=r_sparse+Phi(next)-Phi(previous), Phi=kills/20,
Phi(reset)=Phi(terminal)=0. Emitted rewards are float64 to retain telescoping
precision; PPO's existing float32 buffer remains unchanged. Terminal info keeps
the authoritative sparse ±1 return and separately reports the shaped sum.
No new reset, teacher, extra BC, noise change or timeout bootstrapping is added.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
from within_stage import on_policy as short
from within_stage.precision_policy import load_precision_checkpoint,save_precision_checkpoint,precision_schema,file_sha256
from rust_resources import ResourceMonitor

FORMAT="within-stage-win-preserving-progress-v1"


class ProgressPotentialVecEnv(VecEnvWrapper):
    objective_version=FORMAT
    def __init__(self,venv,*,gamma=1.):
        if isinstance(gamma,bool) or gamma!=1.:
            raise ValueError("Win-preserving finite potential requires gamma1")
        if venv.objective_version!="finite-victory-sparse-v1" \
                or not np.array_equal(venv.initial_remaining(),np.full(venv.num_envs,20)):
            raise ValueError("Wrap the existing sparse full20 TRAIN environment exactly once")
        super().__init__(venv)
        self.phi=np.zeros(self.num_envs,np.float64)
        self.returns=np.zeros(self.num_envs,np.float64)

    def reset(self):
        observation=self.venv.reset()
        self.phi.fill(0);self.returns.fill(0)
        return observation

    def step_wait(self):
        observation,sparse,dones,raw_infos=self.venv.step_wait()
        infos=deepcopy(raw_infos)
        kills=np.asarray(self.venv.current_kills())
        if kills.shape!=(self.num_envs,) or kills.dtype.kind not in "iu":
            raise ValueError("Expected canonical transition-safe integer kill counters")
        rewards=np.empty(self.num_envs,np.float64)
        for i,info in enumerate(infos):
            previous=float(self.phi[i])
            expected=1. if info.get("outcome")=="clear" else -1.
            if info.get("reward_profile")!="finite-victory-sparse-v1" \
                    or (dones[i] and (float(sparse[i])!=expected or info.get("TimeLimit.truncated") is not False)) \
                    or (not dones[i] and float(sparse[i])!=0.):
                raise ValueError("Underlying sparse reward/terminal profile changed")
            current=0. if dones[i] else int(kills[i])/20.
            if not dones[i] and (not 0<=current<=1 or current<previous):
                raise ValueError("Invalid/decreasing within-episode kill potential")
            rewards[i]=float(sparse[i])+current-previous
            self.returns[i]+=rewards[i]
            info.update(reward_profile=FORMAT,underlying_reward_profile="finite-victory-sparse-v1",
                        unshaped_reward=float(sparse[i]),potential_previous=previous,potential_next=current,
                        potential_adjustment=current-previous,shaped_reward=float(rewards[i]))
            if dones[i]:
                if not np.isclose(self.returns[i],expected,rtol=0,atol=1e-12):
                    raise ValueError("Telescoping episode return violated finite victory")
                info.update(sparse_episode_return=expected,shaped_episode_return=float(self.returns[i]))
                info["episode"]["r"]=expected
                self.phi[i]=self.returns[i]=0
            else:self.phi[i]=current
        return observation,rewards,dones,infos


class _Callback(BaseCallback):
    def __init__(self,guard,output):
        super().__init__();self.guard,self.output=guard,output
        self.episodes=[]
    def _on_step(self):
        for i,(reward,done,info) in enumerate(zip(self.locals["rewards"],self.locals["dones"],self.locals["infos"])):
            if info.get("reward_profile")!=FORMAT or not np.isfinite(reward) \
                    or not np.isclose(reward,info["unshaped_reward"]+info["potential_next"]-info["potential_previous"],rtol=0,atol=1e-12):
                raise ValueError("Wrong shaped transition contract")
            event=info["execution_noise"]
            if event["intended_action"]!=int(np.asarray(self.locals["actions"]).reshape(-1)[i]):
                raise ValueError("Intended PPO action was overwritten")
            if done:
                row=dict(outcome=info["outcome"],clear=info["outcome"]=="clear",
                         opening=info["training_opening"],sparse_return=info["sparse_episode_return"],
                         shaped_return=info["shaped_episode_return"],noise=info["execution_noise_episode"],
                         collected_step=int(self.model.num_timesteps))
                self.episodes.append(row)
                with (self.output/"potential-episodes.jsonl").open("a") as stream:
                    stream.write(short.json.dumps(row)+"\n")
        self.guard.check("potential_callback")
        return True


class PotentialRuntime(short.OnlineRuntime):
    def __init__(self,parent,split_path,output,account,**kwargs):
        super().__init__(parent,split_path,output,account,**kwargs)
        try:
            self.env=ProgressPotentialVecEnv(self.env,gamma=self.model.gamma)
            # Already reset once by the verified parent constructor: preserve
            # that observation/opening and start Phi at0, without a second boot.
            self.model.set_env(self.env,force_reset=False)
            self.callback.on_training_end()
            self.callback=_Callback(self.guard,output)
            self.callback.init_callback(self.model);self.callback.on_training_start({}, {})
            self.objective_version=self.env.objective_version
        except BaseException:
            self.close();raise


def contract():
    result=short.contract()
    result.update(format=FORMAT,reward_objective=FORMAT,max_controls=32768,shaping=True,
                  potential="current_kills/20; terminal0; reset0",step_reward_dtype="float64",
                  episode_return="Exactly sparse terminal outcome mathematically; shaped sum audited at1e-12",
                  only_experiment_change="progress potential over unchanged sparse gamma1 objective")
    return result


def sources():
    result=short.source_hashes();result["within_stage\\potential_policy.py"]=file_sha256(Path(__file__).resolve())
    return result


def save_checkpoint(runtime,path,provenance,role):
    if runtime.optimized_steps<=0 or runtime.optimized_steps%1024 or not runtime.state_verified:
        raise ValueError("Save only complete optimized potential rollouts")
    if sources()!=provenance["sources"]:raise ValueError("Frozen potential sources changed")
    runtime.env.verify_provenance()
    config=dict(schema=precision_schema(),datasets=runtime.parent_dataset_config.get("datasets",[]),
                split=runtime.parent_dataset_config.get("split"),purpose=FORMAT,
                parent_checkpoint=provenance["parent"],potential_contract=contract(),
                source_hashes=provenance["sources"],training_env=runtime.env.provenance,
                frozen_split_sha256=short.FROZEN_SPLIT_SHA256,offsets_enabled=True,
                optimized_controls=runtime.optimized_steps,optimizer_stats=deepcopy(runtime.optim_stats),role=role)
    before=runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps=runtime.optimized_steps
        record=save_precision_checkpoint(runtime.model,path,
            migration_record=runtime.model.precision_migration_record,dataset_config=config)
    finally:runtime.model.num_timesteps=before
    short._write(Path(path).with_suffix(".potential.json"),dict(format=FORMAT,contract=contract(),
        sources=provenance["sources"],parent=provenance["parent"],checkpoint_sha256=record["sha256"],
        optimized_controls=runtime.optimized_steps,exact_simulator_resume=False))
    return dict(record,optimized_controls=runtime.optimized_steps)


def load_potential_checkpoint(path,device="cpu"):
    path=Path(path).resolve();extra=short.json.loads(path.with_suffix(".potential.json").read_text())
    if extra.get("format")!=FORMAT or extra.get("contract")!=contract() \
            or extra.get("sources")!=sources() or extra.get("checkpoint_sha256")!=file_sha256(path) \
            or extra.get("exact_simulator_resume") is not False:
        raise ValueError("Invalid potential source/checkpoint/profile contract")
    model=load_precision_checkpoint(path,device=device);config=model.precision_dataset_config
    if config.get("purpose")!=FORMAT or config.get("potential_contract")!=contract() \
            or config.get("source_hashes")!=sources() or config.get("parent_checkpoint")!=extra["parent"] \
            or config.get("frozen_split_sha256")!=short.FROZEN_SPLIT_SHA256 \
            or model.num_timesteps!=extra["optimized_controls"] or not 0<model.num_timesteps<=32768 \
            or model.num_timesteps%1024 or model.rollout_buffer.channels_per_frame!=55:
        raise ValueError("Serialized potential precision provenance differs")
    for key,value in short.PPO.items():
        if key in ("n_steps","batch_size"):continue
        actual=getattr(model,key);actual=actual(1.) if key=="clip_range" else actual
        if isinstance(actual,bool) or actual!=value:raise ValueError(f"PPO profile changed:{key}")
    return model


def run_potential_policy(initial_checkpoint,output,split,budget,*,steps=32768,seed=0,device="cpu"):
    if type(steps) is not int or not 1024<=steps<=32768 or steps%1024:
        raise ValueError("steps must be a multiple of1024 in1024..32768")
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError("seed must be uint32")
    if str(device)=="auto" or torch.device(device).type not in ("cpu","cuda"):
        raise ValueError("Explicit CPU/CUDA device required")
    short.load_train_context(split);output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    parent=load_precision_checkpoint(initial_checkpoint,device=device)
    parent_record=short._parent_record(initial_checkpoint,parent)
    output.mkdir(parents=True);started=time.monotonic()
    report=dict(format=FORMAT,status="running",error=None,stop_reason=None,contract=contract(),
                requested_controls=steps,optimized_controls=0,seed=seed,noise_probability=.001,
                parent=parent_record,updates=[],checkpoints=[],baseline=None,evaluations=[],
                winner_selected=False,final_evaluated=False)
    allowance=account=runtime=provenance=None
    try:
        snap=budget.snapshot()
        if not 0<snap["max_interactions"]<=4000000:raise ValueError("Global cap exceeds4M")
        deadline=started+min(1800.,snap["remaining_seconds"],max(0.,short.GLOBAL_EPOCH-time.time()))
        if deadline<=time.monotonic():raise short.AllowanceStop("No remaining wall time")
        monitor=ResourceMonitor(output/"resources.jsonl",device=str(device).split(":")[0])
        monitor.check("admission",admission_slots=1,force=True)
        amount=min(150000,int(snap["remaining_interactions"]))
        if amount<1:raise short.AllowanceStop("No remaining allowance")
        allowance=short.TrialAllowance(budget,"potential-"+short.hashlib.sha256(str(output).encode()).hexdigest()[:24],
                                      amount,deadline,monitor)
        account=short.TrainingAccount(allowance);provenance=dict(sources=sources(),parent=parent_record)
        short._write(output/"config.json",dict(contract=contract(),provenance=provenance,
                    allowance=allowance.snapshot(),steps=steps,seed=seed,noise_probability=.001))
        report["baseline"]=short._evaluate(parent,initial_checkpoint,split,output/"baseline-dev",allowance)
        runtime=PotentialRuntime(parent,split,output,account,steps=steps,seed=seed,
                                 noise_probability=.001,device=device)
        while runtime.optimized_steps<steps:
            allowance.check("potential_rollout");before=account.agent_completed
            if not runtime.collect() or account.agent_completed-before!=1024:
                raise short.AllowanceStop("Partial potential rollout is not optimized")
            update=runtime.optimize();report["optimized_controls"]=runtime.optimized_steps
            tail=runtime.callback.episodes[-100:]
            update["last100"]=dict(episodes=len(tail),wins=sum(r["clear"] for r in tail))
            report["updates"].append(update)
            if runtime.optimized_steps%8192==0:
                report["checkpoints"].append(save_checkpoint(runtime,
                    output/f"optimized-{runtime.optimized_steps:06d}.zip",provenance,"periodic"))
            short._write(output/"report.json",report)
            if runtime.optimized_steps%16384==0:
                evaluation=short._evaluate(runtime.model,report["checkpoints"][-1]["path"],split,
                    output/f"dev-{runtime.optimized_steps:06d}",allowance)
                evaluation["optimized_controls"]=runtime.optimized_steps
                report["evaluations"].append(evaluation)
        report["checkpoints"].append(save_checkpoint(runtime,output/"final.zip",provenance,"final"))
        report.update(status="completed",stop_reason="control_cap")
    except (short.AllowanceStop,short.ProbeStop,TimeoutError) as error:
        report.update(status="partial",stop_reason="allowance_or_deadline",error=str(error))
    except BaseException as error:
        report.update(status="failed",stop_reason="error",error=f"{type(error).__name__}: {error}")
    finally:
        if runtime is not None:
            report.update(optimized_controls=runtime.optimized_steps,training_episodes=len(runtime.callback.episodes))
            if report["status"]!="completed" and runtime.optimized_steps and runtime.state_verified:
                try:report["checkpoints"].append(save_checkpoint(runtime,output/"partial-optimized.zip",provenance,"partial"))
                except Exception as error:report["checkpoint_error"]=str(error)
            try:runtime.close()
            except Exception as error:report.update(status="failed",cleanup_error=str(error))
        if allowance is not None:
            allowance.finish(certain=account.attempted==account.completed and not allowance.pending)
            report.update(trial_accounting=allowance.snapshot(),training_accounting=account.snapshot(),
                          total_attempted_calls=allowance.used,outer_reservation_settled=allowance.finished)
        report["elapsed_seconds"]=time.monotonic()-started
        short._write(output/"report.json",report)
    return report
