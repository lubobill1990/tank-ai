"""One isolated long-credit precision PPO variant; no implicit GPU/ROM run.

Relative to within_stage.on_policy, PPO changes ONLY n_steps2048 and GAE lambda1.
The gamma1 sparse objective, policy/precision-Adam warm start and noise stay fixed.
This propagates terminal credit across the available trace, but is not guaranteed
Monte Carlo for episodes crossing a2048-step rollout boundary.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch

from within_stage import on_policy as short
from within_stage.precision_policy import (
    load_precision_checkpoint, save_precision_checkpoint, precision_schema,
    state_sha256, file_sha256,
)
from rust_resources import ResourceMonitor

FORMAT="within-stage-precision-episodic-policy-v1"
ROLLOUT=8192


def contract():
    result=short.contract()
    result.update(format=FORMAT,rollout_controls=ROLLOUT,dev_every=32768,whole_trial_cap=200000)
    result["ppo"]=dict(short.PPO,n_steps=2048,gae_lambda=1.)
    result["credit_note"]="Lambda1 gamma1; terminals reach the whole available trace; boundaries still bootstrap."
    return result


def sources():
    result=short.source_hashes()
    result["within_stage\\episodic_policy.py"]=file_sha256(Path(__file__).resolve())
    return result


class EpisodicAllowance(short.TrialAllowance):
    def __init__(self,parent,label,amount,deadline,monitor=None):
        if type(amount) is not int or not 1<=amount<=200000:
            raise ValueError("Long-credit trial allowance must be in1..200000")
        if isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not math.isfinite(deadline):
            raise ValueError("Finite absolute monotonic deadline required")
        self.parent,self.amount,self.deadline,self.monitor=parent,amount,deadline,monitor
        self.used,self.pending,self.evaluations=0,{},[]
        self.finished=False
        self.token=parent.reserve(label,amount)


class EpisodicRuntime(short.OnlineRuntime):
    def __init__(self,parent,split_path,output,account,*,steps,seed,noise_probability,device):
        initial_policy=state_sha256(parent.policy.state_dict())
        initial_adam=state_sha256(parent.policy.optimizer.state_dict())
        super().__init__(parent,split_path,output,account,steps=steps,seed=seed,
                         noise_probability=noise_probability,device=device,n_steps=2048,batch_size=256)
        try:
            # Instance-local profile change before any rollout or optimizer step.
            self.model.gae_lambda=1.
            self.model.rollout_buffer.gae_lambda=1.
            if self.model.rollout_buffer.channels_per_frame!=55 or self.rollout_size!=8192:
                raise ValueError("Long-credit precision rollout allocation is invalid")
            if state_sha256(self.model.policy.state_dict())!=initial_policy \
                    or self.initial_precision_optimizer_sha256!=initial_adam:
                raise ValueError("Parent precision policy/Adam identity changed")
            self.initial_policy_sha256,self.initial_adam_sha256=initial_policy,initial_adam
        except BaseException:
            self.close()
            raise


def save_checkpoint(runtime,path,provenance,role):
    if runtime.optimized_steps<=0 or runtime.optimized_steps%ROLLOUT or not runtime.state_verified:
        raise ValueError("Only completed long-credit optimized rollouts may be saved")
    if sources()!=provenance["sources"]:
        raise ValueError("Frozen long-credit sources changed")
    runtime.env.verify_provenance()
    config=dict(schema=precision_schema(),datasets=runtime.parent_dataset_config.get("datasets",[]),
                split=runtime.parent_dataset_config.get("split"),purpose=FORMAT,
                parent_checkpoint=provenance["parent"],episodic_contract=contract(),
                source_hashes=provenance["sources"],training_env=runtime.env.provenance,
                frozen_split_sha256=short.FROZEN_SPLIT_SHA256,offsets_enabled=True,
                optimized_controls=runtime.optimized_steps,optimizer_stats=deepcopy(runtime.optim_stats),
                initial_policy_sha256=runtime.initial_policy_sha256,
                initial_adam_sha256=runtime.initial_adam_sha256,role=role)
    before=runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps=runtime.optimized_steps
        result=save_precision_checkpoint(runtime.model,path,
            migration_record=runtime.model.precision_migration_record,dataset_config=config)
    finally:
        runtime.model.num_timesteps=before
    short._write(Path(path).with_suffix(".episodic.json"),dict(
        format=FORMAT,contract=contract(),sources=provenance["sources"],parent=provenance["parent"],
        checkpoint_sha256=result["sha256"],optimized_controls=runtime.optimized_steps,
        split_sha256=short.FROZEN_SPLIT_SHA256,exact_simulator_resume=False))
    return dict(result,optimized_controls=runtime.optimized_steps)


def load_episodic_checkpoint(path,device="cpu"):
    path=Path(path).resolve()
    extra=json.loads(path.with_suffix(".episodic.json").read_text())
    if extra.get("format")!=FORMAT or extra.get("contract")!=contract() \
            or extra.get("sources")!=sources() or extra.get("checkpoint_sha256")!=file_sha256(path) \
            or extra.get("split_sha256")!=short.FROZEN_SPLIT_SHA256 \
            or extra.get("exact_simulator_resume") is not False:
        raise ValueError("Invalid episodic profile/source/checkpoint contract")
    model=load_precision_checkpoint(path,device=device)
    config=model.precision_dataset_config
    if config.get("purpose")!=FORMAT or config.get("episodic_contract")!=contract() \
            or config.get("source_hashes")!=sources() or config.get("parent_checkpoint")!=extra.get("parent") \
            or model.num_timesteps!=extra.get("optimized_controls") \
            or not 0<model.num_timesteps<=65536 or model.num_timesteps%ROLLOUT \
            or model.rollout_buffer.channels_per_frame!=55 or model.rollout_buffer.gae_lambda!=1:
        raise ValueError("Serialized precision episodic training identity differs")
    for key,value in contract()["ppo"].items():
        if key in ("n_steps","batch_size"):
            continue
        actual=getattr(model,key)
        actual=actual(1.) if key=="clip_range" else actual
        if isinstance(actual,bool) or actual!=value:
            raise ValueError(f"Serialized long-credit PPO mismatch:{key}")
    return model


def run_episodic_policy(initial_checkpoint,output,split_path,budget,*,steps=65536,seed=0,
                        noise_probability=.001,device="cpu"):
    if type(steps) is not int or not ROLLOUT<=steps<=65536 or steps%ROLLOUT:
        raise ValueError("steps must be a multiple of8192 in8192..65536")
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("seed must be uint32")
    if isinstance(noise_probability,bool) or not isinstance(noise_probability,(int,float)) \
            or not math.isfinite(noise_probability) or not 0<=noise_probability<=1:
        raise ValueError("noise_probability must be finite in[0,1]")
    if str(device)=="auto" or torch.device(device).type not in ("cpu","cuda"):
        raise ValueError("Explicit CPU/CUDA device required")
    short.load_train_context(split_path)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent=load_precision_checkpoint(initial_checkpoint,device=device)
    parent_record=short._parent_record(initial_checkpoint,parent)
    output.mkdir(parents=True)
    started=time.monotonic()
    report=dict(format=FORMAT,status="running",error=None,stop_reason=None,contract=contract(),
                requested_controls=steps,optimized_controls=0,seed=seed,noise_probability=noise_probability,
                device=str(device),parent=parent_record,updates=[],checkpoints=[],baseline=None,
                evaluations=[],winner_selected=False,final_evaluated=False)
    allowance=account=runtime=provenance=None
    try:
        snap=budget.snapshot()
        if not 0<snap["max_interactions"]<=4000000:
            raise ValueError("NEW tranche may not exceed4M interactions")
        deadline=started+min(1800.,snap["remaining_seconds"],max(0.,short.GLOBAL_EPOCH-time.time()))
        if deadline<=time.monotonic():
            raise short.AllowanceStop("No remaining wall time")
        monitor=ResourceMonitor(output/"resources.jsonl",device=str(device).split(":")[0])
        monitor.check("admission",admission_slots=1,force=True)
        amount=min(200000,int(snap["remaining_interactions"]))
        if amount<1:
            raise short.AllowanceStop("No remaining interaction allowance")
        label="episodic-"+short.hashlib.sha256(str(output).encode()).hexdigest()[:24]
        allowance=EpisodicAllowance(budget,label,amount,deadline,monitor)
        account=short.TrainingAccount(allowance)
        provenance=dict(sources=sources(),parent=parent_record)
        short._write(output/"config.json",dict(contract=contract(),provenance=provenance,
                     deadline_monotonic=deadline,allowance=allowance.snapshot(),
                     steps=steps,seed=seed,noise_probability=noise_probability))
        report["baseline"]=short._evaluate(parent,initial_checkpoint,split_path,output/"baseline-dev",allowance)
        runtime=EpisodicRuntime(parent,split_path,output,account,steps=steps,seed=seed,
                               noise_probability=noise_probability,device=device)
        while runtime.optimized_steps<steps:
            allowance.check("long_rollout_start")
            before=account.agent_completed
            complete=runtime.collect()
            if not complete or account.agent_completed-before!=ROLLOUT:
                raise short.AllowanceStop("Partial8192-control rollout not optimized")
            update=runtime.optimize()
            report["optimized_controls"]=runtime.optimized_steps
            tail=runtime.callback.episodes[-100:]
            update["last100"]=dict(episodes=len(tail),wins=sum(r["clear"] for r in tail))
            report["updates"].append(update)
            record=save_checkpoint(runtime,output/f"optimized-{runtime.optimized_steps:06d}.zip",
                                   provenance,"periodic")
            report["checkpoints"].append(record)
            short._write(output/"report.json",report)
            if runtime.optimized_steps%32768==0:
                evaluation=short._evaluate(runtime.model,record["path"],split_path,
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
            report.update(optimized_controls=runtime.optimized_steps,
                          initial_policy_sha256=runtime.initial_policy_sha256,
                          initial_adam_sha256=runtime.initial_adam_sha256,
                          training_episodes=len(runtime.callback.episodes),
                          noise_interventions=runtime.callback.interventions,
                          intended_action_counts=dict(runtime.callback.intended_counts),
                          executed_action_counts=dict(runtime.callback.executed_counts),
                          completed_train_family_counts=dict(runtime.callback.family_counts))
            if report["status"]!="completed" and runtime.optimized_steps and runtime.state_verified:
                try:
                    report["checkpoints"].append(save_checkpoint(runtime,output/"partial-optimized.zip",
                                                                 provenance,"partial_optimized"))
                except Exception as error:
                    report["checkpoint_error"]=str(error)
            try:
                runtime.close()
            except Exception as error:
                report.update(status="failed",stop_reason="cleanup_error",cleanup_error=str(error))
        if allowance is not None:
            certain=account is not None and account.attempted==account.completed and not allowance.pending
            allowance.finish(certain=certain)
            report.update(trial_accounting=allowance.snapshot(),outer_reservation_settled=allowance.finished,
                          total_attempted_calls=allowance.used,training_accounting=account.snapshot())
        report["elapsed_seconds"]=time.monotonic()-started
        short._write(output/"report.json",report)
    return report
