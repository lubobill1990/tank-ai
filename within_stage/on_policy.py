"""Precision110 PPO with bootstrap-inclusive reservations and DEV-only diagnostics.

run_on_policy(initial_checkpoint, output, split_path, budget, steps=65536, ...)
reserves one capped150k-call allowance BEFORE any ROM reset. That is a cap, not
a guarantee the planned controls/evaluations fit: stop rather than extend it.
The unchanged development evaluator uses local sub-reservations of this cap.
Only the outer reservation touches the supplied shared ledger.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from learning_lab import aiming_train as runtime_base
from native_rollout import NativeGPURunRolloutBuffer
from rust_resources import ResourceMonitor
from within_stage.budget import WithinStageBudget
from within_stage.openings import ProbeAccounting, ProbeStop
from within_stage.training_env import make_training_env, load_train_context, FROZEN_SPLIT_SHA256
from within_stage.evaluation import evaluate_development_case
from within_stage.precision_policy import (
    load_precision_checkpoint, save_precision_checkpoint, precision_schema,
    _validate_policy, state_sha256, file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "within-stage-precision-on-policy-v1"
PPO = dict(learning_rate=5e-5, gamma=1., gae_lambda=.95, clip_range=.1,
           ent_coef=.001, target_kl=.01, n_epochs=4, n_steps=256, batch_size=256)
GLOBAL_EPOCH = datetime.fromisoformat("2026-10-01T08:30:25.131+08:00").timestamp()
TRIAL_CAP = 150000


def _write(path, value):
    runtime_base._write(Path(path), value)


def source_hashes():
    return {name: file_sha256(ROOT / name) for name in (
        "within_stage\\on_policy.py", "within_stage\\training_env.py",
        "within_stage\\precision_policy.py", "within_stage\\observation.py",
        "within_stage\\evaluation.py", "within_stage\\execution_noise.py",
        "within_stage\\openings.py", "within_stage\\early_openings.py",
        "within_stage\\opening_pool.py", "learning_lab\\victory_objective.py",
        "learning_lab\\aiming_train.py", "learning_lab\\backplay_train.py",
        "native_rollout.py", "native_codec.py", "compact_rollout.py", "policy.py",
        "battle_city_env.py", "game_state.py")}


class AllowanceStop(RuntimeError):
    pass


class TrialAllowance:
    """Duck-typed budget adapter for unchanged DEV evaluation; local reservations only."""
    def __init__(self, parent, label, amount, deadline, monitor=None):
        if type(amount) is not int or not 1 <= amount <= TRIAL_CAP:
            raise ValueError("Whole trial allowance must be in1..150000")
        if isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not math.isfinite(deadline):
            raise ValueError("Trial deadline must be a finite absolute monotonic time")
        self.parent, self.amount, self.deadline, self.monitor = parent, amount, deadline, monitor
        self.used, self.pending, self.evaluations = 0, {}, []
        self.finished = False
        self.token = parent.reserve(label, amount)

    @property
    def available(self):
        return self.amount - self.used - sum(self.pending.values())

    def check(self, phase="trial"):
        if time.monotonic() >= self.deadline or time.time() >= GLOBAL_EPOCH:
            raise AllowanceStop("Trial/new-tranche absolute deadline")
        if self.monitor is not None:
            self.monitor.check(phase)

    def consume(self):
        self.check("training_frame_or_control")
        if self.available < 1:
            raise AllowanceStop("Bootstrap-inclusive trial allowance exhausted; no extension")
        self.used += 1

    def reserve(self, label, interactions):
        self.check("development_reservation")
        if type(interactions) is not int or not 0 < interactions <= self.available:
            raise AllowanceStop("Insufficient remaining local allowance for DEV upper bound")
        token = f"local-{len(self.evaluations)}"
        self.pending[token] = interactions
        self.evaluations.append(dict(token=token, label=label, reserved=interactions, actual=None))
        return token

    def settle(self, token, interactions):
        if token not in self.pending:
            row = next((r for r in self.evaluations if r["token"] == token), None)
            if row is not None and row["actual"] == interactions:
                return
            raise ValueError("Unknown/conflicting local settlement")
        if type(interactions) is not int or not 0 <= interactions <= self.pending[token]:
            raise ValueError("DEV settlement exceeds its local reservation")
        self.used += interactions
        del self.pending[token]
        next(r for r in self.evaluations if r["token"] == token)["actual"] = interactions

    def snapshot(self):
        return dict(remaining_seconds=max(0., self.deadline-time.monotonic()),
                    remaining_interactions=self.available, used=self.used,
                    outer_reservation=self.token, reserved=self.amount,
                    pending=dict(self.pending), development_reservations=deepcopy(self.evaluations))

    def finish(self, *, certain):
        if self.finished:
            return
        if certain and not self.pending:
            self.parent.settle(self.token, self.used)
            self.finished = True


class TrainingAccount(ProbeAccounting):
    def __init__(self, allowance):
        super().__init__(allowance.amount, deadline=allowance.deadline)
        self.allowance, self.control_limit = allowance, None

    def check(self):
        self.allowance.check("training_account")

    def before(self, kind):
        if kind not in ("raw","agent"):
            raise ValueError("Unexpected accounting domain")
        if kind == "agent" and self.control_limit is not None and self.agent_attempted >= self.control_limit:
            raise AllowanceStop("Training control cap reached")
        self.check()
        self.allowance.consume()
        if kind == "raw":
            self.raw_initialization_attempted += 1
        elif kind == "agent":
            self.agent_attempted += 1


def contract():
    return dict(
        format=FORMAT, schema=precision_schema(), observation_shape=[110,26,26],
        channels_per_frame=55, actions=10, n_envs=4, n_stack=2,
        share_features_extractor=False, separate_critic=True,
        ppo=deepcopy(PPO), reward_objective="finite-victory-sparse-v1",
        gamma=1., shaping=False, bootstrap_timeouts=False,
        buffer="NativeGPURunRolloutBuffer", max_controls=65536, rollout_controls=1024,
        checkpoint_every=8192, dev_every=16384, whole_trial_cap=TRIAL_CAP,
        parent_state="precision policy+precision Adam preserved, new env/counters/RNG",
        offsets_enabled=True, intended_action_and_logprob_preserved=True,
        teacher_takeover=False, anchor_regularization=False, final_evaluated=False,
        exact_simulator_resume=False)


def development_plan(split):
    return [dict(config_index=int(config), intervention_decision=when,
                 noise_seed=801+i, mode="normal" if when is None else "movement_recovery")
            for i, config in enumerate(split["development"]["representative_config_indices"][:6])
            for when in (None,150)]


class _Callback(runtime_base._Callback):
    def __init__(self, guard, output):
        super().__init__(guard, runtime_base.OBJECTIVE)
        self.output, self.episodes = output, []
        self.intended_counts, self.executed_counts, self.family_counts = Counter(), Counter(), Counter()
        self.interventions = 0

    def _on_step(self):
        keep = super()._on_step()
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        for index, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            event = info["execution_noise"]
            if event["intended_action"] != int(actions[index]):
                raise ValueError("Actuator noise must not replace stored PPO intended actions")
            self.intended_counts[event["intended_action"]] += 1
            self.executed_counts[event["executed_action"]] += 1
            self.interventions += int(event["applied"])
            if done:
                opening = info["training_opening"]
                self.family_counts[opening["family_id"]] += 1
                row = dict(outcome=info["outcome"], clear=info["outcome"]=="clear",
                           frames=info["frames"], kills=info["training_state_kills"],
                           reward=float(info["episode"]["r"]), opening=opening,
                           noise_episode=info["execution_noise_episode"],
                           optimized_controls_before_collection=self.model.num_timesteps)
                self.episodes.append(row)
                with (self.output / "training_episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row)+"\n")
        initial_seen=getattr(self,"_seen_openings",set())
        for opening in ([info["training_opening"] for info in self.locals["infos"]]
                        + self.training_env.current_openings()):
            if opening is not None:
                key=(opening["stream"],opening["episode_index"])
                if key not in initial_seen:
                    initial_seen.add(key)
                    with (self.output/"opened_train_families.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(opening)+"\n")
        self._seen_openings=initial_seen
        return keep


class OnlineRuntime(runtime_base.AimingRuntime):
    def __init__(self, parent, split_path, output, account, *, steps, seed, noise_probability,
                 device, n_steps=256, batch_size=256):
        _validate_policy(parent, precision=True)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        self.guard, self.account = account.allowance, account
        self.optimized_steps, self.state_verified = 0, True
        self.target_steps, self.n_steps = steps, n_steps
        self.model = self.env = self.raw = None
        self.optim_stats = dict(adam_steps=0, last_gradient_norm_post_clip=None)
        try:
            self.env = make_training_env(
                split_path, n_envs=4, noise_probability=noise_probability, noise_seed=seed,
                opening_seed=seed, account=account, deadline=account.deadline)
            self.env.verify_provenance()
            self.objective_version = self.env.objective_version
            self.rollout_size = self.env.num_envs*n_steps
            parent_policy = runtime_base._cpu_copy(parent.policy.state_dict())
            parent_adam = runtime_base._cpu_copy(parent.policy.optimizer.state_dict())
            self.initial_precision_optimizer_sha256 = state_sha256(parent_adam)
            self.model = MaskablePPO(
                "CnnPolicy", self.env, seed=seed, device=device, verbose=0,
                **dict(PPO, n_steps=n_steps,batch_size=batch_size),
                policy_kwargs=deepcopy(parent.policy_kwargs),
                rollout_buffer_class=runtime_base._GuardedBuffer,
                rollout_buffer_kwargs={"channels_per_frame":55,"max_storage_bytes":2*1024**3})
            if str(device).startswith("cuda") and self.model.device.type!="cuda":
                raise RuntimeError("Requested CUDA cannot fall back to CPU")
            self.model.policy.load_state_dict(parent_policy, strict=True)
            self.model.policy.optimizer.load_state_dict(parent_adam)
            if state_sha256(self.model.policy.optimizer.state_dict()) != self.initial_precision_optimizer_sha256:
                raise ValueError("Precision Adam did not migrate exactly")
            for group in self.model.policy.optimizer.param_groups:
                group["lr"]=PPO["learning_rate"]
            self.model.precision_migration_record=deepcopy(parent.precision_migration_record)
            self.parent_dataset_config=deepcopy(parent.precision_dataset_config)
            self.model.precision_dataset_config=deepcopy(parent.precision_dataset_config)
            # All actor and separate critic parameters learn under PPO, unlike actor-only BC.
            for parameter in self.model.policy.parameters():
                parameter.requires_grad_(True)
            _validate_policy(self.model,precision=True)
            self.model.rollout_buffer.guard=self.guard
            self._last_good=self._snapshot()
            original_step=self.model.policy.optimizer.step
            guard,stats=self.guard,self.optim_stats
            params=list(self.model.policy.parameters())
            def guarded_step(*args,**kwargs):
                guard.check("adam_step")
                gradients=[p.grad.detach().norm(2) for p in params if p.grad is not None]
                norm=float(torch.stack(gradients).norm(2).item()) if gradients else 0.
                if not np.isfinite(norm):
                    raise FloatingPointError("Nonfinite PPO gradient norm")
                result=original_step(*args,**kwargs)
                stats["adam_steps"]+=1
                stats["last_gradient_norm_post_clip"]=norm
                return result
            self.model.policy.optimizer.step=guarded_step
            self.callback=_Callback(self.guard,output)
            self.total,self.callback=self.model._setup_learn(steps,callback=self.callback)
            self.callback.on_training_start({}, {})
        except BaseException:
            self.close()
            raise

    def collect(self):
        self.env.verify_provenance()
        if self.account.agent_attempted+self.rollout_size>self.target_steps:
            raise AllowanceStop("No control budget for another complete rollout")
        self.account.control_limit=self.account.agent_attempted+self.rollout_size
        return self.model.collect_rollouts(
            self.env, callback=self.callback, rollout_buffer=self.model.rollout_buffer,
            n_rollout_steps=self.n_steps)


def _parent_record(path, model):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=file_sha256(path),
                manifest_sha256=file_sha256(path.with_suffix(".manifest.json")),
                precision_optimizer_sha256=state_sha256(model.policy.optimizer.state_dict()),
                policy_sha256=state_sha256(model.policy.state_dict()),
                old94_optimizer_resumed=False,
                original_precision_migration=deepcopy(model.precision_migration_record),
                precision_dataset_config=deepcopy(model.precision_dataset_config))


def save_online(runtime,path,provenance,role):
    if runtime.optimized_steps<=0 or not runtime.state_verified:
        raise ValueError("Only complete optimized PPO weights can be saved")
    if source_hashes()!=provenance["sources"]:
        raise ValueError("On-policy sources changed")
    runtime.env.verify_provenance()
    config=dict(schema=precision_schema(),datasets=runtime.parent_dataset_config.get("datasets",[]),
                split=runtime.parent_dataset_config.get("split"),
                purpose=FORMAT,offsets_enabled=True,
                parent_checkpoint=provenance["parent"],on_policy_contract=contract(),
                frozen_split_sha256=FROZEN_SPLIT_SHA256,training_env=runtime.env.provenance,
                optimized_controls=runtime.optimized_steps,optimizer_stats=deepcopy(runtime.optim_stats),
                role=role,source_hashes=provenance["sources"])
    before=runtime.model.num_timesteps
    try:
        runtime.model.num_timesteps=runtime.optimized_steps
        record=save_precision_checkpoint(runtime.model,path,
            migration_record=runtime.model.precision_migration_record,dataset_config=config)
    finally:
        runtime.model.num_timesteps=before
    extra=dict(format=FORMAT,contract=contract(),sources=provenance["sources"],
               parent=provenance["parent"],split_sha256=FROZEN_SPLIT_SHA256,
               checkpoint_sha256=record["sha256"],optimized_controls=runtime.optimized_steps,
               objective=runtime.objective_version,accounting=runtime.account.snapshot(),
               exact_simulator_resume=False,role=role)
    _write(Path(path).with_suffix(".on-policy.json"),extra)
    return dict(record,optimized_controls=runtime.optimized_steps)


def load_on_policy_checkpoint(path,device="cpu"):
    path=Path(path).resolve()
    extra=json.loads(path.with_suffix(".on-policy.json").read_text())
    if extra.get("format")!=FORMAT or extra.get("contract")!=contract() \
            or extra.get("sources")!=source_hashes() or extra.get("split_sha256")!=FROZEN_SPLIT_SHA256 \
            or extra.get("checkpoint_sha256")!=file_sha256(path) \
            or extra.get("objective")!=runtime_base.OBJECTIVE or extra.get("exact_simulator_resume") is not False:
        raise ValueError("Invalid precision on-policy source/profile/split manifest")
    model=load_precision_checkpoint(path,device=device)
    config=model.precision_dataset_config
    if config.get("purpose")!=FORMAT or config.get("on_policy_contract")!=contract() \
            or config.get("source_hashes")!=source_hashes() \
            or config.get("frozen_split_sha256")!=FROZEN_SPLIT_SHA256 \
            or config.get("parent_checkpoint")!=extra.get("parent") \
            or model.num_timesteps!=extra.get("optimized_controls") \
            or not 0<model.num_timesteps<=65536 or model.num_timesteps%1024:
        raise ValueError("Precision on-policy update/provenance mismatch")
    for name,value in PPO.items():
        if name in ("n_steps","batch_size"):
            continue
        actual=getattr(model,name)
        if name=="clip_range":
            actual=actual(1.)
        if actual!=value or isinstance(actual,bool):
            raise ValueError(f"Serialized on-policy setting differs: {name}")
    if model.rollout_buffer.channels_per_frame!=55:
        raise ValueError("Compact precision buffer must use55 channels per frame")
    return model


def _evaluate(model,checkpoint,split_path,output,allowance):
    split=json.loads(Path(split_path).read_text())
    if file_sha256(split_path)!=FROZEN_SPLIT_SHA256:
        raise ValueError("Frozen split changed before DEV evaluation")
    rows=[]
    for i,case in enumerate(development_plan(split)):
        allowance.check("development_case")
        row=evaluate_development_case(
            model,checkpoint=checkpoint,split_path=split_path,
            config_index=case["config_index"],output=output/f"case-{i:02d}.json",
            budget=allowance,offsets_enabled=True,intervention_decision=case["intervention_decision"],
            noise_seed=case["noise_seed"],record_video=False,max_frames=12000)
        rows.append(row)
    normal=[r for r in rows if r["intervention_decision"] is None]
    stress=[r for r in rows if r["intervention_decision"] is not None]
    return dict(status="completed",cases=rows,
                normal_clears=sum(r["ordinary_model_clear"] for r in normal),normal_cases=len(normal),
                recovery_successes=sum(r["recovery_success"] for r in stress),stress_cases=len(stress),
                mean_kills=float(np.mean([r["episode"]["kills"] for r in rows])))


def run_on_policy(initial_checkpoint,output,split_path,budget,*,steps=65536,seed=0,
                  noise_probability=.001,device="cpu",anchor_dataset=None):
    if anchor_dataset is not None:
        raise ValueError("First on-policy version intentionally has no BC anchor regularization")
    if type(steps) is not int or not 1024<=steps<=65536 or steps%1024:
        raise ValueError("steps must be a multiple of1024 in1024..65536")
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("seed must be uint32")
    if str(device)=="auto" or torch.device(device).type not in ("cpu","cuda"):
        raise ValueError("Use an explicit CPU or CUDA device")
    if isinstance(noise_probability,bool) or not np.isfinite(noise_probability) or not 0<=noise_probability<=1:
        raise ValueError("Invalid noise_probability")
    context=load_train_context(split_path)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent=load_precision_checkpoint(initial_checkpoint,device=device)
    parent_record=_parent_record(initial_checkpoint,parent)
    output.mkdir(parents=True)
    started=time.monotonic()
    report=dict(format=FORMAT,status="running",stop_reason=None,error=None,contract=contract(),
                optimized_controls=0,requested_controls=steps,noise_probability=noise_probability,
                seed=seed,device=str(device),checkpoints=[],updates=[],evaluations=[],
                baseline=None,parent=parent_record,winner_selected=False,final_evaluated=False)
    runtime=account=allowance=None
    provenance=None
    try:
        snapshot=budget.snapshot()
        if not 0<snapshot["max_interactions"]<=4000000:
            raise ValueError("Global NEW tranche must remain within4M interactions")
        deadline=started+min(1800.,snapshot["remaining_seconds"],max(0.,GLOBAL_EPOCH-time.time()))
        if deadline<=time.monotonic():
            raise AllowanceStop("No remaining trial wall time")
        monitor=ResourceMonitor(output/"resources.jsonl",device=str(device).split(":")[0])
        monitor.check("admission",admission_slots=1,force=True)
        amount=min(TRIAL_CAP,int(snapshot["remaining_interactions"]))
        if amount<1:
            raise AllowanceStop("No remaining global interactions")
        allowance=TrialAllowance(budget,"on-policy-"+hashlib.sha256(str(output).encode()).hexdigest()[:24],
                                  amount,deadline,monitor)
        account=TrainingAccount(allowance)
        provenance=dict(sources=source_hashes(),parent=parent_record,split=context)
        _write(output/"config.json",dict(contract=contract(),parent=parent_record,sources=provenance["sources"],
                                        allowance=allowance.snapshot(),noise_probability=noise_probability,
                                        offsets_note="Full precision offsets enabled for PPO; parent setting retained in parent config",
                                        steps=steps,seed=seed))
        report["baseline"]=_evaluate(parent,initial_checkpoint,split_path,output/"baseline-dev",allowance)
        runtime=OnlineRuntime(parent,split_path,output,account,steps=steps,seed=seed,
                              noise_probability=noise_probability,device=device)
        del parent
        while runtime.optimized_steps<steps:
            allowance.check("rollout_start")
            before=account.agent_completed
            complete=runtime.collect()
            if not complete or account.agent_completed-before!=1024:
                raise AllowanceStop("Partial rollout is not optimized")
            update=runtime.optimize()
            report["optimized_controls"]=runtime.optimized_steps
            tail=runtime.callback.episodes[-100:]
            update["last100"]=dict(episodes=len(tail),wins=sum(r["clear"] for r in tail))
            report["updates"].append(update)
            if runtime.optimized_steps%8192==0:
                record=save_online(runtime,output/f"optimized-{runtime.optimized_steps:06d}.zip",provenance,"periodic")
                report["checkpoints"].append(record)
            _write(output/"report.json",report)
            if runtime.optimized_steps%16384==0:
                evaluation=_evaluate(runtime.model,report["checkpoints"][-1]["path"],split_path,
                                     output/f"dev-{runtime.optimized_steps:06d}",allowance)
                evaluation["optimized_controls"]=runtime.optimized_steps
                report["evaluations"].append(evaluation)
        report["checkpoints"].append(save_online(runtime,output/"final.zip",provenance,"final"))
        report.update(status="completed",stop_reason="control_cap")
    except (AllowanceStop,ProbeStop,TimeoutError) as error:
        report.update(status="partial",stop_reason="allowance_or_deadline",error=str(error))
    except BaseException as error:
        report.update(status="failed",stop_reason="error",error=f"{type(error).__name__}: {error}")
    finally:
        if runtime is not None:
            report["optimized_controls"]=runtime.optimized_steps
            report["training_episodes"]=len(runtime.callback.episodes)
            report["noise_interventions"]=runtime.callback.interventions
            report["intended_action_counts"]=dict(runtime.callback.intended_counts)
            report["executed_action_counts"]=dict(runtime.callback.executed_counts)
            report["completed_train_family_counts"]=dict(runtime.callback.family_counts)
            report["current_train_openings"]=runtime.env.current_openings()
            if report["status"]!="completed" and runtime.optimized_steps and runtime.state_verified:
                try:
                    report["checkpoints"].append(save_online(runtime,output/"partial-optimized.zip",provenance,"partial_optimized"))
                except Exception as error:
                    report["checkpoint_error"]=f"{type(error).__name__}: {error}"
            try:
                runtime.close()
            except Exception as error:
                report.update(status="failed",stop_reason="cleanup_error",
                              cleanup_error=f"{type(error).__name__}: {error}")
        if account is not None:
            report["training_accounting"]=account.snapshot()
        if allowance is not None:
            certain=(account is not None and account.attempted==account.completed and not allowance.pending)
            allowance.finish(certain=certain)
            report["trial_accounting"]=allowance.snapshot()
            report["outer_reservation_settled"]=allowance.finished
            report["total_attempted_calls"]=allowance.used
            report["training_completed_calls"]=account.completed
            report["development_settled_attempted_calls"]=sum(
                row["actual"] or 0 for row in allowance.evaluations)
        report["elapsed_seconds"]=time.monotonic()-started
        _write(output/"report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--split-path",required=True)
    parser.add_argument("--budget",required=True)
    parser.add_argument("--steps",type=int,default=65536)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--noise-probability",type=float,default=.001)
    parser.add_argument("--device",default="cpu")
    args=vars(parser.parse_args(argv))
    args["budget"]=WithinStageBudget(args["budget"])
    result=run_on_policy(**args)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result["status"]=="completed" else 2


if __name__=="__main__":
    raise SystemExit(main())
