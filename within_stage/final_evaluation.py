"""One-shot new-tranche FINAL assessment; no real execution at import/selection.

make_final_selection(model, checkpoint, split_path, loader="precision"|"entity",
                     offsets_enabled=bool|None, record_video=False) -> dict
run_final_evaluation(model, checkpoint, output, budget, selection, split_path)

The caller persists/chooses the selection BEFORE unsealing. The runner exclusively
creates the fixed NEW-tranche receipt before its first case or ROM initialization.
Any receipt, including an incomplete/malformed one, prohibits all future retries.
Partial assessments never qualify. No training, teacher takeover or RAM resets.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from recordings import BestRecordings
from rust_resources import ResourceMonitor, ResourcePressureError
from rust_validation import PrimitiveActions, _evaluation_state, _seed_episode, _primitive
from learning_lab.budget import BudgetExhausted
from within_stage.observation import OBSERVATION_VERSION, PrecisionObservation
from within_stage.entity_observation import entity_space, entity_schema, EntityObservationEnv
from within_stage.precision_policy import precision_schema, state_sha256, load_precision_checkpoint
from within_stage.entity_policy import load_entity_checkpoint
from within_stage.evaluation import ObservationAdapter
from within_stage.execution_noise import RecoveryStressEnv
from within_stage.opening_pool import load_pool, make_opening_env
from within_stage.openings import ProbeAccounting, ProbeStop

ROOT=Path(__file__).resolve().parents[1]
SELECTION_FORMAT="within-stage-final-selection-v1"
REPORT_FORMAT="within-stage-final-assessment-v1"
LOADERS={"precision":"within_stage.precision_policy.load_precision_checkpoint",
         "entity":"within_stage.entity_policy.load_entity_checkpoint"}

RECEIPT_PATH=ROOT/"runs"/"20260930-within-stage"/"within-stage-final-used.json"
FROZEN_SPLIT_SHA256="a94de5882737587e06930d6d782d512228f3248958df26f188fc2b494816eebe"
GLOBAL_DEADLINE_EPOCH=datetime.fromisoformat("2026-10-01T08:30:25.131+08:00").timestamp()


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _write(path,value):
    path=Path(path);pending=path.with_name(path.name+".pending")
    pending.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    pending.replace(path)


def _trusted_load(kind,path):
    if kind=="precision":return load_precision_checkpoint(path,device="cpu")
    if kind=="entity":return load_entity_checkpoint(path,device="cpu")
    raise ValueError("Only own precision/entity strict checkpoint loaders are permitted")


def _source_hashes():
    return {name:_sha(ROOT/name) for name in (
        "within_stage\\final_evaluation.py","within_stage\\precision_policy.py",
        "within_stage\\entity_policy.py","within_stage\\entity_observation.py",
        "within_stage\\observation.py","within_stage\\evaluation.py",
        "within_stage\\execution_noise.py","within_stage\\opening_pool.py",
        "within_stage\\early_openings.py","within_stage\\openings.py",
        "game_state.py","heuristic.py","battle_city_env.py","battle_city.nes",
        "rust_validation.py","recordings.py")}


def _identity(model):
    optimizer=getattr(model.policy,"optimizer",None)
    return dict(policy_sha256=state_sha256(model.policy.state_dict()),
                optimizer_sha256=state_sha256(optimizer.state_dict()) if optimizer is not None else None)


def _schema(kind):
    return precision_schema() if kind=="precision" else entity_schema()


def _model_binding(model,checkpoint,kind):
    if not isinstance(model.action_space,gym.spaces.Discrete) or model.action_space.n!=10 \
            or model.action_space.start!=0:
        raise ValueError("FINAL policy must have exactly ten primitives")
    if kind=="precision":
        if not isinstance(model.observation_space,gym.spaces.Box) \
                or model.observation_space.shape!=(110,26,26) or model.observation_space.dtype!=np.float32:
            raise ValueError("Precision FINAL input must be float32 Box110")
    elif kind=="entity":
        if model.observation_space!=entity_space():
            raise ValueError("Entity FINAL schema differs")
    else:raise ValueError("Unknown selected loader")
    with _evaluation_state(model):
        loaded=_trusted_load(kind,checkpoint)
        if type(model) is not type(loaded) or type(model.policy) is not type(loaded.policy) \
                or getattr(model.predict,"__func__",None) is not getattr(type(model),"predict",None) \
                or _identity(model)!=_identity(loaded):
            raise ValueError("Passed model parameters/Adam do not match the strict checkpoint")
        del loaded
    return _identity(model)


def _load_protocol(split_path):
    split_path=Path(split_path).resolve()
    if _sha(split_path)!=FROZEN_SPLIT_SHA256:
        raise ValueError("FINAL frozen split SHA256 mismatch")
    split=json.loads(split_path.read_text())
    if any(split.get(k)!=v for k,v in (
        ("version","within-stage-family-split-v1"),("stage",1),("max_frames",12000),
        ("post_start_frames",150),("channels_per_frame",55),("n_stack",2),("actions",10),
        ("observation_version",OBSERVATION_VERSION))):
        raise ValueError("FINAL opening/precision/full-game protocol mismatch")
    pool_path=Path(split["pool_path"]).resolve()
    if _sha(pool_path)!=split["pool_sha256"]:
        raise ValueError("Frozen FINAL opening pool hash mismatch")
    pool=load_pool(pool_path)
    target=split["final"]
    families,reps,aliases=target["family_ids"],target["representative_config_indices"],target["all_config_indices"]
    if len(families)!=20 or len(set(families))!=20 or len(reps)!=20 or len(set(reps))!=20 \
            or len(aliases)!=len(set(aliases)):
        raise ValueError("FINAL requires20 distinct frozen families/representatives")
    other_families=set(split["train"]["family_ids"])|set(split["development"]["family_ids"])
    other_configs=set(split["train"]["all_config_indices"])|set(split["development"]["all_config_indices"])
    if set(families)&other_families or set(aliases)&other_configs:
        raise ValueError("FINAL overlaps TRAIN/DEV")
    by_config={r["config_index"]:r for r in pool["specs"]}
    grouped={f["family_id"]:set(f["config_indices"]) for f in pool["families"]}
    if set(aliases)!={i for f in families for i in grouped.get(f,set())}:
        raise ValueError("FINAL alias membership differs from frozen family groups")
    schedule=split["recovery_final_schedule"]
    if len(schedule)!=20:
        raise ValueError("Expected20 frozen movement-stress cases")
    normal,stress=[],[]
    for i,(family,config) in enumerate(zip(families,reps)):
        spec=by_config.get(config)
        if spec is None or spec["family_id"]!=family or config not in aliases \
                or spec["spec"].get("post_start_frames")!=150:
            raise ValueError("FINAL representative does not belong to its frozen family")
        event=schedule[i]
        if event.get("config_index")!=config or event.get("family_id")!=family \
                or type(event.get("decision_index")) is not int or event["decision_index"] not in (100,200) \
                or event.get("noise_seed")!=20261001000+i or event.get("allowed_actions")!=[0,1,2,3,4]:
            raise ValueError("FINAL recovery schedule differs from the predeclared movement protocol")
        common=dict(config_index=config,family_id=family,opening_spec=deepcopy(spec["spec"]),
                    initial_ram_sha256=spec["initial_ram_sha256"],
                    initial_pixels_sha256=spec.get("initial_pixels_sha256"),max_frames=12000,reset_seed=0)
        normal.append(dict(common,case_index=i,mode="normal",intervention_decision=None,noise_seed=0))
        stress.append(dict(common,case_index=20+i,mode="recovery",intervention_decision=event["decision_index"],
                           noise_seed=event["noise_seed"],allowed_actions=[0,1,2,3,4]))
    return split,pool,normal+stress


def make_final_selection(model,checkpoint,split_path,*,loader,offsets_enabled=None,record_video=False):
    """Build selection without receipt/environment; caller persists before unsealing."""
    if RECEIPT_PATH.exists():raise FileExistsError("New-tranche FINAL has already been used")
    if loader not in LOADERS:raise ValueError("loader must be precision or entity")
    if type(record_video) is not bool:raise ValueError("record_video must be boolean")
    if loader=="precision" and type(offsets_enabled) is not bool:
        raise ValueError("Precision offset adapter must be explicitly True or False")
    if loader=="entity" and offsets_enabled is not None:
        raise ValueError("Entity observations have their own schema, not a precision adapter")
    checkpoint,split_path=Path(checkpoint).resolve(),Path(split_path).resolve()
    split,pool,plan=_load_protocol(split_path)
    identity=_model_binding(model,checkpoint,loader)
    selection=dict(
        format=SELECTION_FORMAT,kind=loader,loader=LOADERS[loader],
        checkpoint=str(checkpoint),checkpoint_sha256=_sha(checkpoint),
        checkpoint_manifest_sha256=_sha(checkpoint.with_suffix(".manifest.json")),
        **identity,split_path=str(split_path),split_sha256=FROZEN_SPLIT_SHA256,
        pool_path=str(Path(split["pool_path"]).resolve()),pool_sha256=split["pool_sha256"],
        schema=_schema(loader),schema_sha256=_json_hash(_schema(loader)),
        offsets_enabled=offsets_enabled,record_video=record_video,sources=_source_hashes(),
        deterministic=True,stage=1,max_frames=12000,case_plan=plan,
        targets=dict(normal_required=16,normal_planned=20,recovery_required=16,recovery_planned=20),
        rules="No training/reselection after receipt; no repeated/resampled failures; unapplied stress stays in planned denominator.",
    )
    selection["selection_sha256"]=_json_hash(selection)
    return selection


def _validate_selection(model,checkpoint,selection,split_path):
    claimed=selection.get("selection_sha256")
    content={k:v for k,v in selection.items() if k!="selection_sha256"}
    if _json_hash(content)!=claimed:
        raise ValueError("Final selection was modified")
    rebuilt=make_final_selection(model,checkpoint,split_path,loader=selection.get("kind"),
                                 offsets_enabled=selection.get("offsets_enabled"),
                                 record_video=selection.get("record_video"))
    if rebuilt!=selection:raise ValueError("Selected checkpoint/source/split/protocol identity differs")
    return _load_protocol(split_path)


def _wrap_observation(raw,kind,deadline):
    primitive=PrimitiveActions(raw,deadline=deadline)
    return PrecisionObservation(primitive) if kind=="precision" else EntityObservationEnv(primitive)


def _input(kind,observation,previous=None):
    if kind=="entity":
        if not isinstance(observation,dict) or observation.get("grid",np.zeros(0)).shape!=(110,26,26) \
                or observation.get("entities",np.zeros(0)).shape!=(18,16):
            raise ValueError("Entity wrapper must own exactly one110 history and18x16 entities")
        return deepcopy(observation)
    if not isinstance(observation,np.ndarray) or observation.shape!=(55,26,26) \
            or observation.dtype!=np.float32:
        raise ValueError("Precision wrapper must return55 channels before stacking")
    older=np.zeros_like(observation) if previous is None else previous[55:]
    return np.concatenate((older,observation),axis=0)


def _case(model,checkpoint,selection,pool,case,output,budget,deadline_epoch):
    result=dict(case,status="running",ordinary_model_clear=False,recovery_success=False,
                intervention_applied=False,clear_verified=False,reservation=None,settled=False,
                actions=[],executed_actions=[],accounting=None,error=None)
    path=output/f"case-{case['case_index']:02d}"
    path.mkdir(exist_ok=False)
    raw=env=stress=recorder=account=None
    try:
        if _identity(model)!=dict(policy_sha256=selection["policy_sha256"],
                                 optimizer_sha256=selection["optimizer_sha256"]) \
                or _source_hashes()!=selection["sources"] or _sha(checkpoint)!=selection["checkpoint_sha256"] \
                or _sha(selection["split_path"])!=selection["split_sha256"] \
                or _sha(selection["pool_path"])!=selection["pool_sha256"]:
            raise ValueError("Selected model/checkpoint/source/opening identity changed before case")
        snapshot=budget.snapshot()
        if type(snapshot.get("max_interactions")) is not int or not 0<snapshot["max_interactions"]<=4000000:
            raise ValueError("New-tranche FINAL budget must remain within4M interactions")
        remaining=min(float(snapshot["remaining_seconds"]),float(snapshot["deadline_epoch"])-time.time(),
                      deadline_epoch-time.time())
        if not math.isfinite(remaining):raise ValueError("Invalid FINAL budget deadline")
        if remaining<=0:raise ProbeStop("Final assessment deadline exhausted")
        deadline=time.monotonic()+remaining
        spec=case["opening_spec"]
        allowance=181+10*31+1+150+int(spec["title_wait"])+int(spec["stage_wait"])+3000
        token=budget.reserve("final-"+hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24],allowance)
        result.update(reservation=token,reserved=allowance)
        monitor=ResourceMonitor(path/"resources.jsonl",device=str(model.device).split(":")[0])
        account=ProbeAccounting(allowance,deadline=deadline,monitor=monitor)
        monitor.check("final_case_admission",force=True)
        raw=make_opening_env(pool,case["config_index"],account=account,max_frames=12000)
        env=_wrap_observation(raw,selection["kind"],deadline)
        if case["mode"]=="recovery":
            stress=RecoveryStressEnv(env,(case["intervention_decision"],),
                                     allowed_perturbation_actions=tuple(case["allowed_actions"]),
                                     noise_seed=case["noise_seed"],deadline=deadline)
            env=stress
        predictor=ObservationAdapter(model,offsets_enabled=selection["offsets_enabled"]) \
            if selection["kind"]=="precision" else model
        if selection["record_video"]:
            recorder=BestRecordings(output/"videos",fps=15,scale=3,limit=10)
        with _evaluation_state(model):
            _seed_episode(0)
            observation,_=env.reset(seed=0)
            initial_ram,initial_pixels=raw.nes.ram.copy(),raw.render().copy()
            result["initial_ram_sha256"]=_hash_bytes(initial_ram)
            result["initial_pixels_sha256"]=_hash_bytes(initial_pixels)
            np.savez_compressed(path/"initial.npz",ram=initial_ram,pixels=initial_pixels)
            if result["initial_ram_sha256"]!=case["initial_ram_sha256"] \
                    or (case.get("initial_pixels_sha256") is not None
                        and result["initial_pixels_sha256"]!=case["initial_pixels_sha256"]):
                raise ValueError("Initial RAM/pixels do not match the frozen FINAL opening")
            if raw.state.remaining_enemies!=20 or raw.state.base_destroyed or raw.state.lives!=3:
                raise ValueError("Frozen opening lacks ordinary initial count/lives/base")
            if recorder is not None:
                recorder.begin(initial_pixels,dict(case_index=case["case_index"],mode=case["mode"],
                                                  selection_sha256=selection["selection_sha256"]))
            history=_input(selection["kind"],observation)
            terminated=truncated=False
            with (path/"actions.jsonl").open("x",encoding="utf-8") as trace:
                while not (terminated or truncated):
                    account.check()
                    if account.agent_attempted>=3000:raise ProbeStop("Full12000-frame control bound reached")
                    masks=np.asarray(env.get_wrapper_attr("action_masks")()).copy()
                    if masks.shape!=(10,) or masks.dtype!=np.bool_ or not masks.any():
                        raise ValueError("Invalid FINAL primitive mask")
                    action=_primitive(predictor.predict(history,deterministic=True,action_masks=masks)[0])
                    if not masks[action]:raise ValueError("Selected model returned a masked action")
                    event=dict(decision=len(result["actions"]),intended_action=action,step_completed=False)
                    result["actions"].append(action)
                    try:
                        observation,_,terminated,truncated,info=env.step(action)
                        executed=info.get("execution_noise",{}).get("executed_action",action)
                        result["executed_actions"].append(int(executed))
                        event.update(step_completed=True,executed_action=int(executed),
                                     execution_noise=info.get("execution_noise"),outcome=info.get("outcome",""),
                                     frames=info.get("frames"),kills=info.get("kills"))
                    finally:
                        trace.write(json.dumps(event)+"\n")
                        trace.flush()
                    history=_input(selection["kind"],observation,history)
                    if recorder is not None:recorder.write(raw.render())
            outcome=info.get("outcome")
            if outcome not in ("clear","base_lost","lives_exhausted","timeout"):
                raise ValueError("Missing authoritative full-game terminal outcome")
            state=raw.state
            proof=(state.remaining_enemies==0 and not state.base_destroyed and state.lives>0
                   and not state.game_over)
            result.update(outcome=outcome,frames=int(info["frames"]),kills=int(info["kills"]),
                          deaths=int(info["deaths"]),remaining_enemies=int(state.remaining_enemies),
                          base_destroyed=bool(state.base_destroyed),lives=int(state.lives))
            np.savez_compressed(path/"terminal.npz",ram=raw.nes.ram.copy(),pixels=raw.render().copy())
            if outcome=="clear" and not proof:raise ValueError("Claimed clear lacks terminal-state proof")
            if _identity(model)!=dict(policy_sha256=selection["policy_sha256"],
                                     optimizer_sha256=selection["optimizer_sha256"]) \
                    or _source_hashes()!=selection["sources"] or _sha(checkpoint)!=selection["checkpoint_sha256"]:
                raise ValueError("Model/source/checkpoint mutated during FINAL case")
            result["clear_verified"]=bool(outcome=="clear" and proof)
            result["ordinary_model_clear"]=result["clear_verified"] and stress is None
            if stress is not None:
                record=asdict(stress.last_episode)
                result["stress"]=record
                result["intervention_applied"]=record["applied_interventions"]==1
                result["recovery_success"]=bool(
                    result["clear_verified"] and record["completed"] and record["recovery_rate_valid"]
                    and record["requested_interventions"]==1 and record["applied_interventions"]==1)
            if recorder is not None:
                result["video"]=str(recorder.finish(info))
            result["status"]="completed"
        budget.settle(token,account.attempted);result["settled"]=True
    except (ProbeStop,BudgetExhausted,ResourcePressureError,TimeoutError) as error:
        result.update(status="partial",error=f"{type(error).__name__}: {error}")
        if account is not None and account.attempted==account.completed:
            budget.settle(result["reservation"],account.attempted);result["settled"]=True
    except Exception as error:
        result.update(status="failed",error=f"{type(error).__name__}: {error}")
    finally:
        if raw is not None:
            try:
                np.savez_compressed(path/"last-observed.npz",ram=raw.nes.ram.copy(),pixels=raw.render().copy())
            except Exception as error:
                result.update(status="failed",error=f"Snapshot failure: {type(error).__name__}: {error}",
                              snapshot_error=f"{type(error).__name__}: {error}")
        for resource in (recorder,env if env is not None else raw):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    result.update(status="failed",error=f"Cleanup failure: {type(error).__name__}: {error}",
                                  cleanup_error=f"{type(error).__name__}: {error}")
        if stress is not None and stress.last_episode is not None and "stress" not in result:
            result["stress"]=asdict(stress.last_episode)
            result["intervention_applied"]=result["stress"]["applied_interventions"]==1
        if result["status"]!="completed":
            result["ordinary_model_clear"]=False
            result["recovery_success"]=False
            result["clear_verified"]=False
        if account is not None:result["accounting"]=account.snapshot()
        _write(path/"report.json",result)
    return result


def _hash_bytes(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def summarize(cases):
    normal=[r for r in cases if r.get("mode")=="normal"]
    stress=[r for r in cases if r.get("mode")=="recovery"]
    normals=sum(r.get("status")=="completed" and r.get("ordinary_model_clear",False) for r in normal)
    recoveries=sum(r.get("status")=="completed" and r.get("recovery_success",False)
                   and r.get("intervention_applied",False) for r in stress)
    applied=sum(bool(r.get("intervention_applied",False)) for r in stress)
    complete=len(normal)==20 and len(stress)==20 and all(r.get("status")=="completed" for r in cases)
    return dict(normal_planned=20,normal_completed=sum(r["status"]=="completed" for r in normal),
                normal_clears=normals,normal_planned_rate=normals/20,
                recovery_planned=20,recovery_completed=sum(r["status"]=="completed" for r in stress),
                recovery_applied=applied,recovery_successes=recoveries,recovery_planned_rate=recoveries/20,
                recovery_conditional_rate=recoveries/applied if applied else None,
                recovery_unapplied_or_unrun=20-applied,all40_completed=complete,
                qualified=bool(complete and normals>=16 and recoveries>=16))


def run_final_evaluation(model,checkpoint,output,budget,selection,split_path):
    """Consume the one-shot receipt before any FINAL environment; never retry."""
    if RECEIPT_PATH.exists():raise FileExistsError("FINAL receipt exists; no rerun is authorized")
    checkpoint,output,split_path=Path(checkpoint).resolve(),Path(output).resolve(),Path(split_path).resolve()
    if output.exists():raise FileExistsError(output)
    split,pool,plan=_validate_selection(model,checkpoint,selection,split_path)
    receipt=dict(format=REPORT_FORMAT,status="started_one_shot_consumed",selection=deepcopy(selection),
                 selection_sha256=selection["selection_sha256"],output=str(output),
                 created_epoch=time.time(),global_deadline_epoch=GLOBAL_DEADLINE_EPOCH,
                 rules="Any receipt consumes this FINAL tranche, including partial/failed assessments.")
    RECEIPT_PATH.parent.mkdir(parents=True,exist_ok=True)
    with RECEIPT_PATH.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    output.mkdir(parents=True,exist_ok=False)
    report=dict(format=REPORT_FORMAT,status="running",selection=deepcopy(selection),
                receipt=str(RECEIPT_PATH),receipt_sha256=_sha(RECEIPT_PATH),
                cases=[],error=None,unrun_case_indices=list(range(40)))
    started=time.monotonic()
    try:
        _write(output/"selection.json",selection)
        for case in plan:
            result=_case(model,checkpoint,selection,pool,case,output,budget,GLOBAL_DEADLINE_EPOCH)
            report["cases"].append(result)
            report["unrun_case_indices"].remove(case["case_index"])
            _write(output/"report.json",report)
            if result["status"]!="completed":
                report.update(status=result["status"],error=result["error"])
                break
        else:report["status"]="completed"
    except Exception as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
    finally:
        report["summary"]=summarize(report["cases"])
        accounts=[r["accounting"] for r in report["cases"] if r.get("accounting")]
        report["accounting"]={key:sum(a[key] for a in accounts) for key in
            ("attempted","completed","raw_initialization_attempted","raw_initialization_completed",
             "agent_attempted","agent_completed","agent_raw_frames")}
        report["unsettled_reservations"]=[r["reservation"] for r in report["cases"]
                                          if r.get("reservation") and not r["settled"]]
        report["elapsed_seconds"]=time.monotonic()-started
        if report["status"]!="completed":report["summary"]["qualified"]=False
        _write(output/"report.json",report)
    return report
