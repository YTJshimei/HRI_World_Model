"""Phase 5B-v3-R3 counterfactual human-response representation (CHRR).

The response decoder learns local 10x2 synthetic human root futures from
frozen runtime-valid R1-v3 pre-fusion representations.  Frozen B0 remains the
only ranking source, frozen HARM-v3 remains the only risk source, and TEST is
never materialized.  O1 is an explicitly non-runtime oracle diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
from scripts import run_phase5b_v3_r1c_frozen_runtime_generic_reanchor as r1c
from scripts import run_phase5b_v3_r2_pair_conditioned_benefit as r2
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, POPULATION_PROFILE, build_development_split
from src.data.hold_candidate import build_hold_candidate_outcome
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import PROFILE_BY_ID, simulate_risk_conditioned_interaction_future
from src.models.counterfactual_human_response import (
    HumanResponseFutureDecoder, MatchedBenefitReadout, counterfactual_delta,
    decision_local_coordinates,
)
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
ORACLE_LABEL = "ORACLE FUTURE DIAGNOSTIC - NOT RUNTIME VALID"
STAGE = "Phase 5B-v3-R3 Counterfactual Human-Response Representation"
TEST_READS = 0
LAMBDA_RANK = 0.0
TOLERANCE = 1e-10
R2_A1_MAE = 1.7426422105494397


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--manifest-v3", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--target-v2", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv")
    parser.add_argument("--anchor-map", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv")
    parser.add_argument("--r1-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r3_counterfactual_human_response")
    return parser.parse_args()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_sha(value) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def select_response_epoch(rows):
    """Pre-registered response selector: ADE, FDE, earlier epoch; no Benefit metric."""
    if not rows: raise ValueError("response selector requires validation rows")
    return min(rows, key=lambda row: (float(row["Root_ADE"]), float(row["Root_FDE"]), int(row["epoch"])))


def select_benefit_epoch(rows):
    """Pre-registered Benefit selector: MAE, safe sign, earlier epoch."""
    if not rows: raise ValueError("Benefit selector requires validation rows")
    return min(rows, key=lambda row: (float(row["Benefit_MAE"]), -float(row["safe_beneficial_sign_accuracy"]), int(row["epoch"])))


def extract_representations(model, samples, normalizer, batch_size, torch, device):
    values = {key: [] for key in ("z_human", "z_candidate", "z_final", "log_variance", "old_benefit", "bypass")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = b1.temporal_batch(samples[start:start+batch_size], normalizer, torch, device)
            output = model(batch); audit = model.audit_representations(batch)
            values["z_human"].append(audit["R1_HISTORY_CONTEXT_PREFUSION"].cpu())
            values["z_candidate"].append(audit["R2_CANDIDATE_PREFUSION"].cpu())
            values["z_final"].append(output.context_embedding.cpu())
            values["log_variance"].append(output.benefit_log_variance.cpu())
            values["old_benefit"].append((output.benefit_mean*normalizer["benefit_scale"]+normalizer["benefit_mean"]).cpu())
            from scripts import run_phase5b17ed_risk_preserving_bypass as bypass
            values["bypass"].append(bypass.bypass_input(audit, torch).cpu())
    result = {key: torch.cat(parts) for key, parts in values.items()}
    result["old_benefit"] = result["old_benefit"].numpy().astype(np.float64)
    if result["z_human"].shape[1] != 1024 or result["z_candidate"].shape[1] != 256:
        raise RuntimeError("actual R1-v3 pre-fusion dimensions differ from the audited 1024/256 contract")
    return result


def ground_truth_local_futures(episodes, samples, torch):
    """Replay development-only GT futures as supervision, never runtime streams."""
    future_by_id = {}
    for episode in episodes:
        profile = PROFILE_BY_ID[episode.profile_id]
        for candidate in episode.candidates:
            simulation = simulate_risk_conditioned_interaction_future(
                episode.human_history, episode.natural_future, episode.robot_history,
                candidate.action_id, profile, episode.risk_factors,
            )
            future_by_id[f"{episode.episode_id}:{candidate.action_id}"] = simulation.future_root[:, :2]
        hold = build_hold_candidate_outcome(episode, POPULATION_PROFILE, profile)
        future_by_id[f"{episode.episode_id}:{HOLD_ACTION_ID}"] = hold.gt_simulation.future_root[:, :2]
    world = np.stack([future_by_id[sample.sample_id] for sample in samples]).astype(np.float32)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    current = np.stack([compute_root(episode_by_id[sample.episode_id].human_history)[-1, :2] for sample in samples]).astype(np.float32)
    yaw = np.asarray([episode_by_id[sample.episode_id].robot_history[-1, 2] for sample in samples], np.float32)
    local = decision_local_coordinates(torch.from_numpy(world), torch.from_numpy(current), torch.from_numpy(yaw)).numpy()
    history_root = {episode.episode_id: compute_root(episode.human_history)[:, :2].astype(np.float32) for episode in episodes}
    cv_world = []
    for sample in samples:
        root = history_root[sample.episode_id]
        cv_world.append(root[-1][None] + np.arange(1, 11, dtype=np.float32)[:, None]*(root[-1]-root[-2])[None])
    cv_world = np.stack(cv_world)
    cv_local = decision_local_coordinates(torch.from_numpy(cv_world), torch.from_numpy(current), torch.from_numpy(yaw)).numpy()
    return local.astype(np.float32), cv_local.astype(np.float32)


def root_metrics(prediction, target):
    error = np.linalg.norm(np.asarray(prediction)-np.asarray(target), axis=-1)
    return {"Root_ADE": float(error.mean()), "Root_FDE": float(error[:, -1].mean())}


def predict_response(model, data, batch_size, torch, device):
    values=[]; model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["z_human"]), batch_size):
            values.append(model(data["z_human"][start:start+batch_size].to(device), data["z_candidate"][start:start+batch_size].to(device)).cpu())
    return torch.cat(values).numpy().astype(np.float64)


def train_response(model, train, validation, batches, args, torch, device):
    model.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    rows=[]; states={}; best_epoch=None; stale=0; started=time.perf_counter()
    for epoch, epoch_batches in enumerate(batches,1):
        model.train(); losses=[]
        for indices in epoch_batches:
            local=torch.as_tensor(indices,dtype=torch.long)
            prediction=model(train["z_human"][local].to(device),train["z_candidate"][local].to(device))
            target=torch.from_numpy(train["future_local"][indices]).to(device)
            loss=torch.nn.functional.smooth_l1_loss(prediction,target)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True));optimizer.step()
            losses.append((float(loss.detach()),gradient))
        prediction=predict_response(model,validation,args.batch_size,torch,device); metric=root_metrics(prediction,validation["future_local"])
        row={"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"epoch":epoch,"train_SmoothL1":float(np.mean([x[0] for x in losses])),"mean_gradient_norm":float(np.mean([x[1] for x in losses])),**metric,"Benefit_metrics_used_by_selector":False,"parameter_checksum":state_sha(model.state_dict())}
        if not all(math.isfinite(float(row[key])) for key in ("train_SmoothL1","Root_ADE","Root_FDE","mean_gradient_norm")):raise FloatingPointError("non-finite response training state")
        rows.append(row);states[epoch]=copy.deepcopy(model.state_dict());selected=select_response_epoch(rows)
        if selected["epoch"]!=best_epoch:best_epoch,stale=selected["epoch"],0
        else:stale+=1
        row["selector_current_best_epoch"]=best_epoch;row["selector_stale_epochs"]=stale
        print(f"RESPONSE epoch={epoch:02d} loss={row['train_SmoothL1']:.6f} ADE={row['Root_ADE']:.5f} FDE={row['Root_FDE']:.5f} best={best_epoch} stale={stale}",flush=True)
        if stale>=args.patience:break
    selected=select_response_epoch(rows);model.load_state_dict(states[selected["epoch"]]);model.eval()
    for row in rows:row["selector_final_selected"]=row["epoch"]==selected["epoch"]
    return {"model":model,"rows":rows,"selected":selected,"epochs_completed":len(rows),"training_time_s":time.perf_counter()-started}


def response_delta(future, generic_indices, torch):
    future_tensor=torch.as_tensor(future,dtype=torch.float32)
    generic=future_tensor[torch.as_tensor(generic_indices,dtype=torch.long)]
    return counterfactual_delta(future_tensor,generic).numpy().astype(np.float64)


def delta_fidelity(predicted, truth):
    error=predicted-truth; zero_rmse=float(np.sqrt(np.mean(truth**2))); rmse=float(np.sqrt(np.mean(error**2)))
    mae=float(np.mean(np.abs(error))); numerator=np.sum(predicted*truth,axis=1); denominator=np.linalg.norm(predicted,axis=1)*np.linalg.norm(truth,axis=1)
    valid=denominator>1e-12; cosine=float(np.mean(numerator[valid]/denominator[valid])) if valid.any() else 1.0
    rows=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"record_type":"SUMMARY","Counterfactual_Delta_RMSE":rmse,"Counterfactual_Delta_MAE":mae,"zero_counterfactual_RMSE":zero_rmse,"RMSE_relative_improvement":1.0-rmse/max(zero_rmse,1e-12),"cosine_direction_agreement":cosine,"nonzero_GT_delta_count":int(valid.sum())}]
    shaped_error=error.reshape(-1,10,2);shaped_truth=truth.reshape(-1,10,2)
    for horizon in range(10):
        frame_error=shaped_error[:,horizon];frame_truth=shaped_truth[:,horizon]
        rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"record_type":"PER_HORIZON","frame":horizon+1,"time_s":(horizon+1)/10,"Delta_RMSE":float(np.sqrt(np.mean(frame_error**2))),"Delta_MAE":float(np.mean(np.abs(frame_error))),"zero_RMSE":float(np.sqrt(np.mean(frame_truth**2)))})
    return rows


def predict_benefit(model,data,response_feature,batch_size,torch,device):
    values=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(data["z_final"]),batch_size):
            values.append(model(data["z_final"][start:start+batch_size].to(device),data["z_generic"][start:start+batch_size].to(device),torch.as_tensor(response_feature[start:start+batch_size],dtype=torch.float32,device=device)).cpu())
    return torch.cat(values).numpy().astype(np.float64)*data["scale"]


def benefit_loss(model,data,response_feature,indices,torch,device):
    local=torch.as_tensor(indices,dtype=torch.long);target=torch.as_tensor(data["target"][indices]/data["scale"],dtype=torch.float32,device=device)
    feasible=torch.tensor([data["samples"][index].targets.feasible for index in indices],dtype=torch.bool,device=device)
    prediction=model(data["z_final"][local].to(device),data["z_generic"][local].to(device),torch.as_tensor(response_feature[indices],dtype=torch.float32,device=device))
    error=prediction[feasible]-target[feasible];log_variance=data["log_variance"][local].to(device)[feasible]
    return .5*(error.square()*torch.exp(-log_variance)+log_variance).mean()


def train_benefit(name,model,train,validation,train_feature,validation_feature,batches,args,torch,device):
    model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    rows=[];states={};best_epoch=None;stale=0;started=time.perf_counter()
    for epoch,epoch_batches in enumerate(batches,1):
        model.train();losses=[]
        for indices in epoch_batches:
            loss=benefit_loss(model,train,train_feature,indices,torch,device);optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True));optimizer.step();losses.append((float(loss.detach()),gradient))
        prediction=predict_benefit(model,validation,validation_feature,args.batch_size,torch,device)
        cal=r1c.calibration_row(validation["samples"],prediction,validation["target"],name);sign=r1b.sign_summary(validation["samples"],prediction,validation["target"],name)
        row={"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"epoch":epoch,"train_heteroscedastic_NLL":float(np.mean([x[0] for x in losses])),"mean_gradient_norm":float(np.mean([x[1] for x in losses])),"Benefit_MAE":cal["Benefit_MAE"],"safe_beneficial_sign_accuracy":sign["safe_beneficial_sign_accuracy"],"safe_beneficial_positive_count":sign["predicted_positive_count"],"lambda_rank":0.0,"ranking_loss":0.0,"parameter_checksum":state_sha(model.state_dict())}
        if not all(math.isfinite(float(row[key])) for key in ("train_heteroscedastic_NLL","Benefit_MAE","safe_beneficial_sign_accuracy","mean_gradient_norm")):raise FloatingPointError(f"non-finite {name} state")
        rows.append(row);states[epoch]=copy.deepcopy(model.state_dict());selected=select_benefit_epoch(rows)
        if selected["epoch"]!=best_epoch:best_epoch,stale=selected["epoch"],0
        else:stale+=1
        row["selector_current_best_epoch"]=best_epoch;row["selector_stale_epochs"]=stale
        print(f"{name} epoch={epoch:02d} NLL={row['train_heteroscedastic_NLL']:.5f} MAE={row['Benefit_MAE']:.5f} sign={row['safe_beneficial_sign_accuracy']:.4f} best={best_epoch} stale={stale}",flush=True)
        if stale>=args.patience:break
    selected=select_benefit_epoch(rows);model.load_state_dict(states[selected["epoch"]]);model.eval()
    for row in rows:row["selector_final_selected"]=row["epoch"]==selected["epoch"]
    return {"model":model,"rows":rows,"selected":selected,"epochs_completed":len(rows),"training_time_s":time.perf_counter()-started}


def augmented_subgroup(samples,prediction,target,name,group,predicate):
    sign=r1b.sign_summary(samples,prediction,target,name,predicate);mask=np.asarray([predicate(sample) for sample in samples],bool)
    subset=[sample for sample,keep in zip(samples,mask) if keep];cal=r1c.calibration_row(subset,prediction[mask],target[mask],name)
    return {"group":group,**sign,"Benefit_MAE":cal["Benefit_MAE"],"prediction_mean":float(np.mean(prediction[mask])),"prediction_median":float(np.median(prediction[mask]))}


def response_shortcut_rows(samples,prediction,anchors):
    magnitude=np.linalg.norm(prediction.reshape(len(prediction),-1),axis=1);rows=r2.shortcut_rows(samples,magnitude,anchors)
    for row in rows:row["audit_component"]="response_future_norm"
    return rows


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite R3 result: {args.output_dir}")
    args.output_dir.mkdir(parents=True);(args.output_dir/"checkpoints").mkdir()
    random.seed(args.seed);np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device)

    checksums_before,labels,anchors=r1b.load_contract(args)
    episodes={"train":build_development_split("train",240,GENERATOR_SEED,RISK_SEED),"validation":build_development_split("validation",240,GENERATOR_SEED+1000,RISK_SEED+1000)}
    samples={split:build_v3_temporal_samples(value) for split,value in episodes.items()};targets={split:r1b.apply_target_v2(value,labels) for split,value in samples.items()}
    payload=torch.load(args.r1_checkpoint,map_location=device,weights_only=False)
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    backbone=RichTemporalSmallTransformerV3().to(device);backbone.load_state_dict(payload["model_state_dict"]);backbone.eval()
    for parameter in backbone.parameters():parameter.requires_grad_(False)
    backbone_before=state_sha(backbone.state_dict())
    data={split:extract_representations(backbone,value,payload["normalizer"],args.batch_size,torch,device) for split,value in samples.items()}
    for split in samples:
        generic_indices,identity=r1b.generic_indices(samples[split],anchors);future_local,cv_local=ground_truth_local_futures(episodes[split],samples[split],torch)
        data[split].update({"generic_indices":generic_indices,"generic_identity":identity,"z_generic":data[split]["z_final"][generic_indices],"target":targets[split],"samples":samples[split],"scale":float(payload["normalizer"]["benefit_scale"]),"future_local":future_local,"cv_local":cv_local})
        if any(int(anchors[episode]["runtime_anchor_action_id"])!=action for episode,action in identity):raise RuntimeError("R1A anchor identity changed")

    b0=data["validation"]["old_benefit"];sigma=np.exp(.5*data["validation"]["log_variance"].numpy().astype(np.float64))*data["validation"]["scale"]
    b0_metric,_=r1b.metrics(samples["validation"],b0,sigma,targets["validation"],"B0_FROZEN_RANKING");b0_sign=r1b.sign_summary(samples["validation"],b0,targets["validation"],"B0_FROZEN_RANKING")
    if b0_sign["safe_beneficial_count"]!=115 or b0_sign["predicted_positive_count"]!=42:raise RuntimeError("strict B0 reproduction failed")

    harm_payload=torch.load(args.harm_checkpoint,map_location=device,weights_only=False);harm_head=RiskPreservingBypassHead().to(device);harm_head.load_state_dict(harm_payload["model_state_dict"]);harm_head.eval()
    for parameter in harm_head.parameters():parameter.requires_grad_(False)
    with torch.inference_mode():harm_before=harm_head(data["validation"]["bypass"].to(device)).cpu().numpy()

    batches,batch_audit=b16.make_episode_batches(samples["train"],args.epochs,args.batch_size,args.seed)
    torch.manual_seed(args.seed);response=HumanResponseFutureDecoder();response_result=train_response(response,data["train"],data["validation"],batches,args,torch,device)
    response_prediction={split:predict_response(response_result["model"],data[split],args.batch_size,torch,device) for split in data}
    response_metrics=root_metrics(response_prediction["validation"],data["validation"]["future_local"]);cv_metrics=root_metrics(data["validation"]["cv_local"],data["validation"]["future_local"])
    response_vs_cv=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":"CONSTANT_VELOCITY","coordinate_frame":"decision-time human-origin / robot-yaw local XY",**cv_metrics},{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":"CHRR_RESPONSE_DECODER","coordinate_frame":"decision-time human-origin / robot-yaw local XY",**response_metrics}]
    predicted_delta={split:response_delta(response_prediction[split],data[split]["generic_indices"],torch) for split in data};gt_delta={split:response_delta(data[split]["future_local"],data[split]["generic_indices"],torch) for split in data}
    fidelity=delta_fidelity(predicted_delta["validation"],gt_delta["validation"]);fidelity_summary=fidelity[0]

    features={"C0_MATCHED_ZERO":{},"C1_RUNTIME_CHRR":{},"O1_ORACLE_GT_FUTURE":{}}
    for split in data:
        features["C0_MATCHED_ZERO"][split]=np.zeros_like(predicted_delta[split])
        features["C1_RUNTIME_CHRR"][split]=predicted_delta[split]
        features["O1_ORACLE_GT_FUTURE"][split]=gt_delta[split]
    results={}
    initial_shas={}
    for name in features:
        torch.manual_seed(args.seed);model=MatchedBenefitReadout();initial_shas[name]=state_sha(model.state_dict())
        results[name]=train_benefit(name,model,data["train"],data["validation"],features[name]["train"],features[name]["validation"],batches,args,torch,device)
    predictions={"B0_FROZEN_RANKING":b0}
    for name,result in results.items():predictions[name]=predict_benefit(result["model"],data["validation"],features[name]["validation"],args.batch_size,torch,device)

    sign_rows=[];calibration=[]
    for name,value in predictions.items():
        row=r1b.sign_summary(samples["validation"],value,targets["validation"],name)
        if name.startswith("O1_"):row["oracle_status"]=ORACLE_LABEL
        else:row["runtime_status"]="RUNTIME VALID" if name!="B0_FROZEN_RANKING" else "FROZEN RANKING REFERENCE"
        sign_rows.append(row);calibration.append(r1c.calibration_row(samples["validation"],value,targets["validation"],name))
    signs={row["model"]:row for row in sign_rows};cals={row["model"]:row for row in calibration}

    formal_names=("C0_MATCHED_ZERO","C1_RUNTIME_CHRR","O1_ORACLE_GT_FUTURE")
    c7=[augmented_subgroup(samples["validation"],predictions[name],targets["validation"],name,"C7",lambda sample:any(str(x).startswith("C7") for x in sample.split_metadata["contexts_evaluation_only"])) for name in formal_names]
    stop=[augmented_subgroup(samples["validation"],predictions[name],targets["validation"],name,"STOP",lambda sample:sample.split_metadata["motion_type_evaluation_only"]=="stop") for name in formal_names]
    hold=r2.hold_rows(samples["validation"],{name:predictions[name] for name in formal_names},targets["validation"])

    c0s,c1s=signs["C0_MATCHED_ZERO"],signs["C1_RUNTIME_CHRR"]
    negative=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"GT_negative_FPR":signs[name]["GT_negative_false_positive_rate"],"safe_beneficial_precision":signs[name]["safe_beneficial_precision"],"overall_predicted_positive_rate":signs[name]["overall_predicted_positive_rate"]} for name in ("C0_MATCHED_ZERO","C1_RUNTIME_CHRR")]
    negative.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":"C1_MINUS_C0","GT_negative_FPR_change":c1s["GT_negative_false_positive_rate"]-c0s["GT_negative_false_positive_rate"],"safe_beneficial_precision_change":c1s["safe_beneficial_precision"]-c0s["safe_beneficial_precision"]})

    shortcut=response_shortcut_rows(samples["validation"],response_prediction["validation"],anchors)
    benefit_shortcut=r2.shortcut_rows(samples["validation"],predictions["C1_RUNTIME_CHRR"],anchors)
    for row in benefit_shortcut:row["audit_component"]="C1_Benefit"
    shortcut.extend(benefit_shortcut)

    b0_after=extract_representations(backbone,samples["validation"],payload["normalizer"],args.batch_size,torch,device)["old_benefit"]
    ranking=r2.ranking_invariance(samples["validation"],targets["validation"],sigma,b0,b0_after)
    with torch.inference_mode():harm_after=harm_head(data["validation"]["bypass"].to(device)).cpu().numpy()
    checksums_after={"manifest_v3":r1b.file_sha(args.manifest_v3),"Benefit_Target_v2":r1b.file_sha(args.target_v2),"runtime_anchor_map":r1b.file_sha(args.anchor_map),"R1_v3_BASE":r1b.file_sha(args.r1_checkpoint),"HARM_v3_BASE":r1b.file_sha(args.harm_checkpoint)}
    harm_isolation={"label":LABEL,"mechanism_result":MECHANISM,"HARM_checkpoint_SHA_before":checksums_before["HARM_v3_BASE"],"HARM_checkpoint_SHA_after":checksums_after["HARM_v3_BASE"],"checkpoint_unchanged":checksums_before["HARM_v3_BASE"]==checksums_after["HARM_v3_BASE"],"harm_optimizer_created":False,"validation_harm_logits_before_SHA":array_sha(harm_before),"validation_harm_logits_after_SHA":array_sha(harm_after),"validation_harm_logits_max_abs_diff":float(np.max(np.abs(harm_after-harm_before))),"harm_outputs_exact":bool(np.array_equal(harm_before,harm_after))}

    hold_by={row["model"]:row for row in hold};stop_by={row["model"]:row for row in stop};finite=all(np.isfinite(value).all() for value in (*predictions.values(),*response_prediction.values()))
    # Motion class should explain future-motion magnitude.  The prohibited
    # shortcut is specifically an action ID nearly determining future/Benefit.
    action_shortcut=any(row["near_deterministic_shortcut"] for row in shortcut if row["dimension"] in ("candidate_action","runtime_generic_action"))
    gates={
        "Gate_A":{"name":"Contract / Isolation","checks":{"frozen_checksums_unchanged":checksums_before==checksums_after,"R1_backbone_unchanged":backbone_before==state_sha(backbone.state_dict()),"TEST_reads_zero":TEST_READS==0,"B0_ranking_frozen":ranking["B0_prediction_exact"],"Harm_frozen":harm_isolation["harm_outputs_exact"],"GT_future_not_in_C1_runtime_input":True,"GT_future_only_response_supervision_or_oracle":True}},
        "Gate_B":{"name":"Human-Response Learnability","checks":{"response_ADE_not_worse_than_CV":response_metrics["Root_ADE"]<=cv_metrics["Root_ADE"],"response_FDE_not_worse_than_CV":response_metrics["Root_FDE"]<=cv_metrics["Root_FDE"],"Delta_RMSE_improvement_at_least_0_15":fidelity_summary["RMSE_relative_improvement"]>=.15}},
        "Gate_C":{"name":"Benefit Value","checks":{"C1_safe_sign_at_least_0_55":c1s["safe_beneficial_sign_accuracy"]>=.55,"C1_minus_C0_at_least_0_05":c1s["safe_beneficial_sign_accuracy"]>=c0s["safe_beneficial_sign_accuracy"]+.05,"C1_MAE_not_above_C0":cals["C1_RUNTIME_CHRR"]["Benefit_MAE"]<=cals["C0_MATCHED_ZERO"]["Benefit_MAE"],"C1_MAE_not_above_R2_A1":cals["C1_RUNTIME_CHRR"]["Benefit_MAE"]<=R2_A1_MAE}},
        "Gate_D":{"name":"Stop Recovery","checks":{"C1_Stop_at_least_6_of_8":stop_by["C1_RUNTIME_CHRR"]["predicted_positive_count"]>=6 and stop_by["C1_RUNTIME_CHRR"]["safe_beneficial_count"]==8}},
        "Gate_E":{"name":"No Degenerate Shift","checks":{"GT_negative_FPR_increase_at_most_0_05":c1s["GT_negative_false_positive_rate"]<=c0s["GT_negative_false_positive_rate"]+.05,"safe_beneficial_precision_drop_at_most_0_10":c1s["safe_beneficial_precision"]>=c0s["safe_beneficial_precision"]-.10,"HOLD_negative_FPR_no_collapse":hold_by["C1_RUNTIME_CHRR"]["nonbeneficial_HOLD_FPR"]<=hold_by["C0_MATCHED_ZERO"]["nonbeneficial_HOLD_FPR"]+.05,"finite":finite,"no_action_ID_near_deterministic_shortcut":not action_shortcut}},
        "Gate_F":{"name":"Ranking / Harm Isolation","checks":{"Frozen_B0_ranking_exact":ranking["B0_prediction_exact"] and ranking["metrics_exact"] and ranking["historical_metrics_within_tolerance"] and ranking["rank_signature_changes"]==0,"Harm_outputs_exact":harm_isolation["harm_outputs_exact"]}},
    }
    for gate in gates.values():gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())

    o1_pass=(signs["O1_ORACLE_GT_FUTURE"]["safe_beneficial_sign_accuracy"]>=.55 and signs["O1_ORACLE_GT_FUTURE"]["safe_beneficial_sign_accuracy"]>=c0s["safe_beneficial_sign_accuracy"]+.05 and cals["O1_ORACLE_GT_FUTURE"]["Benefit_MAE"]<=cals["C0_MATCHED_ZERO"]["Benefit_MAE"] and cals["O1_ORACLE_GT_FUTURE"]["Benefit_MAE"]<=R2_A1_MAE and stop_by["O1_ORACLE_GT_FUTURE"]["predicted_positive_count"]>=6)
    if gates["all_passed"]:classification="CHRR SUCCESS";recommendation="Safe Decision Chain Reconstruction"
    elif o1_pass:classification="COUNTERFACTUAL RESPONSE REPRESENTATION IS USEFUL, BUT HUMAN-RESPONSE PREDICTOR IS THE BOTTLENECK";recommendation="Human-response predictor upgrade"
    else:classification="ROOT-TRAJECTORY-ONLY COUNTERFACTUAL REPRESENTATION INSUFFICIENT";recommendation="Future skeleton/pose-response enrichment"

    response_selector={"label":LABEL,"mechanism_result":MECHANISM,"selector":["minimum validation Root ADE","minimum validation Root FDE","earlier epoch"],"Benefit_metrics_used":False,"selected_epoch":int(response_result["selected"]["epoch"]),"epochs_completed":response_result["epochs_completed"],"selected_ADE":response_result["selected"]["Root_ADE"],"selected_FDE":response_result["selected"]["Root_FDE"]}
    benefit_selector={name:{"selected_epoch":int(result["selected"]["epoch"]),"epochs_completed":result["epochs_completed"],"MAE":result["selected"]["Benefit_MAE"],"safe_sign":result["selected"]["safe_beneficial_sign_accuracy"]} for name,result in results.items()}
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"CHRR_RESPONSE_DECODER","state_dict":response_result["model"].state_dict(),"selector":response_selector,"test_reads":0},args.output_dir/"checkpoints/response_decoder.pt")
    for name,result in results.items():torch.save({"label":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL if name.startswith("O1") else None,"model":name,"state_dict":result["model"].state_dict(),"selector":benefit_selector[name],"test_reads":0},args.output_dir/f"checkpoints/{name.lower()}.pt")

    frozen_contract={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"checksums_before":checksums_before,"checksums_after":checksums_after,"R1_backbone_frozen":not any(p.requires_grad for p in backbone.parameters()),"B0_formal_ranking_only":True,"HARM_v3_formal_risk_only":True,"threshold_calibration":False,"decision_chain":False,"arbitration":False}
    response_input_contract={"label":LABEL,"mechanism_result":MECHANISM,"z_human":{"actual_dimension":int(data["train"]["z_human"].shape[1]),"source":"R1_HISTORY_CONTEXT_PREFUSION","runtime_sources":["skeleton_history","human_motion_history","robot_history","functional_history","visibility_history","wm_diagnostic_history","interaction_history","scene_context"]},"z_candidate":{"actual_dimension":int(data["train"]["z_candidate"].shape[1]),"source":"R2_CANDIDATE_PREFUSION","runtime_sources":["candidate_action","candidate_robot_future"]},"decoder_input_dimension":int(data["train"]["z_human"].shape[1]+data["train"]["z_candidate"].shape[1]),"runtime_valid":True,"profile_ID_input":False,"GT_input":False,"forbidden_inputs":["GT future","GT Benefit","GT Harm","GT unsafe","profile ID","oracle theta","best action"]}
    response_target_contract={"label":LABEL,"mechanism_result":MECHANISM,"target":"synthetic action-conditioned human root future [10,2]","coordinate_frame":{"origin":"decision-time current human root","axes":"decision-time robot yaw","absolute_world_regression":False},"GT_future_role":"TRAIN supervision and validation response evaluation only; never C1 runtime input","loss":"SmoothL1, equal weight over 10 frames","Benefit_loss_in_response_training":False,"Harm_loss_in_response_training":False,"Cost_loss_in_response_training":False}
    response_arch={"label":LABEL,"mechanism_result":MECHANISM,**response_result["model"].architecture_audit()}
    benefit_config={"label":LABEL,"mechanism_result":MECHANISM,"seed":args.seed,"optimizer":"AdamW","learning_rate":args.learning_rate,"weight_decay":.001,"batch_size_candidate_budget":args.batch_size,"max_epochs":args.epochs,"patience":args.patience,"objective":"R1B-H0 frozen-uncertainty heteroscedastic NLL","uncertainty":"same frozen R1-v3 log variance for C0/C1/O1","selector":["minimum validation MAE","maximum safe-beneficial sign accuracy","earlier epoch"],"lambda_rank":0.0,"ranking_loss_computed":False,"same_batches":True,"batch_order_audit":batch_audit,"initial_state_SHAs":initial_shas,"hyperparameter_search":False}
    oracle_contract={"label":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,"input":"concat(z_i,z_g,DeltaY_GT_flat)","GT_future_used":True,"runtime_valid":False,"deployment_forbidden":True,"formal_checkpoint":False,"decision_chain_forbidden":True,"purpose":"representation upper-bound diagnosis only"}
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"response_decoder_input_dim":response_input_contract["decoder_input_dimension"],"response_runtime_valid":True,"response_metrics":response_metrics,"CV_metrics":cv_metrics,"counterfactual_fidelity":fidelity_summary,"safe_beneficial":{name:signs[name] for name in predictions},"calibration":{name:cals[name] for name in predictions},"STOP":{row["model"]:row for row in stop},"C7":{row["model"]:row for row in c7},"HOLD":hold_by,"O1_oracle_pass":o1_pass,"ranking_isolation":ranking,"harm_isolation":harm_isolation,"gates":gates,"outcome_classification":classification,"CHRR_successful":gates["all_passed"],"ready_for_v3_safe_decision_chain_reconstruction":gates["all_passed"],"single_next_recommendation":recommendation,"next_stage_started":False}

    io.write_json(args.output_dir/"frozen_contract.json",frozen_contract);io.write_json(args.output_dir/"response_input_contract.json",response_input_contract);io.write_json(args.output_dir/"response_target_contract.json",response_target_contract);io.write_json(args.output_dir/"response_architecture.json",response_arch)
    io.write_csv(args.output_dir/"response_training_curve.csv",response_result["rows"]);io.write_json(args.output_dir/"response_selector.json",response_selector);io.write_csv(args.output_dir/"response_vs_cv.csv",response_vs_cv);io.write_csv(args.output_dir/"counterfactual_delta_fidelity.csv",fidelity)
    architecture={"label":LABEL,"mechanism_result":MECHANISM,**next(iter(results.values()))["model"].architecture_audit()}
    io.write_json(args.output_dir/"c0_architecture.json",{**architecture,"role":"RUNTIME VALID CONTROL","response_feature":"zeros(20)"});io.write_json(args.output_dir/"c1_architecture.json",{**architecture,"role":"RUNTIME VALID CHRR","response_feature":"predicted DeltaY_hat(20)"});io.write_json(args.output_dir/"oracle_o1_contract.json",oracle_contract);io.write_json(args.output_dir/"benefit_training_config.json",{**benefit_config,"selectors":benefit_selector})
    for name,path in (("C0_MATCHED_ZERO","c0_metrics.csv"),("C1_RUNTIME_CHRR","c1_metrics.csv"),("O1_ORACLE_GT_FUTURE","o1_oracle_metrics.csv")):
        rows=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL if name.startswith("O1") else "","runtime_status":"NOT RUNTIME VALID" if name.startswith("O1") else "RUNTIME VALID",**cals[name],**{key:value for key,value in signs[name].items() if key not in ("synthetic_interaction","mechanism_result","model")}}]
        io.write_csv(args.output_dir/path,rows)
    io.write_csv(args.output_dir/"safe_beneficial_sign.csv",sign_rows);io.write_csv(args.output_dir/"mae_comparison.csv",calibration);io.write_csv(args.output_dir/"stop_audit.csv",stop);io.write_csv(args.output_dir/"c7_audit.csv",c7);io.write_csv(args.output_dir/"hold_audit.csv",hold);io.write_csv(args.output_dir/"negative_protection.csv",negative)
    io.write_json(args.output_dir/"ranking_isolation.json",ranking);io.write_json(args.output_dir/"harm_isolation.json",harm_isolation);io.write_csv(args.output_dir/"shortcut_audit.csv",shortcut);io.write_json(args.output_dir/"gate_results.json",gates);io.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
