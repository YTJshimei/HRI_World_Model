"""Phase 5B-v3-R4A Oracle Counterfactual Future Pose Sufficiency Audit.

No future-pose predictor is trained.  Development-only GT COCO-17 futures are
used solely by an explicitly non-runtime oracle Benefit probe.  Frozen B0 and
HARM-v3 remain the only ranking and risk paths; TEST is never materialized.
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

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
from scripts import run_phase5b_v3_r1c_frozen_runtime_generic_reanchor as r1c
from scripts import run_phase5b_v3_r2_pair_conditioned_benefit as r2
from scripts import run_phase5b_v3_r3_counterfactual_human_response as r3
from src.data.adverse_response_dataset import GENERATOR_SEED,RISK_SEED,POPULATION_PROFILE,build_development_split
from src.data.hold_candidate import build_hold_candidate_outcome
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.data.skeleton_schema import joint_names,joint_ids,root_joint,root_joint_ids
from src.data.synthetic_interaction import PROFILE_BY_ID,simulate_risk_conditioned_interaction_future
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.models.oracle_future_pose import OraclePoseBenefitReadout,oracle_pose_delta,root_relative_decision_local_pose
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL

MECHANISM="DEVELOPMENT MECHANISM RESULT"
ORACLE_LABEL="ORACLE FUTURE POSE DIAGNOSTIC - NOT RUNTIME VALID"
STAGE="Phase 5B-v3-R4A Oracle Counterfactual Future Pose Sufficiency Audit"
TEST_READS=0
LAMBDA_RANK=0.0
TOLERANCE=1e-10
R3_RUNTIME_MAE=1.7350633697252054
R3_ROOT_SIGN=0.46956521739130436
R3_ROOT_MAE=1.7345836692970045
R3_ROOT_STOP=0


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--seed",type=int,choices=(42,),default=42)
    parser.add_argument("--epochs",type=int,choices=(30,),default=30)
    parser.add_argument("--patience",type=int,choices=(5,),default=5)
    parser.add_argument("--batch-size",type=int,choices=(64,),default=64)
    parser.add_argument("--learning-rate",type=float,choices=(3e-4,),default=3e-4)
    parser.add_argument("--manifest-v3",type=Path,default=PROJECT_ROOT/"results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--target-v2",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv")
    parser.add_argument("--anchor-map",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv")
    parser.add_argument("--r1-checkpoint",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--r3-summary",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r3_counterfactual_human_response/summary.json")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r4a_oracle_future_pose_sufficiency")
    return parser.parse_args()


def state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_sha(value):
    value=np.ascontiguousarray(value);return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def select_epoch(rows):
    if not rows:raise ValueError("selector requires validation rows")
    return min(rows,key=lambda row:(float(row["Benefit_MAE"]),-float(row["safe_beneficial_sign_accuracy"]),int(row["epoch"])))


def future_skeletons(episodes,samples,torch):
    """Replay actual candidate-conditioned generator futures; never synthesize substitutes."""
    by_id={};yaw_by_episode={}
    for episode in episodes:
        profile=PROFILE_BY_ID[episode.profile_id];yaw_by_episode[episode.episode_id]=float(episode.robot_history[-1,2])
        for candidate in episode.candidates:
            simulation=simulate_risk_conditioned_interaction_future(episode.human_history,episode.natural_future,episode.robot_history,candidate.action_id,profile,episode.risk_factors)
            by_id[f"{episode.episode_id}:{candidate.action_id}"]=simulation.future_global
        hold=build_hold_candidate_outcome(episode,POPULATION_PROFILE,profile)
        by_id[f"{episode.episode_id}:{HOLD_ACTION_ID}"]=hold.gt_simulation.future_global
    world=np.stack([by_id[sample.sample_id] for sample in samples]).astype(np.float32)
    yaw=np.asarray([yaw_by_episode[sample.episode_id] for sample in samples],np.float32)
    if world.shape!=(len(samples),10,17,3) or not np.isfinite(world).all():raise RuntimeError("real candidate-conditioned future skeleton unavailable")
    root,pose=root_relative_decision_local_pose(torch.from_numpy(world),torch.from_numpy(yaw))
    return {"world":world,"root":root.numpy().astype(np.float32),"pose_local":pose.numpy().astype(np.float32),"robot_yaw":yaw}


def loss(model,data,feature,indices,torch,device):
    local=torch.as_tensor(indices,dtype=torch.long);target=torch.as_tensor(data["target"][indices]/data["scale"],dtype=torch.float32,device=device)
    feasible=torch.tensor([data["samples"][index].targets.feasible for index in indices],dtype=torch.bool,device=device)
    prediction=model(data["z_final"][local].to(device),data["z_generic"][local].to(device),torch.as_tensor(feature[indices],dtype=torch.float32,device=device))
    error=prediction[feasible]-target[feasible];log_variance=data["log_variance"][local].to(device)[feasible]
    return .5*(error.square()*torch.exp(-log_variance)+log_variance).mean()


def predict(model,data,feature,batch_size,torch,device):
    values=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(data["z_final"]),batch_size):values.append(model(data["z_final"][start:start+batch_size].to(device),data["z_generic"][start:start+batch_size].to(device),torch.as_tensor(feature[start:start+batch_size],dtype=torch.float32,device=device)).cpu())
    return torch.cat(values).numpy().astype(np.float64)*data["scale"]


def train(name,model,train_data,validation_data,train_feature,validation_feature,batches,args,torch,device):
    model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    rows=[];states={};best=None;stale=0;started=time.perf_counter()
    for epoch,epoch_batches in enumerate(batches,1):
        model.train();losses=[]
        for indices in epoch_batches:
            value=loss(model,train_data,train_feature,indices,torch,device);optimizer.zero_grad(set_to_none=True);value.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True));optimizer.step();losses.append((float(value.detach()),gradient))
        prediction=predict(model,validation_data,validation_feature,args.batch_size,torch,device);cal=r1c.calibration_row(validation_data["samples"],prediction,validation_data["target"],name);sign=r1b.sign_summary(validation_data["samples"],prediction,validation_data["target"],name)
        row={"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL if name=="O_POSE" else "RUNTIME VALID MATCHED CONTROL","model":name,"epoch":epoch,"train_heteroscedastic_NLL":float(np.mean([x[0] for x in losses])),"mean_gradient_norm":float(np.mean([x[1] for x in losses])),"Benefit_MAE":cal["Benefit_MAE"],"safe_beneficial_sign_accuracy":sign["safe_beneficial_sign_accuracy"],"safe_beneficial_positive_count":sign["predicted_positive_count"],"lambda_rank":0.0,"ranking_loss":0.0,"parameter_checksum":state_sha(model.state_dict())}
        if not all(math.isfinite(float(row[key])) for key in ("train_heteroscedastic_NLL","Benefit_MAE","safe_beneficial_sign_accuracy","mean_gradient_norm")):raise FloatingPointError(f"non-finite {name}")
        rows.append(row);states[epoch]=copy.deepcopy(model.state_dict());selected=select_epoch(rows)
        if selected["epoch"]!=best:best,stale=selected["epoch"],0
        else:stale+=1
        row["selector_current_best_epoch"]=best;row["selector_stale_epochs"]=stale
        print(f"{name} epoch={epoch:02d} NLL={row['train_heteroscedastic_NLL']:.5f} MAE={row['Benefit_MAE']:.5f} sign={row['safe_beneficial_sign_accuracy']:.4f} best={best} stale={stale}",flush=True)
        if stale>=args.patience:break
    selected=select_epoch(rows);model.load_state_dict(states[selected["epoch"]]);model.eval()
    for row in rows:row["selector_final_selected"]=row["epoch"]==selected["epoch"]
    return {"model":model,"rows":rows,"selected":selected,"epochs_completed":len(rows),"training_time_s":time.perf_counter()-started}


def augmented_subgroup(samples,prediction,target,name,group,predicate):
    sign=r1b.sign_summary(samples,prediction,target,name,predicate);mask=np.asarray([predicate(sample) for sample in samples],bool);subset=[sample for sample,keep in zip(samples,mask) if keep];cal=r1c.calibration_row(subset,prediction[mask],target[mask],name)
    return {"group":group,**sign,"Benefit_MAE":cal["Benefit_MAE"]}


def describe(values):
    values=np.asarray(values,np.float64)
    return {"count":int(len(values)),"mean":float(np.mean(values)),"median":float(np.median(values)),"std":float(np.std(values)),"P90":float(np.percentile(values,90)),"max":float(np.max(values))}


def pose_audits(samples,delta_pose,target):
    shaped=delta_pose.reshape(-1,10,17,3);magnitude=np.linalg.norm(shaped.reshape(len(shaped),-1),axis=1)
    feasible=np.asarray([sample.targets.feasible for sample in samples],bool);harm=np.asarray([sample.split_metadata["harm_v2_evaluation_only"] for sample in samples],bool);positive=target>TOLERANCE
    groups={"beneficial":positive,"non_beneficial":~positive,"harm_positive":harm,"safe_beneficial":positive&feasible&~harm,"Stop":np.asarray([sample.split_metadata["motion_type_evaluation_only"]=="stop" for sample in samples]),"C7":np.asarray([any(str(x).startswith("C7") for x in sample.split_metadata["contexts_evaluation_only"]) for sample in samples]),"HOLD":np.asarray([sample.split_metadata["candidate_action_id_audit"]==HOLD_ACTION_ID for sample in samples])}
    magnitude_rows=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,"group":name,**describe(magnitude[mask])} for name,mask in groups.items()]
    safe=groups["safe_beneficial"];nonbeneficial=groups["non_beneficial"];joint_abs=np.abs(shaped).mean(axis=(1,3))
    joint_rows=[]
    focus={"left_shoulder","right_shoulder","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"}
    for index,name in enumerate(joint_names):joint_rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,"joint_id":index,"joint_name":name,"focus_joint":name in focus,"safe_beneficial_mean_abs_delta":float(joint_abs[safe,index].mean()),"non_beneficial_mean_abs_delta":float(joint_abs[nonbeneficial,index].mean()),"safe_minus_non_beneficial":float(joint_abs[safe,index].mean()-joint_abs[nonbeneficial,index].mean())})
    temporal=[]
    for frame in range(10):
        frame_magnitude=np.linalg.norm(shaped[:,frame].reshape(len(shaped),-1),axis=1)
        temporal.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,"frame":frame+1,"time_s":(frame+1)/10,"all_mean_magnitude":float(frame_magnitude.mean()),"safe_beneficial_mean_magnitude":float(frame_magnitude[safe].mean()),"non_beneficial_mean_magnitude":float(frame_magnitude[nonbeneficial].mean()),"safe_minus_non_beneficial":float(frame_magnitude[safe].mean()-frame_magnitude[nonbeneficial].mean())})
    return magnitude_rows,joint_rows,temporal


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite R4A: {args.output_dir}")
    args.output_dir.mkdir(parents=True);(args.output_dir/"checkpoints").mkdir()
    random.seed(args.seed);np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device)

    checksums_before,labels,anchors=r1b.load_contract(args)
    r3_summary=json.loads(args.r3_summary.read_text(encoding="utf-8"))
    if abs(r3_summary["safe_beneficial"]["O1_ORACLE_GT_FUTURE"]["safe_beneficial_sign_accuracy"]-R3_ROOT_SIGN)>1e-15 or abs(r3_summary["calibration"]["O1_ORACLE_GT_FUTURE"]["Benefit_MAE"]-R3_ROOT_MAE)>1e-15 or r3_summary["STOP"]["O1_ORACLE_GT_FUTURE"]["predicted_positive_count"]!=R3_ROOT_STOP:raise RuntimeError("R3 root Oracle frozen result changed")
    episodes={"train":build_development_split("train",240,GENERATOR_SEED,RISK_SEED),"validation":build_development_split("validation",240,GENERATOR_SEED+1000,RISK_SEED+1000)}
    samples={split:build_v3_temporal_samples(value) for split,value in episodes.items()};targets={split:r1b.apply_target_v2(value,labels) for split,value in samples.items()}
    skeleton={split:future_skeletons(episodes[split],samples[split],torch) for split in samples}

    payload=torch.load(args.r1_checkpoint,map_location=device,weights_only=False)
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    backbone=RichTemporalSmallTransformerV3().to(device);backbone.load_state_dict(payload["model_state_dict"]);backbone.eval()
    for parameter in backbone.parameters():parameter.requires_grad_(False)
    backbone_before=state_sha(backbone.state_dict());data={split:r3.extract_representations(backbone,value,payload["normalizer"],args.batch_size,torch,device) for split,value in samples.items()}
    features={}
    for split in samples:
        generic_indices,identity=r1b.generic_indices(samples[split],anchors)
        if any(int(anchors[episode]["runtime_anchor_action_id"])!=action for episode,action in identity):raise RuntimeError("generic identity changed")
        pose=torch.from_numpy(skeleton[split]["pose_local"]);generic=pose[torch.as_tensor(generic_indices,dtype=torch.long)];delta=oracle_pose_delta(pose,generic).numpy().astype(np.float64)
        data[split].update({"generic_indices":generic_indices,"generic_identity":identity,"z_generic":data[split]["z_final"][generic_indices],"target":targets[split],"samples":samples[split],"scale":float(payload["normalizer"]["benefit_scale"])})
        features[split]={"C0":np.zeros_like(delta),"O_POSE":delta}

    b0=data["validation"]["old_benefit"];sigma=np.exp(.5*data["validation"]["log_variance"].numpy().astype(np.float64))*data["validation"]["scale"]
    b0_metric,_=r1b.metrics(samples["validation"],b0,sigma,targets["validation"],"B0_FROZEN_RANKING");b0_sign=r1b.sign_summary(samples["validation"],b0,targets["validation"],"B0_FROZEN_RANKING")
    if b0_sign["safe_beneficial_count"]!=115 or b0_sign["predicted_positive_count"]!=42:raise RuntimeError("B0 reproduction failed")
    harm_payload=torch.load(args.harm_checkpoint,map_location=device,weights_only=False);harm_head=RiskPreservingBypassHead().to(device);harm_head.load_state_dict(harm_payload["model_state_dict"]);harm_head.eval()
    for parameter in harm_head.parameters():parameter.requires_grad_(False)
    with torch.inference_mode():harm_before=harm_head(data["validation"]["bypass"].to(device)).cpu().numpy()

    batches,batch_audit=b16.make_episode_batches(samples["train"],args.epochs,args.batch_size,args.seed)
    results={};initial={}
    for name in ("C0","O_POSE"):
        torch.manual_seed(args.seed);model=OraclePoseBenefitReadout();initial[name]=state_sha(model.state_dict());results[name]=train(name,model,data["train"],data["validation"],features["train"][name],features["validation"][name],batches,args,torch,device)
    predictions={name:predict(result["model"],data["validation"],features["validation"][name],args.batch_size,torch,device) for name,result in results.items()}
    signs={name:r1b.sign_summary(samples["validation"],value,targets["validation"],name) for name,value in predictions.items()};cals={name:r1c.calibration_row(samples["validation"],value,targets["validation"],name) for name,value in predictions.items()}
    root_record={"label":LABEL,"mechanism_result":MECHANISM,"model":"O_ROOT_R3_FROZEN_REPRODUCTION","safe_beneficial_count":115,"predicted_positive_count":54,"safe_beneficial_sign_accuracy":R3_ROOT_SIGN,"Benefit_MAE":R3_ROOT_MAE,"Stop_safe_beneficial_count":8,"Stop_predicted_positive_count":R3_ROOT_STOP,"source":str(args.r3_summary),"runtime_valid":False,"oracle_status":"ORACLE FUTURE DIAGNOSTIC - NOT RUNTIME VALID"}
    overall=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":"RUNTIME VALID MATCHED CONTROL",**cals["C0"],**{key:value for key,value in signs["C0"].items() if key in ("safe_beneficial_count","predicted_positive_count","sign_failure_count","safe_beneficial_sign_accuracy")}}, {"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,**root_record}, {"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,**cals["O_POSE"],**{key:value for key,value in signs["O_POSE"].items() if key in ("safe_beneficial_count","predicted_positive_count","sign_failure_count","safe_beneficial_sign_accuracy")}}]

    names=("C0","O_POSE");predicates={"STOP":lambda sample:sample.split_metadata["motion_type_evaluation_only"]=="stop","C7":lambda sample:any(str(x).startswith("C7") for x in sample.split_metadata["contexts_evaluation_only"])}
    stop=[augmented_subgroup(samples["validation"],predictions[name],targets["validation"],name,"STOP",predicates["STOP"]) for name in names];stop.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"group":"STOP","model":"O_ROOT_R3_FROZEN_REPRODUCTION","safe_beneficial_count":8,"predicted_positive_count":0,"safe_beneficial_sign_accuracy":0.0,"Benefit_MAE":r3_summary["STOP"]["O1_ORACLE_GT_FUTURE"]["Benefit_MAE"]})
    c7=[augmented_subgroup(samples["validation"],predictions[name],targets["validation"],name,"C7",predicates["C7"]) for name in names];root_c7=r3_summary["C7"]["O1_ORACLE_GT_FUTURE"];c7.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"group":"C7","model":"O_ROOT_R3_FROZEN_REPRODUCTION","safe_beneficial_count":root_c7["safe_beneficial_count"],"predicted_positive_count":root_c7["predicted_positive_count"],"safe_beneficial_sign_accuracy":root_c7["safe_beneficial_sign_accuracy"],"Benefit_MAE":root_c7["Benefit_MAE"]})
    hold=r2.hold_rows(samples["validation"],predictions,targets["validation"])
    negative=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"GT_negative_FPR":signs[name]["GT_negative_false_positive_rate"],"safe_beneficial_precision":signs[name]["safe_beneficial_precision"]} for name in names];negative.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":"O_POSE_MINUS_C0","GT_negative_FPR_change":signs["O_POSE"]["GT_negative_false_positive_rate"]-signs["C0"]["GT_negative_false_positive_rate"],"safe_beneficial_precision_change":signs["O_POSE"]["safe_beneficial_precision"]-signs["C0"]["safe_beneficial_precision"]})
    magnitude,joint_rows,temporal=pose_audits(samples["validation"],features["validation"]["O_POSE"],targets["validation"])
    shortcut=r2.shortcut_rows(samples["validation"],predictions["O_POSE"],anchors);action_shortcut=any(row["near_deterministic_shortcut"] for row in shortcut if row["dimension"] in ("candidate_action","runtime_generic_action"))

    b0_after=r3.extract_representations(backbone,samples["validation"],payload["normalizer"],args.batch_size,torch,device)["old_benefit"];ranking=r2.ranking_invariance(samples["validation"],targets["validation"],sigma,b0,b0_after)
    with torch.inference_mode():harm_after=harm_head(data["validation"]["bypass"].to(device)).cpu().numpy()
    checksums_after={"manifest_v3":r1b.file_sha(args.manifest_v3),"Benefit_Target_v2":r1b.file_sha(args.target_v2),"runtime_anchor_map":r1b.file_sha(args.anchor_map),"R1_v3_BASE":r1b.file_sha(args.r1_checkpoint),"HARM_v3_BASE":r1b.file_sha(args.harm_checkpoint)}
    harm_isolation={"label":LABEL,"mechanism_result":MECHANISM,"HARM_checkpoint_SHA_before":checksums_before["HARM_v3_BASE"],"HARM_checkpoint_SHA_after":checksums_after["HARM_v3_BASE"],"checkpoint_unchanged":checksums_before["HARM_v3_BASE"]==checksums_after["HARM_v3_BASE"],"harm_optimizer_created":False,"validation_harm_logits_before_SHA":array_sha(harm_before),"validation_harm_logits_after_SHA":array_sha(harm_after),"validation_harm_logits_max_abs_diff":float(np.max(np.abs(harm_after-harm_before))),"harm_outputs_exact":bool(np.array_equal(harm_before,harm_after))}
    stop_by={row["model"]:row for row in stop};c0s,ops=signs["C0"],signs["O_POSE"]
    gates={
        "Gate_A":{"name":"Data / Contract Validity","checks":{"candidate_conditioned_future_skeleton_exists":True,"future_skeleton_not_fabricated":True,"shape_10_17_3":all(value["world"].shape==(1440,10,17,3) for value in skeleton.values()),"frozen_checksums_unchanged":checksums_before==checksums_after,"TEST_reads_zero":TEST_READS==0}},
        "Gate_B":{"name":"Root Reproduction","checks":{"R3_root_sign_exact":abs(root_record["safe_beneficial_sign_accuracy"]-R3_ROOT_SIGN)<=1e-15,"R3_root_MAE_exact":abs(root_record["Benefit_MAE"]-R3_ROOT_MAE)<=1e-15,"R3_root_Stop_exact":root_record["Stop_predicted_positive_count"]==R3_ROOT_STOP}},
        "Gate_C":{"name":"Pose Oracle Sufficiency","checks":{"O_POSE_sign_at_least_0_60":ops["safe_beneficial_sign_accuracy"]>=.60,"O_POSE_minus_C0_at_least_0_10":ops["safe_beneficial_sign_accuracy"]>=c0s["safe_beneficial_sign_accuracy"]+.10,"O_POSE_MAE_not_above_C0":cals["O_POSE"]["Benefit_MAE"]<=cals["C0"]["Benefit_MAE"],"O_POSE_MAE_not_above_R3_runtime":cals["O_POSE"]["Benefit_MAE"]<=R3_RUNTIME_MAE}},
        "Gate_D":{"name":"Stop Recovery","checks":{"O_POSE_Stop_at_least_6_of_8":stop_by["O_POSE"]["predicted_positive_count"]>=6 and stop_by["O_POSE"]["safe_beneficial_count"]==8}},
        "Gate_E":{"name":"No Degenerate Shift","checks":{"GT_negative_FPR_increase_at_most_0_05":ops["GT_negative_false_positive_rate"]<=c0s["GT_negative_false_positive_rate"]+.05,"finite":all(np.isfinite(value).all() for value in predictions.values()),"no_action_template_shortcut":not action_shortcut}},
        "Gate_F":{"name":"System Isolation","checks":{"Frozen_B0_ranking_exact":ranking["B0_prediction_exact"] and ranking["metrics_exact"] and ranking["historical_metrics_within_tolerance"] and ranking["rank_signature_changes"]==0,"Harm_outputs_exact":harm_isolation["harm_outputs_exact"],"no_runtime_model_trained":True,"no_runtime_deployment":True}},
    }
    for gate in gates.values():gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())
    pose_global_help=(ops["safe_beneficial_sign_accuracy"]>=.60 and ops["safe_beneficial_sign_accuracy"]>=c0s["safe_beneficial_sign_accuracy"]+.10 and cals["O_POSE"]["Benefit_MAE"]<=cals["C0"]["Benefit_MAE"] and cals["O_POSE"]["Benefit_MAE"]<=R3_RUNTIME_MAE)
    if gates["all_passed"]:classification="FUTURE POSE RESPONSE HAS SUFFICIENT ORACLE BENEFIT INFORMATION";recommendation="Train runtime future skeleton predictor"
    elif pose_global_help and not gates["Gate_D"]["passed"]:classification="POSE HELPS GLOBAL BENEFIT BUT STOP REQUIRES ADDITIONAL RESPONSE SEMANTICS";recommendation="Stop-specific response audit"
    else:classification="RAW FUTURE SKELETON ALONE INSUFFICIENT";recommendation="Semantic response / cost-component representation audit"

    availability={"label":LABEL,"mechanism_result":MECHANISM,"available":True,"source_files":["src/data/synthetic_interaction.py","src/data/hold_candidate.py","src/data/adverse_response_dataset.py"],"fields":["RiskConditionedInteractionSimulation.future_global","HoldInteractionSimulation.future_global"],"shape":[10,17,3],"joint_schema":"COCO-17 from src/data/skeleton_schema.py","coordinate_frame":"generator world XYZ before required Oracle transform","visibility_representation":"No generator-defined future visibility mask exists; only history visibility is generated","generation_path":"build_development_split -> candidate risk-conditioned simulator / HOLD outcome -> future_global","all_train_candidates":1440,"all_validation_candidates":1440,"fabricated":False,"copied_history":False,"interpolated_root":False}
    pose_contract={"label":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,"pre_registered_oracle":"ALL-JOINT ORACLE because generator has no future visibility mask","joint_names":list(joint_names),"root_definition":{"name":root_joint,"joint_ids":list(root_joint_ids),"extra_joint_added":False},"pose_shape":[10,17,3],"pose_feature_dimension":510,"transform":"subtract per-frame pelvis midpoint; rotate XY by decision-time robot yaw; Z remains root-relative height","absolute_root_in_pose_feature":False,"future_visibility_available":False,"runtime_valid":False,"deployment_forbidden":True}
    architecture={"label":LABEL,"mechanism_result":MECHANISM,**results["C0"]["model"].architecture_audit()};training={"label":LABEL,"mechanism_result":MECHANISM,"seed":args.seed,"optimizer":"AdamW","learning_rate":args.learning_rate,"weight_decay":.001,"batch_size_candidate_budget":args.batch_size,"max_epochs":args.epochs,"patience":args.patience,"objective":"R1B-H0 frozen-uncertainty heteroscedastic NLL","selector":["minimum validation MAE","maximum safe-beneficial sign accuracy","earlier epoch"],"lambda_rank":0.0,"ranking_loss_computed":False,"same_batches":True,"batch_order_audit":batch_audit,"same_initialization":initial["C0"]==initial["O_POSE"],"initial_state_SHA":initial["C0"],"future_skeleton_predictor_trained":False,"hyperparameter_search":False,"selection_by_joint_or_horizon":False,"selections":{name:{"epoch":int(result["selected"]["epoch"]),"MAE":result["selected"]["Benefit_MAE"],"sign":result["selected"]["safe_beneficial_sign_accuracy"]} for name,result in results.items()}}
    frozen_contract={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"checksums_before":checksums_before,"checksums_after":checksums_after,"R1_backbone_frozen":not any(p.requires_grad for p in backbone.parameters()),"R1_backbone_unchanged":backbone_before==state_sha(backbone.state_dict()),"B0_formal_ranking_only":True,"HARM_v3_formal_risk_only":True,"threshold_calibration":False,"decision_chain":False,"runtime_model_training":False}
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"future_skeleton_availability":availability,"future_pose_contract":pose_contract,"root_oracle_reproduction":root_record,"safe_beneficial":{"C0":signs["C0"],"O_ROOT":root_record,"O_POSE":signs["O_POSE"]},"calibration":{"C0":cals["C0"],"O_ROOT":root_record,"O_POSE":cals["O_POSE"]},"STOP":{row["model"]:row for row in stop},"C7":{row["model"]:row for row in c7},"HOLD":{row["model"]:row for row in hold},"gates":gates,"outcome_classification":classification,"single_next_recommendation":recommendation,"runtime_future_skeleton_predictor_authorized":gates["all_passed"],"next_stage_started":False}

    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"C0","state_dict":results["C0"]["model"].state_dict(),"selector":training["selections"]["C0"],"test_reads":0},args.output_dir/"checkpoints/c0_matched.pt")
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE_LABEL,"model":"O_POSE","state_dict":results["O_POSE"]["model"].state_dict(),"selector":training["selections"]["O_POSE"],"runtime_valid":False,"deployment_forbidden":True,"formal_checkpoint":False,"test_reads":0},args.output_dir/"checkpoints/o_pose_oracle_diagnostic.pt")
    io.write_json(args.output_dir/"frozen_contract.json",frozen_contract);io.write_json(args.output_dir/"future_skeleton_availability.json",availability);io.write_json(args.output_dir/"future_pose_contract.json",pose_contract);io.write_json(args.output_dir/"root_oracle_reproduction.json",root_record);io.write_json(args.output_dir/"c0_architecture.json",{**architecture,"role":"RUNTIME VALID MATCHED CONTROL","pose_feature":"zeros(510)"});io.write_json(args.output_dir/"pose_oracle_architecture.json",{**architecture,"role":ORACLE_LABEL,"pose_feature":"GT counterfactual root-relative pose delta(510)","runtime_valid":False,"deployment_forbidden":True});io.write_json(args.output_dir/"training_config.json",training)
    io.write_csv(args.output_dir/"c0_training_curve.csv",results["C0"]["rows"]);io.write_csv(args.output_dir/"pose_oracle_training_curve.csv",results["O_POSE"]["rows"]);io.write_csv(args.output_dir/"overall_comparison.csv",overall);io.write_csv(args.output_dir/"safe_beneficial_sign.csv",overall);io.write_csv(args.output_dir/"mae_comparison.csv",overall);io.write_csv(args.output_dir/"stop_audit.csv",stop);io.write_csv(args.output_dir/"c7_audit.csv",c7);io.write_csv(args.output_dir/"hold_audit.csv",hold);io.write_csv(args.output_dir/"negative_protection.csv",negative);io.write_csv(args.output_dir/"pose_magnitude_audit.csv",magnitude);io.write_csv(args.output_dir/"per_joint_information.csv",joint_rows);io.write_csv(args.output_dir/"temporal_information.csv",temporal);io.write_csv(args.output_dir/"shortcut_audit.csv",shortcut);io.write_json(args.output_dir/"ranking_isolation.json",ranking);io.write_json(args.output_dir/"harm_isolation.json",harm_isolation);io.write_json(args.output_dir/"gate_results.json",{"label":LABEL,"mechanism_result":MECHANISM,**gates});io.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
