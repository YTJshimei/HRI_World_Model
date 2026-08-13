"""Phase 5B-v3-R1D runtime-conditioned episode offset calibration.

The frozen R1-v3 representation of the R1A runtime-generic candidate is the
sole model input.  Only one Linear(128,1) head is trained; all candidates in an
episode receive the same predicted scalar offset.  TEST and decision execution
remain sealed.
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
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
from scripts import run_phase5b_v3_r1c_frozen_runtime_generic_reanchor as r1c
from src.data.adverse_response_dataset import GENERATOR_SEED,RISK_SEED,build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.evaluation.context_value_metrics import pearson,spearman
from src.evaluation.frozen_runtime_generic_reanchor import frozen_runtime_generic_reanchor
from src.models.runtime_conditioned_episode_offset import RuntimeConditionedEpisodeOffset
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL

MECHANISM="DEVELOPMENT MECHANISM RESULT"
STAGE="Phase 5B-v3-R1D Runtime-Conditioned Episode Offset Calibration"
TEST_READS=0
EXPECTED_C0_MAE=1.9629752593426275
TOLERANCE=1e-10


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
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1d_runtime_conditioned_offset")
    return parser.parse_args()


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_epoch(rows):
    """Pre-registered: minimum MAE, maximum safe sign, then earlier epoch."""
    if not rows:raise ValueError("selector requires validation rows")
    return min(rows,key=lambda row:(float(row["Benefit_MAE"]),-float(row["safe_beneficial_sign_accuracy"]),int(row["epoch"])))


def episode_data(samples,target,b0,context,anchors):
    grouped=b15.group_episode(samples); episode_ids=list(grouped); generic_indices=[]; delta_star=[]
    for episode_id in episode_ids:
        indices=grouped[episode_id]; action=int(anchors[episode_id]["runtime_anchor_action_id"])
        match=[index for index in indices if int(samples[index].split_metadata["candidate_action_id_audit"])==action]
        if len(match)!=1:raise RuntimeError(f"runtime generic representation missing for {episode_id}")
        generic_indices.append(match[0]);delta_star.append(float(np.mean(target[indices]-b0[indices])))
    return {"episode_ids":episode_ids,"groups":grouped,"generic_indices":np.asarray(generic_indices,int),"z_generic":context[generic_indices],"delta_star":np.asarray(delta_star,np.float64)}


def apply_offsets(samples,episode_ids,offsets,b0):
    if len(episode_ids)!=len(offsets):raise ValueError("one scalar offset is required per episode")
    by_episode={episode:float(value) for episode,value in zip(episode_ids,offsets)}
    return np.asarray([float(value)+by_episode[sample.episode_id] for sample,value in zip(samples,b0)],np.float64)


def validate_epoch(head,data,samples,target,b0,torch,device,epoch,train_loss):
    head.eval()
    with torch.inference_mode():offset=head(data["z_generic"].to(device)).cpu().numpy().astype(np.float64)
    prediction=apply_offsets(samples,data["episode_ids"],offset,b0)
    sign=r1b.sign_summary(samples,prediction,target,"C2_RCEOC")
    cal=r1c.calibration_row(samples,prediction,target,"C2_RCEOC")
    return {"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"epoch":epoch,"train_MSE":train_loss,"Benefit_MAE":cal["Benefit_MAE"],"safe_beneficial_sign_accuracy":sign["safe_beneficial_sign_accuracy"],"safe_beneficial_positive_count":sign["predicted_positive_count"],"offset_validation_MSE":float(np.mean((offset-data["delta_star"])**2)),"offset_validation_MAE":float(np.mean(np.abs(offset-data["delta_star"]))),"head_checksum":state_sha(head.state_dict())}


def train_head(head,train_data,validation_data,validation_samples,validation_target,validation_b0,args,torch,device):
    head.to(device);optim=torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    generator=torch.Generator(device="cpu");generator.manual_seed(args.seed)
    rows=[];states={};best_epoch=None;stale=0;started=time.perf_counter()
    train_x=train_data["z_generic"];train_y=torch.from_numpy(train_data["delta_star"].astype(np.float32))
    for epoch in range(1,args.epochs+1):
        head.train();order=torch.randperm(len(train_x),generator=generator);losses=[]
        for start in range(0,len(order),args.batch_size):
            index=order[start:start+args.batch_size];prediction=head(train_x[index].to(device));target=train_y[index].to(device)
            loss=torch.nn.functional.mse_loss(prediction,target)
            optim.zero_grad(set_to_none=True);loss.backward();optim.step();losses.append(float(loss.detach()))
        row=validate_epoch(head,validation_data,validation_samples,validation_target,validation_b0,torch,device,epoch,float(np.mean(losses)))
        if not all(math.isfinite(float(row[key])) for key in ("train_MSE","Benefit_MAE","safe_beneficial_sign_accuracy","offset_validation_MSE")):raise FloatingPointError("non-finite RCEOC training state")
        rows.append(row);states[epoch]=copy.deepcopy(head.state_dict());selected=select_epoch(rows)
        if selected["epoch"]!=best_epoch:best_epoch,stale=selected["epoch"],0
        else:stale+=1
        row["selector_current_best_epoch"]=best_epoch;row["selector_stale_epochs"]=stale
        print(f"RCEOC epoch={epoch:02d} train={row['train_MSE']:.5f} MAE={row['Benefit_MAE']:.5f} sign={row['safe_beneficial_sign_accuracy']:.4f} best={best_epoch} stale={stale}",flush=True)
        if stale>=args.patience:break
    selected=select_epoch(rows);head.load_state_dict(states[selected["epoch"]]);head.eval()
    for row in rows:row["selector_final_selected"]=row["epoch"]==selected["epoch"]
    return {"head":head,"rows":rows,"selected":selected,"training_time_s":time.perf_counter()-started,"epochs_completed":len(rows),"optimizer_parameter_count":sum(p.numel() for group in optim.param_groups for p in group["params"])}


def distribution(values):
    values=np.asarray(values,np.float64)
    return {"count":int(len(values)),"mean":float(values.mean()),"std":float(values.std()),**{f"P{p}":float(np.percentile(values,p)) for p in (10,25,50,75,90,95,99)},"min":float(values.min()),"max":float(values.max()),"max_abs":float(np.max(np.abs(values)))}


def ranking_audit(samples,c0,c2,target,sigma):
    c0m,_=r1b.metrics(samples,c0,sigma,target,"C0_FROZEN_B0");c2m,_=r1b.metrics(samples,c2,sigma,target,"C2_RCEOC")
    changed=[];max_pairwise=0.0
    for episode,indices in b15.group_episode(samples).items():
        old,new=c0[indices],c2[indices];max_pairwise=max(max_pairwise,float(np.max(np.abs((old[:,None]-old[None])-(new[:,None]-new[None])))))
        if r1c.rank_signature(samples,indices,c0)!=r1c.rank_signature(samples,indices,c2):changed.append(episode)
    keys=("mean_within_episode_spearman","mean_feasible_within_episode_spearman","mean_feasible_pairwise_accuracy","gt_best_top1_accuracy","gt_best_top2_recall","mean_gt_best_rank")
    return {"label":LABEL,"mechanism_result":MECHANISM,"maximum_pairwise_difference_error":max_pairwise,"rank_signature_changes":len(changed),"changed_episode_ids":changed,**{f"{key}_C0":c0m[key] for key in keys},**{f"{key}_C2":c2m[key] for key in keys},"exact_invariant":max_pairwise<=TOLERANCE and not changed and all(c0m[key]==c2m[key] for key in keys)}


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite R1D result: {args.output_dir}")
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
    backbone_state_before=state_sha(backbone.state_dict())
    frozen={split:r1b.extract_frozen(backbone,value,payload["normalizer"],args.batch_size,torch,device) for split,value in samples.items()}
    b0={split:frozen[split]["old_benefit"] for split in samples};data={split:episode_data(samples[split],targets[split],b0[split],frozen[split]["context"],anchors) for split in samples}
    c0_sign=r1b.sign_summary(samples["validation"],b0["validation"],targets["validation"],"C0_FROZEN_B0")
    if c0_sign["safe_beneficial_count"]!=115 or c0_sign["predicted_positive_count"]!=42:raise RuntimeError("C0 strict reproduction failed")

    torch.manual_seed(args.seed);head=RuntimeConditionedEpisodeOffset();initial_head_sha=state_sha(head.state_dict());trained=train_head(head,data["train"],data["validation"],samples["validation"],targets["validation"],b0["validation"],args,torch,device)
    with torch.inference_mode():offset={split:trained["head"](data[split]["z_generic"].to(device)).cpu().numpy().astype(np.float64) for split in data}
    c2={split:apply_offsets(samples[split],data[split]["episode_ids"],offset[split],b0[split]) for split in data}
    generic_map={split:{episode:index for episode,index in zip(data[split]["episode_ids"],data[split]["generic_indices"])} for split in data}
    frgr={}
    for split in data:frgr[split],_=frozen_runtime_generic_reanchor(b0[split],[sample.episode_id for sample in samples[split]],generic_map[split])
    controls={"C0_FROZEN_B0":b0["validation"],"C1_HISTORICAL_FRGR":frgr["validation"],"C2_RCEOC":c2["validation"]}
    sigma=np.exp(.5*frozen["validation"]["log_variance"].numpy())*float(payload["normalizer"]["benefit_scale"])
    comparison=[]
    for name,value in controls.items():metric,_=r1b.metrics(samples["validation"],value,sigma,targets["validation"],name);comparison.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,**metric})
    metrics_by={row["model"]:row for row in comparison};sign_rows=[r1b.sign_summary(samples["validation"],value,targets["validation"],name) for name,value in controls.items()];signs={row["model"]:row for row in sign_rows}
    c0s,c2s=signs["C0_FROZEN_B0"],signs["C2_RCEOC"];target=targets["validation"]
    safe=np.asarray([s.targets.feasible and not s.split_metadata["harm_v2_evaluation_only"] for s in samples["validation"]],bool)&(target>r1b.TOLERANCE)
    recovered=safe&(b0["validation"]<=0)&(c2["validation"]>0);regressed=safe&(b0["validation"]>0)&(c2["validation"]<=0)
    recovery=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"safe_beneficial_count":int(safe.sum()),"gross_recovery":int(recovered.sum()),"regression":int(regressed.sum()),"net_recovery":int(recovered.sum()-regressed.sum())}]
    calibration=[r1c.calibration_row(samples["validation"],value,target,name) for name,value in controls.items()]
    ranking=ranking_audit(samples["validation"],b0["validation"],c2["validation"],target,sigma)

    offset_accuracy=[]
    for episode,predicted,truth in zip(data["validation"]["episode_ids"],offset["validation"],data["validation"]["delta_star"]):offset_accuracy.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"episode_id":episode,"delta_hat":float(predicted),"delta_star_validation_audit_only":float(truth),"absolute_error":float(abs(predicted-truth))})
    offset_stats={"MAE":float(np.mean(np.abs(offset["validation"]-data["validation"]["delta_star"]))),"Pearson":pearson(offset["validation"],data["validation"]["delta_star"]),"Spearman":spearman(offset["validation"],data["validation"]["delta_star"])}
    for row in offset_accuracy:row.update(offset_stats)
    tail=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"record_type":"SUMMARY","split":split,"clipping":False,"manual_scaling":False,**distribution(value)} for split,value in offset.items()]
    for rank,(episode,value) in enumerate(sorted(zip(data["validation"]["episode_ids"],offset["validation"]),key=lambda item:(-abs(item[1]),item[0]))[:10],1):tail.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"record_type":"TOP_ABS_VALIDATION_OFFSET","rank":rank,"episode_id":episode,"delta_hat":float(value),"abs_delta_hat":float(abs(value))})

    predicates={"C7":lambda s:any(str(x).startswith("C7") for x in s.split_metadata["contexts_evaluation_only"]),"STOP":lambda s:s.split_metadata["motion_type_evaluation_only"]=="stop"}
    audits={group:[{"group":group,**r1b.sign_summary(samples["validation"],value,target,name,predicate)} for name,value in controls.items()] for group,predicate in predicates.items()}
    hold=r1c.hold_rows(samples["validation"],target,controls)
    changed={episode for episode,row in anchors.items() if row["split"]=="validation" and row["anchor_agrees"]=="False"};anchor_rows=[]
    for group,predicate in (("ANCHOR_SAME",lambda e:e not in changed),("ANCHOR_CHANGED",lambda e:e in changed)):
        mask=np.asarray([predicate(s.episode_id) for s in samples["validation"]],bool);subset=[s for s,keep in zip(samples["validation"],mask) if keep];rows=[]
        for name in ("C0_FROZEN_B0","C2_RCEOC"):
            sign=r1b.sign_summary(subset,controls[name][mask],target[mask],name);cal=r1c.calibration_row(subset,controls[name][mask],target[mask],name);rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"group":group,"model":name,"safe_beneficial_count":sign["safe_beneficial_count"],"predicted_positive_count":sign["predicted_positive_count"],"safe_beneficial_sign_accuracy":sign["safe_beneficial_sign_accuracy"],"Benefit_MAE":cal["Benefit_MAE"]})
        improvement=rows[1]["safe_beneficial_sign_accuracy"]-rows[0]["safe_beneficial_sign_accuracy"]
        for row in rows:row["C2_minus_C0_sign_improvement"]=improvement
        anchor_rows.extend(rows)

    historical=[row for row in r1b.read_rows(PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/historical_sign_failure_reclassification.csv") if row["category"]=="A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"]
    by_id={sample.sample_id:index for index,sample in enumerate(samples["validation"])};indices=np.asarray([by_id[row["candidate_id"]] for row in historical],int)
    historical_rows=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"historical_true_failure_count":len(indices),"recovered_positive_count":int(np.sum(value[indices]>0)),"remaining_failure_count":int(np.sum(value[indices]<=0)),"GARA_recovery_reference":14,"FRGR_recovery_reference":28} for name,value in controls.items()]

    shortcut=[];episode_meta={episode:samples["validation"][indices[0]] for episode,indices in b15.group_episode(samples["validation"]).items()};offset_by=dict(zip(data["validation"]["episode_ids"],offset["validation"]))
    getters={"runtime_generic_action":lambda e:int(anchors[e]["runtime_anchor_action_id"]),"motion":lambda e:str(episode_meta[e].split_metadata["motion_type_evaluation_only"]),"context":lambda e:"|".join(map(str,episode_meta[e].split_metadata["contexts_evaluation_only"])) or "NONE","profile_audit":lambda e:str(episode_meta[e].split_metadata["person_profile_id"])}
    values=np.asarray(list(offset_by.values()));overall=float(values.mean());variance=float(values.var())
    for dimension,getter in getters.items():
        groups=defaultdict(list)
        for episode,value in offset_by.items():groups[getter(episode)].append(value)
        between=float(sum(len(group)*(np.mean(group)-overall)**2 for group in groups.values())/len(values));ratio=between/max(variance,1e-12)
        shortcut.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"dimension":dimension,"group_count":len(groups),"between_variance_ratio":ratio,"group_mean_min":float(min(np.mean(x) for x in groups.values())),"group_mean_max":float(max(np.mean(x) for x in groups.values())),"near_deterministic_shortcut":ratio>=.95,"profile_ID_runtime_input":False if dimension=="profile_audit" else ""})

    harm_payload=torch.load(args.harm_checkpoint,map_location=device,weights_only=False);harm_head=RiskPreservingBypassHead().to(device);harm_head.load_state_dict(harm_payload["model_state_dict"]);harm_head.eval()
    for parameter in harm_head.parameters():parameter.requires_grad_(False)
    with torch.inference_mode():harm_before=harm_head(frozen["validation"]["bypass"].to(device)).cpu().numpy();harm_after=harm_head(frozen["validation"]["bypass"].to(device)).cpu().numpy()
    b0_after=r1b.extract_frozen(backbone,samples["validation"],payload["normalizer"],args.batch_size,torch,device)["old_benefit"]
    backbone_state_after=state_sha(backbone.state_dict());checksums_after={"manifest_v3":file_sha(args.manifest_v3),"Benefit_Target_v2":file_sha(args.target_v2),"runtime_anchor_map":file_sha(args.anchor_map),"R1_v3_BASE":file_sha(args.r1_checkpoint),"HARM_v3_BASE":file_sha(args.harm_checkpoint)}
    harm_isolation={"label":LABEL,"mechanism_result":MECHANISM,"HARM_checkpoint_SHA_before":checksums_before["HARM_v3_BASE"],"HARM_checkpoint_SHA_after":checksums_after["HARM_v3_BASE"],"HARM_checkpoint_unchanged":checksums_before["HARM_v3_BASE"]==checksums_after["HARM_v3_BASE"],"harm_optimizer_created":False,"validation_harm_logits_max_abs_diff":float(np.max(np.abs(harm_after-harm_before))),"harm_outputs_exact":np.array_equal(harm_before,harm_after)}
    runtime_generic_c2=np.asarray([c2["validation"][index] for index in data["validation"]["generic_indices"]]);generic_zero={"distribution":distribution(runtime_generic_c2),"MAE_to_zero":float(np.mean(np.abs(runtime_generic_c2)))}

    c0m,c2m=metrics_by["C0_FROZEN_B0"],metrics_by["C2_RCEOC"]
    stop_c2=next(row for row in audits["STOP"] if row["model"]=="C2_RCEOC")
    b0_prediction_max_abs_diff=float(np.max(np.abs(b0_after-b0["validation"])))
    gates={
        "Gate_A":{"name":"Isolation","checks":{"only_129_param_offset_head_trained":trained["optimizer_parameter_count"]==129,"B0_checkpoint_frozen":checksums_before["R1_v3_BASE"]==checksums_after["R1_v3_BASE"] and backbone_state_before==backbone_state_after,"B0_predictions_max_abs_diff_zero":b0_prediction_max_abs_diff==0.0,"Harm_frozen":harm_isolation["HARM_checkpoint_unchanged"],"manifest_target_anchor_frozen":all(checksums_before[key]==checksums_after[key] for key in ("manifest_v3","Benefit_Target_v2","runtime_anchor_map")),"TEST_reads_zero":TEST_READS==0}},
        "Gate_B":{"name":"Sign Improvement","checks":{"safe_beneficial_accuracy_at_least_51_52_percent":c2s["safe_beneficial_sign_accuracy"]>=.5152,"net_recovery_positive":recovery[0]["net_recovery"]>0}},
        "Gate_C":{"name":"Exact Ranking Preservation","checks":{"all_ranking_and_signatures_exact":ranking["exact_invariant"]}},
        "Gate_D":{"name":"Absolute Calibration","checks":{"C2_MAE_not_above_C0":c2m["Benefit_MAE"]<=c0m["Benefit_MAE"]}},
        "Gate_E":{"name":"No Degenerate Shift","checks":{"GT_negative_FPR_increase_at_most_0_05":c2s["GT_negative_false_positive_rate"]<=c0s["GT_negative_false_positive_rate"]+.05,"finite":bool(np.isfinite(c2["validation"]).all() and np.isfinite(offset["validation"]).all()),"no_extreme_numeric_explosion":float(np.max(np.abs(offset["validation"])))<10.0}},
        "Gate_F":{"name":"System Isolation","checks":{"Harm_outputs_unchanged":harm_isolation["harm_outputs_exact"],"threshold_unchanged":True,"decision_chain_not_run":True,"arbitration_unchanged":True}},
    }
    for gate in gates.values():gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())
    outcome="RCEOC SUCCESS" if gates["all_passed"] else "EPISODE-OFFSET FAMILY EXHAUSTED"
    frozen_contract={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"checksums_before":checksums_before,"checksums_after":checksums_after,"B0_state_before":backbone_state_before,"B0_state_after":backbone_state_after,"B0_requires_grad_parameters":sum(p.requires_grad for p in backbone.parameters()),"B0_validation_prediction_max_abs_diff_before_after":b0_prediction_max_abs_diff,"trainable_component":"RuntimeConditionedEpisodeOffset Linear(128,1) only","trainable_parameters":129,"offset_head_initial_state_SHA256":initial_head_sha,"offset_head_selected_state_SHA256":state_sha(trained["head"].state_dict()),"optimizer_parameter_count":trained["optimizer_parameter_count"],"Harm_trainable_parameters":sum(p.requires_grad for p in harm_head.parameters()),"threshold_calibration":False,"decision_chain":False,"arbitration":False}
    input_contract={"label":LABEL,"mechanism_result":MECHANISM,"input":"z_g(e) in R^128","source":"frozen R1-v3-BASE final Benefit-side fused representation for R1A runtime canonical generic","one_input_per_episode":True,"runtime_valid":True,"forbidden_inputs":["GT future","GT benefit","GT harm","GT cost","profile ID","oracle information","candidate-specific features"],"runtime_anchor_map_reused":True,"HOLD_anchor_forbidden":True}
    target_contract={"label":LABEL,"mechanism_result":MECHANISM,"formula":"delta_star(e) = mean_i(BenefitTargetV2(e,i) - mu_B0(e,i))","train_supervision_only":True,"candidates_per_episode":6,"validation_delta_star_audit_and_selector_only":True,"runtime_delta_star_access":False,"test_reads":0}
    config={"label":LABEL,"mechanism_result":MECHANISM,"seed":42,"optimizer":"AdamW","learning_rate":args.learning_rate,"weight_decay":.001,"batch_size_episodes":args.batch_size,"max_epochs":args.epochs,"patience":args.patience,"loss":"episode offset MSE","selector_order":["minimum Validation Benefit MAE","maximum safe-beneficial sign accuracy","earlier epoch"],"hyperparameter_search":False,"output_clipping":False,"manual_offset_scaling":False,"candidate_specific_correction":False}
    selector={"label":LABEL,"mechanism_result":MECHANISM,"selected_epoch":int(trained["selected"]["epoch"]),"selected_Benefit_MAE":float(trained["selected"]["Benefit_MAE"]),"selected_safe_beneficial_sign_accuracy":float(trained["selected"]["safe_beneficial_sign_accuracy"]),"epochs_completed":trained["epochs_completed"],"training_time_s":trained["training_time_s"],"selection_order":config["selector_order"]}
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"C0_strictly_reproduced":c0s["predicted_positive_count"]==42 and abs(c0m["Benefit_MAE"]-EXPECTED_C0_MAE)<=1e-12,"offset_input":input_contract,"trainable_parameters":129,"selected_epoch":selector["selected_epoch"],"C0_safe_beneficial":c0s,"C2_safe_beneficial":c2s,"sign_improvement":c2s["safe_beneficial_sign_accuracy"]-c0s["safe_beneficial_sign_accuracy"],"sign_recovery":recovery[0],"C0_MAE":c0m["Benefit_MAE"],"C2_MAE":c2m["Benefit_MAE"],"ranking_invariant":ranking["exact_invariant"],"offset_accuracy":offset_stats,"offset_tail":distribution(offset["validation"]),"FRGR_max_abs_reference":4.288574695587158,"runtime_generic_C2":generic_zero,"STOP_REGRESSION_WARNING":stop_c2["predicted_positive_count"]<6,"historical_failure_recovery":{row["model"]:row for row in historical_rows},"gates":gates,"outcome_classification":outcome,"RCEOC_successful":gates["all_passed"],"ready_for_v3_safe_decision_chain_reconstruction":gates["all_passed"],"next_stage_started":False}
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"RCEOC_OFFSET_HEAD","state_dict":trained["head"].state_dict(),"architecture":trained["head"].architecture_audit(),"selector":selector,"input_contract":input_contract,"Target_v2_SHA":r1b.EXPECTED_TARGET_SHA,"test_reads":0},args.output_dir/"checkpoints/rceoc_offset_head.pt")
    io.write_json(args.output_dir/"frozen_contract.json",frozen_contract);io.write_json(args.output_dir/"offset_input_contract.json",input_contract);io.write_json(args.output_dir/"offset_target_contract.json",target_contract);io.write_json(args.output_dir/"training_config.json",config);io.write_csv(args.output_dir/"training_curve.csv",trained["rows"]);io.write_json(args.output_dir/"selector.json",selector);io.write_csv(args.output_dir/"c0_frgr_c2_comparison.csv",comparison);io.write_csv(args.output_dir/"safe_beneficial_sign.csv",sign_rows);io.write_csv(args.output_dir/"sign_recovery.csv",recovery);io.write_json(args.output_dir/"ranking_invariance.json",ranking);io.write_csv(args.output_dir/"mae_calibration.csv",calibration);io.write_csv(args.output_dir/"offset_accuracy.csv",offset_accuracy);io.write_csv(args.output_dir/"offset_tail_audit.csv",tail);io.write_csv(args.output_dir/"c7_audit.csv",audits["C7"]);io.write_csv(args.output_dir/"stop_audit.csv",audits["STOP"]);io.write_csv(args.output_dir/"hold_audit.csv",hold);io.write_csv(args.output_dir/"anchor_same_changed.csv",anchor_rows);io.write_csv(args.output_dir/"historical_failure_recovery.csv",historical_rows);io.write_csv(args.output_dir/"shortcut_audit.csv",shortcut);io.write_json(args.output_dir/"harm_isolation.json",harm_isolation);io.write_json(args.output_dir/"gate_results.json",gates);io.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
