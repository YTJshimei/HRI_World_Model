"""Phase 5B-v3-R1C parameter-free frozen runtime-generic re-anchoring.

This development-only synthetic audit applies ``mu_i - mu_g`` to the frozen
R1-v3-BASE Benefit predictions.  It performs no training, checkpoint selection,
threshold calibration, arbitration, or decision-chain evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.evaluation.context_value_metrics import pearson, spearman
from src.evaluation.frozen_runtime_generic_reanchor import FRGR_ALPHA, frozen_runtime_generic_reanchor
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-v3-R1C Frozen Runtime-Generic Post-hoc Re-Anchoring"
TEST_READS = 0
EXPECTED_C0_SAFE_COUNT = 115
EXPECTED_C0_POSITIVE = 42
TOLERANCE = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--manifest-v3", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--target-v2", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv")
    parser.add_argument("--anchor-map", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv")
    parser.add_argument("--r1-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1c_frozen_runtime_generic_reanchor")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, np.float64)
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()), **{f"P{p}": float(np.percentile(values, p)) for p in (10,25,50,75,90)}, "min": float(values.min()), "max": float(values.max())}


def calibration_row(samples, prediction, target, model):
    prediction, target = np.asarray(prediction), np.asarray(target)
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    error = prediction[feasible] - target[feasible]
    return {"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":model,"Benefit_MAE":float(np.mean(np.abs(error))),"mean_error":float(np.mean(error)),"median_error":float(np.median(error)),"global_bias":float(np.mean(error)),"positive_bias":float(np.mean((prediction-target)[target>r1b.TOLERANCE])),"negative_bias":float(np.mean((prediction-target)[target < -r1b.TOLERANCE]))}


def rank_signature(samples, indices, prediction):
    actions=np.asarray([samples[index].split_metadata["candidate_action_id_audit"] for index in indices],int)
    values=np.asarray(prediction)[indices]
    return tuple(int(actions[index]) for index in np.lexsort((actions,-values)))


def hold_rows(samples, target, predictions):
    groups=b15.group_episode(samples); hold=np.asarray([s.split_metadata["candidate_action_id_audit"]==HOLD_ACTION_ID for s in samples],bool)
    feasible=np.asarray([s.targets.feasible for s in samples],bool); harm=np.asarray([s.split_metadata["harm_v2_evaluation_only"] for s in samples],bool)
    beneficial=hold&(target>r1b.TOLERANCE); safe=beneficial&feasible&~harm; rows=[]
    for name,value in predictions.items():
        ranks=[]
        for index in np.flatnonzero(hold):
            indices=groups[samples[index].episode_id]; ranks.append(int(b15.ranks_desc(np.asarray(value)[indices])[indices.index(index)]))
        nonbeneficial=hold&(target<=r1b.TOLERANCE)
        rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"beneficial_HOLD_count":int(beneficial.sum()),"beneficial_HOLD_predicted_positive":int(np.sum(value[beneficial]>0)),"beneficial_HOLD_sign_accuracy":float(np.mean(value[beneficial]>0)),"safe_beneficial_HOLD_count":int(safe.sum()),"safe_beneficial_HOLD_predicted_positive":int(np.sum(value[safe]>0)),"safe_beneficial_HOLD_sign_accuracy":float(np.mean(value[safe]>0)),"nonbeneficial_HOLD_count":int(nonbeneficial.sum()),"nonbeneficial_HOLD_predicted_positive":int(np.sum(value[nonbeneficial]>0)),"nonbeneficial_HOLD_FPR":float(np.mean(value[nonbeneficial]>0)),"rank_signature_SHA256":hashlib.sha256(json.dumps(ranks,separators=(",",":")).encode()).hexdigest(),"rank_mean":float(np.mean(ranks)),"rank_median":float(np.median(ranks)),**{f"rank_P{p}":float(np.percentile(ranks,p)) for p in (10,25,50,75,90)}})
    return rows


def frozen_record(checksums_before, checksums_after, backbone_before, backbone_after):
    return {"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"checksums_before":checksums_before,"checksums_after":checksums_after,"B0_checkpoint_unchanged":checksums_before["R1_v3_BASE"]==checksums_after["R1_v3_BASE"],"R1_state_unchanged":backbone_before==backbone_after,"optimizer_steps":0,"backward_calls":0,"new_head_created":False,"trainable_parameters":0,"FRGR_alpha":FRGR_ALPHA,"tunable_parameters":0,"checkpoint_selection_run":False,"threshold_calibration_run":False,"decision_chain_run":False,"arbitration_run":False}


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite R1C result: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    import torch
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device)
    checksums_before, labels, anchors=r1b.load_contract(args)
    episodes={"train":build_development_split("train",240,GENERATOR_SEED,RISK_SEED),"validation":build_development_split("validation",240,GENERATOR_SEED+1000,RISK_SEED+1000)}
    samples={split:build_v3_temporal_samples(value) for split,value in episodes.items()}
    targets={split:r1b.apply_target_v2(value,labels) for split,value in samples.items()}
    payload=torch.load(args.r1_checkpoint,map_location=device,weights_only=False)
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    model=RichTemporalSmallTransformerV3().to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    backbone_before=state_sha(model.state_dict())
    predictions={}
    for split in ("train","validation"):
        prediction=b1.predict("R1-v3-BASE",model,samples[split],payload["normalizer"],args.batch_size,torch,device)
        predictions[split]=np.asarray(prediction["benefit"],np.float64)
    backbone_after=state_sha(model.state_dict())

    generic_maps={}; c1={}; offsets={}
    for split in ("train","validation"):
        grouped=b15.group_episode(samples[split]); generic_maps[split]={}
        for episode_id,indices in grouped.items():
            action=int(anchors[episode_id]["runtime_anchor_action_id"])
            match=[index for index in indices if int(samples[split][index].split_metadata["candidate_action_id_audit"])==action]
            if len(match)!=1: raise RuntimeError(f"frozen runtime anchor missing for {episode_id}")
            generic_maps[split][episode_id]=match[0]
        c1[split],offsets[split]=frozen_runtime_generic_reanchor(predictions[split],[s.episode_id for s in samples[split]],generic_maps[split])

    validation=samples["validation"]; target=targets["validation"]; c0=predictions["validation"]
    controls={"C0_FROZEN_B0":c0,"C1_FRGR":c1["validation"]}
    c0_sign=r1b.sign_summary(validation,c0,target,"C0_FROZEN_B0"); c1_sign=r1b.sign_summary(validation,c1["validation"],target,"C1_FRGR")
    if c0_sign["safe_beneficial_count"]!=EXPECTED_C0_SAFE_COUNT or c0_sign["predicted_positive_count"]!=EXPECTED_C0_POSITIVE: raise RuntimeError("C0 failed strict B0 Target-v2 reproduction")
    sigma=np.exp(.5*r1b.extract_frozen(model,validation,payload["normalizer"],args.batch_size,torch,device)["log_variance"].numpy())*float(payload["normalizer"]["benefit_scale"])
    comparisons=[]; rank_rows={}
    for name,value in controls.items():
        metric,rank_rows[name]=r1b.metrics(validation,value,sigma,target,name); comparisons.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,**metric})
    metrics_by={row["model"]:row for row in comparisons}; sign_rows=[c0_sign,c1_sign]; signs={row["model"]:row for row in sign_rows}

    grouped=b15.group_episode(validation); max_pairwise=0.0; changed=[]
    for episode_id,indices in grouped.items():
        old=c0[indices]; new=c1["validation"][indices]
        max_pairwise=max(max_pairwise,float(np.max(np.abs((old[:,None]-old[None])-(new[:,None]-new[None])))))
        if rank_signature(validation,indices,c0)!=rank_signature(validation,indices,c1["validation"]): changed.append(episode_id)
    c0m,c1m=metrics_by["C0_FROZEN_B0"],metrics_by["C1_FRGR"]
    ranking={"label":LABEL,"mechanism_result":MECHANISM,"episode_count":len(grouped),"maximum_pairwise_difference_error":max_pairwise,"rank_signature_changes":len(changed),"changed_episode_ids":changed,"mean_within_episode_spearman_C0":c0m["mean_within_episode_spearman"],"mean_within_episode_spearman_C1":c1m["mean_within_episode_spearman"],"feasible_within_episode_spearman_C0":c0m["mean_feasible_within_episode_spearman"],"feasible_within_episode_spearman_C1":c1m["mean_feasible_within_episode_spearman"],"pairwise_C0":c0m["mean_feasible_pairwise_accuracy"],"pairwise_C1":c1m["mean_feasible_pairwise_accuracy"],"Top1_C0":c0m["gt_best_top1_accuracy"],"Top1_C1":c1m["gt_best_top1_accuracy"],"Top2_C0":c0m["gt_best_top2_recall"],"Top2_C1":c1m["gt_best_top2_recall"],"mean_GT_best_rank_C0":c0m["mean_gt_best_rank"],"mean_GT_best_rank_C1":c1m["mean_gt_best_rank"],"exact_invariant":max_pairwise<=TOLERANCE and not changed and all(c0m[key]==c1m[key] for key in ("mean_within_episode_spearman","mean_feasible_within_episode_spearman","mean_feasible_pairwise_accuracy","gt_best_top1_accuracy","gt_best_top2_recall","mean_gt_best_rank"))}

    safe=np.asarray([s.targets.feasible and not s.split_metadata["harm_v2_evaluation_only"] for s in validation],bool)&(target>r1b.TOLERANCE)
    c0p,c1p=c0>0,c1["validation"]>0; recovered=safe&~c0p&c1p; regressed=safe&c0p&~c1p
    sign_recovery=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"safe_beneficial_count":int(safe.sum()),"gross_recovery":int(recovered.sum()),"regression":int(regressed.sum()),"net_recovery":int(recovered.sum()-regressed.sum())}]
    calibration=[calibration_row(validation,value,target,name) for name,value in controls.items()]

    predicates={"C7":lambda s:any(str(x).startswith("C7") for x in s.split_metadata["contexts_evaluation_only"]),"STOP":lambda s:s.split_metadata["motion_type_evaluation_only"]=="stop"}
    audits={name:[{"group":name,**r1b.sign_summary(validation,value,target,model_name,predicate)} for model_name,value in controls.items()] for name,predicate in predicates.items()}
    hold=hold_rows(validation,target,controls)

    changed_episodes={episode for episode,row in anchors.items() if row["split"]=="validation" and row["anchor_agrees"]=="False"}; anchor_rows=[]
    for group,predicate in (("ANCHOR_SAME",lambda e:e not in changed_episodes),("ANCHOR_CHANGED",lambda e:e in changed_episodes)):
        mask=np.asarray([predicate(s.episode_id) for s in validation],bool); subset=[s for s,keep in zip(validation,mask) if keep]
        subgroup=[]
        for name,value in controls.items():
            sign=r1b.sign_summary(subset,value[mask],target[mask],name); cal=calibration_row(subset,value[mask],target[mask],name); subgroup.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"group":group,**{k:v for k,v in sign.items() if k not in ("synthetic_interaction","mechanism_result")},"Benefit_MAE":cal["Benefit_MAE"]})
        base,next_value=subgroup
        for row in subgroup: row["C1_minus_C0_safe_beneficial_accuracy"] = next_value["safe_beneficial_sign_accuracy"]-base["safe_beneficial_sign_accuracy"]
        anchor_rows.extend(subgroup)

    validation_anchors=[row for row in anchors.values() if row["split"]=="validation"]
    offset_by_episode={episode:c0[index] for episode,index in generic_maps["validation"].items()}
    gt_shift=np.asarray([float(row["episode_zero_shift"]) for row in validation_anchors]); correction=np.asarray([-offset_by_episode[row["episode_id"]] for row in validation_anchors])
    alignment_stats={"Pearson":pearson(correction,gt_shift),"Spearman":spearman(correction,gt_shift),"MAE":float(np.mean(np.abs(correction-gt_shift)))}
    alignment=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"episode_id":row["episode_id"],"predicted_correction_negative_mu_B0_generic":float(pred),"GT_Delta_e_audit_only":float(gt),"absolute_error":float(abs(pred-gt)),**alignment_stats} for row,pred,gt in zip(validation_anchors,correction,gt_shift)]

    historical=[row for row in r1b.read_rows(PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/historical_sign_failure_reclassification.csv") if row["category"]=="A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"]
    index_by_id={s.sample_id:i for i,s in enumerate(validation)}; historical_indices=np.asarray([index_by_id[row["candidate_id"]] for row in historical],int)
    historical_rows=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"historical_true_failure_count":len(historical_indices),"recovered_positive_count":int(np.sum(value[historical_indices]>0)),"remaining_failure_count":int(np.sum(value[historical_indices]<=0)),"R1B_GARA_diagnostic_recovery":14} for name,value in controls.items()]

    episode_corrections=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"record_type":"TOP_CORRECTION_EPISODE","episode_id":episode,"runtime_anchor_action_id":int(anchors[episode]["runtime_anchor_action_id"]),"mu_B0_generic":float(offset),"applied_correction":float(-offset),"abs_correction":float(abs(offset))} for episode,offset in offset_by_episode.items()]
    episode_corrections=sorted(episode_corrections,key=lambda row:(-row["abs_correction"],row["episode_id"]))
    degenerate=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"record_type":"SUMMARY","episode_count":len(offset_by_episode),"max_abs_correction":float(max(abs(x) for x in offset_by_episode.values())),"clipping":False,"temperature":False,"offset_scaling":False,"alpha":FRGR_ALPHA}]+episode_corrections[:10]
    generic_distribution=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"split":split,**distribution([predictions[split][index] for index in generic_maps[split].values()])} for split in ("train","validation")]

    checksums_after={"manifest_v3":file_sha(args.manifest_v3),"Benefit_Target_v2":file_sha(args.target_v2),"runtime_anchor_map":file_sha(args.anchor_map),"R1_v3_BASE":file_sha(args.r1_checkpoint),"HARM_v3_BASE":file_sha(args.harm_checkpoint)}
    frozen=frozen_record(checksums_before,checksums_after,backbone_before,backbone_after)
    harm={"label":LABEL,"mechanism_result":MECHANISM,"HARM_checkpoint_sha_before":checksums_before["HARM_v3_BASE"],"HARM_checkpoint_sha_after":checksums_after["HARM_v3_BASE"],"HARM_checkpoint_unchanged":checksums_before["HARM_v3_BASE"]==checksums_after["HARM_v3_BASE"],"harm_model_loaded":False,"harm_outputs_modified":False,"FRGR_scope":"Benefit prediction only"}
    generic_mask=np.asarray([index==generic_maps["validation"][s.episode_id] for index,s in enumerate(validation)],bool); generic_max=float(np.max(np.abs(c1["validation"][generic_mask])))
    gates={
        "Gate_A":{"name":"Isolation / Contract","checks":{"B0_checkpoint_unchanged":frozen["B0_checkpoint_unchanged"],"Target_v2_SHA_unchanged":checksums_before["Benefit_Target_v2"]==checksums_after["Benefit_Target_v2"]==r1b.EXPECTED_TARGET_SHA,"runtime_anchor_map_unchanged":checksums_before["runtime_anchor_map"]==checksums_after["runtime_anchor_map"]==r1b.EXPECTED_ANCHOR_SHA,"TEST_reads_zero":TEST_READS==0,"no_training":frozen["optimizer_steps"]==frozen["backward_calls"]==0,"parameter_free_alpha_one":FRGR_ALPHA==1.0 and frozen["tunable_parameters"]==0}},
        "Gate_B":{"name":"Sign Improvement","checks":{"accuracy_improvement_at_least_0_10":c1_sign["safe_beneficial_sign_accuracy"]>=c0_sign["safe_beneficial_sign_accuracy"]+.10,"positive_count_increased":c1_sign["predicted_positive_count"]>c0_sign["predicted_positive_count"]}},
        "Gate_C":{"name":"Exact Ranking Preservation","checks":{"pairwise_exact":c0m["mean_feasible_pairwise_accuracy"]==c1m["mean_feasible_pairwise_accuracy"] and max_pairwise<=TOLERANCE,"Top1_exact":c0m["gt_best_top1_accuracy"]==c1m["gt_best_top1_accuracy"],"Top2_exact":c0m["gt_best_top2_recall"]==c1m["gt_best_top2_recall"],"rank_signatures_exact":not changed,"within_episode_metrics_exact":ranking["exact_invariant"]}},
        "Gate_D":{"name":"Calibration Guard","checks":{"MAE_worsening_at_most_10_percent":c1m["Benefit_MAE"]<=c0m["Benefit_MAE"]*1.10,"finite":all(np.isfinite(c1["validation"])),"runtime_generic_exact_zero":generic_max<=TOLERANCE}},
        "Gate_E":{"name":"No Degenerate Positive Shift","checks":{"GT_negative_FPR_increase_at_most_0_05":c1_sign["GT_negative_false_positive_rate"]<=c0_sign["GT_negative_false_positive_rate"]+.05,"safe_beneficial_precision_not_collapsed":c1_sign["safe_beneficial_precision"]>=c0_sign["safe_beneficial_precision"]-.10}},
        "Gate_F":{"name":"Harm / System Isolation","checks":{"Harm_checkpoint_unchanged":harm["HARM_checkpoint_unchanged"],"decision_chain_not_run":not frozen["decision_chain_run"],"threshold_not_changed":not frozen["threshold_calibration_run"],"arbitration_not_changed":not frozen["arbitration_run"]}},
    }
    for gate in gates.values():gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())
    stop_by={row["model"]:row for row in audits["STOP"]}; stop_warning=stop_by["C1_FRGR"]["safe_beneficial_sign_accuracy"]<stop_by["C0_FROZEN_B0"]["safe_beneficial_sign_accuracy"]
    if not gates["Gate_B"]["passed"]: outcome="EPISODE-WISE ZERO CORRECTION ALONE INSUFFICIENT"
    elif not gates["Gate_D"]["passed"] or not gates["Gate_E"]["passed"]: outcome="B0 PREDICTED GENERIC OFFSET NOT STABLE ENOUGH; BENEFIT-SPECIFIC REPRESENTATION PROBLEM"
    elif gates["all_passed"]: outcome="FRGR SUCCESS"
    else: outcome="FRGR FAILED OTHER PREREGISTERED GATE"
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"C0_reproduced":True,"C0_safe_beneficial":c0_sign,"C1_safe_beneficial":c1_sign,"safe_beneficial_accuracy_improvement":c1_sign["safe_beneficial_sign_accuracy"]-c0_sign["safe_beneficial_sign_accuracy"],"exceeds_R1B_GARA_41_74_percent":c1_sign["safe_beneficial_sign_accuracy"]>0.41739130434782606,"sign_recovery":sign_recovery[0],"C0_metrics":c0m,"C1_metrics":c1m,"MAE_relative_change":c1m["Benefit_MAE"]/c0m["Benefit_MAE"]-1.0,"generic_transformed_max_abs":generic_max,"target_shift_alignment":alignment_stats,"historical_true_failure_recovery":{row["model"]:row for row in historical_rows},"maximum_abs_episode_correction":degenerate[0]["max_abs_correction"],"STOP_REGRESSION_WARNING":stop_warning,"outcome_classification":outcome,"gates":gates,"FRGR_successful":gates["all_passed"],"ready_for_v3_safe_decision_chain_reconstruction":gates["all_passed"],"next_stage_started":False}
    contract={"label":LABEL,"mechanism_result":MECHANISM,"name":"FRGR","formula":"mu_C1(i) = mu_B0(i) - mu_B0(g_runtime)","alpha":FRGR_ALPHA,"parameter_free":True,"runtime_anchor_source":"frozen R1A runtime_anchor_map.csv","anchor_reselection":False,"GT_reads_for_runtime_transform":0,"clipping":False,"temperature":False,"offset_scaling":False}
    c0_reproduction={"label":LABEL,"mechanism_result":MECHANISM,"strict_reproduction":True,"expected_safe_beneficial_count":115,"actual_safe_beneficial_count":c0_sign["safe_beneficial_count"],"expected_predicted_positive":42,"actual_predicted_positive":c0_sign["predicted_positive_count"],"expected_accuracy":42/115,"actual_accuracy":c0_sign["safe_beneficial_sign_accuracy"]}
    io.write_json(args.output_dir/"frozen_contract.json",frozen);io.write_json(args.output_dir/"c0_reproduction.json",c0_reproduction);io.write_json(args.output_dir/"frgr_contract.json",contract)
    io.write_csv(args.output_dir/"generic_offset_distribution.csv",generic_distribution);io.write_csv(args.output_dir/"target_shift_alignment.csv",alignment);io.write_csv(args.output_dir/"overall_comparison.csv",comparisons);io.write_csv(args.output_dir/"safe_beneficial_sign.csv",sign_rows);io.write_csv(args.output_dir/"sign_recovery.csv",sign_recovery);io.write_json(args.output_dir/"ranking_invariance.json",ranking);io.write_csv(args.output_dir/"mae_calibration.csv",calibration);io.write_csv(args.output_dir/"c7_audit.csv",audits["C7"]);io.write_csv(args.output_dir/"stop_audit.csv",audits["STOP"]);io.write_csv(args.output_dir/"hold_audit.csv",hold);io.write_csv(args.output_dir/"anchor_same_changed.csv",anchor_rows);io.write_csv(args.output_dir/"historical_true_failure_recovery.csv",historical_rows);io.write_csv(args.output_dir/"degenerate_shift_audit.csv",degenerate);io.write_json(args.output_dir/"harm_isolation.json",harm);io.write_json(args.output_dir/"gate_results.json",gates);io.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
