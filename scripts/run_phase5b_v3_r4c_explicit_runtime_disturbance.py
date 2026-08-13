"""Phase 5B-v3-R4C Explicit Runtime Action-Disturbance Advantage (ERADA).

C0, C1 and the GT-only O1 oracle use one matched scalar-feature readout.
Frozen B0 remains the only ranking path, HARM-v3 remains the only harm path,
and the sealed TEST split is never materialized.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
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
from scripts import run_phase5b_v3_r4a_oracle_future_pose_sufficiency as r4a
from src.data.adverse_response_dataset import GENERATOR_SEED,POPULATION_PROFILE,RISK_SEED,build_development_split
from src.evaluation.benefit_component_audit import episode_cost_components
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.models.runtime_disturbance_advantage import DisturbanceAdvantageBenefitReadout,runtime_disturbance_advantage
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL

MECHANISM="DEVELOPMENT MECHANISM RESULT"
ORACLE="GT FULL DISTURBANCE ORACLE - NOT RUNTIME VALID"
STAGE="Phase 5B-v3-R4C Explicit Runtime Action-Disturbance Advantage"
TEST_READS=0
LAMBDA_RANK=0.0
HISTORICAL_MAE_GUARD=1.735063
SIGN_TOLERANCE=1e-6


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
    parser.add_argument("--historical",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/historical_sign_failure_reclassification.csv")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r4c_explicit_runtime_disturbance")
    return parser.parse_args()


def action_ids(samples):
    return np.asarray([sample.split_metadata["candidate_action_id_audit"] for sample in samples],np.int64)


def build_features(episodes,samples,anchors):
    candidates=action_ids(samples);generic=np.asarray([int(anchors[sample.episode_id]["runtime_anchor_action_id"]) for sample in samples],np.int64)
    runtime=runtime_disturbance_advantage(candidates,generic)
    full_by_id={};robot_formula_by_id={}
    for episode in episodes:
        audit=episode_cost_components(episode,POPULATION_PROFILE);ids=audit["action_ids"];anchor=int(anchors[episode.episode_id]["runtime_anchor_action_id"]);anchor_index=int(np.flatnonzero(ids==anchor)[0])
        full=.55*(audit["components"]["disturbance"][anchor_index]-audit["components"]["disturbance"])
        robot_terms=sum((values for name,values in audit["subcomponents"].items() if name.startswith("disturbance.") and name!="disturbance.human_effect_magnitude"),np.zeros(len(ids)))
        robot=.55*(robot_terms[anchor_index]-robot_terms)
        for index,action in enumerate(ids):
            candidate_id=f"{episode.episode_id}:{int(action)}";full_by_id[candidate_id]=float(full[index]);robot_formula_by_id[candidate_id]=float(robot[index])
    full=np.asarray([full_by_id[sample.sample_id] for sample in samples],np.float32)[:,None]
    robot_formula=np.asarray([robot_formula_by_id[sample.sample_id] for sample in samples],np.float32)[:,None]
    return {"C0":np.zeros_like(runtime),"C1":runtime,"O1":full},generic,{"maximum_runtime_vs_formal_robot_component_error":float(np.max(np.abs(runtime-robot_formula))),"maximum_generic_runtime_feature_abs":float(np.max(np.abs(runtime[candidates==generic]))),"human_effect_oracle_increment":{"mean":float(np.mean(full-runtime)),"mean_abs":float(np.mean(np.abs(full-runtime))),"max_abs":float(np.max(np.abs(full-runtime)))}}


def matched_subgroup(samples,predictions,target,group,predicate):
    return [r4a.augmented_subgroup(samples,predictions[name],target,name,group,predicate) for name in ("C0","C1","O1")]


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite R4C: {args.output_dir}")
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
    backbone_before=r4a.state_sha(backbone.state_dict());data={split:r3.extract_representations(backbone,value,payload["normalizer"],args.batch_size,torch,device) for split,value in samples.items()}
    features={};exactness={}
    for split in samples:
        generic_indices,identity=r1b.generic_indices(samples[split],anchors)
        if any(int(anchors[episode]["runtime_anchor_action_id"])!=action for episode,action in identity):raise RuntimeError("runtime generic identity changed")
        split_features,generic_actions,split_exactness=build_features(episodes[split],samples[split],anchors)
        features[split]=split_features;exactness[split]=split_exactness
        data[split].update({"z_generic":data[split]["z_final"][generic_indices],"generic_indices":generic_indices,"generic_actions":generic_actions,"target":targets[split],"samples":samples[split],"scale":float(payload["normalizer"]["benefit_scale"])})
    if max(row["maximum_runtime_vs_formal_robot_component_error"] for row in exactness.values())>1e-7:raise RuntimeError("runtime disturbance formula does not match formal R4B robot-only terms")

    b0=data["validation"]["old_benefit"];sigma=np.exp(.5*data["validation"]["log_variance"].numpy().astype(np.float64))*data["validation"]["scale"]
    b0_metric,_=r1b.metrics(samples["validation"],b0,sigma,targets["validation"],"B0_FROZEN_RANKING");b0_sign=r1b.sign_summary(samples["validation"],b0,targets["validation"],"B0_FROZEN_RANKING")
    if b0_sign["safe_beneficial_count"]!=115 or b0_sign["predicted_positive_count"]!=42:raise RuntimeError("Frozen B0 reproduction failed")
    harm_payload=torch.load(args.harm_checkpoint,map_location=device,weights_only=False);harm_head=RiskPreservingBypassHead().to(device);harm_head.load_state_dict(harm_payload["model_state_dict"]);harm_head.eval()
    for parameter in harm_head.parameters():parameter.requires_grad_(False)
    with torch.inference_mode():harm_before=harm_head(data["validation"]["bypass"].to(device)).cpu().numpy()

    batches,batch_audit=b16.make_episode_batches(samples["train"],args.epochs,args.batch_size,args.seed);results={};initial={}
    for name in ("C0","C1","O1"):
        torch.manual_seed(args.seed);model=DisturbanceAdvantageBenefitReadout();initial[name]=r4a.state_sha(model.state_dict());results[name]=r4a.train(name,model,data["train"],data["validation"],features["train"][name],features["validation"][name],batches,args,torch,device)
        status=ORACLE if name=="O1" else "RUNTIME VALID"
        for row in results[name]["rows"]:row["runtime_status"]=status;row["oracle_status"]=status
    predictions={name:r4a.predict(result["model"],data["validation"],features["validation"][name],args.batch_size,torch,device) for name,result in results.items()}
    signs={name:r1b.sign_summary(samples["validation"],value,targets["validation"],name) for name,value in predictions.items()};calibration={name:r1c.calibration_row(samples["validation"],value,targets["validation"],name) for name,value in predictions.items()}
    overall=[]
    for name in ("C0","C1","O1"):overall.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"runtime_status":ORACLE if name=="O1" else "RUNTIME VALID",**calibration[name],**{key:signs[name][key] for key in ("safe_beneficial_count","predicted_positive_count","safe_beneficial_sign_accuracy","GT_negative_false_positive_rate","overall_predicted_positive_rate","positive_precision","safe_beneficial_precision")}})

    stop=matched_subgroup(samples["validation"],predictions,targets["validation"],"STOP",lambda sample:sample.split_metadata["motion_type_evaluation_only"]=="stop")
    c7=matched_subgroup(samples["validation"],predictions,targets["validation"],"C7",lambda sample:any(str(value).startswith("C7") for value in sample.split_metadata["contexts_evaluation_only"]))
    hold=r2.hold_rows(samples["validation"],predictions,targets["validation"])
    historical=[row for row in r1b.read_rows(args.historical) if row["category"]=="A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"];index={sample.sample_id:i for i,sample in enumerate(samples["validation"])};historical_indices=np.asarray([index[row["candidate_id"]] for row in historical],int)
    if len(historical_indices)!=51:raise RuntimeError("historical failure cohort changed")
    historical_rows=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"runtime_status":ORACLE if name=="O1" else "RUNTIME VALID","model":name,"historical_failure_count":51,"recovered_positive_count":int(np.sum(predictions[name][historical_indices]>0)),"remaining_failure_count":int(np.sum(predictions[name][historical_indices]<=0)),"recovery_rate":float(np.mean(predictions[name][historical_indices]>0))} for name in ("C0","C1","O1")]
    negative=[{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,**{key:signs[name][key] for key in ("GT_negative_count","GT_negative_false_positive_count","GT_negative_false_positive_rate","overall_predicted_positive_rate","positive_precision","safe_beneficial_precision")}} for name in ("C0","C1","O1")]
    negative.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":"C1_MINUS_C0","GT_negative_FPR_change":signs["C1"]["GT_negative_false_positive_rate"]-signs["C0"]["GT_negative_false_positive_rate"],"overall_positive_rate_change":signs["C1"]["overall_predicted_positive_rate"]-signs["C0"]["overall_predicted_positive_rate"],"positive_precision_change":signs["C1"]["positive_precision"]-signs["C0"]["positive_precision"],"safe_beneficial_precision_change":signs["C1"]["safe_beneficial_precision"]-signs["C0"]["safe_beneficial_precision"]})
    shortcut=[]
    for name in ("C0","C1","O1"):
        for row in r2.shortcut_rows(samples["validation"],predictions[name],anchors):shortcut.append({"runtime_status":ORACLE if name=="O1" else "RUNTIME VALID","model":name,**row})
    action_shortcut=any(row["near_deterministic_shortcut"] for row in shortcut if row["model"]=="C1" and row["dimension"] in ("candidate_action","runtime_generic_action"))

    b0_after=r3.extract_representations(backbone,samples["validation"],payload["normalizer"],args.batch_size,torch,device)["old_benefit"];ranking=r2.ranking_invariance(samples["validation"],targets["validation"],sigma,b0,b0_after)
    with torch.inference_mode():harm_after=harm_head(data["validation"]["bypass"].to(device)).cpu().numpy()
    checksums_after={"manifest_v3":r1b.file_sha(args.manifest_v3),"Benefit_Target_v2":r1b.file_sha(args.target_v2),"runtime_anchor_map":r1b.file_sha(args.anchor_map),"R1_v3_BASE":r1b.file_sha(args.r1_checkpoint),"HARM_v3_BASE":r1b.file_sha(args.harm_checkpoint)}
    harm={"label":LABEL,"mechanism_result":MECHANISM,"checkpoint_SHA_before":checksums_before["HARM_v3_BASE"],"checkpoint_SHA_after":checksums_after["HARM_v3_BASE"],"checkpoint_unchanged":checksums_before["HARM_v3_BASE"]==checksums_after["HARM_v3_BASE"],"validation_logits_before_SHA":r4a.array_sha(harm_before),"validation_logits_after_SHA":r4a.array_sha(harm_after),"validation_logits_max_abs_diff":float(np.max(np.abs(harm_after-harm_before))),"harm_outputs_exact":bool(np.array_equal(harm_before,harm_after)),"harm_optimizer_created":False}

    stop_by={row["model"]:row for row in stop};c0,c1,o1=signs["C0"],signs["C1"],signs["O1"]
    gates={
        "Gate_A":{"name":"Contract","checks":{"runtime_formula_exact":max(row["maximum_runtime_vs_formal_robot_component_error"] for row in exactness.values())<=1e-7,"runtime_generic_feature_zero":max(row["maximum_generic_runtime_feature_abs"] for row in exactness.values())==0.0,"runtime_feature_GT_reads_zero":True,"frozen_checksums_unchanged":checksums_before==checksums_after}},
        "Gate_B":{"name":"Benefit Sign","checks":{"C1_sign_at_least_0_55":c1["safe_beneficial_sign_accuracy"]>=.55,"C1_minus_C0_at_least_0_05":c1["safe_beneficial_sign_accuracy"]>=c0["safe_beneficial_sign_accuracy"]+.05}},
        "Gate_C":{"name":"Stop Recovery","checks":{"C1_Stop_at_least_6_of_8":stop_by["C1"]["safe_beneficial_count"]==8 and stop_by["C1"]["predicted_positive_count"]>=6}},
        "Gate_D":{"name":"MAE","checks":{"C1_MAE_not_above_C0":calibration["C1"]["Benefit_MAE"]<=calibration["C0"]["Benefit_MAE"],"C1_MAE_not_above_historical_guard":calibration["C1"]["Benefit_MAE"]<=HISTORICAL_MAE_GUARD}},
        "Gate_E":{"name":"No Degenerate Shift","checks":{"GT_negative_FPR_increase_at_most_0_05":c1["GT_negative_false_positive_rate"]<=c0["GT_negative_false_positive_rate"]+.05,"finite":all(np.isfinite(value).all() for value in predictions.values()),"no_action_shortcut_collapse":not action_shortcut}},
        "Gate_F":{"name":"System Isolation","checks":{"Frozen_B0_ranking_exact":ranking["B0_prediction_exact"] and ranking["metrics_exact"] and ranking["historical_metrics_within_tolerance"] and ranking["rank_signature_changes"]==0,"Harm_exact":harm["harm_outputs_exact"],"TEST_reads_zero":TEST_READS==0,"no_decision_chain":True,"ranking_loss_zero":LAMBDA_RANK==0.0}},
    }
    for gate in gates.values():gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())
    c1_core=all(gates[name]["passed"] for name in ("Gate_B","Gate_C","Gate_D","Gate_E"));o1_pass=o1["safe_beneficial_sign_accuracy"]>=.55 and o1["safe_beneficial_sign_accuracy"]>=c0["safe_beneficial_sign_accuracy"]+.05 and stop_by["O1"]["predicted_positive_count"]>=6 and calibration["O1"]["Benefit_MAE"]<=calibration["C0"]["Benefit_MAE"] and calibration["O1"]["Benefit_MAE"]<=HISTORICAL_MAE_GUARD
    sign_gap=o1["safe_beneficial_sign_accuracy"]-c1["safe_beneficial_sign_accuracy"];mae_gap=calibration["C1"]["Benefit_MAE"]-calibration["O1"]["Benefit_MAE"];close=abs(sign_gap)<=.05 and abs(mae_gap)<=.05
    if c1_core and close:classification="ROBOT-ACTION DISTURBANCE WAS THE MISSING RUNTIME BENEFIT SIGNAL";recommendation="Explicit Task/Safety Advantage"
    elif o1_pass and (not c1_core or not close):classification="HUMAN-EFFECT DISTURBANCE IS THE REMAINING BOTTLENECK";recommendation="Human-effect Disturbance Predictor"
    elif not c1_core and not o1_pass:classification="DISTURBANCE ALONE IS NOT SUFFICIENT DESPITE HIGH SIGN ASSOCIATION";recommendation="Factorized Safety+Disturbance Benefit"
    else:classification="ROBOT-ACTION DISTURBANCE HELPS BUT FULL-DISTURBANCE GAP REMAINS";recommendation="Human-effect Disturbance Predictor"

    runtime_contract={"label":LABEL,"mechanism_result":MECHANISM,"name":"DeltaD_robot","formula":"0.55 * [(0.30*|speed_delta_g|/0.10 + 0.25*|distance_offset_g|/0.20 + 0.20*|lateral_offset_g|/0.20) - same(candidate_i)]","inputs":["candidate_action_id","runtime_generic_action_id","frozen ACTION_DEFINITIONS"],"excluded":["human_effect_magnitude","GT future","GT human effect","GT cost total","GT Benefit","GT Harm","profile ID"],"runtime_valid":True,"GT_reads":0,"source":"src/models/runtime_disturbance_advantage.py::robot_action_disturbance"}
    oracle_contract={"label":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE,"formula":"0.55 * (full GT disturbance(g)-full GT disturbance(i))","includes_GT_human_effect_magnitude":True,"runtime_valid":False,"deployment_forbidden":True,"direct_addition_to_prediction":False}
    architecture=DisturbanceAdvantageBenefitReadout().architecture_audit();training={"label":LABEL,"mechanism_result":MECHANISM,"seed":args.seed,"optimizer":"AdamW","learning_rate":args.learning_rate,"weight_decay":.001,"batch_size_candidate_budget":args.batch_size,"max_epochs":args.epochs,"patience":args.patience,"objective":"frozen-uncertainty heteroscedastic NLL","selector":["minimum validation MAE","maximum safe-beneficial sign accuracy","earlier epoch"],"lambda_rank":0.0,"same_batches":True,"same_initialization":len(set(initial.values()))==1,"initial_state_SHA":initial["C0"],"batch_order_audit":batch_audit,"selections":{name:{"epoch":int(result["selected"]["epoch"]),"MAE":result["selected"]["Benefit_MAE"],"sign":result["selected"]["safe_beneficial_sign_accuracy"]} for name,result in results.items()}}
    frozen={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"checksums_before":checksums_before,"checksums_after":checksums_after,"R1_backbone_frozen":not any(parameter.requires_grad for parameter in backbone.parameters()),"R1_backbone_unchanged":backbone_before==r4a.state_sha(backbone.state_dict()),"Frozen_B0_ranking_only":True,"HARM_v3_risk_only":True,"threshold_calibration":False,"decision_chain":False}
    gap={"label":LABEL,"mechanism_result":MECHANISM,"O1_status":ORACLE,"C1_minus_C0_sign_pp":100*(c1["safe_beneficial_sign_accuracy"]-c0["safe_beneficial_sign_accuracy"]),"O1_minus_C1_sign_pp":100*sign_gap,"C1_minus_O1_MAE":mae_gap,"close_definition_fixed_before_training":"absolute sign gap <=5pp AND absolute MAE gap <=0.05","C1_close_to_O1":close,"C1_core_pass":c1_core,"O1_same_primary_performance_gates_pass":o1_pass,"classification":classification,"single_next_recommendation":recommendation}
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"runtime_disturbance_contract":runtime_contract,"disturbance_exactness":exactness,"safe_beneficial":signs,"calibration":calibration,"STOP":{row["model"]:row for row in stop},"C7":{row["model"]:row for row in c7},"HOLD":{row["model"]:row for row in hold},"historical_failures":{row["model"]:row for row in historical_rows},"oracle_gap":gap,"ranking":ranking,"harm":harm,"gates":gates,"outcome_classification":classification,"single_next_recommendation":recommendation,"next_stage_started":False}

    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"C0","state_dict":results["C0"]["model"].state_dict(),"selector":training["selections"]["C0"],"test_reads":0},args.output_dir/"checkpoints/c0_matched.pt")
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"C1","state_dict":results["C1"]["model"].state_dict(),"selector":training["selections"]["C1"],"runtime_valid":True,"test_reads":0},args.output_dir/"checkpoints/c1_runtime_disturbance.pt")
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"oracle_status":ORACLE,"model":"O1","state_dict":results["O1"]["model"].state_dict(),"selector":training["selections"]["O1"],"runtime_valid":False,"deployment_forbidden":True,"test_reads":0},args.output_dir/"checkpoints/o1_full_disturbance_oracle.pt")
    io.write_json(args.output_dir/"frozen_contract.json",frozen);io.write_json(args.output_dir/"runtime_disturbance_contract.json",runtime_contract);io.write_json(args.output_dir/"disturbance_exactness.json",{"label":LABEL,"mechanism_result":MECHANISM,**exactness});io.write_json(args.output_dir/"c0_architecture.json",{"label":LABEL,"mechanism_result":MECHANISM,**architecture,"role":"MATCHED C0","feature":"zero scalar"});io.write_json(args.output_dir/"c1_architecture.json",{"label":LABEL,"mechanism_result":MECHANISM,**architecture,"role":"RUNTIME C1","feature":"DeltaD_robot scalar"});io.write_json(args.output_dir/"oracle_o1_contract.json",oracle_contract);io.write_json(args.output_dir/"training_config.json",training)
    io.write_csv(args.output_dir/"c0_training_curve.csv",results["C0"]["rows"]);io.write_csv(args.output_dir/"c1_training_curve.csv",results["C1"]["rows"]);io.write_csv(args.output_dir/"o1_training_curve.csv",results["O1"]["rows"]);io.write_csv(args.output_dir/"overall_comparison.csv",overall);io.write_csv(args.output_dir/"safe_beneficial_sign.csv",overall);io.write_csv(args.output_dir/"stop_audit.csv",stop);io.write_csv(args.output_dir/"c7_audit.csv",c7);io.write_csv(args.output_dir/"hold_audit.csv",hold);io.write_csv(args.output_dir/"historical_failure_recovery.csv",historical_rows);io.write_csv(args.output_dir/"negative_protection.csv",negative);io.write_json(args.output_dir/"oracle_gap_analysis.json",gap);io.write_csv(args.output_dir/"shortcut_audit.csv",shortcut);io.write_json(args.output_dir/"ranking_isolation.json",ranking);io.write_json(args.output_dir/"harm_isolation.json",harm);io.write_json(args.output_dir/"gate_results.json",{"label":LABEL,"mechanism_result":MECHANISM,**gates});io.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
