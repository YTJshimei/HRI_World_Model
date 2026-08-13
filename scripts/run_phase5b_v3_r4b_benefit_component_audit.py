"""Phase 5B-v3-R4B Benefit Cost-Component Mechanism Audit (BCCMA).

This is a GT-only target-mechanism audit.  It performs zero optimization,
zero backward calls, zero TEST reads, and does not load or alter the ranking or
risk models.  GT cost components are never proposed as runtime features.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
from src.data.adverse_response_dataset import GENERATOR_SEED,RISK_SEED,POPULATION_PROFILE,build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.decision.decision_cost import DecisionCostWeights,compute_decision_costs
from src.evaluation.benefit_component_audit import COMPONENTS,episode_cost_components
from src.multimodal.temporal_schema import LABEL

MECHANISM="DEVELOPMENT MECHANISM RESULT"
DIAGNOSTIC="GT COST COMPONENT DIAGNOSTIC - NOT RUNTIME VALID"
STAGE="Phase 5B-v3-R4B Benefit Cost-Component Mechanism Audit"
TEST_READS=0
OPTIMIZER_STEPS=0
BACKWARD_CALLS=0
SIGN_TOLERANCE=1e-6


def half_float32_ulp(value):
    """Maximum round-to-nearest error of a value serialized as float32."""
    return 0.5*abs(float(np.spacing(np.float32(value))))


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-v3",type=Path,default=PROJECT_ROOT/"results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--target-v2",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv")
    parser.add_argument("--anchor-map",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv")
    parser.add_argument("--r1-checkpoint",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--historical",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r1a_runtime_anchor_realign/historical_sign_failure_reclassification.csv")
    parser.add_argument("--ranking-isolation",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r4a_oracle_future_pose_sufficiency/ranking_isolation.json")
    parser.add_argument("--harm-isolation",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r4a_oracle_future_pose_sufficiency/harm_isolation.json")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev/phase5b_v3_r4b_benefit_component_audit")
    return parser.parse_args()


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cliffs_delta(left,right):
    left=np.asarray(left,float);right=np.asarray(right,float)
    return float(np.mean((left[:,None]>right[None,:]).astype(float)-(left[:,None]<right[None,:]).astype(float)))


def overlap_coefficient(left,right,bins=40):
    left=np.asarray(left,float);right=np.asarray(right,float);low=min(left.min(),right.min());high=max(left.max(),right.max())
    if high-low<1e-12:return 1.0
    edges=np.linspace(low,high,bins+1);a,_=np.histogram(left,edges,density=True);b,_=np.histogram(right,edges,density=True);width=np.diff(edges)
    return float(np.sum(np.minimum(a,b)*width))


def contribution_stats(values,total):
    values=np.asarray(values,float);total=np.asarray(total,float);absolute=np.abs(values)
    return {"count":int(len(values)),"mean_signed_contribution":float(values.mean()),"median_signed_contribution":float(np.median(values)),"mean_absolute_contribution":float(absolute.mean()),"P90_absolute_contribution":float(np.percentile(absolute,90)),"fraction_positive":float(np.mean(values>0)),"fraction_negative":float(np.mean(values<0)),"absolute_contribution_share":float(absolute.sum()/max(np.sum(np.abs(total)),1e-12))}


def group_rows(contributions,benefit,mask,group):
    return [{"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"group":group,"component":name,**contribution_stats(contributions[name][mask],benefit[mask])} for name in COMPONENTS]


def dominant_label(row):
    values=np.asarray([abs(row[name]) for name in COMPONENTS]);order=np.argsort(-values);total=values.sum()
    if total<=1e-12:return "NO_NONZERO_COMPONENT"
    if values[order[0]]/total>=.40:return COMPONENTS[order[0]].upper()
    if (values[order[0]]+values[order[1]])/total>=.70:return "+".join(sorted((COMPONENTS[order[0]].upper(),COMPONENTS[order[1]].upper())))
    return "DIFFUSE_MULTI_COMPONENT"


def positive_mechanism_label(row):
    values=np.asarray([max(float(row[name]),0.0) for name in COMPONENTS]);order=np.argsort(-values);total=values.sum()
    if total<=1e-12:return "NO_POSITIVE_COMPONENT"
    if values[order[0]]/total>=.40:return COMPONENTS[order[0]].upper()
    if (values[order[0]]+values[order[1]])/total>=.70:return "+".join(sorted((COMPONENTS[order[0]].upper(),COMPONENTS[order[1]].upper())))
    return "DIFFUSE_POSITIVE_MULTI_COMPONENT"


def stable_stop_explanation(stop_rows):
    labels=[row["positive_mechanism_class"] for row in stop_rows];counts=Counter(labels);label,count=counts.most_common(1)[0]
    return {"stable_mechanism":label,"stable_count":count,"required_count":6,"passed":count>=6,"mechanism_counts":dict(counts)}


def subcomponent_contract():
    return {
        "human_response.effect_magnitude":{"formula":"0.30 * mean_(H,J) ||action_effect_xyz||","units":"weighted normalized skeleton displacement"},
        "human_response.speed_effect":{"formula":"0.25 * 10Hz * mean_(H-1) ||diff(root(action_effect)_xy)||","units":"weighted m/s response change"},
        "human_response.lateral_effect":{"formula":"0.20 * abs(final root(action_effect)_world_y)","units":"weighted metres; simulator world-y convention"},
        "human_response.heading_effect":{"formula":"0.25 * abs(wrapped final shoulder-line yaw change)","units":"weighted radians"},
        "disturbance.robot_speed_action":{"formula":"0.30 * abs(action speed_scale_delta)/0.10","semantics":"hand-designed robot action regularizer"},
        "disturbance.target_distance_action":{"formula":"0.25 * abs(action distance_offset_m)/0.20","semantics":"hand-designed robot action regularizer"},
        "disturbance.lateral_action":{"formula":"0.20 * abs(action lateral_offset_m)/0.20","semantics":"hand-designed robot action regularizer"},
        "disturbance.human_effect_magnitude":{"formula":"0.25 * effect_magnitude/0.05","semantics":"human-response magnitude regularizer"},
    }


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite R4B: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    checksums_before,labels,anchors=r1b.load_contract(args)
    episodes=build_development_split("validation",240,GENERATOR_SEED+1000,RISK_SEED+1000)
    label_by_id={candidate_id:row for candidate_id,row in labels.items() if row["split"]=="validation"}
    rows=[];sub_rows=[];maximum_stored_cost_error=0.0;maximum_stored_cost_error_ratio=0.0
    weights=DecisionCostWeights();weight_map={name:getattr(weights,name) for name in COMPONENTS};weight_map["uncertainty_effective"]=0.0
    for episode in episodes:
        audit=episode_cost_components(episode,POPULATION_PROFILE);actions=audit["action_ids"];anchor_action=int(anchors[episode.episode_id]["runtime_anchor_action_id"]);anchor_index=int(np.flatnonzero(actions==anchor_action)[0])
        stored_error=np.abs(audit["components"]["total"]-audit["stored_totals"])
        stored_bounds=np.asarray([half_float32_ulp(value) for value in audit["stored_totals"]])
        maximum_stored_cost_error=max(maximum_stored_cost_error,float(stored_error.max()))
        maximum_stored_cost_error_ratio=max(maximum_stored_cost_error_ratio,float(np.max(stored_error/np.maximum(stored_bounds,np.finfo(float).tiny))))
        weighted={name:weight_map[name]*(audit["components"][name][anchor_index]-audit["components"][name]) if name!="uncertainty" else np.zeros(len(actions)) for name in COMPONENTS}
        for index,action in enumerate(actions):
            sample_id=f"{episode.episode_id}:{int(action)}";label=label_by_id[sample_id];record={"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"episode_id":episode.episode_id,"candidate_id":sample_id,"candidate_action":int(action),"runtime_generic_action":anchor_action,"motion":episode.motion_type,"contexts":"|".join(episode.context_labels) or "NONE","profile_audit_only":episode.profile_id,"GT_Benefit":float(label["benefit_v2_runtime_anchor"]),"harm_v2":label["harm_v2_unchanged"]=="True","feasible":label["feasible_unchanged"]=="True"}
            for name in COMPONENTS:record[name]=float(weighted[name][index])
            record["Benefit_reconstructed"]=float(sum(record[name] for name in COMPONENTS));record["reconstruction_error"]=record["Benefit_reconstructed"]-record["GT_Benefit"]
            record["float32_serialization_bound"]=half_float32_ulp(audit["stored_totals"][anchor_index])+half_float32_ulp(audit["stored_totals"][index])
            record["normalized_reconstruction_error"]=abs(record["reconstruction_error"])/max(record["float32_serialization_bound"],np.finfo(float).tiny)
            record["dominant_positive_component"]=max(COMPONENTS,key=lambda name:record[name]);record["dominant_negative_component"]=min(COMPONENTS,key=lambda name:record[name]);record["mechanism_class"]=dominant_label(record);record["positive_mechanism_class"]=positive_mechanism_label(record);record["distance_to_zero"]=abs(record["GT_Benefit"]);rows.append(record)
            for subname,values in audit["subcomponents"].items():
                parent=subname.split(".")[0];effective=weight_map[parent] if parent!="uncertainty" else 0.0
                sub_rows.append({"candidate_id":sample_id,"episode_id":episode.episode_id,"candidate_action":int(action),"subcomponent":subname,"parent_component":parent,"weighted_contribution":float(effective*(values[anchor_index]-values[index]))})
    benefit=np.asarray([row["GT_Benefit"] for row in rows]);contributions={name:np.asarray([row[name] for row in rows]) for name in COMPONENTS};reconstruction=np.asarray([row["Benefit_reconstructed"] for row in rows]);max_error=float(np.max(np.abs(reconstruction-benefit)));max_normalized_error=max(row["normalized_reconstruction_error"] for row in rows);max_row_tolerance=max(row["float32_serialization_bound"] for row in rows)
    if max_normalized_error>1.0 or maximum_stored_cost_error_ratio>1.0:raise RuntimeError(f"Benefit exact reconstruction exceeds source float32 serialization: ratio={max_normalized_error}")

    safe=np.asarray([row["GT_Benefit"]>SIGN_TOLERANCE and row["feasible"] and not row["harm_v2"] for row in rows]);negative=benefit<-SIGN_TOLERANCE
    stop=safe&np.asarray([row["motion"]=="stop" for row in rows]);c7=safe&np.asarray(["C7" in row["contexts"] for row in rows]);hold=np.asarray([row["candidate_action"]==HOLD_ACTION_ID for row in rows]);beneficial=benefit>SIGN_TOLERANCE
    if safe.sum()!=115 or stop.sum()!=8 or c7.sum()!=38 or (hold&beneficial).sum()!=10 or (hold&safe).sum()!=4 or (hold&~beneficial).sum()!=230:raise RuntimeError("frozen subgroup counts changed")
    overall=group_rows(contributions,benefit,np.ones(len(rows),bool),"ALL_VALIDATION");safe_rows=group_rows(contributions,benefit,safe,"SAFE_BENEFICIAL");negative_rows=group_rows(contributions,benefit,negative,"GT_NEGATIVE")

    stop_cases=[]
    for index in np.flatnonzero(stop):
        row=rows[index];stop_cases.append({key:row[key] for key in ("synthetic_interaction","mechanism_result","diagnostic_status","episode_id","candidate_action","runtime_generic_action","GT_Benefit",*COMPONENTS,"dominant_positive_component","dominant_negative_component","mechanism_class","positive_mechanism_class","distance_to_zero")})
    stop_summary={"label":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"safe_beneficial_Stop_count":8,"component_summary":{name:contribution_stats(contributions[name][stop],benefit[stop]) for name in COMPONENTS},"dominance_frequency":dict(Counter(row["dominant_positive_component"] for row in stop_cases)),**stable_stop_explanation(stop_cases)}

    c7_rows=[]
    for group,mask in (("C7_SAFE_BENEFICIAL",c7),("NON_C7_SAFE_BENEFICIAL",safe&~c7)):c7_rows.extend(group_rows(contributions,benefit,mask,group))
    hold_rows=[]
    for group,mask in (("BENEFICIAL_HOLD",hold&beneficial),("SAFE_BENEFICIAL_HOLD",hold&safe),("NON_BENEFICIAL_HOLD",hold&~beneficial)):hold_rows.extend(group_rows(contributions,benefit,mask,group))

    global_order=sorted(COMPONENTS,key=lambda name:-np.mean(np.abs(contributions[name])));top2=global_order[:2];single=[]
    for name in COMPONENTS:
        value=contributions[name];single.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"feature":name,"definition":"sign(weighted anchor-minus-candidate component contribution)","overall_sign_agreement":float(np.mean(np.sign(value)==np.sign(benefit))),"safe_beneficial_positive_rate":float(np.mean(value[safe]>0)),"Stop_positive_rate":float(np.mean(value[stop]>0)),"C7_positive_rate":float(np.mean(value[c7]>0))})
    top_value=contributions[top2[0]]+contributions[top2[1]];single.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"feature":"TOP2_GLOBAL_MEAN_ABS:"+"+".join(top2),"definition":"predefined by global validation mean absolute contribution, not sign performance","overall_sign_agreement":float(np.mean(np.sign(top_value)==np.sign(benefit))),"safe_beneficial_positive_rate":float(np.mean(top_value[safe]>0)),"Stop_positive_rate":float(np.mean(top_value[stop]>0)),"C7_positive_rate":float(np.mean(top_value[c7]>0))})

    loo=[]
    for name in COMPONENTS:
        value=benefit-contributions[name];loo.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"removed_component":name,"safe_beneficial_sign_preservation":float(np.mean(value[safe]>0)),"overall_sign_agreement":float(np.mean(np.sign(value)==np.sign(benefit))),"Stop_sign_preservation":float(np.mean(value[stop]>0)),"C7_sign_preservation":float(np.mean(value[c7]>0)),"safe_beneficial_flipped_count":int(np.sum(value[safe]<=0))})
    zero=[]
    for name in COMPONENTS:
        critical=safe&(benefit-contributions[name]<=0)
        zero.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"attribution":name.upper()+"_CRITICAL","candidate_count":int(critical.sum()),"fraction_of_safe_beneficial":float(critical.sum()/safe.sum())})
    critical_matrix=np.column_stack([safe&(benefit-contributions[name]<=0) for name in COMPONENTS]);zero.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"attribution":"MULTI_COMPONENT","candidate_count":int(np.sum(safe&(critical_matrix.sum(axis=1)>1))),"fraction_of_safe_beneficial":float(np.sum(safe&(critical_matrix.sum(axis=1)>1))/safe.sum())})

    historical=[row for row in r1b.read_rows(args.historical) if row["category"]=="A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"]
    historical_ids={row["candidate_id"] for row in historical};failure=np.asarray([row["candidate_id"] in historical_ids for row in rows]);correct=safe&~failure
    if failure.sum()!=51 or np.any(failure&~safe):
        raise RuntimeError("frozen historical-failure cohort is no longer exactly 51 safe-beneficial candidates")
    margin_rows=[]
    for group,mask in (("ALL_SAFE_BENEFICIAL",safe),("HISTORICAL_51_FAILURES",failure),("OTHER_SAFE_BENEFICIAL",correct)):
        values=benefit[mask];margin_rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"group":group,"count":int(mask.sum()),**{f"P{p}":float(np.percentile(values,p)) for p in (10,25,50,75,90)},"near_zero_le_0_10_rate":float(np.mean(values<=.10)),"near_zero_le_0_25_rate":float(np.mean(values<=.25))})
    historical_rows=[];clear_differences=[]
    for name in COMPONENTS:
        left,right=contributions[name][failure],contributions[name][correct];effect=cliffs_delta(left,right);overlap=overlap_coefficient(left,right);median_shift=float(np.median(left)-np.median(right));clear=abs(effect)>=.33 or overlap<=.70
        historical_rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"component":name,"failure_count":int(failure.sum()),"correct_count":int(correct.sum()),"failure_mean":float(left.mean()),"correct_mean":float(right.mean()),"failure_median":float(np.median(left)),"correct_median":float(np.median(right)),"median_shift":median_shift,"Cliffs_delta":effect,"distribution_overlap":overlap,"clear_mechanism_difference":clear});clear_differences.append(clear)

    sub_by_name=defaultdict(list)
    for row in sub_rows:sub_by_name[row["subcomponent"]].append(row)
    human_rows=[];disturbance_rows=[];contracts=subcomponent_contract()
    for name,selected in sub_by_name.items():
        if not name.startswith(("human_response.","disturbance.")):continue
        by_id={row["candidate_id"]:row for row in selected};value=np.asarray([by_id[row["candidate_id"]]["weighted_contribution"] for row in rows]);output={"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"subcomponent":name,"parent_component":name.split(".")[0],**contracts[name],**contribution_stats(value,benefit)}
        (human_rows if name.startswith("human_response.") else disturbance_rows).append(output)

    observability={"label":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"components":{
        "task":{"classification":"DERIVABLE_FROM_PREDICTED_FUTURE","reason":"requires future human-robot distance; target distance and current distance are directly observed"},
        "safety":{"classification":"DERIVABLE_FROM_PREDICTED_FUTURE","reason":"requires predicted minimum distance/duration plus directly known feasibility and safety distance"},
        "human_response":{"classification":"DERIVABLE_FROM_PREDICTED_FUTURE","reason":"requires action-conditioned skeleton/action-effect future"},
        "disturbance":{"classification":"DERIVABLE_FROM_PREDICTED_FUTURE","reason":"robot action terms are directly observable; human-effect magnitude requires predicted future response"},
        "uncertainty":{"classification":"DERIVABLE_FROM_PREDICTED_FUTURE","reason":"formal Target-v2 disables this component, but uncertainty would come from predictive distribution"}},"GT_component_values_runtime_inputs":False}

    identifiable=np.asarray([row["mechanism_class"]!="DIFFUSE_MULTI_COMPONENT" for row in rows]);stop_stable=stop_summary["passed"]
    positive_abs={name:float(np.mean(np.maximum(contributions[name][safe],0))) for name in COMPONENTS};sorted_positive=sorted(COMPONENTS,key=lambda name:-positive_abs[name]);negative_mean={name:float(np.mean(np.minimum(contributions[name][safe],0))) for name in COMPONENTS}
    dominant={"dominant_positive_component":sorted_positive[0],"second_dominant_positive_component":sorted_positive[1],"dominant_negative_component":min(COMPONENTS,key=lambda name:negative_mean[name])}
    safe_mechanism_counts=Counter(rows[index]["positive_mechanism_class"] for index in np.flatnonzero(safe))
    prevailing_mechanism,prevailing_count=safe_mechanism_counts.most_common(1)[0]
    major_components=set(prevailing_mechanism.lower().split("+")) if "DIFFUSE" not in prevailing_mechanism else set(sorted_positive[:2])
    bridgeable_classes={"DIRECT_RUNTIME_OBSERVABLE","DERIVABLE_FROM_PREDICTED_FUTURE","DIRECT + DERIVABLE_FROM_PREDICTED_FUTURE"}
    observability["major_drivers"] = sorted(major_components)
    observability["major_drivers_runtime_bridgeable"] = any(observability["components"][name]["classification"] in bridgeable_classes for name in major_components)
    checksums_after={"manifest_v3":file_sha(args.manifest_v3),"Benefit_Target_v2":file_sha(args.target_v2),"runtime_anchor_map":file_sha(args.anchor_map),"R1_v3_BASE":file_sha(args.r1_checkpoint),"HARM_v3_BASE":file_sha(args.harm_checkpoint)}
    ranking=json.loads(args.ranking_isolation.read_text(encoding="utf-8"));harm=json.loads(args.harm_isolation.read_text(encoding="utf-8"))
    gates={
        "Gate_A":{"name":"Contract Reconstruction","checks":{"all_rows_within_source_float32_serialization_bound":max_normalized_error<=1.0,"stored_GT_cost_replay_within_source_float32_serialization_bound":maximum_stored_cost_error_ratio<=1.0,"frozen_contract_checksums_unchanged":checksums_after==checksums_before}},
        "Gate_B":{"name":"Mechanism Identifiability","checks":{"identified_fraction_at_least_0_80":float(np.mean(identifiable[safe]))>=.80}},
        "Gate_C":{"name":"Stop Attribution","checks":{"stable_component_or_pair_at_least_6_of_8":stop_stable}},
        "Gate_D":{"name":"Historical Failure Attribution","descriptive_rule_fixed_before_audit":"abs(Cliff's delta) >= 0.33 OR histogram overlap <= 0.70","checks":{"at_least_one_clear_component_distribution_difference":any(clear_differences)}},
        "Gate_E":{"name":"Runtime Bridgability","checks":{"major_Benefit_driver_runtime_observable_or_predictable":observability["major_drivers_runtime_bridgeable"]}},
        "Gate_F":{"name":"System Isolation","checks":{"optimizer_steps_zero":OPTIMIZER_STEPS==0,"backward_calls_zero":BACKWARD_CALLS==0,"TEST_reads_zero":TEST_READS==0,"Frozen_B0_ranking_unchanged":ranking["B0_prediction_exact"] and ranking["metrics_exact"] and ranking["rank_signature_changes"]==0,"Harm_unchanged":harm["harm_outputs_exact"],"no_decision_chain_run":True}},
    }
    for gate in gates.values():gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())

    # This recommendation rule is fixed before viewing R4B outputs. A mechanism
    # must explain a majority of safe-beneficial cases before a single-factor
    # representation is recommended; otherwise the result is factorized.
    prevailing_fraction=float(prevailing_count/safe.sum())
    if not observability["major_drivers_runtime_bridgeable"]:
        recommendation="Target Redesign Audit"
    elif prevailing_fraction<.50 or "+" in prevailing_mechanism or "DIFFUSE" in prevailing_mechanism:
        recommendation="Factorized Benefit Model"
    elif prevailing_mechanism=="HUMAN_RESPONSE":
        recommendation="Semantic Human-Response Representation"
    elif prevailing_mechanism in ("TASK","SAFETY","DISTURBANCE"):
        recommendation="Explicit Task-Advantage Representation"
    else:
        recommendation="Factorized Benefit Model"

    source=inspect.getsource(compute_decision_costs);contract={"label":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"target":"BENEFIT_TARGET_V2_RUNTIME_ANCHORED","formula":"GT_total_cost(runtime canonical generic) - GT_total_cost(candidate)","components":[
        {"name":"task","weight":1.0,"raw_formula":"final distance error + 0.35 mean distance error + 0.45 progress failure + 0.25 visibility proxy","data_source":"formal GT candidate rollout future_human_robot_distance","units":"weighted distance penalty (metres)","runtime_availability":"DERIVABLE_FROM_PREDICTED_FUTURE","GT_only_in_this_audit":True},
        {"name":"safety","weight":3.0,"raw_formula":"5 violation_proxy + 8 unsafe_duration + 10 close_gap + infeasible*1e4","data_source":"formal GT candidate distance + candidate feasibility","units":"dimensionless safety penalty before outer weight","runtime_availability":"DERIVABLE_FROM_PREDICTED_FUTURE","GT_only_in_this_audit":True,"not_equivalent_to_HARM_v3":True},
        {"name":"human_response","weight":1.4,"raw_formula":"0.30 effect magnitude + 0.25 speed effect + 0.20 lateral effect + 0.25 heading effect","data_source":"formal GT action_effect and natural future","units":"mixed normalized response penalty","runtime_availability":"DERIVABLE_FROM_PREDICTED_FUTURE","GT_only_in_this_audit":True},
        {"name":"disturbance","weight":0.55,"raw_formula":"0.30 robot speed action/0.10 + 0.25 target change/0.20 + 0.20 lateral action/0.20 + 0.25 effect magnitude/0.05","data_source":"known candidate action semantics + formal GT human action effect","units":"dimensionless regularizer","runtime_availability":"DIRECT + DERIVABLE_FROM_PREDICTED_FUTURE","GT_only_in_this_audit":True},
        {"name":"uncertainty","declared_weight":0.85,"effective_weight_in_Target_v2":0.0,"raw_formula":"coordinate uncertainty / 0.05; include_uncertainty=False in formal GT cost generation","data_source":"prediction uncertainty (zero in GT rollout)","units":"dimensionless","runtime_availability":"DERIVABLE_FROM_PREDICTED_FUTURE","GT_only_in_this_audit":True}],"official_source":"src/decision/decision_cost.py::compute_decision_costs","official_source_SHA256":hashlib.sha256(source.encode()).hexdigest(),"components_added_by_audit":False,"negative_Benefit_is_Harm":False}
    exact={"label":LABEL,"mechanism_result":MECHANISM,"diagnostic_status":DIAGNOSTIC,"candidate_count":len(rows),"maximum_absolute_reconstruction_error":max_error,"maximum_row_numerical_tolerance":max_row_tolerance,"maximum_error_to_tolerance_ratio":max_normalized_error,"maximum_replayed_GT_total_cost_vs_stored_error":maximum_stored_cost_error,"maximum_replayed_cost_error_to_half_ULP_ratio":maximum_stored_cost_error_ratio,"numerical_tolerance_policy":"per-row sum of half-ULP bounds from frozen float32 anchor/candidate total-cost serialization; fixed from source dtype, not fitted to results","exact_within_tolerance":max_normalized_error<=1.0 and maximum_stored_cost_error_ratio<=1.0,"formula":"sum weighted component anchor-minus-candidate deltas","TEST_reads":0}
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"diagnostic_status":DIAGNOSTIC,"test_reads":0,"optimizer_steps":0,"backward_calls":0,"exact_reconstruction":exact,"safe_beneficial_dominance":dominant,"safe_identified_fraction":float(np.mean(identifiable[safe])),"safe_positive_mechanism_counts":dict(safe_mechanism_counts),"prevailing_positive_mechanism":{"label":prevailing_mechanism,"count":prevailing_count,"fraction":prevailing_fraction},"Stop":stop_summary,"historical_clear_difference_components":[row["component"] for row in historical_rows if row["clear_mechanism_difference"]],"historical_margin":{row["group"]:row for row in margin_rows},"runtime_observability":observability,"gates":gates,"single_next_recommendation":recommendation,"next_stage_started":False}

    io.write_json(args.output_dir/"benefit_cost_contract.json",contract);io.write_json(args.output_dir/"exact_reconstruction.json",exact);io.write_csv(args.output_dir/"overall_component_contributions.csv",overall);io.write_csv(args.output_dir/"safe_beneficial_components.csv",safe_rows);io.write_csv(args.output_dir/"negative_components.csv",negative_rows);io.write_csv(args.output_dir/"stop_component_cases.csv",stop_cases);io.write_json(args.output_dir/"stop_component_summary.json",stop_summary);io.write_csv(args.output_dir/"c7_component_audit.csv",c7_rows);io.write_csv(args.output_dir/"hold_component_audit.csv",hold_rows);io.write_csv(args.output_dir/"single_component_sign.csv",single);io.write_csv(args.output_dir/"leave_one_component_out.csv",loo);io.write_csv(args.output_dir/"zero_crossing_attribution.csv",zero);io.write_csv(args.output_dir/"benefit_margin_audit.csv",margin_rows);io.write_csv(args.output_dir/"historical_failure_components.csv",historical_rows);io.write_json(args.output_dir/"runtime_observability.json",observability);io.write_csv(args.output_dir/"human_response_subcomponents.csv",human_rows);io.write_csv(args.output_dir/"disturbance_subcomponents.csv",disturbance_rows);io.write_json(args.output_dir/"gate_results.json",{"label":LABEL,"mechanism_result":MECHANISM,**gates});io.write_json(args.output_dir/"summary.json",summary)
    checksums_final={"manifest_v3":file_sha(args.manifest_v3),"Benefit_Target_v2":file_sha(args.target_v2),"runtime_anchor_map":file_sha(args.anchor_map),"R1_v3_BASE":file_sha(args.r1_checkpoint),"HARM_v3_BASE":file_sha(args.harm_checkpoint)}
    if checksums_final!=checksums_before:raise RuntimeError("frozen asset changed during R4B audit")
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
