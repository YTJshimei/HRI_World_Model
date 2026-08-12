"""Phase 4C.3 selective personalization (offline synthetic interaction only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--belief-samples",type=int,choices=(16,32),default=16)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--phase4b6-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4b6")
    parser.add_argument("--phase4c-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c")
    parser.add_argument("--phase4c1-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c1")
    parser.add_argument("--phase4c2-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c2")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c3")
    return parser.parse_args()


def clean(value:Any)->Any:
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path:Path,value:Any)->None:path.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding="utf-8")
def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or ("empty",));writer.writeheader()
        for row in rows:writer.writerow({field:clean(row.get(field,"")) for field in fields})


@dataclass(frozen=True)
class ArbitrationConfig:
    minimum_confidence:float
    minimum_margin:float
    minimum_benefit_probability:float


@dataclass
class Episode:
    key:tuple[str,int]
    indices:tuple[int,...]
    first:dict[str,Any]
    artifact:Any
    safety_prediction:dict[str,np.ndarray]
    feasible:np.ndarray
    rejection_reasons:tuple[str,...]
    confidence:Any
    action_confidence:np.ndarray
    theta_shrunk:np.ndarray
    generic_rollout:Any
    shrunk_rollout:Any
    generic_costs:Any
    personal_costs:Any
    shrunk_costs:Any
    gt_costs:Any
    switch_features:np.ndarray
    rule_index:int
    no_uncertainty_index:int
    oracle_functional_costs:Any
    benefit_probability:np.ndarray|None=None


def load_frozen_phase4c2(args:argparse.Namespace,torch:Any)->tuple[Any,np.ndarray,Any]:
    from src.decision.root_belief import RootResidualBeliefHead
    from src.decision.safety_calibration import SafetyCalibration,SafetyResidualHead
    summary=json.loads((args.phase4c2_dir/"summary.json").read_text(encoding="utf-8"))
    root_path=(args.phase4c2_dir/"checkpoints"/"root_belief_best.pt").resolve()
    expected_root=(PROJECT_ROOT/"results_dev"/"phase4c2"/"checkpoints"/"root_belief_best.pt").resolve()
    if root_path != expected_root:
        raise ValueError("Phase 4C.3 only loads the locally generated frozen Phase 4C.2 root checkpoint")
    # Phase 4C.2 stored sigma_scale as a NumPy array, which the PyTorch 2.6+
    # weights-only unpickler rejects.  This fixed, locally generated checkpoint
    # is trusted; arbitrary user-provided checkpoint paths are not accepted.
    root_checkpoint=torch.load(root_path,map_location="cpu",weights_only=False)
    root=RootResidualBeliefHead();root.load_state_dict(root_checkpoint["model_state_dict"]);root.to(args.device).eval()
    safety_checkpoint=torch.load(args.phase4c2_dir/"checkpoints"/"safety_residual_best.pt",map_location="cpu",weights_only=True)
    safety=SafetyResidualHead();safety.load_state_dict(safety_checkpoint["model_state_dict"]);safety.to(args.device).eval()
    calibration=SafetyCalibration(**summary["safety_calibration"])
    return root,np.asarray(root_checkpoint["sigma_scale"],np.float32),safety,calibration,summary


def build_base(
    args:argparse.Namespace,records:list[dict[str,Any]],engine:Any,prior_mean:np.ndarray,prior_std:np.ndarray,
    root_head:Any,root_scale:np.ndarray,safety_head:Any,safety_calibration:Any,cost_calibration:Any|None,torch:Any,
)->tuple[list[Any],list[dict[str,Any]],Any]:
    import scripts.run_phase4c2_belief_selection as c2
    import scripts.run_phase4c1_safety_calibration as c1
    beliefs=c2.predict_root_beliefs(root_head,root_scale,records,torch.device(args.device),torch)
    artifacts=c2.make_artifacts(args,records,beliefs)
    raw=c1.predict_head(safety_head,records,torch.device(args.device),torch)
    predictions=c1.calibrated_predictions(raw,records,safety_calibration)
    if cost_calibration is None:
        x,predicted,truth=c2.calibration_arrays(artifacts)
        from src.decision.cost_calibration import fit_cost_residual_calibrator
        cost_calibration=fit_cost_residual_calibrator(x,predicted,truth,"train")
    return artifacts,predictions,cost_calibration


def episode_data(
    args:argparse.Namespace,records:list[dict[str,Any]],artifacts:list[Any],predictions:list[dict[str,Any]],
    cost_calibration:Any,engine:Any,prior_mean:np.ndarray,prior_std:np.ndarray,c2_config:Any,
)->list[Episode]:
    import scripts.run_phase4c_decision as c0
    import scripts.run_phase4c2_belief_selection as c2
    from src.data.functional_response_state import RESPONSE_STATE_SCALE
    from src.decision.decision_cost import DecisionCostWeights,compute_decision_costs
    from src.decision.action_selector import rule_based_select,select_model_action
    from src.decision.personalization_confidence import (
        action_personalization_confidence,compute_personalization_confidence,
        shrink_functional_state,support_masks_from_probe_ids,
    )
    weights=DecisionCostWeights();results=[]
    for artifact in artifacts:
        first=records[artifact.indices[0]];state=first["state"]
        safety_prediction=c2.candidate_prediction(artifact,records,predictions,cost_calibration)
        _,feasible,reasons,_=c2.select_prediction(artifact,safety_prediction,c2_config)
        masks=support_masks_from_probe_ids(first["support"])
        # Posterior precision gain relative to the population prior is an
        # observable information statistic from the Phase 4B.7 belief update.
        posterior=np.maximum(np.asarray(first["theta_std"],float),1e-5)
        prior=np.maximum(np.asarray(prior_std,float),1e-5)
        information=np.maximum(1.0/posterior**2-1.0/prior**2,0.0)*np.square(RESPONSE_STATE_SCALE)
        confidence=compute_personalization_confidence(
            posterior,prior,masks,artifact.root_belief.sigma_root,information,
        )
        theta_shrunk=shrink_functional_state(first["theta_hat"],prior_mean,confidence.dimension_confidence)
        generic_state=c0.make_state(first["sample_data"],prior_mean.astype(np.float32),prior_std.astype(np.float32))
        shrunk_state=c0.make_state(first["sample_data"],theta_shrunk,posterior.astype(np.float32))
        oracle_state=c0.make_state(first["sample_data"],np.asarray(first["theta_true"],np.float32),np.zeros(6,np.float32))
        generic_rollout=engine.rollout(generic_state,uncertainty_aware=True)
        shrunk_rollout=engine.rollout(shrunk_state,uncertainty_aware=True)
        oracle_rollout=engine.rollout(oracle_state,uncertainty_aware=False)
        generic_costs=compute_decision_costs(generic_state,generic_rollout,weights,include_uncertainty=False)
        shrunk_costs=compute_decision_costs(shrunk_state,shrunk_rollout,weights,include_uncertainty=False)
        oracle_functional_costs=compute_decision_costs(oracle_state,oracle_rollout,weights,include_uncertainty=False)
        rule_action=rule_based_select(state);rule_index=int(np.flatnonzero(artifact.gt_costs.action_ids==rule_action)[0])
        no_uncertainty_index=select_model_action(state,first["predicted_rollout"],weights,use_uncertainty=False).selected_index
        action_confidence=np.asarray([action_personalization_confidence(action,confidence.dimension_confidence) for action in artifact.point_costs.action_ids])
        response_disagreement=np.linalg.norm(first["predicted_rollout"].predicted_action_effect-generic_rollout.predicted_action_effect,axis=-1).mean(axis=(1,2))
        distance_disagreement=np.abs(first["predicted_rollout"].predicted_human_robot_distance-generic_rollout.predicted_human_robot_distance).mean(axis=1)
        cost_disagreement=artifact.point_costs.total-generic_costs.total
        generic_valid=np.flatnonzero(feasible)
        generic_best=int(generic_valid[np.argmin(generic_costs.total[generic_valid])]) if len(generic_valid) else 0
        features=np.column_stack((
            action_confidence,
            np.full(len(action_confidence),confidence.root_confidence),
            np.full(len(action_confidence),len(first["support"])/5.0),
            cost_disagreement,response_disagreement,distance_disagreement,
            generic_costs.total-generic_costs.total[generic_best],
            shrunk_costs.total-generic_costs.total,
        ))
        results.append(Episode(
            artifact.key,artifact.indices,first,artifact,safety_prediction,feasible,reasons,
            confidence,action_confidence,theta_shrunk,generic_rollout,shrunk_rollout,
            generic_costs,artifact.point_costs,shrunk_costs,artifact.gt_costs,features,
            rule_index,no_uncertainty_index,oracle_functional_costs,
        ))
    return results


def fit_benefit(episodes:list[Episode])->Any:
    from src.decision.personalization_confidence import fit_switch_benefit_calibrator
    features=[];targets=[]
    for episode in episodes:
        valid=np.flatnonzero(episode.feasible)
        if not len(valid):continue
        generic=int(valid[np.argmin(episode.generic_costs.total[valid])])
        for index in valid:
            features.append(episode.switch_features[index])
            targets.append(float(episode.gt_costs.total[index]<episode.gt_costs.total[generic]-1e-6))
    return fit_switch_benefit_calibrator(np.asarray(features),np.asarray(targets),"train")


def add_benefit_probability(episodes:list[Episode],calibrator:Any)->None:
    from src.decision.personalization_confidence import predict_switch_benefit
    for episode in episodes:episode.benefit_probability=predict_switch_benefit(calibrator,episode.switch_features)


def arbitrate(episode:Episode,config:ArbitrationConfig)->Any:
    from src.decision.personalization_confidence import selective_personalization_select
    return selective_personalization_select(
        episode.personal_costs.action_ids,episode.feasible,episode.generic_costs.total,
        episode.personal_costs.total,episode.shrunk_costs.total,episode.action_confidence,
        config.minimum_confidence,config.minimum_margin,config.minimum_benefit_probability,
        episode.benefit_probability,
    )


def validation_score(episodes:list[Episode],config:ArbitrationConfig)->tuple[float,dict[str,float]]:
    regret=[];harmful=[];beneficial=[];violation=[];s9=[];s10=[]
    for episode in episodes:
        decision=arbitrate(episode,config);valid=np.flatnonzero(episode.feasible)
        if decision.selected_index is None:regret.append(.25);violation.append(0.0);continue
        index=decision.selected_index;oracle=int(np.argmin(episode.gt_costs.total));generic=int(valid[np.argmin(episode.generic_costs.total[valid])])
        delta=float(episode.gt_costs.total[index]-episode.gt_costs.total[generic]);switched=index!=generic
        regret.append(float(episode.gt_costs.total[index]-episode.gt_costs.total[oracle]));violation.append(float(episode.gt_costs.unsafe_duration[index]>0))
        harmful.append(float(switched and delta>1e-6));beneficial.append(float(switched and delta<-1e-6))
        if episode.key[0]=="S9_uncertain_new_person":s9.append(regret[-1])
        if episode.key[0]=="S10_action_conflict":s10.append(regret[-1])
    metrics={"regret":float(np.mean(regret)),"harmful":float(np.mean(harmful or (0,))),"beneficial":float(np.mean(beneficial or (0,))),"violation":float(np.mean(violation)),"s9":float(np.mean(s9)),"s10":float(np.mean(s10))}
    score=metrics["regret"]+0.8*metrics["harmful"]+20*metrics["violation"]+0.25*metrics["s9"]+2*metrics["s10"]+0.03*float(metrics["beneficial"]==0)
    return score,metrics


def select_config(episodes:list[Episode])->tuple[ArbitrationConfig,list[dict[str,Any]]]:
    rows=[];best=None
    for confidence in (0.,.15,.25,.35,.45,.55,.7):
        for margin in (0.,.005,.01,.02,.04,.08):
            for probability in (.35,.45,.5,.55,.65):
                config=ArbitrationConfig(confidence,margin,probability);score,metrics=validation_score(episodes,config)
                row={"synthetic_interaction":LABEL,"split":"validation",**asdict(config),"score":score,**metrics};rows.append(row)
                key=(score,metrics["violation"],metrics["harmful"],metrics["regret"])
                if best is None or key<best[0]:best=(key,config)
    assert best is not None
    return best[1],rows


def corr(a:np.ndarray,b:np.ndarray)->float|None:
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>1e-12 and b.std()>1e-12 else None
def rank_corr(a:np.ndarray,b:np.ndarray)->float|None:return corr(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))


def evaluate(args:argparse.Namespace,episodes:list[Episode],config:ArbitrationConfig)->dict[str,Any]:
    from src.data.functional_response_state import RESPONSE_STATE_NAMES
    from src.data.robot_action_schema import RobotAction
    from src.decision.personalization_confidence import decision_margin
    from src.decision.safety_calibration import worst_case_regret
    confidence_rows=[];action_rows=[];disagreement=[];margin_rows=[];switch_rows=[];decision_rows=[]
    s9=[];s10=[];turn=[];distance=[];calibration=[]
    for episode in episodes:
        decision=arbitrate(episode,config);actions=episode.personal_costs.action_ids;valid=np.flatnonzero(episode.feasible)
        oracle=int(np.argmin(episode.gt_costs.total))
        if len(valid):
            generic=int(valid[np.argmin(episode.generic_costs.total[valid])]);full=int(valid[np.argmin(episode.personal_costs.total[valid])])
        else:generic=full=oracle
        if len(valid)>=2:
            generic_margin=decision_margin(actions,episode.generic_costs.total,episode.feasible)
            personal_margin=decision_margin(actions,episode.personal_costs.total,episode.feasible)
        else:generic_margin=personal_margin=None
        selected=decision.selected_index
        if selected is None:
            total=float(episode.gt_costs.total[oracle]+.25);regret=.25;unsafe=False;action=None
        else:
            total=float(episode.gt_costs.total[selected]);regret=total-float(episode.gt_costs.total[oracle]);unsafe=bool(episode.gt_costs.unsafe_duration[selected]>0);action=int(actions[selected])
        for dimension,name in enumerate(RESPONSE_STATE_NAMES):
            confidence_rows.append({"synthetic_interaction":LABEL,"seed":args.seed,"scenario":episode.key[0],"sample":episode.key[1],"profile":episode.first["profile"],"dimension":name,"confidence":episode.confidence.dimension_confidence[dimension],"uncertainty_confidence":episode.confidence.uncertainty_confidence[dimension],"support_coverage":episode.confidence.support_coverage[dimension],"observability":episode.confidence.observability_confidence[dimension],"theta_personal":episode.first["theta_hat"][dimension],"theta_used":episode.theta_shrunk[dimension],"GT_theta_evaluation_only":episode.first["theta_true"][dimension],"personal_theta_absolute_error":abs(episode.first["theta_hat"][dimension]-episode.first["theta_true"][dimension]),"shrunk_theta_absolute_error":abs(episode.theta_shrunk[dimension]-episode.first["theta_true"][dimension])})
        for index,action_id in enumerate(actions):
            action_rows.append({"synthetic_interaction":LABEL,"seed":args.seed,"scenario":episode.key[0],"sample":episode.key[1],"action":int(action_id),"C_action":episode.action_confidence[index],"feasible":bool(episode.feasible[index]),"benefit_probability":episode.benefit_probability[index]})
            response_disagreement=np.linalg.norm(episode.artifact.point_costs.action_ids[index]*0+episode.first["predicted_rollout"].predicted_action_effect[index]-episode.generic_rollout.predicted_action_effect[index],axis=-1).mean()
            distance_disagreement=np.abs(episode.first["predicted_rollout"].predicted_human_robot_distance[index]-episode.generic_rollout.predicted_human_robot_distance[index]).mean()
            disagreement.append({"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"action":int(action_id),"J_generic":episode.generic_costs.total[index],"J_personalized":episode.personal_costs.total[index],"J_shrunk":episode.shrunk_costs.total[index],"Delta_personal":episode.personal_costs.total[index]-episode.generic_costs.total[index],"response_disagreement":response_disagreement,"distance_disagreement":distance_disagreement,"ranking_disagreement":int(np.argsort(np.argsort(episode.generic_costs.total))[index]!=np.argsort(np.argsort(episode.personal_costs.total))[index])})
        predicted_delta=float(episode.shrunk_costs.total[generic]-episode.shrunk_costs.total[decision.selective_index]) if decision.selective_index is not None else 0.
        gt_delta=float(episode.gt_costs.total[generic]-episode.gt_costs.total[decision.selective_index]) if decision.selective_index is not None else 0.
        margin_rows.append({"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"generic_best_action":int(actions[generic]),"personal_best_action":int(actions[full]),"generic_best_second_margin":"" if generic_margin is None else generic_margin.absolute_margin,"personal_best_second_margin":"" if personal_margin is None else personal_margin.absolute_margin,"predicted_switch_margin":predicted_delta,"GT_switch_margin":gt_delta})
        switched=selected is not None and selected!=generic;delta_gt=0. if selected is None else float(episode.gt_costs.total[selected]-episode.gt_costs.total[generic])
        category="NO_SWITCH" if not switched else "BENEFICIAL_SWITCH" if delta_gt<-1e-6 else "HARMFUL_SWITCH" if delta_gt>1e-6 else "NEUTRAL_SWITCH"
        full_delta=float(episode.gt_costs.total[full]-episode.gt_costs.total[generic]);full_category="NO_SWITCH" if full==generic else "BENEFICIAL_SWITCH" if full_delta<-1e-6 else "HARMFUL_SWITCH" if full_delta>1e-6 else "NEUTRAL_SWITCH"
        switch_rows.append({"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"generic_action":int(actions[generic]),"full_action":int(actions[full]),"selective_action":"" if action is None else action,"full_switch":full_category,"selective_switch":category,"full_GT_delta_vs_generic":full_delta,"selective_GT_delta_vs_generic":delta_gt,"C_action":"" if selected is None else episode.action_confidence[selected],"decision_mode":decision.mode.value})
        decision_rows.append({"synthetic_interaction":LABEL,"seed":args.seed,"scenario":episode.key[0],"sample":episode.key[1],"profile":episode.first["profile"],"model":"D2-S Selective","selected_action":"" if action is None else action,"decision_mode":decision.mode.value,"GT_Total_Cost":total,"Oracle_Regret":regret,"Safety_Violation":unsafe,"KEEP":action==int(RobotAction.KEEP),"feasible_mask":"|".join("1" if v else "0" for v in episode.feasible),"reentry":bool(selected is not None and not episode.feasible[selected])})
        base={"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"generic_action":int(actions[generic]),"full_action":int(actions[full]),"selective_action":"" if action is None else action,"generic_GT_cost":episode.gt_costs.total[generic],"full_GT_cost":episode.gt_costs.total[full],"selective_GT_cost":total,"selective_regret":regret,"selective_violation":unsafe,"decision_mode":decision.mode.value}
        if episode.key[0]=="S9_uncertain_new_person":s9.append(base)
        if episode.key[0]=="S10_action_conflict":s10.append(base)
        if episode.key[0]=="S8_high_turn_sensitive":turn.append(base)
        if episode.key[0]=="S6_high_distance_sensitive":distance.append(base)
        calibration.append({"predicted_margin":predicted_delta,"GT_margin":gt_delta,"small_margin":abs(predicted_delta)<.02})
    regrets=np.asarray([row["Oracle_Regret"] for row in decision_rows]);regret_stats=worst_case_regret(regrets)
    full_harmful=sum(row["full_switch"]=="HARMFUL_SWITCH" for row in switch_rows);select_harmful=sum(row["selective_switch"]=="HARMFUL_SWITCH" for row in switch_rows)
    full_beneficial=sum(row["full_switch"]=="BENEFICIAL_SWITCH" for row in switch_rows);select_beneficial=sum(row["selective_switch"]=="BENEFICIAL_SWITCH" for row in switch_rows)
    bins=[]
    for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        values=[]
        for episode in episodes:
            valid=np.flatnonzero(episode.feasible)
            if not len(valid):continue
            generic=int(valid[np.argmin(episode.generic_costs.total[valid])])
            for index in valid:
                if low<=episode.action_confidence[index]<(high if high<1 else high+1e-9):values.append((float(episode.gt_costs.total[generic]-episode.gt_costs.total[index]),float(episode.gt_costs.total[index]<episode.gt_costs.total[generic]-1e-6),float(episode.gt_costs.total[index]>episode.gt_costs.total[generic]+1e-6),float(episode.gt_costs.total[index]-episode.gt_costs.total.min())))
        bins.append({"synthetic_interaction":LABEL,"confidence_bin":f"{low:.1f}-{high:.1f}","count":len(values),"mean_GT_personalization_benefit":float(np.mean([v[0] for v in values])) if values else "","beneficial_rate":float(np.mean([v[1] for v in values])) if values else "","harmful_rate":float(np.mean([v[2] for v in values])) if values else "","oracle_regret":float(np.mean([v[3] for v in values])) if values else ""})
    pred=np.asarray([row["predicted_margin"] for row in calibration]);truth=np.asarray([row["GT_margin"] for row in calibration]);margin_summary={"MAE":float(np.mean(np.abs(pred-truth))),"sign_accuracy":float(np.mean(np.sign(pred)==np.sign(truth))),"spearman":rank_corr(pred,truth),"small_margin_MAE":float(np.mean(np.abs(pred[[r["small_margin"] for r in calibration]]-truth[[r["small_margin"] for r in calibration]]))) if any(r["small_margin"] for r in calibration) else None,"large_margin_MAE":float(np.mean(np.abs(pred[[not r["small_margin"] for r in calibration]]-truth[[not r["small_margin"] for r in calibration]]))) if any(not r["small_margin"] for r in calibration) else None}
    dimension_confidence=np.asarray([row["confidence"] for row in confidence_rows]);dimension_error=np.asarray([row["personal_theta_absolute_error"] for row in confidence_rows])
    action_conf=[];action_benefit=[]
    for episode in episodes:
        valid=np.flatnonzero(episode.feasible)
        if not len(valid):continue
        generic=int(valid[np.argmin(episode.generic_costs.total[valid])])
        for index in valid:
            action_conf.append(episode.action_confidence[index]);action_benefit.append(episode.gt_costs.total[generic]-episode.gt_costs.total[index])
    personalization_calibration={"dimension_confidence_vs_negative_theta_error_spearman":rank_corr(dimension_confidence,-dimension_error),"action_confidence_vs_GT_benefit_spearman":rank_corr(np.asarray(action_conf),np.asarray(action_benefit))}
    metrics={"GT_Total_Cost":float(np.mean([r["GT_Total_Cost"] for r in decision_rows])),"Safety_Violation":float(np.mean([r["Safety_Violation"] for r in decision_rows])),"KEEP_Rate":float(np.mean([r["KEEP"] for r in decision_rows])),"full_harmful_switches":full_harmful,"selective_harmful_switches":select_harmful,"full_beneficial_switches":full_beneficial,"selective_beneficial_switches":select_beneficial,**{f"regret_{k}":v for k,v in regret_stats.items()}}
    baseline_rows=[]
    for episode in episodes:
        valid=np.flatnonzero(episode.feasible);oracle=int(np.argmin(episode.gt_costs.total))
        indices={"D0 Rule":episode.rule_index,"D3 No Uncertainty":episode.no_uncertainty_index}
        if len(valid):
            indices["D1 Generic Safe"]=int(valid[np.argmin(episode.generic_costs.total[valid])])
            indices["D2-FULL"]=int(valid[np.argmin(episode.personal_costs.total[valid])])
            indices["D4 Oracle Functional"]=int(valid[np.argmin(episode.oracle_functional_costs.total[valid])])
        selective=next(row for row in decision_rows if row["scenario"]==episode.key[0] and int(row["sample"])==episode.key[1])
        for model,index in indices.items():
            baseline_rows.append({"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"model":model,"selected_action":int(episode.gt_costs.action_ids[index]),"GT_Total_Cost":episode.gt_costs.total[index],"Oracle_Regret":episode.gt_costs.total[index]-episode.gt_costs.total[oracle],"Safety_Violation":bool(episode.gt_costs.unsafe_duration[index]>0),"KEEP":int(episode.gt_costs.action_ids[index])==int(RobotAction.KEEP)})
        if not len(valid):
            for model in ("D1 Generic Safe","D2-FULL","D4 Oracle Functional"):
                baseline_rows.append({"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"model":model,"selected_action":"","GT_Total_Cost":episode.gt_costs.total[oracle]+.25,"Oracle_Regret":.25,"Safety_Violation":False,"KEEP":False})
        baseline_rows.append({"synthetic_interaction":LABEL,"scenario":episode.key[0],"sample":episode.key[1],"model":"D2-S Selective","selected_action":selective["selected_action"],"GT_Total_Cost":selective["GT_Total_Cost"],"Oracle_Regret":selective["Oracle_Regret"],"Safety_Violation":selective["Safety_Violation"],"KEEP":selective["KEEP"]})
    model_summary={}
    for model in sorted(set(row["model"] for row in baseline_rows)):
        rows=[row for row in baseline_rows if row["model"]==model]
        model_summary[model]={"GT_Total_Cost":float(np.mean([row["GT_Total_Cost"] for row in rows])),"Mean_Regret":float(np.mean([row["Oracle_Regret"] for row in rows])),"P95_Regret":float(np.percentile([row["Oracle_Regret"] for row in rows],95)),"Max_Regret":float(np.max([row["Oracle_Regret"] for row in rows])),"Safety_Violation":float(np.mean([row["Safety_Violation"] for row in rows])),"KEEP_Rate":float(np.mean([row["KEEP"] for row in rows]))}
    return {"confidence":confidence_rows,"actions":action_rows,"disagreement":disagreement,"margins":margin_rows,"switches":switch_rows,"decision":decision_rows,"baselines":baseline_rows,"model_summary":model_summary,"s9":s9,"s10":s10,"turn":turn,"distance":distance,"value_curve":bins,"margin_calibration":[{"synthetic_interaction":LABEL,**row} for row in calibration],"margin_summary":margin_summary,"personalization_calibration":personalization_calibration,"metrics":metrics}


def make_figures(output_dir:Path,evaluation:dict[str,Any])->list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=output_dir/"figures";directory.mkdir(parents=True,exist_ok=True);paths=[]
    def save(name:str)->None:
        path=directory/name;plt.title(LABEL,fontsize=7);plt.tight_layout();plt.savefig(path,dpi=150);plt.close();paths.append(str(path))
    switches=evaluation["switches"]
    plt.figure();plt.scatter([row["C_action"] if row["C_action"]!="" else 0 for row in switches],[row["selective_GT_delta_vs_generic"] for row in switches],alpha=.35);plt.axhline(0,color="k");plt.xlabel("action confidence");plt.ylabel("GT selective cost delta");save("confidence_vs_personalization_benefit.png")
    bins=evaluation["value_curve"];plt.figure();plt.bar([row["confidence_bin"] for row in bins],[0 if row["beneficial_rate"]=="" else row["beneficial_rate"] for row in bins],label="beneficial");plt.plot([row["confidence_bin"] for row in bins],[0 if row["harmful_rate"]=="" else row["harmful_rate"] for row in bins],"ro-",label="harmful");plt.legend();plt.xticks(rotation=30);save("switch_by_confidence.png")
    models=("D1 Generic Safe","D2-FULL","D2-S Selective");plt.figure();plt.bar(models,[evaluation["model_summary"][model]["Mean_Regret"] for model in models]);plt.xticks(rotation=20);plt.ylabel("mean regret");save("generic_full_selective_regret.png")
    margin=evaluation["margin_calibration"];plt.figure();plt.scatter([r["predicted_margin"] for r in margin],[r["GT_margin"] for r in margin],alpha=.35);plt.xlabel("predicted margin");plt.ylabel("GT margin");save("cost_margin_predicted_vs_gt.png")
    for field,name in (("s9","s9_action_comparison.png"),("s10","s10_action_comparison.png"),("turn","turn_sensitive_comparison.png"),("distance","distance_sensitive_comparison.png")):
        rows=evaluation[field];plt.figure();x=np.arange(len(rows));plt.plot(x,[r["generic_GT_cost"] for r in rows],label="generic");plt.plot(x,[r["full_GT_cost"] for r in rows],label="full");plt.plot(x,[r["selective_GT_cost"] for r in rows],label="selective");plt.legend();plt.ylabel("GT cost");save(name)
    modes=[r["decision_mode"] for r in evaluation["decision"]];labels=sorted(set(modes));plt.figure();plt.bar(labels,[modes.count(label) for label in labels]);plt.xticks(rotation=25);save("decision_modes.png")
    regrets=np.sort([r["Oracle_Regret"] for r in evaluation["decision"]]);plt.figure();plt.plot(regrets,np.linspace(0,1,len(regrets)));plt.xlabel("regret");plt.ylabel("CDF");save("regret_cdf.png")
    return paths


def scenario_metrics(rows:list[dict[str,Any]])->dict[str,float]:
    return {"GT_Total_Cost":float(np.mean([r["selective_GT_cost"] for r in rows])),"Mean_Regret":float(np.mean([r["selective_regret"] for r in rows])),"Safety_Violation":float(np.mean([r["selective_violation"] for r in rows])),"KEEP_Rate":float(np.mean([r["selective_action"]==0 for r in rows]))}


def main()->None:
    args=parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    print(LABEL,flush=True)
    import scripts.run_phase4c_decision as c0
    import scripts.run_phase4c1_safety_calibration as c1
    import scripts.run_phase4c2_belief_selection as c2
    from src.decision.counterfactual_rollout import CounterfactualRolloutEngine
    from src.decision.personalization_confidence import predict_switch_benefit
    engine=CounterfactualRolloutEngine.from_phase4b6_checkpoint(args.phase4b6_dir/"checkpoints"/"f2_original_best.pt",args.device)
    prior_mean,prior_std=c0.load_prior(argparse.Namespace(phase4b6_dir=args.phase4b6_dir))
    root_head,root_scale,safety_head,safety_calibration,c2_summary=load_frozen_phase4c2(args,torch)
    c2_selector=c2.SelectorConfig(**c2_summary["selector_config"])
    train=c1.build_records(args,engine,"train",args.seed+101,30,prior_mean,prior_std)
    validation=c1.build_records(args,engine,"validation",args.seed+202,12,prior_mean,prior_std)
    train_artifacts,train_safety,cost_calibration=build_base(args,train,engine,prior_mean,prior_std,root_head,root_scale,safety_head,safety_calibration,None,torch)
    validation_artifacts,validation_safety,_=build_base(args,validation,engine,prior_mean,prior_std,root_head,root_scale,safety_head,safety_calibration,cost_calibration,torch)
    train_episodes=episode_data(args,train,train_artifacts,train_safety,cost_calibration,engine,prior_mean,prior_std,c2_selector)
    benefit_calibrator=fit_benefit(train_episodes);add_benefit_probability(train_episodes,benefit_calibrator)
    validation_episodes=episode_data(args,validation,validation_artifacts,validation_safety,cost_calibration,engine,prior_mean,prior_std,c2_selector);add_benefit_probability(validation_episodes,benefit_calibrator)
    config,validation_rows=select_config(validation_episodes)
    # Test is materialized exactly once after calibrator and arbitration thresholds freeze.
    test=c1.build_records(args,engine,"test",args.seed+303,12,prior_mean,prior_std)
    test_artifacts,test_safety,_=build_base(args,test,engine,prior_mean,prior_std,root_head,root_scale,safety_head,safety_calibration,cost_calibration,torch)
    test_episodes=episode_data(args,test,test_artifacts,test_safety,cost_calibration,engine,prior_mean,prior_std,c2_selector);add_benefit_probability(test_episodes,benefit_calibrator)
    evaluation=evaluate(args,test_episodes,config)
    outputs={"personalization_confidence.csv":evaluation["confidence"],"action_confidence.csv":evaluation["actions"],"generic_personal_disagreement.csv":evaluation["disagreement"],"decision_margin.csv":evaluation["margins"],"switch_audit.csv":evaluation["switches"],"switch_calibration.csv":validation_rows,"s9_analysis.csv":evaluation["s9"],"s10_analysis.csv":evaluation["s10"],"turn_sensitive.csv":evaluation["turn"],"distance_sensitive.csv":evaluation["distance"],"personalization_value_curve.csv":evaluation["value_curve"],"cost_margin_calibration.csv":evaluation["margin_calibration"],"decision_summary.csv":evaluation["baselines"]}
    for name,rows in outputs.items():write_csv(args.output_dir/name,rows)
    figures=make_figures(args.output_dir,evaluation)
    metrics=evaluation["metrics"];models=evaluation["model_summary"]
    s9=scenario_metrics(evaluation["s9"]);s10=scenario_metrics(evaluation["s10"]);turn=scenario_metrics(evaluation["turn"]);distance=scenario_metrics(evaluation["distance"])
    nonempty=[row for row in evaluation["value_curve"] if row["count"]]
    low=next((row for row in nonempty if row["confidence_bin"]=="0.0-0.2"),nonempty[0]);high=nonempty[-1]
    criteria={"GT_total_cost_improved":metrics["GT_Total_Cost"]<1.85346*.97,"mean_regret_improved":metrics["regret_mean"]<.04935*.85,"P95_regret_improved":metrics["regret_P95"]<.25520*.85,"harmful_switches_reduced":metrics["selective_harmful_switches"]<metrics["full_harmful_switches"],"beneficial_switches_preserved":metrics["selective_beneficial_switches"]>0,"S9_regret_improved":s9["Mean_Regret"]<.04406*.95,"S10_regret_preserved":s10["Mean_Regret"]<=.01,"turn_not_worse_than_D1":turn["GT_Total_Cost"]<=.30257*1.01,"distance_advantage_preserved":distance["GT_Total_Cost"]<4.5726,"safety_preserved":metrics["Safety_Violation"]<=.008333333333333333,"rejected_action_reentry_zero":not any(row["reentry"] for row in evaluation["decision"]),"high_confidence_value_better_than_low":float(high["mean_GT_personalization_benefit"])>float(low["mean_GT_personalization_benefit"]),"decision_margin_valid_ranking":evaluation["margin_summary"]["spearman"] is not None and evaluation["margin_summary"]["spearman"]>.15,"selector_does_not_access_test_GT":True}
    criteria["seed42_gate_passed"]=bool(all(criteria.values()))
    write_csv(args.output_dir/"multiseed.csv",[{"synthetic_interaction":LABEL,"seed":args.seed,"metric":k,"value":v,"detail":"seed42 gate only; five-seed forbidden unless all gates pass"} for k,v in metrics.items()])
    summary={"label":LABEL,"seed":args.seed,"phase4c2_safety_frozen":True,"test_materialized_once_after_freeze":True,"arbitration_config":asdict(config),"switch_calibrator_fit_split":benefit_calibrator.fit_split,"metrics":metrics,"models":models,"calibration_separation":{"safety_calibration":{"p_unsafe_target":"GT unsafe","unsafe_candidate_rejection":c2_summary["metrics"]["unsafe_candidate_rejection"],"safe_candidate_retention":c2_summary["metrics"]["safe_candidate_retention"],"frozen_from":"Phase4C.2"},"decision_calibration":evaluation["margin_summary"],"personalization_calibration":evaluation["personalization_calibration"]},"scenarios":{"S9":s9,"S10":s10,"turn_sensitive":turn,"distance_sensitive":distance},"confidence_value":{"low_bin":low,"high_bin":high},"success_criteria":criteria,"five_seed_started":False,"phase4c_ready_to_freeze":False,"phase5_authorized":False,"figures":figures}
    write_json(args.output_dir/"summary.json",summary)
    print(f"cost={metrics['GT_Total_Cost']:.5f} regret={metrics['regret_mean']:.5f} safety={metrics['Safety_Violation']:.4f} gate={criteria['seed42_gate_passed']}",flush=True)


if __name__=="__main__":main()
