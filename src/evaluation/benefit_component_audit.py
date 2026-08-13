"""Exact cost-component decomposition for the Phase 5B-v3-R4B GT audit."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from src.data.hold_candidate import HoldCandidateAction, build_hold_candidate_outcome
from src.data.robot_action_schema import ACTION_DEFINITIONS, HOLD_ACTION_ID
from src.data.skeleton_schema import compute_root, shoulder_joints
from src.data.synthetic_interaction import PROFILE_BY_ID, simulate_risk_conditioned_interaction_future
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.candidate_action import TASK_SAFE_CANDIDATES
from src.decision.decision_cost import DecisionCostWeights, compute_decision_costs
from src.decision.decision_state import DecisionState, FunctionalResponseBelief


COMPONENTS = ("task", "safety", "human_response", "disturbance", "uncertainty")


def _rollout(action_ids, natural, simulations) -> CounterfactualRollout:
    future=np.stack([simulation.future_global for simulation in simulations]);root=compute_root(future);local=future-root[...,None,:]
    effect=np.stack([simulation.action_effect for simulation in simulations])
    return CounterfactualRollout(
        np.asarray(action_ids,np.int64),np.asarray(natural,np.float32),root.astype(np.float32),local.astype(np.float32),future.astype(np.float32),
        np.stack([simulation.robot_future_xy for simulation in simulations]),np.stack([simulation.future_human_robot_distance for simulation in simulations]),
        effect.astype(np.float32),np.zeros_like(effect,np.float32),0,
    )


def _state(episode,candidates) -> DecisionState:
    return DecisionState(
        episode.human_history,episode.robot_history,episode.confidence,episode.visibility,
        FunctionalResponseBelief(np.ones(6,np.float32),np.zeros(6,np.float32)),tuple(candidates),
        episode.target_follow_distance,.80,episode.split,
    )


def _heading_effect(effect,natural):
    left,right=shoulder_joints;predicted=natural[None]+effect;predicted_root=compute_root(predicted);natural_root=compute_root(natural)
    predicted_local=predicted-predicted_root[...,None,:];natural_local=natural-natural_root[:,None,:]
    before=np.arctan2(natural_local[-1,right,1]-natural_local[-1,left,1],natural_local[-1,right,0]-natural_local[-1,left,0])
    after=np.arctan2(predicted_local[:,-1,right,1]-predicted_local[:,-1,left,1],predicted_local[:,-1,right,0]-predicted_local[:,-1,left,0])
    return np.abs(np.arctan2(np.sin(after-before),np.cos(after-before)))


def formal_subcomponents(state: DecisionState,rollout: CounterfactualRollout) -> dict[str,np.ndarray]:
    """Expand only subterms that literally exist in ``compute_decision_costs``."""
    distance=np.asarray(rollout.predicted_human_robot_distance,np.float64);target=float(state.target_follow_distance)
    initial_error=abs(float(state.robot_history[-1,5])-target);final_error=np.abs(distance[:,-1]-target);mean_error=np.mean(np.abs(distance-target),axis=1)
    coordinate_uncertainty=np.linalg.norm(np.asarray(rollout.prediction_uncertainty)[...,:2],axis=-1).mean(axis=(-1,-2));scale=np.maximum(coordinate_uncertainty,.015)
    minimum=distance.min(axis=1);unsafe_duration=np.mean(distance<state.too_close_distance,axis=1);violation=1/(1+np.exp((minimum-state.too_close_distance)/scale));close_gap=np.maximum(state.too_close_distance-minimum,0)
    effect=np.asarray(rollout.predicted_action_effect,np.float64);effect_root=compute_root(effect);effect_magnitude=np.linalg.norm(effect,axis=-1).mean(axis=(1,2));speed=np.linalg.norm(np.diff(effect_root[...,:2],axis=1),axis=-1).mean(axis=1)*10;lateral=np.abs(effect_root[:,-1,1]);heading=_heading_effect(effect,rollout.natural_future)
    robot_speed=[];target_change=[];lateral_change=[]
    for action_id in rollout.action_ids:
        definition=ACTION_DEFINITIONS[int(action_id)];robot_speed.append(abs(definition.speed_scale_delta));target_change.append(abs(definition.distance_offset_m));lateral_change.append(abs(definition.lateral_offset_m))
    return {
        "task.final_error":final_error,
        "task.mean_error":.35*mean_error,
        "task.progress_failure":.45*np.maximum(final_error-initial_error,0),
        "task.visibility_proxy":.25*np.maximum(distance[:,-1]-2.8,0),
        "safety.violation_proxy":5*violation,
        "safety.unsafe_duration":8*unsafe_duration,
        "safety.close_gap":10*close_gap,
        "safety.infeasible_penalty":np.asarray([not candidate.feasible for candidate in state.candidates],np.float64)*1e4,
        "human_response.effect_magnitude":.30*effect_magnitude,
        "human_response.speed_effect":.25*speed,
        "human_response.lateral_effect":.20*lateral,
        "human_response.heading_effect":.25*heading,
        "disturbance.robot_speed_action":.30*np.asarray(robot_speed)/.10,
        "disturbance.target_distance_action":.25*np.asarray(target_change)/.20,
        "disturbance.lateral_action":.20*np.asarray(lateral_change)/.20,
        "disturbance.human_effect_magnitude":.25*effect_magnitude/.05,
        "uncertainty.coordinate_uncertainty":coordinate_uncertainty/.05,
    }


def _combine_costs(left,right):
    return {name:np.r_[getattr(left,name),getattr(right,name)] for name in (*COMPONENTS,"total")}


def episode_cost_components(episode,population_profile) -> dict[str,object]:
    """Run the exact formal GT simulator and cost function for all A0-A4+HOLD."""
    profile=PROFILE_BY_ID[episode.profile_id]
    simulations=tuple(simulate_risk_conditioned_interaction_future(episode.human_history,episode.natural_future,episode.robot_history,candidate.action_id,profile,episode.risk_factors) for candidate in episode.candidates)
    rollout=_rollout([candidate.action_id for candidate in episode.candidates],episode.natural_future,simulations);state=_state(episode,TASK_SAFE_CANDIDATES)
    costs=compute_decision_costs(state,rollout,DecisionCostWeights(),include_uncertainty=False);sub=formal_subcomponents(state,rollout)
    hold=build_hold_candidate_outcome(episode,population_profile,profile);hold_rollout=_rollout((HOLD_ACTION_ID,),episode.natural_future,(hold.gt_simulation,));hold_state=_state(episode,(HoldCandidateAction(),))
    hold_costs=compute_decision_costs(hold_state,hold_rollout,DecisionCostWeights(),include_uncertainty=False);hold_sub=formal_subcomponents(hold_state,hold_rollout)
    combined=_combine_costs(costs,hold_costs);combined_sub={name:np.r_[values,hold_sub[name]] for name,values in sub.items()}
    # Validate the expanded subterms against the official cost object.
    for component in COMPONENTS:
        keys=[name for name in combined_sub if name.startswith(component+".")]
        np.testing.assert_allclose(sum((combined_sub[name] for name in keys),np.zeros(6)),combined[component],rtol=1e-10,atol=1e-10)
    return {"action_ids":np.asarray([*[candidate.action_id for candidate in episode.candidates],HOLD_ACTION_ID],int),"components":combined,"subcomponents":combined_sub,"stored_totals":np.r_[episode.gt_costs,hold.gt_total_cost].astype(np.float64),"weights":asdict(DecisionCostWeights())}
