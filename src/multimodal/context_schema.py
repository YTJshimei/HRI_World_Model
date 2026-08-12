"""Leakage-safe structured context tokens for Phase 5A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FORBIDDEN_MODEL_INPUTS = frozenset({
    "gt_future", "future_global", "gt_theta", "theta_true", "gt_benefit",
    "gt_best_action", "oracle_action", "person_id", "profile_id",
})
TOKEN_ORDER = (
    "skeleton", "motion", "robot", "functional", "candidate",
    "uncertainty", "diagnostic", "interaction", "scene",
)
TOKEN_DIMS = {
    "skeleton": 12, "motion": 12, "robot": 12, "functional": 24,
    "candidate": 8, "uncertainty": 12, "diagnostic": 12,
    "interaction": 8, "scene": 8,
}


@dataclass(frozen=True)
class StructuredContextTokens:
    tokens: dict[str, np.ndarray]
    candidate_action: int
    context_id: str
    initial_state_id: str
    context_split: str

    def __post_init__(self) -> None:
        if set(self.tokens) != set(TOKEN_ORDER):
            raise ValueError("context must contain every canonical token group")
        for name, dimension in TOKEN_DIMS.items():
            value = np.asarray(self.tokens[name])
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"{name} token must be finite with shape [{dimension}]")
        fields = set(self.__dict__) | set(self.tokens)
        if fields & FORBIDDEN_MODEL_INPUTS:
            raise ValueError("ground truth, identity, and oracle fields cannot enter context")

    def flattened(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.tokens[name], np.float32) for name in TOKEN_ORDER])


def _pad(values: np.ndarray, size: int) -> np.ndarray:
    flat = np.asarray(values, np.float32).reshape(-1)
    result = np.zeros(size, np.float32); result[:min(size, len(flat))] = flat[:size]
    return result


def build_context_tokens(
    *, human_history: np.ndarray, robot_history: np.ndarray,
    confidence: np.ndarray, visibility: np.ndarray,
    theta_person: np.ndarray, theta_population: np.ndarray,
    theta_uncertainty: np.ndarray, response_state_mask: np.ndarray,
    support_coverage: np.ndarray, support_action_features: np.ndarray,
    candidate_action: int, candidate_feature: np.ndarray,
    predicted_robot_future: np.ndarray,
    generic_effect: np.ndarray, personalized_effect: np.ndarray,
    generic_distance: np.ndarray, personalized_distance: np.ndarray,
    root_sigma: np.ndarray, minimum_sigma: float, p_unsafe: float,
    motion_state_observable: np.ndarray, scene_observable: np.ndarray,
    context_id: str, initial_state_id: str, context_split: str,
) -> StructuredContextTokens:
    """Build only from runtime-observable/predicted values via an explicit API."""
    history=np.asarray(human_history,float);robot=np.asarray(robot_history,float)
    root=(history[:,11]+history[:,12])*.5
    velocity=np.diff(root[:,:2],axis=0)*10.;acceleration=np.diff(velocity,axis=0)*10.
    visible=np.asarray(visibility,bool);conf=np.asarray(confidence,float)
    distance=np.asarray(generic_distance,float);personal_distance=np.asarray(personalized_distance,float)
    generic=np.asarray(generic_effect,float);personal=np.asarray(personalized_effect,float)
    skeleton=_pad(np.asarray((root[-1,0],root[-1,1],root[-1,2],root[-1,0]-root[0,0],root[-1,1]-root[0,1],conf.mean(),visible.mean(),conf[-1].mean(),visible[-1].mean(),np.std(root[:,0]),np.std(root[:,1]),np.std(root[:,2]))),12)
    motion=_pad(np.concatenate((velocity[-1] if len(velocity) else np.zeros(2),acceleration[-1] if len(acceleration) else np.zeros(2),np.asarray(motion_state_observable))),12)
    robot_token=_pad(np.concatenate((robot[-1],np.asarray(predicted_robot_future)[-1],np.asarray((robot[:,3].mean(),robot[:,4].mean())))),12)
    functional=_pad(np.concatenate((theta_person,theta_population,theta_uncertainty,response_state_mask.astype(float))),24)
    candidate=_pad(np.concatenate((np.asarray((candidate_action,),float),candidate_feature,support_action_features)),8)
    uncertainty=_pad(np.concatenate((np.asarray(root_sigma).mean(axis=0),np.asarray((minimum_sigma,p_unsafe)),support_coverage,theta_uncertainty)),12)
    effect_delta=personal-generic
    diagnostic=_pad(np.asarray((np.linalg.norm(generic,axis=-1).mean(),np.linalg.norm(personal,axis=-1).mean(),np.linalg.norm(effect_delta,axis=-1).mean(),np.mean(personal_distance-generic_distance),np.min(generic_distance),np.min(personal_distance),distance[-1],personal_distance[-1],np.sign(np.mean(personal_distance-generic_distance)),np.std(effect_delta),minimum_sigma,p_unsafe)),12)
    interaction=_pad(np.asarray((robot[-1,5],robot[-1,6],distance[-1]-distance[0],personal_distance[-1]-personal_distance[0],robot[-1,3],robot[-1,4],np.linalg.norm(candidate_feature),visible.mean())),8)
    scene=_pad(scene_observable,8)
    return StructuredContextTokens({"skeleton":skeleton,"motion":motion,"robot":robot_token,"functional":functional,"candidate":candidate,"uncertainty":uncertainty,"diagnostic":diagnostic,"interaction":interaction,"scene":scene},int(candidate_action),context_id,initial_state_id,context_split)


def validate_branch_split_isolation(samples: list[StructuredContextTokens]) -> None:
    seen: dict[str,str]={}
    for sample in samples:
        prior=seen.setdefault(sample.initial_state_id,sample.context_split)
        if prior!=sample.context_split:
            raise ValueError(f"counterfactual branches cross splits: {sample.initial_state_id}")
