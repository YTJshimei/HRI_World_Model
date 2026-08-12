"""Hard safety veto and uncertainty-aware abstention for offline decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.robot_action_schema import RobotAction
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_state import DecisionState


@dataclass(frozen=True)
class SafetyGateResult:
    allowed_mask: np.ndarray
    rejection_reasons: tuple[str, ...]
    abstained: bool
    fallback_action: int | None
    fallback_reason: str


def risk_aware_candidate_mask(
    feasible: np.ndarray,
    predicted_minimum_distance: np.ndarray,
    sigma_minimum_distance: np.ndarray,
    p_unsafe: np.ndarray,
    safe_distance: float,
    probability_threshold: float,
    lcb_multiplier: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Prediction-only v2 gate; no GT safety state is accepted."""
    arrays = tuple(np.asarray(value) for value in (
        feasible, predicted_minimum_distance, sigma_minimum_distance, p_unsafe,
    ))
    if len({item.shape for item in arrays}) != 1:
        raise ValueError("risk-aware gate arrays must have matching shapes")
    allowed = arrays[0].astype(bool).copy()
    reasons = ["" for _ in range(len(allowed))]
    lower_bound = arrays[1] - float(lcb_multiplier) * arrays[2]
    for index in range(len(allowed)):
        if not arrays[0][index]:
            allowed[index] = False; reasons[index] = "candidate_marked_infeasible"
        elif arrays[1][index] < safe_distance:
            allowed[index] = False; reasons[index] = "predicted_minimum_below_threshold"
        elif lower_bound[index] < safe_distance:
            allowed[index] = False; reasons[index] = "distance_lcb_below_threshold"
        elif arrays[3][index] > probability_threshold:
            allowed[index] = False; reasons[index] = "unsafe_probability_above_threshold"
    return allowed, tuple(reasons)


def choose_fallback_action(
    policy: str,
    action_ids: np.ndarray,
    feasible: np.ndarray,
    current_distance: float,
    target_distance: float,
    predicted_risk: np.ndarray,
) -> int:
    """Fallback uses current observations/predicted risk only, never GT."""
    from src.data.robot_action_schema import RobotAction
    actions = np.asarray(action_ids, dtype=int); feasible = np.asarray(feasible, dtype=bool)
    if policy == "FALLBACK_KEEP":
        preferred = (RobotAction.KEEP,)
    elif policy == "FALLBACK_RULE_SAFE":
        preferred = (
            (RobotAction.DISTANCE_PLUS_0_2, RobotAction.SPEED_DOWN_10, RobotAction.KEEP)
            if current_distance < target_distance else
            (RobotAction.KEEP, RobotAction.SPEED_DOWN_10, RobotAction.DISTANCE_PLUS_0_2)
        )
    elif policy == "FALLBACK_MIN_RISK":
        valid = np.flatnonzero(feasible)
        if not len(valid): raise RuntimeError("no feasible fallback")
        return int(actions[valid[np.argmin(np.asarray(predicted_risk)[valid])]])
    else:
        raise ValueError(f"unknown fallback policy: {policy}")
    for action in preferred:
        matches = np.flatnonzero(feasible & (actions == int(action)))
        if len(matches): return int(action)
    raise RuntimeError("requested fallback has no feasible action")


def apply_safety_gate(
    state: DecisionState,
    rollout: CounterfactualRollout,
    uncertainty_aware: bool = True,
    hard_safety: bool = True,
) -> SafetyGateResult:
    allowed = np.asarray([item.feasible for item in state.candidates], dtype=bool)
    reasons = ["" for _ in state.candidates]
    minimum = rollout.predicted_human_robot_distance.min(axis=1)
    coordinate_uncertainty = np.linalg.norm(
        rollout.prediction_uncertainty[..., :2], axis=-1
    ).mean(axis=(-1, -2))
    for index, candidate in enumerate(state.candidates):
        if not candidate.feasible:
            reasons[index] = "candidate_marked_infeasible"
            allowed[index] = False
        elif hard_safety and (
            minimum[index] - 1.64 * coordinate_uncertainty[index]
            < state.too_close_distance + 0.03
        ):
            reasons[index] = "predicted_too_close_hard_veto"
            allowed[index] = False
        elif uncertainty_aware and coordinate_uncertainty[index] > 0.065 and candidate.action in (
            RobotAction.SPEED_UP_10, RobotAction.DISTANCE_MINUS_0_2,
        ):
            reasons[index] = "high_uncertainty_aggressive_action_rejected"
            allowed[index] = False
    fallback_action: int | None = None
    fallback_reason = ""
    abstained = False
    if not allowed.any():
        abstained = True
        too_close_now = float(state.robot_history[-1, 5]) < state.target_follow_distance
        conservative_order = (
            (RobotAction.DISTANCE_PLUS_0_2, RobotAction.SPEED_DOWN_10, RobotAction.KEEP)
            if too_close_now else
            (RobotAction.KEEP, RobotAction.SPEED_DOWN_10, RobotAction.DISTANCE_PLUS_0_2)
        )
        conservative = [
            index for action in conservative_order
            for index, item in enumerate(state.candidates)
            if item.action == action and item.feasible
        ]
        if conservative:
            index = conservative[0]
            allowed[index] = True
            fallback_action = int(state.candidates[index].action)
            fallback_reason = "no_safe_candidate_geometry_aware_conservative_fallback"
        else:
            fallback_reason = "no_feasible_fallback"
    return SafetyGateResult(
        allowed, tuple(reasons), abstained, fallback_action, fallback_reason
    )


def choose_lowest_cost(costs: np.ndarray, gate: SafetyGateResult) -> int:
    if not gate.allowed_mask.any():
        raise RuntimeError("no action survived safety gate")
    masked = np.where(gate.allowed_mask, np.asarray(costs), np.inf)
    return int(np.argmin(masked))
