"""Selectors with separated non-oracle and oracle interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.robot_action_schema import RobotAction
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_cost import (
    DecisionCosts, DecisionCostWeights, compute_decision_costs,
)
from src.decision.decision_state import DecisionState
from src.decision.safety_gate import SafetyGateResult, apply_safety_gate, choose_lowest_cost


@dataclass(frozen=True)
class DecisionResult:
    selected_index: int
    selected_action: int
    costs: DecisionCosts
    safety_gate: SafetyGateResult


def rule_based_select(state: DecisionState) -> int:
    """D0 uses current geometry only and never calls a world model."""
    distance = float(state.robot_history[-1, 5])
    available = {int(item.action) for item in state.candidates if item.feasible}
    preference = (
        RobotAction.DISTANCE_PLUS_0_2 if distance < state.target_follow_distance - 0.12
        else RobotAction.DISTANCE_MINUS_0_2 if distance > state.target_follow_distance + 0.12
        else RobotAction.KEEP
    )
    if int(preference) in available:
        return int(preference)
    for fallback in (RobotAction.KEEP, RobotAction.SPEED_DOWN_10, RobotAction.DISTANCE_PLUS_0_2):
        if int(fallback) in available:
            return int(fallback)
    raise RuntimeError("rule-based controller has no feasible action")


def select_model_action(
    state: DecisionState,
    rollout: CounterfactualRollout,
    weights: DecisionCostWeights = DecisionCostWeights(),
    use_uncertainty: bool = True,
    include_human_response: bool = True,
    include_disturbance: bool = True,
    hard_safety: bool = True,
) -> DecisionResult:
    """D1/D2/D3 selector: deliberately has no GT future/theta parameter."""
    costs = compute_decision_costs(
        state, rollout, weights,
        include_human_response=include_human_response,
        include_disturbance=include_disturbance,
        include_uncertainty=use_uncertainty,
    )
    gate = apply_safety_gate(
        state, rollout, uncertainty_aware=use_uncertainty,
        hard_safety=hard_safety,
    )
    index = choose_lowest_cost(costs.total, gate)
    return DecisionResult(index, int(rollout.action_ids[index]), costs, gate)


def select_oracle_action(
    state: DecisionState,
    oracle_rollout: CounterfactualRollout,
    oracle_theta: np.ndarray,
    weights: DecisionCostWeights = DecisionCostWeights(),
) -> DecisionResult:
    """D4-only entry point; caller must construct rollout from oracle theta."""
    if np.asarray(oracle_theta).shape != (6,):
        raise ValueError("oracle theta must have shape [6]")
    return select_model_action(state, oracle_rollout, weights, use_uncertainty=False)


def decision_regret(selected_gt_cost: float, oracle_gt_cost: float) -> float:
    return float(selected_gt_cost - oracle_gt_cost)
