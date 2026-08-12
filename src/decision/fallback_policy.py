"""Safe fallback state machine with an immutable feasible-action mask."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.data.robot_action_schema import RobotAction


class DecisionMode(str, Enum):
    NORMAL = "NORMAL"
    RULE_FALLBACK = "RULE_FALLBACK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class FallbackDecision:
    mode: DecisionMode
    selected_index: int | None
    selected_action: int | None
    feasible_action_mask: np.ndarray
    reason: str


def rule_safe_order(current_distance: float, target_distance: float) -> tuple[RobotAction, ...]:
    if current_distance < target_distance:
        return (RobotAction.DISTANCE_PLUS_0_2, RobotAction.SPEED_DOWN_10, RobotAction.KEEP)
    return (RobotAction.KEEP, RobotAction.SPEED_DOWN_10, RobotAction.DISTANCE_PLUS_0_2)


def constrained_select_with_fallback(
    action_ids: np.ndarray,
    feasible_action_mask: np.ndarray,
    ranking_cost: np.ndarray,
    current_distance: float,
    target_distance: float,
    normal_candidate_mask: np.ndarray | None = None,
) -> FallbackDecision:
    """Select without ever relaxing or reconstructing ``feasible_action_mask``."""
    actions = np.asarray(action_ids, dtype=int)
    feasible = np.asarray(feasible_action_mask, dtype=bool).copy()
    costs = np.asarray(ranking_cost, dtype=float)
    if actions.shape != feasible.shape or actions.shape != costs.shape:
        raise ValueError("action_ids, feasible mask, and costs must have matching shape")
    normal = feasible.copy() if normal_candidate_mask is None else feasible & np.asarray(normal_candidate_mask, dtype=bool)
    if normal.any():
        index = int(np.argmin(np.where(normal, costs, np.inf)))
        return FallbackDecision(DecisionMode.NORMAL, index, int(actions[index]), feasible, "safe_feasible_set")
    for preferred in rule_safe_order(current_distance, target_distance):
        matches = np.flatnonzero(feasible & (actions == int(preferred)))
        if len(matches):
            index = int(matches[0])
            return FallbackDecision(
                DecisionMode.RULE_FALLBACK, index, int(actions[index]), feasible,
                "rule_safe_action_passed_same_belief_gate",
            )
    return FallbackDecision(
        DecisionMode.ABSTAIN, None, None, feasible,
        "no_action_passed_belief_safety_constraint",
    )
