"""Transparent decomposed cost for offline synthetic high-level decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.robot_action_schema import ACTION_DEFINITIONS
from src.data.skeleton_schema import compute_root, shoulder_joints
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_state import DecisionState


@dataclass(frozen=True)
class DecisionCostWeights:
    task: float = 1.0
    safety: float = 3.0
    human_response: float = 1.4
    disturbance: float = 0.55
    uncertainty: float = 0.85


@dataclass(frozen=True)
class DecisionCosts:
    action_ids: np.ndarray
    task: np.ndarray
    safety: np.ndarray
    human_response: np.ndarray
    disturbance: np.ndarray
    uncertainty: np.ndarray
    total: np.ndarray
    minimum_distance: np.ndarray
    unsafe_duration: np.ndarray
    violation_proxy: np.ndarray
    potential_information_gain: np.ndarray


def _heading_effect(effect: np.ndarray, natural: np.ndarray) -> np.ndarray:
    ls, rs = shoulder_joints
    predicted = natural[None] + effect
    predicted_root = compute_root(predicted)
    natural_root = compute_root(natural)
    predicted_local = predicted - predicted_root[..., None, :]
    natural_local = natural - natural_root[:, None, :]
    before = np.arctan2(
        natural_local[-1, rs, 1] - natural_local[-1, ls, 1],
        natural_local[-1, rs, 0] - natural_local[-1, ls, 0],
    )
    after = np.arctan2(
        predicted_local[:, -1, rs, 1] - predicted_local[:, -1, ls, 1],
        predicted_local[:, -1, rs, 0] - predicted_local[:, -1, ls, 0],
    )
    return np.abs(np.arctan2(np.sin(after - before), np.cos(after - before)))


def compute_decision_costs(
    state: DecisionState,
    rollout: CounterfactualRollout,
    weights: DecisionCostWeights = DecisionCostWeights(),
    include_human_response: bool = True,
    include_disturbance: bool = True,
    include_uncertainty: bool = True,
) -> DecisionCosts:
    distance = np.asarray(rollout.predicted_human_robot_distance, dtype=np.float64)
    target = float(state.target_follow_distance)
    initial_error = abs(float(state.robot_history[-1, 5]) - target)
    final_error = np.abs(distance[:, -1] - target)
    mean_error = np.mean(np.abs(distance - target), axis=1)
    progress_failure = np.maximum(final_error - initial_error, 0.0)
    visibility_proxy = np.maximum(distance[:, -1] - 2.8, 0.0)
    task = final_error + 0.35 * mean_error + 0.45 * progress_failure + 0.25 * visibility_proxy

    minimum = distance.min(axis=1)
    unsafe_duration = np.mean(distance < state.too_close_distance, axis=1)
    coordinate_uncertainty = np.linalg.norm(
        np.asarray(rollout.prediction_uncertainty)[..., :2], axis=-1
    ).mean(axis=(-1, -2))
    scale = np.maximum(coordinate_uncertainty, 0.015)
    violation_proxy = 1.0 / (
        1.0 + np.exp((minimum - state.too_close_distance) / scale)
    )
    close_gap = np.maximum(state.too_close_distance - minimum, 0.0)
    feasibility = np.asarray([item.feasible for item in state.candidates], dtype=bool)
    safety = 5.0 * violation_proxy + 8.0 * unsafe_duration + 10.0 * close_gap
    safety = safety + (~feasibility) * 1e4

    effect = np.asarray(rollout.predicted_action_effect, dtype=np.float64)
    effect_root = compute_root(effect)
    speed_effect = np.linalg.norm(np.diff(effect_root[..., :2], axis=1), axis=-1).mean(axis=1) * 10.0
    lateral_effect = np.abs(effect_root[:, -1, 1])
    heading_effect = _heading_effect(effect, rollout.natural_future)
    effect_magnitude = np.linalg.norm(effect, axis=-1).mean(axis=(1, 2))
    human_response = 0.30 * effect_magnitude + 0.25 * speed_effect + 0.20 * lateral_effect + 0.25 * heading_effect

    robot_speed, target_change, lateral_change = [], [], []
    for action_id in rollout.action_ids:
        definition = ACTION_DEFINITIONS[int(action_id)]
        robot_speed.append(abs(definition.speed_scale_delta))
        target_change.append(abs(definition.distance_offset_m))
        lateral_change.append(abs(definition.lateral_offset_m))
    disturbance = (
        0.30 * np.asarray(robot_speed) / 0.10
        + 0.25 * np.asarray(target_change) / 0.20
        + 0.20 * np.asarray(lateral_change) / 0.20
        + 0.25 * effect_magnitude / 0.05
    )
    uncertainty = coordinate_uncertainty / 0.05
    total = (
        weights.task * task + weights.safety * safety
        + weights.human_response * human_response * float(include_human_response)
        + weights.disturbance * disturbance * float(include_disturbance)
        + weights.uncertainty * uncertainty * float(include_uncertainty)
    )
    return DecisionCosts(
        action_ids=rollout.action_ids.copy(), task=task, safety=safety,
        human_response=human_response, disturbance=disturbance,
        uncertainty=uncertainty, total=total, minimum_distance=minimum,
        unsafe_duration=unsafe_duration, violation_proxy=violation_proxy,
        potential_information_gain=np.zeros_like(total),
    )


def verify_cost_sum(
    costs: DecisionCosts, weights: DecisionCostWeights,
    include_human_response: bool = True,
    include_disturbance: bool = True,
    include_uncertainty: bool = True,
) -> None:
    expected = (
        weights.task * costs.task + weights.safety * costs.safety
        + weights.human_response * costs.human_response * float(include_human_response)
        + weights.disturbance * costs.disturbance * float(include_disturbance)
        + weights.uncertainty * costs.uncertainty * float(include_uncertainty)
    )
    np.testing.assert_allclose(costs.total, expected, rtol=1e-7, atol=1e-8)
