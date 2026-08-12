"""Distributional counterfactual rollout from a calibrated root belief."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.skeleton_schema import compute_root
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_cost import DecisionCosts, DecisionCostWeights
from src.decision.decision_state import DecisionState
from src.decision.root_belief import RootFutureBelief


@dataclass(frozen=True)
class DistributionalRollout:
    action_ids: np.ndarray
    sampled_root: np.ndarray
    sampled_distance: np.ndarray
    sampled_minimum_distance: np.ndarray
    chance_unsafe: np.ndarray
    expected_cost: np.ndarray
    std_cost: np.ndarray
    p95_cost: np.ndarray
    cvar_cost: np.ndarray

    def __post_init__(self) -> None:
        if self.sampled_root.ndim != 4 or self.sampled_root.shape[-1] != 3:
            raise ValueError("sampled_root must have shape [A,N,H,3]")
        if not all(np.isfinite(np.asarray(value)).all() for value in (
            self.sampled_root, self.sampled_distance, self.sampled_minimum_distance,
            self.chance_unsafe, self.expected_cost, self.std_cost, self.p95_cost, self.cvar_cost,
        )):
            raise ValueError("distributional rollout must be finite")
        if np.any(self.chance_unsafe < 0.0) or np.any(self.chance_unsafe > 1.0):
            raise ValueError("chance probabilities must be in [0,1]")


def deterministic_normal_samples(sample_count: int, horizon: int, seed: int = 0) -> np.ndarray:
    """Antithetic standard-normal samples, deterministic for audit/replay."""
    if sample_count < 2 or sample_count % 2:
        raise ValueError("sample_count must be an even integer >= 2")
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((sample_count // 2, horizon, 3))
    values = np.concatenate((half, -half), axis=0)
    return values / np.maximum(values.std(axis=0, keepdims=True), 1e-6)


def _sample_costs(
    state: DecisionState,
    base: DecisionCosts,
    sampled_distance: np.ndarray,
    weights: DecisionCostWeights,
) -> np.ndarray:
    """Re-evaluate only geometry-dependent task/safety terms per belief sample."""
    distance = np.asarray(sampled_distance, dtype=np.float64)
    target = float(state.target_follow_distance)
    initial_error = abs(float(state.robot_history[-1, 5]) - target)
    final_error = np.abs(distance[..., -1] - target)
    mean_error = np.mean(np.abs(distance - target), axis=-1)
    task = final_error + 0.35 * mean_error + 0.45 * np.maximum(final_error - initial_error, 0.0)
    task += 0.25 * np.maximum(distance[..., -1] - 2.8, 0.0)
    minimum = distance.min(axis=-1)
    duration = np.mean(distance < state.too_close_distance, axis=-1)
    close_gap = np.maximum(state.too_close_distance - minimum, 0.0)
    safety = 5.0 * (minimum < state.too_close_distance) + 8.0 * duration + 10.0 * close_gap
    static = (
        weights.human_response * base.human_response[:, None]
        + weights.disturbance * base.disturbance[:, None]
    )
    return weights.task * task + weights.safety * safety + static


def propagate_root_belief(
    state: DecisionState,
    point_rollout: CounterfactualRollout,
    root_belief: RootFutureBelief,
    base_costs: DecisionCosts,
    sample_count: int = 16,
    seed: int = 0,
    weights: DecisionCostWeights = DecisionCostWeights(),
    cvar_alpha: float = 0.90,
) -> DistributionalRollout:
    if point_rollout.predicted_root.shape[1:] != root_belief.mu_root.shape:
        raise ValueError("root belief horizon must match counterfactual rollout")
    natural_point_root = compute_root(point_rollout.natural_future)
    # Candidate-specific response is retained; only the shared natural-root prior
    # is replaced by its calibrated belief.
    action_delta = point_rollout.predicted_root - natural_point_root[None]
    z = deterministic_normal_samples(sample_count, root_belief.mu_root.shape[0], seed)
    natural_samples = root_belief.mu_root[None] + z * root_belief.sigma_root[None]
    sampled_root = action_delta[:, None] + natural_samples[None]
    sampled_distance = np.linalg.norm(
        sampled_root[..., :2] - point_rollout.predicted_robot_xy[:, None], axis=-1,
    )
    sampled_minimum = sampled_distance.min(axis=-1)
    probability = np.mean(sampled_minimum < state.too_close_distance, axis=1)
    sampled_cost = _sample_costs(state, base_costs, sampled_distance, weights)
    tail_start = max(int(np.floor(cvar_alpha * sample_count)), 0)
    sorted_cost = np.sort(sampled_cost, axis=1)
    return DistributionalRollout(
        action_ids=point_rollout.action_ids.copy(), sampled_root=sampled_root.astype(np.float32),
        sampled_distance=sampled_distance.astype(np.float32),
        sampled_minimum_distance=sampled_minimum.astype(np.float32),
        chance_unsafe=probability.astype(np.float32),
        expected_cost=sampled_cost.mean(axis=1), std_cost=sampled_cost.std(axis=1),
        p95_cost=np.percentile(sampled_cost, 95, axis=1),
        cvar_cost=sorted_cost[:, tail_start:].mean(axis=1),
    )
