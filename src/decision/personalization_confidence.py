"""Runtime-only confidence and arbitration for selective personalization.

No function in this module accepts ground-truth theta, future trajectories,
costs, or oracle actions.  Synthetic outcomes are reserved for fitting and
evaluating the separate switch-benefit calibrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from src.data.functional_response_state import RESPONSE_STATE_DIM, RESPONSE_STATE_SCALE
from src.data.response_probe_schema import PROBE_BY_ID, probe_state_mask
from src.data.robot_action_schema import RobotAction


ACTION_RESPONSE_DIMENSIONS: dict[RobotAction, tuple[int, ...]] = {
    RobotAction.KEEP: (),
    RobotAction.SPEED_DOWN_10: (0, 3, 5),
    RobotAction.SPEED_UP_10: (0, 3, 5),
    RobotAction.DISTANCE_PLUS_0_2: (1, 2, 3, 5),
    RobotAction.DISTANCE_MINUS_0_2: (1, 2, 3, 5),
    RobotAction.LEFT_OFFSET: (4, 3, 2),
    RobotAction.RIGHT_OFFSET: (4, 3, 2),
}


@dataclass(frozen=True)
class PersonalizationConfidence:
    dimension_confidence: np.ndarray
    support_coverage: np.ndarray
    uncertainty_confidence: np.ndarray
    observability_confidence: np.ndarray
    root_confidence: float

    def __post_init__(self) -> None:
        vectors = tuple(np.asarray(value) for value in (
            self.dimension_confidence, self.support_coverage,
            self.uncertainty_confidence, self.observability_confidence,
        ))
        if any(value.shape != (RESPONSE_STATE_DIM,) for value in vectors):
            raise ValueError("confidence vectors must have shape [6]")
        if any(np.any((value < 0.0) | (value > 1.0)) for value in vectors):
            raise ValueError("confidence must remain in [0,1]")
        if not 0.0 <= self.root_confidence <= 1.0:
            raise ValueError("root confidence must remain in [0,1]")


def support_masks_from_probe_ids(probe_ids: Iterable[str]) -> np.ndarray:
    masks = []
    for probe_id in probe_ids:
        if probe_id not in PROBE_BY_ID:
            raise ValueError(f"unknown support probe: {probe_id}")
        masks.append(np.asarray(probe_state_mask(PROBE_BY_ID[probe_id]), dtype=bool))
    return np.stack(masks) if masks else np.zeros((0, RESPONSE_STATE_DIM), dtype=bool)


def compute_personalization_confidence(
    theta_uncertainty: np.ndarray,
    population_uncertainty: np.ndarray,
    support_state_masks: np.ndarray,
    root_sigma: np.ndarray,
    information_diagonal: np.ndarray | None = None,
) -> PersonalizationConfidence:
    uncertainty = np.asarray(theta_uncertainty, dtype=np.float64)
    prior = np.asarray(population_uncertainty, dtype=np.float64)
    masks = np.asarray(support_state_masks, dtype=bool)
    if uncertainty.shape != (6,) or prior.shape != (6,):
        raise ValueError("theta uncertainty vectors must have shape [6]")
    if masks.ndim != 2 or masks.shape[1] != 6:
        raise ValueError("support_state_masks must have shape [K,6]")
    if np.any(uncertainty < 0.0) or np.any(prior <= 0.0):
        raise ValueError("uncertainty must be non-negative and population scale positive")
    support_count = masks.sum(axis=0).astype(np.float64)
    coverage = 1.0 - np.exp(-support_count / 1.5)
    uncertainty_confidence = np.clip(1.0 - uncertainty / np.maximum(prior, 1e-6), 0.0, 1.0)
    if information_diagonal is None:
        observability = coverage.copy()
    else:
        information = np.asarray(information_diagonal, dtype=np.float64)
        if information.shape != (6,) or np.any(information < 0.0):
            raise ValueError("information_diagonal must be non-negative with shape [6]")
        observability = information / (information + 1.0)
    root_sigma_xy = np.linalg.norm(np.asarray(root_sigma, dtype=np.float64)[..., :2], axis=-1).mean()
    root_confidence = float(np.exp(-root_sigma_xy / 0.30))
    confidence = root_confidence * (
        0.50 * uncertainty_confidence + 0.30 * coverage + 0.20 * observability
    )
    confidence *= (support_count > 0.0)
    # Phase 4B.7 consistently found adaptation poorly identifiable.  It may earn
    # confidence from observations, but cannot inherit confidence from other dims.
    confidence[5] *= 0.35
    return PersonalizationConfidence(
        np.clip(confidence, 0.0, 1.0).astype(np.float32),
        np.clip(coverage, 0.0, 1.0).astype(np.float32),
        np.clip(uncertainty_confidence, 0.0, 1.0).astype(np.float32),
        np.clip(observability, 0.0, 1.0).astype(np.float32), root_confidence,
    )


def action_personalization_confidence(
    action: int | RobotAction, dimension_confidence: np.ndarray,
) -> float:
    action = RobotAction(int(action))
    confidence = np.asarray(dimension_confidence, dtype=np.float64)
    if confidence.shape != (6,) or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("dimension_confidence must have shape [6] in [0,1]")
    dimensions = ACTION_RESPONSE_DIMENSIONS[action]
    if not dimensions:
        return 1.0
    # Geometric mean makes one unobservable action-relevant dimension matter;
    # unrelated high-confidence dimensions cannot mask it.
    return float(np.prod(np.maximum(confidence[list(dimensions)], 1e-6)) ** (1.0 / len(dimensions)))


def shrink_functional_state(
    personalized: np.ndarray, population: np.ndarray, confidence: np.ndarray,
) -> np.ndarray:
    personal = np.asarray(personalized, dtype=np.float64)
    generic = np.asarray(population, dtype=np.float64)
    weight = np.asarray(confidence, dtype=np.float64)
    if personal.shape != (6,) or generic.shape != (6,) or weight.shape != (6,):
        raise ValueError("functional states and confidence must have shape [6]")
    if np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("confidence must be in [0,1]")
    return (weight * personal + (1.0 - weight) * generic).astype(np.float32)


@dataclass(frozen=True)
class DecisionMargin:
    best_index: int
    second_index: int
    best_action: int
    second_action: int
    absolute_margin: float
    relative_margin: float


def decision_margin(action_ids: np.ndarray, costs: np.ndarray, feasible_mask: np.ndarray) -> DecisionMargin:
    actions = np.asarray(action_ids, dtype=int); values = np.asarray(costs, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)
    if actions.shape != values.shape or actions.shape != feasible.shape or feasible.sum() < 2:
        raise ValueError("decision margin needs at least two feasible aligned candidates")
    valid = np.flatnonzero(feasible)
    # Stable action-ID tie break makes the result candidate-order invariant.
    order = valid[np.lexsort((actions[valid], values[valid]))]
    best, second = int(order[0]), int(order[1])
    margin = float(max(values[second] - values[best], 0.0))
    return DecisionMargin(
        best, second, int(actions[best]), int(actions[second]), margin,
        margin / max(abs(float(values[best])), 1e-6),
    )


class SelectiveDecisionMode(str, Enum):
    PERSONALIZED = "PERSONALIZED"
    SHRUNK_PERSONALIZED = "SHRUNK_PERSONALIZED"
    GENERIC_SAFE = "GENERIC_SAFE"
    RULE_FALLBACK = "RULE_FALLBACK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class SelectiveDecision:
    selected_index: int | None
    selected_action: int | None
    mode: SelectiveDecisionMode
    generic_index: int | None
    selective_index: int | None
    switch_allowed: bool
    reason: str


def selective_personalization_select(
    action_ids: np.ndarray,
    feasible_action_mask: np.ndarray,
    generic_cost: np.ndarray,
    personalized_cost: np.ndarray,
    shrunk_cost: np.ndarray,
    action_confidence: np.ndarray,
    minimum_confidence: float,
    minimum_margin: float,
    minimum_benefit_probability: float,
    predicted_benefit_probability: np.ndarray,
) -> SelectiveDecision:
    """Arbitrate only inside an immutable feasible set; accepts no GT inputs."""
    actions = np.asarray(action_ids, dtype=int); feasible = np.asarray(feasible_action_mask, dtype=bool)
    arrays = tuple(np.asarray(value) for value in (
        generic_cost, personalized_cost, shrunk_cost, action_confidence,
        predicted_benefit_probability,
    ))
    if any(value.shape != actions.shape for value in arrays) or feasible.shape != actions.shape:
        raise ValueError("all selective-selector arrays must align with action_ids")
    if not feasible.any():
        return SelectiveDecision(None, None, SelectiveDecisionMode.ABSTAIN, None, None, False, "empty_frozen_feasible_set")
    valid = np.flatnonzero(feasible)
    generic = int(valid[np.lexsort((actions[valid], arrays[0][valid]))][0])
    selective = int(valid[np.lexsort((actions[valid], arrays[2][valid]))][0])
    if selective == generic:
        return SelectiveDecision(generic, int(actions[generic]), SelectiveDecisionMode.GENERIC_SAFE, generic, selective, False, "same_generic_and_shrunk_optimum")
    gain = float(arrays[2][generic] - arrays[2][selective])
    allowed = (
        arrays[3][selective] >= minimum_confidence
        and gain >= minimum_margin
        and arrays[4][selective] >= minimum_benefit_probability
    )
    if not allowed:
        return SelectiveDecision(generic, int(actions[generic]), SelectiveDecisionMode.GENERIC_SAFE, generic, selective, False, "personalized_switch_not_confident")
    fully_personal_best = int(valid[np.lexsort((actions[valid], arrays[1][valid]))][0])
    mode = SelectiveDecisionMode.PERSONALIZED if selective == fully_personal_best and arrays[3][selective] >= 0.85 else SelectiveDecisionMode.SHRUNK_PERSONALIZED
    return SelectiveDecision(selective, int(actions[selective]), mode, generic, selective, True, "validated_confident_personalization_switch")


@dataclass(frozen=True)
class SwitchBenefitCalibrator:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    fit_split: str

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise ValueError("switch calibrator may only be fitted on train")


def fit_switch_benefit_calibrator(
    features: np.ndarray, beneficial: np.ndarray, split_name: str,
    iterations: int = 400, learning_rate: float = 0.08,
) -> SwitchBenefitCalibrator:
    """Fit benefit probability, never an action-label policy."""
    if split_name != "train":
        raise ValueError("switch benefit calibrator requires train split")
    x = np.asarray(features, dtype=np.float64); y = np.asarray(beneficial, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("invalid switch calibration arrays")
    mean, scale = x.mean(axis=0), x.std(axis=0); scale = np.where(scale < 1e-6, 1.0, scale)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(iterations):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))
        gradient = design.T @ (probability - y) / len(x) + 1e-3 * weights
        weights -= learning_rate * gradient
    return SwitchBenefitCalibrator(weights, mean, scale, split_name)


def predict_switch_benefit(calibrator: SwitchBenefitCalibrator, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    design = np.column_stack(((x - calibrator.feature_mean) / calibrator.feature_scale, np.ones(len(x))))
    return 1.0 / (1.0 + np.exp(-np.clip(design @ calibrator.coefficients, -30.0, 30.0)))
