"""Validated model inputs for one-step synthetic counterfactual decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.functional_response_state import RESPONSE_STATE_DIM
from src.decision.candidate_action import CandidateAction, validate_candidate_actions


@dataclass(frozen=True)
class FunctionalResponseBelief:
    theta_hat: np.ndarray
    theta_uncertainty: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.theta_hat).shape != (RESPONSE_STATE_DIM,):
            raise ValueError("theta_hat must have shape [6]")
        if np.asarray(self.theta_uncertainty).shape != (RESPONSE_STATE_DIM,):
            raise ValueError("theta_uncertainty must have shape [6]")
        if not np.isfinite(self.theta_hat).all() or not np.isfinite(self.theta_uncertainty).all():
            raise ValueError("functional response belief must be finite")
        if np.any(np.asarray(self.theta_uncertainty) < 0.0):
            raise ValueError("theta uncertainty cannot be negative")


@dataclass(frozen=True)
class DecisionState:
    human_history: np.ndarray
    robot_history: np.ndarray
    confidence: np.ndarray
    visibility_mask: np.ndarray
    belief: FunctionalResponseBelief
    candidates: tuple[CandidateAction, ...]
    target_follow_distance: float = 1.5
    too_close_distance: float = 0.80
    scenario_id: str = "synthetic"

    def __post_init__(self) -> None:
        if np.asarray(self.human_history).ndim != 3 or self.human_history.shape[-2:] != (17, 3):
            raise ValueError("human_history must have shape [T,17,3]")
        if np.asarray(self.robot_history).ndim != 2 or self.robot_history.shape[-1] != 7:
            raise ValueError("robot_history must have shape [T,7]")
        if not all(np.isfinite(value).all() for value in (
            self.human_history, self.robot_history,
            self.confidence, self.visibility_mask,
        )):
            raise ValueError("decision state contains non-finite values")
        validate_candidate_actions(self.candidates)
