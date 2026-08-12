"""Residual calibration of existing transparent decision-cost components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COMPONENTS = ("task", "safety", "human_response", "disturbance")


@dataclass(frozen=True)
class CostResidualCalibration:
    coefficients: np.ndarray
    residual_sigma: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    fit_split: str

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise ValueError("cost residual calibration may only be fitted on train")
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != len(COMPONENTS):
            raise ValueError("coefficients must predict four cost residuals")


def fit_cost_residual_calibrator(
    features: np.ndarray,
    predicted_components: np.ndarray,
    observed_components: np.ndarray,
    split_name: str,
    ridge: float = 1e-3,
) -> CostResidualCalibration:
    """Fit component residuals; no action-choice or oracle-label input exists."""
    if split_name != "train":
        raise ValueError("cost calibrator fit requires train split")
    x = np.asarray(features, dtype=np.float64)
    predicted = np.asarray(predicted_components, dtype=np.float64)
    observed = np.asarray(observed_components, dtype=np.float64)
    if x.ndim != 2 or predicted.shape != observed.shape or predicted.shape[1] != len(COMPONENTS):
        raise ValueError("invalid cost calibration arrays")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ (observed - predicted))
    calibrated = predicted + design @ coefficients
    sigma = np.sqrt(np.mean(np.square(observed - calibrated), axis=0))
    return CostResidualCalibration(coefficients, sigma, mean, scale, split_name)


def apply_cost_residual_calibrator(
    calibration: CostResidualCalibration,
    features: np.ndarray,
    predicted_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    predicted = np.asarray(predicted_components, dtype=np.float64)
    design = np.column_stack(((x - calibration.feature_mean) / calibration.feature_scale, np.ones(len(x))))
    corrected = np.maximum(predicted + design @ calibration.coefficients, 0.0)
    sigma = np.broadcast_to(calibration.residual_sigma, corrected.shape).copy()
    return corrected, sigma
