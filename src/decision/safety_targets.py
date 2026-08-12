"""Training/evaluation-only safety targets for synthetic Phase 4C.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyCriticalTargets:
    distance_trajectory: np.ndarray
    minimum_distance: np.ndarray
    time_to_minimum_distance: np.ndarray
    violation_any: np.ndarray
    violation_duration: np.ndarray
    distance_margin: np.ndarray
    distance_at_horizon: np.ndarray


def build_safety_targets_for_training_or_evaluation(
    gt_distance_trajectory: np.ndarray,
    safe_distance: float,
    sample_rate_hz: float = 10.0,
) -> SafetyCriticalTargets:
    """GT-only target builder; not imported by selector or safety gate."""
    distance = np.asarray(gt_distance_trajectory, dtype=np.float32)
    if distance.ndim != 2 or not np.isfinite(distance).all():
        raise ValueError("GT distance trajectory must be finite [A,H]")
    minimum = distance.min(axis=1)
    time_index = distance.argmin(axis=1)
    violation = distance < float(safe_distance)
    return SafetyCriticalTargets(
        distance_trajectory=distance,
        minimum_distance=minimum,
        time_to_minimum_distance=(time_index + 1).astype(np.float32) / sample_rate_hz,
        violation_any=violation.any(axis=1),
        violation_duration=violation.mean(axis=1).astype(np.float32),
        distance_margin=(minimum - safe_distance).astype(np.float32),
        distance_at_horizon=distance[:, -1].astype(np.float32),
    )


def false_safe_rate(predicted_safe: np.ndarray, gt_unsafe: np.ndarray) -> float:
    predicted = np.asarray(predicted_safe, dtype=bool)
    truth = np.asarray(gt_unsafe, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError("false-safe arrays must have matching shapes")
    return float(np.mean(predicted[truth])) if truth.any() else 0.0


def false_veto_rate(predicted_unsafe: np.ndarray, gt_safe: np.ndarray) -> float:
    predicted = np.asarray(predicted_unsafe, dtype=bool)
    truth = np.asarray(gt_safe, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError("false-veto arrays must have matching shapes")
    return float(np.mean(predicted[truth])) if truth.any() else 0.0
