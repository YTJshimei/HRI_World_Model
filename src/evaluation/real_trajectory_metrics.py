"""Additional kinematic metrics for real trajectory evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.trajectory_metrics import ade_fde


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def velocity_and_heading_error(
    prediction: Any, target: Any, timestep_seconds: float
) -> tuple[float, float]:
    predicted = _numpy(prediction)
    expected = _numpy(target)
    if predicted.shape != expected.shape or predicted.shape[-1] != 2:
        raise ValueError("prediction 与 target 必须具有相同的 [..., time, 2] shape")
    if predicted.shape[-2] < 2:
        raise ValueError("velocity/heading 指标至少需要两个 future frame")
    if timestep_seconds <= 0:
        raise ValueError("timestep_seconds 必须大于 0")
    predicted_velocity = np.diff(predicted, axis=-2) / timestep_seconds
    target_velocity = np.diff(expected, axis=-2) / timestep_seconds
    velocity_error = np.linalg.norm(predicted_velocity - target_velocity, axis=-1)
    predicted_heading = np.arctan2(predicted_velocity[..., 1], predicted_velocity[..., 0])
    target_heading = np.arctan2(target_velocity[..., 1], target_velocity[..., 0])
    difference = np.arctan2(
        np.sin(predicted_heading - target_heading),
        np.cos(predicted_heading - target_heading),
    )
    return float(velocity_error.mean()), float(np.abs(difference).mean())


def real_trajectory_metrics(
    prediction: Any,
    target: Any,
    timestep_seconds: float,
    inference_latency_ms_per_sample: float,
) -> dict[str, float]:
    ade, fde = ade_fde(prediction, target)
    velocity_error, heading_error = velocity_and_heading_error(
        prediction, target, timestep_seconds
    )
    return {
        "ADE": ade,
        "FDE": fde,
        "velocity_error": velocity_error,
        "heading_error_rad": heading_error,
        "inference_latency_ms_per_sample": float(inference_latency_ms_per_sample),
    }
