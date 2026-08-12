"""Lightweight decision-critical distance residual and risk calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


SAFETY_FEATURE_DIM = 20


class SafetyResidualHead(nn.Module):
    """Small MLP; it does not replace or enlarge the world-model backbone."""

    def __init__(self, future_frames: int = 10, hidden_size: int = 64) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.network = nn.Sequential(
            nn.Linear(SAFETY_FEATURE_DIM, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.distance_residual = nn.Linear(hidden_size, future_frames)
        self.distance_log_std = nn.Linear(hidden_size, future_frames)
        self.minimum_residual = nn.Linear(hidden_size, 1)
        self.minimum_log_std = nn.Linear(hidden_size, 1)
        self.unsafe_logit = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != SAFETY_FEATURE_DIM:
            raise ValueError("safety features must have shape [B,20]")
        encoded = self.network(features)
        return {
            "distance_residual": self.distance_residual(encoded),
            "distance_log_std": self.distance_log_std(encoded).clamp(-5.0, 1.5),
            "minimum_residual": self.minimum_residual(encoded).squeeze(-1),
            "minimum_log_std": self.minimum_log_std(encoded).squeeze(-1).clamp(-5.0, 1.5),
            "unsafe_logit": self.unsafe_logit(encoded).squeeze(-1),
        }


def safety_features(
    human_history: np.ndarray,
    robot_history: np.ndarray,
    action_feature: np.ndarray,
    predicted_distance: np.ndarray,
    predicted_action_effect: np.ndarray,
    theta_hat: np.ndarray,
    theta_uncertainty: np.ndarray,
) -> np.ndarray:
    from src.data.skeleton_schema import compute_root
    root = compute_root(np.asarray(human_history))
    velocity = (root[-1, :2] - root[-2, :2]) * 10.0
    distance = np.asarray(predicted_distance)
    effect_magnitude = float(np.linalg.norm(predicted_action_effect, axis=-1).mean())
    feature = np.concatenate((
        np.asarray((
            np.linalg.norm(velocity), robot_history[-1, 3], robot_history[-1, 4],
            robot_history[-1, 5], robot_history[-1, 6], distance[0],
            distance.min(), distance[-1], float(distance.argmin()) / max(len(distance) - 1, 1),
            effect_magnitude,
        )),
        np.asarray(action_feature)[:4],
        np.asarray(theta_hat)[[0, 1, 2]],
        np.asarray((np.mean(theta_uncertainty), theta_uncertainty[1], theta_uncertainty[5])),
    )).astype(np.float32)
    if feature.shape != (SAFETY_FEATURE_DIM,) or not np.isfinite(feature).all():
        raise ValueError(f"invalid safety feature shape/value: {feature.shape}")
    return feature


@dataclass(frozen=True)
class SafetyCalibration:
    distance_sigma_scale: float
    minimum_sigma_scale: float
    unsafe_temperature: float
    unsafe_threshold: float
    lcb_multiplier: float
    validation_split: str

    def __post_init__(self) -> None:
        if self.validation_split != "validation":
            raise ValueError("safety calibration must use validation split only")
        if min(self.distance_sigma_scale, self.minimum_sigma_scale, self.unsafe_temperature) <= 0:
            raise ValueError("calibration scales must be positive")


def apply_safety_calibration(
    output: dict[str, np.ndarray], calibration: SafetyCalibration,
) -> dict[str, np.ndarray]:
    distance_sigma = np.exp(output["distance_log_std"]) * calibration.distance_sigma_scale
    minimum_sigma = np.exp(output["minimum_log_std"]) * calibration.minimum_sigma_scale
    logits = output["unsafe_logit"] / calibration.unsafe_temperature
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return {
        **output, "sigma_distance": distance_sigma,
        "sigma_minimum": minimum_sigma, "p_unsafe": probability,
    }


def worst_case_regret(regret: np.ndarray) -> dict[str, float]:
    values = np.asarray(regret, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("regret must be a non-empty vector")
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "P90": float(np.percentile(values, 90)),
        "P95": float(np.percentile(values, 95)), "maximum": float(values.max()),
    }
