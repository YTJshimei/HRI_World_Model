"""Lightweight belief over a frozen natural human-root forecast.

This module is intentionally separate from the Phase 3/4 human-response
backbones.  It predicts a residual correction and heteroscedastic uncertainty
for the pelvis/root trajectory only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RootFutureBelief:
    mu_root: np.ndarray
    sigma_root: np.ndarray
    aleatoric_sigma: np.ndarray
    epistemic_sigma: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (
            self.mu_root, self.sigma_root,
            self.aleatoric_sigma, self.epistemic_sigma,
        ))
        if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 2:
            raise ValueError("root belief arrays must share shape [H,3]")
        if arrays[0].shape[-1] != 3 or not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("root belief must contain finite xyz trajectories")
        if any(np.any(value < 0.0) for value in arrays[1:]):
            raise ValueError("root uncertainty cannot be negative")


class RootResidualBeliefHead(nn.Module):
    """Small translation-invariant MLP for root residual mean and log sigma."""

    def __init__(self, history_frames: int = 20, future_frames: int = 10, hidden_size: int = 96) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.future_frames = int(future_frames)
        input_size = self.history_frames * 3 + self.future_frames * 3 + 6
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.residual = nn.Linear(hidden_size, self.future_frames * 3)
        self.log_sigma = nn.Linear(hidden_size, self.future_frames * 3)

    def forward(self, history_root: torch.Tensor, frozen_prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        if history_root.ndim != 3 or history_root.shape[1:] != (self.history_frames, 3):
            raise ValueError("history_root must have shape [B,T,3]")
        if frozen_prediction.ndim != 3 or frozen_prediction.shape[1:] != (self.future_frames, 3):
            raise ValueError("frozen_prediction must have shape [B,H,3]")
        origin = history_root[:, -1:, :]
        history_local = history_root - origin
        prediction_local = frozen_prediction - origin
        velocity = history_root[:, -1] - history_root[:, -2]
        acceleration = history_root[:, -1] - 2.0 * history_root[:, -2] + history_root[:, -3]
        features = torch.cat((
            history_local.flatten(1), prediction_local.flatten(1), velocity, acceleration,
        ), dim=-1)
        encoded = self.encoder(features)
        residual = self.residual(encoded).view(-1, self.future_frames, 3)
        log_sigma = self.log_sigma(encoded).view(-1, self.future_frames, 3).clamp(-6.0, 1.0)
        return {"residual": residual, "log_sigma": log_sigma}


def make_root_belief(
    frozen_prediction: np.ndarray,
    residual: np.ndarray,
    aleatoric_sigma: np.ndarray,
    epistemic_sigma: np.ndarray | None = None,
) -> RootFutureBelief:
    prediction = np.asarray(frozen_prediction, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    aleatoric = np.asarray(aleatoric_sigma, dtype=np.float32)
    epistemic = np.zeros_like(aleatoric) if epistemic_sigma is None else np.asarray(epistemic_sigma, dtype=np.float32)
    sigma = np.sqrt(np.square(aleatoric) + np.square(epistemic))
    return RootFutureBelief(prediction + residual, sigma, aleatoric, epistemic)


def root_error_components(
    predicted: np.ndarray, target: np.ndarray, sample_rate_hz: float = 10.0,
) -> dict[str, np.ndarray]:
    """Return interpretable pointwise root errors without changing predictions."""
    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError("predicted and target roots must share shape [H,3]")
    error = prediction - truth
    pred_velocity = np.diff(prediction[:, :2], axis=0, prepend=prediction[:1, :2]) * sample_rate_hz
    true_velocity = np.diff(truth[:, :2], axis=0, prepend=truth[:1, :2]) * sample_rate_hz
    pred_heading = np.arctan2(pred_velocity[:, 1], pred_velocity[:, 0])
    true_heading = np.arctan2(true_velocity[:, 1], true_velocity[:, 0])
    heading = np.abs(np.arctan2(np.sin(pred_heading - true_heading), np.cos(pred_heading - true_heading)))
    pred_turn = np.diff(pred_heading, prepend=pred_heading[:1])
    true_turn = np.diff(true_heading, prepend=true_heading[:1])
    return {
        "position_error": np.linalg.norm(error, axis=-1),
        "systematic_bias_x": error[:, 0],
        "systematic_bias_y": error[:, 1],
        "systematic_bias_z": error[:, 2],
        "velocity_bias": np.linalg.norm(pred_velocity - true_velocity, axis=-1),
        "heading_bias": heading,
        "turn_bias": np.abs(np.arctan2(np.sin(pred_turn - true_turn), np.cos(pred_turn - true_turn))),
        "acceleration_bias": np.linalg.norm(np.diff(pred_velocity, axis=0, prepend=pred_velocity[:1]) - np.diff(true_velocity, axis=0, prepend=true_velocity[:1]), axis=-1) * sample_rate_hz,
    }
