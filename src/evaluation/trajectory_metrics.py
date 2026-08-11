"""Trajectory prediction metrics and model-size helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def ade_fde(prediction: Any, target: Any) -> tuple[float, float]:
    """Return average and final displacement errors for [..., time, 2] arrays/tensors."""
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(f"prediction 与 target shape 必须一致：{prediction.shape} != {target.shape}")
    if prediction.shape[-1] != 2 or prediction.shape[-2] < 1:
        raise ValueError("输入 shape 必须为 [..., time, 2]，且 time >= 1")
    if hasattr(prediction, "detach"):
        distances = (prediction - target).square().sum(dim=-1).sqrt()
        return float(distances.mean().item()), float(distances[..., -1].mean().item())
    distances = np.linalg.norm(np.asarray(prediction) - np.asarray(target), axis=-1)
    return float(distances.mean()), float(distances[..., -1].mean())


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
