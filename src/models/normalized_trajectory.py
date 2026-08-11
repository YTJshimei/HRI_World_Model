"""Local-coordinate wrappers for the unchanged v1 neural baselines."""

from __future__ import annotations

import torch
from torch import nn


def local_coordinates(
    history: torch.Tensor, future: torch.Tensor | None = None, scale_by_speed: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    origin = history[:, -1:, :]
    if scale_by_speed:
        mean_velocity = (history[:, -1] - history[:, 0]) / (history.shape[1] - 1)
        scale = mean_velocity.norm(dim=-1, keepdim=True).clamp_min(1e-3)[:, None, :]
    else:
        scale = torch.ones_like(origin[..., :1])
    local_history = (history - origin) / scale
    local_future = None if future is None else (future - origin) / scale
    return local_history, local_future, origin, scale


class NormalizedTrajectoryPredictor(nn.Module):
    """Make a v1 predictor learn future displacement in a local frame."""

    def __init__(self, predictor: nn.Module, scale_by_speed: bool = False) -> None:
        super().__init__()
        self.predictor = predictor
        self.scale_by_speed = scale_by_speed

    def forward_relative(self, history: torch.Tensor) -> torch.Tensor:
        local_history, _, _, _ = local_coordinates(history, scale_by_speed=self.scale_by_speed)
        return self.predictor(local_history)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        local_history, _, origin, scale = local_coordinates(
            history, scale_by_speed=self.scale_by_speed
        )
        relative_future = self.predictor(local_history)
        return origin + relative_future * scale
