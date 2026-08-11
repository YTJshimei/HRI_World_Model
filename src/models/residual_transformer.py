"""A compact Transformer that corrects a constant-velocity prior."""

from __future__ import annotations

import torch
from torch import nn

from src.models.normalized_trajectory import local_coordinates


def constant_velocity_prediction(history: torch.Tensor, future_length: int) -> torch.Tensor:
    velocity = history[:, -1] - history[:, -2]
    steps = torch.arange(
        1, future_length + 1, device=history.device, dtype=history.dtype
    ).view(1, -1, 1)
    return history[:, -1:, :] + steps * velocity[:, None, :]


class ResidualTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        history_length: int = 20,
        future_length: int = 10,
        scale_by_speed: bool = False,
    ) -> None:
        super().__init__()
        self.future_length = future_length
        self.scale_by_speed = scale_by_speed
        self.input_projection = nn.Linear(2, d_model)
        self.position = nn.Parameter(torch.zeros(1, history_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, future_length * 2)
        # Zero residual at initialization makes M0 the exact initial prediction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_residual(self, history: torch.Tensor) -> torch.Tensor:
        local_history, _, _, scale = local_coordinates(
            history, scale_by_speed=self.scale_by_speed
        )
        encoded = self.encoder(
            self.input_projection(local_history)
            + self.position[:, : local_history.shape[1]]
        )
        normalized_residual = self.head(encoded[:, -1]).view(
            history.shape[0], self.future_length, 2
        )
        return normalized_residual * scale

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        prior = constant_velocity_prediction(history, self.future_length)
        return prior + self.forward_residual(history)
