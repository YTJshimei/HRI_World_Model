"""Small Transformer encoder baseline for deterministic trajectory prediction."""

from __future__ import annotations

import torch
from torch import nn


class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2, history_length: int = 20, future_length: int = 10) -> None:
        super().__init__()
        self.future_length = future_length
        self.input_projection = nn.Linear(2, d_model)
        self.position = nn.Parameter(torch.zeros(1, history_length, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, future_length * 2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[1] > self.position.shape[1]:
            raise ValueError("历史序列长度超过模型配置")
        encoded = self.encoder(self.input_projection(history) + self.position[:, : history.shape[1]])
        return self.head(encoded[:, -1]).view(history.shape[0], self.future_length, 2)
