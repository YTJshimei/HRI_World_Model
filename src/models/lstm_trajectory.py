"""LSTM encoder baseline for deterministic trajectory prediction."""

from __future__ import annotations

import torch
from torch import nn


class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, future_length: int = 10) -> None:
        super().__init__()
        self.future_length = future_length
        self.encoder = nn.LSTM(2, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, future_length * 2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(history)
        return self.head(hidden[-1]).view(history.shape[0], self.future_length, 2)
