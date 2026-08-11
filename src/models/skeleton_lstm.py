"""S1 compact LSTM for COCO-17 skeleton prediction."""

from __future__ import annotations

import torch
from torch import nn

from src.data.skeleton_schema import NUM_JOINTS
from src.models.skeleton_baselines import masked_centered_input


class SkeletonLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        future_frames: int = 10,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.missing_token = nn.Parameter(torch.zeros(NUM_JOINTS, 3))
        self.encoder = nn.LSTM(
            NUM_JOINTS * 5, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, future_frames * NUM_JOINTS * 3)

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        features, root = masked_centered_input(
            history_global, confidence, visibility_mask, self.missing_token
        )
        flattened = features.flatten(start_dim=2)
        _, (hidden, _) = self.encoder(flattened)
        relative = self.head(hidden[-1]).view(
            history_global.shape[0], self.future_frames, NUM_JOINTS, 3
        )
        return relative + root[:, None, None, :]
