"""S2/S3 compact spatial-temporal skeleton Transformers."""

from __future__ import annotations

import torch
from torch import nn

from src.data.skeleton_schema import DEFAULT_HISTORY_FRAMES, NUM_JOINTS
from src.models.skeleton_baselines import (
    joint_constant_velocity_prediction,
    masked_centered_input,
)


class SpatialTemporalSkeletonEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        spatial_layers: int = 1,
        temporal_layers: int = 2,
        history_frames: int = DEFAULT_HISTORY_FRAMES,
    ) -> None:
        super().__init__()
        self.missing_token = nn.Parameter(torch.zeros(NUM_JOINTS, 3))
        self.input_projection = nn.Linear(5, d_model)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, NUM_JOINTS, d_model))
        self.time_embedding = nn.Parameter(torch.zeros(1, history_frames, 1, d_model))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, spatial_layers)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, temporal_layers)
        self.d_model = d_model
        self.history_frames = history_frames

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_global.shape[1] > self.history_frames:
            raise ValueError("输入 history 超过模型配置的 history_frames")
        features, root = masked_centered_input(
            history_global, confidence, visibility_mask, self.missing_token
        )
        tokens = self.input_projection(features)
        tokens = (
            tokens
            + self.joint_embedding
            + self.time_embedding[:, : history_global.shape[1]]
        )
        batch, time, joints, channels = tokens.shape
        spatial = self.spatial_encoder(tokens.reshape(batch * time, joints, channels))
        spatial = spatial.view(batch, time, joints, channels)
        temporal_input = spatial.permute(0, 2, 1, 3).reshape(
            batch * joints, time, channels
        )
        temporal = self.temporal_encoder(temporal_input)
        encoded_joints = temporal[:, -1].view(batch, joints, channels)
        return encoded_joints, root


class SpatialTemporalSkeletonTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        spatial_layers: int = 1,
        temporal_layers: int = 2,
        history_frames: int = 20,
        future_frames: int = 10,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.encoder = SpatialTemporalSkeletonEncoder(
            d_model, nhead, spatial_layers, temporal_layers, history_frames
        )
        self.head = nn.Linear(d_model, future_frames * 3)

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded, root = self.encoder(history_global, confidence, visibility_mask)
        relative = self.head(encoded).view(
            history_global.shape[0], NUM_JOINTS, self.future_frames, 3
        ).permute(0, 2, 1, 3)
        return relative + root[:, None, None, :]


class ResidualSkeletonTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        spatial_layers: int = 1,
        temporal_layers: int = 2,
        history_frames: int = 20,
        future_frames: int = 10,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.encoder = SpatialTemporalSkeletonEncoder(
            d_model, nhead, spatial_layers, temporal_layers, history_frames
        )
        self.head = nn.Linear(d_model, future_frames * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _ = self.encoder(history_global, confidence, visibility_mask)
        residual = self.head(encoded).view(
            history_global.shape[0], NUM_JOINTS, self.future_frames, 3
        ).permute(0, 2, 1, 3)
        prior = joint_constant_velocity_prediction(
            history_global, visibility_mask.bool(), self.future_frames
        )
        return prior + residual
