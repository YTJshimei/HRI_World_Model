"""Diagnostic S4 model with explicitly separated root and local-pose branches."""

from __future__ import annotations

import torch
from torch import nn

from src.data.skeleton_schema import NUM_JOINTS, hip_joints
from src.models.skeleton_transformer import SpatialTemporalSkeletonEncoder


def observed_root_sequence(
    history_global: torch.Tensor, visibility_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate each observed root without reading masked joint coordinates."""
    mask = visibility_mask.bool()
    hip_mask = mask[..., list(hip_joints)]
    hip_count = hip_mask.sum(dim=-1)
    hip_sum = (
        history_global[..., list(hip_joints), :] * hip_mask[..., None]
    ).sum(dim=-2)
    hip_root = hip_sum / hip_count.clamp_min(1)[..., None]

    joint_count = mask.sum(dim=-1)
    visible_mean = (history_global * mask[..., None]).sum(dim=-2) / joint_count.clamp_min(1)[..., None]
    zeros = torch.zeros_like(visible_mean)
    fallback = torch.where(joint_count[..., None] > 0, visible_mean, zeros)
    roots = torch.where(hip_count[..., None] > 0, hip_root, fallback)
    valid = (hip_count > 0).to(history_global.dtype)
    return roots, valid


class HybridRootPoseModel(nn.Module):
    """Lightweight root LSTM plus spatial-temporal local-pose Transformer."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        spatial_layers: int = 1,
        temporal_layers: int = 2,
        root_hidden_size: int = 64,
        history_frames: int = 20,
        future_frames: int = 10,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.local_encoder = SpatialTemporalSkeletonEncoder(
            d_model, nhead, spatial_layers, temporal_layers, history_frames
        )
        self.local_head = nn.Linear(d_model, future_frames * 3)
        self.root_encoder = nn.LSTM(7, root_hidden_size, batch_first=True)
        self.root_head = nn.Linear(root_hidden_size, future_frames * 3)

    def forward_components(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        roots, root_valid = observed_root_sequence(history_global, visibility_mask)
        last_root = roots[:, -1]
        relative_roots = roots - last_root[:, None]
        velocity = torch.diff(roots, dim=1, prepend=roots[:, :1])
        root_features = torch.cat((relative_roots, velocity, root_valid[..., None]), dim=-1)
        _, (root_hidden, _) = self.root_encoder(root_features)
        root_displacement = self.root_head(root_hidden[-1]).view(
            history_global.shape[0], self.future_frames, 3
        )
        predicted_root = last_root[:, None] + root_displacement

        encoded_joints, _ = self.local_encoder(
            history_global, confidence, visibility_mask
        )
        predicted_local = self.local_head(encoded_joints).view(
            history_global.shape[0], NUM_JOINTS, self.future_frames, 3
        ).permute(0, 2, 1, 3)
        pelvis = predicted_local[..., list(hip_joints), :].mean(dim=-2)
        predicted_local = predicted_local - pelvis[..., None, :]
        return predicted_root, predicted_local

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        predicted_root, predicted_local = self.forward_components(
            history_global, confidence, visibility_mask
        )
        return predicted_root[..., None, :] + predicted_local
