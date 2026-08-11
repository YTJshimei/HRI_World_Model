"""S4b: independently supervised root-LSTM and local-pose Transformer."""

from __future__ import annotations

import torch
from torch import nn

from src.data.skeleton_schema import NUM_JOINTS, hip_joints
from src.models.hybrid_root_pose import observed_root_sequence
from src.models.skeleton_baselines import estimate_last_root
from src.models.skeleton_transformer import SpatialTemporalSkeletonEncoder


def root_constant_velocity_prior(
    history_global: torch.Tensor,
    visibility_mask: torch.Tensor,
    future_frames: int,
) -> torch.Tensor:
    """Vectorized root-only CV prior from the last two observable hip frames."""
    roots, valid = observed_root_sequence(history_global, visibility_mask)
    batch, time_steps, _ = roots.shape
    time_index = torch.arange(time_steps, device=roots.device).view(1, time_steps)
    valid_mask = valid.bool()
    last_index = torch.where(valid_mask, time_index, -1).amax(dim=1)
    previous_mask = valid_mask & (time_index < last_index[:, None])
    previous_index = torch.where(previous_mask, time_index, -1).amax(dim=1)
    batch_index = torch.arange(batch, device=roots.device)
    gathered_last = roots[batch_index, last_index.clamp_min(0)]
    gathered_previous = roots[batch_index, previous_index.clamp_min(0)]
    fallback = estimate_last_root(history_global, visibility_mask.bool())
    last_root = torch.where(last_index[:, None] >= 0, gathered_last, fallback)
    gap = (last_index - previous_index).clamp_min(1).to(roots.dtype)
    velocity = (gathered_last - gathered_previous) / gap[:, None]
    velocity = torch.where(previous_index[:, None] >= 0, velocity, torch.zeros_like(velocity))
    steps = torch.arange(
        1, future_frames + 1, device=roots.device, dtype=roots.dtype
    ).view(1, future_frames, 1)
    return last_root[:, None] + steps * velocity[:, None]


class DecoupledRootPoseModel(nn.Module):
    """Root dynamics and pelvis-aligned pose are predicted by separate branches."""

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
        self.root_residual_head = nn.Linear(root_hidden_size, future_frames * 3)
        nn.init.zeros_(self.root_residual_head.weight)
        nn.init.zeros_(self.root_residual_head.bias)

    def forward_components(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        roots, root_valid = observed_root_sequence(history_global, visibility_mask)
        anchor = estimate_last_root(history_global, visibility_mask.bool())
        relative_roots = roots - anchor[:, None]
        velocity = torch.diff(roots, dim=1, prepend=roots[:, :1])
        features = torch.cat((relative_roots, velocity, root_valid[..., None]), dim=-1)
        _, (hidden, _) = self.root_encoder(features)
        residual = self.root_residual_head(hidden[-1]).view(
            history_global.shape[0], self.future_frames, 3
        )
        root_prior = root_constant_velocity_prior(
            history_global, visibility_mask, self.future_frames
        )
        predicted_root = root_prior + residual

        encoded, _ = self.local_encoder(history_global, confidence, visibility_mask)
        predicted_local = self.local_head(encoded).view(
            history_global.shape[0], NUM_JOINTS, self.future_frames, 3
        ).permute(0, 2, 1, 3)
        predicted_pelvis = predicted_local[..., list(hip_joints), :].mean(dim=-2)
        predicted_local = predicted_local - predicted_pelvis[..., None, :]
        return predicted_root, predicted_local

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        root, local = self.forward_components(
            history_global, confidence, visibility_mask
        )
        return root[..., None, :] + local
