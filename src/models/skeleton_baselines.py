"""Kinematic utilities and S0 joint constant-velocity baseline."""

from __future__ import annotations

import torch
from torch import nn

from src.data.skeleton_schema import hip_joints


def estimate_last_root_reference(
    history: torch.Tensor, visibility_mask: torch.Tensor
) -> torch.Tensor:
    """Frozen Phase 3A loop implementation retained for equivalence/profiling."""
    batch_size, time_steps, _, _ = history.shape
    roots = []
    for batch in range(batch_size):
        hip_positions = []
        for joint in hip_joints:
            visible_indices = torch.nonzero(visibility_mask[batch, :, joint], as_tuple=False).flatten()
            if len(visible_indices):
                hip_positions.append(history[batch, visible_indices[-1], joint])
        if hip_positions:
            roots.append(torch.stack(hip_positions).mean(dim=0))
            continue
        visible_last = visibility_mask[batch, -1]
        if bool(visible_last.any()):
            roots.append(history[batch, -1, visible_last].mean(dim=0))
        else:
            # This is an explicit no-observation fallback, not a zero-valued joint input.
            roots.append(history.new_zeros(3))
    return torch.stack(roots)


def joint_constant_velocity_prediction_reference(
    history: torch.Tensor, visibility_mask: torch.Tensor, future_frames: int
) -> torch.Tensor:
    """Frozen Phase 3A S0 implementation retained as the diagnostic reference."""
    batch_size, _, joints, _ = history.shape
    fallback_root = estimate_last_root_reference(history, visibility_mask)
    last_positions = history.new_empty((batch_size, joints, 3))
    velocities = history.new_zeros((batch_size, joints, 3))
    for batch in range(batch_size):
        for joint in range(joints):
            indices = torch.nonzero(visibility_mask[batch, :, joint], as_tuple=False).flatten()
            if len(indices) == 0:
                last_positions[batch, joint] = fallback_root[batch]
            else:
                last_positions[batch, joint] = history[batch, indices[-1], joint]
                if len(indices) >= 2:
                    delta_frames = (indices[-1] - indices[-2]).to(history.dtype).clamp_min(1.0)
                    velocities[batch, joint] = (
                        history[batch, indices[-1], joint]
                        - history[batch, indices[-2], joint]
                    ) / delta_frames
    steps = torch.arange(
        1, future_frames + 1, device=history.device, dtype=history.dtype
    ).view(1, -1, 1, 1)
    return last_positions[:, None] + steps * velocities[:, None]


def _gather_joint_positions(history: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one time index for every [batch, joint] without Python loops."""
    gather_index = indices.clamp_min(0)[:, None, :, None].expand(-1, 1, -1, 3)
    return history.gather(1, gather_index).squeeze(1)


def estimate_last_root(history: torch.Tensor, visibility_mask: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of :func:`estimate_last_root_reference`."""
    mask = visibility_mask.bool()
    batch_size, time_steps, joints, coordinates = history.shape
    if coordinates != 3 or mask.shape != (batch_size, time_steps, joints):
        raise ValueError("history/visibility_mask shape mismatch")

    time_index = torch.arange(time_steps, device=history.device).view(1, time_steps, 1)
    hip_mask = mask[..., list(hip_joints)]
    hip_last_index = torch.where(hip_mask, time_index, -1).amax(dim=1)
    hip_positions = _gather_joint_positions(history[..., list(hip_joints), :], hip_last_index)
    hip_valid = hip_last_index.ge(0)
    hip_count = hip_valid.sum(dim=1)
    hip_root = (hip_positions * hip_valid[..., None]).sum(dim=1) / hip_count.clamp_min(1)[..., None]

    visible_last = mask[:, -1]
    visible_count = visible_last.sum(dim=1)
    visible_root = (history[:, -1] * visible_last[..., None]).sum(dim=1) / visible_count.clamp_min(1)[..., None]
    zero_root = history.new_zeros((batch_size, 3))
    fallback = torch.where(visible_count[:, None] > 0, visible_root, zero_root)
    return torch.where(hip_count[:, None] > 0, hip_root, fallback)


def joint_constant_velocity_prediction(
    history: torch.Tensor, visibility_mask: torch.Tensor, future_frames: int
) -> torch.Tensor:
    """Vectorized S0 with exactly the frozen Phase 3A prediction rule."""
    if future_frames <= 0:
        raise ValueError("future_frames must be positive")
    mask = visibility_mask.bool()
    batch_size, time_steps, joints, coordinates = history.shape
    if coordinates != 3 or mask.shape != (batch_size, time_steps, joints):
        raise ValueError("history/visibility_mask shape mismatch")

    time_index = torch.arange(time_steps, device=history.device).view(1, time_steps, 1)
    last_index = torch.where(mask, time_index, -1).amax(dim=1)
    preceding_mask = mask & (time_index < last_index[:, None, :])
    previous_index = torch.where(preceding_mask, time_index, -1).amax(dim=1)

    gathered_last = _gather_joint_positions(history, last_index)
    gathered_previous = _gather_joint_positions(history, previous_index)
    fallback_root = estimate_last_root(history, mask)
    last_positions = torch.where(last_index[..., None] >= 0, gathered_last, fallback_root[:, None])
    delta_frames = (last_index - previous_index).clamp_min(1).to(history.dtype)
    velocities = (gathered_last - gathered_previous) / delta_frames[..., None]
    velocities = torch.where(previous_index[..., None] >= 0, velocities, torch.zeros_like(velocities))

    steps = torch.arange(
        1, future_frames + 1, device=history.device, dtype=history.dtype
    ).view(1, -1, 1, 1)
    return last_positions[:, None] + steps * velocities[:, None]


class JointConstantVelocityReference(nn.Module):
    """Frozen loop implementation, exposed only for before/after diagnostics."""

    def __init__(self, future_frames: int = 10) -> None:
        super().__init__()
        self.future_frames = future_frames

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        del confidence
        return joint_constant_velocity_prediction_reference(
            history_global, visibility_mask.bool(), self.future_frames
        )


class JointConstantVelocity(nn.Module):
    def __init__(self, future_frames: int = 10) -> None:
        super().__init__()
        self.future_frames = future_frames

    def forward(
        self,
        history_global: torch.Tensor,
        confidence: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        del confidence
        return joint_constant_velocity_prediction(
            history_global, visibility_mask.bool(), self.future_frames
        )


def masked_centered_input(
    history_global: torch.Tensor,
    confidence: torch.Tensor,
    visibility_mask: torch.Tensor,
    missing_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = visibility_mask.bool()
    root = estimate_last_root(history_global, mask)
    centered = history_global - root[:, None, None, :]
    safe_coordinates = torch.where(
        mask[..., None], centered, missing_token[None, None, :, :]
    )
    features = torch.cat(
        (safe_coordinates, confidence[..., None], mask.to(history_global.dtype)[..., None]),
        dim=-1,
    )
    return features, root
