"""Oracle future-pose representation for the Phase 5B-v3-R4A sufficiency audit."""
from __future__ import annotations

import torch
from torch import nn

from src.data.skeleton_schema import NUM_JOINTS, root_joint_ids


FUTURE_FRAMES = 10
COORDINATES = 3
POSE_FEATURE_DIM = FUTURE_FRAMES * NUM_JOINTS * COORDINATES
FINAL_CONTEXT_DIM = 128
POSE_BENEFIT_INPUT_DIM = 2 * FINAL_CONTEXT_DIM + POSE_FEATURE_DIM


def root_relative_decision_local_pose(
    future_global: torch.Tensor, robot_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split pelvis-midpoint root and root-relative pose in robot-yaw axes.

    XY is rotated from world axes to decision-time robot-yaw axes. Z remains
    root-relative height. The virtual pelvis midpoint is not an extra joint, so
    all 17 COCO joints remain in the pose tensor.
    """
    if future_global.ndim != 4 or future_global.shape[-3:] != (FUTURE_FRAMES, NUM_JOINTS, COORDINATES):
        raise ValueError("future_global must have shape [B,10,17,3]")
    if robot_yaw.shape != (len(future_global),):
        raise ValueError("robot_yaw must have shape [B]")
    root = (future_global[..., root_joint_ids[0], :] + future_global[..., root_joint_ids[1], :]) / 2
    pose = future_global - root.unsqueeze(-2)
    cosine = torch.cos(robot_yaw).view(-1, 1, 1)
    sine = torch.sin(robot_yaw).view(-1, 1, 1)
    x, y = pose[..., 0], pose[..., 1]
    local = pose.clone()
    local[..., 0] = x * cosine + y * sine
    local[..., 1] = -x * sine + y * cosine
    return root, local


def oracle_pose_delta(candidate_pose: torch.Tensor, generic_pose: torch.Tensor) -> torch.Tensor:
    """Flatten candidate-minus-generic root-relative pose as 510 dimensions."""
    if candidate_pose.ndim != 4 or candidate_pose.shape[-3:] != (FUTURE_FRAMES, NUM_JOINTS, COORDINATES):
        raise ValueError("candidate_pose must have shape [B,10,17,3]")
    if generic_pose.shape != candidate_pose.shape:
        raise ValueError("generic_pose must match candidate_pose")
    return (candidate_pose - generic_pose).flatten(1)


class OraclePoseBenefitReadout(nn.Module):
    """Matched C0/O_POSE head: exactly Linear(766,1)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(POSE_BENEFIT_INPUT_DIM, 1)

    def forward(self, z_i: torch.Tensor, z_g: torch.Tensor, pose_feature: torch.Tensor) -> torch.Tensor:
        if z_i.ndim != 2 or z_i.shape[-1] != FINAL_CONTEXT_DIM:
            raise ValueError("z_i must have shape [B,128]")
        if z_g.shape != z_i.shape:
            raise ValueError("z_g must match z_i [B,128]")
        if pose_feature.shape != (len(z_i), POSE_FEATURE_DIM):
            raise ValueError("pose_feature must have shape [B,510]")
        return self.linear(torch.cat((z_i, z_g, pose_feature), dim=-1)).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": POSE_BENEFIT_INPUT_DIM,
            "pose_feature_dim": POSE_FEATURE_DIM, "layers": ["Linear(766,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "MLP": False, "attention": False, "GNN": False, "ranking_output": False,
        }
