"""Runtime-valid scalar disturbance advantage for Phase 5B-v3-R4C."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.robot_action_schema import ACTION_DEFINITIONS


CONTEXT_DIM = 128
FEATURE_DIM = 1
INPUT_DIM = 2 * CONTEXT_DIM + FEATURE_DIM
DISTURBANCE_WEIGHT = 0.55


def robot_action_disturbance(action_id: int) -> float:
    """Return the directly known robot-action part of the formal disturbance.

    This function deliberately accepts only an action ID.  It has no access to
    trajectories, labels, costs, harm, profile IDs, or simulator state.
    """
    action_id = int(action_id)
    if action_id not in ACTION_DEFINITIONS:
        raise ValueError(f"unknown action ID: {action_id}")
    action = ACTION_DEFINITIONS[action_id]
    raw = (
        0.30 * abs(float(action.speed_scale_delta)) / 0.10
        + 0.25 * abs(float(action.distance_offset_m)) / 0.20
        + 0.20 * abs(float(action.lateral_offset_m)) / 0.20
    )
    return DISTURBANCE_WEIGHT * raw


def runtime_disturbance_advantage(
    candidate_action_ids: np.ndarray, generic_action_ids: np.ndarray,
) -> np.ndarray:
    """Compute ``D_robot(g)-D_robot(i)`` as a canonical ``[B,1]`` array."""
    candidate = np.asarray(candidate_action_ids)
    generic = np.asarray(generic_action_ids)
    if candidate.ndim != 1 or generic.shape != candidate.shape:
        raise ValueError("candidate and generic action IDs must have matching shape [B]")
    value = np.asarray(
        [robot_action_disturbance(g) - robot_action_disturbance(i) for i, g in zip(candidate, generic)],
        dtype=np.float32,
    )
    if not np.isfinite(value).all():
        raise ValueError("runtime disturbance advantage contains non-finite values")
    return value[:, None]


class DisturbanceAdvantageBenefitReadout(nn.Module):
    """Matched C0/C1/O1 readout: exactly ``Linear(257,1)``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, 1)

    def forward(
        self, z_i: torch.Tensor, z_g: torch.Tensor, disturbance_feature: torch.Tensor,
    ) -> torch.Tensor:
        if z_i.ndim != 2 or z_i.shape[-1] != CONTEXT_DIM:
            raise ValueError("z_i must have shape [B,128]")
        if z_g.shape != z_i.shape:
            raise ValueError("z_g must match z_i [B,128]")
        if disturbance_feature.shape != (len(z_i), FEATURE_DIM):
            raise ValueError("disturbance_feature must have shape [B,1]")
        return self.linear(torch.cat((z_i, z_g, disturbance_feature), dim=-1)).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": INPUT_DIM,
            "context_dim_each": CONTEXT_DIM, "disturbance_feature_dim": FEATURE_DIM,
            "layers": ["Linear(257,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "MLP": False, "attention": False, "ranking_output": False,
        }
