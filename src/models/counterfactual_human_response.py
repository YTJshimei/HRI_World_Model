"""Phase 5B-v3-R3 counterfactual human-response representation modules."""
from __future__ import annotations

import torch
from torch import nn


HUMAN_CONTEXT_DIM = 1024
CANDIDATE_CONTEXT_DIM = 256
RESPONSE_INPUT_DIM = HUMAN_CONTEXT_DIM + CANDIDATE_CONTEXT_DIM
FUTURE_FRAMES = 10
ROOT_COORDINATES = 2
RESPONSE_FEATURE_DIM = FUTURE_FRAMES * ROOT_COORDINATES
BENEFIT_INPUT_DIM = 128 + 128 + RESPONSE_FEATURE_DIM


def decision_local_coordinates(
    future_xy: torch.Tensor, current_human_xy: torch.Tensor, robot_yaw: torch.Tensor,
) -> torch.Tensor:
    """Transform world XY points to the decision-time human-origin/robot-yaw frame."""
    if future_xy.ndim < 2 or future_xy.shape[-1] != 2:
        raise ValueError("future_xy must end in [H,2]")
    if current_human_xy.shape != future_xy.shape[:-2] + (2,):
        raise ValueError("current_human_xy must match the leading future dimensions")
    if robot_yaw.shape != future_xy.shape[:-2]:
        raise ValueError("robot_yaw must match the leading future dimensions")
    delta = future_xy - current_human_xy.unsqueeze(-2)
    cosine = torch.cos(robot_yaw).unsqueeze(-1)
    sine = torch.sin(robot_yaw).unsqueeze(-1)
    local_x = delta[..., 0] * cosine + delta[..., 1] * sine
    local_y = -delta[..., 0] * sine + delta[..., 1] * cosine
    return torch.stack((local_x, local_y), dim=-1)


def counterfactual_delta(candidate_future: torch.Tensor, generic_future: torch.Tensor) -> torch.Tensor:
    """Return the only new R3 Benefit feature as a flattened 20-D trajectory delta."""
    expected = candidate_future.shape[:-2] + (FUTURE_FRAMES, ROOT_COORDINATES)
    if candidate_future.shape != expected or generic_future.shape != candidate_future.shape:
        raise ValueError("candidate and generic futures must match [...,10,2]")
    return (candidate_future - generic_future).flatten(-2)


class HumanResponseFutureDecoder(nn.Module):
    """Fixed 1280->128->20 local-root future decoder."""

    def __init__(self, input_dim: int = RESPONSE_INPUT_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim != RESPONSE_INPUT_DIM or hidden_dim != 128:
            raise ValueError("R3 response architecture is frozen at 1280->128->20")
        self.network = nn.Sequential(
            nn.Linear(RESPONSE_INPUT_DIM, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, RESPONSE_FEATURE_DIM),
        )

    def forward(self, z_human: torch.Tensor, z_candidate: torch.Tensor) -> torch.Tensor:
        if z_human.ndim != 2 or z_human.shape[-1] != HUMAN_CONTEXT_DIM:
            raise ValueError("z_human must have shape [B,1024]")
        if z_candidate.ndim != 2 or z_candidate.shape != (len(z_human), CANDIDATE_CONTEXT_DIM):
            raise ValueError("z_candidate must have shape [B,256]")
        return self.network(torch.cat((z_human, z_candidate), dim=-1)).view(
            len(z_human), FUTURE_FRAMES, ROOT_COORDINATES
        )

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": RESPONSE_INPUT_DIM,
            "hidden_dim": 128, "output_dim": RESPONSE_FEATURE_DIM,
            "layers": ["Linear(1280,128)", "GELU", "Linear(128,20)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "transformer": False, "recurrent": False, "attention": False,
        }


class MatchedBenefitReadout(nn.Module):
    """Shared C0/C1/O1 architecture: exactly Linear(276,1)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(BENEFIT_INPUT_DIM, 1)

    def forward(self, z_i: torch.Tensor, z_g: torch.Tensor, response_feature: torch.Tensor) -> torch.Tensor:
        if z_i.ndim != 2 or z_i.shape[-1] != 128:
            raise ValueError("z_i must have shape [B,128]")
        if z_g.shape != z_i.shape:
            raise ValueError("z_g must match z_i [B,128]")
        if response_feature.shape != (len(z_i), RESPONSE_FEATURE_DIM):
            raise ValueError("response_feature must have shape [B,20]")
        return self.linear(torch.cat((z_i, z_g, response_feature), dim=-1)).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": BENEFIT_INPUT_DIM,
            "layers": ["Linear(276,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "ranking_output": False, "MLP": False, "attention": False,
        }
