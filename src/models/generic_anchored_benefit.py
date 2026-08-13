"""Capacity-matched benefit readouts for the Phase 5B-v3 GARA fair test."""
from __future__ import annotations

import torch
from torch import nn


class AbsoluteBenefitReadout(nn.Module):
    """H0 control: one absolute linear score over a frozen 128-D context."""

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        if context_dim != 128:
            raise ValueError("Phase5B-v3 benefit context is frozen at 128 dimensions")
        self.scorer = nn.Linear(context_dim, 1)

    def forward(self, candidate: torch.Tensor, generic: torch.Tensor | None = None) -> torch.Tensor:
        if candidate.ndim != 2 or candidate.shape[-1] != 128:
            raise ValueError("candidate representation must have shape [B,128]")
        return self.scorer(candidate).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "parameterization": "absolute",
            "input_dim": 128, "layers": ["Linear(128,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "extra_projection": False, "extra_normalization": False,
        }


class GenericAnchoredBenefitReadout(nn.Module):
    """H1: shared scorer difference ``f(z_i) - f(z_g)``.

    The same ``nn.Linear`` instance scores both inputs.  Consequently its bias
    cancels algebraically and the generic candidate is exactly zero when the
    two representations are identical.
    """

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        if context_dim != 128:
            raise ValueError("Phase5B-v3 benefit context is frozen at 128 dimensions")
        self.scorer = nn.Linear(context_dim, 1)

    def forward(self, candidate: torch.Tensor, generic: torch.Tensor) -> torch.Tensor:
        if candidate.ndim != 2 or candidate.shape[-1] != 128:
            raise ValueError("candidate representation must have shape [B,128]")
        if generic.shape != candidate.shape:
            raise ValueError("generic representation must match candidate shape [B,128]")
        return (self.scorer(candidate) - self.scorer(generic)).squeeze(-1)

    def score_components(self, candidate: torch.Tensor, generic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate.ndim != 2 or candidate.shape[-1] != 128 or generic.shape != candidate.shape:
            raise ValueError("candidate and generic representations must match [B,128]")
        candidate_score = self.scorer(candidate).squeeze(-1)
        generic_score = self.scorer(generic).squeeze(-1)
        return candidate_score, generic_score, candidate_score - generic_score

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "parameterization": "generic_anchored_relative_advantage",
            "formula": "f(z_i) - f(z_g)", "shared_scorer_object_count": 1,
            "input_dim": 128, "layers": ["Linear(128,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "bias_cancels": True, "anchored_bias": False, "MLP": False,
            "extra_projection": False, "attention": False, "extra_normalization": False,
        }
