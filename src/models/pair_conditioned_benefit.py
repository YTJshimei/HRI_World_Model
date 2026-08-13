"""Minimal absolute and pair-conditioned Benefit readouts for Phase 5B-v3-R2.

The frozen 128-D candidate and runtime-generic representations are produced by
the same R1-v3 backbone layer.  These heads deliberately contain no hidden
layer, normalization, attention, or ranking path.
"""
from __future__ import annotations

import torch
from torch import nn


CONTEXT_DIM = 128
PAIR_DIM = 2 * CONTEXT_DIM


def prepare_pair_input(candidate: torch.Tensor, generic: torch.Tensor) -> torch.Tensor:
    """Validate runtime representations and concatenate them as ``[z_i, z_g]``."""
    if candidate.ndim != 2 or candidate.shape[-1] != CONTEXT_DIM:
        raise ValueError("candidate representation must have shape [B,128]")
    if generic.shape != candidate.shape:
        raise ValueError("generic representation must match candidate shape [B,128]")
    return torch.cat((candidate, generic), dim=-1)


class AbsoluteCandidateBenefitReadout(nn.Module):
    """A0 fairness control: ``Benefit(z_i) = Linear(z_i)``."""

    def __init__(self, context_dim: int = CONTEXT_DIM) -> None:
        super().__init__()
        if context_dim != CONTEXT_DIM:
            raise ValueError("Phase 5B-v3 Benefit context is frozen at 128 dimensions")
        self.linear = nn.Linear(CONTEXT_DIM, 1)

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        if candidate.ndim != 2 or candidate.shape[-1] != CONTEXT_DIM:
            raise ValueError("candidate representation must have shape [B,128]")
        return self.linear(candidate).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__,
            "formula": "Linear(z_i)",
            "runtime_inputs": ["z_i"],
            "input_dim": CONTEXT_DIM,
            "layers": ["Linear(128,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "MLP": False,
            "attention": False,
            "normalization": False,
            "ranking_output": False,
        }


class PairConditionedBenefitReadout(nn.Module):
    """A1 intervention: ``Benefit(z_i,z_g) = Linear(concat(z_i,z_g))``."""

    def __init__(self, context_dim: int = CONTEXT_DIM) -> None:
        super().__init__()
        if context_dim != CONTEXT_DIM:
            raise ValueError("Phase 5B-v3 Benefit context is frozen at 128 dimensions")
        self.linear = nn.Linear(PAIR_DIM, 1)

    def forward(self, candidate: torch.Tensor, generic: torch.Tensor) -> torch.Tensor:
        return self.linear(prepare_pair_input(candidate, generic)).squeeze(-1)

    def weight_halves(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.linear.weight.squeeze(0)
        return weight[:CONTEXT_DIM], weight[CONTEXT_DIM:]

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__,
            "formula": "Linear(concat(z_i,z_g))",
            "runtime_inputs": ["z_i", "z_g"],
            "input_dim": PAIR_DIM,
            "layers": ["Linear(256,1)"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "MLP": False,
            "attention": False,
            "projection": False,
            "normalization": False,
            "ranking_output": False,
        }
