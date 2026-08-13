"""Runtime-conditioned scalar episode offset for Phase 5B-v3-R1D."""
from __future__ import annotations

import torch
from torch import nn


class RuntimeConditionedEpisodeOffset(nn.Module):
    """The preregistered single ``Linear(128,1)`` offset head."""

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        if context_dim != 128:
            raise ValueError("RCEOC input is frozen at 128 dimensions")
        self.linear = nn.Linear(context_dim, 1)

    def forward(self, runtime_generic_context: torch.Tensor) -> torch.Tensor:
        if runtime_generic_context.ndim != 2 or runtime_generic_context.shape[-1] != 128:
            raise ValueError("runtime generic context must have shape [E,128]")
        return self.linear(runtime_generic_context).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": 128, "output_dim": 1,
            "layers": ["Linear(128,1)"], "parameter_count": sum(p.numel() for p in self.parameters()),
            "candidate_specific_input": False, "MLP": False, "projection": False,
            "attention": False, "LayerNorm": False, "generic_embedding": False,
        }
