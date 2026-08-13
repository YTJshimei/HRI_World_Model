"""Independent harm-v2 head used with a frozen Phase 5B representation."""
from __future__ import annotations

from torch import nn


class IndependentHarmV2Head(nn.Module):
    """Architecture-control copy of the frozen historical ``Linear(128, 1)`` head."""

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        if context_dim != 128:
            raise ValueError("Phase5B-1.7E context dimension is frozen at 128")
        self.linear = nn.Linear(context_dim, 1)

    def forward(self, context):
        if context.ndim != 2 or context.shape[-1] != 128:
            raise ValueError("context must have shape [B,128]")
        return self.linear(context).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": 128, "output_dim": 1,
            "layers": ["Linear(128,1)"], "activation": None,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "architecture_control": "exact capacity/structure of frozen historical model.harm",
            "historical_weights_loaded": False,
        }


class MinimalNonlinearHarmV2Probe(nn.Module):
    """The sole preregistered nonlinear readout for Phase 5B-1.7E-A."""

    def __init__(self, context_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        if (context_dim, hidden_dim) != (128, 32):
            raise ValueError("Phase5B-1.7E-A probe is frozen at 128->32->1")
        self.network = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, context):
        if context.ndim != 2 or context.shape[-1] != 128:
            raise ValueError("context must have shape [B,128]")
        return self.network(context).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": 128, "hidden_dim": 32, "output_dim": 1,
            "layers": ["Linear(128,32)", "GELU", "Linear(32,1)"],
            "dropout": False, "batch_norm": False, "attention": False, "residual": False,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "hyperparameter_search": False,
        }


class RiskPreservingBypassHead(nn.Module):
    """Single linear Phase 5B-1.7E-D readout over the frozen bypass input."""

    INPUT_DIM = 1408

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        if input_dim != self.INPUT_DIM:
            raise ValueError("Phase5B-1.7E-D bypass input is frozen at 1408 dimensions")
        self.linear = nn.Linear(self.INPUT_DIM, 1)

    def forward(self, value):
        if value.ndim != 2 or value.shape[-1] != self.INPUT_DIM:
            raise ValueError("bypass input must have shape [B,1408]")
        return self.linear(value).squeeze(-1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "model": type(self).__name__, "input_dim": self.INPUT_DIM, "output_dim": 1,
            "layers": ["Linear(1408,1)"], "activation": None, "projection": False,
            "MLP": False, "attention": False, "normalization": False, "learned_gate": False,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
