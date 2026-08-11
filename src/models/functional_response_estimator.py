"""Lightweight masked estimator of functional human-response state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.functional_response_state import RESPONSE_STATE_DIM, RESPONSE_STATE_SCALE
from src.data.response_statistics import RESPONSE_STATISTIC_DIM


@dataclass(frozen=True)
class FunctionalResponseEstimate:
    theta_mean: torch.Tensor
    theta_log_std: torch.Tensor
    observed_dimension_mask: torch.Tensor


class FunctionalResponseEstimator(nn.Module):
    """Dimension-wise masked attention over observable interaction statistics."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.statistics_encoder = nn.Sequential(
            nn.Linear(RESPONSE_STATISTIC_DIM, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.dimension_attention = nn.Linear(hidden_size, RESPONSE_STATE_DIM)
        self.mean_heads = nn.ModuleList(
            nn.Linear(hidden_size, 1) for _ in range(RESPONSE_STATE_DIM)
        )
        self.log_std_heads = nn.ModuleList(
            nn.Linear(hidden_size, 1) for _ in range(RESPONSE_STATE_DIM)
        )
        scale = torch.tensor(RESPONSE_STATE_SCALE, dtype=torch.float32)
        self.register_buffer("response_scale", scale)
        self.generic_mean_normalized = nn.Parameter(torch.zeros(RESPONSE_STATE_DIM))
        self.generic_log_std_normalized = nn.Parameter(
            torch.full((RESPONSE_STATE_DIM,), -0.25)
        )

    def set_generic_prior(self, mean_state: torch.Tensor) -> None:
        if mean_state.shape != (RESPONSE_STATE_DIM,):
            raise ValueError("generic prior shape mismatch")
        with torch.no_grad():
            self.generic_mean_normalized.copy_(mean_state / self.response_scale)

    def forward(
        self,
        statistics: torch.Tensor,
        support_mask: torch.Tensor,
        response_state_mask: torch.Tensor,
    ) -> FunctionalResponseEstimate:
        if statistics.ndim != 3 or statistics.shape[-1] != RESPONSE_STATISTIC_DIM:
            raise ValueError("statistics must have shape [B,K,response_statistic_dim]")
        if support_mask.shape != statistics.shape[:2]:
            raise ValueError("support_mask shape mismatch")
        if response_state_mask.shape != (*statistics.shape[:2], RESPONSE_STATE_DIM):
            raise ValueError("response_state_mask shape mismatch")
        valid = response_state_mask.to(torch.bool) & support_mask.to(torch.bool)[..., None]
        encoded = self.statistics_encoder(statistics)
        scores = self.dimension_attention(encoded)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        observed = valid.any(dim=1)
        safe_scores = torch.where(observed[:, None, :], scores, torch.zeros_like(scores))
        attention = torch.softmax(safe_scores, dim=1) * valid.to(scores.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bkh,bkd->bdh", encoded, attention)
        observed_mean = torch.cat(
            [head(pooled[:, index]).reshape(-1, 1) for index, head in enumerate(self.mean_heads)],
            dim=1,
        )
        observed_log_std = torch.cat(
            [head(pooled[:, index]).reshape(-1, 1) for index, head in enumerate(self.log_std_heads)],
            dim=1,
        ).clamp(-4.0, 2.0)
        generic_mean = self.generic_mean_normalized[None].expand_as(observed_mean)
        generic_log_std = self.generic_log_std_normalized[None].expand_as(observed_log_std)
        mean_normalized = torch.where(observed, observed_mean, generic_mean)
        log_std_normalized = torch.where(observed, observed_log_std, generic_log_std)
        theta_mean = mean_normalized * self.response_scale
        theta_log_std = log_std_normalized + torch.log(self.response_scale)
        return FunctionalResponseEstimate(theta_mean, theta_log_std, observed)

