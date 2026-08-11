"""Observable-history encoder for Phase 4B few-shot personalization."""

from __future__ import annotations

import torch
from torch import nn

from src.data.personal_interaction_memory import OBSERVABLE_INTERACTION_FEATURE_DIM


class PersonalResponseEncoder(nn.Module):
    """Encode a variable number of completed interactions into ``z_person``.

    The input features are observable pre/post interaction summaries. Person IDs
    and simulator latent profile parameters are intentionally absent.
    """

    def __init__(
        self,
        input_size: int = OBSERVABLE_INTERACTION_FEATURE_DIM,
        hidden_size: int = 64,
        latent_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.interaction_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.sequence_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention_score = nn.Linear(hidden_size, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, latent_size), nn.LayerNorm(latent_size), nn.Tanh()
        )
        self.null_person = nn.Parameter(torch.zeros(latent_size))

    def forward(
        self, support_features: torch.Tensor, support_mask: torch.Tensor
    ) -> torch.Tensor:
        if support_features.ndim != 3 or support_features.shape[-1] != self.input_size:
            raise ValueError(
                f"support_features must have shape [B,K,{self.input_size}]"
            )
        if support_mask.shape != support_features.shape[:2]:
            raise ValueError("support_mask must have shape [B,K]")
        mask = support_mask.to(torch.bool)
        encoded = self.interaction_encoder(support_features)
        sequence, _ = self.sequence_encoder(encoded)
        scores = self.attention_score(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        any_support = mask.any(dim=1)
        safe_scores = torch.where(any_support[:, None], scores, torch.zeros_like(scores))
        attention = torch.softmax(safe_scores, dim=1) * mask.to(scores.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (sequence * attention[..., None]).sum(dim=1)
        latent = self.output(pooled)
        null = self.null_person[None].expand(support_features.shape[0], -1)
        return torch.where(any_support[:, None], latent, null)

