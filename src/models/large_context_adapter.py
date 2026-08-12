"""Model-agnostic adapter contract plus mock and small Phase 5 backbones."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.multimodal.context_schema import TOKEN_DIMS,TOKEN_ORDER


CONTEXT_DIM=sum(TOKEN_DIMS.values())


@dataclass
class ContextValuePrediction:
    context_embedding: torch.Tensor
    benefit_mean: torch.Tensor
    benefit_log_variance: torch.Tensor
    harm_logit: torch.Tensor
    auxiliary: torch.Tensor


class LargeContextAdapter(nn.Module):
    """Stable interface; a future 3B backend plugs in behind ``encode``."""
    def encode(self,features:torch.Tensor)->torch.Tensor:raise NotImplementedError
    def forward(self,features:torch.Tensor)->ContextValuePrediction:raise NotImplementedError


class _Heads(LargeContextAdapter):
    def __init__(self,encoder:nn.Module,embedding_dim:int)->None:
        super().__init__();self.encoder=encoder
        self.benefit=nn.Linear(embedding_dim,1);self.uncertainty=nn.Linear(embedding_dim,1)
        self.harm=nn.Linear(embedding_dim,1);self.auxiliary=nn.Linear(embedding_dim,6)
    def encode(self,features:torch.Tensor)->torch.Tensor:
        if features.ndim!=2 or features.shape[-1]!=CONTEXT_DIM:raise ValueError(f"features must have shape [B,{CONTEXT_DIM}]")
        return self.encoder(features)
    def forward(self,features:torch.Tensor)->ContextValuePrediction:
        encoded=self.encode(features)
        return ContextValuePrediction(encoded,self.benefit(encoded).squeeze(-1),self.uncertainty(encoded).squeeze(-1).clamp(-6,3),self.harm(encoded).squeeze(-1),self.auxiliary(encoded))


class MockLargeContextBackbone(_Heads):
    """Pipeline-only mock; never a formal result model."""
    def __init__(self)->None:
        encoder=nn.Sequential(nn.Linear(CONTEXT_DIM,32),nn.Tanh())
        super().__init__(encoder,32)


class SmallContextNetwork(_Heads):
    """L1 baseline using the exact future L2 structured context."""
    def __init__(self,hidden_size:int=128)->None:
        encoder=nn.Sequential(nn.Linear(CONTEXT_DIM,hidden_size),nn.LayerNorm(hidden_size),nn.GELU(),nn.Dropout(.05),nn.Linear(hidden_size,hidden_size),nn.GELU())
        super().__init__(encoder,hidden_size)
