"""Model-agnostic adapter contract plus mock and small Phase 5 backbones."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from src.multimodal.context_schema import CONTEXT_DIM,TOKEN_DIMS,TOKEN_ORDER


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


class StructuredTokenProjection(nn.Module):
    """Project each canonical token group independently into backbone space."""
    def __init__(self,hidden_size:int)->None:
        super().__init__();self.hidden_size=int(hidden_size)
        self.projections=nn.ModuleDict({name:nn.Sequential(nn.Linear(TOKEN_DIMS[name],hidden_size),nn.LayerNorm(hidden_size),nn.GELU()) for name in TOKEN_ORDER})
        self.group_embedding=nn.Parameter(torch.zeros(len(TOKEN_ORDER),hidden_size))
        nn.init.normal_(self.group_embedding,std=.02)
    def forward(self,features:torch.Tensor)->torch.Tensor:
        if features.ndim!=2 or features.shape[-1]!=CONTEXT_DIM:raise ValueError(f"features must have shape [B,{CONTEXT_DIM}]")
        tokens=[];offset=0
        for index,name in enumerate(TOKEN_ORDER):
            width=TOKEN_DIMS[name];tokens.append(self.projections[name](features[:,offset:offset+width])+self.group_embedding[index]);offset+=width
        return torch.stack(tokens,dim=1)


class StructuredTokenScaleAlignment(nn.Module):
    """Deterministically match every pseudo-token norm to a frozen native scale."""
    def __init__(self,target_norm:float,eps:float=1e-6)->None:
        super().__init__()
        if not math.isfinite(target_norm) or target_norm<=0:raise ValueError("target_norm must be finite and positive")
        self.register_buffer("target_norm",torch.tensor(float(target_norm),dtype=torch.float32))
        self.eps=float(eps)
    def forward(self,tokens:torch.Tensor)->torch.Tensor:
        if tokens.ndim!=3 or tokens.shape[1]!=len(TOKEN_ORDER):raise ValueError(f"tokens must have shape [B,{len(TOKEN_ORDER)},D]")
        norm=tokens.float().norm(dim=-1,keepdim=True)
        direction=tokens/(norm.to(tokens.dtype)+self.eps)
        return direction*self.target_norm.to(device=tokens.device,dtype=tokens.dtype)


def native_embedding_norm_statistics(backbone:nn.Module,chunk_size:int=2048)->dict[str,float]:
    """Read-only row-norm statistics; no text/token data are used."""
    weight=backbone.get_input_embeddings().weight
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(weight),chunk_size):
            chunks.append(weight[start:start+chunk_size].float().norm(dim=-1).cpu())
    values=torch.cat(chunks)
    return {"mean":float(values.mean()),"median":float(values.median()),"P5":float(torch.quantile(values,.05)),"P95":float(torch.quantile(values,.95))}


class FrozenQwen25VLContextAdapter(LargeContextAdapter):
    """Model-specific logic isolated behind the generic context adapter API."""
    MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
    def __init__(self,backbone:nn.Module,hidden_size:int,native_embedding_stats:dict[str,float]|None=None)->None:
        super().__init__();self.backbone=backbone;self.projection=StructuredTokenProjection(hidden_size)
        self.native_embedding_stats=native_embedding_stats or native_embedding_norm_statistics(backbone)
        self.scale_alignment=StructuredTokenScaleAlignment(self.native_embedding_stats["median"])
        self.scale_alignment_enabled=True
        self.benefit=nn.Linear(hidden_size,1);self.uncertainty=nn.Linear(hidden_size,1);self.harm=nn.Linear(hidden_size,1);self.auxiliary=nn.Linear(hidden_size,6)
        for parameter in self.auxiliary.parameters():parameter.requires_grad_(False)
        self.freeze_backbone()
    @classmethod
    def from_pretrained_4bit(cls,model_id:str=MODEL_ID,device_map:dict|str|None=None,cache_dir:str|None=None,local_files_only:bool=False)->"FrozenQwen25VLContextAdapter":
        if model_id!=cls.MODEL_ID:raise ValueError("Stage C-0 is approved only for Qwen/Qwen2.5-VL-3B-Instruct")
        from transformers import BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
        config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
        backbone=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,quantization_config=config,device_map=device_map or {"":0},cache_dir=cache_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=local_files_only)
        hidden_size=int(backbone.config.text_config.hidden_size)
        return cls(backbone,hidden_size,native_embedding_norm_statistics(backbone))
    def freeze_backbone(self)->None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():parameter.requires_grad_(False)
    @property
    def backbone_fully_frozen(self)->bool:return all(not parameter.requires_grad for parameter in self.backbone.parameters())
    def train(self,mode:bool=True):
        super().train(mode);self.backbone.eval();return self
    def encode(self,features:torch.Tensor)->torch.Tensor:
        projected=self.projection(features)
        if self.scale_alignment_enabled:projected=self.scale_alignment(projected)
        device=next(self.backbone.parameters()).device
        dtype=self.backbone.get_input_embeddings().weight.dtype
        inputs=projected.to(device=device,dtype=dtype)
        attention=torch.ones(inputs.shape[:2],device=device,dtype=torch.long)
        output=self.backbone(inputs_embeds=inputs,attention_mask=attention,use_cache=False,output_hidden_states=True,return_dict=True)
        return output.hidden_states[-1].float().mean(dim=1)
    def forward(self,features:torch.Tensor)->ContextValuePrediction:
        encoded=self.encode(features)
        head_device=self.benefit.weight.device
        encoded=encoded.to(head_device,dtype=self.benefit.weight.dtype)
        return ContextValuePrediction(encoded,self.benefit(encoded).squeeze(-1),self.uncertainty(encoded).squeeze(-1).clamp(-6,3),self.harm(encoded).squeeze(-1),self.auxiliary(encoded))
    def trainable_parameter_groups(self)->dict[str,list[nn.Parameter]]:
        return {"projection":list(self.projection.parameters()),"benefit_head":list(self.benefit.parameters()),"harm_head":list(self.harm.parameters()),"uncertainty_head":list(self.uncertainty.parameters())}
    def trainable_state_dict(self)->dict[str,torch.Tensor]:
        prefixes=("projection.","benefit.","harm.","uncertainty.")
        return {name:value.detach().cpu().clone() for name,value in self.state_dict().items() if name.startswith(prefixes)}
    def load_trainable_state_dict(self,state:dict[str,torch.Tensor])->None:
        expected=set(self.trainable_state_dict())
        if set(state)!=expected:raise ValueError("checkpoint must contain exactly projection and value-head state")
        # Load only the four approved trainable modules.  Asking the complete
        # quantized Qwen module to report ``missing_keys`` is backend/version
        # dependent and can incorrectly reject a valid adapter-only checkpoint.
        # Strict submodule loading leaves every backbone tensor untouched.
        for prefix,module in (("projection.",self.projection),("benefit.",self.benefit),("harm.",self.harm),("uncertainty.",self.uncertainty)):
            subset={name[len(prefix):]:value for name,value in state.items() if name.startswith(prefix)}
            result=module.load_state_dict(subset,strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError(f"trainable checkpoint does not match {prefix[:-1]} structure")
