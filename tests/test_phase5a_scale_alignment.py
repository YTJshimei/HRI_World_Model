from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip("torch")
from torch import nn

from src.models.large_context_adapter import CONTEXT_DIM,FrozenQwen25VLContextAdapter,StructuredTokenScaleAlignment
from src.multimodal.context_schema import TOKEN_ORDER


class FakeBackbone(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.embedding=nn.Embedding(32,hidden);self.block=nn.Linear(hidden,hidden);self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden))
    def get_input_embeddings(self):return self.embedding
    def forward(self,inputs_embeds,attention_mask,**kwargs):return SimpleNamespace(hidden_states=(inputs_embeds,self.block(inputs_embeds)))


def test_aligned_token_shape_and_norm():
    alignment=StructuredTokenScaleAlignment(1.009);tokens=torch.randn(3,9,2048);aligned=alignment(tokens)
    assert aligned.shape==(3,9,2048)
    assert torch.allclose(aligned.norm(dim=-1),torch.full((3,9),1.009),atol=1e-5)


def test_zero_vector_alignment_is_finite():
    aligned=StructuredTokenScaleAlignment(1.009)(torch.zeros(2,9,2048))
    assert torch.isfinite(aligned).all() and torch.count_nonzero(aligned)==0


def test_token_order_is_unchanged():
    assert TOKEN_ORDER==("skeleton","motion","robot","functional","candidate","uncertainty","diagnostic","interaction","scene")


def test_candidate_batch_permutation_remains_consistent():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);model.eval();features=torch.randn(4,CONTEXT_DIM);permutation=torch.tensor((2,0,3,1))
    with torch.inference_mode():first=model(features).benefit_mean;second=model(features[permutation]).benefit_mean
    assert torch.allclose(first[permutation],second,atol=1e-6)


def test_qwen_remains_frozen_and_alignment_has_no_trainable_parameters():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24)
    assert model.backbone_fully_frozen
    assert sum(parameter.numel() for parameter in model.scale_alignment.parameters())==0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_scale_audit_forward_does_not_change_parameter_values():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);before={name:value.detach().clone() for name,value in model.state_dict().items()}
    output=model(torch.randn(2,CONTEXT_DIM));torch.autograd.grad(output.benefit_mean.sum(),tuple(parameter for parameter in model.parameters() if parameter.requires_grad),allow_unused=True)
    after=model.state_dict()
    assert before.keys()==after.keys()
    assert all(torch.equal(value,after[name]) for name,value in before.items())


def test_scale_alignment_does_not_change_schema_or_normalizer_source():
    from pathlib import Path
    model_source=Path("src/models/large_context_adapter.py").read_text()
    training_source=Path("scripts/run_phase5a_frozen3b.py").read_text()
    assert "StructuredTokenScaleAlignment" in model_source
    assert 'benefit = np.asarray([target.benefit for target in development["train_targets"]]' in training_source
    assert 'feasible_benefit' not in training_source


def test_scale_audit_contract_has_no_test_or_optimizer_step():
    from pathlib import Path
    source=Path("scripts/audit_phase5a_gradient_sources.py").read_text()
    assert "materialize_test(" not in source and "optimizer.step(" not in source
