from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
from torch import nn

from scripts.audit_phase5a_gradient_sources import exact_autograd_contract
from scripts.run_phase5a_frozen3b import benefit_likelihood_with_detached_variance
from src.models.large_context_adapter import CONTEXT_DIM,FrozenQwen25VLContextAdapter


class FakeBackbone(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.embedding=nn.Embedding(8,hidden);self.block=nn.Linear(hidden,hidden);self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden))
    def get_input_embeddings(self):return self.embedding
    def forward(self,inputs_embeds,attention_mask,**kwargs):return SimpleNamespace(hidden_states=(inputs_embeds,self.block(inputs_embeds)))


def fixture():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);prediction=model(torch.randn(4,CONTEXT_DIM));target=torch.randn(4)
    return model,prediction,target


def test_exact_autograd_contract_is_satisfied():
    model,prediction,target=fixture();audit=exact_autograd_contract(model,prediction,target,torch)
    assert audit["passed"]
    assert audit["benefit_likelihood_uncertainty_head_gradient_norm"]==0
    assert audit["benefit_likelihood_benefit_head_gradient_norm"]>0
    assert audit["benefit_likelihood_projection_gradient_norm"]>0
    assert audit["uncertainty_regularizer_uncertainty_head_gradient_norm"]>0


def test_detach_preserves_forward_value_exactly():
    _,prediction,target=fixture();detached=benefit_likelihood_with_detached_variance(prediction.benefit_mean,target,prediction.benefit_log_variance,torch);original=.5*((prediction.benefit_mean-target).square()*torch.exp(-prediction.benefit_log_variance)).mean()
    assert torch.equal(detached,original)


def test_only_likelihood_variance_path_is_detached():
    source=__import__("pathlib").Path("scripts/run_phase5a_frozen3b.py").read_text();body=source[source.index("def benefit_likelihood_with_detached_variance"):source.index("def training_losses")]
    assert "benefit_log_variance.detach()" in body
    assert "benefit_mean.detach()" not in body and "error.detach()" not in body


def test_prior_repairs_and_safety_contracts_remain():
    adapter=__import__("pathlib").Path("src/models/large_context_adapter.py").read_text();dataset=__import__("pathlib").Path("src/multimodal/context_dataset.py").read_text();audit=__import__("pathlib").Path("scripts/audit_phase5a_gradient_sources.py").read_text()
    assert "self.scale_alignment_enabled=True" in adapter
    assert "fit_benefit_normalizer" in dataset and "benefit normalizer may only be fit on the train split" in dataset
    assert "optimizer.step(" not in audit and "materialize_test(" not in audit
