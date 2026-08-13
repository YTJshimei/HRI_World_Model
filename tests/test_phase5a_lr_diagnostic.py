from pathlib import Path
from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
from torch import nn

from scripts.run_phase5a_frozen3b import gaussian_benefit_likelihood
from src.models.large_context_adapter import CONTEXT_DIM,FrozenQwen25VLContextAdapter


class FakeBackbone(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.embedding=nn.Embedding(8,hidden);self.block=nn.Linear(hidden,hidden);self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden))
    def get_input_embeddings(self):return self.embedding
    def forward(self,inputs_embeds,attention_mask,**kwargs):return SimpleNamespace(hidden_states=(inputs_embeds,self.block(inputs_embeds)))


def fixture():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);prediction=model(torch.randn(4,CONTEXT_DIM));target=torch.randn(4)
    return model,prediction,target


def norm(gradients):return sum(float(gradient.square().sum()) for gradient in gradients if gradient is not None)**.5


def test_formal_gaussian_nll_restores_full_variance_path():
    model,prediction,target=fixture();loss=gaussian_benefit_likelihood(prediction.benefit_mean,target,prediction.benefit_log_variance,torch)
    assert torch.equal(loss,.5*((prediction.benefit_mean-target).square()*torch.exp(-prediction.benefit_log_variance)).mean())
    assert norm(torch.autograd.grad(loss,tuple(model.benefit.parameters()),retain_graph=True,allow_unused=True))>0
    assert norm(torch.autograd.grad(loss,tuple(model.projection.parameters()),retain_graph=True,allow_unused=True))>0
    assert norm(torch.autograd.grad(loss,tuple(model.uncertainty.parameters()),retain_graph=True,allow_unused=True))>0


def test_c_s5_detach_is_not_on_formal_training_path():
    source=Path("scripts/run_phase5a_frozen3b.py").read_text();body=source[source.index("def training_losses"):source.index("def run_stability_preflight")]
    assert "gaussian_benefit_likelihood" in body
    assert "benefit_likelihood_with_detached_variance" not in body


def test_step0_raw_gradient_is_independent_of_learning_rate():
    torch.manual_seed(4);first=FrozenQwen25VLContextAdapter(FakeBackbone(),24);second=FrozenQwen25VLContextAdapter(FakeBackbone(),24);second.load_state_dict(first.state_dict());features=torch.randn(4,CONTEXT_DIM);target=torch.randn(4)
    def gradient(model,lr):
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-3);output=model(features);loss=gaussian_benefit_likelihood(output.benefit_mean,target,output.benefit_log_variance,torch);optimizer.zero_grad();loss.backward();return norm(tuple(p.grad for p in model.parameters() if p.requires_grad))
    assert gradient(first,3e-4)==pytest.approx(gradient(second,3e-5),rel=1e-7)


def test_parameter_delta_audit_math():
    before=(torch.tensor((3.,4.)),torch.tensor((0.,2.)));after=(torch.tensor((4.,4.)),torch.tensor((0.,5.)))
    delta=sum(float((right-left).square().sum()) for left,right in zip(before,after))**.5;parameter=sum(float(left.square().sum()) for left in before)**.5
    assert delta==pytest.approx(10**.5) and delta/(parameter+1e-12)==pytest.approx(10**.5/29**.5)


def test_prior_repairs_qwen_optimizer_and_no_test_contract():
    adapter=Path("src/models/large_context_adapter.py").read_text();dataset=Path("src/multimodal/context_dataset.py").read_text();script=Path("scripts/run_phase5a_lr_diagnostic.py")
    assert "self.scale_alignment_enabled=True" in adapter
    assert "benefit normalizer may only be fit on the train split" in dataset
    if script.exists():
        source=script.read_text();assert "materialize_test(" not in source
