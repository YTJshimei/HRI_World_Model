from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
from torch import nn

from scripts.audit_phase5a_gradient_sources import attribution_parameters,grad_cosine,grad_norm,per_loss_gradients
from src.models.large_context_adapter import CONTEXT_DIM,FrozenQwen25VLContextAdapter


class FakeBackbone(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.embedding=nn.Embedding(8,hidden);self.block=nn.Linear(hidden,hidden);self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden))
    def get_input_embeddings(self):return self.embedding
    def forward(self,inputs_embeds,attention_mask,**kwargs):return SimpleNamespace(hidden_states=(inputs_embeds,self.block(inputs_embeds)))


def fixture():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);prediction=model(torch.randn(4,CONTEXT_DIM));target=torch.randn(4);harm=torch.tensor((0.,1.,0.,1.));error=prediction.benefit_mean-target
    losses={"benefit":.5*(error.square()*torch.exp(-prediction.benefit_log_variance)).mean(),"harm":torch.nn.functional.binary_cross_entropy_with_logits(prediction.harm_logit,harm),"uncertainty":.5*prediction.benefit_log_variance.mean()}
    return model,losses


def checksum(model):return [parameter.detach().clone() for parameter in model.parameters()]


def test_attribution_does_not_step_or_change_parameters():
    model,losses=fixture();before=checksum(model);per_loss_gradients(losses,model,torch);after=checksum(model)
    assert all(torch.equal(left,right) for left,right in zip(before,after))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_per_loss_gradients_do_not_accumulate_or_pollute():
    model,losses=fixture();first=per_loss_gradients(losses,model,torch);second=per_loss_gradients(losses,model,torch)
    for loss_name in ("benefit","harm","uncertainty","total"):
        for left,right in zip(first[loss_name]["all"],second[loss_name]["all"]):
            if left is None:assert right is None
            else:assert torch.equal(left,right)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_cosine_is_finite():
    model,losses=fixture();gradients=per_loss_gradients(losses,model,torch)
    for left,right in (("benefit","harm"),("benefit","uncertainty"),("harm","uncertainty")):
        value=grad_cosine(gradients[left]["all"],gradients[right]["all"]);assert -1<=value<=1


def test_module_gradient_aggregation_matches_total_parameters():
    model,losses=fixture();gradients=per_loss_gradients(losses,model,torch)
    squared=sum(grad_norm(gradients["total"]["modules"][name])**2 for name in ("projection","benefit_head","harm_head","uncertainty_head"))
    assert squared**.5==pytest.approx(grad_norm(gradients["total"]["all"]),rel=1e-6)


def test_qwen_is_not_attribution_target_and_has_no_gradient():
    model,losses=fixture();_,parameters=attribution_parameters(model);backbone={id(parameter) for parameter in model.backbone.parameters()}
    assert not backbone&{id(parameter) for parameter in parameters}
    per_loss_gradients(losses,model,torch)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.backbone.parameters())


def test_diagnostic_source_forbids_optimizer_and_test_materialization():
    from pathlib import Path
    source=Path("scripts/audit_phase5a_gradient_sources.py").read_text()
    assert "optimizer.step(" not in source and "materialize_test(" not in source
