import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip("torch")
from torch import nn

from scripts.run_phase5a_frozen3b import clip_trainable_gradients,gradient_norm,sha256_array,trainable_parameters,trainable_state_checksum
from src.evaluation.context_value_metrics import candidate_metrics,switch_metrics,validation_selection_key
from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
from src.multimodal.context_schema import CONTEXT_DIM,TOKEN_DIMS,TOKEN_ORDER,StructuredContextTokens,prepare_context_batch


class FakeBackbone(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.embedding=nn.Embedding(8,hidden);self.block=nn.Linear(hidden,hidden);self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden))
    def get_input_embeddings(self):return self.embedding
    def forward(self,inputs_embeds,attention_mask,**kwargs):return SimpleNamespace(hidden_states=(inputs_embeds,self.block(inputs_embeds)))


def sample(offset=0.0):
    values=np.arange(CONTEXT_DIM,dtype=np.float32)+offset;cursor=0;groups={}
    for name in TOKEN_ORDER:
        width=TOKEN_DIMS[name];groups[name]=values[cursor:cursor+width];cursor+=width
    return StructuredContextTokens(groups,0,str(offset),str(offset),"validation")


def test_l1_l2_input_equality_checksum():
    canonical=prepare_context_batch((sample(0),sample(1000)))
    assert sha256_array(canonical)==sha256_array(canonical.copy())
    np.testing.assert_array_equal(canonical[0],sample(0).flattened())


def test_qwen_optimizer_exclusion_and_full_step_frozen_gradient_zero():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24)
    parameters=[parameter for group in model.trainable_parameter_groups().values() for parameter in group]
    optimizer=torch.optim.AdamW(parameters,lr=1e-3)
    backbone_ids={id(parameter) for parameter in model.backbone.parameters()}
    assert not backbone_ids&{id(parameter) for parameter in parameters}
    prediction=model(torch.randn(4,CONTEXT_DIM))
    loss=prediction.benefit_mean.square().mean()+prediction.harm_logit.square().mean()+prediction.benefit_log_variance.square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.backbone.parameters())


def populate_all_trainable_gradients(model,value):
    for parameter in trainable_parameters(model):parameter.grad=torch.full_like(parameter,value)


def test_gradient_clipping_only_uses_trainable_params_and_records_pre_post():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);populate_all_trainable_gradients(model,10.0)
    assert not {id(p) for p in model.backbone.parameters()}&{id(p) for p in trainable_parameters(model)}
    audit=clip_trainable_gradients(model,torch,10.0)
    assert audit["pre_clip_grad_norm"]>10.0
    assert audit["post_clip_grad_norm"]<=10.0+1e-5
    assert gradient_norm(trainable_parameters(model))==pytest.approx(audit["post_clip_grad_norm"])
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


@pytest.mark.parametrize("bad",[float("nan"),float("inf")])
def test_nonfinite_gradient_stops_immediately(bad):
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);populate_all_trainable_gradients(model,1.0)
    trainable_parameters(model)[0].grad.flatten()[0]=bad
    with pytest.raises(FloatingPointError):clip_trainable_gradients(model,torch,10.0)


def test_formal_reinitialization_differs_from_preflight_final_state():
    torch.manual_seed(42);preflight=FrozenQwen25VLContextAdapter(FakeBackbone(),24);initial=trainable_state_checksum(preflight)
    with torch.no_grad():next(iter(preflight.projection.parameters())).add_(1.0)
    final=trainable_state_checksum(preflight)
    torch.manual_seed(42);formal=FrozenQwen25VLContextAdapter(FakeBackbone(),24)
    assert trainable_state_checksum(formal)==initial and trainable_state_checksum(formal)!=final


def test_stability_stage_contract_never_materializes_test_or_checkpoint():
    source=Path("scripts/run_phase5a_frozen3b.py").read_text()
    preflight_body=source[source.index("def run_stability_preflight"):source.index("def train_frozen")]
    assert "materialize_test" not in preflight_body and "torch.save" not in preflight_body


def test_trainable_checkpoint_excludes_backbone_and_auxiliary():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24)
    state=model.trainable_state_dict()
    assert state and all(name.startswith(("projection.","benefit.","harm.","uncertainty.")) for name in state)
    assert not any(name.startswith(("backbone.","auxiliary.")) for name in state)
    backbone_before={name:value.detach().clone() for name,value in model.backbone.state_dict().items()}
    model.load_trainable_state_dict(state)
    assert all(torch.equal(value,model.backbone.state_dict()[name]) for name,value in backbone_before.items())


def test_checkpoint_selection_is_validation_only_and_test_cannot_change_it():
    favorable={"Harmful_Switch_Rate":0.0,"Beneficial_Switch_Recall":.5,"Mean_Regret":.1,"Benefit_MAE":.2,"Benefit_Spearman":.3,"Harm_AUROC":.8}
    worse={**favorable,"Beneficial_Switch_Recall":.2}
    selected=min(((validation_selection_key(favorable),3),(validation_selection_key(worse),4)))[1]
    arbitrary_test={**favorable,"Beneficial_Switch_Recall":0.0,"Mean_Regret":99.0}
    assert selected==3
    assert min(((validation_selection_key(favorable),3),(validation_selection_key(worse),4)))[1]==selected
    assert arbitrary_test["Mean_Regret"]==99.0


def test_candidate_metric_benefit_sign_and_switch_counts():
    prediction={"benefit":np.asarray((.4,-.2,0.)),"sigma":np.ones(3),"harm":np.asarray((.1,.9,.1))}
    target={"benefit":np.asarray((.5,-.1,0.)),"harm":np.asarray((False,True,False))}
    metrics=candidate_metrics(prediction,target)
    assert metrics["Benefit_Sign_Accuracy"]==1.0
    decisions=[{"beneficial_switch":True,"harmful_switch":False,"personalized":True,"decision_mode":"PERSONALIZED"},{"beneficial_switch":False,"harmful_switch":True,"personalized":True,"decision_mode":"PERSONALIZED"},{"beneficial_switch":False,"harmful_switch":False,"personalized":False,"decision_mode":"GENERIC_SAFE"}]
    switches=switch_metrics(decisions,2)
    assert switches["Beneficial_Switch_Recall"]==.5 and switches["Harmful_Switch_Count"]==1


def test_context_split_isolation_and_hard_manifest_consistency():
    audit=json.loads(Path("results_dev/phase5a/dataset_audit.json").read_text())
    manifest=json.loads(Path("results_dev/phase5a/hard_case_manifest.json").read_text())
    assert audit["counterfactual_split_isolation"] and sum(audit["context_splits"].values())==audit["test_candidates"]
    assert {"phase4c3_beneficial_cases","phase4c3_harmful_cases","phase4c2_max_regret_cases"}<=set(manifest)


def test_candidate_permutation_keeps_candidate_id_and_result_mapping():
    batch=prepare_context_batch((sample(0),sample(100),sample(200)))
    ids=np.asarray((0,1,2));permutation=np.asarray((2,0,1))
    assert prepare_context_batch(batch[permutation]).shape==(3,CONTEXT_DIM)
    np.testing.assert_array_equal(ids[permutation],np.asarray((2,0,1)))


def test_json_outputs_reject_nan(tmp_path):
    with pytest.raises(ValueError):
        (tmp_path/"bad.json").write_text(json.dumps({"bad":float("nan")},allow_nan=False))
