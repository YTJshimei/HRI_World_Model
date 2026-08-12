import numpy as np
import pytest
torch=pytest.importorskip("torch")
from torch import nn
from types import SimpleNamespace
from src.models.large_context_adapter import CONTEXT_DIM,FrozenQwen25VLContextAdapter,StructuredTokenProjection
from src.multimodal.context_dataset import ContextDataset,ContextTarget
from src.multimodal.context_schema import TOKEN_DIMS,TOKEN_ORDER,StructuredContextTokens,prepare_context_batch

class FakeBackbone(nn.Module):
    def __init__(self,hidden=24):
        super().__init__();self.embedding=nn.Embedding(8,hidden);self.block=nn.Linear(hidden,hidden);self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden))
    def get_input_embeddings(self):return self.embedding
    def forward(self,inputs_embeds,attention_mask,**kwargs):return SimpleNamespace(hidden_states=(inputs_embeds,self.block(inputs_embeds)))

def test_group_projection_shape():
    assert StructuredTokenProjection(32)(torch.randn(3,CONTEXT_DIM)).shape==(3,9,32)

def test_frozen_adapter_output_and_backbone_frozen():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);out=model(torch.randn(2,CONTEXT_DIM))
    assert out.context_embedding.shape==(2,24) and out.benefit_mean.shape==(2,)
    assert model.backbone_fully_frozen

def test_backward_updates_only_projection_and_heads():
    model=FrozenQwen25VLContextAdapter(FakeBackbone(),24);out=model(torch.randn(2,CONTEXT_DIM));loss=out.benefit_mean.mean()+out.harm_logit.mean()+out.benefit_log_variance.mean();loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.projection.parameters())
    assert any(parameter.grad is not None for parameter in model.benefit.parameters())
    assert any(parameter.grad is not None for parameter in model.harm.parameters())
    assert any(parameter.grad is not None for parameter in model.uncertainty.parameters())

def test_model_id_is_locked():
    with pytest.raises(ValueError):FrozenQwen25VLContextAdapter.from_pretrained_4bit("another/model")


def canonical_sample(offset=0.0, action=0):
    cursor=offset
    groups={}
    for name in TOKEN_ORDER:
        width=TOKEN_DIMS[name]
        groups[name]=np.arange(cursor,cursor+width,dtype=np.float32)
        cursor+=width
    return StructuredContextTokens(groups,action,f"context-{offset}",f"state-{offset}","C1_seen_motion_seen_action")


def test_prepare_context_batch_accepts_batch_and_single_sample():
    batch=np.arange(2*CONTEXT_DIM,dtype=np.float32).reshape(2,CONTEXT_DIM)
    prepared=prepare_context_batch(batch)
    assert prepared.shape==(2,CONTEXT_DIM)
    np.testing.assert_array_equal(prepared,batch)
    single=prepare_context_batch(batch[0])
    assert single.shape==(1,CONTEXT_DIM)
    np.testing.assert_array_equal(single[0],batch[0])


@pytest.mark.parametrize("shape",[(107,),(1,107),(2,9,12)])
def test_prepare_context_batch_rejects_unknown_feature_layout(shape):
    with pytest.raises(ValueError):prepare_context_batch(np.zeros(shape,np.float32))


def test_grouped_context_uses_exact_canonical_108_order():
    sample=canonical_sample()
    expected=np.concatenate([sample.tokens[name] for name in TOKEN_ORDER])
    grouped=prepare_context_batch(sample.tokens)
    assert grouped.shape==(1,CONTEXT_DIM)
    np.testing.assert_array_equal(grouped[0],expected)


def test_stage_ab_dataset_and_stage_c_prepare_share_feature_ordering():
    samples=(canonical_sample(0),canonical_sample(1000,1))
    targets=(ContextTarget(0.0,False,np.zeros(6,np.float32)),)*2
    dataset=ContextDataset(samples,targets,"C1_seen_motion_seen_action")
    stage_ab=dataset.features()
    stage_c=prepare_context_batch(samples)
    np.testing.assert_array_equal(stage_ab,stage_c)
    np.testing.assert_array_equal(stage_c[0],samples[0].flattened())


def test_candidate_permutation_reorders_batch_not_feature_axis():
    batch=prepare_context_batch((canonical_sample(0,0),canonical_sample(1000,1),canonical_sample(2000,2)))
    permutation=np.asarray([2,0,1],dtype=np.int64)
    permuted=prepare_context_batch(batch[permutation])
    assert permuted.shape==(3,CONTEXT_DIM)
    np.testing.assert_array_equal(permuted,batch[permutation])


def test_structured_projection_keeps_strict_batched_contract():
    projection=StructuredTokenProjection(16)
    with pytest.raises(ValueError,match=r"features must have shape \[B,108\]"):
        projection(torch.zeros(CONTEXT_DIM))
