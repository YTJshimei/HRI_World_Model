from pathlib import Path

import numpy as np
import pytest

from src.multimodal.context_dataset import ContextTarget,fit_benefit_normalizer
from src.multimodal.context_schema import CONTEXT_DIM,TOKEN_DIMS,TOKEN_ORDER,StructuredContextTokens,prepare_context_batch


def sample(context_id,value=0.0):
    groups={name:np.full(width,value,np.float32) for name,width in TOKEN_DIMS.items()}
    return StructuredContextTokens(groups,0,context_id,context_id.rsplit(":",1)[0],"train")


def fixture():
    samples=[sample(f"train:S1:{index}:0",index) for index in range(4)]
    targets=[ContextTarget(value,False,np.zeros(6,np.float32)) for value in (-2.0,-1.0,1.0,1000.0)]
    meta=[{"feasible":value} for value in (True,True,True,False)]
    return samples,targets,meta


def test_l1_l2_fit_ids_and_statistics_are_identical():
    samples,targets,meta=fixture();l1=fit_benefit_normalizer(samples,targets,meta);l2=fit_benefit_normalizer(samples,targets,meta)
    assert l1.fit_sample_ids==l2.fit_sample_ids
    assert l1.mean==l2.mean and l1.scale==l2.scale and l1.epsilon==l2.epsilon
    np.testing.assert_array_equal(l1.transform(np.asarray((-2.,0.,1.))),l2.transform(np.asarray((-2.,0.,1.))))


def test_infeasible_candidate_is_excluded_without_being_deleted():
    samples,targets,meta=fixture();normalizer=fit_benefit_normalizer(samples,targets,meta)
    assert len(samples)==4 and len(normalizer.fit_sample_ids)==3
    assert "train:S1:3:0" not in normalizer.fit_sample_ids
    assert normalizer.mean==pytest.approx((-2-1+1)/3)


@pytest.mark.parametrize("split",("validation","test"))
def test_validation_and_test_cannot_enter_normalizer_fit(split):
    samples,targets,meta=fixture();samples[0]=sample(f"{split}:S1:0:0")
    with pytest.raises(ValueError,match="train split"):fit_benefit_normalizer(samples,targets,meta)


def test_normalizer_repair_does_not_change_context_contract():
    samples,targets,meta=fixture();before=prepare_context_batch(samples);fit_benefit_normalizer(samples,targets,meta);after=prepare_context_batch(samples)
    assert before.shape==after.shape==(4,CONTEXT_DIM)
    np.testing.assert_array_equal(before,after)
    assert TOKEN_ORDER==("skeleton","motion","robot","functional","candidate","uncertainty","diagnostic","interaction","scene")


def test_scale_alignment_and_frozen_qwen_contract_remain_enabled():
    source=Path("src/models/large_context_adapter.py").read_text()
    assert "self.scale_alignment_enabled=True" in source
    assert "self.freeze_backbone()" in source


def test_s4_audit_does_not_step_or_materialize_test():
    source=Path("scripts/audit_phase5a_gradient_sources.py").read_text()
    assert "optimizer.step(" not in source and "materialize_test(" not in source
