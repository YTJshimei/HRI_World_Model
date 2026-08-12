import inspect
import numpy as np
import pytest
torch=pytest.importorskip("torch")

from src.decision.large_context_arbitrator import ContextDecisionMode,arbitrate_large_context
from src.models.large_context_adapter import CONTEXT_DIM,MockLargeContextBackbone,SmallContextNetwork
from src.multimodal.context_schema import FORBIDDEN_MODEL_INPUTS,StructuredContextTokens,TOKEN_DIMS,TOKEN_ORDER,validate_branch_split_isolation
from src.multimodal.context_dataset import ContextDataset,ContextTarget,validate_global_split_isolation

def sample(action=0,initial="i0",split="C1_seen_motion_seen_action"):
    return StructuredContextTokens({name:np.zeros(TOKEN_DIMS[name],np.float32) for name in TOKEN_ORDER},action,"context",initial,split)

def test_gt_fields_cannot_enter_model_context():
    assert {"gt_benefit","gt_best_action","gt_future","gt_theta","person_id"}<=FORBIDDEN_MODEL_INPUTS
    signature=set(inspect.signature(StructuredContextTokens).parameters)
    assert not signature&FORBIDDEN_MODEL_INPUTS

def test_unseen_person_has_no_identity_shortcut():
    assert "person_id" not in inspect.signature(StructuredContextTokens).parameters
    assert "profile_id" not in inspect.signature(StructuredContextTokens).parameters

def test_counterfactual_branch_split_isolation():
    with pytest.raises(ValueError):validate_branch_split_isolation([sample(0,"same","C1_seen_motion_seen_action"),sample(1,"same","C2_unseen_motion_action")])

def test_counterfactual_branch_dataset_isolation():
    target=ContextTarget(0.,False,np.zeros(6,np.float32))
    first=ContextDataset((sample(0,"same","C1_seen_motion_seen_action"),),(target,),"C1_seen_motion_seen_action")
    second=ContextDataset((sample(1,"same","C2_unseen_motion_action"),),(target,),"C2_unseen_motion_action")
    with pytest.raises(ValueError):validate_global_split_isolation([first,second])

def test_large_model_cannot_bypass_feasible_mask():
    result=arbitrate_large_context(np.arange(3),np.asarray((True,False,True)),np.asarray((1.,-50.,2.)),np.asarray((1.,-100.,2.)),np.asarray((0.,100.,0.)),np.zeros(3),.1,.5)
    assert result.selected_action!=1

def test_mock_pipeline_shapes_and_harm_range():
    model=MockLargeContextBackbone();prediction=model(torch.randn(4,CONTEXT_DIM))
    assert prediction.benefit_mean.shape==(4,)
    probability=prediction.harm_logit.sigmoid();assert torch.all((probability>=0)&(probability<=1))

def test_small_context_shapes():
    prediction=SmallContextNetwork()(torch.randn(3,CONTEXT_DIM));assert prediction.context_embedding.shape==(3,128)

def test_candidate_permutation_invariance():
    actions=np.arange(4);mask=np.asarray((True,True,False,True));generic=np.asarray((.4,.2,-9.,.3));personal=np.asarray((.4,.1,-20.,.3));benefit=np.asarray((0.,.2,10.,0.));harm=np.asarray((.1,.1,0.,.1))
    first=arbitrate_large_context(actions,mask,generic,personal,benefit,harm,.1,.5);p=np.asarray((3,2,0,1));second=arbitrate_large_context(actions[p],mask[p],generic[p],personal[p],benefit[p],harm[p],.1,.5)
    assert first.selected_action==second.selected_action==1

def test_benefit_sign_definition():
    generic_gt_cost,personal_gt_cost=1.2,.8
    benefit=generic_gt_cost-personal_gt_cost
    assert benefit>0

def test_generic_safe_fallback():
    result=arbitrate_large_context(np.arange(2),np.ones(2,bool),np.asarray((0.,1.)),np.asarray((1.,0.)),np.zeros(2),np.ones(2),.1,.5)
    assert result.mode==ContextDecisionMode.GENERIC_SAFE and result.selected_action==0
