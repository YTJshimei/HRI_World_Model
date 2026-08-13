"""Contracts for Phase 5B-v3-R3 counterfactual human-response representation."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.models.counterfactual_human_response import (
    HumanResponseFutureDecoder, MatchedBenefitReadout, counterfactual_delta,
    decision_local_coordinates,
)

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json":"ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv":"ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv":"88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt":"dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt":"2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
}


def test_r3_frozen_contract_checksums():
    for relative,expected in EXPECTED.items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected


def test_decision_local_coordinates_translation_and_rotation():
    torch=pytest.importorskip("torch")
    future=torch.tensor([[[2.0,3.0],[1.0,4.0]]]);origin=torch.tensor([[1.0,2.0]])
    local=decision_local_coordinates(future,origin,torch.tensor([np.pi/2],dtype=torch.float32))
    assert torch.allclose(local,torch.tensor([[[1.0,-1.0],[2.0,0.0]]]),atol=1e-6)


def test_response_decoder_actual_prefusion_dimensions_and_shape():
    torch=pytest.importorskip("torch");model=HumanResponseFutureDecoder()
    output=model(torch.randn(4,1024),torch.randn(4,256))
    assert output.shape==(4,10,2)
    assert model.architecture_audit()["input_dim"]==1280
    assert model.architecture_audit()["layers"]==["Linear(1280,128)","GELU","Linear(128,20)"]


def test_delta_is_candidate_minus_generic_and_flattened_20d():
    torch=pytest.importorskip("torch");candidate=torch.arange(40,dtype=torch.float32).view(2,10,2);generic=torch.ones_like(candidate)
    delta=counterfactual_delta(candidate,generic)
    assert delta.shape==(2,20)
    assert torch.equal(delta,(candidate-generic).reshape(2,20))


def test_c0_c1_o1_share_exact_276d_277_parameter_architecture():
    torch=pytest.importorskip("torch");models=[MatchedBenefitReadout() for _ in range(3)]
    assert all(sum(parameter.numel() for parameter in model.parameters())==277 for model in models)
    z_i=torch.randn(5,128);z_g=torch.randn(5,128);zero=torch.zeros(5,20);predicted=torch.randn(5,20);oracle=torch.randn(5,20)
    assert all(value.shape==(5,) for value in (models[0](z_i,z_g,zero),models[1](z_i,z_g,predicted),models[2](z_i,z_g,oracle)))
    assert torch.count_nonzero(zero)==0


def test_response_selector_uses_only_ade_fde_then_epoch():
    from scripts import run_phase5b_v3_r3_counterfactual_human_response as stage
    rows=[{"epoch":1,"Root_ADE":.2,"Root_FDE":.1,"Benefit_MAE":0.0},{"epoch":2,"Root_ADE":.1,"Root_FDE":.2,"Benefit_MAE":99.0},{"epoch":3,"Root_ADE":.1,"Root_FDE":.1,"Benefit_MAE":100.0},{"epoch":4,"Root_ADE":.1,"Root_FDE":.1,"Benefit_MAE":-100.0}]
    assert stage.select_response_epoch(rows)["epoch"]==3
    source=inspect.getsource(stage.select_response_epoch)
    assert "Benefit_MAE" not in source


def test_gt_future_is_not_a_response_decoder_runtime_argument():
    signature=inspect.signature(HumanResponseFutureDecoder.forward)
    assert tuple(signature.parameters)==("self","z_human","z_candidate")
    from scripts import run_phase5b_v3_r3_counterfactual_human_response as stage
    contract_source=inspect.getsource(stage.main)
    assert '"GT_input":False' in contract_source
    assert '"profile_ID_input":False' in contract_source
    assert stage.ORACLE_LABEL=="ORACLE FUTURE DIAGNOSTIC - NOT RUNTIME VALID"


def test_r3_has_no_test_threshold_ranking_loss_or_decision_chain():
    from scripts import run_phase5b_v3_r3_counterfactual_human_response as stage
    source=inspect.getsource(stage);main_source=inspect.getsource(stage.main)
    assert stage.TEST_READS==0 and stage.LAMBDA_RANK==0.0
    assert 'build_development_split("test"' not in main_source
    assert "pairwise_logistic_ranking_loss" not in source
    assert "select_threshold" not in source
    assert "arbitrate" not in source
    assert "decision_evaluation" not in source


def test_ranking_and_harm_are_only_frozen_paths():
    from scripts import run_phase5b_v3_r3_counterfactual_human_response as stage
    source=inspect.getsource(stage.main)
    assert '"B0_formal_ranking_only":True' in source
    assert '"HARM_v3_formal_risk_only":True' in source
    assert '"harm_optimizer_created":False' in source
