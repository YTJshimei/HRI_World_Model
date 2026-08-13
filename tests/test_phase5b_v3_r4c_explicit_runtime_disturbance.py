"""Contracts for Phase 5B-v3-R4C ERADA."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.models.runtime_disturbance_advantage import (
    DisturbanceAdvantageBenefitReadout,
    robot_action_disturbance,
    runtime_disturbance_advantage,
)


ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json":"ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv":"ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv":"88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt":"dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt":"2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
}


def test_r4c_frozen_contract_checksums():
    for relative,expected in EXPECTED.items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected


def test_robot_only_disturbance_matches_formal_action_regularizers():
    assert robot_action_disturbance(0)==0.0
    assert robot_action_disturbance(1)==pytest.approx(.55*.30)
    assert robot_action_disturbance(2)==pytest.approx(.55*.30)
    assert robot_action_disturbance(3)==pytest.approx(.55*.25)
    assert robot_action_disturbance(4)==pytest.approx(.55*.25)
    assert robot_action_disturbance(5)==pytest.approx(.55*.20)
    assert robot_action_disturbance(6)==pytest.approx(.55*.20)
    assert robot_action_disturbance(7)==pytest.approx(.55*3.0)


def test_runtime_advantage_is_generic_minus_candidate_and_zero_on_identity():
    candidate=np.asarray([0,1,3,7]);generic=np.asarray([3,3,3,7])
    value=runtime_disturbance_advantage(candidate,generic)
    assert value.shape==(4,1) and value.dtype==np.float32
    expected=np.asarray([[robot_action_disturbance(g)-robot_action_disturbance(i)] for i,g in zip(candidate,generic)],np.float32)
    np.testing.assert_array_equal(value,expected)
    assert value[-1,0]==0.0


def test_runtime_feature_function_has_no_gt_simulator_or_profile_inputs():
    signature=inspect.signature(runtime_disturbance_advantage)
    assert tuple(signature.parameters)==("candidate_action_ids","generic_action_ids")
    names=set(runtime_disturbance_advantage.__code__.co_names)|set(robot_action_disturbance.__code__.co_names)
    for forbidden in ("future","human_effect","gt_cost","benefit","harm","profile","simulate"):
        assert all(forbidden not in name.lower() for name in names)


def test_matched_readout_is_exactly_linear_257_for_all_three_arms():
    torch=pytest.importorskip("torch");model=DisturbanceAdvantageBenefitReadout();audit=model.architecture_audit()
    assert audit["input_dim"]==257 and audit["layers"]==["Linear(257,1)"] and audit["parameter_count"]==258
    z_i=torch.randn(4,128);z_g=torch.randn(4,128);feature=torch.randn(4,1)
    assert model(z_i,z_g,feature).shape==(4,)
    with pytest.raises(ValueError):model(z_i,z_g,torch.randn(4,2))


def test_c0_is_zero_c1_is_runtime_and_o1_is_gt_only():
    from scripts import run_phase5b_v3_r4c_explicit_runtime_disturbance as stage
    source=inspect.getsource(stage.build_features)
    assert '"C0":np.zeros_like(runtime)' in source
    assert '"C1":runtime' in source and '"O1":full' in source
    assert stage.ORACLE=="GT FULL DISTURBANCE ORACLE - NOT RUNTIME VALID"
    main=inspect.getsource(stage.main)
    assert '"deployment_forbidden":True' in main and '"direct_addition_to_prediction":False' in main


def test_training_protocol_and_selector_are_matched_and_frozen():
    from scripts import run_phase5b_v3_r4c_explicit_runtime_disturbance as stage
    source=inspect.getsource(stage.main)
    assert 'for name in ("C0","C1","O1")' in source
    assert "torch.manual_seed(args.seed);model=DisturbanceAdvantageBenefitReadout()" in source
    assert "r4a.train" in source and "make_episode_batches" in source
    assert stage.LAMBDA_RANK==0.0 and stage.TEST_READS==0


def test_r4c_has_no_test_threshold_decision_or_new_predictor():
    from scripts import run_phase5b_v3_r4c_explicit_runtime_disturbance as stage
    source=inspect.getsource(stage);main=inspect.getsource(stage.main)
    assert 'build_development_split("test"' not in main
    for forbidden in ("select_threshold","decision_evaluation","arbitrate(","GNN","TransformerSkeleton","Qwen"):
        assert forbidden not in source
    assert '"threshold_calibration":False' in source and '"decision_chain":False' in source


def test_ranking_and_harm_paths_are_frozen_b0_and_harm_v3_only():
    from scripts import run_phase5b_v3_r4c_explicit_runtime_disturbance as stage
    source=inspect.getsource(stage.main)
    assert "r2.ranking_invariance" in source
    assert '"Frozen_B0_ranking_only":True' in source
    assert '"HARM_v3_risk_only":True' in source
    assert '"harm_optimizer_created":False' in source
