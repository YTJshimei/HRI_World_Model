"""Contracts for Phase 5B-v3-R4A Oracle future-pose sufficiency audit."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from src.models.oracle_future_pose import (
    OraclePoseBenefitReadout, oracle_pose_delta, root_relative_decision_local_pose,
)

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json":"ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv":"ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv":"88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt":"dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt":"2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
}


def test_r4a_frozen_contract_checksums():
    for relative,expected in EXPECTED.items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected


def test_candidate_conditioned_future_skeleton_is_real_generator_output():
    from src.data.synthetic_interaction import RiskConditionedInteractionSimulation,simulate_risk_conditioned_interaction_future
    from src.data.hold_candidate import HoldInteractionSimulation
    assert "future_global" in RiskConditionedInteractionSimulation.__dataclass_fields__
    assert "future_global" in HoldInteractionSimulation.__dataclass_fields__
    source=inspect.getsource(simulate_risk_conditioned_interaction_future)
    assert "future_global" in source
    from scripts import run_phase5b_v3_r4a_oracle_future_pose_sufficiency as stage
    future_source=inspect.getsource(stage.future_skeletons)
    assert "simulate_risk_conditioned_interaction_future" in future_source
    assert "build_hold_candidate_outcome" in future_source
    assert "interpol" not in future_source.lower() and "repeat" not in future_source.lower()


def test_root_relative_local_pose_uses_pelvis_midpoint_and_robot_yaw():
    torch=pytest.importorskip("torch")
    future=torch.zeros(1,10,17,3);future[...,11,:]=torch.tensor([1.0,1.0,0.0]);future[...,12,:]=torch.tensor([3.0,1.0,0.0]);future[...,5,:]=torch.tensor([2.0,2.0,1.0])
    root,local=root_relative_decision_local_pose(future,torch.tensor([torch.pi/2]))
    assert torch.allclose(root,torch.tensor([2.0,1.0,0.0]).view(1,1,3).expand(1,10,3))
    assert torch.allclose(local[...,5,:],torch.tensor([1.0,0.0,1.0]).view(1,1,3).expand(1,10,3),atol=1e-6)


def test_pose_delta_is_candidate_minus_generic_510d():
    torch=pytest.importorskip("torch");candidate=torch.randn(3,10,17,3);generic=torch.randn_like(candidate)
    value=oracle_pose_delta(candidate,generic)
    assert value.shape==(3,510)
    assert torch.equal(value,(candidate-generic).flatten(1))


def test_c0_and_pose_oracle_are_identical_766d_linear_heads():
    torch=pytest.importorskip("torch");torch.manual_seed(42);c0=OraclePoseBenefitReadout();torch.manual_seed(42);oracle=OraclePoseBenefitReadout()
    assert sum(parameter.numel() for parameter in c0.parameters())==767
    assert c0.architecture_audit()["layers"]==["Linear(766,1)"]
    assert all(torch.equal(left,right) for left,right in zip(c0.parameters(),oracle.parameters()))
    zeros=torch.zeros(4,510);delta=torch.randn(4,510);z_i=torch.randn(4,128);z_g=torch.randn(4,128)
    assert torch.count_nonzero(zeros)==0 and c0(z_i,z_g,zeros).shape==oracle(z_i,z_g,delta).shape==(4,)


def test_pose_oracle_is_explicitly_nonruntime_and_deployment_forbidden():
    from scripts import run_phase5b_v3_r4a_oracle_future_pose_sufficiency as stage
    assert stage.ORACLE_LABEL=="ORACLE FUTURE POSE DIAGNOSTIC - NOT RUNTIME VALID"
    source=inspect.getsource(stage.main)
    assert '"runtime_valid":False' in source
    assert '"deployment_forbidden":True' in source
    assert '"runtime_model_training":False' in source


def test_r4a_uses_frozen_generic_mapping_and_root_reproduction_constants():
    from scripts import run_phase5b_v3_r4a_oracle_future_pose_sufficiency as stage
    source=inspect.getsource(stage.main)
    assert "r1b.generic_indices" in source
    assert stage.R3_ROOT_SIGN==0.46956521739130436
    assert stage.R3_ROOT_MAE==1.7345836692970045
    assert stage.R3_ROOT_STOP==0


def test_r4a_has_no_predictor_test_threshold_decision_or_ranking_loss():
    from scripts import run_phase5b_v3_r4a_oracle_future_pose_sufficiency as stage
    source=inspect.getsource(stage);main=inspect.getsource(stage.main)
    assert stage.TEST_READS==0 and stage.LAMBDA_RANK==0.0
    assert 'build_development_split("test"' not in main
    assert "pairwise_logistic_ranking_loss" not in source
    assert "select_threshold" not in source and "arbitrate" not in source and "decision_evaluation" not in source
    assert "FutureSkeletonPredictor" not in source


def test_ranking_and_harm_remain_frozen_paths():
    from scripts import run_phase5b_v3_r4a_oracle_future_pose_sufficiency as stage
    source=inspect.getsource(stage.main)
    assert '"B0_formal_ranking_only":True' in source
    assert '"HARM_v3_formal_risk_only":True' in source
    assert '"harm_optimizer_created":False' in source
