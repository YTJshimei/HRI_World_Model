"""Contracts for Phase 5B-v3-R2 decoupled pair-conditioned Benefit readout."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from src.models.pair_conditioned_benefit import (
    AbsoluteCandidateBenefitReadout,
    PairConditionedBenefitReadout,
    prepare_pair_input,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json": "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv": "ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv": "88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt": "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt": "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
}


def test_frozen_contract_checksums_remain_exact():
    for relative, expected in EXPECTED.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_a0_and_a1_are_exact_minimal_linear_heads():
    torch = pytest.importorskip("torch")
    a0 = AbsoluteCandidateBenefitReadout(); a1 = PairConditionedBenefitReadout()
    assert sum(parameter.numel() for parameter in a0.parameters()) == 129
    assert sum(parameter.numel() for parameter in a1.parameters()) == 257
    assert a0.architecture_audit()["layers"] == ["Linear(128,1)"]
    assert a1.architecture_audit()["layers"] == ["Linear(256,1)"]
    assert a0(torch.randn(3, 128)).shape == (3,)
    assert a1(torch.randn(3, 128), torch.randn(3, 128)).shape == (3,)


def test_pair_input_is_ordered_candidate_then_runtime_generic():
    torch = pytest.importorskip("torch")
    candidate = torch.full((2, 128), 1.0); generic = torch.full((2, 128), 2.0)
    pair = prepare_pair_input(candidate, generic)
    assert pair.shape == (2, 256)
    assert torch.equal(pair[:, :128], candidate)
    assert torch.equal(pair[:, 128:], generic)


def test_heads_reject_wrong_shapes_and_a0_has_no_generic_argument():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="128"):
        AbsoluteCandidateBenefitReadout()(torch.randn(2, 127))
    with pytest.raises(ValueError, match="match"):
        PairConditionedBenefitReadout()(torch.randn(2, 128), torch.randn(1, 128))
    with pytest.raises(TypeError):
        AbsoluteCandidateBenefitReadout()(torch.randn(2, 128), torch.randn(2, 128))


def test_selector_contract_is_mae_then_sign_then_earlier_epoch():
    from scripts import run_phase5b_v3_r2_pair_conditioned_benefit as stage
    rows = [
        {"epoch": 1, "Benefit_MAE": 1.0, "safe_beneficial_sign_accuracy": .5},
        {"epoch": 2, "Benefit_MAE": .9, "safe_beneficial_sign_accuracy": .4},
        {"epoch": 3, "Benefit_MAE": .9, "safe_beneficial_sign_accuracy": .6},
        {"epoch": 4, "Benefit_MAE": .9, "safe_beneficial_sign_accuracy": .6},
    ]
    assert stage.select_epoch(rows)["epoch"] == 3


def test_r2_has_no_ranking_loss_test_threshold_or_decision_chain():
    from scripts import run_phase5b_v3_r2_pair_conditioned_benefit as stage
    source = inspect.getsource(stage)
    main_source = inspect.getsource(stage.main)
    assert stage.LAMBDA_RANK == 0.0
    assert "pairwise_logistic_ranking_loss" not in source
    assert 'build_development_split("test"' not in main_source
    assert "select_threshold" not in source
    assert "arbitrate" not in source
    assert "decision_evaluation" not in source
    assert stage.TEST_READS == 0


def test_formal_ranking_contract_is_frozen_b0_only():
    from scripts import run_phase5b_v3_r2_pair_conditioned_benefit as stage
    source = inspect.getsource(stage.ranking_invariance)
    assert '"A0_used_for_ranking": False' in source
    assert '"A1_used_for_ranking": False' in source
    assert stage.EXPECTED_RANKING == {
        "mean_feasible_pairwise_accuracy": 0.8130555555555555,
        "gt_best_top1_accuracy": 0.8416666666666667,
        "gt_best_top2_recall": 0.9541666666666667,
        "mean_gt_best_rank": 1.2291666666666667,
    }


def test_pair_contract_excludes_ground_truth_and_profile_runtime_inputs():
    from scripts import run_phase5b_v3_r2_pair_conditioned_benefit as stage
    source = inspect.getsource(stage.main)
    assert '"profile_ID_input": False' in source
    assert '"GT_input": False' in source
    assert '"GT future"' in source and '"GT benefit"' in source and '"GT harm"' in source
    assert "profile_ID_runtime_input" in inspect.getsource(stage.shortcut_rows)
