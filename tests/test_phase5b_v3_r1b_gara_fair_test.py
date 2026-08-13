"""Contract tests for the Phase 5B-v3-R1B GARA-v2 fair test."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from src.models.generic_anchored_benefit import AbsoluteBenefitReadout, GenericAnchoredBenefitReadout
from src.training.candidate_ranking import LAMBDA_RANK

ROOT = Path(__file__).resolve().parents[1]
TARGET_SHA = "ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1"


def test_h0_h1_capacity_and_identical_initialization():
    torch = pytest.importorskip("torch")
    torch.manual_seed(42); h0 = AbsoluteBenefitReadout()
    torch.manual_seed(42); h1 = GenericAnchoredBenefitReadout()
    assert sum(p.numel() for p in h0.parameters()) == 129
    assert sum(p.numel() for p in h1.parameters()) == 129
    assert all(torch.equal(a, b) for a, b in zip(h0.parameters(), h1.parameters()))


def test_h1_shared_scorer_and_generic_exact_zero():
    torch = pytest.importorskip("torch")
    model = GenericAnchoredBenefitReadout(); value = torch.randn(9, 128)
    prediction = model(value, value)
    assert torch.equal(prediction, torch.zeros_like(prediction))
    assert model.architecture_audit()["shared_scorer_object_count"] == 1


def test_h1_bias_gradient_is_zero():
    torch = pytest.importorskip("torch")
    model = GenericAnchoredBenefitReadout(); a = torch.randn(5, 128); g = torch.randn(5, 128)
    model(a, g).square().mean().backward()
    assert torch.equal(model.scorer.bias.grad, torch.zeros_like(model.scorer.bias.grad))


def test_readouts_reject_invalid_shapes():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="128"):
        AbsoluteBenefitReadout()(torch.randn(2, 127))
    with pytest.raises(ValueError, match="match"):
        GenericAnchoredBenefitReadout()(torch.randn(2, 128), torch.randn(1, 128))


def test_target_v2_checksum_and_ranking_weight_are_frozen():
    path = ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TARGET_SHA
    assert LAMBDA_RANK == 0.25


def test_r1b_script_has_no_test_or_decision_chain_execution():
    from scripts import run_phase5b_v3_r1b_gara_fair_test as stage
    source = inspect.getsource(stage.main)
    assert 'build_development_split("test"' not in source
    assert "select_threshold" not in source
    assert "arbitrate" not in source
    assert "decision_evaluation" not in source
    assert stage.TEST_READS == 0
    assert "r0.V2_B1_REFERENCE" in source
