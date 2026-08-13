"""Isolation and protocol tests for Phase 5B-1.7E-D."""
from __future__ import annotations

import inspect

import pytest

from scripts import run_phase5b17ed_risk_preserving_bypass as stage
from src.models.independent_harm_head import RiskPreservingBypassHead


def test_h1_is_exactly_one_linear_1408_to_one():
    import torch
    head = RiskPreservingBypassHead(); audit = head.architecture_audit()
    assert audit["layers"] == ["Linear(1408,1)"]
    assert audit["parameter_count"] == 1409
    assert not any(audit[name] for name in ("projection", "MLP", "attention", "normalization", "learned_gate"))
    assert head(torch.zeros(3, 1408)).shape == (3,)


def test_other_bypass_dimensions_are_rejected():
    with pytest.raises(ValueError): RiskPreservingBypassHead(1280)


def test_bypass_is_detached_runtime_concat_in_fixed_order():
    import torch
    stages = {"R0_FINAL_FUSED": torch.randn(2, 128, requires_grad=True),
              "R1_HISTORY_CONTEXT_PREFUSION": torch.randn(2, 1024, requires_grad=True),
              "R2_CANDIDATE_PREFUSION": torch.randn(2, 256, requires_grad=True)}
    value = stage.bypass_input(stages, torch)
    assert value.shape == (2, 1408) and not value.requires_grad
    assert torch.equal(value[:, :128], stages["R0_FINAL_FUSED"])
    assert torch.equal(value[:, 128:1152], stages["R1_HISTORY_CONTEXT_PREFUSION"])
    assert torch.equal(value[:, 1152:], stages["R2_CANDIDATE_PREFUSION"])


def test_input_contract_excludes_gt_profile_and_oracle():
    source = inspect.getsource(stage.main)
    assert '"runtime_valid": True' in source
    assert '"profile_id_in_input": False' in source
    assert '"GT_future_in_input": False' in source
    for forbidden in ("GT future human trajectory", "GT harm", "GT benefit", "profile ID", "oracle action"):
        assert forbidden in source


def test_h0_reproduction_is_strict_and_stops_on_failure():
    source = inspect.getsource(stage.main)
    assert stage.H0_TOLERANCE == 1e-12
    assert "H0 failed strict Phase5B-1.7E reproduction" in source
    assert stage.H0_EXPECTED["AUROC"] == pytest.approx(.7750483796995425)


def test_training_reuses_unweighted_phs_v1_protocol():
    source = inspect.getsource(stage.main)
    assert "e.train_head" in source
    assert "prevalence_baseline" in source
    assert '"loss": "unweighted BCEWithLogitsLoss"' in source
    assert '"selector": "PHS-v1: NLL -> Brier -> AUROC -> earlier epoch"' in source


def test_no_forbidden_head_or_training_intervention():
    source = inspect.getsource(RiskPreservingBypassHead)
    assert "nn.Linear" in source and "nn.Sequential" not in source
    main = inspect.getsource(stage.main)
    assert "MinimalNonlinearHarmV2Probe" not in main
    assert "fixed_noisy_or" not in main
    assert '"class_weighting": False' in main and '"oversampling": False' in main and '"focal_loss": False' in main


def test_optimizer_is_checked_as_head_only():
    source = inspect.getsource(stage.main)
    assert '"optimizer_only_harm_heads"' in source
    assert 'h0_training["optimizer_exactly_head"] and h1_training["optimizer_exactly_head"]' in source


def test_all_shared_checksums_and_behavior_are_audited():
    source = inspect.getsource(stage.main)
    for field in ("full_model_checksum_before", "temporal_backbone_checksum_unchanged", "benefit_head_checksum_unchanged",
                  "ranking_behavior_checksum_unchanged", "benefit_output_checksum_unchanged"):
        assert field in source
    assert '"all_shared_parameters_require_grad_false"' in source


def test_gate_thresholds_are_fixed():
    frozen = {"temporal_backbone_checksum_unchanged": True, "benefit_output_checksum_unchanged": True,
              "ranking_behavior_checksum_unchanged": True, "optimizer_only_harm_heads": True, "H0_strict_reproduction": True}
    h0 = {"AUROC": .775, "AUPRC": .57, "NLL": .46, "Brier": .15}
    h1 = {"AUROC": .805, "AUPRC": .60, "NLL": .45, "Brier": .14, "prevalence": .2475}
    baseline = {"NLL": .56, "Brier": .186}
    safe = {"H0": {"mean": .15, "P90": .29, "P95": .36}, "H1": {"mean": .14, "P90": .25, "P95": .32}}
    subtype = {"H0": {"GT_UNSAFE": {"AUROC": .90}}, "H1": {"GT_UNSAFE": {"AUROC": .88}}}
    gates = stage.evaluate_gates(frozen, h0, h1, baseline, safe, subtype)
    assert gates["all_passed"]
    h1["AUROC"] = .799
    assert not stage.evaluate_gates(frozen, h0, h1, baseline, safe, subtype)["Gate_B"]["passed"]


def test_safe_tail_gate_does_not_invent_posthoc_tolerance():
    source = inspect.getsource(stage.evaluate_gates)
    assert "safe_P90_not_higher_than_H0" in source
    assert "safe_P95_not_higher_than_H0" in source
    assert "P90_increase_at_most" not in source and "P95_increase_at_most" not in source


def test_checkpoint_is_impossible_until_all_four_gates_pass(tmp_path):
    import torch
    path = tmp_path / "formal.pt"; head = RiskPreservingBypassHead()
    gates = {"all_passed": False}
    assert not stage.conditional_checkpoint(path, head, gates, {}, {}, torch)
    assert not path.exists()


def test_test_remains_sealed_and_threshold_stage_not_started():
    source = inspect.getsource(stage.main)
    assert '"test_reads": 0' in source
    assert '"threshold_calibration_performed": False' in source
    assert '"phase5b17f_started": False' in source


def test_profile_is_audit_only_not_input():
    source = inspect.getsource(stage.extract_inputs)
    assert "profile" not in source
    source = inspect.getsource(stage.main)
    assert '"profile_id_in_input": False' in source
