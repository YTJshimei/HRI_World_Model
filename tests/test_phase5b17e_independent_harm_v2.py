"""Safety, semantics and selector tests for Phase 5B-1.7E."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts import run_phase5b17d_manifest_v2_rebaseline as frozen
from scripts import run_phase5b17e_independent_harm_v2 as stage
from src.evaluation.probabilistic_harm import harm_metrics, phs_select, prevalence_baseline
from src.models.independent_harm_head import IndependentHarmV2Head
from src.training.independent_harm import harm_v2_target, unweighted_harm_v2_loss

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA = "eb8321e9b4f3cd7213ec52c48169857d37980f5457568fc832ff486157914ce8"


def test_manifest_and_frozen_cracs_checkpoint_hashes():
    assert frozen.EXPECTED_MANIFEST_SHA == "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
    checkpoint = ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt"
    assert stage.file_sha(checkpoint) == EXPECTED_CHECKPOINT_SHA


def test_independent_head_matches_old_capacity_but_is_separate():
    import torch
    head = IndependentHarmV2Head()
    assert head.architecture_audit()["layers"] == ["Linear(128,1)"]
    assert sum(parameter.numel() for parameter in head.parameters()) == 129
    assert set(head.state_dict()) == {"linear.weight", "linear.bias"}
    assert head(torch.zeros(3, 128)).shape == (3,)


@dataclass
class Targets:
    gt_unsafe: bool


@dataclass
class Sample:
    targets: Targets
    split_metadata: dict


@pytest.mark.parametrize("unsafe,decel,lateral,heading,expected", [
    (False, False, False, False, False), (True, False, False, False, True),
    (False, True, False, False, True), (False, False, True, False, True),
    (False, False, False, True, True),
])
def test_harm_v2_is_unsafe_or_adverse_response(unsafe, decel, lateral, heading, expected):
    sample = Sample(Targets(unsafe), {"excessive_deceleration_evaluation_only": decel,
        "abrupt_lateral_response_evaluation_only": lateral, "abrupt_heading_change_evaluation_only": heading,
        "harm_v2_evaluation_only": expected})
    assert harm_v2_target(sample) is expected


def test_old_harm_target_is_not_used_and_bce_is_unweighted():
    source = inspect.getsource(unweighted_harm_v2_loss)
    assert "binary_cross_entropy_with_logits" in source
    call = source.split("return ", 1)[1]
    assert "weight=" not in call and "pos_weight" not in call
    assert "benefit" not in inspect.getsource(harm_v2_target)


def test_prevalence_baseline_is_exact():
    train = np.asarray([1, 0, 0, 0], bool); validation = np.asarray([1, 1, 0, 0], bool)
    result = prevalence_baseline(train, validation)
    expected_nll = -np.mean(validation * np.log(.25) + (~validation) * np.log(.75))
    assert result["train_prevalence_probability"] == .25
    assert result["validation_prevalence"] == .5 and result["AUPRC"] == .5 and result["AUROC"] == .5
    assert result["NLL"] == pytest.approx(expected_nll)
    assert result["Brier"] == pytest.approx(np.mean((.25 - validation.astype(float)) ** 2))


def test_phs_selector_is_deterministic_and_uses_declared_ties():
    rows = [{"epoch": 4, "NLL": .4, "Brier": .2, "AUROC": .9}, {"epoch": 2, "NLL": .4, "Brier": .2, "AUROC": .9}]
    assert phs_select(rows)["epoch"] == 2
    rows.append({"epoch": 3, "NLL": .4, "Brier": .19, "AUROC": .8})
    assert phs_select(rows)["epoch"] == 3
    rows.append({"epoch": 5, "NLL": .39, "Brier": .5, "AUROC": .5})
    assert phs_select(rows)["epoch"] == 5


def test_gate_calculations_are_fixed():
    isolation = {"frozen": True, "optimizer": True}
    metrics = {"AUROC": .81, "AUPRC": .51, "prevalence": .30, "NLL": .50, "Brier": .17}
    baseline = {"NLL": .60, "Brier": .20}
    separation = {"positive": {"mean": .7}, "safe_beneficial": {"mean": .2}, "harm_positive_vs_safe_beneficial_AUROC": .9}
    gates = stage.evaluate_gates(isolation, metrics, baseline, separation)
    assert all(gate["passed"] for gate in gates.values())
    metrics["AUPRC"] = .499
    assert not stage.evaluate_gates(isolation, metrics, baseline, separation)["Gate_B"]["passed"]


def test_only_three_gate_pass_can_save_checkpoint():
    source = inspect.getsource(stage.save_checkpoint)
    assert '("Gate_A", "Gate_B", "Gate_C")' in source and "torch.save" in source


def test_training_source_enforces_isolation_and_no_sampling_or_weighting():
    train_source = inspect.getsource(stage.train_head)
    main_source = inspect.getsource(stage.main)
    assert "optimizer = torch.optim.AdamW(head.parameters()" in train_source
    assert "model.parameters()" not in train_source
    assert "WeightedRandomSampler" not in train_source and "pos_weight" not in train_source
    assert '"oversampling": False' in main_source and '"class_weighting": False' in main_source
    assert "for parameter in model.parameters(): parameter.requires_grad_(False)" in inspect.getsource(stage.load_frozen)


def test_safe_beneficial_and_tradeoff_definitions_are_frozen():
    from src.multimodal import phase5b_v2_dataset as bridge
    source = inspect.getsource(bridge.build_v2_temporal_samples)
    assert 'candidate.benefit > 1e-6 and not candidate.harm_v2' in source
    assert 'candidate.benefit > 1e-6 and candidate.harm_v2' in source


def test_profile_is_audit_only_and_not_runtime_input():
    from src.multimodal import phase5b_v2_dataset as bridge
    source = inspect.getsource(bridge.build_v2_temporal_samples)
    assert '"person_profile_id": episode.profile_id' in source
    assert 'forbidden = {"profile_id", "person_profile_id"' in inspect.getsource(bridge.runtime_contract_audit)
    assert '"profile_id_in_runtime_model": False' in inspect.getsource(stage.main)


def test_test_reads_zero_and_no_threshold_calibration_or_decision_gate():
    source = inspect.getsource(stage.main)
    assert '"test_reads": 0' in source
    assert '"formal_threshold_calibration_performed": False' in source
    assert '"formal_decision_gate_performed": False' in source


def test_harm_metrics_known_values():
    result = harm_metrics(np.asarray((.9, .8, .2, .1)), np.asarray((1, 1, 0, 0), bool))
    assert result["AUROC"] == 1 and result["AUPRC"] == 1
    assert result["Accuracy_at_0_5"] == 1 and result["F1_at_0_5"] == 1
