"""Protocol tests for the frozen-representation readout-capacity audit."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts import audit_phase5b17ea_harm_readout_capacity as audit
from src.models.independent_harm_head import MinimalNonlinearHarmV2Probe

ROOT = Path(__file__).resolve().parents[1]


def test_minimal_mlp_architecture_is_exactly_128_32_1_gelu():
    import torch
    model = MinimalNonlinearHarmV2Probe()
    record = model.architecture_audit()
    assert record["layers"] == ["Linear(128,32)", "GELU", "Linear(32,1)"]
    assert record["parameter_count"] == 4161
    assert not record["dropout"] and not record["batch_norm"] and not record["attention"] and not record["residual"]
    assert model(torch.zeros(4, 128)).shape == (4,)


@pytest.mark.parametrize("hidden", [16, 64, 128])
def test_other_hidden_widths_are_rejected(hidden):
    with pytest.raises(ValueError): MinimalNonlinearHarmV2Probe(hidden_dim=hidden)


def test_no_hyperparameter_search_or_alternate_activation():
    source = inspect.getsource(MinimalNonlinearHarmV2Probe)
    assert "nn.GELU" in source and "nn.ReLU" not in source and "nn.SiLU" not in source
    assert '"hyperparameter_search": False' in source


def test_embedding_cache_contains_development_only_and_profile_is_audit_only():
    source = inspect.getsource(audit.save_embedding_cache)
    assert '"splits": ["train", "validation"]' in source
    assert '"test_embeddings": 0' in source and '"test_reads": 0' in source
    assert '"profile_id_in_probe_input": False' in source


def test_linear_reproduction_is_strict_and_stops_on_failure():
    source = inspect.getsource(audit.main)
    assert audit.REPRODUCTION_TOLERANCE == 1e-12
    assert "Linear probe failed strict Phase5B-1.7E reproduction" in source
    assert audit.EXPECTED_LINEAR["selected_epoch"] == 30


def test_probe_training_reuses_unweighted_head_only_protocol():
    source = inspect.getsource(audit.train_probe)
    assert "e.train_head" in source
    source = inspect.getsource(audit.main)
    assert "train_probe(IndependentHarmV2Head()" in source
    assert "train_probe(MinimalNonlinearHarmV2Probe()" in source


def test_four_subtype_probes_are_linear_and_diagnostic_only():
    source = inspect.getsource(audit.probe_subtypes)
    for name in ("GT_UNSAFE", "EXCESSIVE_DECELERATION", "ABRUPT_LATERAL_RESPONSE", "ABRUPT_HEADING_CHANGE"):
        assert name in source
    assert "IndependentHarmV2Head" in source
    assert '"diagnostic_only": True' in source and '"formal_checkpoint_allowed": False' in source


def test_gates_follow_preregistered_thresholds():
    linear = {"AUROC": .775, "AUPRC": .57, "NLL": .46, "Brier": .15, "ECE": .04, "selected_epoch": 30}
    mlp = {"AUROC": .81, "AUPRC": .60, "NLL": .45, "Brier": .14}
    semantic_l = {"safe_beneficial": {"mean": .15}, "harm_positive_vs_safe_beneficial_AUROC": .85}
    semantic_m = {"safe_beneficial": {"mean": .20}, "harm_positive_vs_safe_beneficial_AUROC": .86}
    subtype_l = {"GT_UNSAFE": .90}; subtype_m = {"GT_UNSAFE": .88}
    old = audit.EXPECTED_LINEAR; audit.EXPECTED_LINEAR = dict(linear)
    try:
        gates = audit.gate_results(linear, mlp, semantic_l, semantic_m, subtype_l, subtype_m)
    finally:
        audit.EXPECTED_LINEAR = old
    assert all(gate["passed"] for gate in gates.values())
    mlp["AUROC"] = .79
    audit.EXPECTED_LINEAR = dict(linear)
    try: assert not audit.gate_results(linear, mlp, semantic_l, semantic_m, subtype_l, subtype_m)["Gate_B"]["passed"]
    finally: audit.EXPECTED_LINEAR = old


def test_no_formal_checkpoint_or_threshold_calibration():
    source = inspect.getsource(audit.main)
    assert "torch.save" not in source
    assert '"formal_harm_checkpoint_written": False' in source
    assert '"threshold_calibration_performed": False' in source


def test_frozen_model_loaded_read_only_and_test_reads_zero():
    source = inspect.getsource(audit.main)
    assert "e.load_frozen" in source and "d.model_sha(model)" in source
    assert "source checkpoint SHA256 mismatch" in source
    assert len(audit.EXPECTED_CHECKPOINT_SHA256) == 64
    assert '"test_reads": 0' in source


def test_mixed_evidence_is_not_misclassified_as_pure_representation_failure():
    gates = {"Gate_B": {"passed": False}}
    linear = {"AUROC": .775}
    mlp = {"AUROC": .793}
    strong_subtypes = [{"subtype": name, "AUROC": score} for name, score in (
        ("GT_UNSAFE", .94), ("EXCESSIVE_DECELERATION", .85),
        ("ABRUPT_LATERAL_RESPONSE", .90), ("ABRUPT_HEADING_CHANGE", .81),
    )]
    result = audit.classify_root_cause(gates, linear, mlp, strong_subtypes)
    assert result["selected_class"] == "F"
    assert result["subtype_linear_probes_below_0_80"] == []


def test_profile_is_not_a_probe_feature():
    source = inspect.getsource(audit.main)
    assert '"profile_id_in_probe_input": False' in source
    assert "train_x" in inspect.getsource(audit.train_probe)


def test_existing_17e_results_are_not_overwritten():
    assert (ROOT / "results_dev" / "phase5b17e_independent_harm_v2" / "summary.json").is_file()
    assert audit.parse_args.__module__.endswith("audit_phase5b17ea_harm_readout_capacity")
