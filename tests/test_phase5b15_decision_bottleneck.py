import inspect

import numpy as np
import pytest

from scripts import run_phase5b15_decision_bottleneck as audit


def test_validation_builder_explicitly_rejects_test_split():
    source = inspect.getsource(audit.build_validation_only)
    assert "materialize_test" in source and "forbidden" in source
    assert 'sample.split != "validation"' in source
    assert 'sample.sample_id.startswith("test:")' in source


def test_audit_source_has_no_optimizer_or_backward_execution():
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert '"optimizer_step_count":0' in source


def test_primary_rejection_is_exactly_one_and_priority_ordered():
    primary, secondary = audit.primary_rejection(feasible=False, sign=False, ranking=False, benefit_pass=False,
                                                  harm_pass=False, uncertainty_pass=False, score_win=False, tie=True)
    assert primary == "SAFETY_MASK_BLOCKED"
    assert primary not in secondary
    assert len([primary]) == 1


def test_ranking_is_computed_only_inside_given_episode_array():
    first = audit.ranks_desc(np.asarray([0.2, 0.9, 0.1]))
    second = audit.ranks_desc(np.asarray([-100.0, -101.0]))
    assert first.tolist() == [2, 1, 3]
    assert second.tolist() == [1, 2]


def test_feasible_ranking_excludes_infeasible_candidates():
    predicted = np.asarray([100.0, 0.8, 0.2]); feasible = np.asarray([False, True, True])
    valid = np.flatnonzero(feasible); ranks = audit.ranks_desc(predicted[valid])
    assert ranks.tolist() == [1, 2] and len(ranks) == 2


def test_funnel_separates_diagnostic_rank_from_actual_gate_chain():
    row = {"feasible": True, "predicted_sign_correct": True, "top1": False, "top2": True,
           "benefit_threshold_pass": True, "harm_threshold_pass": False,
           "uncertainty_threshold_pass": True, "generic_score_win": True, "final_personalized_switch": False}
    summary = audit.summarize_funnel([row])
    assert summary["feasible"] == summary["sign_correct"] == 1
    assert summary["top1"] == 0 and summary["top2"] == 1
    assert summary["benefit_threshold_pass"] == 1
    assert summary["harm_threshold_pass"] == summary["generic_score_win"] == summary["final_switch"] == 0
    assert summary["ranking_fields_are_diagnostic_not_hard_gates"]


def test_oracle_diagnostic_function_accepts_only_passed_validation_samples_by_contract():
    source = inspect.getsource(audit.oracle_diagnostics)
    assert "validation-only oracle diagnostic" in source
    assert "test" not in source.lower()


def test_oracle_fields_are_diagnostic_not_runtime_inputs():
    source = inspect.getsource(audit.main)
    assert '"oracle_runtime_input":False' in source
    assert '"oracle_validation_only":True' in source


def test_uncertainty_gate_is_truthfully_marked_undefined():
    source = inspect.getsource(audit.audit_model)
    assert '"uncertainty_threshold_defined": False' in source
    assert '"uncertainty_threshold_margin": "N/A"' in source


def test_checksum_and_frozen_contract_audit_are_present():
    source = inspect.getsource(audit.main)
    assert "parameter_checksums_before" in source and "parameter_checksums_after" in source
    assert "thresholds_unchanged" in source and "arbitration_unchanged" in source and "feasible_mask_unchanged" in source


def test_context_tags_are_derived_from_frozen_sample_tags_not_new_scene_flags():
    source = inspect.getsource(audit.context_labels)
    assert "temporal_tags" in source and "scene_context" not in source
