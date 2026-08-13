import inspect
from types import SimpleNamespace

import numpy as np

from scripts import run_phase5b17a_target_conflict_audit as audit


def _sample(episode, candidate, benefit, harm, feasible=True, unsafe=False):
    return SimpleNamespace(
        sample_id=f"validation:{episode}:{candidate}", episode_id=f"validation:{episode}",
        targets=SimpleNamespace(benefit=benefit, harm=harm, feasible=feasible, gt_unsafe=unsafe, gt_cost=1.0),
        split_metadata={"candidate_action_id_audit": candidate, "motion_type_evaluation_only": "walk", "person_profile_id": 0},
        temporal_tags=(), context_split="validation",
    )


def test_target_definition_matches_source_contract():
    definition = audit.source_definition()
    assert definition["benefit"]["formula"].startswith("benefit = GT_total_cost")
    assert definition["harm"]["formula"] == "harm = (benefit < -1e-6)"
    assert definition["gt_unsafe"]["formula"].endswith("unsafe_duration[candidate] > 0)")
    assert not definition["semantic_identity"]["GT_harm_equals_GT_unsafe"]


def test_four_quadrant_counts_sum_to_candidate_total():
    samples = [_sample("e", 0, .1, False), _sample("e", 1, .2, True),
               _sample("e", 2, -.1, True), _sample("e", 3, 0, False)]
    rows = audit.overlap_rows("validation", samples)
    assert sum(row["candidate_count"] for row in rows) == len(samples)
    assert [row["candidate_count"] for row in rows] == [1, 1, 1, 1]


def test_episode_counts_are_consistent_with_candidate_groups():
    samples = [_sample("e0", 0, .1, False), _sample("e0", 1, .2, False), _sample("e1", 0, .3, False)]
    q1 = audit.overlap_rows("validation", samples)[0]
    assert q1["candidate_count"] == 3 and q1["episode_count"] == 2


def test_safe_beneficial_definition_is_audit_only_conjunction():
    samples = [_sample("e", 0, .1, False, True), _sample("e", 1, .1, True, True),
               _sample("e", 2, .1, False, False)]
    value = audit.masks(samples)
    assert value["safe_beneficial"].tolist() == [True, False, False]
    source = inspect.getsource(audit.main)
    assert '"GT_targets_runtime_input":False' in source


def test_safe_beneficial_funnel_uses_sequential_gate_counts():
    source = inspect.getsource(audit.safe_beneficial_rows)
    assert "sequential_harm_pass" in source and "sequential_generic_score_win" in source
    assert "benefit_threshold_pass" in source and "harm_threshold_pass" in source


def test_harm_and_benefit_targets_are_audit_only_not_model_inputs():
    source = inspect.getsource(audit)
    assert "model(samples" not in source
    assert '"GT_targets_audit_only":True' in source


def test_person_id_is_audit_metadata_only():
    source = inspect.getsource(audit.main)
    assert '"person_profile_runtime_input":False' in source
    assert '"person_profile_audit_metadata_only":True' in source


def test_test_is_never_materialized_or_read():
    source = inspect.getsource(audit)
    forbidden = "materialize" + "_test"
    assert forbidden not in source
    assert '"test_candidates_read":0' in source and '"test_labels_read":0' in source


def test_no_training_optimizer_or_backward():
    source = inspect.getsource(audit)
    assert "torch.optim" not in source and ".backward(" not in source
    assert '"optimizer_step_count":0' in source and '"backward_call_count":0' in source


def test_thresholds_and_ranking_lambda_are_frozen():
    assert audit.BENEFIT_THRESHOLD == -.02 and audit.HARM_THRESHOLD == .2 and audit.LAMBDA_RANK == .25
    source = inspect.getsource(audit.parse_args)
    assert "threshold" not in source.lower() and "lambda" not in source.lower()


def test_arbitration_and_model_checksum_are_audited_unchanged():
    source = inspect.getsource(audit.main)
    assert "model_checksum_before" in source and "model_checksum_after" in source
    assert "arbitration_sha256_before" in source and '"arbitration_unchanged":True' in source


def test_conditional_calibration_preserves_bin_counts_without_smoothing():
    samples = [_sample("e", 0, .1, False), _sample("e", 1, -.1, True)]
    rows = audit.conditional_calibration_rows("validation", samples, {"harm": np.asarray([.7, .9])})
    beneficial = [row for row in rows if row["subgroup"] == "beneficial" and row["row_type"] == "reliability_bin"]
    assert sum(row["bin_count"] for row in beneficial) == 1
    assert any(row["bin_count"] == 0 and row["observed_harm_frequency"] is None for row in beneficial)


def test_dependency_matrix_marks_shared_total_cost_construction():
    rows = audit.dependency_rows()
    assert len(rows) == 5 and all(row["shared_by_benefit_and_harm"] for row in rows)
    assert all(row["target_construction_confounding"] for row in rows)


def test_oracle_safe_beneficial_requires_real_generic_cost_improvement():
    samples = [_sample("e", 0, .2, False), _sample("e", 1, .1, False)]
    rows = audit.oracle_rows("validation", samples)
    assert len(rows) == 1 and rows[0]["personalized_strictly_better_than_generic"]
    assert np.isclose(rows[0]["personalization_improvement"], .2)
