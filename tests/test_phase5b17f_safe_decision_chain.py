"""Protocol and decision semantics for Phase 5B-1.7F."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_phase5b17f_safe_decision_chain as stage
from src.evaluation.safe_decision_chain import (
    BENEFIT_THRESHOLD, decide_episode, gate_results, safe_beneficial_mask,
    threshold_selection_key, tradeoff_mask,
)


def _episode(episode="e0", *, motion="walk", profile=0, contexts=("C7",)):
    actions = np.asarray([0, 1])
    generic = np.asarray([1.0, 2.0])
    personalized = np.asarray([1.0, 0.1])
    gt = np.asarray([1.0, 0.2])
    result = []
    for local, (benefit, harm, unsafe) in enumerate(((0.0, False, False), (.2, False, False))):
        result.append(SimpleNamespace(
            sample_id=f"validation:{episode}:{local}", episode_id=f"validation:{episode}", split="validation",
            targets=SimpleNamespace(benefit=benefit, feasible=True, gt_unsafe=unsafe),
            split_metadata={
                "candidate_action_id_audit": local, "all_action_ids_evaluation_only": actions,
                "generic_costs_evaluation_only": generic, "personalized_costs_evaluation_only": personalized,
                "gt_costs_evaluation_only": gt, "harm_v2_evaluation_only": harm,
                "person_profile_id": profile, "motion_type_evaluation_only": motion,
                "contexts_evaluation_only": contexts,
                "excessive_deceleration_evaluation_only": False,
                "abrupt_lateral_response_evaluation_only": False,
                "abrupt_heading_change_evaluation_only": False,
            },
        ))
    return result


def test_strict_harm_gate_rejects_probability_equal_to_threshold():
    samples = _episode()
    exact, _ = decide_episode(samples, np.asarray([0.0, .2]), np.asarray([0.0, .5]), .5, "D2")
    below, _ = decide_episode(samples, np.asarray([0.0, .2]), np.asarray([0.0, np.nextafter(.5, 0.0)]), .5, "D2")
    assert not exact.personalized
    assert below.personalized and below.selected_action == 1


def test_d1_disables_only_harm_gate():
    samples = _episode()
    low, rows_low = decide_episode(samples, np.asarray([0.0, .2]), np.asarray([0.0, 0.0]), None, "D1")
    high, rows_high = decide_episode(samples, np.asarray([0.0, .2]), np.asarray([0.0, 1.0]), None, "D1")
    assert low == high and all(row["harm_v2_gate_pass"] for row in rows_low + rows_high)


def test_d0_never_records_personalized_switch():
    decision, _ = decide_episode(_episode(), np.asarray([0.0, 9.0]), np.zeros(2), .5, "D0")
    assert not decision.personalized and not decision.selected_safe_beneficial


def test_safety_violation_counts_any_final_unsafe_but_switch_metric_is_personalized_only():
    samples = _episode()
    samples[0].targets = SimpleNamespace(benefit=0.0, feasible=True, gt_unsafe=True)
    decision, rows = decide_episode(samples, np.asarray([0.0, .2]), np.zeros(2), .5, "D0")
    metrics = stage.summarize_decisions([decision], rows)
    assert metrics["GT_unsafe_switch_count"] == 0
    assert metrics["GT_unsafe_final_selected_count"] == 1
    assert metrics["Safety_Violation"] == 1.0


def test_candidate_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="candidate arrays must align"):
        decide_episode(_episode(), np.zeros(1), np.zeros(2), .5, "D2")


def test_decision_contract_rejects_column_vectors_instead_of_silent_broadcast():
    with pytest.raises(ValueError, match="candidate arrays must align"):
        decide_episode(_episode(), np.zeros((2, 1)), np.zeros((2, 1)), .5, "D2")


def test_threshold_priority_is_safety_first_then_value_then_conservative():
    safer = {"GT_unsafe_switch_count": 0, "GT_harm_v2_risky_switch_count": 0,
             "safe_beneficial_episode_recall": .1, "Mean_Regret": .5, "safe_beneficial_precision": .1}
    unsafe = dict(safer, GT_unsafe_switch_count=1, safe_beneficial_episode_recall=1.0, Mean_Regret=0.0)
    assert threshold_selection_key(safer, .2) < threshold_selection_key(unsafe, .9)
    better = dict(safer, safe_beneficial_episode_recall=.5)
    assert threshold_selection_key(better, .3) < threshold_selection_key(safer, .1)
    assert threshold_selection_key(better, .2) < threshold_selection_key(better, .3)


def test_safe_beneficial_and_tradeoff_definitions_are_disjoint():
    samples = _episode()
    samples[1].split_metadata["harm_v2_evaluation_only"] = True
    assert safe_beneficial_mask(samples).tolist() == [False, False]
    assert tradeoff_mask(samples).tolist() == [False, True]


def test_threshold_selection_accepts_calibration_only():
    assert list(inspect.signature(stage.choose_threshold).parameters) == ["samples", "prediction"]
    source = inspect.getsource(stage.choose_threshold)
    assert "evaluation" not in source.lower() and "test" not in source.lower()


def test_split_is_deterministic_balanced_and_episode_disjoint():
    samples = []
    for index in range(8):
        samples.extend(_episode(f"e{index}", motion="stop" if index % 2 else "walk",
                                profile=index % 2, contexts=("C7", "C8") if index % 3 == 0 else ("C9",)))
    first = stage.split_validation_episodes(samples, 42)
    second = stage.split_validation_episodes(samples, 42)
    assert first == second
    calibration, evaluation, _ = first
    assert len(calibration) == len(evaluation) == 4
    assert not set(calibration) & set(evaluation)
    assert set(calibration) | set(evaluation) == {sample.episode_id for sample in samples}


def test_subsetting_never_splits_candidate_branches():
    samples = _episode("a") + _episode("b")
    arrays = {"benefit": np.arange(4), "harm_v2": np.arange(4) / 10}
    selected, values = stage.subset(samples, arrays, ["validation:a"])
    assert len(selected) == 2 and {item.episode_id for item in selected} == {"validation:a"}
    assert values["benefit"].tolist() == [0, 1]


def test_threshold_candidates_include_strict_boundary_escape():
    values = stage.threshold_candidates(np.asarray([.1, .4, .8]))
    assert 0.0 in values and .1 in values and .8 in values
    assert values.max() > .8


def test_gate_contract_requires_all_five_groups():
    integrity = {"split": True, "sealed": True}
    d0 = {"Mean_Regret": .2, "P95_Regret": .4}
    d1 = {"GT_harm_v2_risky_switch_count": 3, "GT_unsafe_switch_count": 1}
    d2 = {"GT_harm_v2_risky_switch_count": 0, "GT_unsafe_switch_count": 0,
          "safe_beneficial_switch_count": 2, "safe_beneficial_episode_recall": .2,
          "Mean_Regret": .1, "P95_Regret": .3}
    result = gate_results(integrity, d0, d1, d2)
    assert result["all_passed"] and set(result) == {"Gate_A", "Gate_B", "Gate_C", "Gate_D", "Gate_E", "all_passed"}


def test_gate_b_and_gate_e_fail_independently_and_cannot_be_offset_by_regret():
    integrity = {"sealed": True}
    d0 = {"Mean_Regret": 1.0, "P95_Regret": 1.0}
    d1 = {"GT_harm_v2_risky_switch_count": 2, "GT_unsafe_switch_count": 0}
    d2 = {"GT_harm_v2_risky_switch_count": 1, "GT_unsafe_switch_count": 0,
          "safe_beneficial_switch_count": 1, "safe_beneficial_episode_recall": .1,
          "Mean_Regret": 0.0, "P95_Regret": 0.0}
    result = gate_results(integrity, d0, d1, d2)
    assert not result["Gate_B"]["passed"] and result["Gate_E"]["passed"]


def test_frozen_threshold_and_checkpoint_are_explicit_constants():
    assert BENEFIT_THRESHOLD == -.02
    assert stage.EXPECTED_HARM_CHECKPOINT_SHA256 == "68974836d2f515479f63ea8b7b323364e8b5eadb29db0fe9f615843fcb65370d"


def test_script_has_no_training_or_test_materialization_path():
    source = inspect.getsource(stage)
    assert "torch.optim" not in source and ".backward(" not in source
    assert '"optimizer_steps": 0' in source and '"backward_calls": 0' in source and '"test_reads": 0' in source
    assert "build_development_split(\"test\"" not in source


def test_old_harm_is_absent_and_frozen_assets_are_audited():
    source = inspect.getsource(stage.main)
    assert '"old_harm_probability_computed": False' in source
    assert '"old_harm_gate_in_decision_chain": False' in source
    for field in ("R1_checkpoint_unchanged", "harm_v2_head_unchanged", "benefit_threshold_unchanged",
                  "safety_mask_unchanged", "generic_score_unchanged", "arbitration_unchanged",
                  "decision_costs_unchanged", "all_model_parameters_frozen"):
        assert field in source


def test_required_outputs_and_synthetic_label_are_hard_coded():
    source = inspect.getsource(stage.main)
    required = (
        "frozen_contract.json", "validation_threshold_split.json", "threshold_candidate_metrics.csv",
        "harm_v2_threshold_selection.json", "decision_chain_contract.json", "d0_d1_d2_metrics.csv",
        "safe_beneficial_funnel.csv", "risky_candidate_funnel.csv", "benefit_risk_tradeoff.csv",
        "gt_unsafe_audit.csv", "by_harm_subtype.csv", "deceleration_warning_audit.csv",
        "stop_audit.csv", "by_context.csv", "by_motion.csv", "by_action.csv", "gate_results.json", "summary.json",
    )
    assert all(name in source for name in required)
    assert source.count("LABEL") >= 1


def test_failed_gate_cannot_recommend_multiseed():
    source = inspect.getsource(stage.main)
    fail_branch = source[source.index('if not gates["all_passed"]'):source.index("figures = make_figures")]
    assert 'next_recommendation = "Stop and diagnose safety subtype"' in fail_branch
    assert fail_branch.index("Stop and diagnose safety subtype") < fail_branch.index("Multi-seed confirmation")


def test_context_groups_are_independent_multilabel_membership_checks():
    source = inspect.getsource(stage.main)
    assert "context_groups" in source
    assert "next((name" not in source


def test_funnels_are_monotone_cumulative():
    rows = [
        {"episode_id": "e", "GT_benefit_positive": True, "harm_v2": False, "feasible": True,
         "benefit_sign_correct": True, "benefit_threshold_pass": True, "benefit_rank": 1,
         "harm_v2_gate_pass": True, "generic_score_win": True, "personalized_selected": True},
        {"episode_id": "f", "GT_benefit_positive": True, "harm_v2": False, "feasible": True,
         "benefit_sign_correct": False, "benefit_threshold_pass": True, "benefit_rank": 1,
         "harm_v2_gate_pass": True, "generic_score_win": True, "personalized_selected": False},
    ]
    counts = [row["candidate_count"] for row in stage.funnel_rows(rows, "safe")]
    assert counts == sorted(counts, reverse=True)
