"""Phase 5B-1.7F-B isolation, selection, metrics, and gate tests."""
from __future__ import annotations

import inspect

import numpy as np

from scripts import run_phase5b17fb_generic_safety_coverage as stage
from src.decision.generic_risk_coverage import select_with_generic_risk_coverage
from src.evaluation.generic_safety_coverage import gate_results


def _arrays():
    actions = np.asarray([0, 1, 2])
    feasible = np.asarray([True, True, True])
    generic = np.asarray([.1, .2, .3])
    personalized = np.asarray([.1, .2, .3])
    benefit = np.asarray([0.0, 0.0, 0.0])
    risk = np.asarray([.9, .05, .03])
    return actions, feasible, generic, personalized, benefit, risk


def test_generic_replacement_uses_existing_lowest_cost_risk_eligible_candidate():
    result = select_with_generic_risk_coverage(*_arrays(), -.02, .1)
    assert result.selected_index == 1 and result.selected_action == 1
    assert not result.personalized and not result.abstained


def test_strict_risk_threshold_rejects_probability_equal_to_threshold():
    values = list(_arrays()); values[-1] = np.asarray([.1, .09, .08])
    result = select_with_generic_risk_coverage(*values, -.02, .1)
    assert result.selected_index == 1


def test_replacement_never_synthesizes_action():
    values = list(_arrays()); values[0] = np.asarray([10, 20, 30])
    result = select_with_generic_risk_coverage(*values, -.02, .1)
    assert result.selected_action in values[0]


def test_no_safe_generic_reuses_abstain_semantics():
    values = list(_arrays()); values[-1] = np.ones(3)
    result = select_with_generic_risk_coverage(*values, -.02, .1)
    assert result.abstained and result.selected_action is None
    assert result.reason == "NO_SAFE_GENERIC_CANDIDATE"


def test_frozen_d2_personalized_decision_is_preserved_before_generic_filter():
    actions, feasible, generic, personalized, benefit, risk = _arrays()
    personalized[:] = [1.0, 0.0, 1.0]; benefit[:] = [0.0, .5, 0.0]
    result = select_with_generic_risk_coverage(actions, feasible, generic, personalized, benefit, risk, -.02, .1)
    assert result.personalized and result.selected_action == 1
    assert result.reason == "FROZEN_D2_PERSONALIZED_PRESERVED"


def test_generic_self_approval_is_not_mislabeled_as_personalized():
    actions, feasible, generic, personalized, benefit, risk = _arrays()
    risk[:] = [.05, .04, .03]; benefit[:] = [.2, 0.0, 0.0]
    result = select_with_generic_risk_coverage(actions, feasible, generic, personalized, benefit, risk, -.02, .1)
    assert result.selected_action == 0 and not result.personalized


def test_function_accepts_no_gt_inputs():
    parameters = inspect.signature(select_with_generic_risk_coverage).parameters
    assert not any("gt" in name.lower() or "unsafe" in name.lower() for name in parameters)


def _metrics(**updates):
    base = {"total_GT_unsafe_final_count": 0, "total_harm_v2_final_count": 0, "no_safe_generic_count": 0,
            "personalized_harm_v2_count": 0, "safe_beneficial_episode_recall": .44,
            "safe_beneficial_precision": .78, "Overall_Safety_Violation": 0.0,
            "Mean_Regret": .2, "P95_Regret": .3}
    base.update(updates); return base


def test_gate_a_to_f_pass_contract():
    d2 = _metrics(Overall_Safety_Violation=.05, Mean_Regret=.25, P95_Regret=.35)
    d3 = _metrics()
    gates = gate_results({"frozen": True}, d2, d3, 6, 6, 0, 0)
    assert gates["all_passed"]
    assert all(gates[f"Gate_{letter}"]["passed"] for letter in "ABCDEF")


def test_gate_c_fails_for_risk_or_missing_fallback():
    d2 = _metrics(Overall_Safety_Violation=.05)
    gates = gate_results({"frozen": True}, d2, _metrics(total_harm_v2_final_count=1, no_safe_generic_count=2), 6, 6, 0, 0)
    assert not gates["Gate_C"]["passed"] and not gates["all_passed"]


def test_gate_d_detects_risk_transfer_and_latent_debt_activation():
    d2 = _metrics(Overall_Safety_Violation=.05)
    gates = gate_results({"frozen": True}, d2, _metrics(), 6, 6, 1, 1)
    assert not gates["Gate_D"]["passed"]


def test_gate_e_requires_recall_and_precision_preservation():
    d2 = _metrics(Overall_Safety_Violation=.05)
    gates = gate_results({"frozen": True}, d2, _metrics(safe_beneficial_episode_recall=.43), 6, 6, 0, 0)
    assert not gates["Gate_E"]["passed"]


def test_gate_f_requires_both_safety_improvement_and_regret_non_degradation():
    d2 = _metrics(Overall_Safety_Violation=.05, Mean_Regret=.2)
    gates = gate_results({"frozen": True}, d2, _metrics(Mean_Regret=.21), 6, 6, 0, 0)
    assert not gates["Gate_F"]["passed"]


def test_script_has_no_test_training_or_gt_replacement_selection():
    source = inspect.getsource(stage)
    assert 'build_development_split("test"' not in source
    assert "torch.optim" not in source and ".backward(" not in source
    assert '"optimizer_steps": 0' in source and '"backward_calls": 0' in source
    parameters = inspect.signature(select_with_generic_risk_coverage).parameters
    assert not any("gt" in name.lower() or "unsafe" in name.lower() for name in parameters)


def test_threshold_and_benefit_threshold_remain_frozen():
    assert stage.HARM_THRESHOLD == 0.10968538373708725
    assert stage.BENEFIT_THRESHOLD == -.02


def test_required_outputs_and_development_label_are_present():
    source = inspect.getsource(stage.main)
    required = ("frozen_contract.json", "d2_reproduction.json", "generic_risk_gate_contract.json",
                "d2_vs_d3_metrics.csv", "branchwise_safety_metrics.csv", "generic_replacement_trace.csv",
                "six_original_unsafe_cases.csv", "risk_transfer_audit.csv", "deceleration_latent_debt_audit.csv",
                "safe_beneficial_preservation.csv", "by_context.csv", "by_motion.csv", "stop_c7_audit.csv",
                "gate_results.json", "summary.json")
    assert all(name in source for name in required)
    assert stage.MECHANISM == "DEVELOPMENT MECHANISM RESULT"


def test_latent_debt_tracks_personalized_and_generic_replacement_paths_separately():
    source = inspect.getsource(stage.main)
    assert "personalized_path_reached" in source
    assert "generic_replacement_path_reached" in source
    assert "generic_repair_triggered_for_episode" in source


def test_context_and_motion_compare_d2_and_d3():
    source = inspect.getsource(stage.main)
    assert source.count('("D2_CONTROL", d2_decisions)') >= 2
    assert source.count('("D3_GENERIC_HARM_COVERAGE", d3_decisions)') >= 2


def test_formal_history_and_checkpoints_are_checksum_guarded():
    source = inspect.getsource(stage.main)
    for field in ("frozen_history_unchanged", "R1_unchanged", "harm_head_unchanged", "harm_threshold_unchanged",
                  "benefit_threshold_unchanged", "generic_cost_unchanged", "arbitration_weights_unchanged"):
        assert field in source


def test_cost_mask_candidate_and_behavior_checksums_are_before_after_guarded():
    source = inspect.getsource(stage.main)
    for field in ("decision_asset_hashes_before", "decision_asset_hashes_after", "decision_assets_unchanged",
                  "prediction_behavior_before", "prediction_behavior_after", "benefit_ranking_behavior_unchanged",
                  "harm_probability_behavior_unchanged", "frozen_arbitrator_source_sha256_before",
                  "frozen_arbitrator_source_sha256_after"):
        assert field in source


def test_gpu_repeatability_audit_requires_tolerance_ranking_and_action_invariance():
    source = inspect.getsource(stage.main)
    assert "prediction_bytewise_identical" in source and "prediction_max_abs_error" in source
    assert "ranking_before == ranking_after" in source
    assert "D2_final_actions_unchanged_on_replay" in source
    assert "D3_final_actions_unchanged_on_replay" in source
