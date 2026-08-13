"""Phase 5B-1.7F-A post-hoc mechanism audit tests."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from scripts import audit_phase5b17fa_safety_attribution as audit
from src.evaluation.safety_attribution import (
    cumulative_safe_funnel, episode_floor_class, primary_funnel_loss,
    reaches_harm_gate, rejection_reason,
)


def _decision(action=0, harm=False):
    return SimpleNamespace(selected_action=action, selected_harm_v2=harm)


def _row(*, episode="e", candidate="c", harm=False, safe=False, feasible=True,
         benefit=True, sign=True, rank=1, harm_pass=True, score=True, selected=False):
    return {"episode_id": episode, "candidate_id": candidate, "harm_v2": harm,
            "safe_beneficial": safe, "feasible": feasible, "benefit_threshold_pass": benefit,
            "benefit_sign_correct": sign, "benefit_rank": rank, "harm_v2_gate_pass": harm_pass,
            "generic_score_win": score, "personalized_selected": selected,
            "predicted_benefit": .1, "gt_unsafe": False, "selected": selected}


def test_reaches_harm_gate_requires_frozen_feasibility_and_benefit_gate():
    assert reaches_harm_gate(_row())
    assert not reaches_harm_gate(_row(feasible=False))
    assert not reaches_harm_gate(_row(benefit=False))


def test_floor_categories_f0_to_f3_are_exclusive():
    assert episode_floor_class(_decision(), _decision(), [], [])["category"].startswith("F0")
    blocked = [_row(harm=True, benefit=False)]
    assert episode_floor_class(_decision(), _decision(), blocked, blocked)["category"].startswith("F1")
    vetoed = [_row(harm=True, harm_pass=False)]
    assert episode_floor_class(_decision(), _decision(), vetoed, vetoed)["category"].startswith("F2")
    assert episode_floor_class(_decision(1, True), _decision(0), vetoed, vetoed)["category"].startswith("F3")


def test_safe_collateral_is_f4_when_no_decision_changing_risk_veto():
    safe = [_row(safe=True, harm_pass=False)]
    assert episode_floor_class(_decision(), _decision(), safe, safe)["category"].startswith("F4")


def test_d1_rejection_attribution_is_exclusive_and_pipeline_based():
    assert rejection_reason(_row(feasible=False)) == "HARD_FEASIBILITY"
    assert rejection_reason(_row(benefit=False)) == "FAILED_BENEFIT_THRESHOLD"
    assert rejection_reason(_row(rank=2)) == "POOR_RANKING"
    assert rejection_reason(_row(score=False)) == "GENERIC_DOMINANCE"


def test_c7_stop_funnel_is_cumulative_and_primary_loss_is_reproducible():
    rows = [_row(episode="e1", candidate="a", safe=True),
            _row(episode="e2", candidate="b", safe=True, sign=False)]
    funnel = cumulative_safe_funnel(rows, lambda _: True)
    counts = [row["candidate_count"] for row in funnel]
    assert counts == sorted(counts, reverse=True)
    assert primary_funnel_loss(funnel)["transition"] == "feasible -> benefit_sign_correct"


def test_thresholds_are_frozen_exactly():
    assert audit.HARM_THRESHOLD == 0.10968538373708725
    assert audit.BENEFIT_THRESHOLD == -.02


def test_formal_contract_requires_gate_e_and_phase17f_to_remain_failed():
    source = inspect.getsource(audit.formal_contract)
    assert 'summary["phase5b17f_passed"]' in source
    assert 'gates["Gate_E"]["passed"]' in source
    assert "formal 1.7F FAIL/Gate-E FAIL contract was changed" in source


def test_audit_has_no_training_test_or_policy_mutation_path():
    source = inspect.getsource(audit)
    assert "torch.optim" not in source and ".backward(" not in source
    assert 'build_development_split("test"' not in source
    assert '"optimizer_steps": 0' in source and '"backward_calls": 0' in source and '"test_reads": 0' in source
    assert "arbitrate_large_context(" not in inspect.getsource(audit.main)


def test_generic_counterfactual_is_audit_only_and_never_changes_formal_action():
    source = inspect.getsource(audit.main)
    assert '"audit_only": True' in source
    assert '"formal_action_changed": False' in source
    assert '"audit_counterfactual_changed_formal_action": False' in source


def test_profile_id_is_written_only_as_audit_metadata():
    source = inspect.getsource(audit.main)
    assert "profile_id_audit_only" in source
    assert "profile_id" not in inspect.getsource(audit.f.predict)


def test_all_required_artifacts_are_emitted():
    source = inspect.getsource(audit.main)
    required = (
        "frozen_contract.json", "gate_e_floor_effect.csv", "candidate_level_harm_gate_value.csv",
        "generic_decision_path.json", "generic_unsafe_exposures.csv", "generic_harm_counterfactual.csv",
        "hard_safety_vs_gt_unsafe.csv", "generic_unsafe_root_causes.csv", "c7_safe_beneficial_funnel.csv",
        "stop_safe_beneficial_funnel.csv", "benefit_sign_error_audit.csv", "deceleration_latent_risk.csv",
        "by_harm_subtype.csv", "benefit_risk_tradeoff_attribution.csv", "d1_zero_risky_switch_attribution.csv",
        "decision_safety_coverage_matrix.csv", "by_context_motion.csv", "root_cause_classification.json", "summary.json",
    )
    assert all(name in source for name in required)


def test_formal_17f_files_are_read_only_and_hashed_before_after():
    source = inspect.getsource(audit.main)
    assert "formal_before = formal_contract" in source
    assert "formal_1.7F_hashes_before" in source and "formal_1.7F_hashes_after" in source
    assert "formal_1.7F_unchanged" in source


def test_d1_d2_actions_and_metrics_are_replayed_exactly():
    source = inspect.getsource(audit.main)
    assert "formal D1/D2 replay is not reproducible" in source
    assert "formal_D1_D2_metric_reproduction" in source
    assert "D1_D2_action_replay_reproducible" in source


def test_tradeoff_separates_primary_rejection_from_independent_harm_veto():
    source = inspect.getsource(audit.main)
    assert '"rejection_primary": reason' in source
    assert '"also_rejected_by_harm_gate"' in source


def test_floor_output_includes_zero_count_f3_f4_summary_rows():
    source = inspect.getsource(audit.main)
    assert "F3_DECISION_CHANGING_SAFETY_VETO" in source
    assert "F4_SAFE_BENEFICIAL_COLLATERAL_BLOCK" in source
    assert '"record_type": "category_summary"' in source
