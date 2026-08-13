"""Post-hoc safety attribution primitives for frozen Phase 5B-1.7F decisions."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def reaches_harm_gate(row: dict[str, object]) -> bool:
    """The frozen implementation applies harm eligibility after feasibility/benefit threshold."""
    return bool(row["feasible"] and row["benefit_threshold_pass"])


def episode_floor_class(d1_decision, d2_decision, d1_rows, d2_rows) -> dict[str, object]:
    risk_d1 = [row for row in d1_rows if row["harm_v2"]]
    reaching = [row for row in d2_rows if row["harm_v2"] and reaches_harm_gate(row)]
    vetoed = [row for row in reaching if not row["harm_v2_gate_pass"]]
    collateral = [row for row in d2_rows if row["safe_beneficial"] and reaches_harm_gate(row)
                  and not row["harm_v2_gate_pass"]]
    changed = d1_decision.selected_action != d2_decision.selected_action
    if changed and d1_decision.selected_harm_v2:
        category = "F3_DECISION_CHANGING_SAFETY_VETO"
    elif collateral:
        category = "F4_SAFE_BENEFICIAL_COLLATERAL_BLOCK"
    elif vetoed:
        category = "F2_HARM_GATE_VETO_WITHOUT_FINAL_ACTION_CHANGE"
    elif risk_d1:
        category = "F1_RISK_PRESENT_BUT_ALREADY_BLOCKED_UPSTREAM"
    else:
        category = "F0_NO_RISK_OPPORTUNITY"
    return {
        "category": category, "risk_candidate_count": len(risk_d1),
        "risk_reaching_harm_gate_count": len(reaching), "risk_veto_count": len(vetoed),
        "safe_collateral_count": len(collateral), "final_action_changed": bool(changed),
        "d1_action": d1_decision.selected_action, "d2_action": d2_decision.selected_action,
    }


def rejection_reason(row: dict[str, object]) -> str:
    """Assign one exclusive upstream reason to a D1 harm-v2 candidate."""
    if not row["feasible"]:
        return "HARD_FEASIBILITY"
    if not row["benefit_threshold_pass"]:
        return "FAILED_BENEFIT_THRESHOLD"
    if not row["benefit_sign_correct"] and float(row["predicted_benefit"]) < 0:
        return "NEGATIVE_PREDICTED_BENEFIT_SIGN"
    if int(row["benefit_rank"]) != 1:
        return "POOR_RANKING"
    if not row["generic_score_win"]:
        return "GENERIC_DOMINANCE"
    if not row["personalized_selected"]:
        return "OTHER_TIE_OR_FALLBACK"
    return "RISKY_SWITCH"


def cumulative_safe_funnel(rows: Sequence[dict[str, object]], predicate: Callable[[dict[str, object]], bool]):
    base = [row for row in rows if row["safe_beneficial"] and predicate(row)]
    stages = (
        ("GT_safe_beneficial", lambda row: True),
        ("feasible", lambda row: row["feasible"]),
        ("benefit_sign_correct", lambda row: row["feasible"] and row["benefit_sign_correct"]),
        ("benefit_threshold_pass", lambda row: row["feasible"] and row["benefit_sign_correct"] and row["benefit_threshold_pass"]),
        ("rank_eligible_top1", lambda row: row["feasible"] and row["benefit_sign_correct"] and row["benefit_threshold_pass"] and row["benefit_rank"] == 1),
        ("harm_gate_pass", lambda row: row["feasible"] and row["benefit_sign_correct"] and row["benefit_threshold_pass"] and row["benefit_rank"] == 1 and row["harm_v2_gate_pass"]),
        ("generic_score_win", lambda row: row["feasible"] and row["benefit_sign_correct"] and row["benefit_threshold_pass"] and row["benefit_rank"] == 1 and row["harm_v2_gate_pass"] and row["generic_score_win"]),
        ("final_personalized_switch", lambda row: row["personalized_selected"]),
    )
    result = []
    for stage, test in stages:
        selected = [row for row in base if test(row)]
        result.append({"stage": stage, "candidate_count": len(selected),
                       "episode_count": len({row["episode_id"] for row in selected})})
    return result


def primary_funnel_loss(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    losses = []
    for left, right in zip(rows, rows[1:]):
        losses.append((int(left["episode_count"]) - int(right["episode_count"]),
                       int(left["candidate_count"]) - int(right["candidate_count"]),
                       f"{left['stage']} -> {right['stage']}"))
    loss = max(losses, key=lambda item: (item[0], item[1]))
    return {"episode_loss": loss[0], "candidate_loss": loss[1], "transition": loss[2]}


def subgroup_summary(samples, rows, predicate) -> dict[str, object]:
    ids = {sample.sample_id for sample in samples if predicate(sample)}
    selected = [row for row in rows if row["candidate_id"] in ids]
    safe = [row for row in selected if row["safe_beneficial"]]
    risk = [row for row in selected if row["harm_v2"]]
    return {
        "candidate_count": len(selected),
        "episode_count": len({row["episode_id"] for row in selected}),
        "safe_beneficial_candidates": len(safe),
        "safe_beneficial_episodes": len({row["episode_id"] for row in safe}),
        "safe_beneficial_sign_pass": sum(row["benefit_sign_correct"] for row in safe),
        "harm_v2_candidates": len(risk),
        "risk_reaching_harm_gate": sum(reaches_harm_gate(row) for row in risk),
        "harm_gate_veto": sum(reaches_harm_gate(row) and not row["harm_v2_gate_pass"] for row in risk),
        "generic_unsafe_final_exposure": sum(row["selected"] and row["gt_unsafe"] and not row["personalized_selected"] for row in selected),
        "final_gt_unsafe_exposure": sum(row["selected"] and row["gt_unsafe"] for row in selected),
    }

