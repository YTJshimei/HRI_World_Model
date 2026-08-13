"""Frozen Phase 5B-1.7F harm-v2 decision-chain evaluation primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.decision.large_context_arbitrator import arbitrate_large_context

BENEFIT_THRESHOLD = -0.02


@dataclass(frozen=True)
class EpisodeDecision:
    episode_id: str
    selected_local: int
    generic_local: int
    personalized: bool
    selected_safe_beneficial: bool
    selected_harm_v2: bool
    selected_gt_unsafe: bool
    selected_gt_unsafe_any: bool
    regret: float
    selected_action: int


def safe_beneficial_mask(samples) -> np.ndarray:
    return np.asarray([sample.targets.benefit > 0 and not sample.split_metadata["harm_v2_evaluation_only"] and sample.targets.feasible for sample in samples], bool)


def tradeoff_mask(samples) -> np.ndarray:
    return np.asarray([sample.targets.benefit > 0 and sample.split_metadata["harm_v2_evaluation_only"] and sample.targets.feasible for sample in samples], bool)


def decide_episode(samples, benefit, risk, threshold: float | None, mode: str) -> tuple[EpisodeDecision, list[dict[str, object]]]:
    if mode not in {"D0", "D1", "D2"}: raise ValueError("mode must be D0, D1 or D2")
    benefit, risk = np.asarray(benefit, float), np.asarray(risk, float)
    if benefit.shape != risk.shape or benefit.shape != (len(samples),): raise ValueError("candidate arrays must align")
    first = samples[0]; meta = first.split_metadata
    actions = np.asarray([sample.split_metadata["candidate_action_id_audit"] for sample in samples], int)
    all_actions = np.asarray(meta["all_action_ids_evaluation_only"], int)
    full = np.asarray([int(np.flatnonzero(all_actions == action)[0]) for action in actions])
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    generic_cost = np.asarray(meta["generic_costs_evaluation_only"], float)[full]
    personalized_cost = np.asarray(meta["personalized_costs_evaluation_only"], float)[full]
    gt_cost = np.asarray(meta["gt_costs_evaluation_only"], float)[full]
    valid = np.flatnonzero(feasible)
    generic = int(valid[np.lexsort((actions[valid], generic_cost[valid]))][0]) if len(valid) else int(np.argmin(gt_cost))
    if mode == "D0":
        selected, personalized = generic, False
    else:
        # D1 disables only the harm-v2 gate. D2 uses strict p<threshold via
        # the frozen arbitrator's inclusive interface at nextafter(threshold,-inf).
        effective_risk = np.zeros_like(risk) if mode == "D1" else risk
        effective_threshold = 1.0 if mode == "D1" else float(np.nextafter(float(threshold), -np.inf))
        result = arbitrate_large_context(actions, feasible, generic_cost, personalized_cost, benefit,
                                         effective_risk, BENEFIT_THRESHOLD, effective_threshold)
        selected = generic if result.selected_index is None else int(result.selected_index)
        personalized = bool(result.personalization_approved and selected != generic)
    selected_sample = samples[selected]
    harm = bool(selected_sample.split_metadata["harm_v2_evaluation_only"])
    unsafe = bool(selected_sample.targets.gt_unsafe)
    safe = bool(selected_sample.targets.benefit > 0 and not harm and selected_sample.targets.feasible)
    decision = EpisodeDecision(first.episode_id, selected, generic, personalized, bool(personalized and safe),
                               bool(personalized and harm), bool(personalized and unsafe), unsafe,
                               float(gt_cost[selected] - gt_cost.min()), int(actions[selected]))
    ranks = np.empty(len(benefit), int); order = np.argsort(-benefit, kind="stable"); ranks[order] = np.arange(1, len(order) + 1)
    adjusted = personalized_cost - np.maximum(benefit, 0.0); generic_score = generic_cost[generic]
    rows = []
    for i, sample in enumerate(samples):
        harm_pass = True if mode in {"D0", "D1"} else bool(risk[i] < threshold)
        rows.append({"episode_id": first.episode_id, "candidate_id": sample.sample_id, "local_index": i,
                     "feasible": bool(feasible[i]), "GT_benefit_positive": bool(sample.targets.benefit > 0),
                     "safe_beneficial": bool(sample.targets.benefit > 0 and not sample.split_metadata["harm_v2_evaluation_only"] and feasible[i]),
                     "harm_v2": bool(sample.split_metadata["harm_v2_evaluation_only"]), "gt_unsafe": bool(sample.targets.gt_unsafe),
                     "benefit_sign_correct": bool((benefit[i] > 0) == (sample.targets.benefit > 0)),
                     "benefit_threshold_pass": bool(benefit[i] >= BENEFIT_THRESHOLD), "benefit_rank": int(ranks[i]),
                     "harm_v2_gate_pass": harm_pass, "generic_score_win": bool(adjusted[i] < generic_score),
                     "selected": bool(i == selected), "personalized_selected": bool(personalized and i == selected),
                     "risk_probability": float(risk[i]), "predicted_benefit": float(benefit[i])})
    return decision, rows


def summarize_decisions(decisions, candidate_rows) -> dict[str, float | int]:
    opportunities = len({row["episode_id"] for row in candidate_rows if row["safe_beneficial"]})
    all_opportunities = len({row["episode_id"] for row in candidate_rows if row["GT_benefit_positive"] and row["feasible"]})
    safe_count = sum(item.selected_safe_beneficial for item in decisions)
    all_beneficial_count = sum(row["personalized_selected"] and row["GT_benefit_positive"] for row in candidate_rows)
    personalized = sum(item.personalized for item in decisions)
    regrets = np.asarray([item.regret for item in decisions], float)
    return {"episode_count": len(decisions), "safe_beneficial_opportunity_episodes": opportunities,
            "safe_beneficial_switch_count": int(safe_count), "safe_beneficial_episode_recall": float(safe_count / max(opportunities, 1)),
            "safe_beneficial_precision": float(safe_count / max(personalized, 1)),
            "all_beneficial_opportunity_episodes_auxiliary": int(all_opportunities),
            "all_beneficial_switch_count_auxiliary": int(all_beneficial_count),
            "all_beneficial_episode_recall_auxiliary": float(all_beneficial_count / max(all_opportunities, 1)),
            "GT_harm_v2_risky_switch_count": int(sum(item.selected_harm_v2 for item in decisions)),
            "GT_unsafe_switch_count": int(sum(item.selected_gt_unsafe for item in decisions)),
            "GT_unsafe_final_selected_count": int(sum(item.selected_gt_unsafe_any for item in decisions)),
            "Mean_Regret": float(regrets.mean()), "P95_Regret": float(np.percentile(regrets, 95)),
            "Safety_Violation": float(np.mean([item.selected_gt_unsafe_any for item in decisions])),
            "generic_selection_rate": float(np.mean([not item.personalized for item in decisions])),
            "personalized_selection_rate": float(np.mean([item.personalized for item in decisions]))}


def threshold_selection_key(metrics: dict[str, float], threshold: float) -> tuple:
    return (int(metrics["GT_unsafe_switch_count"]), int(metrics["GT_harm_v2_risky_switch_count"]),
            -float(metrics["safe_beneficial_episode_recall"]), float(metrics["Mean_Regret"]),
            -float(metrics["safe_beneficial_precision"]), float(threshold))


def gate_results(integrity, d0, d1, d2) -> dict[str, object]:
    a = dict(integrity)
    b = {"GT_unsafe_final_switch_zero": d2["GT_unsafe_switch_count"] == 0,
         "GT_harm_v2_risky_switch_zero": d2["GT_harm_v2_risky_switch_count"] == 0}
    c = {"at_least_one_safe_beneficial_switch": d2["safe_beneficial_switch_count"] >= 1,
         "safe_beneficial_recall_positive": d2["safe_beneficial_episode_recall"] > 0}
    d = {"Mean_Regret_not_worse_than_D0": d2["Mean_Regret"] <= d0["Mean_Regret"] + 1e-12,
         "P95_Regret_not_worse_than_D0": d2["P95_Regret"] <= d0["P95_Regret"] + 1e-12}
    e = {"risk_switch_reduced_vs_D1": (d2["GT_harm_v2_risky_switch_count"] < d1["GT_harm_v2_risky_switch_count"] or
                                        d2["GT_unsafe_switch_count"] < d1["GT_unsafe_switch_count"]),
         "safe_beneficial_recall_not_zero": d2["safe_beneficial_episode_recall"] > 0}
    result = {"Gate_A": {"name": "Threshold Calibration Integrity", "checks": a, "passed": all(a.values())},
              "Gate_B": {"name": "Safety", "checks": b, "passed": all(b.values())},
              "Gate_C": {"name": "Safe Personalization", "checks": c, "passed": all(c.values())},
              "Gate_D": {"name": "Decision Value", "checks": d, "passed": all(d.values())},
              "Gate_E": {"name": "Harm-v2 Gate Value", "checks": e, "passed": all(e.values())}}
    result["all_passed"] = all(item["passed"] for item in result.values()); return result
