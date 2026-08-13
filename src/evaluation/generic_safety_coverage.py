"""Metrics for the Phase 5B-1.7F-B generic coverage mechanism comparison."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoveredEpisodeDecision:
    episode_id: str
    selected_local: int | None
    generic_local: int | None
    personalized: bool
    abstained: bool
    selected_action: int | None
    selected_harm_v2: bool
    selected_gt_unsafe: bool
    selected_safe_beneficial: bool
    regret: float | None
    reason: str


def summarize_covered(decisions, candidate_rows) -> dict[str, float | int]:
    opportunity = len({row["episode_id"] for row in candidate_rows if row["safe_beneficial"]})
    selected = [item for item in decisions if not item.abstained]
    personalized = [item for item in selected if item.personalized]
    safe = sum(item.selected_safe_beneficial for item in personalized)
    regrets = np.asarray([item.regret for item in selected], dtype=float)
    return {
        "episode_count": len(decisions), "selected_episode_count": len(selected),
        "no_safe_generic_count": sum(item.abstained for item in decisions),
        "safe_beneficial_opportunity_episodes": opportunity,
        "safe_beneficial_switch_count": int(safe),
        "safe_beneficial_episode_recall": float(safe / max(opportunity, 1)),
        "safe_beneficial_precision": float(safe / max(len(personalized), 1)),
        "personalized_GT_unsafe_count": sum(item.personalized and item.selected_gt_unsafe for item in selected),
        "personalized_harm_v2_count": sum(item.personalized and item.selected_harm_v2 for item in selected),
        "generic_GT_unsafe_count": sum(not item.personalized and item.selected_gt_unsafe for item in selected),
        "generic_harm_v2_count": sum(not item.personalized and item.selected_harm_v2 for item in selected),
        "total_GT_unsafe_final_count": sum(item.selected_gt_unsafe for item in selected),
        "total_harm_v2_final_count": sum(item.selected_harm_v2 for item in selected),
        "Overall_Safety_Violation": float(np.mean([item.selected_gt_unsafe for item in decisions])),
        "Mean_Regret": float(regrets.mean()) if len(regrets) == len(decisions) else None,
        "P95_Regret": float(np.percentile(regrets, 95)) if len(regrets) == len(decisions) else None,
        "generic_selection_rate": float(np.mean([not item.personalized and not item.abstained for item in decisions])),
        "personalized_selection_rate": float(np.mean([item.personalized for item in decisions])),
        "abstain_rate": float(np.mean([item.abstained for item in decisions])),
    }


def branchwise_rows(system: str, metrics: dict[str, object]):
    return [
        {"system": system, "branch": "Personalized", "GT_unsafe_final": metrics["personalized_GT_unsafe_count"],
         "harm_v2_risky_final": metrics["personalized_harm_v2_count"]},
        {"system": system, "branch": "Generic", "GT_unsafe_final": metrics["generic_GT_unsafe_count"],
         "harm_v2_risky_final": metrics["generic_harm_v2_count"]},
        {"system": system, "branch": "Total", "GT_unsafe_final": metrics["total_GT_unsafe_final_count"],
         "harm_v2_risky_final": metrics["total_harm_v2_final_count"]},
    ]


def gate_results(isolation, d2, d3, original_unsafe_blocked: int, original_unsafe_total: int,
                 risk_transfer_count: int, latent_selected: int):
    a = dict(isolation)
    b = {"all_original_generic_unsafe_blocked": original_unsafe_blocked == original_unsafe_total == 6}
    c = {"D3_total_GT_unsafe_zero": d3["total_GT_unsafe_final_count"] == 0,
         "D3_total_harm_v2_zero": d3["total_harm_v2_final_count"] == 0,
         "no_safe_generic_zero": d3["no_safe_generic_count"] == 0}
    d = {"personalized_harm_v2_zero": d3["personalized_harm_v2_count"] == 0,
         "risk_transfer_zero": risk_transfer_count == 0,
         "deceleration_latent_selected_zero": latent_selected == 0}
    e = {"safe_recall_preserved": d3["safe_beneficial_episode_recall"] >= d2["safe_beneficial_episode_recall"] - 1e-12,
         "precision_not_decreased": d3["safe_beneficial_precision"] >= d2["safe_beneficial_precision"] - 1e-12}
    d3_mean, d3_p95 = d3["Mean_Regret"], d3["P95_Regret"]
    f = {"overall_safety_violation_decreased": d3["Overall_Safety_Violation"] < d2["Overall_Safety_Violation"],
         "Mean_Regret_not_worse": d3_mean is not None and d3_mean <= d2["Mean_Regret"] + 1e-12,
         "P95_Regret_not_worse": d3_p95 is not None and d3_p95 <= d2["P95_Regret"] + 1e-12}
    result = {"Gate_A": {"name": "Isolation", "checks": a, "passed": all(a.values())},
              "Gate_B": {"name": "Generic Coverage Mechanism", "checks": b, "passed": all(b.values())},
              "Gate_C": {"name": "Total Safety", "checks": c, "passed": all(c.values())},
              "Gate_D": {"name": "No Risk Transfer", "checks": d, "passed": all(d.values())},
              "Gate_E": {"name": "Safe Personalization Preservation", "checks": e, "passed": all(e.values())},
              "Gate_F": {"name": "Decision Value", "checks": f, "passed": all(f.values())}}
    result["all_passed"] = all(item["passed"] for item in result.values())
    return result
