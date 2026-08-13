"""Phase 5B-1.7F-C safe-fallback semantics and support audit helpers.

This module deliberately does not implement a robot action.  It only classifies
already existing behavior and computes development-only audit gates.  Ground
truth may be used by the oracle/evaluation helpers, but no runtime selector is
defined here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
MECHANISM = "DEVELOPMENT MECHANISM RESULT"


def classify_abstain_semantics(
    *, selected_index: int | None, selected_action: int | None,
    candidate_rollout_available: bool, candidate_cost_available: bool,
) -> dict[str, Any]:
    """Classify ABSTAIN without assigning invented motion or cost semantics."""
    executable = bool(
        selected_index is not None
        and selected_action is not None
        and candidate_rollout_available
        and candidate_cost_available
    )
    return {
        "selected_index": selected_index,
        "selected_action": selected_action,
        "candidate_rollout_available": bool(candidate_rollout_available),
        "candidate_cost_available": bool(candidate_cost_available),
        "is_executable_robot_action": executable,
        "classification": "EXECUTABLE_ACTION" if executable else "EVALUATION_PLACEHOLDER",
        "can_define_regret": executable,
    }


def oracle_safe_fallback_availability(
    candidate_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Count GT-safe feasible candidates per episode for audit only.

    This function is intentionally named ``oracle`` and is not accepted by any
    runtime decision function.  A GT-safe candidate is feasible and has neither
    the frozen GT-unsafe nor harm-v2 label.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for episode_id, rows in grouped.items():
        safe = [
            row for row in rows
            if bool(row["feasible"])
            and not bool(row["GT_harm_v2"])
            and not bool(row["GT_unsafe"])
        ]
        count = len(safe)
        bucket = "0 candidates" if count == 0 else "1 candidate" if count == 1 else ">1 candidates"
        result.append({
            "episode_id": episode_id,
            "oracle_safe_candidate_count": count,
            "availability_bucket": bucket,
            "oracle_safe_candidate_ids": "|".join(str(row["candidate_id"]) for row in safe),
            "oracle_safe_action_ids": "|".join(str(row["candidate_action_id"]) for row in safe),
            "oracle_safe_actions": "|".join(str(row["action"]) for row in safe),
        })
    counts = Counter(row["availability_bucket"] for row in result)
    summary = {
        "episode_count": len(result),
        "0 candidates": counts["0 candidates"],
        "1 candidate": counts["1 candidate"],
        ">1 candidates": counts[">1 candidates"],
        "episodes_with_at_least_one": len(result) - counts["0 candidates"],
    }
    return result, summary


def runtime_fallback_is_verified(
    registry_entry: Mapping[str, Any], *, feasible: bool,
    predicted_harm: float, harm_threshold: float,
) -> bool:
    """Check an existing fallback using runtime facts only (never GT labels)."""
    required = (
        "exists", "is_executable_robot_action", "protocol_defined_safe_semantics",
        "deterministic_robot_rollout", "human_response_rollout", "GT_cost_available",
    )
    return bool(
        all(bool(registry_entry.get(name, False)) for name in required)
        and feasible
        and np.isfinite(predicted_harm)
        and float(predicted_harm) < float(harm_threshold)
    )


def defined_regret(selected_gt_cost: float | None, oracle_gt_cost: float) -> float:
    """Compute regret only for a real selected rollout with a defined GT cost."""
    if selected_gt_cost is None:
        raise ValueError("fallback must have an actual candidate rollout and GT cost")
    values = np.asarray((selected_gt_cost, oracle_gt_cost), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("regret costs must be finite")
    return float(values[0] - values[1])


def safe_fallback_gates(
    *, semantics_valid: bool, rollout_evaluation_supported: bool,
    original_no_safe_count: int, remaining_undefined_count: int,
    fallback_gt_unsafe_count: int | None, fallback_harm_v2_count: int | None,
    personalized_risky_selected: int, latent_deceleration_selected: int,
    evaluation_episode_count: int, defined_action_count: int,
    defined_gt_cost_count: int, defined_regret_count: int,
) -> dict[str, Any]:
    """Compute the preregistered A--F fallback-design gates."""
    a_checks = {
        "existing_executable_behavior": bool(semantics_valid),
        "protocol_defined_not_post_hoc": bool(semantics_valid),
    }
    b_checks = {
        "robot_and_human_rollout_supported": bool(rollout_evaluation_supported),
        "GT_cost_harm_unsafe_supported": bool(rollout_evaluation_supported),
    }
    c_checks = {
        "original_no_safe_generic_count_is_14": int(original_no_safe_count) == 14,
        "remaining_undefined_count_zero": int(remaining_undefined_count) == 0,
    }
    d_checks = {
        "fallback_defined_for_all_original_14": int(remaining_undefined_count) == 0,
        "fallback_GT_unsafe_zero": fallback_gt_unsafe_count == 0,
        "fallback_harm_v2_zero": fallback_harm_v2_count == 0,
    }
    e_checks = {
        "personalized_risky_selected_zero": int(personalized_risky_selected) == 0,
        "deceleration_latent_debt_selected_zero": int(latent_deceleration_selected) == 0,
    }
    f_checks = {
        "all_final_actions_defined": int(defined_action_count) == int(evaluation_episode_count),
        "all_GT_costs_defined": int(defined_gt_cost_count) == int(evaluation_episode_count),
        "all_regrets_defined": int(defined_regret_count) == int(evaluation_episode_count),
    }
    gates = {
        "Gate_A": {"name": "Semantics Validity", "checks": a_checks, "passed": all(a_checks.values())},
        "Gate_B": {"name": "Rollout/Evaluation Support", "checks": b_checks, "passed": all(b_checks.values())},
        "Gate_C": {"name": "No-Safe-Generic Coverage", "checks": c_checks, "passed": all(c_checks.values())},
        "Gate_D": {"name": "Fallback Safety", "checks": d_checks, "passed": all(d_checks.values())},
        "Gate_E": {"name": "No Risk Transfer", "checks": e_checks, "passed": all(e_checks.values())},
        "Gate_F": {"name": "Decision Evaluability", "checks": f_checks, "passed": all(f_checks.values())},
    }
    gates["all_passed"] = all(gate["passed"] for gate in gates.values())
    return gates
