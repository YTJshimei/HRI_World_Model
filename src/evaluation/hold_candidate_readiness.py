"""Readiness summaries and preregistered gates for manifest-v3 HOLD support."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def summarize_hold_group(rows: Iterable[dict[str, Any]], group_by: Callable[[dict[str, Any]], Iterable[str]], dimension: str):
    groups = defaultdict(list)
    for row in rows:
        for group in group_by(row):
            groups[str(group)].append(row)
    result = []
    for group in sorted(groups):
        values = groups[group]
        result.append({
            "split": values[0]["split"] if len({row["split"] for row in values}) == 1 else "development",
            "dimension": dimension, "group": group, "candidate_count": len(values),
            "harm_v2_count": sum(row["harm_v2"] for row in values),
            "harm_v2_rate": float(np.mean([row["harm_v2"] for row in values])),
            "GT_unsafe_count": sum(row["GT_unsafe"] for row in values),
            "GT_unsafe_rate": float(np.mean([row["GT_unsafe"] for row in values])),
            "beneficial_count": sum(row["GT_benefit"] > 1e-6 for row in values),
            "beneficial_rate": float(np.mean([row["GT_benefit"] > 1e-6 for row in values])),
            "mean_GT_total_cost": float(np.mean([row["GT_total_cost"] for row in values])),
        })
    return result


def readiness_gates(
    *, action_semantics: bool, rollout_complete: bool, no_label_shortcut: bool,
    train_hold_count: int, validation_hold_count: int,
    expected_train_count: int, expected_validation_count: int,
    motion_coverage: bool, context_coverage: bool, profile_coverage: bool,
    original_no_safe_count: int, original_hold_safe_count: int,
    all_hold_costs_finite: bool,
):
    checks = {
        "Gate_A": {
            "name": "Action Semantics", "checks": {
                "executable_brake_to_zero_then_hold": action_semantics,
                "distinct_from_ABSTAIN_KEEP": action_semantics,
            },
        },
        "Gate_B": {
            "name": "Rollout Completeness", "checks": {
                "robot_human_cost_unsafe_events_harm_benefit_complete": rollout_complete,
            },
        },
        "Gate_C": {
            "name": "No Label Shortcut", "checks": {
                "rollout_derived_labels_and_costs": no_label_shortcut,
            },
        },
        "Gate_D": {
            "name": "Development Support", "checks": {
                "train_HOLD_complete": train_hold_count == expected_train_count,
                "validation_HOLD_complete": validation_hold_count == expected_validation_count,
                "motion_coverage": motion_coverage, "context_coverage": context_coverage,
                "profile_coverage": profile_coverage,
            },
        },
        "Gate_E": {
            "name": "Fallback Utility", "checks": {
                "original_no_safe_count_is_14": original_no_safe_count == 14,
                "multiple_original_episodes_have_GT_safe_HOLD": original_hold_safe_count >= 2,
            },
        },
        "Gate_F": {
            "name": "Evaluability", "checks": {
                "all_HOLD_GT_costs_finite": all_hold_costs_finite,
                "HOLD_selected_regret_would_be_defined": all_hold_costs_finite,
            },
        },
    }
    for gate in checks.values():
        gate["passed"] = all(gate["checks"].values())
    checks["all_passed"] = all(gate["passed"] for gate in checks.values())
    return checks
