"""Audit-stage generic harm-v2 coverage with all frozen scores unchanged."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.decision.large_context_arbitrator import arbitrate_large_context


@dataclass(frozen=True)
class GenericCoveredDecision:
    selected_index: int | None
    selected_action: int | None
    generic_index: int | None
    personalized: bool
    abstained: bool
    reason: str


def select_with_generic_risk_coverage(
    action_ids: np.ndarray,
    feasible_action_mask: np.ndarray,
    generic_cost: np.ndarray,
    personalized_cost: np.ndarray,
    benefit_mean: np.ndarray,
    harm_probability: np.ndarray,
    benefit_threshold: float,
    harm_threshold: float,
) -> GenericCoveredDecision:
    """Apply the same strict harm eligibility to generic and personalized candidates.

    This mirrors the frozen arbitration except for ``generic_eligible``. No GT
    target is accepted, no action is synthesized, and all cost formulae remain
    unchanged. Existing ABSTAIN semantics are reused if the eligible generic
    set is empty.
    """
    actions = np.asarray(action_ids, dtype=int)
    feasible = np.asarray(feasible_action_mask, dtype=bool)
    generic_cost = np.asarray(generic_cost, dtype=float)
    personalized_cost = np.asarray(personalized_cost, dtype=float)
    benefit = np.asarray(benefit_mean, dtype=float)
    risk = np.asarray(harm_probability, dtype=float)
    if any(value.shape != actions.shape for value in (feasible, generic_cost, personalized_cost, benefit, risk)):
        raise ValueError("candidate arrays must align")

    feasible_indices = np.flatnonzero(feasible)
    if not len(feasible_indices):
        return GenericCoveredDecision(None, None, None, False, True, "NO_SAFE_GENERIC_CANDIDATE")
    frozen_generic = int(feasible_indices[np.lexsort((actions[feasible_indices], generic_cost[feasible_indices]))][0])

    # Preserve the entire frozen D2 personalized path first. Generic coverage
    # changes only an outcome that D2 would return as its baseline/fallback.
    frozen = arbitrate_large_context(actions, feasible, generic_cost, personalized_cost, benefit, risk,
                                     benefit_threshold, float(np.nextafter(float(harm_threshold), -np.inf)))
    if frozen.personalization_approved and int(frozen.selected_index) != frozen_generic:
        return GenericCoveredDecision(int(frozen.selected_index), int(frozen.selected_action), frozen_generic, True, False,
                                      "FROZEN_D2_PERSONALIZED_PRESERVED")

    generic_eligible = feasible & (risk < float(harm_threshold))
    valid = np.flatnonzero(generic_eligible)
    if not len(valid):
        return GenericCoveredDecision(None, None, None, False, True, "NO_SAFE_GENERIC_CANDIDATE")
    generic = int(valid[np.lexsort((actions[valid], generic_cost[valid]))][0])

    return GenericCoveredDecision(generic, int(actions[generic]), generic, False, False, "GENERIC_RISK_ELIGIBLE_FALLBACK")
