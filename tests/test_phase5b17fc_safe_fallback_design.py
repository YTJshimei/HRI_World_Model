import inspect

import numpy as np
import pytest

from src.evaluation.safe_fallback_design import (
    classify_abstain_semantics, defined_regret,
    oracle_safe_fallback_availability, runtime_fallback_is_verified,
    safe_fallback_gates,
)


def _candidate(episode, action, *, safe=True, feasible=True):
    return {
        "episode_id": episode, "candidate_id": f"{episode}:{action}",
        "candidate_action_id": action, "action": f"A{action}", "feasible": feasible,
        "GT_harm_v2": not safe, "GT_unsafe": False,
    }


def test_abstain_is_placeholder_not_keep_or_stop() -> None:
    audit = classify_abstain_semantics(
        selected_index=None, selected_action=None,
        candidate_rollout_available=False, candidate_cost_available=False,
    )
    assert audit["classification"] == "EVALUATION_PLACEHOLDER"
    assert not audit["is_executable_robot_action"]
    assert not audit["can_define_regret"]


def test_runtime_fallback_cannot_use_ground_truth() -> None:
    parameters = set(inspect.signature(runtime_fallback_is_verified).parameters)
    assert not parameters.intersection({"gt", "gt_unsafe", "harm_v2_label", "gt_cost", "oracle"})


def test_runtime_fallback_requires_existing_rollout_cost_and_safe_semantics() -> None:
    complete = {
        "exists": True, "is_executable_robot_action": True,
        "protocol_defined_safe_semantics": True,
        "deterministic_robot_rollout": True, "human_response_rollout": True,
        "GT_cost_available": True,
    }
    assert runtime_fallback_is_verified(complete, feasible=True, predicted_harm=.1, harm_threshold=.2)
    for missing in complete:
        broken = dict(complete); broken[missing] = False
        assert not runtime_fallback_is_verified(broken, feasible=True, predicted_harm=.1, harm_threshold=.2)
    assert not runtime_fallback_is_verified(complete, feasible=True, predicted_harm=.2, harm_threshold=.2)


def test_fallback_regret_rejects_placeholder_cost() -> None:
    with pytest.raises(ValueError, match="actual candidate rollout"):
        defined_regret(None, 1.0)
    assert defined_regret(1.4, 1.1) == pytest.approx(.3)


def test_oracle_availability_buckets_are_correct() -> None:
    rows = [
        _candidate("e0", 0, safe=False),
        _candidate("e1", 0, safe=True), _candidate("e1", 1, safe=False),
        _candidate("e2", 0, safe=True), _candidate("e2", 1, safe=True),
    ]
    per_episode, summary = oracle_safe_fallback_availability(rows)
    assert [row["availability_bucket"] for row in per_episode] == ["0 candidates", "1 candidate", ">1 candidates"]
    assert summary == {"episode_count": 3, "0 candidates": 1, "1 candidate": 1, ">1 candidates": 1, "episodes_with_at_least_one": 2}


def test_oracle_availability_respects_feasibility() -> None:
    _, summary = oracle_safe_fallback_availability([_candidate("e", 0, safe=True, feasible=False)])
    assert summary["0 candidates"] == 1


def test_gate_a_f_failure_when_no_fallback_exists() -> None:
    gates = safe_fallback_gates(
        semantics_valid=False, rollout_evaluation_supported=False,
        original_no_safe_count=14, remaining_undefined_count=14,
        fallback_gt_unsafe_count=None, fallback_harm_v2_count=None,
        personalized_risky_selected=0, latent_deceleration_selected=0,
        evaluation_episode_count=120, defined_action_count=106,
        defined_gt_cost_count=106, defined_regret_count=106,
    )
    assert not gates["Gate_A"]["passed"]
    assert not gates["Gate_B"]["passed"]
    assert not gates["Gate_C"]["passed"]
    assert not gates["Gate_D"]["passed"]
    assert gates["Gate_E"]["passed"]
    assert not gates["Gate_F"]["passed"]
    assert not gates["all_passed"]


def test_gate_a_f_pass_for_complete_existing_fallback() -> None:
    gates = safe_fallback_gates(
        semantics_valid=True, rollout_evaluation_supported=True,
        original_no_safe_count=14, remaining_undefined_count=0,
        fallback_gt_unsafe_count=0, fallback_harm_v2_count=0,
        personalized_risky_selected=0, latent_deceleration_selected=0,
        evaluation_episode_count=120, defined_action_count=120,
        defined_gt_cost_count=120, defined_regret_count=120,
    )
    assert gates["all_passed"]
