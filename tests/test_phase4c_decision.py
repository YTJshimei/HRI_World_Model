import inspect
from dataclasses import replace

import numpy as np
import pytest

from src.data.robot_action_schema import RobotAction
from src.decision.action_selector import (
    decision_regret, rule_based_select, select_model_action, select_oracle_action,
)
from src.decision.candidate_action import (
    CandidateAction, TASK_SAFE_CANDIDATES, validate_candidate_actions,
)
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_cost import (
    DecisionCostWeights, compute_decision_costs, verify_cost_sum,
)
from src.decision.decision_state import DecisionState, FunctionalResponseBelief
from src.decision.safety_gate import apply_safety_gate


def make_state(candidates=TASK_SAFE_CANDIDATES):
    history = np.zeros((20, 17, 3), dtype=np.float32)
    history[:, :, 0] = np.linspace(0.0, 0.5, 20)[:, None]
    robot = np.zeros((20, 7), dtype=np.float32)
    robot[:, 0] = history[:, 0, 0] - 1.5
    robot[:, 3] = 0.25; robot[:, 5] = 1.5
    confidence = np.ones((20, 17), dtype=np.float32)
    visibility = np.ones((20, 17), dtype=bool)
    belief = FunctionalResponseBelief(
        np.asarray((0.6, 0.8, 0.5, 0.3, 0.4, 2.0), dtype=np.float32),
        np.asarray((0.2, 0.2, 0.2, 0.1, 0.2, 0.8), dtype=np.float32),
    )
    return DecisionState(history, robot, confidence, visibility, belief, tuple(candidates))


def make_rollout(distances=None, uncertainty=0.01):
    actions = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    natural = np.zeros((10, 17, 3), dtype=np.float32)
    effect = np.zeros((5, 10, 17, 3), dtype=np.float32)
    effect[1:, :, :, 0] = np.asarray((0.01, 0.02, -0.01, 0.03))[:, None, None]
    global_future = natural[None] + effect
    root = global_future[:, :, 11:13].mean(axis=2)
    local = global_future - root[:, :, None]
    if distances is None:
        distances = np.asarray([
            np.full(10, 1.50), np.full(10, 1.55), np.full(10, 1.45),
            np.full(10, 1.70), np.full(10, 1.30),
        ])
    robot_xy = np.zeros((5, 10, 2), dtype=np.float32)
    return CounterfactualRollout(
        actions, natural, root, local, global_future, robot_xy,
        np.asarray(distances, dtype=np.float32), effect,
        np.full_like(effect, uncertainty), 1,
    )


def test_candidate_action_schema_separates_task_actions_and_probes() -> None:
    assert len(validate_candidate_actions(TASK_SAFE_CANDIDATES)) == 5
    probe = CandidateAction(RobotAction.LEFT_OFFSET, task_action=False, identification_probe=True)
    with pytest.raises(ValueError):
        validate_candidate_actions((*TASK_SAFE_CANDIDATES, probe))


def test_invalid_candidate_is_rejected() -> None:
    candidates = list(TASK_SAFE_CANDIDATES)
    candidates[2] = replace(candidates[2], feasible=False)
    gate = apply_safety_gate(make_state(candidates), make_rollout())
    assert not gate.allowed_mask[2]
    assert gate.rejection_reasons[2] == "candidate_marked_infeasible"


def test_safety_hard_veto() -> None:
    distances = np.full((5, 10), 1.5); distances[4] = 0.55
    gate = apply_safety_gate(make_state(), make_rollout(distances))
    assert not gate.allowed_mask[4]
    assert "hard_veto" in gate.rejection_reasons[4]


def test_gt_future_cannot_enter_model_selector() -> None:
    signature = inspect.signature(select_model_action)
    assert "gt_future" not in signature.parameters
    assert "gt_theta" not in signature.parameters
    assert "oracle_theta" not in signature.parameters


def test_d0_does_not_accept_or_call_world_model() -> None:
    assert tuple(inspect.signature(rule_based_select).parameters) == ("state",)
    assert rule_based_select(make_state()) == int(RobotAction.KEEP)


def test_d1_d2_interfaces_do_not_accept_personal_or_gt_theta() -> None:
    parameters = inspect.signature(select_model_action).parameters
    assert "personal_theta" not in parameters
    assert "gt_theta" not in parameters


def test_d4_is_only_selector_with_oracle_theta() -> None:
    assert "oracle_theta" in inspect.signature(select_oracle_action).parameters
    result = select_oracle_action(
        make_state(), make_rollout(), np.ones(6, dtype=np.float32)
    )
    assert result.selected_action in range(5)


def test_uncertainty_penalty_increases_uncertain_action_cost() -> None:
    state, rollout = make_state(), make_rollout(uncertainty=0.05)
    with_uncertainty = compute_decision_costs(state, rollout, include_uncertainty=True)
    without = compute_decision_costs(state, rollout, include_uncertainty=False)
    assert np.all(with_uncertainty.total >= without.total)
    assert np.any(with_uncertainty.total > without.total)


def test_fallback_logic_when_all_candidates_predicted_unsafe() -> None:
    distances = np.full((5, 10), 0.50)
    gate = apply_safety_gate(make_state(), make_rollout(distances))
    assert gate.abstained
    assert gate.fallback_action in (
        int(RobotAction.KEEP), int(RobotAction.SPEED_DOWN_10),
        int(RobotAction.DISTANCE_PLUS_0_2),
    )
    assert gate.allowed_mask.sum() == 1


def test_candidate_permutation_invariance() -> None:
    state, rollout = make_state(), make_rollout()
    selected = select_model_action(state, rollout).selected_action
    permutation = np.asarray((4, 2, 0, 3, 1))
    candidates = tuple(state.candidates[index] for index in permutation)
    permuted_state = replace(state, candidates=candidates)
    permuted = CounterfactualRollout(
        rollout.action_ids[permutation], rollout.natural_future,
        rollout.predicted_root[permutation], rollout.predicted_local[permutation],
        rollout.predicted_global[permutation], rollout.predicted_robot_xy[permutation],
        rollout.predicted_human_robot_distance[permutation],
        rollout.predicted_action_effect[permutation],
        rollout.prediction_uncertainty[permutation], 1,
    )
    assert select_model_action(permuted_state, permuted).selected_action == selected


def test_cost_components_sum_exactly() -> None:
    weights = DecisionCostWeights()
    costs = compute_decision_costs(make_state(), make_rollout(), weights)
    verify_cost_sum(costs, weights)


def test_regret_calculation() -> None:
    assert decision_regret(1.4, 0.9) == pytest.approx(0.5)


def test_keep_is_not_hardcoded_optimal() -> None:
    state = replace(make_state(), target_follow_distance=1.30)
    distances = np.asarray([
        np.full(10, 1.50), np.full(10, 1.48), np.full(10, 1.45),
        np.full(10, 1.70), np.full(10, 1.30),
    ])
    result = select_model_action(
        state, make_rollout(distances),
        DecisionCostWeights(human_response=0.0, disturbance=0.0, uncertainty=0.0),
    )
    assert result.selected_action == int(RobotAction.DISTANCE_MINUS_0_2)
