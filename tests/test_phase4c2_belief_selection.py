import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.robot_action_schema import RobotAction
from src.decision.belief_rollout import deterministic_normal_samples
from src.decision.cost_calibration import fit_cost_residual_calibrator
from src.decision.fallback_policy import (
    DecisionMode, constrained_select_with_fallback,
)
from src.decision.root_belief import RootResidualBeliefHead, make_root_belief
from src.decision.safety_gate import chance_constrained_candidate_mask


def test_root_belief_shape_and_finite() -> None:
    head = RootResidualBeliefHead()
    output = head(torch.randn(4, 20, 3), torch.randn(4, 10, 3))
    assert output["residual"].shape == (4, 10, 3)
    belief = make_root_belief(
        np.zeros((10, 3)), output["residual"][0].detach().numpy(),
        np.exp(output["log_sigma"][0].detach().numpy()),
    )
    assert belief.mu_root.shape == belief.sigma_root.shape == (10, 3)
    assert np.isfinite(belief.sigma_root).all()


def test_distributional_samples_are_finite_and_antithetic() -> None:
    values = deterministic_normal_samples(16, 10, seed=42)
    assert values.shape == (16, 10, 3)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[:8], -values[8:], atol=1e-7)


def test_chance_probability_in_range() -> None:
    with pytest.raises(ValueError):
        chance_constrained_candidate_mask(
            np.ones(1, bool), np.ones(1), np.ones(1) * .1,
            np.zeros(1), np.asarray((1.1,)), .8, .1, 1.64,
        )


def test_rejected_candidate_never_reenters_selection() -> None:
    actions = np.arange(5)
    mask = np.asarray((False, True, False, True, False))
    # Rejected actions deliberately receive the lowest costs.
    result = constrained_select_with_fallback(
        actions, mask, np.asarray((-100.0, 2.0, -50.0, 1.0, -30.0)), 1.0, 1.5,
    )
    assert result.selected_index == 3
    assert result.feasible_action_mask[result.selected_index]


def test_selector_and_fallback_use_same_feasible_mask() -> None:
    actions = np.arange(5)
    mask = np.asarray((False, True, False, False, False))
    normal = np.zeros(5, dtype=bool)
    result = constrained_select_with_fallback(actions, mask, np.ones(5), 1.0, 1.5, normal)
    assert result.mode == DecisionMode.RULE_FALLBACK
    assert result.selected_action == int(RobotAction.SPEED_DOWN_10)
    np.testing.assert_array_equal(result.feasible_action_mask, mask)


def test_abstain_is_distinct_from_keep() -> None:
    result = constrained_select_with_fallback(
        np.arange(5), np.zeros(5, bool), np.zeros(5), 1.0, 1.5,
    )
    assert result.mode == DecisionMode.ABSTAIN
    assert result.selected_action is None
    assert result.selected_action != int(RobotAction.KEEP)


def test_rule_safe_must_pass_same_gate_and_unsafe_fallback_not_selected() -> None:
    actions = np.arange(5)
    mask = np.asarray((True, False, False, False, False))
    result = constrained_select_with_fallback(
        actions, mask, np.zeros(5), 1.0, 1.5, np.zeros(5, bool),
    )
    assert result.selected_action == int(RobotAction.KEEP)
    assert result.selected_action != int(RobotAction.DISTANCE_PLUS_0_2)


def test_empty_feasible_set_triggers_abstain() -> None:
    result = constrained_select_with_fallback(
        np.arange(5), np.zeros(5, bool), np.zeros(5), 2.0, 1.5,
    )
    assert result.mode == DecisionMode.ABSTAIN


def test_candidate_permutation_invariance() -> None:
    actions = np.asarray((0, 1, 2, 3, 4))
    mask = np.asarray((True, False, True, True, False))
    costs = np.asarray((.4, .1, .2, .3, .0))
    first = constrained_select_with_fallback(actions, mask, costs, 1.5, 1.5)
    permutation = np.asarray((4, 2, 0, 3, 1))
    second = constrained_select_with_fallback(
        actions[permutation], mask[permutation], costs[permutation], 1.5, 1.5,
    )
    assert first.selected_action == second.selected_action == 2


def test_cost_calibrator_cannot_fit_validation_or_test_labels() -> None:
    x = np.ones((5, 3)); components = np.ones((5, 4))
    for split in ("validation", "test"):
        with pytest.raises(ValueError):
            fit_cost_residual_calibrator(x, components, components, split)


def test_selector_interfaces_do_not_accept_ground_truth() -> None:
    signature = inspect.signature(constrained_select_with_fallback)
    forbidden = {"gt", "oracle", "label", "target_cost", "test_labels"}
    assert not forbidden.intersection(signature.parameters)


def test_s10_keep_can_remain_valid_optimum() -> None:
    result = constrained_select_with_fallback(
        np.arange(5), np.ones(5, bool), np.asarray((0.0, .2, .3, .4, .5)), 1.72, 1.45,
    )
    assert result.mode == DecisionMode.NORMAL
    assert result.selected_action == int(RobotAction.KEEP)
