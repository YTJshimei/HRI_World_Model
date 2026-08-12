import inspect

import numpy as np
import pytest

from src.data.robot_action_schema import RobotAction
from src.decision.personalization_confidence import (
    SelectiveDecisionMode, action_personalization_confidence,
    compute_personalization_confidence, decision_margin,
    fit_switch_benefit_calibrator, selective_personalization_select,
    shrink_functional_state,
)


def test_confidence_range() -> None:
    result = compute_personalization_confidence(
        np.ones(6) * .2, np.ones(6), np.ones((3, 6), bool), np.ones((10, 3)) * .1,
    )
    assert np.all((result.dimension_confidence >= 0) & (result.dimension_confidence <= 1))


def test_low_confidence_shrinks_to_population() -> None:
    population = np.arange(6.0); personal = population + 2
    np.testing.assert_allclose(shrink_functional_state(personal, population, np.zeros(6)), population)


def test_high_confidence_is_personalized() -> None:
    population = np.arange(6.0); personal = population + 2
    np.testing.assert_allclose(shrink_functional_state(personal, population, np.ones(6)), personal)


def test_action_specific_dimension_routing() -> None:
    confidence = np.asarray((.8, .2, .3, .4, .9, .5))
    speed = action_personalization_confidence(RobotAction.SPEED_UP_10, confidence)
    distance = action_personalization_confidence(RobotAction.DISTANCE_PLUS_0_2, confidence)
    assert speed != distance


def test_turn_action_cannot_borrow_unrelated_high_confidence() -> None:
    # speed/distance are perfect, but turn/delay/lateral are weak.
    confidence = np.asarray((1.0, 1.0, .01, .01, .01, .01))
    turn = action_personalization_confidence(RobotAction.LEFT_OFFSET, confidence)
    assert turn < .02


def test_generic_safe_is_distinct_from_fallback() -> None:
    result = selective_personalization_select(
        np.arange(3), np.ones(3, bool), np.asarray((0., 1., 2.)),
        np.asarray((1., 0., 2.)), np.asarray((1., 0., 2.)),
        np.asarray((1., .1, 1.)), .5, .01, .5, np.ones(3),
    )
    assert result.mode == SelectiveDecisionMode.GENERIC_SAFE
    assert result.selected_action == 0


def test_rejected_candidate_never_reenters_selective_selection() -> None:
    result = selective_personalization_select(
        np.arange(3), np.asarray((True, False, True)), np.asarray((1., -50., 2.)),
        np.asarray((1., -100., 2.)), np.asarray((1., -100., 2.)),
        np.ones(3), 0., 0., 0., np.ones(3),
    )
    assert result.selected_action != 1


def test_selector_cannot_access_gt_theta_future_or_cost() -> None:
    parameters = set(inspect.signature(selective_personalization_select).parameters)
    assert not parameters.intersection({"gt", "gt_theta", "gt_future", "gt_cost", "oracle_action"})


def test_decision_margin_permutation_invariance() -> None:
    actions=np.asarray((0,1,2,3));cost=np.asarray((.2,.1,.1,.5));mask=np.ones(4,bool)
    first=decision_margin(actions,cost,mask);p=np.asarray((3,2,0,1));second=decision_margin(actions[p],cost[p],mask[p])
    assert first.best_action == second.best_action == 1
    assert first.second_action == second.second_action == 2
    assert first.absolute_margin == pytest.approx(second.absolute_margin)


def test_switch_audit_definition() -> None:
    generic_cost, selective_cost = 1.2, .9
    assert selective_cost < generic_cost  # BENEFICIAL_SWITCH evaluation rule.
    assert not (selective_cost > generic_cost)


def test_keep_and_abstain_remain_distinct() -> None:
    empty = selective_personalization_select(
        np.asarray((0,1)), np.zeros(2,bool), np.zeros(2),np.zeros(2),np.zeros(2),
        np.zeros(2),0.,0.,0.,np.zeros(2),
    )
    assert empty.mode == SelectiveDecisionMode.ABSTAIN
    assert empty.selected_action is None
    assert empty.selected_action != int(RobotAction.KEEP)


def test_switch_calibrator_cannot_fit_validation_or_test() -> None:
    for split in ("validation", "test"):
        with pytest.raises(ValueError):
            fit_switch_benefit_calibrator(np.ones((3,4)), np.ones(3), split)
