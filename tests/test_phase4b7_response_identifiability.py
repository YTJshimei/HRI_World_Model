import inspect

import numpy as np
import pytest

from src.data.functional_response_state import functional_state_from_profile
from src.data.personal_interaction_memory import (
    PersonalInteractionMemory, generate_personal_interaction_corpus,
)
from src.data.response_probe_schema import (
    PROBE_BY_ID, PROBE_CATALOG, probe_state_mask, validate_probe_catalog,
)
from src.data.synthetic_interaction import PROFILE_BY_ID, generate_interaction_split
from src.evaluation.response_identifiability import (
    PROBE_OBSERVABLE_DIM, FunctionalBelief, information_score,
    local_observability_diagnostics, posterior_std, response_jacobian,
    select_probe_without_oracle, simulate_functional_probe,
    uncertainty_specificity_score,
)


@pytest.fixture(scope="module")
def state():
    split = generate_interaction_split(
        2, 4781, "phase4b7_test", profile_ids=(1,),
        noise_std=0.0, occlusion_rate=0.0,
    )
    return (
        split.human_history[0], split.natural_future[0], split.robot_history[0],
        functional_state_from_profile(PROFILE_BY_ID[1]),
    )


def test_probe_schema_is_synthetic_high_level_and_has_no_cmd_vel_sequence() -> None:
    validate_probe_catalog()
    assert len(PROBE_CATALOG) >= 14
    assert all(probe.synthetic_only and probe.high_level_action for probe in PROBE_CATALOG)
    assert not any(hasattr(probe, "cmd_vel") for probe in PROBE_CATALOG)


def test_probe_statistics_and_selection_interfaces_cannot_receive_hidden_profile() -> None:
    assert "profile" not in inspect.signature(simulate_functional_probe).parameters
    signature = inspect.signature(select_probe_without_oracle)
    assert "theta_true" not in signature.parameters
    assert "profile" not in signature.parameters
    assert "person_id" not in signature.parameters


def test_response_jacobian_shape_and_finite_values(state) -> None:
    history, natural, robot, theta = state
    jacobian = response_jacobian(
        history, natural, robot, PROBE_BY_ID["SPEED_UP_10"], theta
    )
    assert jacobian.shape == (PROBE_OBSERVABLE_DIM, 6)
    assert np.isfinite(jacobian).all()


def test_observability_matrix_shape(state) -> None:
    history, natural, robot, theta = state
    jacobians = [
        response_jacobian(history, natural, robot, probe, theta)
        for probe in PROBE_CATALOG[:3]
    ]
    diagnostics = local_observability_diagnostics(jacobians)
    assert diagnostics.information_matrix.shape == (6, 6)
    assert diagnostics.singular_values.shape == (6,)
    assert 0 <= diagnostics.rank <= 6


def test_keep_has_lower_information_than_active_probe(state) -> None:
    history, natural, robot, theta = state
    prior = np.asarray((0.5, 0.5, 0.4, 0.3, 0.5, 1.0))
    keep = response_jacobian(history, natural, robot, PROBE_BY_ID["KEEP"], theta)
    speed = response_jacobian(history, natural, robot, PROBE_BY_ID["SPEED_UP_15"], theta)
    assert information_score(prior, (keep,)) < information_score(prior, (speed,))


def test_speed_probe_is_sensitive_to_speed_gain(state) -> None:
    history, natural, robot, theta = state
    keep = response_jacobian(history, natural, robot, PROBE_BY_ID["KEEP"], theta)
    speed = response_jacobian(history, natural, robot, PROBE_BY_ID["SPEED_UP_15"], theta)
    assert np.linalg.norm(speed[:, 0]) > np.linalg.norm(keep[:, 0]) + 1e-6


def test_distance_probe_is_sensitive_to_distance_and_lateral(state) -> None:
    history, natural, robot, theta = state
    jacobian = response_jacobian(
        history, natural, robot, PROBE_BY_ID["DISTANCE_PLUS_0_3"], theta
    )
    assert np.linalg.norm(jacobian[:, 1]) > 1e-3
    assert np.linalg.norm(jacobian[:, 2]) > 1e-3


def test_turn_probe_is_sensitive_to_turn_gain(state) -> None:
    history, natural, robot, theta = state
    keep = response_jacobian(history, natural, robot, PROBE_BY_ID["KEEP"], theta)
    turn = response_jacobian(history, natural, robot, PROBE_BY_ID["TURN_LEFT_SMALL"], theta)
    assert np.linalg.norm(turn[:, 4]) > np.linalg.norm(keep[:, 4]) + 1e-3


def test_repeated_probe_has_diminishing_uncertainty_return(state) -> None:
    history, natural, robot, theta = state
    prior = np.asarray((0.5, 0.5, 0.4, 0.3, 0.5, 1.0))
    jacobian = response_jacobian(
        history, natural, robot, PROBE_BY_ID["SPEED_UP_15"], theta
    )
    after_one = posterior_std(prior, (jacobian,))
    after_two = posterior_std(prior, (jacobian, jacobian))
    first_gain = float(np.sum(prior - after_one))
    second_gain = float(np.sum(after_one - after_two))
    assert first_gain > 0.0
    assert second_gain < first_gain


def test_support_query_remains_past_only() -> None:
    corpus = generate_personal_interaction_corpus(
        (1,), 1, 12, 10, 9812, "phase4b7_leakage",
        noise_std=0.0, occlusion_rate=0.0,
    )
    query = corpus.records[int(corpus.query_indices[0])]
    support = PersonalInteractionMemory(corpus.records).select_support(query, 5, "recent")
    assert all(item.person_instance_id == query.person_instance_id for item in support)
    assert all(item.timestamp < query.timestamp and item.order_index < query.order_index for item in support)


def test_uncertainty_specificity_metric() -> None:
    before = np.ones(6)
    after = np.asarray((0.2, 0.9, 0.9, 0.3, 0.4, 0.5))
    score = uncertainty_specificity_score(before, after, (True, False, False, True, True, True))
    assert 0.0 <= score["specificity_score"] <= 1.0
    assert score["relevant_reduction"] > score["irrelevant_reduction"]


def test_candidate_probe_permutation_preserves_set_observability(state) -> None:
    history, natural, robot, theta = state
    jacobians = [
        response_jacobian(history, natural, robot, PROBE_BY_ID[name], theta)
        for name in ("SPEED_UP_15", "DISTANCE_PLUS_0_3", "TURN_LEFT_SMALL")
    ]
    first = local_observability_diagnostics(jacobians)
    second = local_observability_diagnostics(jacobians[::-1])
    np.testing.assert_allclose(first.information_matrix, second.information_matrix, atol=1e-9)
    np.testing.assert_allclose(first.singular_values, second.singular_values, atol=1e-9)


def test_probe_state_masks_are_action_specific() -> None:
    assert not any(probe_state_mask(PROBE_BY_ID["KEEP"]))
    speed = probe_state_mask(PROBE_BY_ID["SPEED_DOWN_10"])
    distance = probe_state_mask(PROBE_BY_ID["DISTANCE_PLUS_0_2"])
    turn = probe_state_mask(PROBE_BY_ID["TURN_LEFT_SMALL"])
    assert speed[0] and not speed[1]
    assert distance[1] and distance[2] and not distance[0]
    assert turn[4] and not turn[0] and not turn[1]
