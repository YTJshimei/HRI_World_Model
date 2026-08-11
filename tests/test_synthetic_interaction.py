import numpy as np
import pytest

from src.data.robot_action_schema import (
    PHASE4A_ACTIONS,
    ROBOT_HISTORY_FIELDS,
    RobotAction,
    action_feature,
)
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import (
    PROFILE_BY_ID,
    create_interaction_splits,
    generate_interaction_split,
    simulate_interaction_future,
)


@pytest.fixture(scope="module")
def small_split():
    return generate_interaction_split(10, 17, "fixture", noise_std=0.0, occlusion_rate=0.0)


def test_robot_action_schema_is_structured_and_stable() -> None:
    assert len(ROBOT_HISTORY_FIELDS) == 7
    assert [int(action) for action in PHASE4A_ACTIONS] == [0, 1, 2, 3, 4]
    assert action_feature(RobotAction.SPEED_UP_10).shape == (4,)


def test_counterfactual_sample_shapes_and_keep_is_natural(small_split) -> None:
    assert small_split.human_history.shape == (10, 20, 17, 3)
    assert small_split.robot_history.shape == (10, 20, 7)
    assert small_split.future_by_action.shape == (10, 5, 10, 17, 3)
    assert small_split.action_effect_by_action.shape == (10, 5, 10, 17, 3)
    np.testing.assert_allclose(
        small_split.future_by_action[:, 0], small_split.natural_future, atol=1e-7
    )
    np.testing.assert_allclose(small_split.action_effect_by_action[:, 0], 0.0, atol=1e-7)
    assert float(small_split.robot_history[..., 3].max()) <= 2.5
    assert float(small_split.robot_history[..., 5].min()) >= 1.0
    assert float(small_split.robot_history[..., 5].max()) <= 1.9


def test_same_state_actions_produce_different_delayed_futures(small_split) -> None:
    effects = small_split.action_effect_by_action[0]
    assert np.max(np.abs(effects[1:] - effects[:1])) > 1e-4
    profile = PROFILE_BY_ID[int(small_split.person_profile_id[0])]
    delay = int(np.ceil(profile.response_delay * 10.0 - 1e-9))
    np.testing.assert_allclose(effects[:, :delay], 0.0, atol=1e-7)
    assert np.max(np.abs(effects[1:, delay:])) > 1e-5


def test_profile_changes_response_for_same_state(small_split) -> None:
    kwargs = (
        small_split.human_history[0],
        small_split.natural_future[0],
        small_split.robot_history[0],
        RobotAction.DISTANCE_MINUS_0_2,
    )
    first = simulate_interaction_future(*kwargs, PROFILE_BY_ID[0])
    second = simulate_interaction_future(*kwargs, PROFILE_BY_ID[4])
    assert np.max(np.abs(first.action_effect - second.action_effect)) > 1e-4


def test_residual_response_preserves_natural_bone_lengths(small_split) -> None:
    from src.data.skeleton_schema import skeleton_edges

    future = small_split.future_by_action[0, 4]
    natural = small_split.natural_future[0]
    for left, right in skeleton_edges:
        predicted_length = np.linalg.norm(future[:, left] - future[:, right], axis=-1)
        natural_length = np.linalg.norm(natural[:, left] - natural[:, right], axis=-1)
        np.testing.assert_allclose(predicted_length, natural_length, atol=1e-5)


def test_counterfactual_branches_never_cross_group_splits() -> None:
    splits = create_interaction_splits(12, 6, 6, seed=9, noise_std=0.0)
    groups = [
        splits.train,
        splits.val,
        splits.test_seen_person_seen_context,
        splits.test_unseen_interaction_state,
        splits.test_unseen_person_profile,
        splits.test_unseen_action_context,
    ]
    identifiers = [set(group.initial_state_id.tolist()) for group in groups]
    for index, first in enumerate(identifiers):
        for second in identifiers[index + 1 :]:
            assert first.isdisjoint(second)
    assert not splits.train.action_supervision_mask.all()
    assert set(splits.test_unseen_person_profile.person_profile_id).isdisjoint(
        set(splits.train.person_profile_id)
    )
