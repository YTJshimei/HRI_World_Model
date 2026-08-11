import numpy as np

from src.data.synthetic_trajectory import generate_split
from src.data.synthetic_trajectory_diagnostics import (
    create_labeled_splits,
    dataset_statistics,
    generate_labeled_split,
)


def test_labeled_view_exactly_matches_v1_generator() -> None:
    v1_history, v1_future = generate_split(31, 123)
    labeled = generate_labeled_split(31, 123)
    np.testing.assert_array_equal(labeled.history, v1_history)
    np.testing.assert_array_equal(labeled.future, v1_future)


def test_type_counts_and_variable_horizon() -> None:
    splits = create_labeled_splits(25, 10, 15, seed=42, future_length=30)
    assert splits.test.history.shape == (15, 20, 2)
    assert splits.test.future.shape == (15, 30, 2)
    stats = dataset_statistics(splits)
    assert stats["history_frames"] == 20
    assert stats["future_frames"] == 30
    assert stats["trajectory_type_counts"]["train"] == {
        "straight": 5,
        "acceleration": 5,
        "deceleration": 5,
        "left_turn": 5,
        "right_turn": 5,
    }
