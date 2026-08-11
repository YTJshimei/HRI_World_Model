import numpy as np

from src.data.synthetic_trajectory import FUTURE_LENGTH, HISTORY_LENGTH, create_splits


def test_data_shapes_and_reproducibility() -> None:
    first = create_splits(train_size=30, val_size=10, test_size=8, seed=7)
    second = create_splits(train_size=30, val_size=10, test_size=8, seed=7)
    assert first.train[0].shape == (30, HISTORY_LENGTH, 2)
    assert first.train[1].shape == (30, FUTURE_LENGTH, 2)
    assert first.val[0].shape == (10, HISTORY_LENGTH, 2)
    assert first.test[1].shape == (8, FUTURE_LENGTH, 2)
    assert first.train[0].dtype == np.float32
    np.testing.assert_array_equal(first.train[0], second.train[0])
    assert not np.array_equal(first.train[0][:8], first.test[0])
