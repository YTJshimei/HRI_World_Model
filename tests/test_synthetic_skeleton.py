import numpy as np

from src.data.skeleton_schema import skeleton_edges, validate_reconstruction
from src.data.synthetic_skeleton import (
    ACTION_TYPES,
    create_skeleton_splits,
    generate_skeleton_split,
)


def test_skeleton_tensor_shapes_and_reconstruction() -> None:
    split = generate_skeleton_split(18, seed=7, noise_std=0.0, occlusion_rate=0.2)
    assert split.history_global.shape == (18, 20, 17, 3)
    assert split.future_global.shape == (18, 10, 17, 3)
    assert split.root_global.shape == (18, 20, 3)
    assert split.joint_local.shape == (18, 20, 17, 3)
    assert split.confidence.shape == (18, 20, 17)
    assert split.visibility_mask.shape == (18, 20, 17)
    assert split.visibility_mask.dtype == np.bool_
    assert validate_reconstruction(
        split.history_global, split.root_global, split.joint_local
    )


def test_all_schema_bone_lengths_are_constant_over_time() -> None:
    split = generate_skeleton_split(
        len(ACTION_TYPES), seed=11, noise_std=0.0, occlusion_rate=0.0
    )
    sequence = np.concatenate((split.history_global, split.future_global), axis=1)
    for left, right in skeleton_edges:
        lengths = np.linalg.norm(sequence[..., left, :] - sequence[..., right, :], axis=-1)
        assert np.max(np.ptp(lengths, axis=1)) < 1e-5


def test_occlusion_uses_mask_and_confidence_not_zero_coordinates() -> None:
    split = generate_skeleton_split(9, seed=5, noise_std=0.0, occlusion_rate=0.8)
    assert (~split.visibility_mask).any()
    assert np.all(split.confidence[~split.visibility_mask] == 0.0)
    hidden_coordinates = split.history_global[~split.visibility_mask]
    assert not np.all(hidden_coordinates == 0.0)


def test_train_validation_test_are_independent_and_reproducible() -> None:
    first = create_skeleton_splits(18, 9, 9, seed=42, noise_std=0.0)
    second = create_skeleton_splits(18, 9, 9, seed=42, noise_std=0.0)
    np.testing.assert_array_equal(first.train.history_global, second.train.history_global)
    assert not np.array_equal(first.train.history_global[:9], first.test.history_global)
    assert not np.array_equal(first.val.history_global, first.test.history_global)
