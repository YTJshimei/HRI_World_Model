import numpy as np
import pytest

from src.data.skeleton_occlusion import (
    STRUCTURED_GROUPS,
    apply_random_occlusion,
    apply_structured_occlusion,
)
from src.data.synthetic_skeleton import generate_skeleton_split
from src.evaluation.skeleton_decomposition import (
    build_oracle_predictions,
    compose_global,
    decompose_global,
    shapley_root_local_contribution,
)


def test_root_local_decomposition_reconstructs_exactly() -> None:
    values = np.random.default_rng(3).normal(size=(2, 10, 17, 3)).astype(np.float32)
    root, local = decompose_global(values)
    np.testing.assert_allclose(compose_global(root, local), values, atol=1e-6)


def test_all_seven_oracles_have_target_shape() -> None:
    target = np.zeros((2, 10, 17, 3), dtype=np.float32)
    predictions = {name: target + index for index, name in enumerate(("S1", "S2", "S3"), 1)}
    oracles = build_oracle_predictions(predictions, target)
    assert len(oracles) == 7
    assert all(value.shape == target.shape for value in oracles.values())


def test_shapley_contributions_sum_to_full_error() -> None:
    result = shapley_root_local_contribution(0.10, 0.06, 0.05)
    assert result["root_contribution"] + result["local_contribution"] == pytest.approx(0.10)
    assert result["root_fraction"] + result["local_fraction"] == pytest.approx(1.0)


def test_random_occlusion_is_deterministic_and_absolute() -> None:
    split = generate_skeleton_split(8, 9, occlusion_rate=0.1)
    first = apply_random_occlusion(split, 0.3, 123)
    second = apply_random_occlusion(split, 0.3, 123)
    np.testing.assert_array_equal(first.visibility_mask, second.visibility_mask)
    np.testing.assert_array_equal(first.confidence[~first.visibility_mask], 0.0)
    assert np.all(first.confidence[first.visibility_mask] >= 0.80)
    assert np.all(first.confidence[first.visibility_mask] <= 1.0)


@pytest.mark.parametrize("group", sorted(STRUCTURED_GROUPS))
@pytest.mark.parametrize("frames", (3, 5, 10))
def test_structured_occlusion_masks_final_consecutive_frames(group, frames) -> None:
    split = generate_skeleton_split(4, 7, occlusion_rate=0.0)
    masked = apply_structured_occlusion(split, group, frames)
    joints = list(STRUCTURED_GROUPS[group])
    assert not masked.visibility_mask[:, -frames:, joints].any()
    if frames < split.history_global.shape[1]:
        assert masked.visibility_mask[:, : -frames, joints].all()
