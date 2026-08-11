import numpy as np
import pytest

from src.evaluation.skeleton_metrics import (
    bone_length_error,
    mpjpe,
    root_aligned_mpjpe,
    skeleton_metrics,
)


def test_mpjpe_for_unit_translation() -> None:
    target = np.zeros((2, 10, 17, 3), dtype=np.float32)
    prediction = target.copy()
    prediction[..., 0] = 1.0
    assert mpjpe(prediction, target) == pytest.approx(1.0)


def test_bone_length_error_ignores_rigid_translation() -> None:
    rng = np.random.default_rng(3)
    target = rng.normal(size=(2, 10, 17, 3)).astype(np.float32)
    prediction = target + np.array([4.0, -2.0, 1.0], dtype=np.float32)
    assert bone_length_error(prediction, target) == pytest.approx(0.0, abs=1e-6)


def test_root_aligned_mpjpe_removes_global_translation() -> None:
    target = np.zeros((2, 10, 17, 3), dtype=np.float32)
    prediction = target + np.array([1.0, -2.0, 0.5], dtype=np.float32)
    assert mpjpe(prediction, target) > 0.0
    assert root_aligned_mpjpe(prediction, target) == pytest.approx(0.0, abs=1e-7)


def test_complete_metric_set_contains_occluded_joint_error() -> None:
    target = np.zeros((1, 10, 17, 3), dtype=np.float32)
    prediction = target.copy()
    prediction[:, :, 5, 0] = 0.5
    visibility = np.ones((1, 20, 17), dtype=bool)
    visibility[:, -1, 5] = False
    metrics = skeleton_metrics(prediction, target, visibility, sample_rate_hz=10.0)
    assert metrics["Occluded_Joint_MPJPE"] == pytest.approx(0.5)
    for name in (
        "MPJPE", "Global_MPJPE", "Local_MPJPE", "Root_ADE", "Root_FDE", "Joint_Velocity_Error",
        "Bone_Length_Error", "Heading_Error_rad", "Lower_Limb_MPJPE",
    ):
        assert name in metrics
