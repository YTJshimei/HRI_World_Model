"""Phase 3 metrics for global COCO-17 skeleton predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.skeleton_schema import (
    compute_root,
    lower_limb_joints,
    shoulder_joints,
    skeleton_edges,
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mpjpe(prediction: Any, target: Any, mask: Any | None = None) -> float:
    predicted, expected = _numpy(prediction), _numpy(target)
    if predicted.shape != expected.shape or predicted.shape[-2:] != (17, 3):
        raise ValueError("prediction/target 必须具有相同的 [..., 17, 3] shape")
    errors = np.linalg.norm(predicted - expected, axis=-1)
    if mask is None:
        return float(errors.mean())
    selected = np.broadcast_to(np.asarray(mask, dtype=bool), errors.shape)
    return float(errors[selected].mean()) if selected.any() else float("nan")


def root_aligned_mpjpe(prediction: Any, target: Any, mask: Any | None = None) -> float:
    """MPJPE after independently aligning prediction and GT roots per future frame."""
    predicted, expected = _numpy(prediction), _numpy(target)
    if predicted.shape != expected.shape or predicted.shape[-2:] != (17, 3):
        raise ValueError("prediction/target must have identical [..., 17, 3] shapes")
    predicted_local = predicted - compute_root(predicted)[..., None, :]
    expected_local = expected - compute_root(expected)[..., None, :]
    return mpjpe(predicted_local, expected_local, mask)


def bone_length_error(prediction: Any, target: Any) -> float:
    predicted, expected = _numpy(prediction), _numpy(target)
    errors = []
    for left, right in skeleton_edges:
        predicted_length = np.linalg.norm(
            predicted[..., left, :] - predicted[..., right, :], axis=-1
        )
        target_length = np.linalg.norm(
            expected[..., left, :] - expected[..., right, :], axis=-1
        )
        errors.append(np.abs(predicted_length - target_length))
    return float(np.stack(errors, axis=-1).mean())


def joint_velocity_error(
    prediction: Any, target: Any, sample_rate_hz: float
) -> float:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz 必须大于 0")
    predicted, expected = _numpy(prediction), _numpy(target)
    if predicted.shape[-3] < 2:
        raise ValueError("Joint Velocity Error 至少需要两个 future frame")
    predicted_velocity = np.diff(predicted, axis=-3) * sample_rate_hz
    target_velocity = np.diff(expected, axis=-3) * sample_rate_hz
    return float(np.linalg.norm(predicted_velocity - target_velocity, axis=-1).mean())


def heading_error(prediction: Any, target: Any) -> float:
    """Body yaw error from the left-to-right shoulder axis, in radians."""
    predicted, expected = _numpy(prediction), _numpy(target)
    left, right = shoulder_joints
    predicted_axis = predicted[..., right, :2] - predicted[..., left, :2]
    expected_axis = expected[..., right, :2] - expected[..., left, :2]
    predicted_angle = np.arctan2(predicted_axis[..., 1], predicted_axis[..., 0])
    expected_angle = np.arctan2(expected_axis[..., 1], expected_axis[..., 0])
    difference = np.arctan2(
        np.sin(predicted_angle - expected_angle),
        np.cos(predicted_angle - expected_angle),
    )
    return float(np.abs(difference).mean())


def skeleton_metrics(
    prediction: Any,
    target: Any,
    history_visibility_mask: Any,
    sample_rate_hz: float = 10.0,
) -> dict[str, float]:
    predicted, expected = _numpy(prediction), _numpy(target)
    predicted_root = compute_root(predicted)
    expected_root = compute_root(expected)
    root_errors = np.linalg.norm(predicted_root - expected_root, axis=-1)
    lower_error = np.linalg.norm(
        predicted[..., lower_limb_joints, :] - expected[..., lower_limb_joints, :],
        axis=-1,
    ).mean()
    history_mask = np.asarray(history_visibility_mask, dtype=bool)
    occluded_at_prediction = ~history_mask[:, -1, :]
    occluded_mask = np.broadcast_to(
        occluded_at_prediction[:, None, :], predicted.shape[:-1]
    )
    global_error = mpjpe(predicted, expected)
    local_error = root_aligned_mpjpe(predicted, expected)
    return {
        # MPJPE is retained as the Phase 3A backward-compatible global metric.
        "MPJPE": global_error,
        "Global_MPJPE": global_error,
        "Local_MPJPE": local_error,
        "Root_ADE": float(root_errors.mean()),
        "Root_FDE": float(root_errors[:, -1].mean()),
        "Joint_Velocity_Error": joint_velocity_error(
            predicted, expected, sample_rate_hz
        ),
        "Bone_Length_Error": bone_length_error(predicted, expected),
        "Heading_Error_rad": heading_error(predicted, expected),
        "Lower_Limb_MPJPE": float(lower_error),
        "Occluded_Joint_MPJPE": mpjpe(predicted, expected, occluded_mask),
    }


def metrics_by_action(
    prediction: Any,
    target: Any,
    history_visibility_mask: Any,
    action_types: np.ndarray,
    sample_rate_hz: float = 10.0,
) -> dict[str, dict[str, float]]:
    predicted, expected = _numpy(prediction), _numpy(target)
    visibility = _numpy(history_visibility_mask)
    result = {}
    for action in np.unique(action_types):
        selected = action_types == action
        result[str(action)] = skeleton_metrics(
            predicted[selected], expected[selected], visibility[selected], sample_rate_hz
        )
    return result
