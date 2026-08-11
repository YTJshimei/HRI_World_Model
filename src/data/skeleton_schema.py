"""Single authoritative COCO-17 skeleton definition for Phase 3."""

from __future__ import annotations

from typing import Any

import numpy as np

joint_names = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
joint_ids = {name: index for index, name in enumerate(joint_names)}

# COCO has no explicit pelvis joint. The root is a virtual midpoint and does not add a joint.
root_joint = "pelvis_midpoint"
root_joint_ids = (joint_ids["left_hip"], joint_ids["right_hip"])

skeleton_edge_names = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)
skeleton_edges = tuple((joint_ids[left], joint_ids[right]) for left, right in skeleton_edge_names)

left_lower_limb_joints = tuple(
    joint_ids[name] for name in ("left_hip", "left_knee", "left_ankle")
)
right_lower_limb_joints = tuple(
    joint_ids[name] for name in ("right_hip", "right_knee", "right_ankle")
)
lower_limb_joints = left_lower_limb_joints + right_lower_limb_joints
shoulder_joints = (joint_ids["left_shoulder"], joint_ids["right_shoulder"])
hip_joints = root_joint_ids

NUM_JOINTS = len(joint_names)
DEFAULT_HISTORY_FRAMES = 20
DEFAULT_FUTURE_FRAMES = 10
DEFAULT_SAMPLE_RATE_HZ = 10.0


def compute_root(global_skeleton: Any) -> Any:
    """Return the midpoint of left/right hips for [..., 17, 3]."""
    return (
        global_skeleton[..., root_joint_ids[0], :]
        + global_skeleton[..., root_joint_ids[1], :]
    ) / 2


def global_to_local(global_skeleton: Any) -> tuple[Any, Any]:
    root = compute_root(global_skeleton)
    return root, global_skeleton - root[..., None, :]


def validate_reconstruction(
    global_skeleton: Any, root_global: Any, joint_local: Any, atol: float = 1e-6
) -> bool:
    reconstructed = root_global[..., None, :] + joint_local
    if hasattr(global_skeleton, "detach"):
        import torch

        return bool(torch.allclose(global_skeleton, reconstructed, atol=atol, rtol=0.0))
    return bool(np.allclose(global_skeleton, reconstructed, atol=atol, rtol=0.0))


def validate_schema() -> None:
    if len(joint_names) != NUM_JOINTS or len(set(joint_names)) != NUM_JOINTS:
        raise ValueError("joint_names 必须包含 17 个唯一名称")
    for left, right in skeleton_edges:
        if left == right or not (0 <= left < NUM_JOINTS and 0 <= right < NUM_JOINTS):
            raise ValueError(f"非法 skeleton edge：{left, right}")


validate_schema()
