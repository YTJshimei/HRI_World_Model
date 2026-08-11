"""Deterministic random and structured history occlusion for robustness tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.data.skeleton_schema import joint_ids
from src.data.synthetic_skeleton import SkeletonSplit


STRUCTURED_GROUPS = {
    "left_leg": tuple(joint_ids[name] for name in ("left_hip", "left_knee", "left_ankle")),
    "right_leg": tuple(joint_ids[name] for name in ("right_hip", "right_knee", "right_ankle")),
    "left_arm": tuple(joint_ids[name] for name in ("left_shoulder", "left_elbow", "left_wrist")),
    "right_arm": tuple(joint_ids[name] for name in ("right_shoulder", "right_elbow", "right_wrist")),
    "lower_body": tuple(
        joint_ids[name]
        for name in (
            "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
        )
    ),
}


def _with_visibility(split: SkeletonSplit, visibility: np.ndarray) -> SkeletonSplit:
    confidence = split.confidence.copy()
    confidence[~visibility] = 0.0
    return replace(split, confidence=confidence, visibility_mask=visibility)


def apply_random_occlusion(
    split: SkeletonSplit, occlusion_rate: float, seed: int
) -> SkeletonSplit:
    """Apply an absolute deterministic random occlusion rate to the same history."""
    if not 0.0 <= occlusion_rate < 1.0:
        raise ValueError("occlusion_rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    visibility = rng.random(split.visibility_mask.shape) >= occlusion_rate
    # Draw one rate-independent confidence template from the training distribution.
    # Reusing the same seed at every rate isolates visibility rather than confidence shift.
    confidence = rng.uniform(0.80, 1.0, visibility.shape).astype(np.float32)
    confidence[~visibility] = 0.0
    return replace(split, confidence=confidence, visibility_mask=visibility)


def apply_structured_occlusion(
    split: SkeletonSplit,
    group: str,
    consecutive_frames: int,
) -> SkeletonSplit:
    """Mask a body group for the final N observed frames, preserving other masks."""
    if group not in STRUCTURED_GROUPS:
        raise ValueError(f"unknown structured occlusion group: {group}")
    history_frames = split.visibility_mask.shape[1]
    if consecutive_frames <= 0 or consecutive_frames > history_frames:
        raise ValueError("consecutive_frames must be within history length")
    visibility = split.visibility_mask.copy()
    frame_slice = slice(history_frames - consecutive_frames, history_frames)
    visibility[:, frame_slice, list(STRUCTURED_GROUPS[group])] = False
    return _with_visibility(split, visibility)
