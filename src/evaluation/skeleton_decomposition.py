"""Oracle root/local recombination utilities for Phase 3B."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.data.skeleton_schema import compute_root


def decompose_global(skeleton: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(skeleton)
    if values.shape[-2:] != (17, 3):
        raise ValueError("skeleton must have shape [..., 17, 3]")
    root = np.asarray(compute_root(values))
    return root, values - root[..., None, :]


def compose_global(root: Any, local_pose: Any) -> np.ndarray:
    roots, local = np.asarray(root), np.asarray(local_pose)
    if local.shape[-2:] != (17, 3) or roots.shape != local.shape[:-2] + (3,):
        raise ValueError("root/local pose shapes are incompatible")
    return roots[..., None, :] + local


def build_oracle_predictions(
    predictions: Mapping[str, Any], target: Any
) -> dict[str, np.ndarray]:
    """Build O1--O7 exactly from S1/S2/S3 and ground-truth components."""
    required = ("S1", "S2", "S3")
    if any(name not in predictions for name in required):
        raise ValueError("predictions must contain S1, S2, and S3")
    gt_root, gt_local = decompose_global(target)
    components = {name: decompose_global(predictions[name]) for name in required}
    return {
        "O1_GTroot_S1local": compose_global(gt_root, components["S1"][1]),
        "O2_GTroot_S2local": compose_global(gt_root, components["S2"][1]),
        "O3_GTroot_S3local": compose_global(gt_root, components["S3"][1]),
        "O4_S1root_GTlocal": compose_global(components["S1"][0], gt_local),
        "O5_S2root_GTlocal": compose_global(components["S2"][0], gt_local),
        "O6_S3root_GTlocal": compose_global(components["S3"][0], gt_local),
        "O7_S1root_S2local": compose_global(
            components["S1"][0], components["S2"][1]
        ),
    }


def shapley_root_local_contribution(
    full_global_mpjpe: float,
    root_only_global_mpjpe: float,
    local_only_global_mpjpe: float,
) -> dict[str, float]:
    """Split the non-additive interaction equally between root and local errors."""
    interaction = full_global_mpjpe - root_only_global_mpjpe - local_only_global_mpjpe
    root = root_only_global_mpjpe + interaction / 2.0
    local = local_only_global_mpjpe + interaction / 2.0
    denominator = full_global_mpjpe if full_global_mpjpe else 1.0
    return {
        "root_contribution": root,
        "local_contribution": local,
        "interaction": interaction,
        "root_fraction": root / denominator,
        "local_fraction": local / denominator,
    }
