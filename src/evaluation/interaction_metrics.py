"""Metrics for synthetic action-conditioned counterfactual forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.skeleton_schema import compute_root
from src.evaluation.skeleton_metrics import skeleton_metrics


def _mask_or_all(split: Any, action_mask: np.ndarray | None) -> np.ndarray:
    if action_mask is None:
        return np.ones(split.candidate_actions.shape, dtype=bool)
    values = np.asarray(action_mask, dtype=bool)
    if values.shape != split.candidate_actions.shape:
        raise ValueError("action_mask shape mismatch")
    return values


def counterfactual_ranking_per_sample(
    prediction: np.ndarray,
    split: Any,
    action_mask: np.ndarray | None = None,
) -> np.ndarray:
    selected = _mask_or_all(split, action_mask)
    predicted_root = compute_root(prediction)
    predicted_distance = np.linalg.norm(
        predicted_root[..., :2] - split.robot_future_xy_by_action, axis=-1
    )
    expected_distance = split.future_human_robot_distance
    scores = []
    for sample in range(len(split)):
        actions = np.flatnonzero(selected[sample])
        correct, count = 0, 0
        for left_index, left in enumerate(actions):
            for right in actions[left_index + 1 :]:
                expected_difference = expected_distance[sample, left, -1] - expected_distance[sample, right, -1]
                if abs(float(expected_difference)) < 1e-6:
                    continue
                predicted_difference = predicted_distance[sample, left, -1] - predicted_distance[sample, right, -1]
                correct += int(np.sign(predicted_difference) == np.sign(expected_difference))
                count += 1
        scores.append(correct / count if count else float("nan"))
    return np.asarray(scores, dtype=np.float64)


def interaction_metrics(
    prediction: np.ndarray,
    predicted_natural: np.ndarray,
    split: Any,
    sample_rate_hz: float = 10.0,
    action_mask: np.ndarray | None = None,
) -> dict[str, float]:
    predicted = np.asarray(prediction)
    mask = _mask_or_all(split, action_mask)
    target = split.future_by_action
    visibility = np.broadcast_to(
        split.visibility_mask[:, None],
        (len(split), target.shape[1], *split.visibility_mask.shape[1:]),
    )
    base = skeleton_metrics(
        predicted[mask], target[mask], visibility[mask], sample_rate_hz
    )

    predicted_effect = predicted - predicted_natural[:, None]
    expected_effect = split.action_effect_by_action
    nonkeep = mask & (split.candidate_actions != 0)
    if nonkeep.any():
        effect_error = np.linalg.norm(
            predicted_effect[nonkeep] - expected_effect[nonkeep], axis=-1
        ).mean()
        predicted_sensitivity = np.linalg.norm(predicted_effect[nonkeep], axis=-1).mean()
        expected_sensitivity = np.linalg.norm(expected_effect[nonkeep], axis=-1).mean()
    else:
        effect_error = predicted_sensitivity = expected_sensitivity = 0.0

    predicted_root_effect = compute_root(predicted_effect)[..., -1, :2]
    expected_root_effect = compute_root(expected_effect)[..., -1, :2]
    predicted_vectors = predicted_root_effect[nonkeep]
    expected_vectors = expected_root_effect[nonkeep]
    denominator = np.linalg.norm(predicted_vectors, axis=-1) * np.linalg.norm(
        expected_vectors, axis=-1
    )
    valid_direction = denominator > 1e-8
    cosine = np.zeros_like(denominator)
    cosine[valid_direction] = (
        (predicted_vectors[valid_direction] * expected_vectors[valid_direction]).sum(axis=-1)
        / denominator[valid_direction]
    )

    predicted_root = compute_root(predicted)
    predicted_distance = np.linalg.norm(
        predicted_root[..., :2] - split.robot_future_xy_by_action, axis=-1
    )
    distance_error = np.abs(
        predicted_distance[mask] - split.future_human_robot_distance[mask]
    ).mean()
    ranking = counterfactual_ranking_per_sample(predicted, split, mask)
    ranking_accuracy = float(np.nanmean(ranking)) if np.isfinite(ranking).any() else 0.0
    return {
        "Global_MPJPE": float(base["Global_MPJPE"]),
        "Local_MPJPE": float(base["Local_MPJPE"]),
        "Root_ADE": float(base["Root_ADE"]),
        "Root_FDE": float(base["Root_FDE"]),
        "Bone_Length_Error": float(base["Bone_Length_Error"]),
        "Action_Sensitivity": float(predicted_sensitivity),
        "GT_Action_Sensitivity": float(expected_sensitivity),
        "Action_Sensitivity_Ratio": float(
            predicted_sensitivity / max(expected_sensitivity, 1e-12)
        ),
        "Action_Effect_Error": float(effect_error),
        "Action_Direction_Accuracy": float(
            np.mean(cosine[valid_direction] > 0.0) if valid_direction.any() else 0.0
        ),
        "Action_Direction_Cosine": float(
            cosine[valid_direction].mean() if valid_direction.any() else 0.0
        ),
        "Counterfactual_Ranking_Accuracy": ranking_accuracy,
        "Human_Robot_Distance_Error": float(distance_error),
    }
