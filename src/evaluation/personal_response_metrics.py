"""Phase 4B response-amplitude, human-only ranking and uncertainty metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.skeleton_schema import compute_root
from src.evaluation.interaction_metrics import counterfactual_ranking_per_sample
from src.evaluation.skeleton_metrics import skeleton_metrics


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def safe_correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float | None:
    x, y = np.asarray(left, dtype=np.float64).ravel(), np.asarray(right, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None
    if rank:
        x, y = _rank(x), _rank(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def action_sensitivity_per_sample(effect: np.ndarray, actions: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(effect, axis=-1).mean(axis=(-1, -2))
    nonkeep = actions != 0
    result = np.zeros(effect.shape[0], dtype=np.float64)
    for sample in range(effect.shape[0]):
        selected = magnitude[sample, nonkeep[sample]]
        result[sample] = float(selected.mean()) if len(selected) else 0.0
    return result


def human_response_ranking_per_sample(
    predicted_effect: np.ndarray,
    expected_effect: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Pairwise ranking from human response magnitude only (no robot reward)."""
    predicted_score = np.linalg.norm(predicted_effect, axis=-1).mean(axis=(-1, -2))
    expected_score = np.linalg.norm(expected_effect, axis=-1).mean(axis=(-1, -2))
    scores = []
    for sample in range(len(predicted_effect)):
        candidates = np.flatnonzero(actions[sample] != 0)
        correct = count = 0
        for left_position, left in enumerate(candidates):
            for right in candidates[left_position + 1:]:
                expected_difference = expected_score[sample, left] - expected_score[sample, right]
                if abs(float(expected_difference)) < 1e-10:
                    continue
                predicted_difference = predicted_score[sample, left] - predicted_score[sample, right]
                correct += int(np.sign(expected_difference) == np.sign(predicted_difference))
                count += 1
        scores.append(correct / count if count else 0.0)
    return np.asarray(scores, dtype=np.float64)


def uncertainty_metrics(
    predicted_root: np.ndarray,
    target_root: np.ndarray,
    root_log_std: np.ndarray,
) -> tuple[dict[str, float | None], list[dict[str, float]]]:
    sigma = np.exp(np.clip(root_log_std, -5.0, 1.0))
    error = target_root - predicted_root
    nll = 0.5 * ((error / sigma) ** 2 + 2.0 * np.log(sigma) + np.log(2.0 * np.pi))
    metrics: dict[str, float | None] = {"Root_NLL": float(nll.mean())}
    for level, z in ((50, 0.67448975), (80, 1.28155157), (90, 1.64485363)):
        covered = np.abs(error) <= z * sigma
        metrics[f"Coverage_{level}"] = float(covered.mean())
        metrics[f"Interval_Width_{level}"] = float((2.0 * z * sigma).mean())
    uncertainty = np.linalg.norm(sigma, axis=-1).ravel()
    magnitude_error = np.linalg.norm(error, axis=-1).ravel()
    metrics["Uncertainty_Error_Correlation"] = safe_correlation(uncertainty, magnitude_error)
    order = np.argsort(uncertainty)
    curve = []
    for retained in (0.2, 0.4, 0.6, 0.8, 1.0):
        count = max(1, int(np.ceil(len(order) * retained)))
        indices = order[:count]
        curve.append({
            "retained_fraction": retained,
            "mean_root_error": float(magnitude_error[indices].mean()),
            "mean_uncertainty": float(uncertainty[indices].mean()),
        })
    return metrics, curve


def personal_response_metrics(
    prediction: np.ndarray,
    predicted_natural: np.ndarray,
    root_log_std: np.ndarray,
    action_effect_root_log_std: np.ndarray,
    split: Any,
    indices: np.ndarray,
    sample_rate_hz: float = 10.0,
) -> tuple[dict[str, float | None], dict[str, np.ndarray], list[dict[str, float]]]:
    selected = np.asarray(indices, dtype=np.int64)
    target = split.future_by_action[selected]
    natural = split.natural_future[selected]
    actions = split.candidate_actions[selected]
    visibility = np.broadcast_to(
        split.visibility_mask[selected, None],
        (len(selected), target.shape[1], *split.visibility_mask.shape[1:]),
    )
    base = skeleton_metrics(
        prediction.reshape(-1, *prediction.shape[2:]),
        target.reshape(-1, *target.shape[2:]),
        visibility.reshape(-1, *visibility.shape[2:]),
        sample_rate_hz,
    )
    predicted_effect = prediction - predicted_natural[:, None]
    expected_effect = target - natural[:, None]
    nonkeep = actions != 0
    effect_error_per = np.linalg.norm(predicted_effect - expected_effect, axis=-1).mean(axis=(-1, -2))
    predicted_sensitivity = action_sensitivity_per_sample(predicted_effect, actions)
    gt_sensitivity = action_sensitivity_per_sample(expected_effect, actions)
    ranking = human_response_ranking_per_sample(predicted_effect, expected_effect, actions)
    predicted_root, target_root = compute_root(prediction), compute_root(target)
    root_errors = np.linalg.norm(predicted_root - target_root, axis=-1)
    robot_future = split.robot_future_xy_by_action[selected]
    predicted_distance = np.linalg.norm(predicted_root[..., :2] - robot_future, axis=-1)
    expected_distance = split.future_human_robot_distance[selected]
    uncertainty, curve = uncertainty_metrics(predicted_root, target_root, root_log_std)
    effect_uncertainty, _ = uncertainty_metrics(
        compute_root(predicted_effect)[nonkeep],
        compute_root(expected_effect)[nonkeep],
        action_effect_root_log_std[nonkeep],
    )
    renamed_effect_uncertainty = {
        name.replace("Root_", "Action_Effect_Root_")
        .replace("Coverage_", "Action_Effect_Coverage_")
        .replace("Interval_", "Action_Effect_Interval_")
        .replace("Uncertainty_", "Action_Effect_Uncertainty_"): value
        for name, value in effect_uncertainty.items()
    }
    pred_vectors = compute_root(predicted_effect)[..., -1, :2][nonkeep]
    gt_vectors = compute_root(expected_effect)[..., -1, :2][nonkeep]
    valid = (np.linalg.norm(pred_vectors, axis=-1) * np.linalg.norm(gt_vectors, axis=-1)) > 1e-8
    direction = float(np.mean(np.sum(pred_vectors[valid] * gt_vectors[valid], axis=-1) > 0.0)) if valid.any() else 0.0
    full_ranking = counterfactual_ranking_per_sample(
        prediction, _IndexedSplit(split, selected)
    )
    metrics: dict[str, float | None] = {
        "Global_MPJPE": float(base["Global_MPJPE"]),
        "Local_MPJPE": float(base["Local_MPJPE"]),
        "Root_ADE": float(root_errors.mean()),
        "Root_FDE": float(root_errors[..., -1].mean()),
        "Action_Effect_Error": float(effect_error_per[nonkeep].mean()),
        "Action_Direction_Accuracy": direction,
        "Action_Sensitivity": float(predicted_sensitivity.mean()),
        "GT_Action_Sensitivity": float(gt_sensitivity.mean()),
        "Sensitivity_MAE": float(np.abs(predicted_sensitivity - gt_sensitivity).mean()),
        "Sensitivity_Pearson": safe_correlation(predicted_sensitivity, gt_sensitivity),
        "Sensitivity_Spearman": safe_correlation(predicted_sensitivity, gt_sensitivity, rank=True),
        "Amplitude_Ratio": float(predicted_sensitivity.mean() / max(gt_sensitivity.mean(), 1e-12)),
        "Human_Response_Ranking_Accuracy": float(ranking.mean()),
        "Full_Decision_Ranking_Accuracy": float(full_ranking.mean()),
        "Human_Robot_Distance_Error": float(np.abs(predicted_distance - expected_distance).mean()),
        **uncertainty,
        **renamed_effect_uncertainty,
    }
    per_sample = {
        "predicted_sensitivity": predicted_sensitivity,
        "gt_sensitivity": gt_sensitivity,
        "sensitivity_absolute_error": np.abs(predicted_sensitivity - gt_sensitivity),
        "human_response_ranking": ranking,
        "effect_error": (
            (effect_error_per * nonkeep).sum(axis=1)
            / nonkeep.sum(axis=1).clip(min=1)
        ),
    }
    return metrics, per_sample, curve


class _IndexedSplit:
    def __init__(self, split: Any, indices: np.ndarray) -> None:
        for name in (
            "candidate_actions", "robot_future_xy_by_action",
            "future_human_robot_distance",
        ):
            setattr(self, name, getattr(split, name)[indices])

    def __len__(self) -> int:
        return len(self.candidate_actions)


def oracle_gap(p0: float, p2: float, p3: float) -> float | None:
    denominator = p0 - p3
    if denominator <= 0.0:
        return None
    return float((p0 - p2) / denominator)
