"""Statistical audits and validation-only calibration for Phase 4B.5."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.personal_response_metrics import safe_correlation


def profile_response_correlations(
    profile_rows: list[dict[str, float | int | str]],
    parameter_names: tuple[str, ...],
    response_metric_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for parameter in parameter_names:
        x = np.asarray([float(item[parameter]) for item in profile_rows])
        for metric in response_metric_names:
            y = np.asarray([float(item[metric]) for item in profile_rows])
            rows.append({
                "parameter": parameter,
                "response_metric": metric,
                "pearson": safe_correlation(x, y),
                "spearman": safe_correlation(x, y, rank=True),
                "profile_count": len(profile_rows),
            })
    return rows


def distribution_coverage(
    train_values: np.ndarray, test_values: np.ndarray
) -> dict[str, float]:
    train = np.asarray(train_values, dtype=np.float64).ravel()
    test = np.asarray(test_values, dtype=np.float64).ravel()
    train, test = train[np.isfinite(train)], test[np.isfinite(test)]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("coverage audit requires non-empty finite train/test values")
    lower, upper = float(train.min()), float(train.max())
    q05, q95 = np.quantile(train, (0.05, 0.95))
    nearest = np.min(np.abs(test[:, None] - train[None, :]), axis=1)
    return {
        "train_min": lower,
        "train_max": upper,
        "test_min": float(test.min()),
        "test_max": float(test.max()),
        "range_coverage": float(np.mean((test >= lower) & (test <= upper))),
        "quantile_coverage_05_95": float(np.mean((test >= q05) & (test <= q95))),
        "nearest_train_response_distance_mean": float(nearest.mean()),
        "nearest_train_response_distance_p95": float(np.quantile(nearest, 0.95)),
    }


def person_effect_recovery_ratio(
    predicted_effects: np.ndarray, expected_effects: np.ndarray
) -> dict[str, float]:
    predicted = np.asarray(predicted_effects)
    expected = np.asarray(expected_effects)
    if predicted.shape != expected.shape or predicted.shape[0] < 2:
        raise ValueError("need aligned effects for at least two persons")
    predicted_differences, expected_differences = [], []
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_differences.append(
                float(np.linalg.norm(predicted[left] - predicted[right], axis=-1).mean())
            )
            expected_differences.append(
                float(np.linalg.norm(expected[left] - expected[right], axis=-1).mean())
            )
    predicted_mean = float(np.mean(predicted_differences))
    expected_mean = float(np.mean(expected_differences))
    return {
        "predicted_between_person_effect_difference": predicted_mean,
        "gt_between_person_effect_difference": expected_mean,
        "person_effect_recovery_ratio": predicted_mean / max(expected_mean, 1e-12),
    }


def fit_uncertainty_scale(
    errors: np.ndarray,
    sigma: np.ndarray,
    fit_split_role: str,
    minimum: float = 0.10,
    maximum: float = 10.0,
) -> float:
    """Closed-form Gaussian scale fit; only validation data are accepted."""
    if fit_split_role != "validation":
        raise ValueError("uncertainty scale may only be fit on validation data")
    residual = np.asarray(errors, dtype=np.float64)
    base_sigma = np.asarray(sigma, dtype=np.float64)
    if residual.shape != base_sigma.shape:
        raise ValueError("error/sigma shape mismatch")
    standardized_square = np.square(residual / np.clip(base_sigma, 1e-8, None))
    scale = float(np.sqrt(standardized_square.mean()))
    return float(np.clip(scale, minimum, maximum))


def calibrated_uncertainty_metrics(
    errors: np.ndarray, sigma: np.ndarray, scale: float
) -> dict[str, float | None]:
    calibrated = np.asarray(sigma) * float(scale)
    residual = np.asarray(errors)
    nll = 0.5 * (
        np.square(residual / calibrated)
        + 2.0 * np.log(calibrated)
        + np.log(2.0 * np.pi)
    )
    result: dict[str, float | None] = {
        "scale": float(scale),
        "NLL": float(nll.mean()),
    }
    for level, z in ((50, 0.67448975), (80, 1.28155157), (90, 1.64485363)):
        result[f"Coverage_{level}"] = float(np.mean(np.abs(residual) <= z * calibrated))
        result[f"Interval_Width_{level}"] = float(np.mean(2.0 * z * calibrated))
    uncertainty = np.linalg.norm(calibrated, axis=-1).ravel()
    error_magnitude = np.linalg.norm(residual, axis=-1).ravel()
    result["Uncertainty_Error_Correlation"] = safe_correlation(
        uncertainty, error_magnitude
    )
    return result
