"""Validation metrics and PHS-v1 selection for independent harm-v2."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from src.evaluation.context_value_metrics import average_precision, binary_auc

PHS_TIE_TOLERANCE = 1e-12


def binary_nll(probability, target) -> float:
    probability = np.clip(np.asarray(probability, np.float64), 1e-12, 1 - 1e-12)
    target = np.asarray(target, np.float64)
    return float(np.mean(-(target * np.log(probability) + (1 - target) * np.log(1 - probability))))


def expected_calibration_error(probability, target, bins: int = 10) -> float:
    probability = np.asarray(probability, np.float64); target = np.asarray(target, bool)
    if bins <= 0: raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1); result = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability < edges[index + 1] if index < bins - 1 else probability <= edges[index + 1])
        if mask.any(): result += float(mask.mean()) * abs(float(probability[mask].mean()) - float(target[mask].mean()))
    return float(result)


def harm_metrics(probability, target) -> dict[str, float | int | None]:
    probability = np.asarray(probability, np.float64); target = np.asarray(target, bool)
    if probability.shape != target.shape or probability.ndim != 1 or not len(target):
        raise ValueError("probability and target must be non-empty rank-1 arrays with equal shape")
    predicted = probability >= .5; tp = int(np.sum(predicted & target)); fp = int(np.sum(predicted & ~target)); fn = int(np.sum(~predicted & target))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {
        "candidate_count": int(len(target)), "positive_count": int(target.sum()), "prevalence": float(target.mean()),
        "AUROC": binary_auc(probability, target), "AUPRC": average_precision(probability, target),
        "NLL": binary_nll(probability, target), "Brier": float(np.mean((probability - target.astype(float)) ** 2)),
        "ECE": expected_calibration_error(probability, target), "Accuracy_at_0_5": float(np.mean(predicted == target)),
        "Precision_at_0_5": float(precision), "Recall_at_0_5": float(recall),
        "F1_at_0_5": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "mean_probability": float(probability.mean()),
        "mean_positive_probability": float(probability[target].mean()) if target.any() else None,
        "mean_negative_probability": float(probability[~target].mean()) if (~target).any() else None,
    }


def prevalence_baseline(train_target, validation_target) -> dict[str, float]:
    train = np.asarray(train_target, bool); validation = np.asarray(validation_target, bool)
    prevalence = float(train.mean()); probability = np.full(len(validation), prevalence, np.float64)
    return {"train_prevalence_probability": prevalence, "validation_prevalence": float(validation.mean()),
            "NLL": binary_nll(probability, validation), "Brier": float(np.mean((probability - validation.astype(float)) ** 2)),
            "AUROC": .5, "AUPRC": float(validation.mean())}


def phs_select(rows: Sequence[Mapping[str, float]]) -> dict:
    """PHS-v1: min NLL, min Brier, max AUROC, earliest epoch."""
    best = None
    required = ("epoch", "NLL", "Brier", "AUROC")
    for original in rows:
        row = dict(original)
        if not all(name in row and math.isfinite(float(row[name])) for name in required):
            raise ValueError("PHS-v1 requires finite epoch/NLL/Brier/AUROC")
        if best is None:
            best = row; continue
        nll_delta = float(row["NLL"]) - float(best["NLL"])
        if nll_delta < -PHS_TIE_TOLERANCE: best = row; continue
        if abs(nll_delta) > PHS_TIE_TOLERANCE: continue
        brier_delta = float(row["Brier"]) - float(best["Brier"])
        if brier_delta < -PHS_TIE_TOLERANCE: best = row; continue
        if abs(brier_delta) > PHS_TIE_TOLERANCE: continue
        auc_delta = float(row["AUROC"]) - float(best["AUROC"])
        if auc_delta > PHS_TIE_TOLERANCE: best = row; continue
        if abs(auc_delta) <= PHS_TIE_TOLERANCE and int(row["epoch"]) < int(best["epoch"]): best = row
    if best is None: raise RuntimeError("PHS-v1 received no validation epochs")
    return best
