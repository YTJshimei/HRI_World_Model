"""CRACS-v1 validation checkpoint selection for Phase 5B-1.7D-B.

The selector is deliberately independent of models, optimizers, losses and
datasets.  It consumes validation metrics only and never reads TEST data.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

MAX_MAE_RATIO = 1.25
MAX_SIGN_DROP = 0.05
SCORE_TIE_TOLERANCE = 1e-12

MAE = "Benefit_MAE"
SIGN = "Benefit_Sign_Accuracy"
SPEARMAN = "mean_feasible_within_episode_spearman"
PAIRWISE = "mean_feasible_pairwise_accuracy"
TOP1 = "gt_best_top1_accuracy"
TOP2 = "gt_best_top2_recall"
BIAS = "global_bias"
EPOCH = "epoch"

REQUIRED_METRICS = (MAE, SIGN, SPEARMAN, PAIRWISE, TOP1, TOP2)


def calibration_limits(reference_mae: float, reference_sign_accuracy: float) -> dict[str, float]:
    """Return the fixed preregistered B1-relative calibration limits."""
    if not math.isfinite(reference_mae) or reference_mae < 0:
        raise ValueError("reference_mae must be finite and non-negative")
    if not math.isfinite(reference_sign_accuracy):
        raise ValueError("reference_sign_accuracy must be finite")
    return {
        "max_mae": float(MAX_MAE_RATIO * reference_mae),
        "min_sign_accuracy": float(reference_sign_accuracy - MAX_SIGN_DROP),
    }


def spearman_score(value: float) -> float:
    """Map a finite Spearman coefficient from [-1, 1] to [0, 1]."""
    value = float(value)
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("Spearman must be finite and in [-1, 1]")
    return (value + 1.0) / 2.0


def ranking_score(metrics: Mapping[str, float]) -> float:
    """Compute the preregistered equal-weight four-component score."""
    components = (
        spearman_score(metrics[SPEARMAN]),
        float(metrics[PAIRWISE]),
        float(metrics[TOP1]),
        float(metrics[TOP2]),
    )
    if not all(math.isfinite(value) for value in components):
        raise ValueError("all RankingScore components must be finite")
    return float(sum(components) / 4.0)


def eligibility(metrics: Mapping[str, float], reference_mae: float, reference_sign_accuracy: float) -> tuple[bool, list[str]]:
    """Apply the fixed calibration and all-metrics-finite eligibility gate."""
    reasons: list[str] = []
    values = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append((name, float(value)))
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        reasons.append("missing_metrics:" + ",".join(missing))
    nonfinite = [name for name, value in values if not math.isfinite(value)]
    if nonfinite:
        reasons.append("nonfinite_metrics:" + ",".join(nonfinite))
    limits = calibration_limits(reference_mae, reference_sign_accuracy)
    if MAE in metrics and math.isfinite(float(metrics[MAE])) and float(metrics[MAE]) > limits["max_mae"]:
        reasons.append("mae_above_limit")
    if SIGN in metrics and math.isfinite(float(metrics[SIGN])) and float(metrics[SIGN]) < limits["min_sign_accuracy"]:
        reasons.append("sign_below_limit")
    return not reasons, reasons


def annotate(metrics: Mapping[str, float], reference_mae: float, reference_sign_accuracy: float) -> dict:
    """Return an audit row without mutating the caller's validation metrics."""
    row = dict(metrics)
    eligible, reasons = eligibility(row, reference_mae, reference_sign_accuracy)
    row["cracs_eligible"] = eligible
    row["cracs_ineligibility_reasons"] = "|".join(reasons)
    row["S_spearman"] = spearman_score(row[SPEARMAN]) if math.isfinite(float(row.get(SPEARMAN, math.nan))) else math.nan
    row["S_pairwise"] = float(row.get(PAIRWISE, math.nan))
    row["S_top1"] = float(row.get(TOP1, math.nan))
    row["S_top2"] = float(row.get(TOP2, math.nan))
    row["RankingScore"] = ranking_score(row) if eligible else math.nan
    return row


def _prefer(candidate: Mapping[str, float], incumbent: Mapping[str, float]) -> bool:
    score_difference = float(candidate["RankingScore"]) - float(incumbent["RankingScore"])
    if score_difference > SCORE_TIE_TOLERANCE:
        return True
    if abs(score_difference) > SCORE_TIE_TOLERANCE:
        return False
    candidate_mae, incumbent_mae = float(candidate[MAE]), float(incumbent[MAE])
    if candidate_mae != incumbent_mae:
        return candidate_mae < incumbent_mae
    candidate_bias = abs(float(candidate.get(BIAS, math.inf)))
    incumbent_bias = abs(float(incumbent.get(BIAS, math.inf)))
    if candidate_bias != incumbent_bias:
        return candidate_bias < incumbent_bias
    return int(candidate[EPOCH]) < int(incumbent[EPOCH])


def select_cracs(rows: Sequence[Mapping[str, float]], reference_mae: float, reference_sign_accuracy: float) -> tuple[dict, list[dict]]:
    """Select one eligible validation epoch using CRACS-v1."""
    audited = [annotate(row, reference_mae, reference_sign_accuracy) for row in rows]
    best = None
    for row in audited:
        if row["cracs_eligible"] and (best is None or _prefer(row, best)):
            best = row
    if best is None:
        raise RuntimeError("CRACS-v1 found no calibration-eligible validation epoch")
    return dict(best), audited
