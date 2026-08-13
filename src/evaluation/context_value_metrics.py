"""Candidate, switch, and decision metrics for Phase 5 context value models."""
from __future__ import annotations

import numpy as np


def _as_float(values):
    return np.asarray(values, dtype=np.float64)


def pearson(prediction, target):
    prediction, target = _as_float(prediction), _as_float(target)
    if len(prediction) < 2 or np.std(prediction) <= 1e-12 or np.std(target) <= 1e-12:
        return None
    return float(np.corrcoef(prediction, target)[0, 1])


def spearman(prediction, target):
    prediction, target = _as_float(prediction), _as_float(target)
    if len(prediction) < 2 or np.std(prediction) <= 1e-12 or np.std(target) <= 1e-12:
        return None
    pred_rank = np.argsort(np.argsort(prediction, kind="stable"), kind="stable")
    target_rank = np.argsort(np.argsort(target, kind="stable"), kind="stable")
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def binary_auc(probability, truth):
    probability, truth = _as_float(probability), np.asarray(truth, dtype=bool)
    positive, negative = probability[truth], probability[~truth]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def average_precision(probability, truth):
    probability, truth = _as_float(probability), np.asarray(truth, dtype=bool)
    positives = int(truth.sum())
    if not positives:
        return None
    order = np.argsort(-probability, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def candidate_metrics(prediction, targets, feasible=None):
    benefit = _as_float(prediction["benefit"])
    sigma = np.maximum(_as_float(prediction["sigma"]), 1e-6)
    harm_probability = _as_float(prediction["harm"])
    target_benefit = _as_float(targets["benefit"])
    target_harm = np.asarray(targets["harm"], dtype=bool)
    keep = np.ones(len(benefit), dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
    benefit, sigma, harm_probability = benefit[keep], sigma[keep], harm_probability[keep]
    target_benefit, target_harm = target_benefit[keep], target_harm[keep]
    error = benefit - target_benefit
    classified = harm_probability >= 0.5
    true_positive = int(np.sum(classified & target_harm))
    predicted_positive = int(classified.sum())
    actual_positive = int(target_harm.sum())
    nll = 0.5 * (error / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)
    return {
        "Benefit_MAE": float(np.mean(np.abs(error))),
        "Benefit_Pearson": pearson(benefit, target_benefit),
        "Benefit_Spearman": spearman(benefit, target_benefit),
        "Benefit_Sign_Accuracy": float(np.mean(np.sign(benefit) == np.sign(target_benefit))),
        "Harm_AUROC": binary_auc(harm_probability, target_harm),
        "Harm_AUPRC": average_precision(harm_probability, target_harm),
        "Harm_Precision": float(true_positive / max(predicted_positive, 1)),
        "Harm_Recall": float(true_positive / max(actual_positive, 1)),
        "Benefit_Uncertainty_NLL": float(np.mean(nll)),
        "Uncertainty_Error_Correlation": pearson(sigma, np.abs(error)),
        "Candidate_Count": int(len(benefit)),
    }


def switch_metrics(decisions, beneficial_opportunity_count):
    beneficial = sum(bool(row["beneficial_switch"]) for row in decisions)
    harmful = sum(bool(row["harmful_switch"]) for row in decisions)
    personalized = sum(bool(row["personalized"]) for row in decisions)
    switched = [row for row in decisions if bool(row["personalized"])]
    neutral = sum(not bool(row["beneficial_switch"]) and not bool(row["harmful_switch"]) for row in switched)
    no_switch = len(decisions) - personalized
    return {
        "Beneficial_Switch_Recall": float(beneficial / max(beneficial_opportunity_count, 1)),
        "Beneficial_Switch_Precision": float(beneficial / max(personalized, 1)),
        "Beneficial_Switch_Count": int(beneficial),
        "Harmful_Switch_Count": int(harmful),
        "Harmful_Switch_Rate": float(harmful / max(len(decisions), 1)),
        "Neutral_Switch_Count": int(neutral),
        "No_Switch_Count": int(no_switch),
        "Personalized_Decision_Rate": float(personalized / max(len(decisions), 1)),
        "Generic_Safe_Rate": float(sum(row["decision_mode"] == "GENERIC_SAFE" for row in decisions) / max(len(decisions), 1)),
        "ABSTAIN_Rate": float(sum(row["decision_mode"] == "ABSTAIN" for row in decisions) / max(len(decisions), 1)),
    }


def decision_metrics(decisions):
    regret = _as_float([row["Oracle_Regret"] for row in decisions])
    selected = [str(row["selected_action"]) for row in decisions]
    return {
        "GT_Total_Cost": float(np.mean([float(row["GT_Total_Cost"]) for row in decisions])),
        "Mean_Regret": float(np.mean(regret)),
        "Median_Regret": float(np.median(regret)),
        "P90_Regret": float(np.percentile(regret, 90)),
        "P95_Regret": float(np.percentile(regret, 95)),
        "Max_Regret": float(np.max(regret)),
        "Safety_Violation": float(np.mean([bool(row["Safety_Violation"]) for row in decisions])),
        "KEEP_Rate": float(np.mean([value in ("0", "0.0") for value in selected])),
        "Fallback_Rate": float(np.mean([row["decision_mode"] != "PERSONALIZED" for row in decisions])),
    }


def validation_selection_key(metrics, harmful_switch_cap=0.01):
    """Transparent validation-only lexicographic checkpoint criterion."""
    harmful = float(metrics["Harmful_Switch_Rate"])
    return (
        harmful > harmful_switch_cap,
        harmful,
        -float(metrics["Beneficial_Switch_Recall"]),
        float(metrics["Mean_Regret"]),
        float(metrics["Benefit_MAE"]),
        -float(metrics.get("Benefit_Spearman") or -1.0),
        -float(metrics.get("Harm_AUROC") or -1.0),
    )
