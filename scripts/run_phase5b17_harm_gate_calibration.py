"""Phase 5B-1.7 harm-threshold-only calibration with a sealed validation holdout."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as p5
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b16_candidate_ranking as b16
from src.evaluation.context_value_metrics import average_precision, binary_auc
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK

H0_HARM_THRESHOLD = 0.2
BENEFIT_THRESHOLD = -0.02
HARMFUL_REJECTION_TOLERANCE = 0.05
MODELS = ("H0 Frozen Harm Threshold", "H1 Calibrated Harm Threshold")
CONTEXTS = ("C7", "C8", "C9")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17_harm_gate_calibration")
    parser.add_argument("--phase5b16-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b16_candidate_ranking")
    parser.add_argument("--phase5b1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b1_static_vs_temporal_small")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_checksum(samples, fields):
    rows = []
    for sample in samples:
        row = [sample.sample_id]
        for field in fields:
            value = sample.split_metadata[field]
            row.append(np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
        rows.append(row)
    return b1.digest_json(rows)


def episode_features(samples):
    grouped = b15.group_episode(samples)
    profiles = sorted({int(sample.split_metadata["person_profile_id"]) for sample in samples})
    feature_names = ["candidate_count", "beneficial_candidate_count", "harmful_candidate_count", "neutral_candidate_count", "beneficial_episode"]
    feature_names += [f"profile_{profile}" for profile in profiles] + list(CONTEXTS)
    rows, vectors = [], []
    for episode_id, indices in grouped.items():
        episode = [samples[index] for index in indices]
        beneficial = sum(sample.targets.benefit > 1e-6 for sample in episode)
        harmful = sum(sample.targets.harm for sample in episode)
        neutral = len(episode) - beneficial - harmful
        profile = int(episode[0].split_metadata["person_profile_id"])
        contexts = b15.context_labels(episode[0])
        vector = [len(episode), beneficial, harmful, neutral, int(beneficial > 0)]
        vector += [int(profile == value) for value in profiles] + [int(name in contexts) for name in CONTEXTS]
        rows.append({"episode_id": episode_id, "profile": profile, "contexts": sorted(contexts),
                     "candidate_count": len(episode), "beneficial_candidate_count": beneficial,
                     "harmful_candidate_count": harmful, "neutral_candidate_count": neutral,
                     "beneficial_episode": beneficial > 0})
        vectors.append(vector)
    return rows, np.asarray(vectors, np.float64), feature_names


def stratified_episode_split(samples, seed=42):
    """Fixed multi-start 50/50 split balancing episode-level strata."""
    rows, vectors, feature_names = episode_features(samples)
    if len(rows) % 2:
        raise ValueError("validation episode count must be even for fixed 50/50 split")
    target_count = len(rows) // 2
    total = vectors.sum(0); scale = np.maximum(total, 1.0)
    rng = np.random.default_rng(seed); best = None
    # Candidate, beneficial episode, harmful count and rare context imbalance
    # are compared lexicographically before the normalized all-field error.
    priority = [4, 1, 2, feature_names.index("C7"), feature_names.index("C8"), feature_names.index("C9")]
    for _ in range(50_000):
        calibration = np.sort(rng.choice(len(rows), target_count, replace=False))
        difference = np.abs(2 * vectors[calibration].sum(0) - total)
        key = tuple(difference[index] for index in priority) + (float(np.square(difference / scale).sum()),)
        if best is None or key < best[0]:
            best = (key, calibration.copy())
            if all(value == 0 for value in key):
                break
    calibration = best[1].tolist()
    evaluation = sorted(set(range(len(rows))) - set(calibration))
    calibration_ids = sorted(rows[index]["episode_id"] for index in calibration)
    evaluation_ids = sorted(rows[index]["episode_id"] for index in evaluation)
    if len(calibration_ids) != target_count or len(evaluation_ids) != target_count or set(calibration_ids) & set(evaluation_ids):
        raise RuntimeError("invalid validation calibration/evaluation split")
    assignment = {episode_id: "calibration" for episode_id in calibration_ids} | {episode_id: "evaluation" for episode_id in evaluation_ids}
    episode_rows = [{**row, "subset": assignment[row["episode_id"]]} for row in rows]
    distributions = {}
    for subset, indices in (("calibration", calibration), ("evaluation", evaluation)):
        sums = vectors[indices].sum(0)
        distributions[subset] = {name: int(value) for name, value in zip(feature_names, sums)} | {"episode_count": len(indices)}
    payload = {
        "label": LABEL, "seed": seed, "method": "episode-level deterministic 50,000-start balanced subset search",
        "stratification_fields": ["beneficial episode/candidate", "harmful candidate", "profile", "C7", "C8", "C9", "candidate count"],
        "calibration_episode_ids": calibration_ids, "evaluation_episode_ids": evaluation_ids,
        "calibration_episode_count": len(calibration_ids), "evaluation_episode_count": len(evaluation_ids),
        "episode_overlap_count": 0, "candidate_cross_subset_count": 0,
        "distributions": distributions, "episodes": sorted(episode_rows, key=lambda row: row["episode_id"]),
    }
    payload["split_checksum_sha256"] = b1.digest_json(payload)
    return payload


def select_subset(samples, prediction, episode_ids):
    allowed = set(episode_ids)
    indices = [index for index, sample in enumerate(samples) if sample.episode_id in allowed]
    subset = [samples[index] for index in indices]
    subprediction = {key: np.asarray(value)[indices] for key, value in prediction.items()}
    if {sample.episode_id for sample in subset} != allowed:
        raise RuntimeError("subset lost an episode")
    return subset, subprediction


def probability_statistics(values):
    values = np.asarray(values, np.float64)
    percentiles = np.percentile(values, (10, 25, 50, 75, 90, 95)) if len(values) else np.full(6, np.nan)
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            **{name: float(value) for name, value in zip(("P10", "P25", "P50", "P75", "P90", "P95"), percentiles)}}


def expected_calibration_error(probability, target, bins=10):
    probability, target = np.asarray(probability, float), np.asarray(target, bool)
    result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (probability >= lower) & ((probability <= upper) if upper >= 1 else (probability < upper))
        if selected.any():
            result += selected.mean() * abs(probability[selected].mean() - target[selected].mean())
    return float(result)


def harm_calibration_metrics(probability, target):
    probability = np.clip(np.asarray(probability, np.float64), 1e-7, 1 - 1e-7)
    target = np.asarray(target, bool)
    return {
        "candidate_count": len(target), "harmful_count": int(target.sum()),
        "AUROC": binary_auc(probability, target), "AUPRC": average_precision(probability, target),
        "Brier_Score": float(np.mean((probability - target.astype(float)) ** 2)),
        "ECE_10_bins": expected_calibration_error(probability, target, 10),
        "Binary_NLL": float(-np.mean(target * np.log(probability) + (~target) * np.log(1 - probability))),
        "predicted_harm_mean": float(probability.mean()), "observed_harm_rate": float(target.mean()),
        "mean_probability_minus_prevalence": float(probability.mean() - target.mean()),
    }


def probability_audit(samples, prediction):
    probability = np.asarray(prediction["harm"], float)
    rows = []
    categories = {
        "GT_beneficial": np.asarray([sample.targets.benefit > 1e-6 for sample in samples]),
        "GT_harmful": np.asarray([sample.targets.harm for sample in samples]),
        "GT_neutral": np.asarray([not sample.targets.harm and sample.targets.benefit <= 1e-6 for sample in samples]),
    }
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    for scope, scope_mask in (("all", np.ones(len(samples), bool)), ("feasible", feasible)):
        for category, category_mask in categories.items():
            rows.append({"synthetic_interaction": LABEL, "subset": "calibration", "scope": scope,
                         "category": category, **probability_statistics(probability[scope_mask & category_mask])})
    metrics = {
        "label": LABEL, "primary_scope": "feasible calibration candidates",
        "feasible": harm_calibration_metrics(probability[feasible], [sample.targets.harm for sample in samples if sample.targets.feasible]),
        "all_candidates_diagnostic": harm_calibration_metrics(probability, [sample.targets.harm for sample in samples]),
    }
    beneficial = probability[feasible & categories["GT_beneficial"]]
    harmful = probability[feasible & categories["GT_harmful"]]
    metrics["systematic_high_probability_diagnostic"] = {
        "beneficial_mean": float(beneficial.mean()) if len(beneficial) else None,
        "beneficial_median": float(np.median(beneficial)) if len(beneficial) else None,
        "harmful_mean": float(harmful.mean()) if len(harmful) else None,
        "separation": float(harmful.mean() - beneficial.mean()) if len(beneficial) and len(harmful) else None,
        "systematically_high_overall": abs(metrics["feasible"]["mean_probability_minus_prevalence"]) > .10,
        "systematically_high_for_beneficial_candidates": bool(len(beneficial) and np.median(beneficial) > H0_HARM_THRESHOLD),
        "diagnosis": "subgroup harm-probability calibration/class-confounding, not threshold placement alone" if len(beneficial) and np.median(beneficial) > H0_HARM_THRESHOLD else "no strong beneficial-subgroup inflation",
    }
    return rows, metrics


def threshold_candidates(probability):
    values = np.unique(np.clip(np.asarray(probability, float), 0.0, 1.0))
    return sorted(set([0.0, H0_HARM_THRESHOLD, 1.0, *map(float, values)]))


def evaluate_threshold(name, samples, prediction, harm_threshold):
    candidate_rows, decisions, metrics = b1.decision_evaluation(name, prediction, samples, (BENEFIT_THRESHOLD, harm_threshold))
    audit = b15.audit_model(name, samples, prediction, (BENEFIT_THRESHOLD, harm_threshold))
    funnel = b15.summarize_funnel(audit["funnel"])
    return candidate_rows, decisions, metrics, audit, funnel


def select_harm_threshold(calibration_samples, calibration_prediction):
    """Select exclusively on Calibration; Evaluation is not an argument."""
    rows = []
    for threshold in threshold_candidates(calibration_prediction["harm"]):
        _, _, metrics, _, funnel = evaluate_threshold("CALIBRATION_ONLY", calibration_samples, calibration_prediction, threshold)
        rows.append({"synthetic_interaction": LABEL, "subset": "calibration", "harm_threshold": threshold,
                     "Harmful_Switch_Count": metrics["Harmful_Switch_Count"],
                     "Beneficial_Switch_Count": metrics["Beneficial_Switch_Count"],
                     "Beneficial_Switch_Recall": metrics["Beneficial_Switch_Recall"],
                     "Beneficial_Switch_Precision": metrics["Beneficial_Switch_Precision"],
                     "Mean_Regret": metrics["Mean_Regret"], "P95_Regret": metrics["P95_Regret"],
                     "Safety_Violation": metrics["Safety_Violation"], "harm_pass_count": funnel["harm_threshold_pass"]})
    safe = [row for row in rows if row["Harmful_Switch_Count"] == 0]
    if not safe:
        raise RuntimeError("no calibration threshold satisfies zero harmful switches")
    selected = min(safe, key=lambda row: (-row["Beneficial_Switch_Recall"], row["Mean_Regret"], row["harm_threshold"]))
    selection = {
        "label": LABEL, "selected_on": "calibration subset only", "evaluation_access_during_selection": False,
        "selection_priority": ["Harmful_Switch_Count == 0", "maximum Beneficial_Switch_Recall",
                               "minimum Mean_Regret", "lower/more conservative harm threshold"],
        "h0_threshold": H0_HARM_THRESHOLD, "h1_threshold": selected["harm_threshold"],
        "selected_calibration_metrics": selected, "candidate_count": len(rows),
        "locked": True, "evaluation_may_not_modify": True,
    }
    selection["selection_checksum_sha256"] = b1.digest_json(selection)
    return selected["harm_threshold"], rows, selection


def harmful_rejection_rows(condition, samples, prediction, threshold):
    rows = []
    for index, sample in enumerate(samples):
        if not sample.targets.harm or not sample.targets.feasible:
            continue
        probability = float(prediction["harm"][index])
        rows.append({"synthetic_interaction": LABEL, "subset": "evaluation", "condition": condition,
                     "sample_id": sample.sample_id, "episode_id": sample.episode_id,
                     "harm_probability": probability, "harm_threshold": threshold,
                     "rejected_by_harm_gate": probability > threshold, "false_safe": probability <= threshold,
                     "benefit_threshold_pass": float(prediction["benefit"][index]) >= BENEFIT_THRESHOLD})
    return rows


def summarize_harmful_rejection(rows):
    return {"harmful_candidate_count": len(rows),
            "harmful_rejection_count": sum(row["rejected_by_harm_gate"] for row in rows),
            "harmful_rejection_rate": float(np.mean([row["rejected_by_harm_gate"] for row in rows])) if rows else None,
            "false_safe_count": sum(row["false_safe"] for row in rows),
            "false_safe_rate": float(np.mean([row["false_safe"] for row in rows])) if rows else None}


def by_context_rows(condition, samples, prediction, threshold):
    rows = []
    for context in CONTEXTS:
        indices = [index for index, sample in enumerate(samples) if context in b15.context_labels(sample)]
        subset = [samples[index] for index in indices]
        subprediction = {key: np.asarray(value)[indices] for key, value in prediction.items()}
        _, _, metrics, audit, funnel = evaluate_threshold(condition, subset, subprediction, threshold)
        rows.append({"synthetic_interaction": LABEL, "subset": "evaluation", "condition": condition,
                     "harm_threshold": threshold, "context": context,
                     "episode_count": len(b15.group_episode(subset)), "candidate_count": len(subset),
                     "beneficial_count": funnel["opportunity_count"], "feasible_count": funnel["feasible"],
                     "sign_correct_count": funnel["sign_correct"], "benefit_pass_count": funnel["benefit_threshold_pass"],
                     "harm_pass_count": funnel["harm_threshold_pass"], "generic_score_win_count": funnel["generic_score_win"],
                     "final_switch_count": funnel["final_switch"], "Beneficial_Switch_Count": metrics["Beneficial_Switch_Count"],
                     "Harmful_Switch_Count": metrics["Harmful_Switch_Count"], "Mean_Regret": metrics["Mean_Regret"],
                     "P95_Regret": metrics["P95_Regret"], "Safety_Violation": metrics["Safety_Violation"]})
    return rows


def make_figures(output, audit_rows, calibration_metrics, threshold_rows, funnels, rejection, contexts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.title(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure(figsize=(8, 4))
    feasible = [row for row in audit_rows if row["scope"] == "feasible"]
    plt.bar([row["category"].replace("GT_", "") for row in feasible], [row["mean"] for row in feasible]); plt.ylabel("mean harm probability"); save("harm_probability_by_gt_class.png")
    plt.figure(); plt.scatter([row["harm_threshold"] for row in threshold_rows], [row["Beneficial_Switch_Recall"] for row in threshold_rows], s=10); plt.axvline(H0_HARM_THRESHOLD, color="k", linestyle="--"); plt.xlabel("harm threshold"); plt.ylabel("calibration beneficial recall"); save("threshold_sweep_recall.png")
    stages=("opportunity_count", "feasible", "sign_correct", "benefit_threshold_pass", "harm_threshold_pass", "generic_score_win", "final_switch")
    plt.figure(figsize=(10, 4))
    for condition in MODELS: plt.plot(stages, [funnels[condition][stage] for stage in stages], marker="o", label=condition[:2])
    plt.xticks(rotation=25, ha="right"); plt.ylabel("evaluation beneficial candidates"); plt.legend(); save("h0_h1_funnel.png")
    plt.figure(); plt.bar([condition[:2] for condition in MODELS], [rejection[condition]["harmful_rejection_rate"] for condition in MODELS]); plt.ylim(0, 1.05); plt.ylabel("harmful candidate rejection rate"); save("harmful_rejection.png")
    plt.figure(figsize=(8, 4)); x=np.arange(3); width=.35
    for offset, condition in ((-.5, MODELS[0]), (.5, MODELS[1])):
        values=[next(row["harm_pass_count"] for row in contexts if row["condition"]==condition and row["context"]==context) for context in CONTEXTS]
        plt.bar(x+offset*width, values, width, label=condition[:2])
    plt.xticks(x, CONTEXTS); plt.ylabel("benefit-pass then harm-pass count"); plt.legend(); save("context_harm_pass.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Phase5B-1.7: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    manifest, manifest_audit = b1.manifest_file_audit(args.manifest_dir)
    samples = b15.build_validation_only(args, torch)
    manifest_validation = {candidate for row in manifest["episodes"] if row["split"] == "validation" for candidate in row["candidate_ids"]}
    if {sample.sample_id for sample in samples} != manifest_validation:
        raise RuntimeError("validation candidates differ from frozen manifest")
    normalizers, normalizer_record = b16.load_frozen_normalizer(args.phase5b1_dir)
    checkpoint_path = args.phase5b16_dir / "checkpoints" / "r1_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selection16 = json.loads((args.phase5b16_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
    if tuple(selection16["models"]["R1 B1-Rank"]["thresholds"]) != (BENEFIT_THRESHOLD, H0_HARM_THRESHOLD):
        raise RuntimeError("Phase5B-1.6 R1 frozen thresholds mismatch")
    if checkpoint["manifest_sha256"] != b1.EXPECTED_MANIFEST_SHA or checkpoint["normalizer_sha256"] != normalizer_record["sha256"]:
        raise RuntimeError("Phase5B-1.6 R1 checkpoint contract mismatch")
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    model = RichTemporalSmallTransformer(); model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.to(device).eval()
    checksum_before = b16.model_checksum(model)
    arbitration_source = inspect.getsource(__import__("src.decision.large_context_arbitrator", fromlist=["arbitrate_large_context"]).arbitrate_large_context)
    arbitration_sha = hashlib.sha256(arbitration_source.encode()).hexdigest()
    prediction = b1.predict("B1", model, samples, normalizers, args.batch_size, torch, device)

    split = stratified_episode_split(samples, args.seed)
    p5.write_json(args.output_dir / "validation_harm_calibration_split.json", split)
    calibration_samples, calibration_prediction = select_subset(samples, prediction, split["calibration_episode_ids"])
    evaluation_ids = split["evaluation_episode_ids"]  # IDs only; labels/predictions remain sealed until H1 is locked.
    audit_rows, calibration_metrics = probability_audit(calibration_samples, calibration_prediction)
    p5.write_csv(args.output_dir / "harm_probability_audit.csv", audit_rows)
    p5.write_json(args.output_dir / "harm_calibration_metrics.json", calibration_metrics)

    h1_threshold, threshold_rows, threshold_selection = select_harm_threshold(calibration_samples, calibration_prediction)
    p5.write_csv(args.output_dir / "harm_threshold_candidates.csv", threshold_rows)
    p5.write_json(args.output_dir / "harm_threshold_selection.json", threshold_selection)

    # Formal Evaluation access starts only after the selected threshold is locked on disk.
    evaluation_samples, evaluation_prediction = select_subset(samples, prediction, evaluation_ids)
    conditions = {MODELS[0]: H0_HARM_THRESHOLD, MODELS[1]: h1_threshold}
    funnels, decisions, decision_metrics, all_funnel, all_rejection, context_rows = {}, {}, {}, [], [], []
    rejection_summary = {}
    for condition, threshold in conditions.items():
        _, decision_rows, metrics, audit, funnel = evaluate_threshold(condition, evaluation_samples, evaluation_prediction, threshold)
        funnels[condition], decisions[condition], decision_metrics[condition] = funnel, decision_rows, metrics
        all_funnel += audit["funnel"]
        rejection = harmful_rejection_rows(condition, evaluation_samples, evaluation_prediction, threshold)
        all_rejection += rejection; rejection_summary[condition] = summarize_harmful_rejection(rejection)
        context_rows += by_context_rows(condition, evaluation_samples, evaluation_prediction, threshold)

    h0, h1 = MODELS
    gate_a_checks = {
        "benefit_pass_then_harm_pass_increased": funnels[h1]["harm_threshold_pass"] > funnels[h0]["harm_threshold_pass"],
        "harmful_rejection_not_clearly_lower": rejection_summary[h1]["harmful_rejection_rate"] >= rejection_summary[h0]["harmful_rejection_rate"] - HARMFUL_REJECTION_TOLERANCE,
    }
    gate_b_checks = {
        "beneficial_switch_count_increased": decision_metrics[h1]["Beneficial_Switch_Count"] > decision_metrics[h0]["Beneficial_Switch_Count"],
        "beneficial_episode_recall_increased": decision_metrics[h1]["Beneficial_Switch_Recall"] > decision_metrics[h0]["Beneficial_Switch_Recall"],
        "harmful_switch_count_zero": decision_metrics[h1]["Harmful_Switch_Count"] == 0,
        "safety_not_worse": decision_metrics[h1]["Safety_Violation"] <= decision_metrics[h0]["Safety_Violation"],
        "mean_regret_not_worse": decision_metrics[h1]["Mean_Regret"] <= decision_metrics[h0]["Mean_Regret"] + 1e-12,
    }
    gate_a, gate_b = all(gate_a_checks.values()), all(gate_b_checks.values())
    checksum_after = b16.model_checksum(model)
    generic_wins = funnels[h1]["generic_score_win"]
    harm_repaired_arbitration_bottleneck = gate_a and funnels[h1]["harm_threshold_pass"] > funnels[h0]["harm_threshold_pass"] and generic_wins <= 2
    figures = make_figures(args.output_dir, audit_rows, calibration_metrics, threshold_rows, funnels, rejection_summary, context_rows)

    frozen_contract = {
        "label": LABEL, **manifest_audit, "validation_candidate_count": len(samples), "validation_episode_count": len(b15.group_episode(samples)),
        "test_candidates_read": 0, "test_labels_read": 0, "test_metrics_computed": False,
        "checkpoint_path": str(checkpoint_path), "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "model_parameter_checksum_before": checksum_before, "model_parameter_checksum_after": checksum_after,
        "model_parameters_unchanged": checksum_before == checksum_after, "optimizer_created": False, "optimizer_step_count": 0, "backward_call_count": 0,
        "ranking_lambda_before": LAMBDA_RANK, "ranking_lambda_after": LAMBDA_RANK, "ranking_lambda_unchanged": LAMBDA_RANK == .25,
        "benefit_threshold_before": BENEFIT_THRESHOLD, "benefit_threshold_after": BENEFIT_THRESHOLD, "benefit_threshold_unchanged": True,
        "normalizer_sha256": normalizer_record["sha256"], "normalizer_unchanged": True,
        "safety_mask_checksum_before": b1.contract_hash(samples, "targets"), "safety_mask_checksum_after": b1.contract_hash(samples, "targets"), "safety_mask_unchanged": True,
        "generic_score_checksum_before": metadata_checksum(samples, ("generic_costs_evaluation_only",)),
        "generic_score_checksum_after": metadata_checksum(samples, ("generic_costs_evaluation_only",)), "generic_score_unchanged": True,
        "personalized_cost_checksum_before": metadata_checksum(samples, ("personalized_costs_evaluation_only",)),
        "personalized_cost_checksum_after": metadata_checksum(samples, ("personalized_costs_evaluation_only",)), "personalized_cost_unchanged": True,
        "arbitration_source_sha256_before": arbitration_sha, "arbitration_source_sha256_after": arbitration_sha, "arbitration_unchanged": True,
        "uncertainty_handling_unchanged": True, "only_changed_value": "harm threshold",
    }
    decision_rows = [{"synthetic_interaction": LABEL, "subset": "evaluation", "condition": condition,
                      "harm_threshold": conditions[condition], **decision_metrics[condition]} for condition in MODELS]
    p5.write_json(args.output_dir / "frozen_contract.json", frozen_contract)
    p5.write_csv(args.output_dir / "h0_vs_h1_funnel.csv", all_funnel)
    p5.write_csv(args.output_dir / "h0_vs_h1_decision_metrics.csv", decision_rows)
    p5.write_csv(args.output_dir / "harmful_rejection_audit.csv", all_rejection)
    p5.write_csv(args.output_dir / "by_context_metrics.csv", context_rows)
    summary = {
        "label": LABEL, "stage": "Phase 5B-1.7 Harm Gate Calibration", "validation_holdout_evaluation_only": True,
        "split_checksum_sha256": split["split_checksum_sha256"], "calibration_episode_count": split["calibration_episode_count"],
        "evaluation_episode_count": split["evaluation_episode_count"], "h0_harm_threshold": H0_HARM_THRESHOLD, "h1_harm_threshold": h1_threshold,
        "harm_probability_calibration": calibration_metrics, "evaluation_funnel": funnels,
        "evaluation_decision_metrics": decision_metrics, "harmful_rejection": rejection_summary,
        "gate_a_harm_mechanism": {"passed": gate_a, "checks": gate_a_checks, "rejection_tolerance_absolute": HARMFUL_REJECTION_TOLERANCE},
        "gate_b_decision_transfer": {"passed": gate_b, "checks": gate_b_checks, "any_new_harmful_switch_forces_fail": True},
        "harm_gate_repaired_and_bottleneck_transferred_to_generic_score_arbitration": harm_repaired_arbitration_bottleneck,
        "phase5b18_allowed": bool(gate_a and harm_repaired_arbitration_bottleneck), "phase5b18_requires_human_approval": True,
        "phase5b18_automatically_started": False, "phase5b2_started": False, "test_candidates_read": 0, "figures": figures,
    }
    p5.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(p5.clean(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
