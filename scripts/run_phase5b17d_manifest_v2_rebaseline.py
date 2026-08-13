"""Phase 5B-1.7D manifest-v2 fair static/temporal/ranking rebaseline."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b16_candidate_ranking as b16
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.context_value_metrics import candidate_metrics, validation_selection_key
from src.multimodal.phase5b_v2_dataset import DEPRECATED_HARM_TARGET, build_v2_temporal_samples, runtime_contract_audit
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK, pairwise_logistic_ranking_loss

EXPECTED_MANIFEST_SHA = "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
MODELS = ("B0-v2 Static Small", "B1-v2 Rich Temporal Small", "R1-v2 Rich Temporal + Ranking")
FROZEN_THRESHOLDS = (-0.02, 0.2)
TRAIN_SIZE = VALIDATION_SIZE = 240
MAE_TOLERANCE_ABSOLUTE = .015
MAE_TOLERANCE_RELATIVE = .20
TOP_ACCURACY_TOLERANCE = .02


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17d_manifest_v2_rebaseline")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    return parser.parse_args()


def file_sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def model_sha(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def manifest_contract(path: Path, samples):
    digest = file_sha(path)
    if digest != EXPECTED_MANIFEST_SHA: raise RuntimeError(f"manifest_v2 checksum mismatch: {digest}")
    manifest = json.loads(path.read_text(encoding="utf-8")); expected = {}; test_rows_seen = 0
    for row in manifest["episodes"]:
        if row["split"] == "test":
            test_rows_seen += 1
            continue  # Never inspect TEST candidate IDs, trajectories, benefit or labels.
        if row["split"] not in ("train", "validation"): raise ValueError("unknown manifest split")
        labels = row["harm_v2_labels"]
        for candidate_id in row["candidate_ids"]:
            action = candidate_id.rsplit(":", 1)[1]
            expected[candidate_id] = (row["episode_id"], row["split"], bool(labels[action]))
    actual = {sample.sample_id: (sample.episode_id, sample.split, bool(sample.split_metadata["harm_v2_evaluation_only"])) for sample in samples}
    episode_splits = defaultdict(set)
    for sample in samples: episode_splits[sample.episode_id].add(sample.split)
    manifest_v1 = PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage" / "phase5b_manifest_v1.json"
    manifest_v1_sha = file_sha(manifest_v1)
    result = {
        "label": LABEL, "manifest_version": manifest["version"], "manifest_sha256": digest,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA, "checksum_matches": digest == EXPECTED_MANIFEST_SHA,
        "development_candidate_ids_identical": expected == actual, "development_candidate_count": len(actual),
        "train_candidates": sum(sample.split == "train" for sample in samples),
        "validation_candidates": sum(sample.split == "validation" for sample in samples),
        "sealed_test_episode_metadata_rows_skipped": test_rows_seen, "test_candidate_reads": 0,
        "test_trajectory_reads": 0, "test_benefit_reads": 0, "test_harm_v2_reads": 0, "test_reads": 0,
        "same_episode_cross_split": sorted(key for key, value in episode_splits.items() if len(value) > 1),
        "split_before_candidate_branching": True,
        "manifest_v1_sha256": manifest_v1_sha, "manifest_v1_expected_sha256": b1.EXPECTED_MANIFEST_SHA,
        "manifest_v1_unchanged": manifest_v1_sha == b1.EXPECTED_MANIFEST_SHA,
    }
    result["passed"] = result["checksum_matches"] and result["development_candidate_ids_identical"] and result["manifest_v1_unchanged"] and not result["same_episode_cross_split"] and result["test_reads"] == 0
    if not result["passed"]: raise RuntimeError("manifest_v2 runtime contract failed")
    return result


def normalizer_record(normalizers):
    record = {"label": LABEL, "fit_split": normalizers["fit_split"], "fit_scope": normalizers["fit_scope"],
              "fit_sample_ids": normalizers["fit_sample_ids"], "static_mean": normalizers["static_mean"],
              "static_scale": normalizers["static_scale"], "benefit_mean": normalizers["benefit_mean"],
              "benefit_scale": normalizers["benefit_scale"], "benefit_raw_std": normalizers["benefit_raw_std"],
              "stream": normalizers["stream"], "harm_pos_weight": normalizers["harm_pos_weight"],
              "harm_target": DEPRECATED_HARM_TARGET, "harm_v2_used_for_fit": False}
    record["sha256"] = b1.digest_json(record)
    return record


def loss_terms(output, selected, normalizers, torch, device):
    target = torch.tensor([(sample.targets.benefit - normalizers["benefit_mean"]) / normalizers["benefit_scale"] for sample in selected], dtype=torch.float32, device=device)
    old_harm = torch.tensor([sample.targets.harm for sample in selected], dtype=torch.float32, device=device)
    feasible = torch.tensor([sample.targets.feasible for sample in selected], dtype=torch.bool, device=device)
    error = output.benefit_mean[feasible] - target[feasible]; log_variance = output.benefit_log_variance[feasible]
    nll = .5 * (error.square() * torch.exp(-log_variance) + log_variance).mean()
    old_harm_loss = torch.nn.functional.binary_cross_entropy_with_logits(output.harm_logit[feasible], old_harm[feasible], pos_weight=torch.tensor(2.0, device=device))
    rank, audit = pairwise_logistic_ranking_loss(output.benefit_mean, target, [sample.episode_id for sample in selected], feasible)
    return {"nll": nll, "deprecated_old_harm": old_harm_loss, "base": nll + old_harm_loss,
            "rank": rank, "weighted_rank": LAMBDA_RANK * rank, "rank_audit": audit}


def rank_evaluation(model_name, samples, prediction):
    rows = []
    for episode_id, indices in b15.group_episode(samples).items():
        predicted = np.asarray(prediction["benefit"])[indices]
        target = np.asarray([samples[index].targets.benefit for index in indices], float)
        feasible = np.asarray([samples[index].targets.feasible for index in indices], bool)
        actions = np.asarray([samples[index].split_metadata["candidate_action_id_audit"] for index in indices], int)
        valid = np.flatnonzero(feasible); best = int(np.lexsort((actions, -target))[0])
        valid_best = int(valid[np.lexsort((actions[valid], -target[valid]))[0]])
        rank = b15.ranks_desc(predicted); valid_rank = b15.ranks_desc(predicted[valid])
        beneficial = float(target.max()) > 1e-6
        rows.append({"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                     "within_episode_spearman": b15.spearman(predicted, target),
                     "feasible_within_episode_spearman": b15.spearman(predicted[valid], target[valid]),
                     "feasible_pairwise_accuracy": b15.pairwise_accuracy(predicted[valid], target[valid]),
                     "gt_best_top1": int(rank[best] == 1), "gt_best_top2": int(rank[best] <= 2),
                     "gt_best_rank": int(rank[best]), "beneficial_episode": beneficial,
                     "beneficial_gt_best_rank": int(rank[best]) if beneficial else "",
                     "feasible_gt_best_rank": int(valid_rank[np.flatnonzero(valid == valid_best)[0]]),
                     "candidate_count": len(indices), "feasible_count": len(valid)})
    summary = b15.summarize_ranking([{**row, "pairwise_ranking_accuracy": row["feasible_pairwise_accuracy"],
                                      "generic_vs_best_personalized_pair_accuracy": row["feasible_pairwise_accuracy"]} for row in rows])
    # Explicit aliases used by the preregistered 1.7D report.
    summary["mean_feasible_pairwise_accuracy"] = float(np.mean([row["feasible_pairwise_accuracy"] for row in rows]))
    summary["benefit_sign_accuracy"] = float(np.mean(np.sign(prediction["benefit"]) == np.sign([sample.targets.benefit for sample in samples])))
    return rows, summary


def validation_snapshot(model_name, model, samples, normalizers, args, torch, device):
    prediction = b1.predict(model_name, model, samples, normalizers, args.batch_size, torch, device)
    targets = {"benefit": np.asarray([sample.targets.benefit for sample in samples]), "harm": np.asarray([sample.targets.harm for sample in samples])}
    metrics = candidate_metrics(prediction, targets, np.asarray([sample.targets.feasible for sample in samples]))
    _, ranking = rank_evaluation(model_name, samples, prediction)
    # Reuse the frozen Phase5B-1 checkpoint selector exactly.  These deprecated
    # arbitration-derived fields are not published as decision/safety results.
    _, _, historical = b1.decision_evaluation(model_name, prediction, samples, FROZEN_THRESHOLDS)
    return prediction, metrics, ranking, historical, validation_selection_key(historical)


def train_model(model_name, model, train, validation, normalizers, epoch_batches, args, torch, device, use_ranking):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    curve, best, stale, start_time = [], None, 0, time.perf_counter(); model.to(device)
    for epoch, batches in enumerate(epoch_batches, 1):
        model.train(); batch_rows = []
        for indices in batches:
            selected = [train[index] for index in indices]
            output = model(b1.model_batch(model_name, selected, normalizers, torch, device))
            terms = loss_terms(output, selected, normalizers, torch, device)
            contribution = terms["weighted_rank"] if use_ranking else terms["rank"] * 0.0
            loss = terms["base"] + contribution
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)); optimizer.step()
            if not bool(torch.isfinite(loss)) or not np.isfinite(gradient): raise FloatingPointError("non-finite 1.7D training")
            batch_rows.append({name: float(terms[name].detach()) for name in ("nll", "deprecated_old_harm", "base", "rank", "weighted_rank")} | {"total": float(loss.detach()), "ranking_pairs": terms["rank_audit"].pair_count})
        _, metrics, ranking, historical, key = validation_snapshot(model_name, model, validation, normalizers, args, torch, device)
        row = {"synthetic_interaction": LABEL, "model": model_name, "epoch": epoch,
               **{f"train_{name}": float(np.mean([item[name] for item in batch_rows])) for name in batch_rows[0]},
               **metrics, **ranking, "checkpoint_selection_semantics": "frozen historical validation-only; old harm/decision semantics deprecated"}
        curve.append(row)
        if best is None or key < best[0]: best = (key, epoch, copy.deepcopy(model.state_dict()), metrics, ranking, historical); stale = 0
        else: stale += 1
        print(f"{model_name} epoch={epoch:02d} total={row['train_total']:.5f} val_mae={metrics['Benefit_MAE']:.5f} rank={ranking['mean_gt_best_rank']:.3f} stale={stale}", flush=True)
        if stale >= args.patience: break
    model.load_state_dict(best[2]); model.eval()
    selection = {"best_epoch": best[1], "selection_key": list(best[0]), "validation_candidate_metrics": best[3],
                 "validation_ranking_metrics": best[4], "historical_selection_metrics_deprecated_not_a_decision_result": best[5],
                 "epochs_completed": len(curve), "early_stopped": len(curve) < args.epochs,
                 "training_time_s": time.perf_counter() - start_time, "checkpoint_selected_on": "validation only",
                 "checkpoint_criterion": "unchanged Phase5B-1 validation_selection_key; decision interpretation prohibited",
                 "ranking_loss_contribution": LAMBDA_RANK if use_ranking else 0.0}
    return model, curve, selection


def evaluate(model_name, model, validation, normalizers, args, torch, device):
    prediction = b1.predict(model_name, model, validation, normalizers, args.batch_size, torch, device)
    targets = {"benefit": np.asarray([sample.targets.benefit for sample in validation]), "harm": np.asarray([sample.targets.harm for sample in validation])}
    metrics = candidate_metrics(prediction, targets, np.asarray([sample.targets.feasible for sample in validation]))
    rows, ranking = rank_evaluation(model_name, validation, prediction)
    return prediction, rows, {**metrics, **ranking}


def subset_metrics(model_name, samples, prediction, predicate, dimension, group):
    episodes = {sample.episode_id for sample in samples if predicate(sample)}
    indices = [index for index, sample in enumerate(samples) if sample.episode_id in episodes]
    if not indices: return {"synthetic_interaction": LABEL, "model": model_name, "dimension": dimension, "group": group, "episode_count": 0, "candidate_count": 0}
    subset = [samples[index] for index in indices]; subprediction = {key: np.asarray(value)[indices] for key, value in prediction.items()}
    targets = {"benefit": np.asarray([sample.targets.benefit for sample in subset]), "harm": np.asarray([sample.targets.harm for sample in subset])}
    candidate = candidate_metrics(subprediction, targets, np.asarray([sample.targets.feasible for sample in subset]))
    candidate.pop("Candidate_Count")
    _, ranking = rank_evaluation(model_name, subset, subprediction)
    harm_v2 = np.asarray([sample.split_metadata["harm_v2_evaluation_only"] for sample in subset], bool)
    safe_beneficial = np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in subset], bool)
    return {"synthetic_interaction": LABEL, "model": model_name, "dimension": dimension, "group": group,
            "scope": "all candidates in episodes matching group", "episode_count": len(episodes), "candidate_count": len(indices),
            "harm_v2_positive_candidates": int(harm_v2.sum()), "harm_v2_positive_rate": float(harm_v2.mean()),
            "safe_beneficial_candidates": int(safe_beneficial.sum()), **candidate, **ranking}


def grouped_rows(samples, predictions):
    adverse = {
        "EXCESSIVE_DECELERATION": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
        "ABRUPT_LATERAL_RESPONSE": lambda sample: sample.split_metadata["abrupt_lateral_response_evaluation_only"],
        "ABRUPT_HEADING_CHANGE": lambda sample: sample.split_metadata["abrupt_heading_change_evaluation_only"],
        "GT_UNSAFE": lambda sample: sample.targets.gt_unsafe,
    }
    contexts = {name: (lambda sample, name=name: any(str(value).startswith(name) for value in sample.split_metadata["contexts_evaluation_only"])) for name in ("C7", "C8", "C9")}
    motions = sorted({str(sample.split_metadata["motion_type_evaluation_only"]) for sample in samples})
    by_adverse, by_context, by_motion = [], [], []
    for model_name, prediction in predictions.items():
        by_adverse += [subset_metrics(model_name, samples, prediction, predicate, "adverse_event", name) for name, predicate in adverse.items()]
        by_context += [subset_metrics(model_name, samples, prediction, predicate, "context", name) for name, predicate in contexts.items()]
        by_motion += [subset_metrics(model_name, samples, prediction, lambda sample, motion=motion: sample.split_metadata["motion_type_evaluation_only"] == motion, "motion", motion) for motion in motions]
    return by_adverse, by_context, by_motion


def harm_v2_audit(samples, predictions):
    groups = {
        "harm_v2_positive": lambda sample: sample.split_metadata["harm_v2_evaluation_only"],
        "harm_v2_negative": lambda sample: not sample.split_metadata["harm_v2_evaluation_only"],
        "safe_beneficial": lambda sample: sample.split_metadata["safe_beneficial_evaluation_only"],
        "benefit_risk_tradeoff": lambda sample: sample.split_metadata["benefit_risk_tradeoff_evaluation_only"],
    }
    rows = []
    for model_name, prediction in predictions.items():
        for group, predicate in groups.items():
            indices = [index for index, sample in enumerate(samples) if predicate(sample)]
            target = np.asarray([samples[index].targets.benefit for index in indices]); predicted = np.asarray(prediction["benefit"])[indices]
            rows.append({"synthetic_interaction": LABEL, "model": model_name, "group": group, "audit_only": True,
                         "harm_v2_entered_optimizer": False, "candidate_count": len(indices),
                         "Benefit_MAE": float(np.mean(np.abs(predicted - target))) if len(indices) else None,
                         "Benefit_Sign_Accuracy": float(np.mean(np.sign(predicted) == np.sign(target))) if len(indices) else None})
    return rows


def make_figures(output, curves, metrics, adverse_rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.suptitle(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure()
    for name, rows in curves.items(): plt.plot([row["epoch"] for row in rows], [row["Benefit_MAE"] for row in rows], label=name)
    plt.xlabel("epoch"); plt.ylabel("validation Benefit MAE"); plt.legend(fontsize=7); save("training_validation_benefit_mae.png")
    plt.figure(figsize=(10, 4)); names = list(metrics); x = np.arange(len(names)); width = .25
    for offset, field in enumerate(("Benefit_Spearman", "mean_feasible_within_episode_spearman", "mean_feasible_pairwise_accuracy")):
        plt.bar(x + (offset - 1) * width, [metrics[name][field] for name in names], width, label=field)
    plt.xticks(x, names, rotation=12); plt.legend(fontsize=6); save("global_ranking_metrics.png")
    plt.figure(figsize=(10, 4)); groups = sorted({row["group"] for row in adverse_rows}); x = np.arange(len(groups)); width = .25
    for offset, model in enumerate(MODELS):
        plt.bar(x + (offset - 1) * width, [next(row["mean_feasible_pairwise_accuracy"] for row in adverse_rows if row["model"] == model and row["group"] == group) for group in groups], width, label=model)
    plt.xticks(x, groups, rotation=18); plt.ylabel("pairwise accuracy"); plt.legend(fontsize=6); save("adverse_event_pairwise_accuracy.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7D: {args.output_dir}")
    args.output_dir.mkdir(parents=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    episodes = {"train": build_development_split("train", TRAIN_SIZE, GENERATOR_SEED, RISK_SEED),
                "validation": build_development_split("validation", VALIDATION_SIZE, GENERATOR_SEED + 1000, RISK_SEED + 1000)}
    splits = {name: build_v2_temporal_samples(values) for name, values in episodes.items()}; all_samples = splits["train"] + splits["validation"]
    contract = manifest_contract(args.manifest, all_samples); runtime_audit = runtime_contract_audit(all_samples)
    normalizers = b1.fit_normalizers(splits["train"]); normalizer = normalizer_record(normalizers); io.write_json(args.output_dir / "normalizer.json", normalizer)

    from src.models.large_context_adapter import SmallContextNetwork
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    torch.manual_seed(args.seed); b0 = SmallContextNetwork()
    torch.manual_seed(args.seed); b1_model = RichTemporalSmallTransformer(); b1_initial = model_sha(b1_model)
    r1_model = copy.deepcopy(b1_model); r1_initial = model_sha(r1_model)
    models = {MODELS[0]: b0, MODELS[1]: b1_model, MODELS[2]: r1_model}
    batches, batch_audit = b16.make_episode_batches(splits["train"], args.epochs, args.batch_size, args.seed)
    curves, selections = {}, {}
    for name, model in models.items():
        model, curve, selection = train_model(name, model, splits["train"], splits["validation"], normalizers, batches, args, torch, device, name == MODELS[2])
        models[name], curves[name], selections[name] = model, curve, selection

    predictions, ranking_rows, metrics = {}, [], {}
    for name, model in models.items():
        prediction, rows, result = evaluate(name, model, splits["validation"], normalizers, args, torch, device)
        predictions[name], metrics[name] = prediction, result; ranking_rows += rows
    adverse_rows, context_rows, motion_rows = grouped_rows(splits["validation"], predictions)
    harm_rows = harm_v2_audit(splits["validation"], predictions)

    b0m, b1m, r1m = metrics[MODELS[0]], metrics[MODELS[1]], metrics[MODELS[2]]
    mae_limit = b0m["Benefit_MAE"] + max(MAE_TOLERANCE_ABSOLUTE, MAE_TOLERANCE_RELATIVE * b0m["Benefit_MAE"])
    gate_a_checks = {"benefit_or_within_ranking_improved": bool((b1m["Benefit_Spearman"] or -1) > (b0m["Benefit_Spearman"] or -1) or b1m["mean_within_episode_spearman"] > b0m["mean_within_episode_spearman"]),
                     "benefit_mae_not_clearly_worse": b1m["Benefit_MAE"] <= mae_limit}
    gate_b_checks = {"feasible_spearman_improved": r1m["mean_feasible_within_episode_spearman"] > b1m["mean_feasible_within_episode_spearman"],
                     "pairwise_accuracy_improved": r1m["mean_feasible_pairwise_accuracy"] > b1m["mean_feasible_pairwise_accuracy"],
                     "mean_gt_best_rank_improved": r1m["mean_gt_best_rank"] < b1m["mean_gt_best_rank"],
                     "top1_not_clearly_worse": r1m["gt_best_top1_accuracy"] + TOP_ACCURACY_TOLERANCE >= b1m["gt_best_top1_accuracy"],
                     "top2_not_clearly_worse": r1m["gt_best_top2_recall"] + TOP_ACCURACY_TOLERANCE >= b1m["gt_best_top2_recall"]}
    gate_a, gate_b = all(gate_a_checks.values()), all(gate_b_checks.values())
    old_harm_source = inspect.getsource(loss_terms)
    training_config = {"label": LABEL, "stage": "Phase 5B-1.7D Manifest-v2 Fair Rebaseline", "seed": 42,
                       "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-3, "batch_size": 64,
                       "max_epochs": 30, "patience": 5, "gradient_clip": 10.0, "episode_grouped_batches": True,
                       "base_loss": "heteroscedastic Gaussian NLL + BCE deprecated old-harm auxiliary",
                       "old_harm_target": DEPRECATED_HARM_TARGET, "old_harm_interpretation": "historical reproducibility only; not safety/adverse/independent harm probability",
                       "harm_v2_used_in_optimizer": "harm_v2" in old_harm_source, "R1_only_training_difference": "lambda_rank=0.25 candidate-set pairwise logistic loss",
                       "lambda_rank": LAMBDA_RANK, "ranking_feasible_only": True, "ranking_same_episode_only": True, "ranking_ties_excluded": True,
                       "checkpoint_selection": "unchanged Phase5B-1 validation_selection_key on validation only; historical decision fields deprecated and not reported as decision gate",
                       "test_reads": 0, "hyperparameter_search": False}
    if training_config["harm_v2_used_in_optimizer"]: raise RuntimeError("harm_v2 leaked into optimizer loss")
    audits = {
        MODELS[0]: {"model": "SmallContextNetwork", "input": "canonical Phase5A-compatible [B,108]", "parameter_count": sum(p.numel() for p in b0.parameters()),
                    "trainable_parameter_count": sum(p.numel() for p in b0.parameters() if p.requires_grad), "architecture_changed_for_phase5b17d": False},
        MODELS[1]: b1_model.architecture_audit(), MODELS[2]: r1_model.architecture_audit(),
    }
    audits[MODELS[1]]["initial_checksum"] = b1_initial; audits[MODELS[2]]["initial_checksum"] = r1_initial
    audits[MODELS[2]]["architecture_identical_to_B1"] = b1_model.architecture_audit() == r1_model.architecture_audit()
    audits[MODELS[2]]["initial_checksum_identical_to_B1"] = b1_initial == r1_initial
    figures = make_figures(args.output_dir, curves, metrics, adverse_rows)

    global_rows = [{"synthetic_interaction": LABEL, "model": name, **value,
                    "Harm_AUROC_semantics": "DEPRECATED OLD-HARM SEMANTICS",
                    "Harm_AUPRC_semantics": "DEPRECATED OLD-HARM SEMANTICS"} for name, value in metrics.items()]
    ranking_summary = {"label": LABEL, "models": metrics, "gate_a_temporal_reproducibility": {"passed": gate_a, "checks": gate_a_checks, "mae_not_worse_limit": mae_limit},
                       "gate_b_ranking_reproducibility": {"passed": gate_b, "checks": gate_b_checks, "top_accuracy_tolerance": TOP_ACCURACY_TOLERANCE}}
    selection_record = {"label": LABEL, "manifest_sha256": EXPECTED_MANIFEST_SHA, "normalizer_sha256": normalizer["sha256"], "validation_only": True,
                        "test_reads": 0, "models": selections, "freeze_allowed": bool(gate_a and gate_b)}
    summary = {"label": LABEL, "stage": "Phase 5B-1.7D Manifest-v2 Fair Rebaseline", "models": metrics,
               "manifest_contract_passed": contract["passed"], "runtime_contract": runtime_audit,
               "B1_R1_architecture_identical": audits[MODELS[2]]["architecture_identical_to_B1"],
               "B1_R1_initial_checksum_identical": b1_initial == r1_initial,
               "harm_v2_entered_optimizer": False, "test_reads": 0,
               "old_harm_semantics": "DEPRECATED OLD-HARM SEMANTICS; no safety conclusion",
               "formal_decision_gate_performed": False, "decision_arbitration_modified": False,
               "gate_a_temporal_reproducibility": ranking_summary["gate_a_temporal_reproducibility"],
               "gate_b_ranking_reproducibility": ranking_summary["gate_b_ranking_reproducibility"],
               "checkpoints_frozen": bool(gate_a and gate_b), "R1_v2_ready_as_harm_v2_backbone": bool(gate_a and gate_b),
               "ready_for_phase5b17e": bool(gate_a and gate_b), "phase5b17e_started": False, "phase5b2_started": False,
               "figures": figures}

    io.write_json(args.output_dir / "manifest_contract.json", contract); io.write_json(args.output_dir / "training_config.json", training_config)
    for filename, name in (("b0_model_audit.json", MODELS[0]), ("b1_model_audit.json", MODELS[1]), ("r1_model_audit.json", MODELS[2])): io.write_json(args.output_dir / filename, audits[name])
    for filename, name in (("b0_training_curve.csv", MODELS[0]), ("b1_training_curve.csv", MODELS[1]), ("r1_training_curve.csv", MODELS[2])): io.write_csv(args.output_dir / filename, curves[name])
    io.write_csv(args.output_dir / "global_metrics.csv", global_rows); io.write_csv(args.output_dir / "episode_ranking.csv", ranking_rows)
    io.write_json(args.output_dir / "ranking_summary.json", ranking_summary); io.write_csv(args.output_dir / "by_adverse_event.csv", adverse_rows)
    io.write_csv(args.output_dir / "by_context.csv", context_rows); io.write_csv(args.output_dir / "by_motion.csv", motion_rows)
    io.write_csv(args.output_dir / "harm_v2_audit_only.csv", harm_rows); io.write_json(args.output_dir / "checkpoint_selection.json", selection_record)
    io.write_json(args.output_dir / "batch_order_audit.json", {"label": LABEL, "epochs": batch_audit})
    if gate_a and gate_b:
        for name, filename in zip(MODELS, ("b0_v2_best.pt", "b1_v2_best.pt", "r1_v2_best.pt")):
            torch.save({"model_state_dict": models[name].state_dict(), "model": name, "manifest_sha256": EXPECTED_MANIFEST_SHA,
                        "normalizer_sha256": normalizer["sha256"], "selection": selections[name], "label": LABEL}, args.output_dir / "checkpoints" / filename)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()
