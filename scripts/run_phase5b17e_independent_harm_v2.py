"""Phase 5B-1.7E independent harm-v2 head-only training.

All data are synthetic DEVELOPMENT/VALIDATION data. TEST is sealed. The
R1-v2-CRACS representation and every historical output head remain frozen;
only a newly initialized architecture-control harm-v2 head is optimized.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
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
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.context_value_metrics import binary_auc
from src.evaluation.probabilistic_harm import harm_metrics, phs_select, prevalence_baseline
from src.models.independent_harm_head import IndependentHarmV2Head
from src.multimodal.phase5b_v2_dataset import DEPRECATED_HARM_TARGET, build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL
from src.training.independent_harm import harm_v2_target, unweighted_harm_v2_loss

STAGE = "Phase 5B-1.7E Independent Harm-v2 Head Training"
ACTION_NAMES = {0: "KEEP", 1: "SPEED_DOWN", 2: "SPEED_UP", 3: "DISTANCE_PLUS", 4: "DISTANCE_MINUS"}
SUBTYPES = {
    "GT_UNSAFE": lambda sample: sample.targets.gt_unsafe,
    "EXCESSIVE_DECELERATION": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
    "ABRUPT_LATERAL_RESPONSE": lambda sample: sample.split_metadata["abrupt_lateral_response_evaluation_only"],
    "ABRUPT_HEADING_CHANGE": lambda sample: sample.split_metadata["abrupt_heading_change_evaluation_only"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17e_independent_harm_v2")
    return parser.parse_args()


def file_sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(state: dict, prefixes: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if prefixes is not None and not name.startswith(prefixes): continue
        if name.startswith(exclude): continue
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_sha(array) -> str:
    value = np.ascontiguousarray(array); digest = hashlib.sha256()
    digest.update(str(value.shape).encode()); digest.update(str(value.dtype).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def load_frozen(checkpoint_path, torch, device):
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("manifest_sha256") != d.EXPECTED_MANIFEST_SHA or payload.get("test_reads") != 0:
        raise RuntimeError("invalid R1-v2-CRACS checkpoint contract")
    model = RichTemporalSmallTransformer(); model.load_state_dict(payload["model_state_dict"])
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.to(device).eval()
    return model, payload


def encode_samples(model, samples, normalizers, batch_size, torch, device):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = b1.temporal_batch(samples[start:start + batch_size], normalizers, torch, device)
            chunks.append(model.encode(batch).detach().cpu())
    return torch.cat(chunks), torch.tensor([harm_v2_target(sample) for sample in samples], dtype=torch.float32)


def probabilities(head, embeddings, batch_size, torch, device):
    chunks = []; head.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            chunks.append(torch.sigmoid(head(embeddings[start:start + batch_size].to(device))).cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def train_head(head, train_x, train_y, validation_x, validation_y, args, torch, device):
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    allowed_ids = {id(parameter) for parameter in head.parameters()}
    generator = torch.Generator(device="cpu"); generator.manual_seed(args.seed)
    rows, states, stale, best_epoch = [], {}, 0, None; start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train(); order = torch.randperm(len(train_x), generator=generator); losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            logits = head(train_x[indices].to(device)); loss = unweighted_harm_v2_loss(logits, train_y[indices].to(device), torch)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        probability = probabilities(head, validation_x, args.batch_size, torch, device)
        metrics = harm_metrics(probability, validation_y.numpy().astype(bool))
        row = {"synthetic_interaction": LABEL, "epoch": epoch, "train_BCEWithLogitsLoss": float(np.mean(losses)), **metrics,
               "head_checksum": state_sha(head.state_dict())}
        rows.append(row); states[epoch] = copy.deepcopy(head.state_dict())
        selected = phs_select(rows)
        if selected["epoch"] != best_epoch: best_epoch = selected["epoch"]; stale = 0
        else: stale += 1
        row["phs_current_best_epoch"] = best_epoch; row["phs_stale_epochs"] = stale
        print(f"harm-v2 epoch={epoch:02d} train_bce={row['train_BCEWithLogitsLoss']:.5f} val_nll={row['NLL']:.5f} auc={row['AUROC']:.4f} best={best_epoch} stale={stale}", flush=True)
        if stale >= args.patience: break
    selected = phs_select(rows); head.load_state_dict(states[selected["epoch"]]); head.eval()
    return {"head": head, "rows": rows, "selected": selected, "optimizer_ids": optimizer_ids,
            "optimizer_exactly_head": optimizer_ids == allowed_ids, "states": states,
            "epochs_completed": len(rows), "training_time_s": time.perf_counter() - start_time}


def describe_probability(values) -> dict:
    values = np.asarray(values, np.float64)
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            **{f"P{p}": float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90, 95)}}


def subset_row(samples, probability, predicate, dimension, group):
    mask = np.asarray([bool(predicate(sample)) for sample in samples]); target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    result = harm_metrics(probability[mask], target[mask]) if mask.any() else {"candidate_count": 0}
    return {"synthetic_interaction": LABEL, "dimension": dimension, "group": group,
            "episode_count": len({sample.episode_id for sample, keep in zip(samples, mask) if keep}),
            "positive_episode_count": len({sample.episode_id for sample, keep, positive in zip(samples, mask, target) if keep and positive}), **result}


def subtype_rows(samples, probability):
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool); rows = []
    for name, predicate in SUBTYPES.items():
        subtype = np.asarray([bool(predicate(sample)) for sample in samples]); values = probability[subtype]
        comparison = subtype | ~target
        subtype_labels = subtype[comparison]
        enough = int(subtype_labels.sum()) >= 20 and int((~subtype_labels).sum()) >= 20
        auc = binary_auc(probability[comparison], subtype_labels)
        rows.append({"synthetic_interaction": LABEL, "subtype": name, "positive_candidate_count": int(subtype.sum()),
                     **describe_probability(values), "true_harm_negative_comparison_count": int((~target).sum()),
                     "one_vs_harm_v2_negative_AUROC": auc,
                     "Recall_at_0_5_diagnostic": float(np.mean(values >= .5)) if len(values) else None,
                     "sufficient_support": enough, "near_random_warning": bool(enough and auc is not None and auc < .60),
                     "all_subtype_positive_are_harm_v2": bool(np.all(target[subtype]))})
    return rows


def context_rows(samples, probability):
    return [subset_row(samples, probability, lambda sample, name=name: any(str(value).startswith(name) for value in sample.split_metadata["contexts_evaluation_only"]), "context", name) for name in ("C7", "C8", "C9")]


def categorical_rows(samples, probability, dimension, values, getter):
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool); rows = []
    for value in values:
        mask = np.asarray([getter(sample) == value for sample in samples]); metrics = harm_metrics(probability[mask], target[mask])
        rows.append({"synthetic_interaction": LABEL, "dimension": dimension, "group": value,
                     "episode_count": len({sample.episode_id for sample, keep in zip(samples, mask) if keep}),
                     "positive_episode_count": len({sample.episode_id for sample, keep, positive in zip(samples, mask, target) if keep and positive}), **metrics,
                     "true_harm_rate": float(target[mask].mean()), "predicted_harm_mean": float(probability[mask].mean()),
                     "calibration_error_signed": float(probability[mask].mean() - target[mask].mean())})
    return rows


def variance_shortcut(rows, samples, probability, getter):
    groups = defaultdict(list)
    for sample, value in zip(samples, probability): groups[getter(sample)].append(float(value))
    overall_mean = float(np.mean(probability)); overall = float(np.var(probability))
    between = float(sum(len(values) * (float(np.mean(values)) - overall_mean) ** 2 for values in groups.values()) / len(probability))
    return {"group_count": len(groups), "predicted_mean_range": float(max(map(np.mean, groups.values())) - min(map(np.mean, groups.values()))),
            "between_group_variance_fraction": between / max(overall, 1e-12),
            "near_deterministic_shortcut": bool(between / max(overall, 1e-12) >= .90)}


def separation(samples, probability):
    positive = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    safe = np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool)
    pos, neg = probability[positive], probability[safe]
    pooled = math.sqrt((float(pos.var()) + float(neg.var())) / 2); effect = (float(pos.mean()) - float(neg.mean())) / max(pooled, 1e-12)
    low, high = min(probability), max(probability); edges = np.linspace(low, high + 1e-12, 21)
    left, _ = np.histogram(pos, edges, density=False); right, _ = np.histogram(neg, edges, density=False)
    overlap = float(np.minimum(left / len(pos), right / len(neg)).sum())
    labels = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]; values = np.r_[pos, neg]
    return {"positive": describe_probability(pos), "safe_beneficial": describe_probability(neg), "mean_difference": float(pos.mean() - neg.mean()),
            "cohens_d": effect, "histogram_overlap_coefficient_20_bins": overlap,
            "harm_positive_vs_safe_beneficial_AUROC": binary_auc(values, labels),
            "safe_beneficial_false_positive_rate_at_0_5_diagnostic": float(np.mean(neg >= .5))}


def tradeoff_rows(samples, probability):
    groups = defaultdict(list)
    for index, sample in enumerate(samples): groups[sample.episode_id].append(index)
    global_rank = np.argsort(np.argsort(-probability, kind="stable"), kind="stable") + 1; rows = []
    for index, sample in enumerate(samples):
        if not sample.split_metadata["benefit_risk_tradeoff_evaluation_only"]: continue
        episode = groups[sample.episode_id]; episode_order = np.argsort(np.argsort(-probability[episode], kind="stable"), kind="stable") + 1
        rows.append({"synthetic_interaction": LABEL, "sample_id": sample.sample_id, "episode_id": sample.episode_id,
                     "action": ACTION_NAMES[int(sample.split_metadata["candidate_action_id_audit"])], "benefit": sample.targets.benefit,
                     "harm_v2_probability": float(probability[index]), "global_harm_probability_rank": int(global_rank[index]),
                     "within_episode_harm_rank": int(episode_order[episode.index(index)]), "rejected_at_0_5_diagnostic": bool(probability[index] >= .5)})
    return rows


def save_checkpoint(path, head, payload, architecture, selector, gates, torch):
    if not all(gates[name]["passed"] for name in ("Gate_A", "Gate_B", "Gate_C")): return False
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"label": LABEL, "stage": STAGE, "model_state_dict": head.state_dict(), "architecture": architecture,
                "selector": selector, "manifest_sha256": d.EXPECTED_MANIFEST_SHA,
                "backbone_checkpoint_sha256": payload["source_checkpoint_sha256"], "backbone_checksum": payload["backbone_checksum"],
                "test_reads": 0}, path)
    return True


def evaluate_gates(isolation_checks, metrics, baseline, separation_metrics):
    """Apply the fixed, preregistered Stage 1.7E gates without tuning."""
    gate_b_checks = {"AUROC_at_least_0_80": metrics["AUROC"] >= .80,
                     "AUPRC_at_least_prevalence_plus_0_20": metrics["AUPRC"] >= metrics["prevalence"] + .20,
                     "NLL_better_than_constant": metrics["NLL"] < baseline["NLL"],
                     "Brier_better_than_constant": metrics["Brier"] < baseline["Brier"]}
    gate_c_checks = {"positive_mean_above_safe_beneficial": separation_metrics["positive"]["mean"] > separation_metrics["safe_beneficial"]["mean"],
                     "separation_AUROC_at_least_0_80": separation_metrics["harm_positive_vs_safe_beneficial_AUROC"] >= .80}
    return {"Gate_A": {"name": "Training Isolation", "checks": isolation_checks, "passed": all(isolation_checks.values())},
            "Gate_B": {"name": "Independent Harm Learnability", "checks": gate_b_checks, "passed": all(gate_b_checks.values())},
            "Gate_C": {"name": "Semantic Separation", "checks": gate_c_checks, "passed": all(gate_c_checks.values())}}


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7E: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    source_checkpoint_sha = file_sha(args.checkpoint); model, source_payload = load_frozen(args.checkpoint, torch, device)
    frozen_state_before = copy.deepcopy(model.state_dict()); model_before = d.model_sha(model)
    backbone_before = state_sha(frozen_state_before, exclude=("benefit.", "uncertainty.", "harm."))
    benefit_before = state_sha(frozen_state_before, prefixes=("benefit.", "uncertainty."))
    ranking_before = state_sha(frozen_state_before, exclude=("harm.",))
    benefit_behavior_before = None

    episodes = {"train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED),
                "validation": build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)}
    samples = {name: build_v2_temporal_samples(values) for name, values in episodes.items()}
    manifest_contract = d.manifest_contract(args.manifest, samples["train"] + samples["validation"])
    normalizers = source_payload["normalizer"]
    if normalizers["sha256"] != "dc4e412b5313d5b8d96b7ad6521b03e0a7672419c5ba50076ecf002740011d2c": raise RuntimeError("normalizer checksum mismatch")
    train_x, train_y = encode_samples(model, samples["train"], normalizers, args.batch_size, torch, device)
    validation_x, validation_y = encode_samples(model, samples["validation"], normalizers, args.batch_size, torch, device)
    with torch.no_grad(): benefit_behavior_before = model.benefit(validation_x.to(device)).cpu().numpy()

    torch.manual_seed(args.seed); head = IndependentHarmV2Head().to(device)
    old_harm_identical = all(torch.equal(head.state_dict()["linear." + name], model.harm.state_dict()[name]) for name in ("weight", "bias"))
    if old_harm_identical: raise RuntimeError("new harm-v2 head unexpectedly equals historical harm weights")
    training = train_head(head, train_x, train_y, validation_x, validation_y, args, torch, device)
    probability = probabilities(head, validation_x, args.batch_size, torch, device); target = validation_y.numpy().astype(bool)
    metrics = harm_metrics(probability, target); baseline = prevalence_baseline(train_y.numpy().astype(bool), target)

    frozen_state_after = model.state_dict(); model_after = d.model_sha(model)
    backbone_after = state_sha(frozen_state_after, exclude=("benefit.", "uncertainty.", "harm."))
    benefit_after = state_sha(frozen_state_after, prefixes=("benefit.", "uncertainty."))
    ranking_after = state_sha(frozen_state_after, exclude=("harm.",))
    with torch.no_grad(): benefit_behavior_after = model.benefit(validation_x.to(device)).cpu().numpy()

    safe_separation = separation(samples["validation"], probability)
    subtypes = subtype_rows(samples["validation"], probability)
    contexts = context_rows(samples["validation"], probability)
    motions = sorted({sample.split_metadata["motion_type_evaluation_only"] for sample in samples["validation"]})
    motion_rows = categorical_rows(samples["validation"], probability, "motion", motions, lambda sample: sample.split_metadata["motion_type_evaluation_only"])
    actions = sorted({int(sample.split_metadata["candidate_action_id_audit"]) for sample in samples["validation"]})
    action_rows = categorical_rows(samples["validation"], probability, "action", actions, lambda sample: int(sample.split_metadata["candidate_action_id_audit"]))
    for row in action_rows: row["action_name"] = ACTION_NAMES[int(row["group"])]
    profiles = sorted({int(sample.split_metadata["person_profile_id"]) for sample in samples["validation"]})
    profile_rows = categorical_rows(samples["validation"], probability, "profile_audit_only", profiles, lambda sample: int(sample.split_metadata["person_profile_id"]))
    tradeoffs = tradeoff_rows(samples["validation"], probability)
    safe_mask = np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples["validation"]], bool)
    safe_row = {"synthetic_interaction": LABEL, "group": "safe_beneficial", **describe_probability(probability[safe_mask]),
                "false_positive_rate_at_0_5_diagnostic": float(np.mean(probability[safe_mask] >= .5)),
                "episode_count": len({sample.episode_id for sample, keep in zip(samples["validation"], safe_mask) if keep})}
    positive_row = {"synthetic_interaction": LABEL, "group": "harm_v2_positive", **describe_probability(probability[target])}

    gate_a_checks = {"only_harm_v2_head_trainable": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad) == 129,
                     "optimizer_exactly_harm_v2_head": training["optimizer_exactly_head"], "backbone_checksum_unchanged": backbone_before == backbone_after,
                     "benefit_head_checksum_unchanged": benefit_before == benefit_after, "ranking_related_checksum_unchanged": ranking_before == ranking_after,
                     "entire_frozen_model_checksum_unchanged": model_before == model_after,
                     "benefit_behavior_checksum_unchanged": array_sha(benefit_behavior_before) == array_sha(benefit_behavior_after),
                     "deprecated_harm_absent_from_loss": "DEPRECATED" not in inspect.getsource(unweighted_harm_v2_loss)}
    gates = evaluate_gates(gate_a_checks, metrics, baseline, safe_separation)
    subtype_warnings = [row["subtype"] for row in subtypes if row["near_random_warning"]]

    architecture = {"label": LABEL, **head.architecture_audit(), "old_harm_structure": "Linear(128,1)",
                    "old_harm_parameter_count": sum(parameter.numel() for parameter in model.harm.parameters()),
                    "new_parameters_equal_old_weights": old_harm_identical, "fresh_initialization_seed": args.seed}
    phs = {"label": LABEL, "name": "Probabilistic Harm Selector v1", "short_name": "PHS-v1", "validation_only": True, "test_reads": 0,
           "selection_order": ["minimum Validation NLL", "minimum Brier", "maximum AUROC", "earlier epoch"],
           "tie_tolerance": 1e-12, "selected_epoch": training["selected"]["epoch"], "selected_metrics": training["selected"]}
    config = {"label": LABEL, "stage": STAGE, "seed": args.seed, "optimizer": "AdamW", "learning_rate": args.learning_rate,
              "weight_decay": 1e-3, "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
              "loss": "unweighted BCEWithLogitsLoss", "class_weighting": False, "focal_loss": False, "oversampling": False,
              "undersampling": False, "benefit_loss_contribution": 0, "ranking_loss_contribution": 0,
              "deprecated_harm_loss_contribution": 0, "hyperparameter_search": False, "threshold_0_5_diagnostic_only": True, "test_reads": 0}
    freeze_audit = {"label": LABEL, "frozen_model_requires_grad_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                    "new_head_trainable_parameter_count": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
                    "optimizer_parameter_count": sum(parameter.numel() for parameter in head.parameters() if id(parameter) in training["optimizer_ids"]),
                    "checksums_before": {"entire_model": model_before, "backbone": backbone_before, "benefit_head": benefit_before, "ranking_related": ranking_before,
                                         "benefit_behavior": array_sha(benefit_behavior_before)},
                    "checksums_after": {"entire_model": model_after, "backbone": backbone_after, "benefit_head": benefit_after, "ranking_related": ranking_after,
                                        "benefit_behavior": array_sha(benefit_behavior_after)}, "gate_a": gates["Gate_A"]}
    frozen_contract = {"label": LABEL, "source_checkpoint": str(args.checkpoint), "source_checkpoint_sha256": source_checkpoint_sha,
                       "manifest_sha256": file_sha(args.manifest), "expected_manifest_sha256": d.EXPECTED_MANIFEST_SHA,
                       "normalizer_sha256": normalizers["sha256"], "test_candidate_reads": 0, "test_trajectory_reads": 0,
                       "test_benefit_reads": 0, "test_harm_reads": 0, "test_reads": 0, "manifest_contract_passed": manifest_contract["passed"],
                       "backbone_checksum": backbone_before, "source_checkpoint_unchanged": source_checkpoint_sha == file_sha(args.checkpoint)}
    baseline_record = {"label": LABEL, "definition": "constant probability p = TRAIN harm_v2 prevalence", **baseline}
    global_record = {"label": LABEL, "split": "validation", **metrics, "threshold_0_5_is_diagnostic_only": True}
    shortcut = {"action": variance_shortcut(action_rows, samples["validation"], probability, lambda sample: int(sample.split_metadata["candidate_action_id_audit"])),
                "profile": variance_shortcut(profile_rows, samples["validation"], probability, lambda sample: int(sample.split_metadata["person_profile_id"])),
                "motion": variance_shortcut(motion_rows, samples["validation"], probability, lambda sample: sample.split_metadata["motion_type_evaluation_only"]),
                "profile_id_in_runtime_model": False}

    tradeoff_probability = np.asarray([row["harm_v2_probability"] for row in tradeoffs], np.float64)
    tradeoff_summary = {"candidate_count": len(tradeoffs), "episode_count": len({row["episode_id"] for row in tradeoffs}),
                        **describe_probability(tradeoff_probability),
                        "rejected_at_0_5_diagnostic_count": sum(row["rejected_at_0_5_diagnostic"] for row in tradeoffs),
                        "threshold_0_5_is_diagnostic_only": True}
    stop_audit = next(row for row in motion_rows if row["group"] == "stop")
    stop_audit_summary = {"candidate_count": stop_audit["candidate_count"], "positive_count": stop_audit["positive_count"],
                          "true_harm_rate": stop_audit["true_harm_rate"], "predicted_harm_mean": stop_audit["predicted_harm_mean"],
                          "AUROC": stop_audit["AUROC"], "AUPRC": stop_audit["AUPRC"], "Recall_at_0_5_diagnostic": stop_audit["Recall_at_0_5"],
                          "simple_high_probability_stop_shortcut": bool(stop_audit["predicted_harm_mean"] >= .5),
                          "underestimation_gap": float(stop_audit["predicted_harm_mean"] - stop_audit["true_harm_rate"])}

    gates["all_passed"] = all(item["passed"] for item in gates.values())
    output_checkpoint = args.output_dir / "checkpoints" / "harm_v2_head_best.pt"
    checkpoint_payload = {"source_checkpoint_sha256": source_checkpoint_sha, "backbone_checksum": backbone_before}
    checkpoint_written = save_checkpoint(output_checkpoint, head, checkpoint_payload, architecture, phs, gates, torch)
    gates["checkpoint_frozen"] = checkpoint_written
    summary = {"label": LABEL, "stage": STAGE, "development_validation_only": True, "test_reads": 0,
               "train_candidates": len(train_y), "validation_candidates": len(validation_y), "train_positive_count": int(train_y.sum()),
               "validation_positive_count": int(validation_y.sum()), "selected_epoch": phs["selected_epoch"], "global_metrics": global_record,
               "prevalence_baseline": baseline_record, "safe_beneficial_separation": safe_separation,
               "benefit_risk_tradeoff": tradeoff_summary,
               "GT_unsafe": next(row for row in subtypes if row["subtype"] == "GT_UNSAFE"), "subtype_generalization_warnings": subtype_warnings,
               "stop_motion_audit": stop_audit_summary, "shortcuts": shortcut, "gates": gates, "harm_v2_head_frozen": checkpoint_written,
               "formal_threshold_calibration_performed": False, "formal_decision_gate_performed": False,
               "ready_for_phase5b17f": gates["all_passed"], "phase5b17f_started": False}

    io.write_json(args.output_dir / "frozen_contract.json", frozen_contract); io.write_json(args.output_dir / "harm_head_architecture.json", architecture)
    io.write_json(args.output_dir / "training_config.json", config); io.write_csv(args.output_dir / "training_curve.csv", training["rows"])
    io.write_json(args.output_dir / "phs_selector.json", phs); io.write_json(args.output_dir / "prevalence_baseline.json", baseline_record)
    io.write_json(args.output_dir / "global_harm_metrics.json", global_record); io.write_csv(args.output_dir / "safe_beneficial_separation.csv", [positive_row, safe_row, {"synthetic_interaction": LABEL, "group": "separation_metrics", **safe_separation}])
    io.write_csv(args.output_dir / "benefit_risk_tradeoff.csv", tradeoffs); io.write_csv(args.output_dir / "by_harm_subtype.csv", subtypes)
    io.write_csv(args.output_dir / "by_context.csv", contexts); io.write_csv(args.output_dir / "by_motion.csv", motion_rows)
    io.write_csv(args.output_dir / "by_action.csv", action_rows); io.write_csv(args.output_dir / "by_profile.csv", profile_rows)
    io.write_json(args.output_dir / "parameter_freeze_audit.json", freeze_audit); io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()
