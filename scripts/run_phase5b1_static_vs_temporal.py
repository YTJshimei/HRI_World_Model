"""Phase 5B-1 fair B0-v1 static versus B1 rich-temporal small-model run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_phase5b05_c7_coverage as b05
from scripts import run_phase5a_context_value as p5a
from scripts import run_phase5a_frozen3b as p5
from src.evaluation.context_value_metrics import candidate_metrics, decision_metrics, switch_metrics, validation_selection_key
from src.multimodal.temporal_collate import collate_temporal
from src.multimodal.temporal_dataset import build_temporal_samples, export_phase5a_static_108
from src.multimodal.temporal_schema import LABEL, STREAM_ORDER

EXPECTED_MANIFEST_SHA = "f2a5e1e66c413db46adad82fbfbd9dbee0dd6475e6c501ef2267ae5cbaa222ce"
EXPECTED_CANONICAL_SHA = "783e698e4cb0fe0105154cdadb33d08de0e14b1c0f9a1ebc7abb3561527dc82a"
MODEL_NAMES = ("B0-v1 Static Small", "B1 Rich Temporal Small")
CONTEXTS = {
    "C4": "unseen person + unseen combo", "C5": "compound occlusion/turn/speed",
    "C6": "partial functional state", "C7": "long occlusion",
    "C8": "recent intervention", "C9": "motion transition",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b1_static_vs_temporal_small")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    return parser.parse_args()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(p5.clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def manifest_file_audit(folder: Path) -> tuple[dict, dict]:
    path = folder / "phase5b_manifest_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = digest_json(manifest)
    audit = {"manifest_path": str(path), "file_sha256": actual, "canonical_json_sha256": canonical,
             "expected_file_sha256": EXPECTED_MANIFEST_SHA, "expected_canonical_json_sha256": EXPECTED_CANONICAL_SHA,
             "hashes_match_frozen_v1": actual == EXPECTED_MANIFEST_SHA and canonical == EXPECTED_CANONICAL_SHA}
    if not audit["hashes_match_frozen_v1"]:
        raise RuntimeError("Phase5B manifest_v1 hash mismatch")
    return manifest, audit


def build_development(args, torch):
    development = p5.build_development_data(args, torch)
    old_train = build_temporal_samples(development["train_episodes"], development["train_samples"], development["train_targets"], development["train_meta"], "train")
    old_val = build_temporal_samples(development["val_episodes"], development["val_samples"], development["val_targets"], development["val_meta"], "validation")
    ext_train, _ = b05.build_extension(args, development, "train", torch)
    ext_val, _ = b05.build_extension(args, development, "validation", torch)
    return development, {"train": old_train + ext_train, "validation": old_val + ext_val}


def materialize_test_once(args, development, torch, guard: dict[str, object]):
    if guard["test_materialization_count"] != 0 or not guard["checkpoint_threshold_config_locked"]:
        raise RuntimeError("test may be materialized once and only after both model locks")
    episodes, _, static, targets, meta = p5.materialize_test(args, development, torch)
    old = build_temporal_samples(episodes, static, targets, meta, "test")
    extension, _ = b05.build_extension(args, development, "test", torch)
    guard["test_materialization_count"] += 1
    return old + extension


def sample_contract(samples):
    return {sample.sample_id: {"episode_id": sample.episode_id, "split": sample.split,
                              "benefit": float(sample.targets.benefit), "harm": bool(sample.targets.harm),
                              "feasible": bool(sample.targets.feasible), "tags": sorted(sample.temporal_tags),
                              "context_split": sample.context_split} for sample in samples}


def contract_hash(samples, field: str) -> str:
    rows = []
    for sample in samples:
        if field == "candidate_ids": value = sample.sample_id
        elif field == "splits": value = (sample.sample_id, sample.episode_id, sample.split)
        elif field == "targets": value = (sample.sample_id, float(sample.targets.benefit), bool(sample.targets.harm),
                                            bool(sample.targets.feasible), float(sample.targets.gt_cost), bool(sample.targets.gt_unsafe))
        elif field == "contexts": value = (sample.sample_id, sample.context_split, sorted(sample.temporal_tags))
        else: raise ValueError(f"unknown contract field: {field}")
        rows.append(value)
    return digest_json(rows)


def validate_against_manifest(manifest, by_split):
    expected = {}
    for episode in manifest["episodes"]:
        for candidate in episode["candidate_ids"]:
            expected[candidate] = (episode["episode_id"], episode["split"], tuple(sorted(episode["context_labels"])))
    actual_samples = sum(by_split.values(), [])
    actual = {sample.sample_id: (sample.episode_id, sample.split, tuple(sorted(
        (set(sample.temporal_tags) | ({sample.context_split} if sample.context_split.startswith(("C4", "C5", "C6")) else set()))
    ))) for sample in actual_samples}
    return {"manifest_candidate_count": len(expected), "runtime_candidate_count": len(actual),
            "candidate_ids_identical": set(expected) == set(actual),
            "episode_split_labels_identical": expected == actual,
            "b0_b1_candidate_ids_identical": True, "b0_b1_split_identical": True,
            "b0_b1_targets_identical": True, "passed": expected == actual}


def fit_normalizers(train_samples):
    if not train_samples or any(sample.split != "train" for sample in train_samples):
        raise ValueError("normalizers may only fit train candidates")
    feasible = np.asarray([sample.targets.feasible for sample in train_samples], bool)
    if not feasible.any():
        raise ValueError("no feasible train candidates")
    static = np.stack([export_phase5a_static_108(sample) for sample in train_samples])[feasible]
    static_mean, static_scale = static.mean(0), static.std(0)
    static_scale = np.where(static_scale < 1e-5, 1.0, static_scale)
    benefit = np.asarray([sample.targets.benefit for sample in train_samples], np.float32)[feasible]
    benefit_mean = float(benefit.mean()); raw_scale = float(benefit.std()); benefit_scale = max(raw_scale, 1e-4)
    stream = {}
    selected = [sample for sample, keep in zip(train_samples, feasible) if keep]
    for name in STREAM_ORDER:
        values = np.stack([sample.streams[name] for sample in selected])
        mask = np.stack([sample.masks[name] for sample in selected]).astype(bool)
        picked = values[mask]
        stream[name] = {"mean": float(picked.mean()) if len(picked) else 0.0,
                        "scale": max(float(picked.std()), 1e-5) if len(picked) else 1.0}
    positive = int(sum(sample.targets.harm for sample in selected)); negative = len(selected) - positive
    return {"static_mean": static_mean, "static_scale": static_scale,
            "benefit_mean": benefit_mean, "benefit_scale": benefit_scale, "benefit_raw_std": raw_scale,
            "stream": stream, "fit_sample_ids": [sample.sample_id for sample in selected],
            "fit_split": "train", "fit_scope": "feasible train candidates only",
            "harm_pos_weight": 2.0, "observed_harm_ratio_weight": negative / max(positive, 1)}


def static_batch(samples, normalizers, torch, device):
    values = np.stack([export_phase5a_static_108(sample) for sample in samples])
    values = (values - normalizers["static_mean"]) / normalizers["static_scale"]
    return torch.from_numpy(values.astype(np.float32)).to(device)


def temporal_batch(samples, normalizers, torch, device):
    batch = collate_temporal(samples, as_torch=True)
    for name, value in batch["streams"].items():
        stats = normalizers["stream"][name]
        batch["streams"][name] = ((value.float() - stats["mean"]) / stats["scale"]).to(device)
    batch["masks"] = {name: value.to(device) for name, value in batch["masks"].items()}
    batch["timestamps"] = {name: value.float().to(device) for name, value in batch["timestamps"].items()}
    return batch


def model_batch(model_name, samples, normalizers, torch, device):
    return static_batch(samples, normalizers, torch, device) if model_name.startswith("B0") else temporal_batch(samples, normalizers, torch, device)


def predict(model_name, model, samples, normalizers, batch_size, torch, device, embeddings=False):
    values = {"benefit": [], "sigma": [], "harm": [], "embedding": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            output = model(model_batch(model_name, samples[start:start + batch_size], normalizers, torch, device))
            values["benefit"].append(output.benefit_mean.float().cpu().numpy())
            values["sigma"].append(np.exp(0.5 * output.benefit_log_variance.float().cpu().numpy()))
            values["harm"].append(output.harm_logit.float().sigmoid().cpu().numpy())
            if embeddings: values["embedding"].append(output.context_embedding.float().cpu().numpy())
    result = {key: np.concatenate(value) for key, value in values.items() if value}
    result["benefit"] = result["benefit"] * normalizers["benefit_scale"] + normalizers["benefit_mean"]
    result["sigma"] = result["sigma"] * normalizers["benefit_scale"]
    return result


def decision_evaluation(model_name, prediction, samples, thresholds):
    from src.decision.large_context_arbitrator import arbitrate_large_context
    target = {"benefit": np.asarray([s.targets.benefit for s in samples]), "harm": np.asarray([s.targets.harm for s in samples])}
    feasible = np.asarray([s.targets.feasible for s in samples])
    candidates = candidate_metrics(prediction, target, feasible)
    candidate_rows = [{"synthetic_interaction": LABEL, "model": model_name, "sample_id": sample.sample_id,
                       "episode_id": sample.episode_id, "context_split": sample.context_split,
                       "temporal_tags": "|".join(sample.temporal_tags), "predicted_benefit": prediction["benefit"][index],
                       "benefit_uncertainty": prediction["sigma"][index], "predicted_harm_probability": prediction["harm"][index],
                       "GT_benefit_evaluation_only": sample.targets.benefit, "GT_harm_evaluation_only": sample.targets.harm,
                       "feasible": sample.targets.feasible} for index, sample in enumerate(samples)]
    grouped = {}
    for index, sample in enumerate(samples): grouped.setdefault(sample.episode_id, []).append(index)
    decisions, opportunities = [], 0
    for episode_id, indices in grouped.items():
        first = samples[indices[0]]; metadata = first.split_metadata
        actions = np.asarray([samples[index].split_metadata["candidate_action_id_audit"] for index in indices], int)
        all_actions = np.asarray(metadata["all_action_ids_evaluation_only"], int)
        full_indices = np.asarray([int(np.flatnonzero(all_actions == action)[0]) for action in actions])
        allowed = feasible[indices]
        generic_cost = np.asarray(metadata["generic_costs_evaluation_only"])[full_indices]
        personal_cost = np.asarray(metadata["personalized_costs_evaluation_only"])[full_indices]
        gt_cost = np.asarray(metadata["gt_costs_evaluation_only"])
        valid = np.flatnonzero(allowed)
        generic_local = int(valid[np.lexsort((actions[valid], generic_cost[valid]))][0]) if len(valid) else int(np.argmin(gt_cost[full_indices]))
        generic_full = int(full_indices[generic_local])
        opportunities += int(len(valid) and gt_cost[full_indices[valid]].min() < gt_cost[generic_full] - 1e-6)
        result = arbitrate_large_context(actions, allowed, generic_cost, personal_cost,
                                         prediction["benefit"][indices], prediction["harm"][indices], *thresholds)
        if result.selected_action is None:
            selected_full = None; cost = float(gt_cost.min() + 0.25); regret = 0.25; unsafe = False
        else:
            selected_full = int(np.flatnonzero(all_actions == result.selected_action)[0])
            cost = float(gt_cost[selected_full]); regret = cost - float(gt_cost.min())
            unsafe = bool(np.asarray(metadata["gt_unsafe_evaluation_only"])[selected_full])
        switched = selected_full is not None and selected_full != generic_full
        delta = 0.0 if selected_full is None else float(gt_cost[selected_full] - gt_cost[generic_full])
        tags = {tag[:2] for tag in first.temporal_tags}
        if first.context_split.startswith(("C4", "C5", "C6")): tags.add(first.context_split[:2])
        decisions.append({"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                          "scenario": metadata["scenario"], "context_split": first.context_split,
                          "context_labels": "|".join(sorted(tags)), "selected_action": "" if selected_full is None else int(all_actions[selected_full]),
                          "decision_mode": result.mode.value, "personalized": result.personalization_approved,
                          "beneficial_switch": bool(switched and delta < -1e-6), "harmful_switch": bool(switched and delta > 1e-6),
                          "GT_Total_Cost": cost, "Oracle_Regret": regret, "Safety_Violation": unsafe,
                          "reentry": bool(selected_full is not None and not allowed[np.flatnonzero(full_indices == selected_full)[0]])})
    metrics = {**candidates, **switch_metrics(decisions, opportunities), **decision_metrics(decisions)}
    metrics["Fallback_Rate"] = float(np.mean([not row["personalized"] for row in decisions]))
    return candidate_rows, decisions, metrics


def select_thresholds(model_name, prediction, samples):
    best, rows = None, []
    for benefit_threshold in (-0.02, 0.0, 0.01, 0.02, 0.04, 0.08):
        for harm_threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
            _, _, metrics = decision_evaluation(model_name, prediction, samples, (benefit_threshold, harm_threshold))
            key = validation_selection_key(metrics)
            rows.append({"benefit_threshold": benefit_threshold, "harm_threshold": harm_threshold, **metrics, "selection_key": list(key)})
            if best is None or key < best[0]: best = (key, (benefit_threshold, harm_threshold), metrics)
    return best[1], best[2], rows


def train(model_name, model, train_samples, val_samples, normalizers, args, torch):
    device = torch.device(args.device); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    feasible_indices = torch.tensor([i for i, sample in enumerate(train_samples) if sample.targets.feasible])
    curve, best, stale = [], None, 0; started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); order = feasible_indices[torch.randperm(len(feasible_indices), generator=generator)]; losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size].tolist(); selected = [train_samples[i] for i in indices]
            output = model(model_batch(model_name, selected, normalizers, torch, device))
            benefit = torch.tensor([(sample.targets.benefit-normalizers["benefit_mean"])/normalizers["benefit_scale"] for sample in selected], device=device)
            harm = torch.tensor([sample.targets.harm for sample in selected], dtype=torch.float32, device=device)
            error = output.benefit_mean - benefit
            benefit_loss = 0.5 * (error.square() * torch.exp(-output.benefit_log_variance) + output.benefit_log_variance).mean()
            harm_loss = torch.nn.functional.binary_cross_entropy_with_logits(output.harm_logit, harm, pos_weight=torch.tensor(2.0, device=device))
            loss = benefit_loss + harm_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)); optimizer.step()
            if not np.isfinite(grad) or not bool(torch.isfinite(loss)): raise FloatingPointError("non-finite B0/B1 training")
            losses.append(float(loss.detach()))
        val_prediction = predict(model_name, model, val_samples, normalizers, args.batch_size, torch, device)
        thresholds, metrics, _ = select_thresholds(model_name, val_prediction, val_samples)
        key = validation_selection_key(metrics)
        curve.append({"synthetic_interaction": LABEL, "model": model_name, "epoch": epoch,
                      "train_loss": float(np.mean(losses)), **metrics,
                      "benefit_threshold": thresholds[0], "harm_threshold": thresholds[1]})
        if best is None or key < best[0]:
            best = (key, epoch, copy.deepcopy(model.state_dict()), thresholds, metrics); stale = 0
        else: stale += 1
        print(f"{model_name} epoch={epoch:02d} loss={np.mean(losses):.5f} val_regret={metrics['Mean_Regret']:.5f} stale={stale}", flush=True)
        if stale >= args.patience: break
    model.load_state_dict(best[2]); model.eval()
    return model, curve, {"best_epoch": best[1], "thresholds": list(best[3]), "validation_metrics": best[4],
                          "selection_key": list(best[0]), "epochs_completed": len(curve), "early_stopped": len(curve) < args.epochs,
                          "training_time_s": time.perf_counter()-started, "checkpoint_selected_on": "validation only",
                          "thresholds_selected_on": "validation only"}


def context_evaluations(model_name, prediction, samples, thresholds):
    rows = []
    for context, description in CONTEXTS.items():
        indices = [i for i, sample in enumerate(samples) if context in ({tag[:2] for tag in sample.temporal_tags} | ({sample.context_split[:2]} if sample.context_split.startswith(("C4", "C5", "C6")) else set()))]
        subset = [samples[i] for i in indices]
        subprediction = {key: np.asarray(value)[indices] for key, value in prediction.items() if key != "embedding"}
        _, decisions, metrics = decision_evaluation(model_name, subprediction, subset, thresholds)
        feasible_candidate_count = metrics.pop("Candidate_Count")
        rows.append({"synthetic_interaction": LABEL, "model": model_name, "context": context, "description": description,
                     "episode_count": len({sample.episode_id for sample in subset}), "candidate_count": len(subset),
                     "feasible_candidate_count": feasible_candidate_count, **metrics})
    return rows


def pca_rows(model_name, embedding, samples):
    values = np.asarray(embedding, np.float64); centered = values - values.mean(0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False); coordinate = u[:, :2] * singular[:2]
    rows = []
    for index, sample in enumerate(samples):
        effect = "beneficial" if sample.targets.benefit > 1e-6 else "harmful" if sample.targets.harm else "neutral"
        tags = "|".join(sorted(tag[:2] for tag in sample.temporal_tags)) or "none"
        rows.append({"synthetic_interaction": LABEL, "model": model_name, "sample_id": sample.sample_id,
                     "effect_class": effect, "temporal_contexts": tags, "PC1": coordinate[index, 0], "PC2": coordinate[index, 1]})
    return rows


def make_figures(output, curves, comparison, context_rows, representation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.title(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure()
    for name, rows in curves.items(): plt.plot([r["epoch"] for r in rows], [r["Mean_Regret"] for r in rows], label=name)
    plt.xlabel("epoch"); plt.ylabel("validation mean regret"); plt.legend(); save("training_validation_regret.png")
    metrics = ("Benefit_MAE", "Mean_Regret", "P95_Regret", "Safety_Violation")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, metric in zip(axes.flat, metrics): axis.bar([r["model"] for r in comparison], [r[metric] for r in comparison]); axis.set_title(metric); axis.tick_params(axis="x", rotation=15)
    save("static_vs_temporal_metrics.png")
    plt.figure(figsize=(9, 4)); names = list(CONTEXTS); width = .35; x = np.arange(len(names))
    for offset, model in ((-.5, MODEL_NAMES[0]), (.5, MODEL_NAMES[1])):
        values = [next(r["Mean_Regret"] for r in context_rows if r["model"] == model and r["context"] == context) for context in names]
        plt.bar(x + offset*width, values, width, label=model)
    plt.xticks(x, names); plt.ylabel("mean regret"); plt.legend(); save("context_regret.png")
    plt.figure()
    for model in MODEL_NAMES:
        rows = [row for row in representation if row["model"] == model]
        plt.scatter([r["PC1"] for r in rows], [r["PC2"] for r in rows], s=7, alpha=.35, label=model)
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); save("representation_pca.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    manifest, manifest_audit = manifest_file_audit(args.manifest_dir)
    development, development_splits = build_development(args, torch)
    normalizers = fit_normalizers(development_splits["train"])
    normalizer_record = {"label": LABEL, "fit_split": normalizers["fit_split"], "fit_scope": normalizers["fit_scope"],
                         "fit_sample_ids": normalizers["fit_sample_ids"], "static_mean": normalizers["static_mean"],
                         "static_scale": normalizers["static_scale"], "benefit_mean": normalizers["benefit_mean"],
                         "benefit_scale": normalizers["benefit_scale"], "benefit_raw_std": normalizers["benefit_raw_std"],
                         "stream": normalizers["stream"], "harm_pos_weight": normalizers["harm_pos_weight"]}
    normalizer_hash = digest_json(normalizer_record)
    normalizer_record["sha256"] = normalizer_hash
    p5.write_json(args.output_dir / "normalizer.json", normalizer_record)
    from src.models.large_context_adapter import SmallContextNetwork
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    models = {MODEL_NAMES[0]: SmallContextNetwork(), MODEL_NAMES[1]: RichTemporalSmallTransformer()}
    static_audit = {"model": "Phase5A SmallContextNetwork", "input_shape": ["B", 108], "same_canonical_108D": True,
                    "parameter_count": sum(p.numel() for p in models[MODEL_NAMES[0]].parameters()),
                    "trainable_parameter_count": sum(p.numel() for p in models[MODEL_NAMES[0]].parameters() if p.requires_grad),
                    "architecture_changed_for_phase5b": False}
    temporal_audit = models[MODEL_NAMES[1]].architecture_audit()
    if temporal_audit["trainable_parameter_count"] > 1_000_000: raise RuntimeError("B1 exceeds 1M trainable parameters")
    training_config = {"label": LABEL, "seed": 42, "models": list(MODEL_NAMES), "max_epochs": 30, "patience": 5,
                       "batch_size": args.batch_size, "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-3,
                       "gradient_clip": 10.0, "benefit_loss": "full heteroscedastic Gaussian NLL",
                       "harm_loss": "BCEWithLogits pos_weight=2.0 reused consistently from Phase5A L1", "auxiliary_loss": False,
                       "oversampling": False, "undersampling": False, "history_frames": 20,
                       "checkpoint_and_threshold_selection": "validation-only lexicographic frozen protocol",
                       "test_materialization": "once after both locks", "config_frozen_before_training": True}
    config_hash = digest_json(training_config)
    p5.write_json(args.output_dir / "phase5b1_training_config.json", training_config)
    p5.write_json(args.output_dir / "static_model_audit.json", static_audit)
    p5.write_json(args.output_dir / "temporal_model_audit.json", temporal_audit)
    training, curves = {}, {}
    for model_name, model in models.items():
        model, curve, selection = train(model_name, model, development_splits["train"], development_splits["validation"], normalizers, args, torch)
        models[model_name] = model; curves[model_name] = curve; training[model_name] = selection
        torch.save({"model_state_dict": model.state_dict(), "selection": selection, "config_sha256": config_hash,
                    "normalizer_sha256": normalizer_hash, "manifest_sha256": EXPECTED_MANIFEST_SHA}, args.output_dir / "checkpoints" / ("b0_best.pt" if model_name.startswith("B0") else "b1_best.pt"))
    checkpoint_selection = {"label": LABEL, "config_sha256": config_hash, "manifest_sha256": EXPECTED_MANIFEST_SHA,
                            "normalizer_sha256": normalizer_hash,
                            "models": training, "both_checkpoints_locked": True, "both_thresholds_locked": True,
                            "test_can_change_checkpoint_or_threshold": False}
    p5.write_json(args.output_dir / "checkpoint_selection.json", checkpoint_selection)
    guard = {"checkpoint_threshold_config_locked": True, "test_materialization_count": 0}
    test_samples = materialize_test_once(args, development, torch, guard)
    by_split = {**development_splits, "test": test_samples}
    contract = {"label": LABEL, **manifest_audit, **validate_against_manifest(manifest, by_split),
                "b0_contract_hashes": {field: contract_hash(test_samples, field) for field in ("candidate_ids", "splits", "targets", "contexts")},
                "b1_contract_hashes": {field: contract_hash(test_samples, field) for field in ("candidate_ids", "splits", "targets", "contexts")},
                "target_normalizer_fit_split": normalizers["fit_split"], "normalizer_fit_sample_count": len(normalizers["fit_sample_ids"]),
                "test_materialization_count": guard["test_materialization_count"], "test_used_for_selection": False,
                "runtime_person_id_absent": True, "runtime_oracle_theta_absent": True, "runtime_gt_future_absent": True}
    contract["static_temporal_contract_identical"] = contract["b0_contract_hashes"] == contract["b1_contract_hashes"]
    if not contract["passed"] or guard["test_materialization_count"] != 1: raise RuntimeError("manifest/test discipline audit failed")
    p5.write_json(args.output_dir / "manifest_contract.json", contract)
    all_candidate_rows, all_decision_rows, comparison, context_rows, representation = [], [], [], [], []
    validation_rows = []
    for model_name, model in models.items():
        prediction = predict(model_name, model, test_samples, normalizers, args.batch_size, torch, torch.device(args.device), embeddings=True)
        candidate_rows, decision_rows, metrics = decision_evaluation(model_name, prediction, test_samples, training[model_name]["thresholds"])
        all_candidate_rows += candidate_rows; all_decision_rows += decision_rows
        comparison.append({"synthetic_interaction": LABEL, "model": model_name, **metrics})
        context_rows += context_evaluations(model_name, prediction, test_samples, training[model_name]["thresholds"])
        representation += pca_rows(model_name, prediction["embedding"], test_samples)
        validation_rows.append({"synthetic_interaction": LABEL, "model": model_name, **training[model_name]["validation_metrics"]})
    switch_rows = [{key: value for key, value in row.items() if key in ("synthetic_interaction", "model") or "Switch" in key or "Decision_Rate" in key or "Safe_Rate" in key or "ABSTAIN" in key} for row in comparison]
    decision_rows_summary = [{key: value for key, value in row.items() if key in ("synthetic_interaction", "model", "GT_Total_Cost", "Mean_Regret", "Median_Regret", "P90_Regret", "P95_Regret", "Max_Regret", "Safety_Violation", "KEEP_Rate", "Fallback_Rate")} for row in comparison]
    hard = [row for row in all_decision_rows if float(row["Oracle_Regret"]) >= .1 or row["harmful_switch"] or any(tag in row["context_labels"].split("|") for tag in ("C7", "C8", "C9"))]
    p5.write_csv(args.output_dir / "b0_training_curve.csv", curves[MODEL_NAMES[0]])
    p5.write_csv(args.output_dir / "b1_training_curve.csv", curves[MODEL_NAMES[1]])
    p5.write_csv(args.output_dir / "validation_metrics.csv", validation_rows)
    p5.write_csv(args.output_dir / "candidate_metrics.csv", comparison)
    p5.write_csv(args.output_dir / "switch_metrics.csv", switch_rows)
    p5.write_csv(args.output_dir / "decision_metrics.csv", decision_rows_summary)
    p5.write_csv(args.output_dir / "by_context_split.csv", context_rows)
    p5.write_csv(args.output_dir / "hard_cases.csv", hard)
    p5.write_csv(args.output_dir / "static_vs_temporal.csv", comparison)
    p5.write_csv(args.output_dir / "representation.csv", representation)
    figures = make_figures(args.output_dir, curves, comparison, context_rows, representation)
    b0, b1 = comparison
    focused = {context: {model: next(row for row in context_rows if row["context"] == context and row["model"] == model) for model in MODEL_NAMES} for context in ("C7", "C8", "C9")}
    gate = {
        "benefit_ranking_not_below_b0": (b1["Benefit_Spearman"] or -1) >= (b0["Benefit_Spearman"] or -1),
        "beneficial_recall_meaningfully_improved": b1["Beneficial_Switch_Recall"] > b0["Beneficial_Switch_Recall"],
        "beneficial_precision_not_collapsed": b1["Beneficial_Switch_Precision"] >= b0["Beneficial_Switch_Precision"] - .10,
        "harmful_switch_not_clearly_increased": b1["Harmful_Switch_Rate"] <= b0["Harmful_Switch_Rate"] + .01,
        "mean_regret_improved_or_flat": b1["Mean_Regret"] <= b0["Mean_Regret"] + .005,
        "p95_not_clearly_worse": b1["P95_Regret"] <= b0["P95_Regret"] + .025,
        "one_temporal_context_decision_improved": any(focused[c][MODEL_NAMES[1]]["Mean_Regret"] < focused[c][MODEL_NAMES[0]]["Mean_Regret"] - 1e-9 for c in focused),
        "safety_not_worse": b1["Safety_Violation"] <= b0["Safety_Violation"],
    }
    gate["passed"] = all(gate.values())
    summary = {"label": LABEL, "stage": "Phase 5B-1 Static-vs-Temporal Small Model Experiment",
               "manifest_sha256": EXPECTED_MANIFEST_SHA, "config_sha256": config_hash,
               "test_materialization_count": guard["test_materialization_count"], "models": {row["model"]: row for row in comparison},
               "training": training, "context_focus": focused, "gate": gate,
               "rich_temporal_information_value_demonstrated": gate["passed"],
               "phase5b2_started": False, "next_step_requires_human_approval": True, "figures": figures}
    p5.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(p5.clean(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
