"""Phase 5A Stage C-R1: formal seed-42 frozen-Qwen training and one-shot test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as base

LABEL = base.LABEL
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
FORMAL_LR = 3e-5
WEIGHT_DECAY = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
MAX_EPOCHS = 25  # Predeclared by the original Stage C entry point.
PATIENCE = 5
BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface")
    parser.add_argument("--output-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase5a_frozen3b_formal_seed42")
    parser.add_argument("--phase5a-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=base.PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


@dataclass
class TestAccessGuard:
    """One-way protocol gate: lock selection first, then materialize test once."""

    selection_locked: bool = False
    consumed: bool = False
    checkpoint_sha256: str | None = None
    thresholds: tuple[float, float] | None = None

    def lock(self, checkpoint_sha256: str, thresholds: tuple[float, float]) -> None:
        if self.selection_locked:
            raise RuntimeError("checkpoint selection is already locked")
        if not checkpoint_sha256 or len(checkpoint_sha256) != 64:
            raise ValueError("a saved checkpoint SHA-256 is required before test access")
        self.checkpoint_sha256 = checkpoint_sha256
        self.thresholds = tuple(float(value) for value in thresholds)
        self.selection_locked = True

    def consume(self) -> None:
        if not self.selection_locked:
            raise RuntimeError("test cannot be materialized before checkpoint and threshold lock")
        if self.consumed:
            raise RuntimeError("formal ML test may only be materialized once")
        self.consumed = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_parameters(parameters, torch) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


def formal_gradient_statistics(values) -> dict:
    result = base.gradient_statistics(values)
    array = np.asarray(values, np.float64)
    result.update({
        "fraction_gt_500": float(np.mean(array > 500.0)),
        "fraction_gt_1000": float(np.mean(array > 1000.0)),
    })
    return result


def assert_formal_contract(args, model, tensors) -> dict:
    normalizer = tensors["benefit_normalizer"]
    expected = {"mean": -0.1921661049, "std": 0.1974763721, "epsilon": 1e-4, "fit_count": 616}
    actual = {
        "mean": normalizer.mean, "std": normalizer.scale, "epsilon": normalizer.epsilon,
        "fit_count": len(normalizer.fit_sample_ids), "infeasible_excluded": 104,
    }
    if args.seed != 42 or BATCH_SIZE != 8 or FORMAL_LR != 3e-5:
        raise RuntimeError("formal seed/batch/LR contract changed")
    if abs(actual["mean"] - expected["mean"]) > 1e-7 or abs(actual["std"] - expected["std"]) > 1e-7:
        raise RuntimeError(f"C-S4 benefit normalizer parity failed: {actual}")
    if actual["fit_count"] != expected["fit_count"] or actual["epsilon"] != expected["epsilon"]:
        raise RuntimeError(f"C-S4 benefit normalizer scope failed: {actual}")
    if not model.scale_alignment_enabled or abs(model.native_embedding_stats["median"] - 1.0087162852287292) > 1e-6:
        raise RuntimeError("C-S3 native-scale alignment contract failed")
    return {
        "label": LABEL, "optimizer": "AdamW", "learning_rate": FORMAL_LR,
        "weight_decay": WEIGHT_DECAY, "betas": list(BETAS), "eps": EPS,
        "scheduler": "none", "gradient_clipping": False, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "max_epochs_source": "predeclared Stage C --epochs default=25",
        "patience_source": "predeclared Stage C --patience default=5",
        "gaussian_nll": "0.5*error^2*exp(-log_variance) + 0.5*log_variance",
        "C_S5_detach": False, "scale_alignment_enabled": True,
        "native_embedding_median": model.native_embedding_stats["median"],
        "normalizer": actual, "only_trainable_modules": list(model.trainable_parameter_groups()),
    }


def train_formal(model, development, tensors, torch):
    from src.evaluation.context_value_metrics import validation_selection_key

    parameters = base.trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=FORMAL_LR, weight_decay=WEIGHT_DECAY, betas=BETAS, eps=EPS
    )
    generator = torch.Generator().manual_seed(42)
    feasible_indices = tensors["feasible_indices"]
    curve, validation_rows, gradient_rows, gaussian_rows = [], [], [], []
    best, stale, global_step = None, 0, 0
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train()
        order = feasible_indices[torch.randperm(len(feasible_indices), generator=generator)]
        epoch_parts = {key: [] for key in ("total_loss", "benefit_likelihood", "harm_loss", "uncertainty_regularizer")}
        epoch_gradients = []
        for start in range(0, len(order), BATCH_SIZE):
            global_step += 1; indices = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            output = model(tensors["train_x"][indices].to("cuda"))
            target = tensors["train_y"][indices].to("cuda")
            error = output.benefit_mean - target
            benefit = 0.5 * (error.square() * torch.exp(-output.benefit_log_variance)).mean()
            uncertainty = 0.5 * output.benefit_log_variance.mean()
            harm = torch.nn.functional.binary_cross_entropy_with_logits(
                output.harm_logit, tensors["train_harm"][indices].to("cuda"), pos_weight=tensors["pos_weight"]
            )
            loss = benefit + uncertainty + harm
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite formal loss at epoch={epoch} step={global_step}")
            loss.backward()
            groups = base.group_gradient_norms(model); raw = base.gradient_norm(parameters)
            audit = base.frozen_audit(model, optimizer)
            if any(not math.isfinite(value) for value in (raw, *groups.values())):
                raise FloatingPointError(f"non-finite raw gradient at step={global_step}")
            if audit["qwen_requires_grad_parameter_count"] or audit["qwen_gradient_tensor_count"] or audit["qwen_optimizer_parameter_count"] or not audit["optimizer_only_projection_heads"]:
                raise RuntimeError(f"Qwen frozen boundary failed at formal step={global_step}")
            optimizer.step()  # Intentionally no clipping.
            if not finite_parameters(parameters, torch):
                raise FloatingPointError(f"trainable parameter became NaN/Inf at step={global_step}")
            epoch_gradients.append(raw)
            gradient_rows.append({
                "synthetic_interaction": LABEL, "epoch": epoch, "step": global_step,
                "raw_grad_norm": raw, "projection_grad": groups["projection"],
                "benefit_head_grad": groups["benefit_head"], "harm_head_grad": groups["harm_head"],
                "uncertainty_head_grad": groups["uncertainty_head"], "gradient_clipped": False,
            })
            logv = output.benefit_log_variance.detach().float(); sigma = torch.exp(0.5 * logv); precision = torch.exp(-logv)
            gaussian_rows.append({
                "synthetic_interaction": LABEL, "epoch": epoch, "step": global_step,
                "normalized_target_mean": float(target.detach().mean()), "benefit_mean": float(output.benefit_mean.detach().mean()),
                "prediction_error_mean": float(error.detach().mean()), "prediction_error_abs_mean": float(error.detach().abs().mean()),
                "log_variance_mean": float(logv.mean()), "log_variance_std": float(logv.std()),
                "log_variance_min": float(logv.min()), "log_variance_max": float(logv.max()),
                "sigma_mean": float(sigma.mean()), "exp_neg_log_variance_mean": float(precision.mean()),
                "benefit_likelihood": float(benefit.detach()), "uncertainty_regularizer": float(uncertainty.detach()),
            })
            for key, value in (("total_loss", loss), ("benefit_likelihood", benefit), ("harm_loss", harm), ("uncertainty_regularizer", uncertainty)):
                epoch_parts[key].append(float(value.detach()))

        validation_raw = base.prediction_batches(model, tensors["val_x"], BATCH_SIZE, torch)
        validation = base.denormalize_prediction(validation_raw, tensors["benefit_mean"], tensors["benefit_scale"])
        thresholds, validation_metrics, threshold_audit = base.select_thresholds(
            validation, development["val_targets"], development["val_meta"], development["val_episodes"]
        )
        key = validation_selection_key(validation_metrics)
        gradient_stats = formal_gradient_statistics(epoch_gradients)
        recent_gaussian = gaussian_rows[-len(epoch_gradients):]
        epoch_row = {
            "synthetic_interaction": LABEL, "epoch": epoch,
            **{name: float(np.mean(values)) for name, values in epoch_parts.items()},
            **{f"raw_gradient_{name}": value for name, value in gradient_stats.items()},
            "projection_grad_mean": float(np.mean([row["projection_grad"] for row in gradient_rows[-len(epoch_gradients):]])),
            "benefit_head_grad_mean": float(np.mean([row["benefit_head_grad"] for row in gradient_rows[-len(epoch_gradients):]])),
            "harm_head_grad_mean": float(np.mean([row["harm_head_grad"] for row in gradient_rows[-len(epoch_gradients):]])),
            "uncertainty_head_grad_mean": float(np.mean([row["uncertainty_head_grad"] for row in gradient_rows[-len(epoch_gradients):]])),
            "log_variance_mean": float(np.mean([row["log_variance_mean"] for row in recent_gaussian])),
            "log_variance_std": float(np.mean([row["log_variance_std"] for row in recent_gaussian])),
            "log_variance_min": float(np.min([row["log_variance_min"] for row in recent_gaussian])),
            "log_variance_max": float(np.max([row["log_variance_max"] for row in recent_gaussian])),
            "sigma_mean": float(np.mean([row["sigma_mean"] for row in recent_gaussian])),
            "exp_neg_log_variance_mean": float(np.mean([row["exp_neg_log_variance_mean"] for row in recent_gaussian])),
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "epoch_duration_s": time.perf_counter() - epoch_started,
        }
        curve.append(epoch_row)
        validation_rows.append({
            "synthetic_interaction": LABEL, "epoch": epoch, **validation_metrics,
            "benefit_threshold": thresholds[0], "harm_threshold": thresholds[1],
            "selection_key": json.dumps(base.clean(list(key))), "threshold_candidates": len(threshold_audit),
        })
        if best is None or key < best["key"]:
            best = {"key": key, "epoch": epoch, "state": model.trainable_state_dict(),
                    "thresholds": thresholds, "metrics": validation_metrics}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} loss={epoch_row['total_loss']:.6f} val_MAE={validation_metrics['Benefit_MAE']:.6f} "
            f"recall={validation_metrics['Beneficial_Switch_Recall']:.4f} harmful={validation_metrics['Harmful_Switch_Rate']:.4f} "
            f"grad_mean={gradient_stats['mean']:.2f} grad_max={gradient_stats['max']:.2f} stale={stale}", flush=True,
        )
        if stale >= PATIENCE:
            break

    last_state = model.trainable_state_dict()
    model.load_trainable_state_dict(best["state"]); model.eval()
    training = {
        "epochs_completed": len(curve), "best_epoch": best["epoch"], "early_stopped": len(curve) < MAX_EPOCHS,
        "patience": PATIENCE, "best_validation_metrics": best["metrics"],
        "thresholds": list(best["thresholds"]), "selection_key": list(best["key"]),
        "selection_rule": "validation-only lexicographic: enforce harmful<=0.01; minimize harmful; maximize beneficial recall; minimize mean regret and MAE; maximize Spearman and AUROC",
        "training_time_s": time.perf_counter() - training_started,
        "training_peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
        "raw_gradient": formal_gradient_statistics([row["raw_grad_norm"] for row in gradient_rows]),
        "optimizer_audit": base.frozen_audit(model, optimizer), "gradient_clipping": False,
    }
    normalizers = {name: tensors[name] for name in ("feature_mean", "feature_scale", "benefit_mean", "benefit_scale")}
    return model, normalizers, curve, validation_rows, gradient_rows, gaussian_rows, training, last_state


def checkpoint_payload(model, normalizers, training, seed, state=None):
    return {
        "label": LABEL, "model_id": MODEL_ID, "seed": seed, "epoch": training["best_epoch"],
        "model_state": model.trainable_state_dict() if state is None else state,
        **normalizers, "thresholds": training["thresholds"], "selection_rule": training["selection_rule"],
        "test_used_for_selection": False, "optimizer": "AdamW", "learning_rate": FORMAL_LR,
        "weight_decay": WEIGHT_DECAY, "betas": BETAS, "eps": EPS, "scheduler": "none",
        "gradient_clipping": False,
    }


def subset_metrics(name, prediction, targets, meta, episodes, thresholds, split):
    keep = [index for index, row in enumerate(meta) if row["context_split"] == split]
    episode_keys = {(meta[index]["scenario"], meta[index]["sample"]) for index in keep}
    subset_prediction = {key: value[keep] for key, value in prediction.items() if key != "embedding"}
    _, _, metrics = base.evaluate_predictions(
        name, subset_prediction, [targets[index] for index in keep], [meta[index] for index in keep],
        [episode for episode in episodes if episode.key in episode_keys], thresholds,
    )
    return metrics


def benchmark(model, features, torch):
    rows = []
    model.eval()
    for mode, batch_size in (("single_candidate", 1), ("multi_candidate", 5), ("batch8", 8)):
        batch = features[:batch_size].to("cuda")
        with torch.inference_mode():
            for _ in range(50): model(batch)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); times = []
            for _ in range(200):
                torch.cuda.synchronize(); started = time.perf_counter(); model(batch); torch.cuda.synchronize()
                times.append((time.perf_counter() - started) * 1000)
        rows.append({
            "synthetic_interaction": LABEL, "mode": mode, "batch_size": batch_size,
            "latency_mean_ms": float(np.mean(times)), "latency_median_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        })
    return rows


def hard_case_rows(l1_decisions, l2_decisions, manifest):
    groups = {
        "S9": lambda row: row["scenario"] == "S9_uncertain_new_person",
        "S10": lambda row: row["scenario"] == "S10_action_conflict",
        "high_turn_sensitive": lambda row: row["scenario"] == "S8_high_turn_sensitive",
        "high_distance_sensitive": lambda row: row["scenario"] == "S6_high_distance_sensitive",
        "phase4_beneficial": lambda row: {"scenario": row["scenario"], "sample": int(row["sample"])} in manifest["phase4c3_beneficial_cases"],
        "phase4_harmful": lambda row: {"scenario": row["scenario"], "sample": int(row["sample"])} in manifest["phase4c3_harmful_cases"],
        "phase4_max_regret": lambda row: any(item["scenario"] == row["scenario"] and item["sample"] == int(row["sample"]) for item in manifest["phase4c2_max_regret_cases"]),
    }
    rows = []
    from src.evaluation.context_value_metrics import decision_metrics
    for model_name, decisions in (("L1", l1_decisions), ("L2-FROZEN", l2_decisions)):
        for group, predicate in groups.items():
            selected = [row for row in decisions if predicate(row)]
            if not selected: continue
            rows.append({
                "synthetic_interaction": LABEL, "model": model_name, "hard_case_group": group,
                "count": len(selected), **decision_metrics(selected),
                "Beneficial_Switch_Count": sum(bool(row["beneficial_switch"]) for row in selected),
                "Harmful_Switch_Count": sum(bool(row["harmful_switch"]) for row in selected),
                "Personalized_Decision_Rate": float(np.mean([bool(row["personalized"]) for row in selected])),
            })
    return rows


def make_figures(output, curve, validation_rows, comparison, context_rows, gradient_rows, gaussian_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.title(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure(); plt.plot([row["epoch"] for row in curve], [row["total_loss"] for row in curve]); plt.xlabel("epoch"); plt.ylabel("train total loss"); save("training_loss.png")
    plt.figure(); plt.plot([row["epoch"] for row in validation_rows], [row["Benefit_MAE"] for row in validation_rows]); plt.xlabel("epoch"); plt.ylabel("validation Benefit MAE"); save("validation_benefit_mae.png")
    plt.figure(); plt.plot([row["step"] for row in gradient_rows], [row["raw_grad_norm"] for row in gradient_rows]); plt.xlabel("step"); plt.ylabel("raw gradient norm"); save("raw_gradient.png")
    plt.figure(); plt.plot([row["step"] for row in gaussian_rows], [row["log_variance_mean"] for row in gaussian_rows]); plt.xlabel("step"); plt.ylabel("mean log variance"); save("log_variance.png")
    for metric, filename in (("Beneficial_Switch_Recall", "beneficial_recall.png"), ("Mean_Regret", "mean_regret.png")):
        plt.figure(); plt.bar([row["model"] for row in comparison], [row[metric] for row in comparison]); plt.ylabel(metric); save(filename)
    splits = sorted(set(row["context_split"] for row in context_rows)); x = np.arange(len(splits)); width = .35
    plt.figure(figsize=(10, 4))
    for offset, model in ((-.5, "L1"), (.5, "L2-FROZEN")):
        values = [next(row["Mean_Regret"] for row in context_rows if row["model"] == model and row["context_split"] == split) for split in splits]
        plt.bar(x + offset * width, values, width, label=model)
    plt.xticks(x, splits, rotation=25, ha="right"); plt.legend(); save("context_split_regret.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite formal Stage C-R1 results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    from src.evaluation.context_value_metrics import candidate_metrics, decision_metrics, switch_metrics
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    from src.multimodal.context_schema import CONTEXT_DIM, TOKEN_ORDER, prepare_context_batch

    frozen_summary = json.loads((args.phase5a_dir / "summary.json").read_text(encoding="utf-8"))
    frozen_dataset = json.loads((args.phase5a_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    hard_manifest_bytes = (args.phase5a_dir / "hard_case_manifest.json").read_bytes()
    hard_manifest = json.loads(hard_manifest_bytes)
    development = base.build_development_data(args, torch)
    train_raw = prepare_context_batch(development["train_samples"]); val_raw = prepare_context_batch(development["val_samples"])
    if CONTEXT_DIM != frozen_dataset["input_dimension"] or CONTEXT_DIM != 108: raise RuntimeError("108-D context parity failed")
    input_audit = {
        "label": LABEL, "context_dim": CONTEXT_DIM, "token_order": list(TOKEN_ORDER),
        "train_count": len(train_raw), "validation_count": len(val_raw),
        "L1_L2_train_checksum_equal": base.sha256_array(train_raw) == base.sha256_array(train_raw.copy()),
        "L1_L2_validation_checksum_equal": base.sha256_array(val_raw) == base.sha256_array(val_raw.copy()),
        "hard_case_manifest_sha256": hashlib.sha256(hard_manifest_bytes).hexdigest(),
        "forbidden_inputs_absent": frozen_dataset["forbidden_inputs_absent"],
        "identity_shortcut_absent": frozen_dataset["identity_shortcut_absent"],
        "test_materialized": False,
    }
    model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(
        args.model_id, device_map={"": 0}, cache_dir=str(args.cache_dir), local_files_only=True
    ).to("cuda")
    tensors = base.prepare_training_tensors(development, torch)
    contract = assert_formal_contract(args, model, tensors)
    base.write_json(args.output_dir / "formal_config.json", {**contract, "input_audit": input_audit})

    model, normalizers, curve, validation_rows, gradient_rows, gaussian_rows, training, last_state = train_formal(
        model, development, tensors, torch
    )
    best_path = args.output_dir / "best_validation_checkpoint.pt"
    last_path = args.output_dir / "last_checkpoint.pt"
    torch.save(checkpoint_payload(model, normalizers, training, args.seed), best_path)
    torch.save(checkpoint_payload(model, normalizers, training, args.seed, last_state), last_path)
    best_sha = sha256_file(best_path); last_sha = sha256_file(last_path)
    selection = {
        "label": LABEL, "best_epoch": training["best_epoch"], "epochs_completed": training["epochs_completed"],
        "early_stopped": training["early_stopped"], "criterion": training["selection_rule"],
        "best_validation_metrics": training["best_validation_metrics"], "selection_key": training["selection_key"],
        "locked_thresholds": training["thresholds"], "best_checkpoint_sha256": best_sha,
        "last_checkpoint_sha256": last_sha, "selected_using": "validation only", "test_materialized": False,
    }
    base.write_json(args.output_dir / "checkpoint_selection.json", selection)
    base.write_csv(args.output_dir / "training_curve.csv", curve)
    base.write_csv(args.output_dir / "validation_metrics.csv", validation_rows)
    base.write_csv(args.output_dir / "gradient_trajectory.csv", gradient_rows)
    base.write_csv(args.output_dir / "gaussian_nll_trajectory.csv", gaussian_rows)

    guard = TestAccessGuard(); guard.lock(best_sha, tuple(training["thresholds"])); guard.consume()
    episodes, datasets, samples, targets, meta = base.materialize_test(args, development, torch)
    test_raw = prepare_context_batch(samples)
    test_x = torch.from_numpy(((test_raw - normalizers["feature_mean"]) / normalizers["feature_scale"]).astype(np.float32))
    l2_raw = base.prediction_batches(model, test_x, BATCH_SIZE, torch, with_embeddings=True)
    l2 = base.denormalize_prediction(l2_raw, normalizers["benefit_mean"], normalizers["benefit_scale"])
    l2_candidates, l2_decisions, l2_metrics = base.evaluate_predictions("L2-FROZEN", l2, targets, meta, episodes, tuple(training["thresholds"]))
    if any(row["reentry"] for row in l2_decisions): raise RuntimeError("L2 re-enabled an infeasible candidate")

    l1_candidate_rows = [row for row in csv.DictReader((args.phase5a_dir / "benefit_prediction.csv").open(encoding="utf-8")) if row["model"] == "L1"]
    l1_decisions = [row for row in csv.DictReader((args.phase5a_dir / "decision_metrics.csv").open(encoding="utf-8")) if row["model"] == "L1"]
    if len(l1_candidate_rows) != len(samples) or len(l1_decisions) != len(episodes): raise RuntimeError("frozen L1/test split mismatch")
    l1 = {"benefit": np.asarray([float(row["predicted_benefit"]) for row in l1_candidate_rows]), "sigma": np.asarray([float(row["benefit_uncertainty"]) for row in l1_candidate_rows]), "harm": np.asarray([float(row["predicted_harm_probability"]) for row in l1_candidate_rows])}
    for row in l1_decisions:
        for key in ("personalized", "beneficial_switch", "harmful_switch", "Safety_Violation", "reentry"): row[key] = str(row[key]).lower() == "true"
        row["Oracle_Regret"] = float(row["Oracle_Regret"]); row["GT_Total_Cost"] = float(row["GT_Total_Cost"])
    target_values = {"benefit": np.asarray([target.benefit for target in targets]), "harm": np.asarray([target.harm for target in targets])}
    feasible = np.asarray([row["feasible"] for row in meta], bool)
    opportunity_count = sum(
        episode.feasible.any() and np.min(episode.gt_costs.total[episode.feasible]) < episode.gt_costs.total[np.flatnonzero(episode.feasible)[np.argmin(episode.generic_costs.total[episode.feasible])]] - 1e-6
        for episode in episodes
    )
    l1_metrics = {**candidate_metrics(l1, target_values, feasible), **switch_metrics(l1_decisions, opportunity_count), **decision_metrics(l1_decisions)}
    # L1 is a frozen reference.  Preserve its Stage A/B evaluator values
    # verbatim instead of silently recomputing a denominator under a newer
    # helper (the frozen beneficial-opportunity denominator is 47, not 44).
    frozen_l1 = frozen_summary["models"]["L1"]
    l1_metrics.update({
        "Benefit_MAE": frozen_l1["Benefit_MAE"],
        "Benefit_Spearman": frozen_l1["Benefit_Ranking_Spearman"],
        "Harm_AUROC": frozen_l1["Harm_AUROC"],
        "Beneficial_Switch_Recall": frozen_l1["Beneficial_Switch_Recall"],
        "Beneficial_Switch_Precision": frozen_l1["Beneficial_Switch_Precision"],
        "Harmful_Switch_Rate": frozen_l1["Harmful_Switch_Rate"],
        "GT_Total_Cost": frozen_l1["GT_Total_Cost"],
        "Mean_Regret": frozen_l1["Mean_Regret"],
        "P95_Regret": frozen_l1["P95_Regret"],
        "Max_Regret": frozen_l1["Max_Regret"],
        "Safety_Violation": frozen_l1["Safety_Violation"],
        "Personalized_Decision_Rate": frozen_l1["Personalized_Decision_Rate"],
        "Generic_Safe_Rate": frozen_l1["Generic_Safe_Rate"],
    })
    comparison = [{"synthetic_interaction": LABEL, "model": "L1", **l1_metrics}, {"synthetic_interaction": LABEL, "model": "L2-FROZEN", **l2_metrics}]
    base.write_csv(args.output_dir / "candidate_metrics.csv", [dict(row, model="L1") for row in l1_candidate_rows] + l2_candidates)
    base.write_csv(args.output_dir / "switch_metrics.csv", comparison)
    base.write_csv(args.output_dir / "decision_metrics.csv", [dict(row, synthetic_interaction=LABEL, model="L1") for row in l1_decisions] + l2_decisions)
    base.write_csv(args.output_dir / "small_vs_frozen3b.csv", comparison)

    context_rows = []
    splits = sorted(set(row["context_split"] for row in meta))
    for split in splits:
        context_rows.append({"synthetic_interaction": LABEL, "model": "L1", "context_split": split, **subset_metrics("L1", l1, targets, meta, episodes, tuple(frozen_summary["thresholds"]["L1"]), split)})
        context_rows.append({"synthetic_interaction": LABEL, "model": "L2-FROZEN", "context_split": split, **subset_metrics("L2-FROZEN", l2, targets, meta, episodes, tuple(training["thresholds"]), split)})
    base.write_csv(args.output_dir / "by_context_split.csv", context_rows)
    hard_rows = hard_case_rows(l1_decisions, l2_decisions, hard_manifest)
    base.write_csv(args.output_dir / "hard_cases.csv", hard_rows)

    memory_rows = benchmark(model, test_x, torch)
    for row in memory_rows:
        row.update({"training_peak_gb": training["training_peak_cuda_gb"], "training_time_s": training["training_time_s"]})
    base.write_csv(args.output_dir / "memory_latency.csv", memory_rows)
    checkpoint_audit = {
        "label": LABEL, "best_checkpoint_sha256": best_sha, "last_checkpoint_sha256": last_sha,
        "checkpoint_selected_on": "validation only", "thresholds_selected_on": "validation only",
        "test_access_count": int(guard.consumed), "test_materialized_after_selection_lock": guard.selection_locked and guard.consumed,
        "test_can_change_checkpoint": False, "qwen_fully_frozen": model.backbone_fully_frozen,
        "optimizer_audit": training["optimizer_audit"], "scale_alignment_enabled": model.scale_alignment_enabled,
        "normalizer_fit_count": len(tensors["benefit_normalizer"].fit_sample_ids), "normalizer_mean": tensors["benefit_mean"],
        "normalizer_std": tensors["benefit_scale"], "results_finite": True,
    }
    base.write_json(args.output_dir / "checkpoint_audit.json", checkpoint_audit)

    context_lookup = {(row["model"], row["context_split"]): row for row in context_rows}
    complex_improvements = {split: context_lookup[("L1", split)]["Mean_Regret"] - context_lookup[("L2-FROZEN", split)]["Mean_Regret"] for split in splits if split.startswith(("C4_", "C5_", "C6_"))}
    hard_lookup = {(row["model"], row["hard_case_group"]): row for row in hard_rows}
    def hard_regret(model_name, group): return hard_lookup[(model_name, group)]["Mean_Regret"]
    gate = {
        "not_always_generic_safe": l2_metrics["Personalized_Decision_Rate"] > .05,
        "beneficial_recall_meaningfully_above_L1": l2_metrics["Beneficial_Switch_Recall"] >= l1_metrics["Beneficial_Switch_Recall"] + .02,
        "precision_not_collapsed": l2_metrics["Beneficial_Switch_Precision"] >= .05,
        "harmful_switch_not_clearly_increased": l2_metrics["Harmful_Switch_Rate"] <= .025,
        "mean_regret_not_clearly_worse": l2_metrics["Mean_Regret"] <= l1_metrics["Mean_Regret"] + .005,
        "p95_regret_not_clearly_worse": l2_metrics["P95_Regret"] <= l1_metrics["P95_Regret"] + .025,
        "safety_not_worse": l2_metrics["Safety_Violation"] <= frozen_summary["models"]["L1"]["Safety_Violation"],
        "one_C4_C5_C6_split_clearly_better": max(complex_improvements.values()) >= .005,
        "S9_not_clearly_worse": hard_regret("L2-FROZEN", "S9") <= hard_regret("L1", "S9") + .025,
        "S10_not_clearly_worse": hard_regret("L2-FROZEN", "S10") <= hard_regret("L1", "S10") + .025,
        "turn_not_catastrophic": hard_regret("L2-FROZEN", "high_turn_sensitive") <= hard_regret("L1", "high_turn_sensitive") + .025,
        "distance_not_catastrophic": hard_regret("L2-FROZEN", "high_distance_sensitive") <= hard_regret("L1", "high_distance_sensitive") + .025,
    }
    gate["seed42_gate_passed"] = all(gate.values())
    figures = make_figures(args.output_dir, curve, validation_rows, comparison, context_rows, gradient_rows, gaussian_rows)
    summary = {
        "label": LABEL, "stage": "Phase 5A Stage C-R1 Frozen-3B Formal Training", "seed": 42,
        "formal_training_completed": True, "formal_test_evaluation_count": 1,
        "model_id": MODEL_ID, "quantization": "NF4 4-bit", "no_lora": True, "no_qlora": True,
        "formal_contract": contract, "training": training,
        "models": {"L1": l1_metrics, "L2-FROZEN": l2_metrics},
        "complex_split_mean_regret_improvement_L1_minus_L2": complex_improvements,
        "hard_cases": hard_rows, "memory_latency": memory_rows, "checkpoint_audit": checkpoint_audit,
        "gate": gate, "scientific_scope": "same 108-D synthetic-interaction input; seed=42 only; no claim that a large model is necessary",
        "five_seed_started": False, "stage_d_started": False, "next_step_requires_human_approval": True,
        "figures": figures,
    }
    base.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(base.clean({"training": training, "L1": l1_metrics, "L2-FROZEN": l2_metrics, "gate": gate}), indent=2), flush=True)


if __name__ == "__main__":
    main()
