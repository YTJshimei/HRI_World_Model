"""Phase 5A Stage D-R1: formal seed-42 LoRA-adapted Qwen training.

All inputs and outputs in this experiment are synthetic interaction data.  The
script deliberately implements a single declared configuration and refuses to
overwrite an existing formal result directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as base
from scripts import run_phase5a_frozen3b_formal as formal
from scripts import run_phase5a_lora_smoke as d0

LABEL = base.LABEL
MODEL_ID = d0.MODEL_ID
L3_NAME = "L3-PREPARED-LORA"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_lora_formal_seed42")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--original-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_frozen3b_formal_seed42")
    parser.add_argument("--prepared-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_prepared_nolora_control")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(base.clean(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalizer_sha256(normalizers) -> str:
    digest = hashlib.sha256()
    for name in sorted(normalizers):
        digest.update(name.encode())
        digest.update(np.asarray(normalizers[name]).tobytes())
    return digest.hexdigest()


def lora_config_contract(targets: list[str]) -> dict:
    return {
        "r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
        "target_modules": list(targets), "target_count": 72,
        "target_scope": "language transformer self_attn.q_proj/v_proj only",
    }


def compare_prepared_contract(current: dict, reference: dict) -> dict:
    """Compare only the frozen, pre-LoRA base state required by D-C0."""
    fields = (
        "parameter_dtype_counts", "module_dtype_counts", "layer_norm_dtype_counts",
        "embedding_dtype", "lm_head_dtype", "linear4bit_count", "is_loaded_in_4bit",
        "requires_grad_parameter_count",
    )
    checks = {f"after.{field}": current["after"].get(field) == reference["after"].get(field) for field in fields}
    checks["preparation_source_sha256"] = current.get("preparation_source_sha256") == reference.get("preparation_source_sha256")
    return {
        "label": LABEL, "reference": "Phase 5A Stage D-C0 prepared_base_contract.json",
        "checks": checks, "matched": all(checks.values()),
        "current_contract_sha256": canonical_json_sha256({key: current[key] for key in ("preparation_source_sha256", "after")}),
        "reference_contract_sha256": canonical_json_sha256({key: reference[key] for key in ("preparation_source_sha256", "after")}),
        "test_materialized": False,
    }


def lora_named_parameters(model):
    return [(name, parameter) for name, parameter in model.backbone.named_parameters() if "lora_" in name]


def task_state_dict(model) -> dict[str, object]:
    return {
        "projection_heads": model.trainable_state_dict(),
        "lora": {name: parameter.detach().cpu().clone() for name, parameter in lora_named_parameters(model)},
    }


def load_task_state_dict(model, state: dict[str, object]) -> None:
    model.load_trainable_state_dict(state["projection_heads"])
    named = dict(lora_named_parameters(model))
    if set(named) != set(state["lora"]):
        raise ValueError("LoRA checkpoint paths do not exactly match the injected adapter")
    with __import__("torch").no_grad():
        for name, value in state["lora"].items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def state_checksum(state: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for section in ("projection_heads", "lora"):
        for name, value in sorted(state[section].items()):
            digest.update(section.encode()); digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_lora_training_mode(model) -> None:
    model.train()  # Adapter override keeps the original backbone in eval mode.
    for name, module in model.backbone.named_modules():
        if "lora_dropout" in name:
            module.train()


def lora_boundary(model, optimizer=None) -> dict:
    audit = d0.trainable_audit(model, optimizer)
    lora_paths = [name for name, _ in model.backbone.named_modules() if name.endswith(".lora_A.default")]
    selected = sorted(name[: -len(".lora_A.default")].replace("base_model.model.", "", 1) for name in lora_paths)
    audit.update({
        "selected_module_path_count": len(selected),
        "selected_module_paths": selected,
        "lora_A_module_count": len(lora_paths),
        "vision_lora_module_count": sum(d0.is_vision_path(name) for name in lora_paths),
    })
    return audit


def assert_boundary(audit: dict, require_gradients: bool = False) -> None:
    if audit["selected_module_path_count"] != 72 or audit["lora_A_module_count"] != 72:
        raise RuntimeError("D-R1 requires exactly 72 language q/v LoRA paths")
    if audit["vision_lora_parameter_count"] or audit["vision_lora_module_count"]:
        raise RuntimeError("vision LoRA is forbidden")
    if audit["qwen_original_base_requires_grad_count"] or audit["qwen_original_base_optimizer_count"]:
        raise RuntimeError("original Qwen parameters entered the trainable/optimizer boundary")
    if audit["embedding_trainable_count"] or audit["lm_head_trainable_count"]:
        raise RuntimeError("embedding or LM head became trainable")
    if not audit["optimizer_exactly_lora_projection_heads"]:
        raise RuntimeError("optimizer must contain exactly LoRA plus projection/heads")
    if require_gradients and audit["qwen_original_base_gradient_tensor_count"]:
        raise RuntimeError("original Qwen received gradients")


def train_lora(model, development, tensors, torch):
    from src.evaluation.context_value_metrics import validation_selection_key

    original, lora, heads = d0.parameter_groups(model)
    parameters = [parameter for _, parameter in lora + heads]
    optimizer = torch.optim.AdamW(parameters, lr=formal.FORMAL_LR, weight_decay=formal.WEIGHT_DECAY,
                                  betas=formal.BETAS, eps=formal.EPS)
    boundary = lora_boundary(model, optimizer); assert_boundary(boundary)
    generator = torch.Generator().manual_seed(42)
    feasible = tensors["feasible_indices"]
    curve, validation_rows, gradient_rows, gaussian_rows = [], [], [], []
    best = None; stale = 0; global_step = 0
    torch.cuda.reset_peak_memory_stats(); training_started = time.perf_counter()
    for epoch in range(1, formal.MAX_EPOCHS + 1):
        epoch_started = time.perf_counter(); set_lora_training_mode(model)
        order = feasible[torch.randperm(len(feasible), generator=generator)]
        parts = {key: [] for key in ("total_loss", "benefit_likelihood", "harm_loss", "uncertainty_regularizer")}
        epoch_gradients = []; epoch_group_gradients = {key: [] for key in ("lora", "projection", "benefit_head", "harm_head", "uncertainty_head")}
        epoch_gaussian = []
        for start in range(0, len(order), formal.BATCH_SIZE):
            global_step += 1; indices = order[start:start + formal.BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            output = model(tensors["train_x"][indices].to("cuda")); target = tensors["train_y"][indices].to("cuda")
            error = output.benefit_mean - target
            benefit = 0.5 * (error.square() * torch.exp(-output.benefit_log_variance)).mean()
            uncertainty = 0.5 * output.benefit_log_variance.mean()
            harm = torch.nn.functional.binary_cross_entropy_with_logits(
                output.harm_logit, tensors["train_harm"][indices].to("cuda"), pos_weight=tensors["pos_weight"]
            )
            loss = benefit + uncertainty + harm
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite D-R1 loss at epoch={epoch} step={global_step}")
            loss.backward()
            group_values = {
                "lora": d0.gradient_norm(lora),
                "projection": base.gradient_norm(model.trainable_parameter_groups()["projection"]),
                "benefit_head": base.gradient_norm(model.trainable_parameter_groups()["benefit_head"]),
                "harm_head": base.gradient_norm(model.trainable_parameter_groups()["harm_head"]),
                "uncertainty_head": base.gradient_norm(model.trainable_parameter_groups()["uncertainty_head"]),
            }
            raw = base.gradient_norm(parameters)
            original_gradient_tensors = sum(parameter.grad is not None for _, parameter in original)
            if original_gradient_tensors:
                raise RuntimeError(f"original Qwen gradient detected at step={global_step}")
            if not all(math.isfinite(value) and value > 0 for value in (raw, *group_values.values())):
                raise FloatingPointError(f"missing/non-finite trainable gradient at step={global_step}")
            optimizer.step()  # The declared protocol intentionally has no clipping.
            if not formal.finite_parameters(parameters, torch):
                raise FloatingPointError(f"trainable parameter became NaN/Inf at step={global_step}")
            logv = output.benefit_log_variance.detach().float()
            gaussian = {
                "log_variance_mean": float(logv.mean()), "log_variance_std": float(logv.std()),
                "log_variance_min": float(logv.min()), "log_variance_max": float(logv.max()),
            }
            epoch_gaussian.append(gaussian); epoch_gradients.append(raw)
            for key, value in group_values.items(): epoch_group_gradients[key].append(value)
            for key, value in (("total_loss", loss), ("benefit_likelihood", benefit), ("harm_loss", harm), ("uncertainty_regularizer", uncertainty)):
                parts[key].append(float(value.detach()))
            gradient_rows.append({
                "synthetic_interaction": LABEL, "epoch": epoch, "step": global_step, "raw_grad_norm": raw,
                "lora_grad_norm": group_values["lora"], "projection_grad_norm": group_values["projection"],
                "benefit_head_grad_norm": group_values["benefit_head"], "harm_head_grad_norm": group_values["harm_head"],
                "uncertainty_head_grad_norm": group_values["uncertainty_head"],
                "qwen_original_gradient_tensor_count": original_gradient_tensors, "gradient_clipped": False,
            })
            gaussian_rows.append({"synthetic_interaction": LABEL, "epoch": epoch, "step": global_step, **gaussian})

        if all(row["log_variance_mean"] <= -5.99 for row in epoch_gaussian):
            raise FloatingPointError(f"log-variance collapsed to -6 throughout epoch={epoch}")
        model.eval()
        validation_raw = base.prediction_batches(model, tensors["val_x"], formal.BATCH_SIZE, torch)
        validation = base.denormalize_prediction(validation_raw, tensors["benefit_mean"], tensors["benefit_scale"])
        thresholds, metrics, threshold_audit = base.select_thresholds(
            validation, development["val_targets"], development["val_meta"], development["val_episodes"]
        )
        key = validation_selection_key(metrics); stats = formal.formal_gradient_statistics(epoch_gradients)
        row = {
            "synthetic_interaction": LABEL, "epoch": epoch,
            **{name: float(np.mean(values)) for name, values in parts.items()},
            **{f"raw_gradient_{name}": value for name, value in stats.items()},
            **{f"{name}_grad_mean": float(np.mean(values)) for name, values in epoch_group_gradients.items()},
            "log_variance_mean": float(np.mean([item["log_variance_mean"] for item in epoch_gaussian])),
            "log_variance_std": float(np.mean([item["log_variance_std"] for item in epoch_gaussian])),
            "log_variance_min": float(np.min([item["log_variance_min"] for item in epoch_gaussian])),
            "log_variance_max": float(np.max([item["log_variance_max"] for item in epoch_gaussian])),
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "epoch_duration_s": time.perf_counter() - epoch_started,
        }
        curve.append(row)
        validation_rows.append({"synthetic_interaction": LABEL, "epoch": epoch, **metrics,
                                "benefit_threshold": thresholds[0], "harm_threshold": thresholds[1],
                                "selection_key": json.dumps(base.clean(list(key))), "threshold_candidates": len(threshold_audit)})
        if best is None or key < best["key"]:
            best = {"key": key, "epoch": epoch, "state": task_state_dict(model), "thresholds": thresholds, "metrics": metrics}
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch:02d} loss={row['total_loss']:.6f} val_MAE={metrics['Benefit_MAE']:.6f} "
              f"recall={metrics['Beneficial_Switch_Recall']:.4f} harmful={metrics['Harmful_Switch_Rate']:.4f} "
              f"grad_mean={stats['mean']:.2f} lora_grad={row['lora_grad_mean']:.2f} stale={stale}", flush=True)
        if len(curve) >= 3 and all(item["raw_gradient_max"] > 1e6 for item in curve[-3:]) and all(
                curve[-index]["total_loss"] > curve[-index-1]["total_loss"] for index in (1, 2)):
            raise FloatingPointError("persistent extreme gradients accompanied by divergent loss")
        if stale >= formal.PATIENCE: break

    last_state = task_state_dict(model); load_task_state_dict(model, best["state"]); model.eval()
    all_gradients = [row["raw_grad_norm"] for row in gradient_rows]
    training = {
        "epochs_completed": len(curve), "best_epoch": best["epoch"], "early_stopped": len(curve) < formal.MAX_EPOCHS,
        "patience": formal.PATIENCE, "best_validation_metrics": best["metrics"], "thresholds": list(best["thresholds"]),
        "selection_key": list(best["key"]),
        "selection_rule": "validation-only lexicographic: enforce harmful<=0.01; minimize harmful; maximize beneficial recall; minimize mean regret and MAE; maximize Spearman and AUROC",
        "training_time_s": time.perf_counter() - training_started,
        "training_peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
        "raw_gradient": formal.formal_gradient_statistics(all_gradients),
        "optimizer_audit": lora_boundary(model, optimizer), "gradient_clipping": False,
    }
    assert_boundary(training["optimizer_audit"], require_gradients=True)
    normalizers = {name: tensors[name] for name in ("feature_mean", "feature_scale", "benefit_mean", "benefit_scale")}
    return model, normalizers, curve, validation_rows, gradient_rows, gaussian_rows, training, last_state


def checkpoint_payload(model, normalizers, training, lora_config, state=None):
    task_state = task_state_dict(model) if state is None else state
    return {
        "label": LABEL, "model_id": MODEL_ID, "seed": 42, "epoch": training["best_epoch"],
        "task_state": task_state, **normalizers, "thresholds": training["thresholds"],
        "selection_rule": training["selection_rule"], "test_used_for_selection": False,
        "optimizer": "AdamW", "learning_rate": formal.FORMAL_LR, "weight_decay": formal.WEIGHT_DECAY,
        "betas": formal.BETAS, "eps": formal.EPS, "scheduler": "none", "gradient_clipping": False,
        "lora_config": lora_config,
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def convert_decisions(rows):
    converted = []
    for source in rows:
        row = dict(source)
        for key in ("personalized", "beneficial_switch", "harmful_switch", "Safety_Violation", "reentry"):
            row[key] = str(row[key]).lower() == "true"
        row["Oracle_Regret"] = float(row["Oracle_Regret"]); row["GT_Total_Cost"] = float(row["GT_Total_Cost"])
        converted.append(row)
    return converted


def reference_data(args):
    original_summary = json.loads((args.original_dir / "summary.json").read_text(encoding="utf-8"))
    prepared_summary = json.loads((args.prepared_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "metrics": {
            "L1": original_summary["models"]["L1"],
            "L2-ORIGINAL-FROZEN": original_summary["models"]["L2-FROZEN"],
            "L2-P-PREPARED-NO-LORA": prepared_summary["models"]["L2-P-PREPARED-NO-LORA"],
        },
        "context": read_csv(args.original_dir / "by_context_split.csv") + [row for row in read_csv(args.prepared_dir / "by_context_split.csv") if row["model"] == "L2-P-PREPARED-NO-LORA"],
        "hard": read_csv(args.original_dir / "hard_cases.csv") + [row for row in read_csv(args.prepared_dir / "hard_cases.csv") if row["model"] == "L2-P-PREPARED-NO-LORA"],
        "l1_decisions": convert_decisions([row for row in read_csv(args.original_dir / "decision_metrics.csv") if row["model"] == "L1"]),
        "prepared_metrics": prepared_summary["models"]["L2-P-PREPARED-NO-LORA"],
    }


def decision_gain_by_complex_split(context_rows, reference_model="L2-P-PREPARED-NO-LORA") -> dict:
    lookup = {(row["model"], row["context_split"]): row for row in context_rows}
    result = {}
    for split in ("C4_unseen_person_unseen_motion_action", "C5_compound_occlusion_turn_speed", "C6_partial_functional_observation"):
        prepared = lookup[(reference_model, split)]; lora = lookup[(L3_NAME, split)]
        result[split[:2]] = {
            "mean_regret_delta": float(lora["Mean_Regret"]) - float(prepared["Mean_Regret"]),
            "beneficial_switch_delta": int(float(lora["Beneficial_Switch_Count"])) - int(float(prepared["Beneficial_Switch_Count"])),
            "harmful_switch_delta": int(float(lora["Harmful_Switch_Count"])) - int(float(prepared["Harmful_Switch_Count"])),
        }
        result[split[:2]]["decision_level_improved"] = (
            result[split[:2]]["mean_regret_delta"] < -1e-12 or
            (result[split[:2]]["beneficial_switch_delta"] > 0 and result[split[:2]]["harmful_switch_delta"] <= 0)
        )
    return result


def gates(metrics, references, context_gains, context_gains_vs_l1) -> dict:
    prepared = references["L2-P-PREPARED-NO-LORA"]; l1 = references["L1"]
    complex_gain = any(row["decision_level_improved"] for row in context_gains.values())
    gate_a_checks = {
        "beneficial_recall_above_prepared": metrics["Beneficial_Switch_Recall"] > prepared["Beneficial_Switch_Recall"],
        "beneficial_precision_not_worse": metrics["Beneficial_Switch_Precision"] >= prepared["Beneficial_Switch_Precision"],
        "harmful_switch_rate_at_most_1pct": metrics["Harmful_Switch_Rate"] <= 0.01,
        "regret_or_complex_split_gain": metrics["Mean_Regret"] < prepared["Mean_Regret"] or complex_gain,
        "ranking_not_materially_worse": metrics["Benefit_Spearman"] >= prepared["Benefit_Spearman"] - 0.02,
        "harm_discrimination_not_materially_worse": metrics["Harm_AUROC"] >= prepared["Harm_AUROC"] - 0.02,
        "C4_C5_C6_decision_gain": complex_gain,
    }
    complex_gain_vs_l1 = any(row["decision_level_improved"] for row in context_gains_vs_l1.values())
    gate_b_checks = {
        "beneficial_recall_above_L1": metrics["Beneficial_Switch_Recall"] > l1["Beneficial_Switch_Recall"],
        "beneficial_precision_not_collapsed": metrics["Beneficial_Switch_Precision"] >= max(0.0, l1["Beneficial_Switch_Precision"] - 0.02),
        "mean_regret_close_or_better": metrics["Mean_Regret"] <= l1["Mean_Regret"] + 0.002,
        "safety_not_worse": metrics["Safety_Violation"] <= l1["Safety_Violation"] + 1e-12,
        "C4_C5_C6_decision_gain_vs_L1": complex_gain_vs_l1,
    }
    return {
        "Gate_A": {"passed": all(gate_a_checks.values()), "checks": gate_a_checks},
        "Gate_B": {"passed": all(gate_b_checks.values()), "checks": gate_b_checks},
        "five_seed_allowed": all(gate_a_checks.values()) and all(gate_b_checks.values()),
        "thresholds_predeclared_before_test": True,
    }


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite D-R1 results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42)
    import torch
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")

    development = base.build_development_data(args, torch); tensors = base.prepare_training_tensors(development, torch)
    normalizer = tensors["benefit_normalizer"]
    if len(normalizer.fit_sample_ids) != 616 or abs(normalizer.mean + .1921661049) > 1e-7 or abs(normalizer.scale - .1974763721) > 1e-7:
        raise RuntimeError("C-S4 train-feasible-only normalizer parity failed")

    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(
        args.model_id, device_map={"": 0}, cache_dir=str(args.cache_dir), local_files_only=True
    ).to("cuda")
    before = d0.prepared_base_snapshot(model.backbone)
    model = d0.prepare_kbit_backbone(model)
    after = d0.prepared_base_snapshot(model.backbone)
    current_contract = d0.prepared_base_contract(before, after)
    reference_contract = json.loads((args.prepared_dir / "prepared_base_contract.json").read_text(encoding="utf-8"))
    contract_match = compare_prepared_contract(current_contract, reference_contract)
    if not contract_match["matched"]: raise RuntimeError(f"D-C0 prepared-base contract mismatch: {contract_match['checks']}")
    module_audit = d0.discover_attention_modules(model.backbone)
    targets = module_audit["explicit_target_modules"]
    if len(targets) != 72: raise RuntimeError("expected exactly 72 q/v target paths")
    config = lora_config_contract(targets)
    model = d0.inject_language_lora(model, targets)
    contract = formal.assert_formal_contract(args, model, tensors)

    model, normalizers, curve, validation_rows, gradient_rows, gaussian_rows, training, last_state = train_lora(model, development, tensors, torch)
    best_path = args.output_dir / "best_validation_checkpoint.pt"; last_path = args.output_dir / "last_checkpoint.pt"
    torch.save(checkpoint_payload(model, normalizers, training, config), best_path)
    torch.save(checkpoint_payload(model, normalizers, training, config, last_state), last_path)
    best_sha = formal.sha256_file(best_path); last_sha = formal.sha256_file(last_path)
    selection = {
        "label": LABEL, "best_epoch": training["best_epoch"], "epochs_completed": training["epochs_completed"],
        "criterion": training["selection_rule"], "best_validation_metrics": training["best_validation_metrics"],
        "selection_key": training["selection_key"], "locked_thresholds": training["thresholds"],
        "best_checkpoint_sha256": best_sha, "last_checkpoint_sha256": last_sha,
        "normalizer_sha256": normalizer_sha256(normalizers), "lora_config_sha256": canonical_json_sha256(config),
        "selected_using": "validation only", "test_materialized": False,
    }
    base.write_csv(args.output_dir / "training_curve.csv", curve)
    base.write_csv(args.output_dir / "validation_metrics.csv", validation_rows)
    base.write_csv(args.output_dir / "gradient_trajectory.csv", gradient_rows)
    base.write_json(args.output_dir / "checkpoint_selection.json", selection)

    # Verify exact recovery before irreversibly unlocking the one formal test read.
    expected_state_sha = state_checksum(task_state_dict(model)); checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_task_state_dict(model, checkpoint["task_state"]); recovered_state_sha = state_checksum(task_state_dict(model))
    if expected_state_sha != recovered_state_sha: raise RuntimeError("best LoRA checkpoint recovery mismatch")
    guard = formal.TestAccessGuard(); guard.lock(best_sha, tuple(training["thresholds"])); guard.consume()
    episodes, datasets, samples, targets_data, meta = base.materialize_test(args, development, torch)
    from src.multimodal.context_schema import prepare_context_batch
    test_raw = prepare_context_batch(samples)
    test_x = torch.from_numpy(((test_raw - normalizers["feature_mean"]) / normalizers["feature_scale"]).astype(np.float32))
    prediction_raw = base.prediction_batches(model, test_x, formal.BATCH_SIZE, torch)
    prediction = base.denormalize_prediction(prediction_raw, normalizers["benefit_mean"], normalizers["benefit_scale"])
    candidates, decisions, metrics = base.evaluate_predictions(L3_NAME, prediction, targets_data, meta, episodes, tuple(training["thresholds"]))
    if any(row["reentry"] for row in decisions): raise RuntimeError("L3 re-enabled an infeasible candidate")

    refs = reference_data(args); all_metrics = {**refs["metrics"], L3_NAME: metrics}
    comparison = [{"synthetic_interaction": LABEL, "model": name, **values} for name, values in all_metrics.items()]
    prepared_comparison = [row for row in comparison if row["model"] in ("L2-P-PREPARED-NO-LORA", L3_NAME)]
    context_rows = list(refs["context"])
    for split in sorted(set(row["context_split"] for row in meta)):
        context_rows.append({"synthetic_interaction": LABEL, "model": L3_NAME, "context_split": split,
                             **formal.subset_metrics(L3_NAME, prediction, targets_data, meta, episodes, tuple(training["thresholds"]), split)})
    hard_rows = list(refs["hard"])
    l3_hard = formal.hard_case_rows(refs["l1_decisions"], decisions, json.loads((args.phase5a_dir / "hard_case_manifest.json").read_text(encoding="utf-8")))
    for row in l3_hard:
        if row["model"] == "L2-FROZEN": row["model"] = L3_NAME; hard_rows.append(row)
    complex_gains = decision_gain_by_complex_split(context_rows)
    complex_gains_vs_l1 = decision_gain_by_complex_split(context_rows, "L1")
    gate_report = gates(metrics, all_metrics, complex_gains, complex_gains_vs_l1)

    base.write_csv(args.output_dir / "candidate_metrics.csv", candidates)
    base.write_csv(args.output_dir / "switch_metrics.csv", comparison)
    base.write_csv(args.output_dir / "decision_metrics.csv", decisions)
    base.write_csv(args.output_dir / "by_context_split.csv", context_rows)
    base.write_csv(args.output_dir / "hard_cases.csv", hard_rows)
    base.write_csv(args.output_dir / "prepared_vs_lora.csv", prepared_comparison)
    base.write_csv(args.output_dir / "all_models_comparison.csv", comparison)
    memory_rows = formal.benchmark(model, tensors["val_x"], torch)
    adapter_bytes = sum(value.numel() * value.element_size() for value in task_state_dict(model)["lora"].values())
    for row in memory_rows: row["lora_adapter_bytes"] = adapter_bytes
    base.write_csv(args.output_dir / "memory_latency.csv", memory_rows)

    post_audit = lora_boundary(model, None)
    checkpoint_audit = {
        "label": LABEL, "best_checkpoint_sha256": best_sha, "last_checkpoint_sha256": last_sha,
        "checkpoint_selected_on": "validation only", "thresholds_selected_on": "validation only",
        "normalizer_sha256": selection["normalizer_sha256"], "lora_config_sha256": selection["lora_config_sha256"],
        "best_checkpoint_recovery_state_sha256": recovered_state_sha, "test_access_count": int(guard.consumed),
        "test_materialized_after_selection_lock": guard.selection_locked and guard.consumed,
        "test_can_change_checkpoint_or_threshold": False, "prepared_contract": contract_match,
        "trainable_boundary": training["optimizer_audit"], "post_test_boundary": post_audit,
    }
    base.write_json(args.output_dir / "checkpoint_audit.json", checkpoint_audit)
    formal.make_figures(args.output_dir, curve, validation_rows, comparison, context_rows, gradient_rows, gaussian_rows)
    deltas = {key: metrics[key] - refs["prepared_metrics"][key] for key in (
        "Benefit_MAE", "Benefit_Spearman", "Harm_AUROC", "Beneficial_Switch_Recall",
        "Beneficial_Switch_Precision", "Beneficial_Switch_Count", "Harmful_Switch_Count",
        "Mean_Regret", "P95_Regret", "Safety_Violation")}
    summary = {
        "label": LABEL, "stage": "Phase 5A Stage D-R1 LoRA-Adapted Qwen 3B Formal Training seed=42",
        "success": True, "formal_test_evaluation_count": 1, "formal_contract": contract,
        "prepared_base_contract_match": contract_match, "lora_config": config,
        "training": training, "models": all_metrics, "L3_minus_L2_P": deltas,
        "complex_split_attribution": {"vs_L2_P": complex_gains, "vs_L1": complex_gains_vs_l1}, "gates": gate_report,
        "checkpoint_audit": checkpoint_audit, "memory_latency": memory_rows,
        "lora_adapter_bytes": adapter_bytes, "test_used_for_tuning": False,
        "five_seed_started": False, "next_step_requires_human_approval": True,
    }
    base.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(base.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()
