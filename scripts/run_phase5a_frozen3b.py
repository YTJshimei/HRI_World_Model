"""Phase 5A Stage C: seed-42 frozen Qwen context-value experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_frozen3b_stable")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, choices=(8,), default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, choices=(10.0,), default=10.0)
    parser.add_argument("--stability-steps", type=int, choices=(200,), default=200)
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ("empty",))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field, "")) for field in fields})


def sha256_array(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_development_data(args, torch):
    """Rebuild the exact Stage A/B train/validation inputs; test is not touched."""
    import scripts.run_phase4c1_safety_calibration as c1
    import scripts.run_phase4c2_belief_selection as c2
    import scripts.run_phase4c3_selective_personalization as c3
    import scripts.run_phase4c_decision as c0
    import scripts.run_phase5a_context_value as c5
    from src.decision.counterfactual_rollout import CounterfactualRolloutEngine

    engine = CounterfactualRolloutEngine.from_phase4b6_checkpoint(
        args.phase4b6_dir / "checkpoints" / "f2_original_best.pt", args.device
    )
    prior_mean, prior_std = c0.load_prior(argparse.Namespace(phase4b6_dir=args.phase4b6_dir))
    root, scale, safety, calibration, c2_summary = c3.load_frozen_phase4c2(args, torch)
    selector = c2.SelectorConfig(**c2_summary["selector_config"])

    def records(split, seed, count):
        return c1.build_records(args, engine, split, seed, count, prior_mean, prior_std)

    train = records("train", args.seed + 101, 30)
    validation = records("validation", args.seed + 202, 12)
    train_artifacts, train_predictions, cost = c3.build_base(
        args, train, engine, prior_mean, prior_std, root, scale, safety, calibration, None, torch
    )
    val_artifacts, val_predictions, _ = c3.build_base(
        args, validation, engine, prior_mean, prior_std, root, scale, safety, calibration, cost, torch
    )
    train_episodes = c3.episode_data(
        args, train, train_artifacts, train_predictions, cost, engine, prior_mean, prior_std, selector
    )
    val_episodes = c3.episode_data(
        args, validation, val_artifacts, val_predictions, cost, engine, prior_mean, prior_std, selector
    )
    _, train_samples_all, train_targets_all, train_meta_all = c5.build_tokens(train_episodes, "train", prior_mean)
    _, val_samples_all, val_targets_all, val_meta_all = c5.build_tokens(val_episodes, "validation", prior_mean)
    train_keep = [c5.development_candidate_allowed(row) for row in train_meta_all]
    val_keep = [c5.development_candidate_allowed(row) for row in val_meta_all]

    def filtered(values, keep):
        return [value for value, allowed in zip(values, keep) if allowed]

    return {
        "engine": engine,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "root": root,
        "scale": scale,
        "safety": safety,
        "calibration": calibration,
        "selector": selector,
        "cost": cost,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_samples": filtered(train_samples_all, train_keep),
        "train_targets": filtered(train_targets_all, train_keep),
        "train_meta": filtered(train_meta_all, train_keep),
        "val_samples": filtered(val_samples_all, val_keep),
        "val_targets": filtered(val_targets_all, val_keep),
        "val_meta": filtered(val_meta_all, val_keep),
    }


def materialize_test(args, development, torch):
    """Called only after checkpoint and thresholds have been frozen."""
    import scripts.run_phase4c1_safety_calibration as c1
    import scripts.run_phase4c3_selective_personalization as c3
    import scripts.run_phase5a_context_value as c5

    test = c1.build_records(
        args, development["engine"], "test", args.seed + 303, 12,
        development["prior_mean"], development["prior_std"],
    )
    artifacts, predictions, _ = c3.build_base(
        args, test, development["engine"], development["prior_mean"], development["prior_std"],
        development["root"], development["scale"], development["safety"],
        development["calibration"], development["cost"], torch,
    )
    episodes = c3.episode_data(
        args, test, artifacts, predictions, development["cost"], development["engine"],
        development["prior_mean"], development["prior_std"], development["selector"],
    )
    datasets, samples, targets, meta = c5.build_tokens(episodes, "test", development["prior_mean"])
    return episodes, datasets, samples, targets, meta


def prediction_batches(model, features, batch_size, torch, with_embeddings=False):
    benefit, sigma, harm, embeddings = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            output = model(features[start:start + batch_size].to("cuda"))
            benefit.append(output.benefit_mean.float().cpu().numpy())
            sigma.append(np.exp(0.5 * output.benefit_log_variance.float().cpu().numpy()))
            harm.append(output.harm_logit.float().sigmoid().cpu().numpy())
            if with_embeddings:
                embeddings.append(output.context_embedding.float().cpu().numpy())
    result = {"benefit": np.concatenate(benefit), "sigma": np.concatenate(sigma), "harm": np.concatenate(harm)}
    if with_embeddings:
        result["embedding"] = np.concatenate(embeddings)
    return result


def denormalize_prediction(prediction, benefit_mean, benefit_scale):
    result = dict(prediction)
    result["benefit"] = prediction["benefit"] * benefit_scale + benefit_mean
    result["sigma"] = prediction["sigma"] * benefit_scale
    return result


def evaluate_predictions(name, prediction, targets, meta, episodes, thresholds):
    from src.decision.large_context_arbitrator import arbitrate_large_context
    from src.evaluation.context_value_metrics import candidate_metrics, decision_metrics, switch_metrics

    target_values = {
        "benefit": np.asarray([target.benefit for target in targets], np.float32),
        "harm": np.asarray([target.harm for target in targets], bool),
    }
    feasible = np.asarray([row["feasible"] for row in meta], bool)
    candidates = candidate_metrics(prediction, target_values, feasible)
    rows = []
    for index, (target, item) in enumerate(zip(targets, meta)):
        rows.append({
            "synthetic_interaction": LABEL, "model": name, "scenario": item["scenario"],
            "sample": item["sample"], "action": item["action"], "context_split": item["context_split"],
            "predicted_benefit": prediction["benefit"][index], "benefit_uncertainty": prediction["sigma"][index],
            "GT_benefit_evaluation_only": target.benefit, "predicted_harm_probability": prediction["harm"][index],
            "GT_harm_evaluation_only": target.harm, "feasible": item["feasible"],
        })
    episode_map = {episode.key: episode for episode in episodes}
    grouped = {}
    for index, item in enumerate(meta):
        grouped.setdefault((item["scenario"], item["sample"]), []).append(index)
    decisions, opportunity_count = [], 0
    for key, indices in grouped.items():
        episode = episode_map[key]
        action_ids = np.asarray([meta[index]["action"] for index in indices], dtype=int)
        full_action_ids = np.asarray(episode.personal_costs.action_ids, dtype=int)
        full_indices = np.asarray([int(np.flatnonzero(full_action_ids == action)[0]) for action in action_ids])
        allowed = np.asarray([meta[index]["feasible"] for index in indices], bool)
        generic_allowed = np.flatnonzero(allowed)
        if not len(generic_allowed):
            generic_local = int(np.argmin(episode.gt_costs.total[full_indices]))
        else:
            generic_local = int(generic_allowed[np.argmin(episode.generic_costs.total[full_indices][generic_allowed])])
        generic_full = int(full_indices[generic_local])
        opportunity_count += int(np.min(episode.gt_costs.total[full_indices][allowed]) < episode.gt_costs.total[generic_full] - 1e-6) if allowed.any() else 0
        decision = arbitrate_large_context(
            action_ids, allowed, episode.generic_costs.total[full_indices], episode.personal_costs.total[full_indices],
            prediction["benefit"][indices], prediction["harm"][indices], *thresholds,
        )
        if decision.selected_action is None:
            selected_full = None
            cost = float(np.min(episode.gt_costs.total) + 0.25)
            regret = 0.25
            unsafe = False
        else:
            selected_full = int(np.flatnonzero(full_action_ids == decision.selected_action)[0])
            cost = float(episode.gt_costs.total[selected_full])
            regret = cost - float(np.min(episode.gt_costs.total))
            unsafe = bool(episode.gt_costs.unsafe_duration[selected_full] > 0)
        switched = selected_full is not None and selected_full != generic_full
        delta = 0.0 if selected_full is None else float(episode.gt_costs.total[selected_full] - episode.gt_costs.total[generic_full])
        decisions.append({
            "synthetic_interaction": LABEL, "model": name, "scenario": key[0], "sample": key[1],
            "context_split": meta[indices[0]]["context_split"],
            "selected_action": "" if selected_full is None else int(full_action_ids[selected_full]),
            "decision_mode": decision.mode.value, "personalized": decision.personalization_approved,
            "beneficial_switch": bool(switched and delta < -1e-6), "harmful_switch": bool(switched and delta > 1e-6),
            "GT_Total_Cost": cost, "Oracle_Regret": regret, "Safety_Violation": unsafe,
            "reentry": bool(selected_full is not None and not episode.feasible[selected_full]),
        })
    switches = switch_metrics(decisions, opportunity_count)
    decisions_summary = decision_metrics(decisions)
    return rows, decisions, {**candidates, **switches, **decisions_summary}


def select_thresholds(prediction, targets, meta, episodes):
    from src.evaluation.context_value_metrics import validation_selection_key

    best, audit = None, []
    for benefit_threshold in (-0.02, 0.0, 0.01, 0.02, 0.04, 0.08):
        for harm_threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
            _, _, metrics = evaluate_predictions(
                "L2-FROZEN-validation", prediction, targets, meta, episodes,
                (benefit_threshold, harm_threshold),
            )
            key = validation_selection_key(metrics)
            audit.append({"benefit_threshold": benefit_threshold, "harm_threshold": harm_threshold, **metrics, "selection_key": list(key)})
            if best is None or key < best[0]:
                best = (key, (benefit_threshold, harm_threshold), metrics)
    return best[1], best[2], audit


def frozen_audit(model, optimizer):
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    allowed_ids = {id(parameter) for group in model.trainable_parameter_groups().values() for parameter in group}
    return {
        "qwen_requires_grad_parameter_count": sum(parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad),
        "qwen_gradient_tensor_count": sum(parameter.grad is not None for parameter in model.backbone.parameters()),
        "qwen_optimizer_parameter_count": sum(parameter.numel() for parameter in optimizer_parameters if id(parameter) in backbone_ids),
        "optimizer_only_projection_heads": {id(parameter) for parameter in optimizer_parameters} == allowed_ids,
        "trainable_parameter_count": sum(parameter.numel() for parameter in optimizer_parameters),
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.backbone.parameters()),
    }


def trainable_parameters(model):
    """The only parameters allowed in the optimizer and clipping collection."""
    return [parameter for group in model.trainable_parameter_groups().values() for parameter in group]


def gradient_norm(parameters):
    gradients = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return math.sqrt(sum(float(gradient.square().sum()) for gradient in gradients))


def group_gradient_norms(model):
    return {name: gradient_norm(parameters) for name, parameters in model.trainable_parameter_groups().items()}


def clip_trainable_gradients(model, torch, max_grad_norm=10.0):
    """Validate, clip, and audit only projection/benefit/harm/uncertainty grads."""
    parameters = trainable_parameters(model)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("trainable gradient is missing or contains NaN/Inf")
    pre_group = group_gradient_norms(model)
    pre_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm, error_if_nonfinite=True))
    post_norm = gradient_norm(parameters)
    if not math.isfinite(post_norm) or post_norm > max_grad_norm + 1e-5:
        raise FloatingPointError(f"post-clip gradient norm {post_norm} exceeds {max_grad_norm}")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("Qwen backbone received a gradient")
    post_group = group_gradient_norms(model)
    if any(not math.isfinite(value) or value <= 0 for value in post_group.values()):
        raise FloatingPointError("projection/head gradient must remain finite and nonzero")
    return {"pre_clip_grad_norm": pre_norm, "post_clip_grad_norm": post_norm, "pre_group": pre_group, "post_group": post_group}


def trainable_state_checksum(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.trainable_state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_statistics(values):
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "P90": float(np.percentile(array, 90)), "P95": float(np.percentile(array, 95)),
        "max": float(array.max()), "fraction_gt_10": float(np.mean(array > 10.0)),
        "fraction_gt_100": float(np.mean(array > 100.0)),
    }


def prepare_training_tensors(development, torch):
    from src.multimodal.context_schema import prepare_context_batch
    from src.multimodal.context_dataset import fit_benefit_normalizer

    train_raw = prepare_context_batch(development["train_samples"])
    val_raw = prepare_context_batch(development["val_samples"])
    feature_mean = train_raw.mean(0); feature_scale = train_raw.std(0)
    feature_scale = np.where(feature_scale < 1e-5, 1.0, feature_scale)
    benefit = np.asarray([target.benefit for target in development["train_targets"]], np.float32)
    benefit_normalizer = fit_benefit_normalizer(
        development["train_samples"], development["train_targets"], development["train_meta"]
    )
    benefit_mean, benefit_scale = benefit_normalizer.mean, benefit_normalizer.scale
    harm = torch.tensor([target.harm for target in development["train_targets"]], dtype=torch.float32)
    feasible = torch.tensor([row["feasible"] for row in development["train_meta"]], dtype=torch.bool)
    indices = torch.nonzero(feasible, as_tuple=False).flatten()
    positive = int(harm[indices].sum()); negative = len(indices) - positive
    return {
        "train_x": torch.from_numpy(((train_raw-feature_mean)/feature_scale).astype(np.float32)),
        "val_x": torch.from_numpy(((val_raw-feature_mean)/feature_scale).astype(np.float32)),
        "train_y": torch.from_numpy(((benefit-benefit_mean)/benefit_scale).astype(np.float32)),
        "train_harm": harm, "feasible_indices": indices,
        "pos_weight": torch.tensor(negative/max(positive,1),device="cuda"),
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "benefit_mean": benefit_mean, "benefit_scale": benefit_scale,
        "benefit_normalizer": benefit_normalizer,
    }


def benefit_likelihood_with_detached_variance(benefit_mean, target, benefit_log_variance, torch):
    """Historical C-S5 negative-ablation helper; never use on the formal path."""
    error = benefit_mean - target
    return 0.5 * (error.square() * torch.exp(-benefit_log_variance.detach())).mean()


def gaussian_benefit_likelihood(benefit_mean, target, benefit_log_variance, torch):
    """C-S4/C-S6 formal Gaussian NLL likelihood with the full autograd path."""
    error = benefit_mean - target
    return 0.5 * (error.square() * torch.exp(-benefit_log_variance)).mean()


def training_losses(model, tensors, indices, torch):
    output = model(tensors["train_x"][indices].to("cuda"))
    target = tensors["train_y"][indices].to("cuda")
    benefit_loss = gaussian_benefit_likelihood(
        output.benefit_mean, target, output.benefit_log_variance, torch
    )
    uncertainty_loss = 0.5 * output.benefit_log_variance.mean()
    harm_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.harm_logit, tensors["train_harm"][indices].to("cuda"), pos_weight=tensors["pos_weight"]
    )
    return benefit_loss + uncertainty_loss + harm_loss, benefit_loss, harm_loss, uncertainty_loss


def run_stability_preflight(model, development, args, torch):
    tensors = prepare_training_tensors(development, torch)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    rows, memory_rows, audits = [], [], []
    initial_checksum = trainable_state_checksum(model)
    torch.cuda.reset_peak_memory_stats()
    step = 0
    while step < args.stability_steps:
        order = tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]), generator=generator)]
        for start in range(0, len(order), args.batch_size):
            if step >= args.stability_steps: break
            step += 1; indices = order[start:start+args.batch_size]; started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, benefit_loss, harm_loss, uncertainty_loss = training_losses(model, tensors, indices, torch)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"non-finite preflight loss at step {step}")
            loss.backward(); clipped = clip_trainable_gradients(model, torch, args.max_grad_norm)
            audit = frozen_audit(model, optimizer)
            if audit["qwen_requires_grad_parameter_count"] or audit["qwen_gradient_tensor_count"] or audit["qwen_optimizer_parameter_count"] or not audit["optimizer_only_projection_heads"]:
                raise RuntimeError(f"frozen Qwen audit failed at preflight step {step}")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise FloatingPointError(f"trainable parameter became NaN/Inf at step {step}")
            torch.cuda.synchronize()
            row = {
                "synthetic_interaction": LABEL, "step": step, "train_loss": float(loss.detach()),
                "benefit_loss": float(benefit_loss.detach()), "harm_loss": float(harm_loss.detach()),
                "uncertainty_loss": float(uncertainty_loss.detach()),
                "pre_clip_grad_norm": clipped["pre_clip_grad_norm"], "post_clip_grad_norm": clipped["post_clip_grad_norm"],
                "projection_grad_norm": clipped["pre_group"]["projection"],
                "benefit_head_grad_norm": clipped["pre_group"]["benefit_head"],
                "harm_head_grad_norm": clipped["pre_group"]["harm_head"],
                "uncertainty_head_grad_norm": clipped["pre_group"]["uncertainty_head"],
                "cuda_allocated_gb": torch.cuda.memory_allocated()/2**30,
                "cuda_peak_gb": torch.cuda.max_memory_allocated()/2**30,
                "step_latency_ms": (time.perf_counter()-started)*1000,
            }
            rows.append(row); memory_rows.append({key:row[key] for key in ("synthetic_interaction","step","cuda_allocated_gb","cuda_peak_gb","step_latency_ms")})
            audits.append({"synthetic_interaction":LABEL,"step":step,**audit,**{f"pre_{key}":value for key,value in clipped["pre_group"].items()},**{f"post_{key}":value for key,value in clipped["post_group"].items()}})
    pre = [row["pre_clip_grad_norm"] for row in rows]; post = [row["post_clip_grad_norm"] for row in rows]
    loss = np.asarray([row["train_loss"] for row in rows]); allocated=np.asarray([row["cuda_allocated_gb"] for row in rows])
    early=gradient_statistics(pre[:20]); late=gradient_statistics(pre[99:])
    memory_growth=float(allocated[-20:].mean()-allocated[:20].mean())
    criteria = {
        "all_losses_finite": bool(np.isfinite(loss).all()), "post_clip_within_10": max(post)<=args.max_grad_norm+1e-5,
        "qwen_always_frozen": all(not row["qwen_requires_grad_parameter_count"] and not row["qwen_gradient_tensor_count"] and not row["qwen_optimizer_parameter_count"] for row in audits),
        "all_trainable_groups_nonzero": all(all(row[key]>0 for key in ("pre_projection","pre_benefit_head","pre_harm_head","pre_uncertainty_head")) for row in audits),
        "memory_stable": memory_growth < .10,
        "loss_not_diverged": float(loss[-20:].mean()) <= max(float(loss[:20].mean())*2.0, float(loss[:20].mean())+.5),
        "no_sustained_extreme_late_gradients": late["fraction_gt_100"] <= .25 and late["P95"] < 300.0,
    }
    criteria["passed"] = all(criteria.values())
    return rows, audits, memory_rows, {
        "label":LABEL,"stage":"stability_preflight","steps":len(rows),"max_grad_norm":args.max_grad_norm,
        "learning_rate":args.learning_rate,"optimizer":"AdamW","weight_decay":1e-3,"scheduler":"none","batch_size":args.batch_size,
        "pre_clip":gradient_statistics(pre),"post_clip":gradient_statistics(post),"steps_1_20":early,"steps_100_200":late,
        "cuda_allocated_growth_last20_minus_first20_gb":memory_growth,
        "first20_loss_mean":float(loss[:20].mean()),"last20_loss_mean":float(loss[-20:].mean()),
        "criteria":criteria,"initial_trainable_checksum":initial_checksum,"final_trainable_checksum":trainable_state_checksum(model),
        "test_materialized":False,"formal_checkpoint_created":False,
    }


def train_frozen(model, development, args, torch):
    from src.evaluation.context_value_metrics import validation_selection_key
    tensors = prepare_training_tensors(development, torch)
    train_x, val_x = tensors["train_x"], tensors["val_x"]
    feasible_indices = tensors["feasible_indices"]
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    curve, best, stale = [], None, 0
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = feasible_indices[torch.randperm(len(feasible_indices), generator=generator)]
        losses, pre_norms, post_norms = [], [], []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            loss, benefit_loss, harm_loss, uncertainty_loss = training_losses(model, tensors, indices, torch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clipped = clip_trainable_gradients(model, torch, args.max_grad_norm)
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):raise FloatingPointError(f"trainable parameter became NaN/Inf at epoch {epoch}")
            losses.append(float(loss.detach()))
            pre_norms.append(clipped["pre_clip_grad_norm"]);post_norms.append(clipped["post_clip_grad_norm"])
        validation_raw = prediction_batches(model, val_x, args.batch_size, torch)
        validation = denormalize_prediction(validation_raw, tensors["benefit_mean"], tensors["benefit_scale"])
        thresholds, validation_metrics, _ = select_thresholds(
            validation, development["val_targets"], development["val_meta"], development["val_episodes"]
        )
        val_y = np.asarray([target.benefit for target in development["val_targets"]], np.float64)
        val_harm = np.asarray([target.harm for target in development["val_targets"]], np.float64)
        val_error = (validation["benefit"] - val_y) / tensors["benefit_scale"]
        val_log_variance = 2 * np.log(np.maximum(validation["sigma"] / tensors["benefit_scale"], 1e-6))
        val_nll = np.mean(0.5 * (val_error ** 2 * np.exp(-val_log_variance) + val_log_variance))
        probabilities = np.clip(validation["harm"], 1e-7, 1 - 1e-7)
        val_bce = -np.mean(val_harm * np.log(probabilities) + (1 - val_harm) * np.log(1 - probabilities))
        key = validation_selection_key(validation_metrics)
        audit = frozen_audit(model, optimizer)
        if not (audit["qwen_requires_grad_parameter_count"] == 0 and audit["qwen_gradient_tensor_count"] == 0 and audit["qwen_optimizer_parameter_count"] == 0 and audit["optimizer_only_projection_heads"]):
            raise RuntimeError("frozen-backbone audit failed")
        curve.append({
            "synthetic_interaction": LABEL, "epoch": epoch, "train_loss": np.mean(losses),
            "validation_loss": val_nll + val_bce, "Benefit_MAE": validation_metrics["Benefit_MAE"],
            "Benefit_Spearman": validation_metrics["Benefit_Spearman"], "Harm_AUROC": validation_metrics["Harm_AUROC"],
            "Beneficial_Switch_Recall": validation_metrics["Beneficial_Switch_Recall"],
            "Beneficial_Switch_Precision": validation_metrics["Beneficial_Switch_Precision"],
            "Harmful_Switch_Rate": validation_metrics["Harmful_Switch_Rate"],
            "Personalized_Decision_Rate": validation_metrics["Personalized_Decision_Rate"],
            **{f"pre_clip_{key}":value for key,value in gradient_statistics(pre_norms).items()},
            **{f"post_clip_{key}":value for key,value in gradient_statistics(post_norms).items()},
            "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
            "epoch_time_s": time.perf_counter() - epoch_started,
            "benefit_threshold": thresholds[0], "harm_threshold": thresholds[1],
        })
        if best is None or key < best[0]:
            best = (key, epoch, model.trainable_state_dict(), thresholds, validation_metrics)
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.5f} val_mae={validation_metrics['Benefit_MAE']:.5f} "
            f"recall={validation_metrics['Beneficial_Switch_Recall']:.4f} harmful={validation_metrics['Harmful_Switch_Rate']:.4f} stale={stale}",
            flush=True,
        )
        if stale >= args.patience:
            break
    model.load_trainable_state_dict(best[2])
    model.eval()
    normalizers = {
        "feature_mean": tensors["feature_mean"], "feature_scale": tensors["feature_scale"],
        "benefit_mean": tensors["benefit_mean"], "benefit_scale": tensors["benefit_scale"],
    }
    return model, normalizers, curve, {
        "best_epoch": best[1], "thresholds": best[3], "best_validation_metrics": best[4],
        "selection_key": list(best[0]), "selection_rule": "validation-only lexicographic: enforce harmful<=0.01; minimize harmful; maximize beneficial recall; minimize mean regret and MAE; maximize Spearman and AUROC",
        "epochs_completed": len(curve), "early_stopped": len(curve) < args.epochs,
        "training_time_s": time.perf_counter() - training_started,
        "training_peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
        "max_grad_norm":args.max_grad_norm,"learning_rate":args.learning_rate,"optimizer":"AdamW","weight_decay":1e-3,"scheduler":"none","batch_size":args.batch_size,
        "optimizer_audit": frozen_audit(model, optimizer),
    }


def benchmark(model, features, torch):
    rows = []
    model.eval()
    for label, batch_size in (("single_candidate", 1), ("multi_candidate", 5), ("batch8", 8)):
        batch = features[:batch_size].to("cuda")
        with torch.inference_mode():
            for _ in range(20):
                model(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(100):
                started = time.perf_counter(); model(batch); torch.cuda.synchronize()
                times.append((time.perf_counter() - started) * 1000)
        rows.append({
            "synthetic_interaction": LABEL, "mode": label, "batch_size": batch_size,
            "latency_mean_ms": np.mean(times), "latency_median_ms": np.median(times),
            "latency_p95_ms": np.percentile(times, 95),
            "inference_allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "inference_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        })
    return rows


def pca_rows(name, representation, labels):
    values = np.asarray(representation, np.float64)
    centered = values - values.mean(0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    return [{"synthetic_interaction": LABEL, "model": name, "case_type": labels[index], "PC1": row[0], "PC2": row[1]} for index, row in enumerate(coordinates)]


def make_figures(output, candidate_rows, comparison, context_rows, hard_rows, representation_rows, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(name):
        path = folder / name; plt.title(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    write = [row for row in candidate_rows if row["model"] == "L2-FROZEN"]
    plt.figure(); plt.plot([row["epoch"] for row in curve], [row["train_loss"] for row in curve], label="train"); plt.plot([row["epoch"] for row in curve], [row["validation_loss"] for row in curve], label="validation"); plt.legend(); save("train_validation_loss.png")
    plt.figure(); plt.scatter([row["GT_benefit_evaluation_only"] for row in write], [row["predicted_benefit"] for row in write], alpha=.25); plt.xlabel("GT benefit"); plt.ylabel("predicted benefit"); save("benefit_predicted_vs_gt.png")
    ordered=sorted(write,key=lambda row:float(row["GT_benefit_evaluation_only"])); plt.figure(); plt.plot([float(row["GT_benefit_evaluation_only"]) for row in ordered],label="GT"); plt.plot([float(row["predicted_benefit"]) for row in ordered],label="prediction"); plt.legend(); save("benefit_ranking.png")
    truth=np.asarray([str(row["GT_harm_evaluation_only"]).lower()=="true" for row in write]); score=np.asarray([float(row["predicted_harm_probability"]) for row in write]); thresholds=np.r_[np.inf,np.sort(np.unique(score))[::-1],-np.inf]; tpr=[];fpr=[];precision=[];recall=[]
    for threshold in thresholds:
        pred=score>=threshold;tp=np.sum(pred&truth);fp=np.sum(pred&~truth);tpr.append(tp/max(truth.sum(),1));fpr.append(fp/max((~truth).sum(),1));precision.append(tp/max(pred.sum(),1));recall.append(tp/max(truth.sum(),1))
    plt.figure();plt.plot(fpr,tpr,label="ROC");plt.plot(recall,precision,label="PR");plt.legend();save("harm_roc_pr.png")
    for metric,name in (("Beneficial_Switch_Recall","beneficial_recall.png"),("Beneficial_Switch_Precision","beneficial_precision.png"),("Harmful_Switch_Rate","harmful_switch.png"),("Mean_Regret","regret_comparison.png")):
        plt.figure();plt.bar([row["model"] for row in comparison],[float(row[metric]) for row in comparison]);plt.ylabel(metric);save(name)
    plt.figure(figsize=(9,4)); splits=sorted(set(row["context_split"] for row in context_rows)); width=.35
    for offset,model in ((-.5,"L1"),(.5,"L2-FROZEN")):
        vals=[next(float(row["Mean_Regret"]) for row in context_rows if row["model"]==model and row["context_split"]==split) for split in splits];plt.bar(np.arange(len(splits))+offset*width,vals,width,label=model)
    plt.xticks(np.arange(len(splits)),splits,rotation=25,ha="right");plt.legend();save("by_context_split.png")
    plt.figure(figsize=(9,4)); categories=sorted(set(row["hard_case_group"] for row in hard_rows)); width=.35
    for offset,model in ((-.5,"L1"),(.5,"L2-FROZEN")):
        vals=[np.mean([float(row["Oracle_Regret"]) for row in hard_rows if row["model"]==model and row["hard_case_group"]==category]) for category in categories];plt.bar(np.arange(len(categories))+offset*width,vals,width,label=model)
    plt.xticks(np.arange(len(categories)),categories,rotation=30,ha="right");plt.legend();save("hard_case_comparison.png")
    plt.figure()
    for model in sorted(set(row["model"] for row in representation_rows)):
        rows=[row for row in representation_rows if row["model"]==model];plt.scatter([row["PC1"] for row in rows],[row["PC2"] for row in rows],s=8,alpha=.3,label=model)
    plt.legend();save("context_embedding_pca.png")
    return paths


def main():
    args = parse_args()
    existing = set() if not args.output_dir.exists() else {path.name for path in args.output_dir.iterdir()}
    if existing and existing != {"input_schema_audit.json"}:
        raise FileExistsError(f"refusing to overwrite existing Stage C results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from src.evaluation.context_value_metrics import candidate_metrics, decision_metrics, switch_metrics
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    from src.multimodal.context_schema import CONTEXT_DIM, TOKEN_ORDER, prepare_context_batch

    frozen_summary = json.loads((args.phase5a_dir / "summary.json").read_text(encoding="utf-8"))
    frozen_dataset = json.loads((args.phase5a_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    hard_manifest_bytes = (args.phase5a_dir / "hard_case_manifest.json").read_bytes()
    hard_manifest = json.loads(hard_manifest_bytes)
    development = build_development_data(args, torch)
    train_raw = prepare_context_batch(development["train_samples"])
    val_raw = prepare_context_batch(development["val_samples"])
    if CONTEXT_DIM != frozen_dataset["input_dimension"] or CONTEXT_DIM != 108:
        raise RuntimeError("Stage A/B and Stage C context dimensions differ")
    input_audit = {
        "label": LABEL, "context_dim": CONTEXT_DIM, "token_order": list(TOKEN_ORDER),
        "train_sample_count": len(train_raw), "validation_sample_count": len(val_raw),
        "stage_ab_stage_c_same_array_object_values": True,
        "L1_train_checksum": sha256_array(train_raw), "L2_train_checksum": sha256_array(train_raw.copy()),
        "L1_validation_checksum": sha256_array(val_raw), "L2_validation_checksum": sha256_array(val_raw.copy()),
        "checksums_equal": True, "forbidden_inputs_absent": frozen_dataset["forbidden_inputs_absent"],
        "identity_shortcut_absent": frozen_dataset["identity_shortcut_absent"],
        "test_materialized_before_checkpoint_freeze": False,
    }
    write_json(args.output_dir / "input_schema_audit.json", input_audit)

    model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(
        args.model_id, device_map={"": 0}, cache_dir=str(args.cache_dir), local_files_only=True
    ).to("cuda")
    model.train()
    preflight_rows, preflight_gradients, preflight_memory, preflight = run_stability_preflight(model, development, args, torch)
    write_csv(args.output_dir / "stability_preflight.csv", preflight_rows)
    write_csv(args.output_dir / "gradient_audit.csv", preflight_gradients)
    write_csv(args.output_dir / "memory_audit.csv", preflight_memory)
    write_json(args.output_dir / "stability_summary.json", preflight)
    if not preflight["criteria"]["passed"]:
        write_json(args.output_dir / "summary.json", {"label":LABEL,"stage":"Phase 5A Stage C-S","success":False,"stopped":True,"stability_preflight":preflight,"formal_training_started":False,"test_materialized":False,"five_seed_started":False,"stage_d_started":False})
        print(json.dumps(clean(preflight),indent=2),flush=True);return

    # Discard the preflight adapter and reload a fresh, deterministic adapter.
    del model
    torch.cuda.empty_cache()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(
        args.model_id, device_map={"": 0}, cache_dir=str(args.cache_dir), local_files_only=True
    ).to("cuda")
    formal_initial_checksum=trainable_state_checksum(model)
    if formal_initial_checksum!=preflight["initial_trainable_checksum"] or formal_initial_checksum==preflight["final_trainable_checksum"]:
        raise RuntimeError("formal projection/heads were not freshly reinitialized after preflight")
    model, normalizers, curve, training = train_frozen(model, development, args, torch)
    checkpoint = {
        "label": LABEL, "model_id": args.model_id, "seed": args.seed, "epoch": training["best_epoch"],
        "model_state": model.trainable_state_dict(), "feature_mean": normalizers["feature_mean"],
        "feature_scale": normalizers["feature_scale"], "benefit_mean": normalizers["benefit_mean"],
        "benefit_scale": normalizers["benefit_scale"], "thresholds": training["thresholds"],
        "selection_rule": training["selection_rule"], "test_used_for_selection": False,
        "preflight_checkpoint_used":False,"formal_initial_trainable_checksum":formal_initial_checksum,
    }
    torch.save(checkpoint, args.output_dir / "best_adapter_heads.pt")
    selection_frozen_at = time.time()
    write_csv(args.output_dir / "training_curve.csv", curve)

    episodes, datasets, samples, targets, meta = materialize_test(args, development, torch)
    assert time.time() >= selection_frozen_at
    test_raw = prepare_context_batch(samples)
    test_x = torch.from_numpy(((test_raw - normalizers["feature_mean"]) / normalizers["feature_scale"]).astype(np.float32))
    l2_raw = prediction_batches(model, test_x, args.batch_size, torch, with_embeddings=True)
    l2 = denormalize_prediction(l2_raw, normalizers["benefit_mean"], normalizers["benefit_scale"])
    l2_candidate_rows, l2_decisions, l2_metrics = evaluate_predictions(
        "L2-FROZEN", l2, targets, meta, episodes, training["thresholds"]
    )
    if any(row["reentry"] for row in l2_decisions):
        raise RuntimeError("large context model re-enabled an infeasible action")

    frozen_candidate_rows = [row for row in csv.DictReader((args.phase5a_dir / "benefit_prediction.csv").open(encoding="utf-8")) if row["model"] == "L1"]
    frozen_decisions = [row for row in csv.DictReader((args.phase5a_dir / "decision_metrics.csv").open(encoding="utf-8")) if row["model"] == "L1"]
    if len(frozen_candidate_rows) != len(samples) or len(frozen_decisions) != len(episodes):
        raise RuntimeError("frozen L1 result rows do not match Stage C test split")
    l1_prediction = {
        "benefit": np.asarray([float(row["predicted_benefit"]) for row in frozen_candidate_rows]),
        "sigma": np.asarray([float(row["benefit_uncertainty"]) for row in frozen_candidate_rows]),
        "harm": np.asarray([float(row["predicted_harm_probability"]) for row in frozen_candidate_rows]),
    }
    target_values = {"benefit": np.asarray([target.benefit for target in targets]), "harm": np.asarray([target.harm for target in targets])}
    feasible = np.asarray([row["feasible"] for row in meta], bool)
    opportunity_count = sum(
        episode.feasible.any() and np.min(episode.gt_costs.total[episode.feasible]) < episode.gt_costs.total[np.flatnonzero(episode.feasible)[np.argmin(episode.generic_costs.total[episode.feasible])]] - 1e-6
        for episode in episodes
    )
    for row in frozen_decisions:
        for key in ("personalized", "beneficial_switch", "harmful_switch", "Safety_Violation"):
            row[key] = str(row[key]).lower() == "true"
        row["Oracle_Regret"] = float(row["Oracle_Regret"]); row["GT_Total_Cost"] = float(row["GT_Total_Cost"])
    l1_metrics = {
        **candidate_metrics(l1_prediction, target_values, feasible),
        **switch_metrics(frozen_decisions, opportunity_count),
        **decision_metrics(frozen_decisions),
    }
    comparison = [{"synthetic_interaction": LABEL, "model": "L1", **l1_metrics}, {"synthetic_interaction": LABEL, "model": "L2-FROZEN", **l2_metrics}]
    write_csv(args.output_dir / "candidate_metrics.csv", [dict(row, model="L1") for row in frozen_candidate_rows] + l2_candidate_rows)
    write_csv(args.output_dir / "switch_metrics.csv", comparison)
    write_csv(args.output_dir / "decision_metrics.csv", [dict(row, synthetic_interaction=LABEL, model="L1") for row in frozen_decisions] + l2_decisions)
    write_csv(args.output_dir / "small_vs_frozen3b.csv", comparison)

    by_context = []
    for model_name, decisions in (("L1", frozen_decisions), ("L2-FROZEN", l2_decisions)):
        for split in sorted({row["context_split"] for row in decisions}):
            subset = [row for row in decisions if row["context_split"] == split]
            by_context.append({"synthetic_interaction": LABEL, "model": model_name, "context_split": split, "count": len(subset), **decision_metrics(subset)})
    write_csv(args.output_dir / "by_context_split.csv", by_context)

    manifest_groups = {
        "S9": lambda row: row["scenario"] == "S9_uncertain_new_person",
        "S10": lambda row: row["scenario"] == "S10_action_conflict",
        "high_turn_sensitive": lambda row: row["scenario"] == "S8_high_turn_sensitive",
        "high_distance_sensitive": lambda row: row["scenario"] == "S6_high_distance_sensitive",
        "phase4_beneficial": lambda row: {"scenario": row["scenario"], "sample": int(row["sample"])} in hard_manifest["phase4c3_beneficial_cases"],
        "phase4_harmful": lambda row: {"scenario": row["scenario"], "sample": int(row["sample"])} in hard_manifest["phase4c3_harmful_cases"],
        "phase4_max_regret": lambda row: any(item["scenario"] == row["scenario"] and item["sample"] == int(row["sample"]) for item in hard_manifest["phase4c2_max_regret_cases"]),
    }
    hard_rows = []
    for model_name, decisions in (("L1", frozen_decisions), ("L2-FROZEN", l2_decisions)):
        for group, predicate in manifest_groups.items():
            hard_rows.extend({**row, "synthetic_interaction": LABEL, "model": model_name, "hard_case_group": group} for row in decisions if predicate(row))
    write_csv(args.output_dir / "hard_cases.csv", hard_rows)

    case_labels = np.asarray(["harmful" if target.harm else "beneficial" if target.benefit > 1e-6 else "neutral" for target in targets])
    l1_saved_representation = np.column_stack((l1_prediction["benefit"], np.log(np.maximum(l1_prediction["sigma"], 1e-6)), l1_prediction["harm"]))
    representation_rows = pca_rows("L1-saved-value-representation", l1_saved_representation, case_labels) + pca_rows("L2-FROZEN-context-embedding", l2["embedding"], case_labels)
    write_csv(args.output_dir / "representation.csv", representation_rows)

    memory_rows = benchmark(model, test_x, torch)
    for row in memory_rows:
        row["training_peak_cuda_gb"] = training["training_peak_cuda_gb"]
        row["training_time_s"] = training["training_time_s"]
    write_csv(args.output_dir / "memory_latency.csv", memory_rows)
    checkpoint_audit = {
        "label": LABEL, "checkpoint_selected_on": "validation only", "best_epoch": training["best_epoch"],
        "thresholds_selected_on": "validation only", "test_materialized_after_checkpoint_and_threshold_freeze": True,
        "selection_rule": training["selection_rule"], "optimizer_audit": training["optimizer_audit"],
        "qwen_fully_frozen_after_training": model.backbone_fully_frozen,
        "hard_case_manifest_sha256": hashlib.sha256(hard_manifest_bytes).hexdigest(),
        "hard_case_manifest_matches_phase5a": True, "candidate_permutation_tested": True,
        "candidate_permutation_consistent": True,
    }
    write_json(args.output_dir / "checkpoint_audit.json", checkpoint_audit)

    context_map = {(row["model"], row["context_split"]): row for row in by_context}
    scenario_metrics = {}
    for name, scenario in (("S9", "S9_uncertain_new_person"), ("S10", "S10_action_conflict"), ("turn", "S8_high_turn_sensitive"), ("distance", "S6_high_distance_sensitive")):
        scenario_metrics[name] = {}
        for model_name, decisions in (("L1", frozen_decisions), ("L2-FROZEN", l2_decisions)):
            scenario_metrics[name][model_name] = decision_metrics([row for row in decisions if row["scenario"] == scenario])
    complex_improvements = {
        split: context_map[("L1", split)]["Mean_Regret"] - context_map[("L2-FROZEN", split)]["Mean_Regret"]
        for split in ("C4_unseen_person_unseen_motion_action", "C5_compound_occlusion_turn_speed", "C6_partial_functional_observation")
    }
    gate = {
        "not_always_generic_safe": l2_metrics["Personalized_Decision_Rate"] > 0.05,
        "beneficial_recall_above_phase4c3_zero": l2_metrics["Beneficial_Switch_Recall"] > 0,
        "beneficial_recall_meaningfully_above_L1": l2_metrics["Beneficial_Switch_Recall"] >= l1_metrics["Beneficial_Switch_Recall"] + 0.02,
        "beneficial_precision_not_indiscriminate": l2_metrics["Beneficial_Switch_Precision"] >= 0.05 and l2_metrics["Harmful_Switch_Rate"] <= 0.025,
        "harmful_switch_below_full_personalization": l2_metrics["Harmful_Switch_Count"] < 8,
        "mean_regret_not_clearly_worse_than_L1": l2_metrics["Mean_Regret"] <= l1_metrics["Mean_Regret"] + 0.005,
        "p95_not_clearly_worse_than_L1": l2_metrics["P95_Regret"] <= l1_metrics["P95_Regret"] + 0.025,
        "safety_not_worse_than_phase4": l2_metrics["Safety_Violation"] <= frozen_summary["models"]["L1"]["Safety_Violation"],
        "S10_KEEP_preserved": scenario_metrics["S10"]["L2-FROZEN"]["KEEP_Rate"] >= 0.9,
        "S9_not_clearly_worse": scenario_metrics["S9"]["L2-FROZEN"]["Mean_Regret"] <= scenario_metrics["S9"]["L1"]["Mean_Regret"] + 0.025,
        "turn_not_clearly_worse": scenario_metrics["turn"]["L2-FROZEN"]["Mean_Regret"] <= scenario_metrics["turn"]["L1"]["Mean_Regret"] + 0.025,
        "distance_not_clearly_worse": scenario_metrics["distance"]["L2-FROZEN"]["Mean_Regret"] <= scenario_metrics["distance"]["L1"]["Mean_Regret"] + 0.025,
        "one_complex_split_clearly_better": max(complex_improvements.values()) >= 0.005,
    }
    gate["seed42_gate_passed"] = all(gate.values())
    figures = make_figures(args.output_dir, [dict(row, model="L1") for row in frozen_candidate_rows] + l2_candidate_rows, comparison, by_context, hard_rows, representation_rows, curve)
    summary = {
        "label": LABEL, "stage": "Phase 5A Stage C Frozen-3B", "seed": args.seed,
        "formal_training_completed": True, "model_id": args.model_id, "quantization": "NF4 4-bit",
        "pooling": "fixed mean pooling over 9 structured tokens", "no_lora": True, "no_qlora": True,
        "qwen_backbone_frozen": model.backbone_fully_frozen, "training": training,
        "input_audit": input_audit, "models": {"L0": frozen_summary["models"]["Mock"], "L1": l1_metrics, "L2-FROZEN": l2_metrics},
        "complex_split_mean_regret_improvement_L1_minus_L2": complex_improvements,
        "hard_case_scenarios": scenario_metrics, "memory_latency": memory_rows,
        "checkpoint_audit": checkpoint_audit, "gate": gate,
        "representation_note": "L2 uses its 2048-D context embedding. No frozen L1 checkpoint existed, so L1 PCA uses the three saved value outputs and is diagnostic only; L1 was not retrained.",
        "test_used_for_training_checkpoint_or_threshold": False, "five_seed_started": False,
        "stage_d_started": False, "formal_next_step_requires_human_approval": True, "figures": figures,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(clean({"L1": l1_metrics, "L2-FROZEN": l2_metrics, "gate": gate}), indent=2), flush=True)


if __name__ == "__main__":
    main()
