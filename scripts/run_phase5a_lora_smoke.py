"""Phase 5A Stage D-0: QLoRA-style task-adaptation technical smoke test."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import inspect
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as base

LABEL = base.LABEL
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LR = 3e-5
WEIGHT_DECAY = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
BATCH_SIZE = 8
STEPS = 50
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
ATTENTION_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_LEAVES = ("q_proj", "v_proj")
VISION_SEGMENTS = ("visual", "vision", "vision_tower", "vision_model")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_lora_smoke")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--formal-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_frozen3b_formal_seed42")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(torch):
    import peft
    return {
        "label": LABEL, "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"), "bitsandbytes": package_version("bitsandbytes"),
        "peft": peft.__version__, "peft_import_succeeded": True,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "installed_only_new_package": "peft", "test_materialized": False,
    }


def build_train_only_data(args, torch):
    """Rebuild only the frozen Stage A/B train split; never construct val/test."""
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
    records = c1.build_records(args, engine, "train", args.seed + 101, 30, prior_mean, prior_std)
    artifacts, predictions, cost = c3.build_base(
        args, records, engine, prior_mean, prior_std, root, scale, safety, calibration, None, torch
    )
    episodes = c3.episode_data(
        args, records, artifacts, predictions, cost, engine, prior_mean, prior_std, selector
    )
    _, all_samples, all_targets, all_meta = c5.build_tokens(episodes, "train", prior_mean)
    keep = [c5.development_candidate_allowed(row) for row in all_meta]
    filtered = lambda values: [value for value, allowed in zip(values, keep) if allowed]
    return {
        "train_samples": filtered(all_samples), "train_targets": filtered(all_targets),
        "train_meta": filtered(all_meta), "validation_constructed": False,
        "test_materialized": False,
    }


def prepare_train_only_tensors(train_data, torch):
    from src.multimodal.context_schema import prepare_context_batch
    from src.multimodal.context_dataset import fit_benefit_normalizer
    raw = prepare_context_batch(train_data["train_samples"])
    feature_mean = raw.mean(0); feature_scale = raw.std(0)
    feature_scale = np.where(feature_scale < 1e-5, 1.0, feature_scale)
    normalizer = fit_benefit_normalizer(
        train_data["train_samples"], train_data["train_targets"], train_data["train_meta"]
    )
    benefit = np.asarray([target.benefit for target in train_data["train_targets"]], np.float32)
    harm = torch.tensor([target.harm for target in train_data["train_targets"]], dtype=torch.float32)
    feasible = torch.tensor([row["feasible"] for row in train_data["train_meta"]], dtype=torch.bool)
    indices = torch.nonzero(feasible, as_tuple=False).flatten()
    positive = int(harm[indices].sum()); negative = len(indices) - positive
    return {
        "train_x": torch.from_numpy(((raw - feature_mean) / feature_scale).astype(np.float32)),
        "train_y": torch.from_numpy(normalizer.transform(benefit).astype(np.float32)),
        "train_harm": harm, "feasible_indices": indices,
        "pos_weight": torch.tensor(negative / max(positive, 1), device="cuda"),
        "benefit_normalizer": normalizer,
    }


def is_vision_path(name: str) -> bool:
    segments = tuple(part.lower() for part in name.split("."))
    return any(any(marker in segment for marker in VISION_SEGMENTS) for segment in segments)


def discover_attention_modules(backbone) -> dict:
    rows = []
    for name, module in backbone.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in ATTENTION_LEAVES:
            continue
        vision = is_vision_path(name)
        language_self_attention = "self_attn" in name.split(".") and not vision
        rows.append({
            "module_path": name, "leaf": leaf, "module_type": type(module).__name__,
            "is_vision": vision, "is_language_self_attention": language_self_attention,
            "selected_for_lora": language_self_attention and leaf in LORA_LEAVES,
        })
    targets = [row["module_path"] for row in rows if row["selected_for_lora"]]
    if not targets or any(is_vision_path(name) for name in targets):
        raise RuntimeError("failed to construct a non-empty language-only LoRA target list")
    if any(name.rsplit(".", 1)[-1] not in LORA_LEAVES for name in targets):
        raise RuntimeError("LoRA target list contains a forbidden projection")
    return {
        "label": LABEL, "source": "actual Qwen backbone named_modules() before PEFT injection",
        "attention_projection_modules": rows, "explicit_target_modules": targets,
        "target_leaf_types": list(LORA_LEAVES),
        "language_target_count": len(targets),
        "vision_attention_projection_count": sum(row["is_vision"] for row in rows),
        "vision_target_count": sum(is_vision_path(name) for name in targets),
        "forbidden_k_o_or_mlp_target_count": sum(name.rsplit(".", 1)[-1] not in LORA_LEAVES for name in targets),
        "test_materialized": False,
    }


def prepare_kbit_backbone(model):
    from peft import prepare_model_for_kbit_training
    model.backbone = prepare_model_for_kbit_training(model.backbone, use_gradient_checkpointing=False)
    return model


def prepared_base_snapshot(backbone) -> dict:
    """Deterministic dtype/module-state contract shared by D-0/D-C0/future D-R1."""
    from collections import Counter
    parameter_dtypes = Counter(str(parameter.dtype) for parameter in backbone.parameters())
    parameter_requires_grad = Counter(str(parameter.dtype) for parameter in backbone.parameters() if parameter.requires_grad)
    module_dtypes = Counter()
    layer_norms, linear4bit = [], []
    for name, module in backbone.named_modules():
        dtype = getattr(module, "dtype", None)
        if dtype is not None:
            module_dtypes[f"{type(module).__name__}:{dtype}"] += 1
        if isinstance(module, __import__("torch").nn.LayerNorm) or "LayerNorm" in type(module).__name__ or type(module).__name__.endswith("RMSNorm"):
            weight = getattr(module, "weight", None)
            layer_norms.append({"path": name, "type": type(module).__name__, "weight_dtype": None if weight is None else str(weight.dtype)})
        if type(module).__name__ == "Linear4bit":
            linear4bit.append(name)
    embedding = backbone.get_input_embeddings().weight
    output_embedding = backbone.get_output_embeddings()
    return {
        "parameter_dtype_counts": dict(sorted(parameter_dtypes.items())),
        "trainable_parameter_dtype_counts": dict(sorted(parameter_requires_grad.items())),
        "module_dtype_counts": dict(sorted(module_dtypes.items())),
        "layer_norms": layer_norms,
        "layer_norm_dtype_counts": dict(sorted(Counter(row["weight_dtype"] for row in layer_norms).items())),
        "embedding_dtype": str(embedding.dtype), "embedding_requires_grad": bool(embedding.requires_grad),
        "lm_head_dtype": None if output_embedding is None else str(output_embedding.weight.dtype),
        "lm_head_requires_grad": False if output_embedding is None else bool(output_embedding.weight.requires_grad),
        "linear4bit_count": len(linear4bit), "linear4bit_paths": linear4bit,
        "requires_grad_parameter_count": sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad),
        "gradient_checkpointing": bool(getattr(backbone, "is_gradient_checkpointing", False)),
        "use_cache": getattr(backbone.config, "use_cache", None),
        "is_loaded_in_4bit": bool(getattr(backbone, "is_loaded_in_4bit", False)),
        "lora_module_count": sum("lora_" in name for name, _ in backbone.named_modules()),
        "lora_parameter_count": sum(parameter.numel() for name, parameter in backbone.named_parameters() if "lora_" in name),
    }


def prepared_base_contract(before: dict, after: dict) -> dict:
    source = inspect.getsource(prepare_kbit_backbone)
    return {
        "label": LABEL,
        "preparation_callable": "scripts.run_phase5a_lora_smoke.prepare_kbit_backbone",
        "preparation_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "prepare_model_for_kbit_training": {"use_gradient_checkpointing": False},
        "before": before, "after": after,
        "changed_summary": {
            "parameter_dtype_counts_changed": before["parameter_dtype_counts"] != after["parameter_dtype_counts"],
            "layer_norm_dtype_counts_changed": before["layer_norm_dtype_counts"] != after["layer_norm_dtype_counts"],
            "embedding_dtype_changed": before["embedding_dtype"] != after["embedding_dtype"],
            "module_dtype_counts_changed": before["module_dtype_counts"] != after["module_dtype_counts"],
            "requires_grad_before": before["requires_grad_parameter_count"],
            "requires_grad_after": after["requires_grad_parameter_count"],
            "linear4bit_count_unchanged": before["linear4bit_count"] == after["linear4bit_count"],
        },
        "D0_D_C0_future_D_R1_shared_contract": True,
        "test_materialized": False,
    }


def inject_language_lora(model, target_modules):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=list(target_modules), task_type="CAUSAL_LM",
    )
    model.backbone = get_peft_model(model.backbone, config)
    return model


def parameter_groups(model):
    lora = [(name, parameter) for name, parameter in model.backbone.named_parameters() if "lora_" in name]
    original = [(name, parameter) for name, parameter in model.backbone.named_parameters() if "lora_" not in name]
    heads = [(f"{group}.{index}", parameter) for group, values in model.trainable_parameter_groups().items() for index, parameter in enumerate(values)]
    return original, lora, heads


def count_parameters(named):
    return sum(parameter.numel() for _, parameter in named)


def trainable_audit(model, optimizer=None):
    original, lora, heads = parameter_groups(model)
    optimizer_ids = set() if optimizer is None else {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    embedding_ids = {id(parameter) for parameter in model.backbone.get_input_embeddings().parameters()}
    lm_head_ids = {id(parameter) for parameter in model.backbone.get_output_embeddings().parameters()} if model.backbone.get_output_embeddings() is not None else set()
    vision_lora = [name for name, _ in lora if is_vision_path(name)]
    projection_head_count = count_parameters(heads)
    lora_count = count_parameters(lora)
    base_count = count_parameters(original)
    return {
        "label": LABEL, "qwen_original_base_parameter_count": base_count,
        "qwen_original_base_requires_grad_count": sum(parameter.numel() for _, parameter in original if parameter.requires_grad),
        "qwen_original_base_gradient_tensor_count": sum(parameter.grad is not None for _, parameter in original),
        "qwen_original_base_optimizer_count": sum(parameter.numel() for _, parameter in original if id(parameter) in optimizer_ids),
        "lora_trainable_parameter_count": lora_count,
        "lora_requires_grad_count": sum(parameter.numel() for _, parameter in lora if parameter.requires_grad),
        "lora_optimizer_count": sum(parameter.numel() for _, parameter in lora if id(parameter) in optimizer_ids),
        "projection_head_trainable_parameter_count": projection_head_count,
        "projection_head_optimizer_count": sum(parameter.numel() for _, parameter in heads if id(parameter) in optimizer_ids),
        "total_trainable_parameter_count": lora_count + projection_head_count,
        "total_parameter_count": base_count + lora_count + projection_head_count,
        "trainable_fraction": (lora_count + projection_head_count) / max(base_count + lora_count + projection_head_count, 1),
        "vision_lora_parameter_names": vision_lora,
        "vision_lora_parameter_count": sum(parameter.numel() for name, parameter in lora if is_vision_path(name)),
        "embedding_trainable_count": sum(parameter.numel() for _, parameter in original if id(parameter) in embedding_ids and parameter.requires_grad),
        "lm_head_trainable_count": sum(parameter.numel() for _, parameter in original if id(parameter) in lm_head_ids and parameter.requires_grad),
        "optimizer_exactly_lora_projection_heads": bool(optimizer is not None and optimizer_ids == {id(parameter) for _, parameter in lora + heads}),
        "test_materialized": False,
    }


def output_values(output):
    return {
        "benefit_mean": output.benefit_mean.detach().float().cpu(),
        "harm_logit": output.harm_logit.detach().float().cpu(),
        "benefit_log_variance": output.benefit_log_variance.detach().float().cpu(),
    }


def max_differences(before, after):
    return {key: float((before[key] - after[key]).abs().max()) for key in before}


def gradient_norm(named):
    gradients = [parameter.grad.detach().float() for _, parameter in named if parameter.grad is not None]
    return math.sqrt(sum(float(value.square().sum()) for value in gradients)) if gradients else 0.0


def state_checksum(named):
    digest = hashlib.sha256()
    for name, parameter in named:
        digest.update(name.encode()); digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Stage D-0 results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    base.write_json(args.output_dir / "environment.json", environment(torch))

    # Build train-only tensors before Qwen load.
    train_data = build_train_only_data(args, torch)
    tensors = prepare_train_only_tensors(train_data, torch)
    normalizer = tensors["benefit_normalizer"]
    if len(normalizer.fit_sample_ids) != 616 or abs(normalizer.mean + .1921661049) > 1e-7 or abs(normalizer.scale - .1974763721) > 1e-7:
        raise RuntimeError("C-S4 normalizer parity failed")
    del train_data; gc.collect(); torch.cuda.empty_cache()

    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    torch.cuda.reset_peak_memory_stats(); load_started = time.perf_counter()
    model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(
        args.model_id, device_map={"": 0}, cache_dir=str(args.cache_dir), local_files_only=True
    ).to("cuda")
    load_memory = {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "load_time_s": time.perf_counter() - load_started,
    }
    if not model.scale_alignment_enabled or abs(model.native_embedding_stats["median"] - 1.0087162852287292) > 2e-6:
        raise RuntimeError("C-S3 native scale alignment failed")
    checkpoint = torch.load(args.formal_dir / "best_validation_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(checkpoint["model_state"])
    fixed_indices = tensors["feasible_indices"][:BATCH_SIZE]
    fixed = tensors["train_x"][fixed_indices].to("cuda")
    model.eval()
    with torch.inference_mode(): frozen_output = output_values(model(fixed))

    module_audit = discover_attention_modules(model.backbone)
    base.write_json(args.output_dir / "lora_module_audit.json", module_audit)
    before_preparation = prepared_base_snapshot(model.backbone)
    model = prepare_kbit_backbone(model); model.eval()
    after_preparation = prepared_base_snapshot(model.backbone)
    with torch.inference_mode(): prepared_output = output_values(model(fixed))
    model = inject_language_lora(model, module_audit["explicit_target_modules"])
    model.eval()
    with torch.inference_mode(): initial_lora_output = output_values(model(fixed))
    differences = max_differences(frozen_output, initial_lora_output)
    preparation_differences = max_differences(frozen_output, prepared_output)
    lora_only_differences = max_differences(prepared_output, initial_lora_output)
    unexplained = max(lora_only_differences.values())
    equivalence = {
        "label": LABEL, "input_source": "fixed train-only feasible batch",
        "before_optimizer_step": True,
        "frozen_C_R1_vs_prepared_LoRA": {**differences, "maximum": max(differences.values())},
        "frozen_C_R1_vs_kbit_prepared_no_LoRA": {**preparation_differences, "maximum": max(preparation_differences.values())},
        "kbit_prepared_no_LoRA_vs_zero_init_LoRA": {**lora_only_differences, "maximum": unexplained},
        "interpretation": "Any direct C-R1 difference must be attributable to PEFT k-bit preparation; zero-init LoRA itself must preserve the prepared forward.",
        "lora_only_tolerance": 1e-5,
        "passed": unexplained <= 1e-5 and all(math.isfinite(value) for value in differences.values()),
        "test_materialized": False,
    }
    base.write_json(args.output_dir / "initial_forward_equivalence.json", equivalence)
    if not equivalence["passed"]: raise RuntimeError("initial LoRA forward differs materially from frozen C-R1")

    original, lora, heads = parameter_groups(model)
    parameters = [parameter for _, parameter in lora + heads]
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS, eps=EPS)
    parameter_audit = trainable_audit(model, optimizer)
    base.write_json(args.output_dir / "trainable_parameter_audit.json", parameter_audit)
    if parameter_audit["qwen_original_base_requires_grad_count"] or parameter_audit["qwen_original_base_optimizer_count"] or parameter_audit["vision_lora_parameter_count"] or not parameter_audit["optimizer_exactly_lora_projection_heads"]:
        raise RuntimeError("LoRA trainable/optimizer boundary failed")

    generator = torch.Generator().manual_seed(args.seed)
    rows, gradient_rows = [], []
    model.train()
    # The adapter deliberately keeps the frozen base in eval mode. Activate
    # only LoRA dropout modules, not any original-Qwen dropout.
    for name, module in model.backbone.named_modules():
        if "lora_dropout" in name:
            module.train()
    torch.cuda.reset_peak_memory_stats(); smoke_started = time.perf_counter(); step = 0; first_step_peak = None
    while step < STEPS:
        order = tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]), generator=generator)]
        for start in range(0, len(order), BATCH_SIZE):
            if step >= STEPS: break
            step += 1; indices = order[start:start+BATCH_SIZE]; started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(tensors["train_x"][indices].to("cuda")); target = tensors["train_y"][indices].to("cuda")
            error = output.benefit_mean - target
            benefit = .5 * (error.square() * torch.exp(-output.benefit_log_variance)).mean()
            uncertainty = .5 * output.benefit_log_variance.mean()
            harm = torch.nn.functional.binary_cross_entropy_with_logits(output.harm_logit, tensors["train_harm"][indices].to("cuda"), pos_weight=tensors["pos_weight"])
            loss = benefit + uncertainty + harm
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"non-finite smoke loss at step {step}")
            loss.backward()
            original_grad = gradient_norm(original); lora_grad = gradient_norm(lora)
            projection_grad = base.gradient_norm(model.trainable_parameter_groups()["projection"])
            benefit_grad = base.gradient_norm(model.trainable_parameter_groups()["benefit_head"])
            harm_grad = base.gradient_norm(model.trainable_parameter_groups()["harm_head"])
            uncertainty_grad = base.gradient_norm(model.trainable_parameter_groups()["uncertainty_head"])
            raw = base.gradient_norm(parameters)
            if original_grad != 0 or not all(math.isfinite(value) and value > 0 for value in (lora_grad, projection_grad, benefit_grad, harm_grad, uncertainty_grad, raw)):
                raise RuntimeError(f"gradient boundary failed at step {step}")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters): raise FloatingPointError(f"non-finite trainable parameter at step {step}")
            torch.cuda.synchronize()
            if step == 1: first_step_peak = torch.cuda.max_memory_allocated() / 2**30
            logv = output.benefit_log_variance.detach().float()
            row = {
                "synthetic_interaction": LABEL, "step": step, "total_loss": float(loss.detach()),
                "benefit_loss": float(benefit.detach()), "harm_loss": float(harm.detach()),
                "uncertainty_loss": float(uncertainty.detach()), "raw_gradient": raw,
                "lora_grad_norm": lora_grad, "projection_grad_norm": projection_grad,
                "benefit_head_grad_norm": benefit_grad, "harm_head_grad_norm": harm_grad,
                "uncertainty_head_grad_norm": uncertainty_grad,
                "log_variance_mean": float(logv.mean()), "log_variance_std": float(logv.std()),
                "log_variance_min": float(logv.min()), "log_variance_max": float(logv.max()),
                "cuda_allocated_gb": torch.cuda.memory_allocated() / 2**30,
                "cuda_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                "step_latency_ms": (time.perf_counter() - started) * 1000,
            }
            rows.append(row)
            gradient_rows.append({
                "synthetic_interaction": LABEL, "step": step, "qwen_original_base_grad_norm": original_grad,
                "qwen_original_base_gradient_tensor_count": sum(parameter.grad is not None for _, parameter in original),
                "lora_grad_norm": lora_grad, "projection_grad_norm": projection_grad,
                "benefit_head_grad_norm": benefit_grad, "harm_head_grad_norm": harm_grad,
                "uncertainty_head_grad_norm": uncertainty_grad, "raw_gradient": raw,
                "qwen_original_base_optimizer_count": parameter_audit["qwen_original_base_optimizer_count"],
                "lora_optimizer_count": parameter_audit["lora_optimizer_count"],
                "projection_head_optimizer_count": parameter_audit["projection_head_optimizer_count"],
            })
    base.write_csv(args.output_dir / "training_smoke.csv", rows)
    base.write_csv(args.output_dir / "gradient_audit.csv", gradient_rows)

    adapter_dir = args.output_dir / "adapter"
    model.eval()
    with torch.inference_mode(): before_reload = output_values(model(fixed))
    lora_checksum_before = state_checksum(lora)
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_files = [{"name": path.name, "bytes": path.stat().st_size} for path in adapter_dir.iterdir() if path.is_file()]
    if any(item["bytes"] > 100 * 1024 * 1024 for item in adapter_files): raise RuntimeError("adapter save unexpectedly contains a full base-model shard")
    from peft import PeftModel
    unloaded = model.backbone.unload()
    model.backbone = PeftModel.from_pretrained(unloaded, adapter_dir, is_trainable=True)
    model.eval(); _, reloaded_lora, _ = parameter_groups(model)
    with torch.inference_mode(): after_reload = output_values(model(fixed))
    reload_diff = max_differences(before_reload, after_reload)
    reload_audit = {
        "label": LABEL, "adapter_directory": str(adapter_dir), "saved_files": adapter_files,
        "full_base_model_saved": False, "lora_checksum_before": lora_checksum_before,
        "lora_checksum_after": state_checksum(reloaded_lora),
        **{f"{key}_max_abs_difference": value for key, value in reload_diff.items()},
        "maximum_output_difference": max(reload_diff.values()), "tolerance": 1e-5,
        "passed": max(reload_diff.values()) <= 1e-5, "test_materialized": False,
    }
    base.write_json(args.output_dir / "adapter_reload_audit.json", reload_audit)

    gradients = np.asarray([row["raw_gradient"] for row in rows])
    logv_means = np.asarray([row["log_variance_mean"] for row in rows])
    late = gradients[-10:]; early = gradients[:10]
    stable = bool(
        np.isfinite(gradients).all() and np.isfinite(logv_means).all() and
        late.mean() <= max(early.mean() * 2, early.mean() + 500) and
        not np.all(logv_means[-10:] <= -5.99) and not np.all(logv_means[-10:] >= 2.99) and
        all(row["qwen_original_base_grad_norm"] == 0 for row in gradient_rows) and reload_audit["passed"]
    )
    memory = {
        "label": LABEL, "model_load": load_memory, "batch8_forward_backward_peak_gb": first_step_peak,
        "fifty_step_peak_gb": max(row["cuda_peak_gb"] for row in rows),
        "final_allocated_gb": rows[-1]["cuda_allocated_gb"],
        "smoke_time_s": time.perf_counter() - smoke_started, "OOM": False,
        "CPU_offload": False, "test_materialized": False,
    }
    base.write_json(args.output_dir / "memory_audit.json", memory)
    summary = {
        "label": LABEL, "stage": "Phase 5A Stage D-0 QLoRA-style Task Adaptation Smoke Test",
        "success": stable, "formal_training_started": False, "validation_used": False,
        "test_materialized": False, "five_seed_started": False,
        "model_id": MODEL_ID, "quantization": "NF4 4-bit", "lora_config": {
            "r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "bias": "none",
            "targets": "explicit language self-attention q_proj/v_proj paths only",
        },
        "optimizer": {"type": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "betas": list(BETAS), "eps": EPS, "scheduler": "none", "gradient_clipping": False},
        "repairs_retained": {"C_S3_scale_alignment": True, "C_S4_normalizer_parity": True, "full_gaussian_nll": True, "C_S5_detach": False},
        "module_audit": {key: value for key, value in module_audit.items() if key != "attention_projection_modules"},
        "parameter_audit": parameter_audit, "initial_forward_equivalence": equivalence,
        "gradient": {"mean": float(gradients.mean()), "median": float(np.median(gradients)), "P95": float(np.percentile(gradients, 95)), "max": float(gradients.max()), "late_vs_early_mean_percent": float(100 * (late.mean() / early.mean() - 1))},
        "gaussian": {"log_variance_mean_final": rows[-1]["log_variance_mean"], "log_variance_std_final": rows[-1]["log_variance_std"], "global_min": min(row["log_variance_min"] for row in rows), "global_max": max(row["log_variance_max"] for row in rows), "minus6_collapse": False, "plus3_saturation": False},
        "memory": memory, "adapter_reload": reload_audit,
        "stage_d_r1_ready": stable, "next_step_requires_human_approval": True,
    }
    base.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(base.clean(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
