"""Stage C-0: frozen Qwen2.5-VL-3B 4-bit structured-context smoke test."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import math
import platform
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
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_stagec")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
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


def load_train_features(path, seed):
    """Create small train-shape inputs; never read Phase 5 test predictions."""
    from src.multimodal.context_schema import CONTEXT_DIM, prepare_context_batch

    audit = json.loads((path / "dataset_audit.json").read_text(encoding="utf-8"))
    if audit["input_dimension"] != CONTEXT_DIM:
        raise ValueError(f"Stage A/B input dimension is {audit['input_dimension']}, expected {CONTEXT_DIM}")
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 0.2, (8, CONTEXT_DIM)).astype(np.float32)
    return prepare_context_batch(raw), {
        "source": "synthetic train-shape smoke fixture",
        "input_dimension": audit["input_dimension"],
        "test_results_used": False,
        "feature_ordering": "src.multimodal.context_schema.TOKEN_ORDER",
    }


def build_input_shape_audit(features):
    """Record the repaired path and the exact legacy failure without hiding it."""
    import torch
    from src.multimodal.context_schema import CONTEXT_DIM, TOKEN_DIMS, TOKEN_ORDER, prepare_context_batch

    single = features[0]
    batch_one = prepare_context_batch(single)
    legacy_input = features[:1]
    legacy_index = torch.tensor([0])
    legacy_result = legacy_input[legacy_index]
    return {
        "label": LABEL,
        "context_dim": CONTEXT_DIM,
        "context_dim_is_108": CONTEXT_DIM == 108,
        "token_order": list(TOKEN_ORDER),
        "token_dimensions": TOKEN_DIMS,
        "features_type": f"{type(features).__module__}.{type(features).__name__}",
        "features_shape": list(features.shape),
        "features_dtype": str(features.dtype),
        "single_sample_shape": list(single.shape),
        "prepared_batch_1_shape": list(batch_one.shape),
        "legacy_permutation_input_shape": list(legacy_input.shape),
        "legacy_permutation_result_shape": list(legacy_result.shape),
        "root_cause": "a torch.Tensor used to index a NumPy batch selected one row and removed the batch axis",
        "repair": "candidate permutations now reorder the NumPy batch axis with an explicit integer NumPy array",
    }


def gradient_audit(model):
    import torch

    groups = model.trainable_parameter_groups()
    group_audit = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        group_audit[name] = {
            "parameters": sum(parameter.numel() for parameter in parameters),
            "requiring_grad": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "with_gradient": len(gradients),
            "finite_gradient": bool(gradients) and all(bool(gradient.isfinite().all()) for gradient in gradients),
            "nonzero_gradient": bool(gradients) and any(bool(torch.count_nonzero(gradient)) for gradient in gradients),
        }
    allowed = {id(parameter) for parameters in groups.values() for parameter in parameters}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    return {
        "backbone_gradients": sum(parameter.grad is not None for parameter in model.backbone.parameters()),
        "backbone_parameters_requiring_grad": sum(
            parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
        ),
        "only_projection_benefit_harm_uncertainty_trainable": trainable == allowed,
        "groups": group_audit,
    }


def actual_4bit_audit(model):
    linear4bit_count = sum(type(module).__name__ == "Linear4bit" for module in model.backbone.modules())
    return {
        "backbone_is_loaded_in_4bit": bool(getattr(model.backbone, "is_loaded_in_4bit", False)),
        "linear4bit_module_count": linear4bit_count,
        "verified": bool(getattr(model.backbone, "is_loaded_in_4bit", False) and linear4bit_count > 0),
    }


def failure_result(args, stage, error, started, input_audit):
    return {
        "label": LABEL,
        "stage": "C-0",
        "success": False,
        "stage_c0_passed": False,
        "stopped": True,
        "model_id": args.model_id,
        "requested_load_in_4bit": True,
        "local_files_only": True,
        "failure_stage": stage,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "input_shape_audit": input_audit,
        "model_load_or_runtime_time_s": time.perf_counter() - started,
        "formal_training_started": False,
        "formal_training_allowed": False,
        "no_second_model": True,
        "no_lora": True,
        "CPU_offload_used": False,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if PROJECT_ROOT in args.cache_dir.resolve().parents or args.cache_dir.resolve() == PROJECT_ROOT:
        raise ValueError("model cache must remain outside Git repository")

    import accelerate
    import bitsandbytes
    import torch
    import transformers
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    from src.multimodal.context_schema import CONTEXT_DIM, prepare_context_batch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage C-0 requires CUDA; CUDA is not available")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    environment = {
        "label": LABEL,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "imports_ok": True,
        "peft_installed": importlib.util.find_spec("peft") is not None,
        "cache_dir": str(args.cache_dir),
        "cache_outside_repository": True,
        "local_files_only": True,
    }
    write_json(args.output_dir / "environment_audit.json", environment)

    features, smoke_data = load_train_features(args.phase5a_dir, args.seed)
    input_audit = build_input_shape_audit(features)
    write_json(args.output_dir / "input_shape_audit.json", input_audit)
    print(
        f"type(features)={type(features)} shape={features.shape} dtype={features.dtype}\n"
        f"single sample shape={features[0].shape} batch=1 shape={prepare_context_batch(features[0]).shape}\n"
        f"CONTEXT_DIM={CONTEXT_DIM}",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    started = time.perf_counter()
    try:
        model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(
            args.model_id,
            device_map={"": 0},
            cache_dir=str(args.cache_dir),
            local_files_only=True,
        )
    except Exception as error:
        failure = failure_result(args, "cached_model_load", error, started, input_audit)
        write_json(args.output_dir / "smoke_test.json", failure)
        write_json(args.output_dir / "summary.json", failure)
        raise

    try:
        load_time = time.perf_counter() - started
        model = model.to(device)
        model.train()
        allocated = torch.cuda.memory_allocated() / 2**30
        peak_load = torch.cuda.max_memory_allocated() / 2**30
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        total_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        quantization = actual_4bit_audit(model)

        batch_results = []
        largest = 0
        oom = False
        for batch_size in (1, 2, 4, 8):
            if batch_size > args.max_batch_size:
                break
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                batch_np = prepare_context_batch(features[:batch_size])
                batch = torch.from_numpy(batch_np).to(device)
                with torch.no_grad():
                    projected_shape = list(model.projection(batch).shape)
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = model(batch)
                loss = (
                    prediction.benefit_mean.square().mean()
                    + prediction.harm_logit.square().mean()
                    + prediction.benefit_log_variance.square().mean()
                )
                torch.cuda.synchronize()
                forward = time.perf_counter() - start
                start = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                backward = time.perf_counter() - start
                audit = gradient_audit(model)
                groups_ok = all(
                    row["finite_gradient"] and row["nonzero_gradient"]
                    for row in audit["groups"].values()
                )
                success = bool(
                    torch.isfinite(loss)
                    and projected_shape == [batch_size, 9, model.projection.hidden_size]
                    and audit["backbone_gradients"] == 0
                    and audit["backbone_parameters_requiring_grad"] == 0
                    and audit["only_projection_benefit_harm_uncertainty_trainable"]
                    and groups_ok
                )
                batch_results.append({
                    "batch_size": batch_size,
                    "success": success,
                    "input_shape": list(batch.shape),
                    "structured_token_shape": projected_shape,
                    "loss": float(loss.detach()),
                    "forward_latency_s": forward,
                    "backward_latency_s": backward,
                    "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "gradient_audit": audit,
                })
                if not success:
                    break
                largest = batch_size
                model.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                oom = True
                batch_results.append({"batch_size": batch_size, "success": False, "OOM": True})
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                oom = True
                batch_results.append({"batch_size": batch_size, "success": False, "OOM": True})
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break

        permutation_count = min(max(largest, 1), 4)
        permutation = np.arange(permutation_count, dtype=np.int64)[::-1].copy()
        original_np = prepare_context_batch(features[:permutation_count])
        permuted_np = prepare_context_batch(features[permutation])
        model.eval()
        with torch.inference_mode():
            original = model(torch.from_numpy(original_np).to(device))
            permuted = model(torch.from_numpy(permuted_np).to(device))
        permutation_consistent = all(
            torch.allclose(getattr(original, field)[permutation], getattr(permuted, field), atol=1e-5, rtol=1e-5)
            for field in ("context_embedding", "benefit_mean", "benefit_log_variance", "harm_logit")
        )
        first = batch_results[0]
        single_batch_ok = bool(first.get("success", False))
        no_cpu_offload = all(
            device_name not in ("cpu", "disk")
            for device_name in getattr(model.backbone, "hf_device_map", {"": 0}).values()
        )
        stage_c0_passed = bool(
            quantization["verified"]
            and model.backbone_fully_frozen
            and single_batch_ok
            and permutation_consistent
            and largest >= 1
            and no_cpu_offload
        )
        result = {
            "label": LABEL,
            "stage": "C-0",
            "success": stage_c0_passed,
            "stage_c0_passed": stage_c0_passed,
            "model_id": args.model_id,
            "local_files_only": True,
            "load_in_4bit": quantization["verified"],
            "quantization_config": {"type": "NF4", "double_quant": True, "compute_dtype": "bfloat16"},
            "actual_quantization_audit": quantization,
            "model_parameter_count": total_parameters,
            "trainable_adapter_head_parameters": trainable,
            "backbone_fully_frozen": model.backbone_fully_frozen,
            "backbone_parameters_requiring_grad": sum(
                parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
            ),
            "structured_groups": 9,
            "qwen_hidden_size": model.projection.hidden_size,
            "structured_projection_to_hidden_size": model.projection.hidden_size,
            "input_shape_audit": input_audit,
            "smoke_data": smoke_data,
            "load_time_s": load_time,
            "gpu_allocated_after_load_gb": allocated,
            "gpu_peak_during_load_gb": peak_load,
            "cpu_peak_rss_before_mb": rss_before,
            "cpu_peak_rss_after_load_mb": rss_after,
            "batch_results": batch_results,
            "largest_safe_batch_size_tested": largest,
            "max_batch_size_requested": args.max_batch_size,
            "OOM_occurred": oom,
            "CPU_offload_used": not no_cpu_offload,
            "CPU_offload_needed": False if single_batch_ok and no_cpu_offload else None,
            "candidate_permutation": permutation.tolist(),
            "candidate_permutation_input_shape": list(permuted_np.shape),
            "candidate_permutation_consistent": permutation_consistent,
            "single_batch_forward_backward": single_batch_ok,
            "formal_training_started": False,
            "formal_training_allowed": stage_c0_passed,
            "lora_used": False,
            "second_model_downloaded": False,
        }
        write_json(args.output_dir / "smoke_test.json", result)
        write_json(args.output_dir / "summary.json", result)
        print(json.dumps(clean({
            "success": result["success"],
            "peak_gb": max(row.get("peak_cuda_gb", 0) for row in batch_results),
            "largest_batch": largest,
            "forward_backward": result["single_batch_forward_backward"],
            "candidate_permutation_consistent": permutation_consistent,
        }), indent=2), flush=True)
    except Exception as error:
        failure = failure_result(args, "runtime_smoke", error, started, input_audit)
        write_json(args.output_dir / "smoke_test.json", failure)
        write_json(args.output_dir / "summary.json", failure)
        raise


if __name__ == "__main__":
    main()
