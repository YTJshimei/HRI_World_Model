"""Run Phase 3A.5 metrics, vectorization benchmarks, plots, and diagnostic S4."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase3a" / "phase3_skeleton_baseline.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase3a5"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-repetitions", type=int, default=200)
    parser.add_argument("--training-profile-warmup", type=int, default=5)
    parser.add_argument("--training-profile-steps", type=int, default=20)
    return parser.parse_args()


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_state(torch: Any, model: Any, path: str | Path) -> Any:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class ResidualReferencePrior:
    """Factory namespace filled after torch import to avoid import-time dependency errors."""


def make_residual_reference_wrapper(torch: Any, model: Any) -> Any:
    from src.data.skeleton_schema import NUM_JOINTS
    from src.models.skeleton_baselines import joint_constant_velocity_prediction_reference

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, history: Any, confidence: Any, visibility: Any) -> Any:
            encoded, _ = self.wrapped.encoder(history, confidence, visibility)
            residual = self.wrapped.head(encoded).view(
                history.shape[0], NUM_JOINTS, self.wrapped.future_frames, 3
            ).permute(0, 2, 1, 3)
            return joint_constant_velocity_prediction_reference(
                history, visibility.bool(), self.wrapped.future_frames
            ) + residual

    return Wrapper(model)


def predict(model: Any, loader: Any, device: Any, torch: Any) -> np.ndarray:
    rows = []
    model.to(device).eval()
    with torch.inference_mode():
        for history, _, confidence, visibility, _ in loader:
            rows.append(
                model(history.to(device), confidence.to(device), visibility.to(device)).cpu()
            )
    return torch.cat(rows).numpy()


def benchmark_inference(
    model: Any,
    sample: tuple[Any, Any, Any],
    device: Any,
    torch: Any,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int | None]:
    if warmup < 50 or repetitions < 200:
        raise ValueError("strict benchmark requires warmup >= 50 and repetitions >= 200")
    history, confidence, visibility = (value.to(device) for value in sample)
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(history, confidence, visibility)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(history, confidence, visibility)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(timings)
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "batch_size": int(history.shape[0]),
        "warmup": warmup,
        "repetitions": repetitions,
        "mean_ms_per_batch": float(values.mean()),
        "median_ms_per_batch": float(np.median(values)),
        "p95_ms_per_batch": float(np.percentile(values, 95)),
        "mean_ms_per_sample": float(values.mean() / history.shape[0]),
        "peak_cuda_memory_mib": peak_memory,
    }


def benchmark_training_throughput(
    model: Any,
    batch: tuple[Any, Any, Any, Any],
    device: Any,
    torch: Any,
    warmup: int,
    steps: int,
    learning_rate: float,
) -> dict[str, float | int]:
    history, future, confidence, visibility = (value.to(device) for value in batch)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(history, confidence, visibility), future)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
    values = np.asarray(timings)
    mean_seconds = float(values.mean())
    return {
        "warmup_steps": warmup,
        "measured_steps": steps,
        "mean_ms_per_step": mean_seconds * 1000.0,
        "samples_per_second": float(history.shape[0] / mean_seconds),
    }


def focused_by_action(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ("Global_MPJPE", "Local_MPJPE", "Root_ADE", "Root_FDE")
    return {action: {key: metrics[key] for key in keys} for action, metrics in rows.items()}


def main() -> int:
    args = parse_args()
    if args.benchmark_warmup < 50 or args.benchmark_repetitions < 200:
        raise SystemExit("benchmark-warmup must be >= 50 and repetitions >= 200")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("Error: the existing PyTorch environment is required; nothing is installed automatically.", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but torch.cuda.is_available() is False.", file=sys.stderr)
        return 2
    if not args.baseline_json.exists():
        print(f"Error: frozen Phase 3A result not found: {args.baseline_json}", file=sys.stderr)
        return 3

    from src.data.synthetic_skeleton import as_tensor_dataset, create_skeleton_splits
    from src.evaluation.skeleton_diagnostic_plots import (
        plot_local_pose_comparison,
        plot_root_trajectories,
    )
    from src.evaluation.skeleton_metrics import metrics_by_action, skeleton_metrics
    from src.models.hybrid_root_pose import HybridRootPoseModel
    from src.models.skeleton_baselines import (
        JointConstantVelocity,
        JointConstantVelocityReference,
    )
    from src.models.skeleton_lstm import SkeletonLSTM
    from src.models.skeleton_transformer import (
        ResidualSkeletonTransformer,
        SpatialTemporalSkeletonTransformer,
    )
    from src.training.train_skeleton import train_skeleton_model

    frozen = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    config = frozen["config"]
    epochs = int(config["epochs"] if args.epochs is None else args.epochs)
    if epochs <= 0:
        raise SystemExit("epochs must be positive")
    seed = int(config["seed"])
    history_frames = int(config["history_frames"])
    future_frames = int(config["future_frames"])
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    sample_rate = float(config["sample_rate"])
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(torch, seed)
    splits = create_skeleton_splits(
        int(config["train_size"]),
        int(config["validation_size"]),
        int(config["test_size"]),
        seed,
        history_frames=history_frames,
        future_frames=future_frames,
        sample_rate_hz=sample_rate,
        noise_std=float(config["noise_std"]),
        occlusion_rate=float(config["occlusion_rate"]),
    )
    train_dataset = as_tensor_dataset(splits.train)
    validation_dataset = as_tensor_dataset(splits.val)

    constructors = {
        "S1_SkeletonLSTM": lambda: SkeletonLSTM(future_frames=future_frames),
        "S2_SpatialTemporalSkeletonTransformer": lambda: SpatialTemporalSkeletonTransformer(
            history_frames=history_frames, future_frames=future_frames
        ),
        "S3_ResidualSkeletonTransformer": lambda: ResidualSkeletonTransformer(
            history_frames=history_frames, future_frames=future_frames
        ),
    }
    models = {}
    for name, constructor in constructors.items():
        models[name] = load_state(
            torch, constructor(), frozen["training"][name]["checkpoint"]
        )

    # S4 alone is newly trained. Frozen S1/S2/S3 checkpoints are only loaded.
    set_seed(torch, seed)
    s4 = HybridRootPoseModel(
        history_frames=history_frames, future_frames=future_frames
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    print("Training diagnostic S4 HybridRootPoseModel")
    s4_training = train_skeleton_model(
        s4,
        train_loader,
        validation_loader,
        device,
        epochs,
        args.output_dir / "s4_hybridrootposemodel_best.pt",
        learning_rate,
    )
    s4.to("cpu")
    models["S4_HybridRootPoseModel"] = s4

    # The test loader is created only after validation has selected S4's checkpoint.
    test_dataset = as_tensor_dataset(splits.test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    first_batch = next(iter(test_loader))
    inference_sample = (first_batch[0], first_batch[2], first_batch[3])
    training_sample = (first_batch[0], first_batch[1], first_batch[2], first_batch[3])

    evaluation_models = {
        "S0_JointConstantVelocity": JointConstantVelocity(future_frames),
        **models,
    }
    predictions = {}
    evaluations = {}
    for name, model in evaluation_models.items():
        prediction = predict(model, test_loader, device, torch)
        predictions[name] = prediction
        by_action = metrics_by_action(
            prediction,
            splits.test.future_global,
            splits.test.visibility_mask,
            splits.test.action_type,
            sample_rate,
        )
        evaluations[name] = {
            "metrics": skeleton_metrics(
                prediction,
                splits.test.future_global,
                splits.test.visibility_mask,
                sample_rate,
            ),
            "by_action": focused_by_action(by_action),
            "parameters": parameter_count(model),
        }
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    figure_predictions = {
        "S1": predictions["S1_SkeletonLSTM"],
        "S2": predictions["S2_SpatialTemporalSkeletonTransformer"],
        "S4": predictions["S4_HybridRootPoseModel"],
    }
    figure_dir = args.output_dir / "figures"
    plot_root_trajectories(
        splits.test.history_global,
        splits.test.future_global,
        figure_predictions,
        splits.test.action_type,
        figure_dir / "root_trajectories.png",
    )
    plot_local_pose_comparison(
        splits.test.future_global,
        figure_predictions,
        splits.test.action_type,
        figure_dir,
    )

    # Controlled old/new equivalence and strict timing on one identical batch.
    s0_reference = JointConstantVelocityReference(future_frames)
    s0_vectorized = JointConstantVelocity(future_frames)
    s3_vectorized = models["S3_ResidualSkeletonTransformer"]
    s3_reference = make_residual_reference_wrapper(torch, s3_vectorized)
    history_cuda, confidence_cuda, visibility_cuda = (
        value.to(device) for value in inference_sample
    )
    with torch.inference_mode():
        s0_error = float(
            (s0_reference.to(device)(history_cuda, confidence_cuda, visibility_cuda)
             - s0_vectorized.to(device)(history_cuda, confidence_cuda, visibility_cuda))
            .abs().max().item()
        )
        s3_error = float(
            (s3_reference.to(device)(history_cuda, confidence_cuda, visibility_cuda)
             - s3_vectorized.to(device)(history_cuda, confidence_cuda, visibility_cuda))
            .abs().max().item()
        )
    equivalence = {"S0_max_abs_error": s0_error, "S3_max_abs_error": s3_error}
    if max(s0_error, s3_error) > 1e-6:
        raise RuntimeError(f"vectorization changed prediction: {equivalence}")

    strict_models = {
        "S0_reference": s0_reference,
        "S0_vectorized": s0_vectorized,
        "S1": models["S1_SkeletonLSTM"],
        "S2": models["S2_SpatialTemporalSkeletonTransformer"],
        "S3_reference_prior": s3_reference,
        "S3_vectorized_prior": s3_vectorized,
    }
    inference_benchmarks = {}
    for name, model in strict_models.items():
        print(f"Strict benchmark {name}")
        inference_benchmarks[name] = benchmark_inference(
            model,
            inference_sample,
            device,
            torch,
            args.benchmark_warmup,
            args.benchmark_repetitions,
        )

    old_training_model = make_residual_reference_wrapper(torch, copy.deepcopy(s3_vectorized).cpu())
    new_training_model = copy.deepcopy(s3_vectorized).cpu()
    set_seed(torch, seed)
    old_throughput = benchmark_training_throughput(
        old_training_model,
        training_sample,
        device,
        torch,
        args.training_profile_warmup,
        args.training_profile_steps,
        learning_rate,
    )
    set_seed(torch, seed)
    new_throughput = benchmark_training_throughput(
        new_training_model,
        training_sample,
        device,
        torch,
        args.training_profile_warmup,
        args.training_profile_steps,
        learning_rate,
    )

    s0_old = inference_benchmarks["S0_reference"]["mean_ms_per_batch"]
    s0_new = inference_benchmarks["S0_vectorized"]["mean_ms_per_batch"]
    s3_old = inference_benchmarks["S3_reference_prior"]["mean_ms_per_batch"]
    s3_new = inference_benchmarks["S3_vectorized_prior"]["mean_ms_per_batch"]
    comparison = {
        "S0_latency_speedup_x": float(s0_old / s0_new),
        "S3_latency_speedup_x": float(s3_old / s3_new),
        "S3_training_throughput_speedup_x": float(
            new_throughput["samples_per_second"] / old_throughput["samples_per_second"]
        ),
        "frozen_phase3a": {
            "S0_mean_ms_per_batch": frozen["test_results"]["S0_JointConstantVelocity"]["inference"]["mean_ms_per_batch"],
            "S3_mean_ms_per_batch": frozen["test_results"]["S3_ResidualSkeletonTransformer"]["inference"]["mean_ms_per_batch"],
            "S3_training_seconds": frozen["training"]["S3_ResidualSkeletonTransformer"]["training_time_seconds"],
        },
    }
    loop_profile = {
        "frozen_reference_batch_size": batch_size,
        "python_sample_loop_iterations_per_forward": batch_size * 2,
        "python_joint_loop_iterations_per_forward": batch_size * (17 + 2),
        "torch_nonzero_calls_for_joint_prior": batch_size * 17,
        "torch_nonzero_calls_for_root_fallback": batch_size * 2,
        "explicit_item_calls": 0,
        "numpy_conversions": 0,
        "explicit_cpu_gpu_copies_inside_prior": 0,
        "implicit_sync_sources": [
            "CUDA torch.nonzero has data-dependent output shape and synchronizes the host",
            "Python bool(tensor.any()) synchronizes when the no-hip fallback is reached",
            "many tiny indexed reads, writes, and arithmetic CUDA kernel launches",
        ],
    }

    payload = {
        "config": {
            "device": args.device,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "history_frames": history_frames,
            "future_frames": future_frames,
            "noise_std": float(config["noise_std"]),
            "occlusion_rate": float(config["occlusion_rate"]),
            "learning_rate": learning_rate,
            "benchmark_warmup": args.benchmark_warmup,
            "benchmark_repetitions": args.benchmark_repetitions,
        },
        "frozen_models_retrained": False,
        "test_materialization_count": 1,
        "equivalence": equivalence,
        "loop_profile": loop_profile,
        "evaluation": evaluations,
        "inference_benchmarks": inference_benchmarks,
        "training_throughput": {
            "S3_reference_prior": old_throughput,
            "S3_vectorized_prior": new_throughput,
        },
        "optimization_comparison": comparison,
        "S4_training": {
            "best_epoch": s4_training.best_epoch,
            "best_validation_MPJPE": s4_training.best_validation_mpjpe,
            "training_time_seconds": s4_training.training_time_seconds,
            "checkpoint": s4_training.checkpoint_path,
            "history": list(s4_training.history),
        },
    }
    result_path = args.output_dir / "phase3_diagnostics.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output_dir / "phase3_by_action.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "action", "Global_MPJPE", "Local_MPJPE", "Root_ADE", "Root_FDE"))
        for model_name, result in evaluations.items():
            for action, metrics in result["by_action"].items():
                writer.writerow((model_name, action, *(metrics[key] for key in (
                    "Global_MPJPE", "Local_MPJPE", "Root_ADE", "Root_FDE"
                ))))
    print(f"Diagnostics written to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
