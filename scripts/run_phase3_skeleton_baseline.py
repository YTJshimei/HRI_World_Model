"""Run the single-seed Phase 3A synthetic COCO-17 skeleton baseline."""

from __future__ import annotations

import argparse
import gc
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-frames", type=int, default=20)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--occlusion-rate", type=float, default=0.10)
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--validation-size", type=int, default=270)
    parser.add_argument("--test-size", type=int, default=270)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-repetitions", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase3a"
    )
    return parser.parse_args()


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, args: argparse.Namespace) -> Any:
    from src.models.skeleton_baselines import JointConstantVelocity
    from src.models.skeleton_lstm import SkeletonLSTM
    from src.models.skeleton_transformer import (
        ResidualSkeletonTransformer,
        SpatialTemporalSkeletonTransformer,
    )

    if name == "S0_JointConstantVelocity":
        return JointConstantVelocity(args.future_frames)
    if name == "S1_SkeletonLSTM":
        return SkeletonLSTM(future_frames=args.future_frames)
    if name == "S2_SpatialTemporalSkeletonTransformer":
        return SpatialTemporalSkeletonTransformer(
            history_frames=args.history_frames, future_frames=args.future_frames
        )
    return ResidualSkeletonTransformer(
        history_frames=args.history_frames, future_frames=args.future_frames
    )


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def predict_model(model: Any, loader: Any, device: Any, torch: Any) -> np.ndarray:
    predictions = []
    model.to(device).eval()
    with torch.inference_mode():
        for history, _, confidence, visibility, _ in loader:
            predictions.append(
                model(
                    history.to(device), confidence.to(device), visibility.to(device)
                ).cpu()
            )
    return torch.cat(predictions).numpy()


def benchmark_model(
    model: Any,
    sample: tuple[Any, Any, Any],
    device: Any,
    torch: Any,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int | None]:
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
            timings.append((time.perf_counter() - started) * 1000)
    values = np.asarray(timings)
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**2
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
        "peak_cuda_memory_mib": peak,
    }


def main() -> int:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.history_frames < 2
        or args.future_frames < 2
        or args.sample_rate <= 0
        or args.benchmark_warmup < 1
        or args.benchmark_repetitions < 1
    ):
        raise SystemExit("epochs/batch/sample-rate/benchmark 必须为正，history/future 至少为 2")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("错误：需要项目已有的 PyTorch；不会自动安装依赖。", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("错误：请求 CUDA，但 torch.cuda.is_available() 为 False。", file=sys.stderr)
        return 2

    from src.data.synthetic_skeleton import as_tensor_dataset, create_skeleton_splits
    from src.evaluation.skeleton_metrics import metrics_by_action, skeleton_metrics
    from src.training.train_skeleton import train_skeleton_model

    set_seed(torch, args.seed)
    device = torch.device(args.device)
    splits = create_skeleton_splits(
        args.train_size,
        args.validation_size,
        args.test_size,
        args.seed,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        sample_rate_hz=args.sample_rate,
        noise_std=args.noise_std,
        occlusion_rate=args.occlusion_rate,
    )
    train_dataset = as_tensor_dataset(splits.train)
    validation_dataset = as_tensor_dataset(splits.val)
    trained_models = {}
    training_results = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = (
        "S1_SkeletonLSTM",
        "S2_SpatialTemporalSkeletonTransformer",
        "S3_ResidualSkeletonTransformer",
    )
    for name in model_names:
        set_seed(torch, args.seed)
        model = build_model(name, args)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=args.batch_size
        )
        print(f"\n训练 {name}")
        result = train_skeleton_model(
            model,
            train_loader,
            validation_loader,
            device,
            args.epochs,
            args.output_dir / f"{name.lower()}_best.pt",
            args.learning_rate,
        )
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        trained_models[name] = model
        training_results[name] = {
            "best_epoch": result.best_epoch,
            "best_validation_MPJPE": result.best_validation_mpjpe,
            "checkpoint": result.checkpoint_path,
            "training_time_seconds": result.training_time_seconds,
            "history": list(result.history),
        }

    # Test data is materialized only after all validation-selected models are frozen.
    test_dataset = as_tensor_dataset(splits.test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    benchmark_batch = next(iter(test_loader))
    benchmark_sample = (
        benchmark_batch[0],
        benchmark_batch[2],
        benchmark_batch[3],
    )
    models = {"S0_JointConstantVelocity": build_model("S0_JointConstantVelocity", args)}
    models.update(trained_models)
    results = {}
    for name, model in models.items():
        prediction = predict_model(model, test_loader, device, torch)
        metrics = skeleton_metrics(
            prediction,
            splits.test.future_global,
            splits.test.visibility_mask,
            args.sample_rate,
        )
        benchmark = benchmark_model(
            model,
            benchmark_sample,
            device,
            torch,
            args.benchmark_warmup,
            args.benchmark_repetitions,
        )
        results[name] = {
            "metrics": metrics,
            "by_action": metrics_by_action(
                prediction,
                splits.test.future_global,
                splits.test.visibility_mask,
                splits.test.action_type,
                args.sample_rate,
            ),
            "parameters": parameter_count(model),
            "training_time_seconds": training_results.get(name, {}).get(
                "training_time_seconds", 0.0
            ),
            "inference": benchmark,
        }
        model.to("cpu")
        del prediction
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined_history = np.concatenate(
        [splits.train.history_global, splits.val.history_global, splits.test.history_global]
    )
    combined_future = np.concatenate(
        [splits.train.future_global, splits.val.future_global, splits.test.future_global]
    )
    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "test_materialization_count": 1,
        "tensor_contract": {
            "history_global": ["B", args.history_frames, 17, 3],
            "future_global": ["B", args.future_frames, 17, 3],
            "root_global": ["B", args.history_frames, 3],
            "joint_local": ["B", args.history_frames, 17, 3],
            "confidence": ["B", args.history_frames, 17],
            "visibility_mask": ["B", args.history_frames, 17],
        },
        "synthetic_data_range": {
            "history_coordinate_min_m": combined_history.min(axis=(0, 1, 2)).tolist(),
            "history_coordinate_max_m": combined_history.max(axis=(0, 1, 2)).tolist(),
            "future_coordinate_min_m": combined_future.min(axis=(0, 1, 2)).tolist(),
            "future_coordinate_max_m": combined_future.max(axis=(0, 1, 2)).tolist(),
            "body_scale_range": [0.90, 1.10],
            "root_initial_xy_range_m": [-3.0, 3.0],
            "root_height_range_m": [0.88, 1.02],
            "speed_range_mps": [0.0, 2.0],
            "cadence_range_hz": [0.0, 2.8],
            "turn_yaw_rate_rad_s": [-0.5, 0.5],
            "noise_std_m": args.noise_std,
            "occlusion_rate": args.occlusion_rate,
        },
        "training": training_results,
        "test_results": results,
    }
    result_path = args.output_dir / "phase3_skeleton_baseline.json"
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nPhase 3A 结果：{result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
