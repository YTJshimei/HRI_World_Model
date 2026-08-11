"""Train an unchanged Phase 2 model on synthetic or standardized ROS trajectories."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--data-source", choices=("synthetic", "ros"), default="synthetic")
    parser.add_argument("--model", choices=("m1", "m2", "m3"), default="m3")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-by-speed", action="store_true")
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--max-interpolation-gap", type=float, default=0.3)
    parser.add_argument("--jump-speed-threshold", type=float, default=3.0)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase2c")
    return parser.parse_args()


def build_model(name: str, scale_by_speed: bool) -> Any:
    from src.models.lstm_trajectory import LSTMTrajectoryPredictor
    from src.models.normalized_trajectory import NormalizedTrajectoryPredictor
    from src.models.residual_transformer import ResidualTransformer
    from src.models.transformer_trajectory import TransformerTrajectoryPredictor

    if name == "m1":
        return NormalizedTrajectoryPredictor(LSTMTrajectoryPredictor(), scale_by_speed)
    if name == "m2":
        return NormalizedTrajectoryPredictor(TransformerTrajectoryPredictor(), scale_by_speed)
    return ResidualTransformer(scale_by_speed=scale_by_speed)


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("错误：需要项目已有的 PyTorch；不会自动安装依赖。", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("错误：请求 CUDA，但 torch.cuda.is_available() 为 False。", file=sys.stderr)
        return 2
    if args.epochs <= 0 or args.batch_size <= 0 or args.target_hz <= 0:
        raise SystemExit("epochs、batch-size、target-hz 必须大于 0")
    set_seed(torch, args.seed)
    device = torch.device(args.device)
    quality_payload = None
    test_load_count = 0

    if args.data_source == "synthetic":
        from src.data.synthetic_trajectory import create_splits

        splits = create_splits(
            args.train_size, args.validation_size, args.test_size, args.seed
        )
        train_arrays, validation_arrays = splits.train, splits.val

        def load_test_once() -> tuple[np.ndarray, np.ndarray]:
            nonlocal test_load_count
            test_load_count += 1
            if test_load_count > 1:
                raise RuntimeError("test set 只能在训练完成后读取一次")
            return splits.test

    else:
        if args.trajectory_csv is None or args.split_manifest is None:
            raise SystemExit("--data-source ros 需要 --trajectory-csv 和 --split-manifest")
        from src.data.data_quality import inspect_data_quality
        from src.data.ros_trajectory_loader import (
            SplitManifest,
            load_trajectory_csv,
            validate_manifest_coverage,
        )
        from src.data.trajectory_window_builder import build_windows

        records = load_trajectory_csv(args.trajectory_csv)
        manifest = SplitManifest.load(args.split_manifest)
        validate_manifest_coverage(manifest, records)
        partitions = manifest.partition_records(records)
        window_options = {
            "target_hz": args.target_hz,
            "max_interpolation_gap_seconds": args.max_interpolation_gap,
        }
        train_windows = build_windows(partitions["train"], **window_options)
        validation_windows = build_windows(partitions["validation"], **window_options)
        train_arrays = (train_windows.history, train_windows.future)
        validation_arrays = (validation_windows.history, validation_windows.future)
        quality_payload = {
            "train": inspect_data_quality(
                partitions["train"], args.target_hz, args.jump_speed_threshold, len(train_windows)
            ).to_dict(),
            "validation": inspect_data_quality(
                partitions["validation"], args.target_hz, args.jump_speed_threshold, len(validation_windows)
            ).to_dict(),
        }
        def load_test_once() -> tuple[np.ndarray, np.ndarray]:
            nonlocal test_load_count, quality_payload
            test_load_count += 1
            if test_load_count > 1:
                raise RuntimeError("test set 只能在训练完成后读取一次")
            test_windows = build_windows(partitions["test"], **window_options)
            quality_payload["test"] = inspect_data_quality(
                partitions["test"], args.target_hz, args.jump_speed_threshold, len(test_windows)
            ).to_dict()
            return test_windows.history, test_windows.future

    if len(train_arrays[0]) == 0 or len(validation_arrays[0]) == 0:
        raise SystemExit("train/validation 没有有效窗口；请检查数据质量和 split manifest")
    train_dataset = TensorDataset(torch.from_numpy(train_arrays[0]), torch.from_numpy(train_arrays[1]))
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_arrays[0]), torch.from_numpy(validation_arrays[1])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    from src.evaluation.real_trajectory_metrics import real_trajectory_metrics
    from src.training.train_trajectory_phase2c import train_with_best_validation_checkpoint

    model = build_model(args.model, args.scale_by_speed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.data_source}_{args.model}_best_validation.pt"
    training = train_with_best_validation_checkpoint(
        model,
        train_loader,
        validation_loader,
        device,
        args.epochs,
        checkpoint,
        args.learning_rate,
    )

    # This is the only point at which final test windows are materialized.
    test_history, test_future = load_test_once()
    if len(test_history) == 0:
        raise SystemExit("test split 没有有效窗口")
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_history), torch.from_numpy(test_future)),
        batch_size=args.batch_size,
    )
    model.eval()
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for history, _ in test_loader:
            predictions.append(model(history.to(device)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = torch.cat(predictions).numpy()
    latency = elapsed * 1000 / len(test_history)
    learned_metrics = real_trajectory_metrics(
        prediction, test_future, 1.0 / args.target_hz, latency
    )

    from src.evaluation.trajectory_diagnostics import ConstantVelocityModule

    m0_model = ConstantVelocityModule(test_future.shape[1]).to(device).eval()
    m0_predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    m0_started = time.perf_counter()
    with torch.inference_mode():
        for history, _ in test_loader:
            m0_predictions.append(m0_model(history.to(device)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    m0_latency = (time.perf_counter() - m0_started) * 1000 / len(test_history)
    m0_prediction = torch.cat(m0_predictions).numpy()
    m0_metrics = real_trajectory_metrics(
        m0_prediction, test_future, 1.0 / args.target_hz, m0_latency
    )
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "split_protocol": "trial/session/person split before resampling and windowing",
        "test_evaluation_count": 1,
        "best_validation": {
            "epoch": training.best_epoch,
            "ADE": training.best_validation_ade,
            "checkpoint": training.checkpoint_path,
        },
        "training_history": list(training.history),
        "test_metrics": {"M0_ConstantVelocity": m0_metrics, args.model: learned_metrics},
        "data_quality": quality_payload,
    }
    result_path = args.output_dir / f"{args.data_source}_{args.model}_phase2c.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Phase 2C 结果：{result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
