"""Run Phase 2B normalized, grouped, multiseed, horizon and GPU diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = (42, 123, 3407, 2026, 7777)
HORIZONS = (5, 10, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-by-speed", action="store_true")
    parser.add_argument("--benchmark-batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    return parser.parse_args()


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_constant_velocity(history: np.ndarray, future_length: int) -> np.ndarray:
    velocity = history[:, -1] - history[:, -2]
    steps = np.arange(1, future_length + 1, dtype=np.float32)[None, :, None]
    return history[:, -1:, :] + steps * velocity[:, None, :]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_models(future_length: int, scale_by_speed: bool) -> dict[str, Any]:
    from src.models.lstm_trajectory import LSTMTrajectoryPredictor
    from src.models.normalized_trajectory import NormalizedTrajectoryPredictor
    from src.models.residual_transformer import ResidualTransformer
    from src.models.transformer_trajectory import TransformerTrajectoryPredictor

    return {
        "M1_LSTM_normalized": NormalizedTrajectoryPredictor(
            LSTMTrajectoryPredictor(future_length=future_length), scale_by_speed
        ),
        "M2_Transformer_normalized": NormalizedTrajectoryPredictor(
            TransformerTrajectoryPredictor(future_length=future_length), scale_by_speed
        ),
        "M3_ResidualTransformer": ResidualTransformer(
            future_length=future_length, scale_by_speed=scale_by_speed
        ),
    }


def run_learned_models(
    torch: Any,
    splits: Any,
    seed: int,
    future_length: int,
    args: argparse.Namespace,
    device: Any,
    keep_predictions: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, float]]], dict[str, np.ndarray], dict[str, Any]]:
    from torch.utils.data import DataLoader

    from src.data.synthetic_trajectory_diagnostics import as_tensor_dataset
    from src.evaluation.trajectory_diagnostics import benchmark_inference, predict_batches
    from src.evaluation.trajectory_metrics import ade_fde, parameter_count
    from src.training.train_trajectory import train_model

    results: dict[str, dict[str, float]] = {}
    curves = {}
    predictions = {}
    benchmarks = {}
    model_names = tuple(build_models(future_length, args.scale_by_speed))
    for name in model_names:
        # Every reported experiment seed directly controls initialization and shuffling.
        set_seed(torch, seed)
        model = build_models(future_length, args.scale_by_speed)[name]
        train_loader = DataLoader(
            as_tensor_dataset(splits.train),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(as_tensor_dataset(splits.val), batch_size=args.batch_size)
        test_loader = DataLoader(as_tensor_dataset(splits.test), batch_size=args.batch_size)
        print(f"\nseed={seed} horizon={future_length} training {name}")
        curves[name] = train_model(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.learning_rate,
        )
        prediction = predict_batches(model, test_loader, device)
        target = torch.from_numpy(splits.test.future)
        ade, fde = ade_fde(prediction, target)
        results[name] = {"ADE": ade, "FDE": fde, "parameters": parameter_count(model)}
        if keep_predictions:
            predictions[name] = prediction.numpy()
            benchmark_size = min(args.benchmark_batch_size, len(splits.test.history))
            sample = torch.from_numpy(splits.test.history[:benchmark_size])
            benchmarks[name] = benchmark_inference(
                model, sample, device, args.warmup, args.repetitions
            )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results, curves, predictions, benchmarks


def load_v1_predictions(torch: Any, splits: Any, args: argparse.Namespace, device: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from torch.utils.data import DataLoader

    from src.data.synthetic_trajectory_diagnostics import as_tensor_dataset
    from src.evaluation.trajectory_diagnostics import benchmark_inference, predict_batches
    from src.models.lstm_trajectory import LSTMTrajectoryPredictor
    from src.models.transformer_trajectory import TransformerTrajectoryPredictor

    paths_and_models = {
        "M1_LSTM_v1_absolute": (
            PROJECT_ROOT / "results_dev" / "m1_lstm.pt",
            LSTMTrajectoryPredictor(),
        ),
        "M2_Transformer_v1_absolute": (
            PROJECT_ROOT / "results_dev" / "m2_transformer.pt",
            TransformerTrajectoryPredictor(),
        ),
    }
    loader = DataLoader(as_tensor_dataset(splits.test), batch_size=args.batch_size)
    predictions, benchmarks = {}, {}
    for name, (path, model) in paths_and_models.items():
        if not path.exists():
            print(f"警告：未找到 v1 权重，跳过 {name} 分类型预测：{path}")
            continue
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.to(device)
        predictions[name] = predict_batches(model, loader, device).numpy()
        benchmark_size = min(args.benchmark_batch_size, len(splits.test.history))
        benchmarks[name] = benchmark_inference(
            model,
            torch.from_numpy(splits.test.history[:benchmark_size]),
            device,
            args.warmup,
            args.repetitions,
        )
    return predictions, benchmarks


def summarize_multiseed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for model in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        for statistic in ("mean", "std"):
            reducer = np.mean if statistic == "mean" else lambda x: np.std(x, ddof=1)
            summary.append(
                {
                    "seed": statistic,
                    "model": model,
                    "ADE": float(reducer([row["ADE"] for row in selected])),
                    "FDE": float(reducer([row["FDE"] for row in selected])),
                }
            )
    return summary


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("epochs 和 batch-size 必须为正整数")
    if args.warmup < 50 or args.repetitions < 200:
        raise SystemExit("warmup 必须 >= 50，repetitions 必须 >= 200")
    try:
        import torch
    except ImportError:
        print("错误：需要项目已有的 PyTorch；不会自动安装依赖。", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("错误：请求 CUDA，但 torch.cuda.is_available() 为 False。", file=sys.stderr)
        return 2

    from src.data.synthetic_trajectory_diagnostics import (
        TYPE_NAMES,
        create_labeled_splits,
        dataset_statistics,
    )
    from src.evaluation.trajectory_diagnostics import (
        ConstantVelocityModule,
        benchmark_inference,
        metrics_by_type,
    )
    from src.evaluation.trajectory_metrics import ade_fde
    from src.evaluation.trajectory_plots import plot_training_curves, plot_trajectory_examples

    device = torch.device(args.device)
    output_dir = PROJECT_ROOT / "results_dev"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    primary_splits = create_labeled_splits(
        args.train_size, args.val_size, args.test_size, args.seed, future_length=10
    )
    statistics = dataset_statistics(primary_splits)
    print("\n数据集统计（生成逻辑未修改）")
    print(json.dumps(statistics, indent=2, ensure_ascii=False))

    primary_results, curves, normalized_predictions, benchmarks = run_learned_models(
        torch, primary_splits, args.seed, 10, args, device, keep_predictions=True
    )
    m0_prediction = numpy_constant_velocity(primary_splits.test.history, 10)
    m0_ade, m0_fde = ade_fde(m0_prediction, primary_splits.test.future)
    primary_results = {
        "M0_ConstantVelocity": {"ADE": m0_ade, "FDE": m0_fde, "parameters": 0},
        **primary_results,
    }
    benchmark_size = min(args.benchmark_batch_size, len(primary_splits.test.history))
    benchmarks["M0_ConstantVelocity"] = benchmark_inference(
        ConstantVelocityModule(10),
        torch.from_numpy(primary_splits.test.history[:benchmark_size]),
        device,
        args.warmup,
        args.repetitions,
    )
    v1_predictions, v1_benchmarks = load_v1_predictions(torch, primary_splits, args, device)
    benchmarks.update(v1_benchmarks)

    all_predictions = {"M0_ConstantVelocity": m0_prediction, **v1_predictions, **normalized_predictions}
    by_type_rows = []
    by_type_json = {}
    for model_name, prediction in all_predictions.items():
        grouped = metrics_by_type(
            prediction, primary_splits.test.future, primary_splits.test.trajectory_type
        )
        by_type_json[model_name] = grouped
        for trajectory_type, values in grouped.items():
            by_type_rows.append({"model": model_name, "trajectory_type": trajectory_type, **values})
    write_csv(
        output_dir / "phase2_by_type.csv",
        by_type_rows,
        ["model", "trajectory_type", "count", "ADE", "FDE"],
    )

    v1_plot_predictions = {"M0": m0_prediction}
    if "M1_LSTM_v1_absolute" in v1_predictions:
        v1_plot_predictions["LSTM"] = v1_predictions["M1_LSTM_v1_absolute"]
    if "M2_Transformer_v1_absolute" in v1_predictions:
        v1_plot_predictions["Transformer"] = v1_predictions["M2_Transformer_v1_absolute"]
    figure_paths = plot_trajectory_examples(
        figures_dir,
        primary_splits.test.history,
        primary_splits.test.future,
        primary_splits.test.trajectory_type,
        v1_plot_predictions,
        filename_prefix="v1_trajectories",
    )
    figure_paths += plot_trajectory_examples(
        figures_dir,
        primary_splits.test.history,
        primary_splits.test.future,
        primary_splits.test.trajectory_type,
        {
            "M0": m0_prediction,
            "LSTM": normalized_predictions["M1_LSTM_normalized"],
            "Transformer": normalized_predictions["M2_Transformer_normalized"],
            "M3": normalized_predictions["M3_ResidualTransformer"],
        },
        filename_prefix="normalized_trajectories",
    )
    figure_paths.append(plot_training_curves(figures_dir, curves))

    multiseed_rows = []
    horizon_results: dict[str, Any] = {}
    cached_primary = primary_results
    for seed in SEEDS:
        if seed == args.seed:
            seed_results = cached_primary
        else:
            splits = create_labeled_splits(
                args.train_size, args.val_size, args.test_size, seed, future_length=10
            )
            learned, _, _, _ = run_learned_models(torch, splits, seed, 10, args, device)
            m0 = numpy_constant_velocity(splits.test.history, 10)
            ade, fde = ade_fde(m0, splits.test.future)
            seed_results = {"M0_ConstantVelocity": {"ADE": ade, "FDE": fde}, **learned}
        for model_name, values in seed_results.items():
            if model_name in (
                "M0_ConstantVelocity",
                "M1_LSTM_normalized",
                "M2_Transformer_normalized",
                "M3_ResidualTransformer",
            ):
                multiseed_rows.append(
                    {"seed": seed, "model": model_name, "ADE": values["ADE"], "FDE": values["FDE"]}
                )
    multiseed_summary = summarize_multiseed(multiseed_rows)
    write_csv(
        output_dir / "phase2_multiseed.csv",
        multiseed_rows + multiseed_summary,
        ["seed", "model", "ADE", "FDE"],
    )

    for horizon in HORIZONS:
        if horizon == 10:
            horizon_results[str(horizon)] = cached_primary
            continue
        splits = create_labeled_splits(
            args.train_size,
            args.val_size,
            args.test_size,
            args.seed,
            future_length=horizon,
        )
        learned, _, _, _ = run_learned_models(
            torch, splits, args.seed, horizon, args, device
        )
        m0 = numpy_constant_velocity(splits.test.history, horizon)
        ade, fde = ade_fde(m0, splits.test.future)
        horizon_results[str(horizon)] = {
            "M0_ConstantVelocity": {"ADE": ade, "FDE": fde, "parameters": 0},
            **learned,
        }

    v1_path = output_dir / "phase2_baseline.json"
    v1_baseline = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else None
    diagnostics = {
        "config": vars(args),
        "fixed_seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "v1_baseline_unchanged": v1_baseline,
        "v1_training_target_audit": {
            "uses_absolute_coordinates": True,
            "evidence": "v1 model(history) is compared directly with absolute future using MSELoss; no origin subtraction or scale transform exists in run_phase2_baseline.py/train_trajectory.py.",
        },
        "dataset_statistics": statistics,
        "primary_comparison": primary_results,
        "by_type": by_type_json,
        "multiseed_summary": multiseed_summary,
        "horizon_results": horizon_results,
        "training_curves": curves,
        "gpu_inference_benchmark": benchmarks,
        "figures": figure_paths,
    }
    diagnostics_path = output_dir / "phase2_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n诊断完成：{diagnostics_path}")
    print(f"分类型结果：{output_dir / 'phase2_by_type.csv'}")
    print(f"多随机种子结果：{output_dir / 'phase2_multiseed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
