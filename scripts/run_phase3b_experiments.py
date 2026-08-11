"""Run Phase 3B oracle, multi-seed, horizon, occlusion, and loss diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = (42, 123, 3407, 2026, 7777)
HORIZONS = (5, 10, 20, 30)
METRICS = (
    "Global_MPJPE",
    "Local_MPJPE",
    "Root_ADE",
    "Root_FDE",
    "Bone_Length_Error",
    "Joint_Velocity_Error",
    "Occluded_Joint_MPJPE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--history-frames", type=int, default=20)
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--train-occlusion-rate", type=float, default=0.10)
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--validation-size", type=int, default=270)
    parser.add_argument("--test-size", type=int, default=270)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-repetitions", type=int, default=200)
    parser.add_argument(
        "--phase3a-json",
        type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase3a" / "phase3_skeleton_baseline.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase3b"
    )
    return parser.parse_args()


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(name: str, history_frames: int, future_frames: int) -> Any:
    from src.models.decoupled_root_pose import DecoupledRootPoseModel
    from src.models.skeleton_lstm import SkeletonLSTM
    from src.models.skeleton_transformer import (
        ResidualSkeletonTransformer,
        SpatialTemporalSkeletonTransformer,
    )

    if name == "S1":
        return SkeletonLSTM(future_frames=future_frames)
    if name == "S2":
        return SpatialTemporalSkeletonTransformer(
            history_frames=history_frames, future_frames=future_frames
        )
    if name == "S3":
        return ResidualSkeletonTransformer(
            history_frames=history_frames, future_frames=future_frames
        )
    if name == "S4b":
        return DecoupledRootPoseModel(
            history_frames=history_frames, future_frames=future_frames
        )
    raise ValueError(f"unknown model: {name}")


def key(model: str, seed: int, horizon: int, variant: str = "base") -> str:
    return f"{model}|seed={seed}|horizon={horizon}|loss={variant}"


def safe_name(value: str) -> str:
    return value.lower().replace("+", "_").replace(" ", "_")


def train_or_resume(
    model_name: str,
    seed: int,
    horizon: int,
    variant: str,
    weights: Any,
    splits: Any,
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    device: Any,
) -> dict[str, Any]:
    from src.data.synthetic_skeleton import as_tensor_dataset
    from src.training.train_skeleton import train_skeleton_model

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.lower()}_seed{seed}_h{horizon}_{safe_name(variant)}"
    checkpoint = checkpoint_dir / f"{stem}.pt"
    metadata_path = checkpoint_dir / f"{stem}.json"
    signature = {
        "model": model_name,
        "seed": seed,
        "horizon": horizon,
        "variant": variant,
        "weights": asdict(weights),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "noise_std": args.noise_std,
        "train_occlusion_rate": args.train_occlusion_rate,
    }
    if checkpoint.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            print(f"CACHE {stem}", flush=True)
            return cached

    print(f"TRAIN {stem}", flush=True)
    set_seed(torch, seed)
    model = build_model(model_name, args.history_frames, horizon)
    train_loader = DataLoader(
        as_tensor_dataset(splits.train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        as_tensor_dataset(splits.val), batch_size=args.batch_size
    )
    result = train_skeleton_model(
        model,
        train_loader,
        validation_loader,
        device,
        args.epochs,
        checkpoint,
        args.learning_rate,
        weights,
        args.sample_rate,
        verbose=False,
    )
    metadata = {
        "signature": signature,
        "checkpoint": str(checkpoint),
        "best_epoch": result.best_epoch,
        "best_validation_MPJPE": result.best_validation_mpjpe,
        "training_time_seconds": result.training_time_seconds,
        "parameters": parameter_count(model),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"DONE {stem} val={result.best_validation_mpjpe:.6f} "
        f"seconds={result.training_time_seconds:.1f}",
        flush=True,
    )
    model.to("cpu")
    del model, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def load_checkpoint_model(
    model_name: str,
    seed: int,
    horizon: int,
    registry: dict[str, dict[str, Any]],
    torch: Any,
    variant: str = "base",
) -> Any:
    model = build_model(model_name, 20, horizon)
    checkpoint = torch.load(
        registry[key(model_name, seed, horizon, variant)]["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict(model: Any, loader: Any, device: Any, torch: Any) -> np.ndarray:
    predictions = []
    model.to(device).eval()
    with torch.inference_mode():
        for history, _, confidence, visibility, _ in loader:
            predictions.append(
                model(history.to(device), confidence.to(device), visibility.to(device)).cpu()
            )
    model.to("cpu")
    result = torch.cat(predictions).numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metric_values(
    prediction: np.ndarray,
    split: Any,
    sample_rate: float,
) -> dict[str, float]:
    from src.evaluation.skeleton_metrics import skeleton_metrics

    metrics = skeleton_metrics(
        prediction, split.future_global, split.visibility_mask, sample_rate
    )
    return {name: float(metrics[name]) for name in METRICS}


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...],
    metric_keys: tuple[str, ...] = METRICS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in group_keys)].append(row)
    output = []
    for values, selected in grouped.items():
        aggregate = {name: value for name, value in zip(group_keys, values)}
        aggregate["row_type"] = "aggregate"
        aggregate["n"] = len(selected)
        for metric in metric_keys:
            numbers = np.asarray([row[metric] for row in selected], dtype=float)
            mean = float(np.nanmean(numbers)) if not np.isnan(numbers).all() else float("nan")
            std = (
                float(np.nanstd(numbers, ddof=1))
                if np.count_nonzero(~np.isnan(numbers)) > 1
                else 0.0
            )
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_mean_std"] = f"{mean:.6f} ± {std:.6f}"
        output.append(aggregate)
    return output


def write_csv(path: Path, raw: list[dict], aggregates: list[dict]) -> None:
    rows = [{"row_type": "seed", **row} for row in raw] + aggregates
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    """Replace non-standard NaN/Infinity values with JSON null recursively."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("epochs and batch-size must be positive")
    if args.benchmark_warmup < 50 or args.benchmark_repetitions < 200:
        raise SystemExit("benchmark requires warmup >= 50 and repetitions >= 200")
    if 10 not in args.horizons or 42 not in args.seeds:
        raise SystemExit("the full protocol requires seed 42 and horizon 10")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("Error: existing PyTorch is required; no dependency is installed automatically.", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but unavailable.", file=sys.stderr)
        return 2

    from src.data.skeleton_occlusion import (
        STRUCTURED_GROUPS,
        apply_random_occlusion,
        apply_structured_occlusion,
    )
    from src.data.synthetic_skeleton import as_tensor_dataset, create_skeleton_splits
    from src.evaluation.phase3b_plots import (
        plot_contributions,
        plot_grouped_lines,
        plot_oracle_decomposition,
    )
    from src.evaluation.skeleton_benchmark import benchmark_skeleton_model
    from src.evaluation.skeleton_decomposition import (
        build_oracle_predictions,
        shapley_root_local_contribution,
    )
    from src.training.train_skeleton import SkeletonLossWeights

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_weights = SkeletonLossWeights()
    s4b_weights = SkeletonLossWeights(global_loss=1.0, root_loss=1.0, local_loss=1.0)
    ablations = {
        "Base": SkeletonLossWeights(),
        "Base+root": SkeletonLossWeights(global_loss=1.0, root_loss=1.0),
        "Base+local": SkeletonLossWeights(global_loss=1.0, local_loss=1.0),
        "Base+bone": SkeletonLossWeights(global_loss=1.0, bone_loss=0.1),
        "Base+velocity": SkeletonLossWeights(global_loss=1.0, velocity_loss=0.01),
        "Base+root+local": s4b_weights,
        "Base+root+local+bone+velocity": SkeletonLossWeights(
            global_loss=1.0,
            root_loss=1.0,
            local_loss=1.0,
            bone_loss=0.1,
            velocity_loss=0.01,
        ),
    }

    registry: dict[str, dict[str, Any]] = {}
    frozen = json.loads(args.phase3a_json.read_text(encoding="utf-8"))
    frozen_names = {
        "S1": "S1_SkeletonLSTM",
        "S2": "S2_SpatialTemporalSkeletonTransformer",
        "S3": "S3_ResidualSkeletonTransformer",
    }
    can_reuse_frozen = (
        args.epochs == int(frozen["config"]["epochs"])
        and args.batch_size == int(frozen["config"]["batch_size"])
        and args.history_frames == int(frozen["config"]["history_frames"])
        and args.noise_std == float(frozen["config"]["noise_std"])
        and args.train_occlusion_rate == float(frozen["config"]["occlusion_rate"])
    )

    print("TRAINING PHASE: test split is not read during this phase", flush=True)
    for seed in args.seeds:
        for horizon in args.horizons:
            splits = create_skeleton_splits(
                args.train_size,
                args.validation_size,
                args.test_size,
                seed,
                history_frames=args.history_frames,
                future_frames=horizon,
                sample_rate_hz=args.sample_rate,
                noise_std=args.noise_std,
                occlusion_rate=args.train_occlusion_rate,
            )
            for model_name in ("S1", "S2", "S3"):
                registry_key = key(model_name, seed, horizon)
                if seed == 42 and horizon == 10 and can_reuse_frozen:
                    frozen_name = frozen_names[model_name]
                    model = build_model(model_name, args.history_frames, horizon)
                    registry[registry_key] = {
                        "checkpoint": frozen["training"][frozen_name]["checkpoint"],
                        "best_epoch": frozen["training"][frozen_name]["best_epoch"],
                        "best_validation_MPJPE": frozen["training"][frozen_name]["best_validation_MPJPE"],
                        "training_time_seconds": frozen["training"][frozen_name]["training_time_seconds"],
                        "parameters": parameter_count(model),
                        "source": "frozen_phase3a",
                    }
                    print(f"FROZEN {registry_key}", flush=True)
                else:
                    registry[registry_key] = train_or_resume(
                        model_name,
                        seed,
                        horizon,
                        "base",
                        base_weights,
                        splits,
                        args,
                        torch,
                        DataLoader,
                        device,
                    )
            if horizon == 10:
                registry[key("S4b", seed, horizon, "root_local")] = train_or_resume(
                    "S4b",
                    seed,
                    horizon,
                    "root_local",
                    s4b_weights,
                    splits,
                    args,
                    torch,
                    DataLoader,
                    device,
                )
            del splits

    # All loss variants and their weights are fixed before any test access.
    seed42_splits = create_skeleton_splits(
        args.train_size,
        args.validation_size,
        args.test_size,
        42,
        history_frames=args.history_frames,
        future_frames=10,
        sample_rate_hz=args.sample_rate,
        noise_std=args.noise_std,
        occlusion_rate=args.train_occlusion_rate,
    )
    ablation_metadata = {}
    for variant, weights in ablations.items():
        variant_key = "root_local" if variant == "Base+root+local" else safe_name(variant)
        if variant == "Base+root+local":
            metadata = registry[key("S4b", 42, 10, "root_local")]
        else:
            metadata = train_or_resume(
                "S4b", 42, 10, variant_key, weights, seed42_splits,
                args, torch, DataLoader, device,
            )
            registry[key("S4b", 42, 10, variant_key)] = metadata
        ablation_metadata[variant] = {"variant_key": variant_key, "weights": asdict(weights), **metadata}
    validation_selected_variant = min(
        ablation_metadata,
        key=lambda name: ablation_metadata[name]["best_validation_MPJPE"],
    )
    print(f"VALIDATION SELECTED LOSS: {validation_selected_variant}", flush=True)

    # Test access starts only after every model/loss is fixed by validation.
    print("EVALUATION PHASE: test access starts now", flush=True)
    horizon_rows: list[dict] = []
    oracle_rows: list[dict] = []
    multiseed_rows: list[dict] = []
    contribution_rows: list[dict] = []
    occlusion_rows: list[dict] = []
    structured_rows: list[dict] = []
    seed10_models: dict[int, dict[str, Any]] = {}

    for seed in args.seeds:
        for horizon in args.horizons:
            splits = create_skeleton_splits(
                args.train_size, args.validation_size, args.test_size, seed,
                history_frames=args.history_frames, future_frames=horizon,
                sample_rate_hz=args.sample_rate, noise_std=args.noise_std,
                occlusion_rate=args.train_occlusion_rate,
            )
            test_loader = DataLoader(as_tensor_dataset(splits.test), batch_size=args.batch_size)
            predictions = {}
            for model_name in ("S1", "S2", "S3"):
                model = load_checkpoint_model(model_name, seed, horizon, registry, torch)
                predictions[model_name] = predict(model, test_loader, device, torch)
                metrics = metric_values(predictions[model_name], splits.test, args.sample_rate)
                horizon_rows.append({
                    "seed": seed,
                    "future_frames": horizon,
                    "horizon_seconds": horizon / args.sample_rate,
                    "model": model_name,
                    **metrics,
                })
                del model
            oracles = build_oracle_predictions(predictions, splits.test.future_global)
            o7_metrics = metric_values(oracles["O7_S1root_S2local"], splits.test, args.sample_rate)
            horizon_rows.append({
                "seed": seed,
                "future_frames": horizon,
                "horizon_seconds": horizon / args.sample_rate,
                "model": "O7",
                **o7_metrics,
            })

            if horizon == 10:
                s4b = load_checkpoint_model("S4b", seed, 10, registry, torch, "root_local")
                s4b_prediction = predict(s4b, test_loader, device, torch)
                s4b_metrics = metric_values(s4b_prediction, splits.test, args.sample_rate)
                for model_name in ("S1", "S2", "S3"):
                    row = next(
                        row for row in reversed(horizon_rows)
                        if row["seed"] == seed and row["future_frames"] == 10 and row["model"] == model_name
                    )
                    multiseed_rows.append(dict(row))
                multiseed_rows.extend((
                    {"seed": seed, "future_frames": 10, "horizon_seconds": 1.0, "model": "O7", **o7_metrics},
                    {"seed": seed, "future_frames": 10, "horizon_seconds": 1.0, "model": "S4b", **s4b_metrics},
                ))
                for oracle_name, oracle_prediction in oracles.items():
                    oracle_rows.append({
                        "seed": seed,
                        "model": oracle_name,
                        **metric_values(oracle_prediction, splits.test, args.sample_rate),
                    })
                oracle_lookup = {
                    "S1": ("O4_S1root_GTlocal", "O1_GTroot_S1local"),
                    "S2": ("O5_S2root_GTlocal", "O2_GTroot_S2local"),
                    "S3": ("O6_S3root_GTlocal", "O3_GTroot_S3local"),
                }
                for model_name, (root_oracle, local_oracle) in oracle_lookup.items():
                    full = metric_values(predictions[model_name], splits.test, args.sample_rate)["Global_MPJPE"]
                    root_only = metric_values(oracles[root_oracle], splits.test, args.sample_rate)["Global_MPJPE"]
                    local_only = metric_values(oracles[local_oracle], splits.test, args.sample_rate)["Global_MPJPE"]
                    contribution_rows.append({
                        "seed": seed,
                        "model": model_name,
                        "full_global_mpjpe": full,
                        "root_only_global_mpjpe": root_only,
                        "local_only_global_mpjpe": local_only,
                        **shapley_root_local_contribution(full, root_only, local_only),
                    })
                seed10_models[seed] = {
                    "S1": registry[key("S1", seed, 10)],
                    "S2": registry[key("S2", seed, 10)],
                    "S3": registry[key("S3", seed, 10)],
                    "S4b": registry[key("S4b", seed, 10, "root_local")],
                }
                del s4b
            del test_loader, predictions, oracles, splits
            gc.collect()

    # Robustness uses the five independently trained 1-second models.
    for seed in args.seeds:
        splits = create_skeleton_splits(
            args.train_size, args.validation_size, args.test_size, seed,
            history_frames=args.history_frames, future_frames=10,
            sample_rate_hz=args.sample_rate, noise_std=args.noise_std,
            occlusion_rate=args.train_occlusion_rate,
        )
        models = {
            name: load_checkpoint_model(
                name, seed, 10, registry, torch,
                "root_local" if name == "S4b" else "base",
            )
            for name in ("S1", "S2", "S3", "S4b")
        }

        def evaluate_condition(condition_split: Any) -> dict[str, np.ndarray]:
            loader = DataLoader(as_tensor_dataset(condition_split), batch_size=args.batch_size)
            return {name: predict(model, loader, device, torch) for name, model in models.items()}

        for rate in (0.0, 0.1, 0.2, 0.3, 0.4):
            condition = apply_random_occlusion(splits.test, rate, seed + 91_000)
            condition_predictions = evaluate_condition(condition)
            condition_predictions["O7"] = build_oracle_predictions(
                {name: condition_predictions[name] for name in ("S1", "S2", "S3")},
                condition.future_global,
            )["O7_S1root_S2local"]
            for model_name, prediction in condition_predictions.items():
                occlusion_rows.append({
                    "seed": seed,
                    "occlusion_rate": rate,
                    "occlusion_percent": int(rate * 100),
                    "model": model_name,
                    **metric_values(prediction, condition, args.sample_rate),
                })
        for group in STRUCTURED_GROUPS:
            for duration in (3, 5, 10):
                condition = apply_structured_occlusion(splits.test, group, duration)
                condition_predictions = evaluate_condition(condition)
                condition_predictions["O7"] = build_oracle_predictions(
                    {name: condition_predictions[name] for name in ("S1", "S2", "S3")},
                    condition.future_global,
                )["O7_S1root_S2local"]
                for model_name, prediction in condition_predictions.items():
                    structured_rows.append({
                        "seed": seed,
                        "occlusion_type": group,
                        "consecutive_frames": duration,
                        "model": model_name,
                        **metric_values(prediction, condition, args.sample_rate),
                    })
        del models, splits
        gc.collect()

    # Test all predeclared ablations only after validation has selected one.
    loss_rows = []
    test_loader = DataLoader(as_tensor_dataset(seed42_splits.test), batch_size=args.batch_size)
    for variant, metadata in ablation_metadata.items():
        model = load_checkpoint_model(
            "S4b", 42, 10, registry, torch, metadata["variant_key"]
        )
        prediction = predict(model, test_loader, device, torch)
        loss_rows.append({
            "loss_variant": variant,
            "validation_selected": variant == validation_selected_variant,
            "best_validation_MPJPE": metadata["best_validation_MPJPE"],
            "best_epoch": metadata["best_epoch"],
            "parameters": metadata["parameters"],
            **{f"weight_{name}": value for name, value in metadata["weights"].items()},
            **metric_values(prediction, seed42_splits.test, args.sample_rate),
        })
        del model, prediction

    # Strict synchronized benchmark on the central seed/horizon setting.
    benchmark_batch = next(iter(test_loader))
    benchmark_sample = (benchmark_batch[0], benchmark_batch[2], benchmark_batch[3])
    benchmarks = {}
    for model_name in ("S1", "S2", "S3", "S4b"):
        model = load_checkpoint_model(
            model_name, 42, 10, registry, torch,
            "root_local" if model_name == "S4b" else "base",
        )
        benchmarks[model_name] = {
            "parameters": parameter_count(model),
            **benchmark_skeleton_model(
                model, benchmark_sample, device, torch,
                args.benchmark_warmup, args.benchmark_repetitions,
            ),
        }
        model.to("cpu")
        del model

    horizon_aggregate = aggregate_rows(
        horizon_rows, ("future_frames", "horizon_seconds", "model")
    )
    multiseed_aggregate = aggregate_rows(multiseed_rows, ("model",))
    oracle_aggregate = aggregate_rows(oracle_rows, ("model",))
    occlusion_aggregate = aggregate_rows(
        occlusion_rows, ("occlusion_rate", "occlusion_percent", "model")
    )
    structured_aggregate = aggregate_rows(
        structured_rows, ("occlusion_type", "consecutive_frames", "model")
    )
    contribution_aggregate = aggregate_rows(
        contribution_rows,
        ("model",),
        (
            "full_global_mpjpe", "root_only_global_mpjpe", "local_only_global_mpjpe",
            "root_contribution", "local_contribution", "interaction",
            "root_fraction", "local_fraction",
        ),
    )

    write_csv(args.output_dir / "oracle.csv", oracle_rows, oracle_aggregate)
    write_csv(args.output_dir / "multiseed.csv", multiseed_rows, multiseed_aggregate)
    write_csv(args.output_dir / "horizon.csv", horizon_rows, horizon_aggregate)
    write_csv(args.output_dir / "occlusion.csv", occlusion_rows, occlusion_aggregate)
    write_csv(
        args.output_dir / "structured_occlusion.csv", structured_rows, structured_aggregate
    )
    write_csv(args.output_dir / "loss_ablation.csv", loss_rows, [])

    for metric, filename in (
        ("Global_MPJPE", "global_mpjpe_vs_horizon.png"),
        ("Local_MPJPE", "local_mpjpe_vs_horizon.png"),
        ("Root_FDE", "root_fde_vs_horizon.png"),
        ("Bone_Length_Error", "bone_error_vs_horizon.png"),
    ):
        plot_grouped_lines(
            horizon_aggregate, "horizon_seconds", metric, figures_dir / filename,
            f"{metric} vs prediction horizon", "Prediction horizon (s)",
        )
    for metric, filename in (
        ("Global_MPJPE", "mpjpe_vs_occlusion.png"),
        ("Occluded_Joint_MPJPE", "occluded_joint_mpjpe_vs_occlusion.png"),
    ):
        plot_grouped_lines(
            occlusion_aggregate, "occlusion_percent", metric, figures_dir / filename,
            f"{metric} vs random joint occlusion", "Random joint occlusion (%)",
        )
    original_oracle_context = [
        row for row in multiseed_aggregate if row["model"] in ("S1", "S2", "S3")
    ] + oracle_aggregate
    plot_oracle_decomposition(
        original_oracle_context, figures_dir / "oracle_error_decomposition.png"
    )
    plot_contributions(
        contribution_aggregate, figures_dir / "root_local_contribution.png"
    )

    summary = {
        "config": {
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "horizons": args.horizons,
            "history_frames": args.history_frames,
            "sample_rate_hz": args.sample_rate,
            "noise_std": args.noise_std,
            "train_occlusion_rate": args.train_occlusion_rate,
            "loss_weights_declared_before_test": {
                name: asdict(weights) for name, weights in ablations.items()
            },
        },
        "frozen_s1_s2_s3_structure_changed": False,
        "test_used_for_selection": False,
        "validation_selected_loss_variant": validation_selected_variant,
        "registry": registry,
        "multiseed_aggregate": multiseed_aggregate,
        "horizon_aggregate": horizon_aggregate,
        "oracle_aggregate": oracle_aggregate,
        "root_local_contribution": contribution_aggregate,
        "occlusion_aggregate": occlusion_aggregate,
        "structured_occlusion_aggregate": structured_aggregate,
        "loss_ablation": loss_rows,
        "cuda_benchmark": benchmarks,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Phase 3B complete: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
