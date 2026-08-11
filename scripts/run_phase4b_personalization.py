"""Run Phase 4B synthetic-only few-shot personal-response experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYNTHETIC_LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
K_VALUES = (0, 1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-frames", type=int, default=20)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--occlusion-rate", type=float, default=0.10)
    parser.add_argument("--persons-per-profile", type=int, default=2)
    parser.add_argument("--interactions-per-person", type=int, default=30)
    parser.add_argument("--benchmark-batch-size", type=int, default=32)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-repetitions", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase4b",
    )
    return parser.parse_args()


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8"
    )


CSV_FIELDS = (
    "seed", "split", "person", "profile", "K", "model", "metric", "value",
    "synthetic_interaction", "detail",
)


def row(
    seed: int, split: str, person: str, profile: str | int, k: int,
    model: str, metric: str, value: Any, detail: str = "",
) -> dict[str, Any]:
    if isinstance(value, float) and not math.isfinite(value):
        value = ""
    if value is None:
        value = ""
    return {
        "seed": seed, "split": split, "person": person, "profile": profile,
        "K": k, "model": model, "metric": metric, "value": value,
        "synthetic_interaction": SYNTHETIC_LABEL, "detail": detail,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({name: item.get(name, "") for name in CSV_FIELDS})


def build_corpora(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    from src.data.personal_interaction_memory import generate_personal_interaction_corpus

    if args.stage == "smoke":
        seen_profiles, unseen_profiles = (0, 1), (5,)
        persons, interactions = 1, 15
        auxiliary_interactions = 12
    else:
        seen_profiles, unseen_profiles = (0, 1, 2, 3, 4), (5, 6)
        persons, interactions = args.persons_per_profile, args.interactions_per_person
        auxiliary_interactions = max(20, interactions - 5)
    common = dict(
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        sample_rate_hz=args.sample_rate,
        noise_std=args.noise_std,
        occlusion_rate=args.occlusion_rate,
    )
    seen = generate_personal_interaction_corpus(
        seen_profiles, persons, interactions, 10, args.seed + 100,
        "seen_timeline", mask_unseen_combinations=True, **common,
    )
    if args.stage == "smoke":
        train_orders, val_orders, test_orders = range(10, 12), range(12, 13), range(13, 15)
    else:
        train_orders, val_orders, test_orders = range(10, 20), range(20, 23), range(23, interactions)

    def with_orders(corpus: Any, orders: Any, label: str) -> Any:
        selected = np.flatnonzero(np.isin(corpus.order_indices, list(orders)))
        return replace(corpus, query_indices=selected, split_label=label)

    train = with_orders(seen, train_orders, "train_seen_person")
    validation = with_orders(seen, val_orders, "validation_seen_person")
    splits = {"T1_seen_person_seen_context": with_orders(seen, test_orders, "T1_seen_person_seen_context")}
    definitions = (
        ("T2_unseen_interaction_state", seen_profiles, "unseen", False),
        ("T3_unseen_person_profile", unseen_profiles, "seen", False),
        ("T4_unseen_action_context", seen_profiles, "seen", False),
        ("T5_unseen_person_unseen_context", unseen_profiles, "unseen", False),
    )
    for offset, (label, profiles, state_mode, mask) in enumerate(definitions):
        splits[label] = generate_personal_interaction_corpus(
            profiles,
            1 if args.stage == "smoke" else persons,
            auxiliary_interactions,
            10,
            args.seed + 1000 + offset * 101,
            label,
            state_mode=state_mode,
            mask_unseen_combinations=mask,
            **common,
        )
    return train, validation, splits


def person_index_for(corpus: Any) -> dict[str, int]:
    return {
        person: index
        for index, person in enumerate(sorted(set(corpus.person_instance_ids.tolist())))
    }


def build_model(mode: str, args: argparse.Namespace, people: int) -> Any:
    from src.models.personalized_response_world_model import PersonalizedRootPoseWorldModel
    return PersonalizedRootPoseWorldModel(
        mode,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        number_of_seen_people=people if mode == "P1" else 0,
    )


def train_models(
    args: argparse.Namespace,
    train: Any,
    validation: Any,
    device: Any,
    torch: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    from torch.utils.data import DataLoader
    from src.training.train_personal_response import (
        PersonalInteractionQueryDataset,
        train_personal_response_model,
    )

    indices = person_index_for(train)
    epochs = 1 if args.stage == "smoke" else args.epochs
    metadata: dict[str, dict[str, Any]] = {}
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("P0", "P1", "P2", "P3"):
        signature = {
            "stage": args.stage, "seed": args.seed, "epochs": epochs,
            "batch_size": args.batch_size, "mode": mode,
            "train_queries": len(train.query_indices),
            "validation_queries": len(validation.query_indices),
            "synthetic_only": True, "protocol_version": 3,
            "equal_query_exposure_across_models": True,
        }
        checkpoint = checkpoint_dir / f"{mode.lower()}_best.pt"
        metadata_path = checkpoint_dir / f"{mode.lower()}_training.json"
        if checkpoint.exists() and metadata_path.exists():
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                print(f"CACHE {mode}", flush=True)
                metadata[mode] = cached
                continue
        set_seed(torch, args.seed + int(mode[1:]))
        model = build_model(mode, args, len(indices))
        # All models receive the same number of query/optimizer exposures. Only
        # P2 consumes the varying support; P0/P1/P3 ignore those support tensors.
        train_k = K_VALUES
        validation_k = 5 if mode == "P2" else 0
        train_loader = DataLoader(
            PersonalInteractionQueryDataset(
                train, train_k, "earliest", indices, seed=args.seed
            ),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        validation_loader = DataLoader(
            PersonalInteractionQueryDataset(
                validation, validation_k, "earliest", indices, seed=args.seed
            ),
            batch_size=args.batch_size,
        )
        print(f"TRAIN {mode} - {SYNTHETIC_LABEL}", flush=True)
        result = train_personal_response_model(
            model, train_loader, validation_loader, device, epochs, checkpoint,
            learning_rate=args.learning_rate, verbose=args.stage == "smoke",
        )
        info = {
            "signature": signature,
            "checkpoint": str(checkpoint),
            "best_epoch": result.best_epoch,
            "best_validation_Global_MPJPE": result.best_validation_global_mpjpe,
            "training_time_seconds": result.training_time_seconds,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        write_json(metadata_path, info)
        metadata[mode] = info
        print(
            f"DONE {mode} val={result.best_validation_global_mpjpe:.6f} "
            f"seconds={result.training_time_seconds:.1f}", flush=True,
        )
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return metadata, indices


def load_model(mode: str, args: argparse.Namespace, people: int, metadata: dict[str, Any], torch: Any) -> Any:
    model = build_model(mode, args, people)
    state = torch.load(metadata["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model


def predict(
    model: Any, corpus: Any, k: int, strategy: str, person_indices: dict[str, int],
    batch_size: int, device: Any, torch: Any, seed: int,
) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from src.training.train_personal_response import (
        PersonalInteractionQueryDataset, model_forward, move_batch,
    )
    dataset = PersonalInteractionQueryDataset(
        corpus, k, strategy, person_indices, seed=seed
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    result: dict[str, list[Any]] = {
        "future": [], "natural": [], "log_std": [], "effect_log_std": [],
        "z": [], "source": [], "profile": [],
    }
    model.to(device).eval()
    with torch.inference_mode():
        for raw in loader:
            batch = move_batch(raw, device)
            output = model_forward(model, batch)
            result["future"].append(output.future_by_action.cpu())
            result["natural"].append(output.natural_future.cpu())
            result["log_std"].append(output.root_log_std_by_action.cpu())
            result["effect_log_std"].append(
                output.action_effect_root_log_std_by_action.cpu()
            )
            result["z"].append(output.z_person.cpu())
            result["source"].append(batch["source_index"].cpu())
            result["profile"].append(batch["profile_id"].cpu())
    model.to("cpu")
    return {name: torch.cat(values).numpy() for name, values in result.items()}


def evaluate_all(
    args: argparse.Namespace,
    splits: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    seen_person_indices: dict[str, int],
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[tuple[str, str, int], dict[str, Any]]]:
    from src.evaluation.personal_response_metrics import personal_response_metrics

    rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (
            "fewshot", "by_person", "profile_calibration", "uncertainty",
            "oracle_gap", "human_response_ranking", "support_strategy", "multiseed",
        )
    }
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    predictions: dict[tuple[str, str, int], dict[str, Any]] = {}
    for mode in ("P0", "P1", "P2", "P3"):
        model = load_model(mode, args, len(seen_person_indices), metadata[mode], torch)
        eligible_splits = ("T1_seen_person_seen_context",) if mode == "P1" else tuple(splits)
        eligible_k = (0,) if mode == "P1" else K_VALUES
        for split_name in eligible_splits:
            corpus = splits[split_name]
            for k in eligible_k:
                output = predict(
                    model, corpus, k, "earliest", seen_person_indices,
                    args.batch_size, device, torch, args.seed,
                )
                metrics, per_sample, curve = personal_response_metrics(
                    output["future"], output["natural"], output["log_std"],
                    output["effect_log_std"],
                    corpus.split, output["source"], args.sample_rate,
                )
                key = (mode, split_name, k)
                results[key] = metrics
                predictions[key] = {**output, "per_sample": per_sample}
                for metric, value in metrics.items():
                    rows["fewshot"].append(row(args.seed, split_name, "ALL", "ALL", k, mode, metric, value))
                    if "Coverage_" in metric or "Interval_" in metric or "NLL" in metric or "Uncertainty_Error_Correlation" in metric:
                        rows["uncertainty"].append(row(args.seed, split_name, "ALL", "ALL", k, mode, metric, value))
                for curve_point in curve:
                    rows["uncertainty"].append(row(
                        args.seed, split_name, "ALL", "ALL", k, mode,
                        "coverage_risk_mean_root_error", curve_point["mean_root_error"],
                        detail=f"retained_fraction={curve_point['retained_fraction']};mean_uncertainty={curve_point['mean_uncertainty']}",
                    ))
                source = output["source"]
                persons = corpus.person_instance_ids[source]
                profiles = output["profile"]
                for person in sorted(set(persons.tolist())):
                    mask = persons == person
                    profile = int(profiles[mask][0])
                    target = corpus.split.future_by_action[source[mask]]
                    global_error = float(np.linalg.norm(output["future"][mask] - target, axis=-1).mean())
                    person_metrics = {
                        "Global_MPJPE": global_error,
                        "Action_Effect_Error": float(per_sample["effect_error"][mask].mean()),
                        "GT_Action_Sensitivity": float(per_sample["gt_sensitivity"][mask].mean()),
                        "Action_Sensitivity": float(per_sample["predicted_sensitivity"][mask].mean()),
                        "Sensitivity_MAE": float(per_sample["sensitivity_absolute_error"][mask].mean()),
                        "Human_Response_Ranking_Accuracy": float(per_sample["human_response_ranking"][mask].mean()),
                    }
                    for metric, value in person_metrics.items():
                        rows["by_person"].append(row(args.seed, split_name, person, profile, k, mode, metric, value))
                for profile in sorted(set(profiles.tolist())):
                    mask = profiles == profile
                    calibration = {
                        "GT_Action_Sensitivity": float(per_sample["gt_sensitivity"][mask].mean()),
                        "Action_Sensitivity": float(per_sample["predicted_sensitivity"][mask].mean()),
                        "Sensitivity_MAE": float(per_sample["sensitivity_absolute_error"][mask].mean()),
                    }
                    for metric, value in calibration.items():
                        rows["profile_calibration"].append(row(args.seed, split_name, "ALL", int(profile), k, mode, metric, value))
                rows["human_response_ranking"].append(row(
                    args.seed, split_name, "ALL", "ALL", k, mode,
                    "Human_Response_Ranking_Accuracy", metrics["Human_Response_Ranking_Accuracy"],
                    detail="uses predicted human response only; excludes robot task/progress reward",
                ))
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Passive support strategy ablation: selection only, no active data acquisition.
    model = load_model("P2", args, len(seen_person_indices), metadata["P2"], torch)
    main = splits["T3_unseen_person_profile"]
    for strategy in ("random", "recent", "diverse_action"):
        output = predict(model, main, 5, strategy, seen_person_indices, args.batch_size, device, torch, args.seed)
        metrics, _, _ = personal_response_metrics(
            output["future"], output["natural"], output["log_std"],
            output["effect_log_std"], main.split,
            output["source"], args.sample_rate,
        )
        for metric in ("Global_MPJPE", "Action_Effect_Error", "Sensitivity_MAE", "Human_Response_Ranking_Accuracy"):
            rows["support_strategy"].append(row(
                args.seed, main.split_label, "ALL", "ALL", 5, "P2", metric,
                metrics[metric], detail=f"passive_strategy={strategy}",
            ))
    model.to("cpu")

    # Oracle gap is defined only when the oracle improves over generic P0.
    from src.evaluation.personal_response_metrics import oracle_gap
    for split_name in splits:
        if split_name == "T1_seen_person_seen_context":
            pass
        for k in K_VALUES:
            for metric in ("Global_MPJPE", "Action_Effect_Error", "Sensitivity_MAE"):
                p0 = results[("P0", split_name, k)][metric]
                p2 = results[("P2", split_name, k)][metric]
                p3 = results[("P3", split_name, k)][metric]
                gap = oracle_gap(float(p0), float(p2), float(p3))
                rows["oracle_gap"].append(row(
                    args.seed, split_name, "ALL", "ALL", k, "P2", f"Oracle_Gap_{metric}", gap,
                    detail="undefined when P3 is not better than P0",
                ))
    main_name = "T3_unseen_person_profile"
    for k in K_VALUES:
        for metric in ("Global_MPJPE", "Action_Effect_Error", "Sensitivity_MAE"):
            rows["multiseed"].append(row(
                args.seed, main_name, "ALL", "ALL", k, "P2", metric,
                results[("P2", main_name, k)][metric],
                detail="single_seed_stage2_only; five-seed experiment not started",
            ))
    return results, rows, predictions


def latent_diagnostics(predictions: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    output = predictions[("P2", "T3_unseen_person_profile", 10)]
    z = output["z"].astype(np.float64)
    profiles = output["profile"].astype(int)
    centered = z - z.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right[:2].T
    nearest = []
    for index, value in enumerate(z):
        distance = np.linalg.norm(z - value, axis=1)
        distance[index] = np.inf
        nearest.append(int(profiles[int(np.argmin(distance))]))
    nearest = np.asarray(nearest)
    within, between = [], []
    for left in range(len(z)):
        for right_index in range(left + 1, len(z)):
            target = within if profiles[left] == profiles[right_index] else between
            target.append(float(np.linalg.norm(z[left] - z[right_index])))
    total_variance = float(np.square(singular).sum())
    explained = (
        (np.square(singular[:2]) / total_variance).tolist() if total_variance > 0 else [0.0, 0.0]
    )
    return {
        "coordinates": coordinates,
        "profiles": profiles,
        "pca_explained_variance_ratio": explained,
        "leave_one_out_nearest_neighbor_profile_accuracy": float(
            np.mean(nearest == profiles)
        ),
        "between_to_within_latent_distance_ratio": (
            float(np.mean(between) / max(np.mean(within), 1e-12))
            if within and between else None
        ),
    }


def benchmark(
    args: argparse.Namespace, corpus: Any, metadata: dict[str, dict[str, Any]],
    indices: dict[str, int], device: Any, torch: Any,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from src.training.train_personal_response import PersonalInteractionQueryDataset, model_forward, move_batch
    warmup = 2 if args.stage == "smoke" else max(50, args.benchmark_warmup)
    repetitions = 5 if args.stage == "smoke" else max(200, args.benchmark_repetitions)
    dataset = PersonalInteractionQueryDataset(corpus, 5, "earliest", indices, seed=args.seed)
    raw = next(iter(DataLoader(dataset, batch_size=min(args.benchmark_batch_size, len(dataset)))))
    batch = move_batch(raw, device)
    report: dict[str, Any] = {
        "device": str(device), "batch_size": int(batch["history"].shape[0]),
        "warmup": warmup, "repetitions": repetitions,
    }
    for mode in ("P0", "P2"):
        model = load_model(mode, args, len(indices), metadata[mode], torch).to(device).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(warmup):
                model_forward(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples = []
            for _ in range(repetitions):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model_forward(model, batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
        report[mode] = {
            "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda" else 0.0
            ),
        }
        if mode == "P2":
            encoder = model.personal_response_encoder
            with torch.inference_mode():
                for _ in range(warmup):
                    encoder(batch["support_features"], batch["support_mask"])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                encoder_samples = []
                for _ in range(repetitions):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    encoder(batch["support_features"], batch["support_mask"])
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    encoder_samples.append((time.perf_counter() - start) * 1000.0)
            report["PersonalResponseEncoder"] = {
                "mean_ms": statistics.fmean(encoder_samples),
                "median_ms": statistics.median(encoder_samples),
                "p95_ms": float(np.percentile(encoder_samples, 95)),
                "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            }
        model.to("cpu")
    return report


def make_figures(
    output_dir: Path,
    results: dict[tuple[str, str, int], dict[str, Any]],
    predictions: dict[tuple[str, str, int], dict[str, Any]],
    latent: dict[str, Any],
    split: Any,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(name: str) -> None:
        path = figures / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        written.append(str(path))

    main = "T3_unseen_person_profile"
    for metric, filename, ylabel in (
        ("Global_MPJPE", "global_mpjpe_vs_k.png", "Global MPJPE (m)"),
        ("Action_Effect_Error", "effect_error_vs_k.png", "Action Effect Error (m)"),
        ("Sensitivity_MAE", "sensitivity_mae_vs_k.png", "Sensitivity MAE (m)"),
        ("Human_Response_Ranking_Accuracy", "human_response_ranking_vs_k.png", "Ranking accuracy"),
        ("Root_NLL", "uncertainty_vs_k.png", "Root NLL"),
    ):
        plt.figure(figsize=(5.5, 3.8))
        for mode in ("P0", "P2", "P3"):
            plt.plot(K_VALUES, [results[(mode, main, k)][metric] for k in K_VALUES], marker="o", label=mode)
        plt.xlabel("Support interactions K")
        plt.ylabel(ylabel)
        plt.title(SYNTHETIC_LABEL)
        plt.legend()
        save(filename)

    output = predictions[("P2", main, 10)]
    per = output["per_sample"]
    plt.figure(figsize=(4.5, 4.2))
    plt.scatter(per["gt_sensitivity"], per["predicted_sensitivity"], c=output["profile"], s=14)
    limit = max(float(per["gt_sensitivity"].max()), float(per["predicted_sensitivity"].max()), 1e-3)
    plt.plot([0, limit], [0, limit], "k--", linewidth=1)
    plt.xlabel("GT action sensitivity")
    plt.ylabel("Predicted action sensitivity")
    plt.title(SYNTHETIC_LABEL)
    save("gt_vs_predicted_sensitivity.png")

    plt.figure(figsize=(5.2, 3.8))
    x = np.arange(3)
    values = [results[(mode, main, 10)]["Action_Effect_Error"] for mode in ("P0", "P2", "P3")]
    plt.bar(x, values)
    plt.xticks(x, ("Generic P0", "Adapted P2", "Oracle P3"))
    plt.ylabel("Action Effect Error (m)")
    plt.title(SYNTHETIC_LABEL)
    save("generic_adapted_oracle.png")

    predicted_root = output["future"][:, :, :, 11:13].mean(axis=3)
    target_root = split.split.future_by_action[output["source"]][:, :, :, 11:13].mean(axis=3)
    sigma = np.exp(output["log_std"])
    uncertainty = np.linalg.norm(sigma, axis=-1).ravel()
    error = np.linalg.norm(predicted_root - target_root, axis=-1).ravel()
    order = np.argsort(uncertainty)
    fractions = np.linspace(0.1, 1.0, 10)
    risk = [error[order[:max(1, int(len(order) * fraction))]].mean() for fraction in fractions]
    plt.figure(figsize=(5.2, 3.8))
    plt.plot(fractions, risk, marker="o")
    plt.xlabel("Retained lowest-uncertainty fraction")
    plt.ylabel("Root error (m)")
    plt.title(SYNTHETIC_LABEL)
    save("coverage_risk.png")

    coordinates, profiles = latent["coordinates"], latent["profiles"]
    plt.figure(figsize=(5.2, 4.2))
    for profile in sorted(set(profiles.tolist())):
        mask = profiles == profile
        plt.scatter(coordinates[mask, 0], coordinates[mask, 1], label=f"profile {profile}", s=18)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(SYNTHETIC_LABEL)
    plt.legend()
    save("latent_pca.png")

    source = int(output["source"][0])
    action_index = 4
    plt.figure(figsize=(5.2, 4.2))
    gt = target_root[0, action_index]
    plt.plot(gt[:, 0], gt[:, 1], "k-o", label="GT")
    for mode in ("P0", "P2", "P3"):
        pred = predictions[(mode, main, 10)]["future"][0, action_index, :, 11:13].mean(axis=1)
        plt.plot(pred[:, 0], pred[:, 1], marker=".", label=mode)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Same action, virtual person profile {int(output['profile'][0])}\n{SYNTHETIC_LABEL}")
    plt.legend()
    save("virtual_person_same_action.png")
    return written


def main() -> None:
    args = parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(SYNTHETIC_LABEL, flush=True)
    set_seed(torch, args.seed)
    train, validation, splits = build_corpora(args)
    metadata, indices = train_models(args, train, validation, device, torch)
    results, csv_rows, predictions = evaluate_all(
        args, splits, metadata, indices, device, torch
    )
    latent = latent_diagnostics(predictions)
    bench = benchmark(args, splits["T3_unseen_person_profile"], metadata, indices, device, torch)
    figure_paths = make_figures(
        args.output_dir, results, predictions, latent,
        splits["T3_unseen_person_profile"],
    )
    for name, values in csv_rows.items():
        write_csv(args.output_dir / f"{name}.csv", values)
    main_split = "T3_unseen_person_profile"
    summary = {
        "label": SYNTHETIC_LABEL,
        "stage": args.stage,
        "seed": args.seed,
        "protocol": {
            "support_K": K_VALUES,
            "support_is_strictly_past_only": True,
            "counterfactual_branches_are_atomic": True,
            "P2_observations_only": True,
            "P3_oracle_only": True,
            "P1_seen_person_only": True,
            "test_not_used_for_model_selection": True,
            "five_seed_run_started": False,
        },
        "training": metadata,
        "T3_unseen_person_profile": {
            mode: {str(k): results[(mode, main_split, k)] for k in K_VALUES}
            for mode in ("P0", "P2", "P3")
        },
        "latent_diagnostics": {
            name: value for name, value in latent.items() if name not in ("coordinates", "profiles")
        },
        "benchmark": bench,
        "figures": figure_paths,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"Phase 4B {args.stage} complete: {args.output_dir}\n"
        f"T3 P0 Global={results[('P0', main_split, 0)]['Global_MPJPE']:.6f} "
        f"P2 K10 Global={results[('P2', main_split, 10)]['Global_MPJPE']:.6f} "
        f"P3 Global={results[('P3', main_split, 0)]['Global_MPJPE']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
