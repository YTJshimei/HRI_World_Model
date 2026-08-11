"""Phase 4B.6 synthetic functional human-response identification experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from argparse import Namespace
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--phase4b-dir", type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase4b",
    )
    parser.add_argument(
        "--phase4b5-dir", type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase4b5",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase4b6",
    )
    return parser.parse_args()


def phase4b_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        stage="full", seed=args.seed, epochs=20, batch_size=args.batch_size,
        history_frames=20, future_frames=10, sample_rate=10.0,
        learning_rate=1e-3, noise_std=0.005, occlusion_rate=0.10,
        persons_per_profile=2, interactions_per_person=30,
        benchmark_batch_size=32, benchmark_warmup=50,
        benchmark_repetitions=200, output_dir=args.phase4b_dir,
        device=args.device,
    )


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    fields: list[str] = []
    for item in rows:
        for field in item:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({field: clean(item.get(field, "")) for field in fields})


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    population_mean: np.ndarray, p0_checkpoint: str,
    device: Any, torch: Any,
) -> Any:
    from src.models.functional_response_decoder import FunctionalResponseWorldModel
    model = FunctionalResponseWorldModel()
    state = torch.load(p0_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    model.decoder.natural_backbone.load_state_dict(state)
    model.decoder.freeze_natural_backbone()
    model.estimator.set_generic_prior(torch.from_numpy(population_mean))
    return model


def train_or_load_f2(
    name: str, args: argparse.Namespace, train: Any, validation: Any,
    population_mean: np.ndarray, p0_checkpoint: str,
    device: Any, torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from torch.utils.data import DataLoader
    from src.training.train_functional_response import (
        FunctionalEpisodeDataset, train_functional_model,
    )
    checkpoint = args.output_dir / "checkpoints" / f"{name.lower()}_best.pt"
    metadata_path = checkpoint.with_name(f"{name.lower()}_training.json")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "seed": args.seed, "epochs": args.epochs, "protocol": 1,
        "train_split": train.split_label, "functional_state": True,
    }
    model = build_model(population_mean, p0_checkpoint, device, torch)
    if checkpoint.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
            return model, metadata
    train_loader = DataLoader(
        FunctionalEpisodeDataset(train, K_VALUES, "random", args.seed),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        FunctionalEpisodeDataset(validation, 5, "diverse_action", args.seed),
        batch_size=args.batch_size,
    )
    result = train_functional_model(
        model, train_loader, validation_loader, device, args.epochs, checkpoint,
        args.learning_rate,
    )
    metadata = {
        "signature": signature, "checkpoint": str(checkpoint),
        "best_epoch": result.best_epoch,
        "best_validation_Action_Effect_Error": result.best_validation_effect_error,
        "training_time_seconds": result.training_time_seconds,
        "parameters": result.parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    write_json(metadata_path, metadata)
    return model, metadata


def predict_f2(
    model: Any, corpus: Any, k: int, args: argparse.Namespace,
    device: Any, torch: Any, support_type: str = "earliest",
) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from src.training.train_functional_response import FunctionalEpisodeDataset
    loader = DataLoader(
        FunctionalEpisodeDataset(corpus, k, support_type, args.seed),
        batch_size=args.batch_size,
    )
    values = {name: [] for name in (
        "future", "natural", "theta", "theta_log_std", "theta_target",
        "observed_mask", "source", "profile",
    )}
    model.to(device).eval()
    with torch.inference_mode():
        for raw in loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            output = model(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"], batch["statistics"],
                batch["support_mask"], batch["response_state_mask"],
            )
            values["future"].append(output.future_by_action.cpu())
            values["natural"].append(output.natural_future.cpu())
            values["theta"].append(output.theta_response.cpu())
            values["theta_log_std"].append(output.theta_log_std.cpu())
            values["theta_target"].append(batch["theta_target"].cpu())
            values["observed_mask"].append(batch["response_state_mask"].any(dim=1).cpu())
            values["source"].append(batch["source_index"].cpu())
            values["profile"].append(batch["profile_id"].cpu())
    model.to("cpu")
    return {name: torch.cat(items).numpy() for name, items in values.items()}


def predict_decoder_baseline(
    decoder: Any, corpus: Any, theta_mode: str, population_mean: np.ndarray,
    args: argparse.Namespace, device: Any, torch: Any,
) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from src.training.train_functional_response import FunctionalEpisodeDataset
    loader = DataLoader(
        FunctionalEpisodeDataset(corpus, 0, "earliest", args.seed),
        batch_size=args.batch_size,
    )
    values = {name: [] for name in (
        "future", "natural", "theta", "theta_target", "source", "profile",
    )}
    decoder.to(device).eval()
    with torch.inference_mode():
        for raw in loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            theta = (
                batch["theta_target"] if theta_mode == "oracle"
                else torch.from_numpy(population_mean).to(device)[None].expand(
                    batch["history"].shape[0], -1
                )
            )
            output = decoder(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"], theta,
            )
            values["future"].append(output.future_by_action.cpu())
            values["natural"].append(output.natural_future.cpu())
            values["theta"].append(theta.cpu())
            values["theta_target"].append(batch["theta_target"].cpu())
            values["source"].append(batch["source_index"].cpu())
            values["profile"].append(batch["profile_id"].cpu())
    decoder.to("cpu")
    return {name: torch.cat(items).numpy() for name, items in values.items()}


def future_metrics(output: dict[str, np.ndarray], corpus: Any) -> dict[str, Any]:
    from src.evaluation.personal_response_metrics import personal_response_metrics
    source = output["source"]
    shape = (*output["future"].shape[:3], 3)
    zeros = np.zeros(shape, dtype=np.float32)
    metrics, per_sample, _ = personal_response_metrics(
        output["future"], output["natural"], zeros, zeros,
        corpus.split, source,
    )
    return {"metrics": metrics, "per_sample": per_sample}


def state_metrics(output: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.data.functional_response_state import RESPONSE_STATE_NAMES
    from src.evaluation.personal_response_metrics import safe_correlation
    error = np.abs(output["theta"] - output["theta_target"])
    summary = {"Response_State_MAE": float(error.mean())}
    rows = []
    for dimension, name in enumerate(RESPONSE_STATE_NAMES):
        predicted = output["theta"][:, dimension]
        target = output["theta_target"][:, dimension]
        metrics = {
            "MAE": float(np.abs(predicted - target).mean()),
            "Pearson": safe_correlation(predicted, target),
            "Spearman": safe_correlation(predicted, target, rank=True),
        }
        summary[f"{name}_MAE"] = metrics["MAE"]
        for metric, value in metrics.items():
            rows.append({"dimension": name, "metric": metric, "value": value})
    return summary, rows


def response_uncertainty_rows(
    output: dict[str, np.ndarray], k: int, model_name: str, seed: int,
) -> list[dict[str, Any]]:
    from src.data.functional_response_state import RESPONSE_STATE_NAMES
    from src.evaluation.personal_response_metrics import safe_correlation
    sigma = np.exp(output["theta_log_std"])
    error = np.abs(output["theta"] - output["theta_target"])
    observed = output["observed_mask"]
    rows = []
    for dimension, name in enumerate(RESPONSE_STATE_NAMES):
        for status, mask in (
            ("all", np.ones(len(error), dtype=bool)),
            ("observed", observed[:, dimension]),
            ("unobserved", ~observed[:, dimension]),
        ):
            if not mask.any():
                continue
            values = error[mask, dimension]; uncertainty = sigma[mask, dimension]
            base = {
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": seed,
                "split": "T3_unseen_person_profile", "K": k,
                "model": model_name, "dimension": name,
                "information_status": status,
            }
            metrics = {
                "MAE": float(values.mean()),
                "Mean_Uncertainty": float(uncertainty.mean()),
                "Coverage_50": float(np.mean(values <= 0.67448975 * uncertainty)),
                "Coverage_80": float(np.mean(values <= 1.28155157 * uncertainty)),
                "Coverage_90": float(np.mean(values <= 1.64485363 * uncertainty)),
                "Uncertainty_Error_Correlation": safe_correlation(uncertainty, values),
            }
            for metric, value in metrics.items():
                rows.append({**base, "metric": metric, "value": value})
    return rows


def fixed_query_recovery(
    decoder: Any, corpus: Any, output: dict[str, np.ndarray],
    args: argparse.Namespace, device: Any, torch: Any,
) -> dict[str, float]:
    from src.data.skeleton_schema import compute_root
    from src.evaluation.personalization_diagnostics import person_effect_recovery_ratio
    profiles = sorted(set(output["profile"].tolist()))
    theta_by_profile = {
        profile: output["theta"][output["profile"] == profile].mean(axis=0)
        for profile in profiles
    }
    target_by_profile = {
        profile: output["theta_target"][output["profile"] == profile].mean(axis=0)
        for profile in profiles
    }
    source = int(output["source"][0]); split = corpus.split
    history = torch.from_numpy(split.human_history[source:source + 1]).to(device)
    robot = torch.from_numpy(split.robot_history[source:source + 1]).to(device)
    actions = torch.from_numpy(split.candidate_actions[source:source + 1]).to(device)
    confidence = torch.from_numpy(split.confidence[source:source + 1]).to(device)
    visibility = torch.from_numpy(split.visibility_mask[source:source + 1]).to(device)
    decoder.to(device).eval()
    predicted, expected = [], []
    with torch.inference_mode():
        for profile in profiles:
            pred = decoder(
                history, robot, actions, confidence, visibility,
                torch.from_numpy(theta_by_profile[profile])[None].to(device),
            )
            oracle = decoder(
                history, robot, actions, confidence, visibility,
                torch.from_numpy(target_by_profile[profile])[None].to(device),
            )
            predicted.append(pred.action_effect_by_action.cpu().numpy()[0])
            expected.append(oracle.action_effect_by_action.cpu().numpy()[0])
    predicted, expected = np.stack(predicted), np.stack(expected)
    overall = person_effect_recovery_ratio(predicted, expected)
    speed = person_effect_recovery_ratio(predicted[:, 1:3], expected[:, 1:3])
    distance = person_effect_recovery_ratio(predicted[:, 3:5], expected[:, 3:5])
    predicted_root = compute_root(predicted)[:, 3:5, -1, :2]
    expected_root = compute_root(expected)[:, 3:5, -1, :2]
    lateral_axis = np.asarray((-1.0, 1.0), dtype=np.float32)
    pred_lateral = (predicted_root * lateral_axis).sum(axis=-1)
    gt_lateral = (expected_root * lateral_axis).sum(axis=-1)
    lateral_ratio = float(
        np.mean(np.abs(np.diff(pred_lateral, axis=0)))
        / max(np.mean(np.abs(np.diff(gt_lateral, axis=0))), 1e-12)
    )
    decoder.to("cpu")
    return {
        "Person_Effect_Recovery_Ratio": overall["person_effect_recovery_ratio"],
        "Speed_Effect_Recovery_Ratio": speed["person_effect_recovery_ratio"],
        "Distance_Effect_Recovery_Ratio": distance["person_effect_recovery_ratio"],
        "Lateral_Effect_Recovery_Ratio": lateral_ratio,
        "Delay_Recovery_Error": float(np.mean([
            abs(theta_by_profile[p][3] - target_by_profile[p][3]) for p in profiles
        ])),
    }


def functional_interventions(
    decoder: Any, corpus: Any, theta: np.ndarray,
    args: argparse.Namespace, device: Any, torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    from src.data.skeleton_schema import compute_root
    source = int(corpus.query_indices[0]); split = corpus.split
    tensors = (
        torch.from_numpy(split.human_history[source:source + 1]).to(device),
        torch.from_numpy(split.robot_history[source:source + 1]).to(device),
        torch.from_numpy(split.candidate_actions[source:source + 1]).to(device),
        torch.from_numpy(split.confidence[source:source + 1]).to(device),
        torch.from_numpy(split.visibility_mask[source:source + 1]).to(device),
    )
    history_root = compute_root(split.human_history[source])
    robot_xy = split.robot_history[source, -1, :2]
    away = history_root[-1, :2] - robot_xy
    away = away / max(float(np.linalg.norm(away)), 1e-8)
    lateral_axis = np.asarray((-away[1], away[0]), dtype=np.float32)
    decoder.to(device).eval()
    definitions = (
        ("speed_response_gain", 0, 2, "increasing"),
        ("distance_response_gain", 1, 4, "increasing"),
        ("lateral_response_gain", 2, 4, "increasing"),
        ("response_delay", 3, 2, "decreasing"),
    )
    rows, passed = [], {}
    for name, dimension, action_index, direction in definitions:
        values = []
        for factor in (0.5, 1.0, 1.5):
            altered = theta.copy(); altered[dimension] *= factor
            with torch.inference_mode():
                output = decoder(
                    *tensors[:5], torch.from_numpy(altered)[None].to(device)
                )
            effect = output.action_effect_by_action.cpu().numpy()[0, action_index]
            if name == "lateral_response_gain":
                root_effect = compute_root(effect)[-1, :2]
                # Project onto the decoder's actual avoidance-lateral axis;
                # global x/y components are not themselves lateral measures.
                metric = float(abs(np.dot(root_effect, lateral_axis)))
            else:
                metric = float(np.linalg.norm(effect, axis=-1).mean())
            values.append(metric)
            rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                "dimension": name, "factor": factor,
                "action": int(split.candidate_actions[source, action_index]),
                "response_measure": metric, "expected_direction": direction,
            })
        passed[name] = bool(
            np.all(np.diff(values) > 0) if direction == "increasing"
            else np.all(np.diff(values) < 0)
        )
    decoder.to("cpu")
    return rows, passed


def functional_swap(
    decoder: Any, corpus: Any, args: argparse.Namespace, device: Any, torch: Any,
) -> tuple[list[dict[str, Any]], bool]:
    from src.data.functional_response_state import functional_state_from_profile
    from src.data.synthetic_interaction import PROFILE_BY_ID
    source = int(corpus.query_indices[0]); split = corpus.split
    tensors = (
        torch.from_numpy(split.human_history[source:source + 1]).to(device),
        torch.from_numpy(split.robot_history[source:source + 1]).to(device),
        torch.from_numpy(split.candidate_actions[source:source + 1]).to(device),
        torch.from_numpy(split.confidence[source:source + 1]).to(device),
        torch.from_numpy(split.visibility_mask[source:source + 1]).to(device),
    )
    # Profile 2: high speed/low distance; profile 1: low speed/high distance.
    states = {
        "A_high_speed_low_distance": functional_state_from_profile(PROFILE_BY_ID[2]),
        "B_low_speed_high_distance": functional_state_from_profile(PROFILE_BY_ID[1]),
    }
    rows, measures = [], {}
    decoder.to(device).eval()
    with torch.inference_mode():
        for identity_label in ("history_A", "history_B"):
            for state_name, theta in states.items():
                output = decoder(*tensors, torch.from_numpy(theta)[None].to(device))
                effect = output.action_effect_by_action.cpu().numpy()[0]
                speed = float(np.linalg.norm(effect[1:3], axis=-1).mean())
                distance = float(np.linalg.norm(effect[3:5], axis=-1).mean())
                measures[(identity_label, state_name)] = (speed, distance)
                rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "fixed_query": source, "history_identity_label": identity_label,
                    "functional_state": state_name,
                    "speed_response_magnitude": speed,
                    "distance_response_magnitude": distance,
                })
    follows_theta = bool(
        measures[("history_A", "A_high_speed_low_distance")][0]
        > measures[("history_A", "B_low_speed_high_distance")][0]
        and measures[("history_A", "A_high_speed_low_distance")][1]
        < measures[("history_A", "B_low_speed_high_distance")][1]
        and measures[("history_A", "A_high_speed_low_distance")]
        == measures[("history_B", "A_high_speed_low_distance")]
    )
    decoder.to("cpu")
    return rows, follows_theta


def paired_bootstrap(improvement: np.ndarray, seed: int) -> dict[str, float]:
    values = np.asarray(improvement, dtype=np.float64)
    rng = np.random.default_rng(seed + 66_600)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean_improvement": float(values.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "sample_count": int(len(values)),
    }


def make_figures(
    output_dir: Path, metric_lookup: dict[tuple[str, int], dict[str, Any]],
    state_lookup: dict[int, dict[str, Any]], recovery_lookup: dict[int, dict[str, Any]],
    identification_rows: list[dict[str, Any]], intervention_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]], uncertainty_rows: list[dict[str, Any]],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name: str) -> None:
        path = figures / name; plt.tight_layout(); plt.savefig(path, dpi=150)
        plt.close(); written.append(str(path))

    plt.figure(figsize=(5.5, 4))
    plt.plot(K_VALUES, [state_lookup[k]["Response_State_MAE"] for k in K_VALUES], marker="o")
    plt.xlabel("K"); plt.ylabel("Response State MAE"); plt.title(SYNTHETIC_LABEL); save("response_state_mae_vs_k.png")
    for metric, filename in (
        ("Action_Effect_Error", "effect_error_vs_k.png"),
        ("Sensitivity_MAE", "sensitivity_mae_vs_k.png"),
    ):
        plt.figure(figsize=(6, 4))
        for model in ("F0", "F1", "F2", "F3"):
            keys = sorted(k for candidate, k in metric_lookup if candidate == model)
            plt.plot(keys, [metric_lookup[(model, k)][metric] for k in keys], marker="o", label=model)
        plt.xlabel("K"); plt.ylabel(metric); plt.title(SYNTHETIC_LABEL); plt.legend(); save(filename)
    plt.figure(figsize=(5.5, 4))
    plt.plot(K_VALUES, [recovery_lookup[k]["Person_Effect_Recovery_Ratio"] for k in K_VALUES], marker="o")
    plt.xlabel("K"); plt.ylabel("Recovery Ratio"); plt.title(SYNTHETIC_LABEL); save("recovery_ratio_vs_k.png")
    scatter = [item for item in identification_rows if item["K"] == 10 and item["metric"] == "sample"]
    plt.figure(figsize=(6, 5))
    for dimension in sorted(set(item["dimension"] for item in scatter)):
        selected = [item for item in scatter if item["dimension"] == dimension]
        plt.scatter([item["target"] for item in selected], [item["prediction"] for item in selected], s=12, label=dimension)
    plt.xlabel("GT theta"); plt.ylabel("Predicted theta"); plt.title(SYNTHETIC_LABEL); plt.legend(fontsize=6); save("predicted_vs_gt_response_dimensions.png")
    plt.figure(figsize=(6, 4))
    for dimension in sorted(set(item["dimension"] for item in intervention_rows)):
        selected = [item for item in intervention_rows if item["dimension"] == dimension]
        plt.plot([item["factor"] for item in selected], [item["response_measure"] for item in selected], marker="o", label=dimension)
    plt.xlabel("theta multiplier"); plt.ylabel("response measure"); plt.title(SYNTHETIC_LABEL); plt.legend(fontsize=7); save("functional_intervention_curves.png")
    dimensions = sorted(set(item["dimension"] for item in support_rows))
    types = ("random", "speed_only", "distance_only", "diverse_action")
    matrix = np.asarray([[next(item["value"] for item in support_rows if item["support_type"] == kind and item["dimension"] == dim and item["metric"] == "MAE") for dim in dimensions] for kind in types])
    plt.figure(figsize=(8, 4)); plt.imshow(matrix, aspect="auto"); plt.colorbar(label="MAE")
    plt.xticks(range(len(dimensions)), dimensions, rotation=30); plt.yticks(range(len(types)), types); plt.title(SYNTHETIC_LABEL); save("support_type_information_matrix.png")
    uncertainty = [item for item in uncertainty_rows if item["metric"] == "Mean_Uncertainty" and item["information_status"] == "all"]
    plt.figure(figsize=(6, 4))
    for dimension in sorted(set(item["dimension"] for item in uncertainty)):
        selected = [item for item in uncertainty if item["dimension"] == dimension]
        plt.plot([item["K"] for item in selected], [item["value"] for item in selected], marker="o", label=dimension)
    plt.xlabel("K"); plt.ylabel("theta uncertainty"); plt.title(SYNTHETIC_LABEL); plt.legend(fontsize=6); save("uncertainty_vs_k.png")
    plt.figure(figsize=(6, 4))
    values = [metric_lookup[(model, 10 if model in ("F1", "F2") else 0)]["Action_Effect_Error"] for model in ("F0", "F1", "F2", "F3")]
    plt.bar(("F0", "F1", "F2", "F3"), values); plt.ylabel("Action Effect Error"); plt.title(SYNTHETIC_LABEL); save("f0_f1_f2_f3_comparison.png")
    return written


def main() -> None:
    args = parse_args()
    import torch
    import scripts.run_phase4b_personalization as phase4b
    from src.data.functional_response_state import (
        RESPONSE_STATE_NAMES, population_mean_response_state,
    )
    from src.data.personalization_diagnostics import (
        RESPONSE_COVERED_PROFILES, generate_covered_personal_corpus,
    )
    from src.data.synthetic_interaction import (
        SEEN_PROFILE_IDS, VIRTUAL_PERSON_PROFILES,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device); set_seed(torch, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(SYNTHETIC_LABEL, flush=True)
    train, validation, splits = phase4b.build_corpora(phase4b_args(args))
    t3 = splits["T3_unseen_person_profile"]
    phase4b_summary = json.loads((args.phase4b_dir / "summary.json").read_text(encoding="utf-8"))
    p0_checkpoint = phase4b_summary["training"]["P0"]["checkpoint"]
    original_profiles = tuple(
        profile for profile in VIRTUAL_PERSON_PROFILES
        if profile.profile_id in SEEN_PROFILE_IDS
    )
    original_mean = population_mean_response_state(original_profiles)

    print(f"TRAIN F2-original - {SYNTHETIC_LABEL}", flush=True)
    f2, f2_training = train_or_load_f2(
        "f2_original", args, train, validation, original_mean,
        p0_checkpoint, device, torch,
    )
    covered_full = generate_covered_personal_corpus(seed=args.seed + 60_000)
    covered_train = replace(
        covered_full,
        query_indices=np.flatnonzero(np.isin(covered_full.order_indices, range(10, 20))),
        split_label="functional_response_covered_train",
    )
    covered_validation = replace(
        covered_full,
        query_indices=np.flatnonzero(np.isin(covered_full.order_indices, range(20, 23))),
        split_label="functional_response_covered_validation",
    )
    covered_mean = population_mean_response_state(RESPONSE_COVERED_PROFILES)
    print(f"TRAIN F2-response-covered - {SYNTHETIC_LABEL}", flush=True)
    f2_covered, covered_training = train_or_load_f2(
        "f2_response_covered", args, covered_train, covered_validation,
        covered_mean, p0_checkpoint, device, torch,
    )

    decoder = f2.decoder
    f0_output = predict_decoder_baseline(
        decoder, t3, "generic", original_mean, args, device, torch
    )
    f3_output = predict_decoder_baseline(
        decoder, t3, "oracle", original_mean, args, device, torch
    )
    f0_metrics = future_metrics(f0_output, t3)
    f3_metrics = future_metrics(f3_output, t3)
    metric_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    state_lookup: dict[int, dict[str, Any]] = {}
    f2_outputs: dict[int, dict[str, np.ndarray]] = {}
    fewshot_rows, identification_rows, uncertainty_rows = [], [], []

    for k in K_VALUES:
        metric_lookup[("F0", k)] = f0_metrics["metrics"]
        metric_lookup[("F1", k)] = phase4b_summary["T3_unseen_person_profile"]["P2"][str(k)]
        metric_lookup[("F3", k)] = f3_metrics["metrics"]
        output = predict_f2(f2, t3, k, args, device, torch, "earliest")
        f2_outputs[k] = output
        evaluated = future_metrics(output, t3)
        metric_lookup[("F2", k)] = evaluated["metrics"]
        state_summary, state_rows = state_metrics(output)
        state_lookup[k] = state_summary
        for item in state_rows:
            identification_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                "split": t3.split_label, "K": k, "model": "F2", **item,
            })
        for sample in range(len(output["theta"])):
            for dimension, name in enumerate(RESPONSE_STATE_NAMES):
                identification_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "split": t3.split_label, "K": k, "model": "F2",
                    "profile": int(output["profile"][sample]),
                    "dimension": name, "metric": "sample",
                    "prediction": float(output["theta"][sample, dimension]),
                    "target": float(output["theta_target"][sample, dimension]),
                })
        uncertainty_rows.extend(response_uncertainty_rows(output, k, "F2", args.seed))
    for (model, k), metrics in metric_lookup.items():
        for metric, value in metrics.items():
            fewshot_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                "split": t3.split_label, "person": "ALL", "profile": "ALL",
                "K": k, "model": model, "metric": metric, "value": value,
            })

    recovery_lookup: dict[int, dict[str, Any]] = {}
    recovery_rows = []
    f0_recovery = fixed_query_recovery(decoder, t3, f0_output, args, device, torch)
    f3_recovery = fixed_query_recovery(decoder, t3, f3_output, args, device, torch)
    old_recovery = json.loads(
        (args.phase4b5_dir / "summary.json").read_text(encoding="utf-8")
    )["conditioning_recovery"]["Phase4B-P2"]
    for k in K_VALUES:
        f2_recovery = fixed_query_recovery(
            decoder, t3, f2_outputs[k], args, device, torch
        )
        recovery_lookup[k] = f2_recovery
        for model, recovery in (
            ("F0", f0_recovery),
            ("F1", {
                "Person_Effect_Recovery_Ratio": old_recovery["person_effect_recovery_ratio"]
            }),
            ("F2", f2_recovery), ("F3", f3_recovery),
        ):
            for metric, value in recovery.items():
                recovery_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "split": t3.split_label, "K": k, "model": model,
                    "metric": metric, "value": value,
                })

    support_rows = []
    support_summaries = {}
    for support_type in ("random", "speed_only", "distance_only", "diverse_action"):
        output = predict_f2(f2, t3, 3, args, device, torch, support_type)
        summary, per_dimension = state_metrics(output)
        support_summaries[support_type] = summary
        for item in per_dimension:
            support_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                "split": t3.split_label, "K": 3, "model": "F2",
                "support_type": support_type, **item,
            })

    coverage_rows, coverage_summary = [], {}
    for setting, model in (
        ("original_coverage", f2),
        ("response_covered_training", f2_covered),
    ):
        coverage_summary[setting] = {}
        for k in K_VALUES:
            output = predict_f2(model, t3, k, args, device, torch, "earliest")
            future = future_metrics(output, t3)["metrics"]
            state, _ = state_metrics(output)
            recovery = fixed_query_recovery(model.decoder, t3, output, args, device, torch)
            metrics = {
                "Response_State_MAE": state["Response_State_MAE"],
                "Action_Effect_Error": future["Action_Effect_Error"],
                "Sensitivity_MAE": future["Sensitivity_MAE"],
                "Person_Effect_Recovery_Ratio": recovery["Person_Effect_Recovery_Ratio"],
            }
            coverage_summary[setting][str(k)] = metrics
            for metric, value in metrics.items():
                coverage_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "split": t3.split_label, "K": k, "model": setting,
                    "metric": metric, "value": value,
                    "test_profiles_in_training": False,
                })

    intervention_theta = f3_output["theta_target"].mean(axis=0).astype(np.float32)
    intervention_rows, intervention_passed = functional_interventions(
        decoder, t3, intervention_theta, args, device, torch
    )
    swap_rows, swap_passed = functional_swap(
        decoder, t3, args, device, torch
    )

    state_mae_per_sample = {
        k: np.abs(f2_outputs[k]["theta"] - f2_outputs[k]["theta_target"]).mean(axis=1)
        for k in K_VALUES
    }
    state_evidence = paired_bootstrap(
        state_mae_per_sample[0] - state_mae_per_sample[10], args.seed
    )
    effect_evidence = paired_bootstrap(
        future_metrics(f2_outputs[0], t3)["per_sample"]["effect_error"]
        - future_metrics(f2_outputs[10], t3)["per_sample"]["effect_error"],
        args.seed,
    )
    k_effect = [metric_lookup[("F2", k)]["Action_Effect_Error"] for k in K_VALUES]
    k_sensitivity = [metric_lookup[("F2", k)]["Sensitivity_MAE"] for k in K_VALUES]
    diverse_complete = bool(
        support_summaries["diverse_action"]["Response_State_MAE"]
        < min(
            support_summaries["speed_only"]["Response_State_MAE"],
            support_summaries["distance_only"]["Response_State_MAE"],
        )
    )
    criteria = {
        "F3_oracle_better_than_F0": bool(
            f3_metrics["metrics"]["Action_Effect_Error"]
            < f0_metrics["metrics"]["Action_Effect_Error"]
            and f3_metrics["metrics"]["Sensitivity_MAE"]
            < f0_metrics["metrics"]["Sensitivity_MAE"]
        ),
        "F2_state_MAE_credible_K_improvement": bool(
            state_evidence["ci95_lower"] > 0
            and state_lookup[10]["Response_State_MAE"] < state_lookup[0]["Response_State_MAE"]
        ),
        "F2_response_credible_K_improvement": bool(
            effect_evidence["ci95_lower"] > 0
            and min(k_effect[2:]) < k_effect[0]
        ),
        "F2_recovery_above_phase4b": bool(
            recovery_lookup[10]["Person_Effect_Recovery_Ratio"]
            > max(0.05, old_recovery["person_effect_recovery_ratio"] * 2)
        ),
        "functional_interventions_directional": bool(all(intervention_passed.values())),
        "functional_swap_follows_theta": swap_passed,
        "diverse_support_more_complete": diverse_complete,
    }
    criteria["five_seed_gate_passed"] = bool(all(criteria.values()))
    criteria["ready_for_phase4c"] = False

    write_csv(args.output_dir / "response_state_identification.csv", identification_rows)
    write_csv(args.output_dir / "fewshot.csv", fewshot_rows)
    write_csv(args.output_dir / "effect_recovery.csv", recovery_rows)
    write_csv(args.output_dir / "functional_intervention.csv", intervention_rows)
    write_csv(args.output_dir / "functional_swap.csv", swap_rows)
    write_csv(args.output_dir / "support_information.csv", support_rows)
    write_csv(args.output_dir / "coverage_experiment.csv", coverage_rows)
    write_csv(args.output_dir / "response_uncertainty.csv", uncertainty_rows)
    multiseed_rows = [
        {
            "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
            "split": t3.split_label, "K": k, "model": "F2",
            "metric": metric, "value": metric_lookup[("F2", k)][metric],
            "detail": "seed42_gate_run_only; five-seed not started",
        }
        for k in K_VALUES
        for metric in ("Global_MPJPE", "Action_Effect_Error", "Sensitivity_MAE")
    ]
    write_csv(args.output_dir / "multiseed.csv", multiseed_rows)
    figures = make_figures(
        args.output_dir, metric_lookup, state_lookup, recovery_lookup,
        identification_rows, intervention_rows, support_rows, uncertainty_rows,
    )
    summary = {
        "label": SYNTHETIC_LABEL, "seed": args.seed,
        "five_seed_started": False,
        "phase4a_4b_4b5_results_untouched": True,
        "training": {
            "F2_original": f2_training,
            "F2_response_covered": covered_training,
        },
        "F0": f0_metrics["metrics"],
        "F1_Phase4B_P2_K10": metric_lookup[("F1", 10)],
        "F2_by_K": {str(k): {**metric_lookup[("F2", k)], **state_lookup[k]} for k in K_VALUES},
        "F3": f3_metrics["metrics"],
        "effect_recovery_by_K": recovery_lookup,
        "support_information": support_summaries,
        "coverage_experiment": coverage_summary,
        "intervention_passed": intervention_passed,
        "swap_passed": swap_passed,
        "paired_evidence": {
            "Response_State_MAE_K0_minus_K10": state_evidence,
        "F2_K0_Effect_Error_minus_F2_K10": effect_evidence,
        },
        "success_criteria": criteria,
        "figures": figures,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"F2 theta MAE K0={state_lookup[0]['Response_State_MAE']:.6f} "
        f"K10={state_lookup[10]['Response_State_MAE']:.6f} "
        f"recovery={recovery_lookup[10]['Person_Effect_Recovery_Ratio']:.4f} "
        f"five_seed_gate={criteria['five_seed_gate_passed']}", flush=True,
    )


if __name__ == "__main__":
    main()
