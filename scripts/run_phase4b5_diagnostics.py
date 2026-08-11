"""Phase 4B.5 synthetic personalization diagnosis and gated meta-adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYNTHETIC_LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "all"), default="a")
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
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase4b5",
    )
    return parser.parse_args()


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase4b_arguments(args: argparse.Namespace) -> Namespace:
    return Namespace(
        stage="full", seed=args.seed, epochs=20, batch_size=args.batch_size,
        history_frames=20, future_frames=10, sample_rate=10.0,
        learning_rate=1e-3, noise_std=0.005, occlusion_rate=0.10,
        persons_per_profile=2, interactions_per_person=30,
        benchmark_batch_size=32, benchmark_warmup=50,
        benchmark_repetitions=200, output_dir=args.phase4b_dir, device=args.device,
    )


def distribution_rows(corpus: Any, split_name: str) -> list[dict[str, Any]]:
    from src.data.synthetic_interaction import PROFILE_BY_ID
    rows = []
    split = corpus.split
    for source in corpus.query_indices:
        source = int(source)
        effects = split.action_effect_by_action[source]
        actions = split.candidate_actions[source]
        nonkeep = actions != 0
        sensitivity = float(np.linalg.norm(effects[nonkeep], axis=-1).mean())
        profile_id = int(split.person_profile_id[source])
        delay = float(PROFILE_BY_ID[profile_id].response_delay)
        for action_index, action in enumerate(actions):
            rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL,
                "seed": 42,
                "split": split_name,
                "person": str(corpus.person_instance_ids[source]),
                "profile": profile_id,
                "action": int(action),
                "gt_action_sensitivity": sensitivity,
                "effect_magnitude": float(np.linalg.norm(effects[action_index], axis=-1).mean()),
                "response_delay": delay,
            })
    return rows


def profile_sufficiency_audit(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from src.data.personalization_diagnostics import (
        PROFILE_PARAMETER_NAMES, descriptors_for_split,
    )
    from src.data.synthetic_interaction import (
        PROFILE_BY_ID, VIRTUAL_PERSON_PROFILES, generate_interaction_split,
    )
    from src.evaluation.personalization_diagnostics import profile_response_correlations
    profile_rows = []
    for profile in VIRTUAL_PERSON_PROFILES:
        split = generate_interaction_split(
            80, seed + 10_000 + profile.profile_id, f"profile_audit_{profile.profile_id}",
            profile_ids=(profile.profile_id,), noise_std=0.0, occlusion_rate=0.0,
        )
        effect_magnitude = np.linalg.norm(split.action_effect_by_action, axis=-1).mean(axis=(-1, -2))
        nonkeep = split.candidate_actions != 0
        descriptors = descriptors_for_split(split, PROFILE_BY_ID)
        item: dict[str, Any] = {
            "synthetic_interaction": SYNTHETIC_LABEL,
            "profile": profile.profile_id,
            "profile_name": profile.name,
            **{name: float(getattr(profile, name)) for name in PROFILE_PARAMETER_NAMES},
            "gt_action_sensitivity": float(effect_magnitude[nonkeep].mean()),
            "response_delay_metric": float(profile.response_delay),
            "lateral_response_magnitude": float(np.abs(descriptors[..., 4][nonkeep]).mean()),
            "speed_response_magnitude": float(np.abs(descriptors[..., 2][nonkeep]).mean()),
        }
        for action in range(5):
            item[f"effect_magnitude_action_{action}"] = float(
                effect_magnitude[split.candidate_actions == action].mean()
            )
        profile_rows.append(item)
    response_metrics = (
        "gt_action_sensitivity", "response_delay_metric",
        "lateral_response_magnitude", "speed_response_magnitude",
        "effect_magnitude_action_1", "effect_magnitude_action_2",
        "effect_magnitude_action_3", "effect_magnitude_action_4",
    )
    correlations = profile_response_correlations(
        profile_rows, PROFILE_PARAMETER_NAMES, response_metrics
    )
    response_vectors = np.asarray([
        [float(row[name]) for name in response_metrics] for row in profile_rows
    ])
    parameter_vectors = np.asarray([
        [float(row[name]) for name in PROFILE_PARAMETER_NAMES] for row in profile_rows
    ])
    response_vectors = (response_vectors - response_vectors.mean(0)) / response_vectors.std(0).clip(min=1e-8)
    parameter_vectors = (parameter_vectors - parameter_vectors.mean(0)) / parameter_vectors.std(0).clip(min=1e-8)
    pairs = []
    for left in range(len(profile_rows)):
        for right in range(left + 1, len(profile_rows)):
            pairs.append({
                "left": int(profile_rows[left]["profile"]),
                "right": int(profile_rows[right]["profile"]),
                "parameter_distance": float(np.linalg.norm(parameter_vectors[left] - parameter_vectors[right])),
                "response_distance": float(np.linalg.norm(response_vectors[left] - response_vectors[right])),
            })
    closest = min(pairs, key=lambda item: item["response_distance"])
    summary = {
        "profile_count": len(profile_rows),
        "closest_response_pair": closest,
        "non_identifiability_warning": bool(
            closest["response_distance"] < 0.25 and closest["parameter_distance"] > 1.0
        ),
    }
    return profile_rows, correlations, summary


def response_distribution_audit(corpora: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from src.evaluation.personalization_diagnostics import distribution_coverage
    rows = []
    for name, corpus in corpora.items():
        rows.extend(distribution_rows(corpus, name))
    coverage_rows = []
    train_rows = [item for item in rows if item["split"] == "train"]
    for test_name in ("T3_unseen_person_profile", "T5_unseen_person_unseen_context"):
        test_rows = [item for item in rows if item["split"] == test_name]
        for action in range(5):
            for metric in ("gt_action_sensitivity", "effect_magnitude", "response_delay"):
                train_values = np.asarray([item[metric] for item in train_rows if item["action"] == action])
                test_values = np.asarray([item[metric] for item in test_rows if item["action"] == action])
                coverage_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL,
                    "train_split": "train", "test_split": test_name,
                    "action": action, "metric": metric,
                    **distribution_coverage(train_values, test_values),
                })
    return rows, coverage_rows


def train_or_load_oracle_effect(
    args: argparse.Namespace, train: Any, validation: Any, phase4b_metadata: dict[str, Any],
    device: Any, torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from torch.utils.data import DataLoader
    from src.models.personalization_diagnostics import OracleEffectWorldModel
    from src.training.train_meta_personalization import MetaEpisodeDataset
    checkpoint = args.output_dir / "checkpoints" / "o_effect_best.pt"
    metadata_path = args.output_dir / "checkpoints" / "o_effect_training.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    signature = {"seed": args.seed, "epochs": 0, "protocol": 2,
                 "structured_descriptor_decoder": True}
    model = OracleEffectWorldModel()
    p0_state = torch.load(
        phase4b_metadata["P0"]["checkpoint"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    model.natural_backbone.load_state_dict(p0_state)
    for parameter in model.natural_backbone.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "signature": signature, "checkpoint": None,
        "best_epoch": 0, "best_validation_Global_MPJPE": None,
        "training_time_seconds": 0.0,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    write_json(metadata_path, metadata)
    return model, metadata


def predict_oracle_effect(model: Any, corpus: Any, args: argparse.Namespace, device: Any, torch: Any) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from src.training.train_meta_personalization import MetaEpisodeDataset
    loader = DataLoader(
        MetaEpisodeDataset(corpus, 0, oracle_access=True),
        batch_size=args.batch_size,
    )
    output: dict[str, list[Any]] = {name: [] for name in ("future", "natural", "log_std", "effect_log_std", "source")}
    model.to(device).eval()
    with torch.inference_mode():
        for raw in loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            result = model(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"], batch["effect_descriptors"],
            )
            output["future"].append(result.future_by_action.cpu())
            output["natural"].append(result.natural_future.cpu())
            output["log_std"].append(result.root_log_std_by_action.cpu())
            output["effect_log_std"].append(result.action_effect_root_log_std_by_action.cpu())
            output["source"].append(batch["source_index"].cpu())
    model.to("cpu")
    return {name: torch.cat(values).numpy() for name, values in output.items()}


def old_p2_conditioning_audit(
    args: argparse.Namespace, corpus: Any, metadata: dict[str, Any], device: Any, torch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import scripts.run_phase4b_personalization as phase4b
    from torch.utils.data import DataLoader
    from src.data.synthetic_interaction import PROFILE_BY_ID, simulate_interaction_future
    from src.evaluation.personalization_diagnostics import person_effect_recovery_ratio
    from src.models.personalization_diagnostics import prediction_from_encoded
    from src.training.train_meta_personalization import MetaEpisodeDataset
    phase_args = phase4b_arguments(args)
    model = phase4b.load_model("P2", phase_args, 10, metadata["P2"], torch).to(device).eval()
    person_indices = phase4b.person_index_for(corpus)
    predictions = phase4b.predict(
        model, corpus, 10, "earliest", person_indices, args.batch_size, device,
        torch, args.seed,
    )
    model.to(device).eval()
    centroids = {
        int(profile): predictions["z"][predictions["profile"] == profile].mean(axis=0)
        for profile in np.unique(predictions["profile"])
    }
    loader = DataLoader(MetaEpisodeDataset(corpus, 10, "earliest"), batch_size=1)
    raw = next(iter(loader))
    batch = {name: value.to(device) for name, value in raw.items()}
    with torch.inference_mode():
        encoded = model.encode_context(
            batch["history"], batch["robot"], batch["confidence"], batch["visibility"],
            support_features=batch["support_features"], support_mask=batch["support_mask"],
        )
    predicted_effects, expected_effects, sensitivity_rows = [], [], []
    source = int(batch["source_index"].item())
    split = corpus.split
    for profile_id in sorted(centroids):
        z = torch.from_numpy(centroids[profile_id]).to(device=device, dtype=batch["history"].dtype)[None]
        with torch.inference_mode():
            output = prediction_from_encoded(model, (*encoded[:-1], z), batch["actions"])
        predicted_effect = (output.future_by_action - output.natural_future[:, None]).cpu().numpy()[0]
        profile = PROFILE_BY_ID[profile_id]
        simulations = [
            simulate_interaction_future(
                split.human_history[source], split.natural_future[source],
                split.robot_history[source], int(action), profile,
            )
            for action in split.candidate_actions[source]
        ]
        expected_effect = np.stack([item.action_effect for item in simulations])
        predicted_effects.append(predicted_effect)
        expected_effects.append(expected_effect)
        sensitivity_rows.append({
            "synthetic_interaction": SYNTHETIC_LABEL,
            "model": "Phase4B_P2", "query_source": source, "profile": profile_id,
            "person_conditioning_sensitivity": float(np.linalg.norm(predicted_effect, axis=-1).mean()),
            "gt_effect_magnitude": float(np.linalg.norm(expected_effect, axis=-1).mean()),
        })
    recovery = person_effect_recovery_ratio(
        np.stack(predicted_effects), np.stack(expected_effects)
    )
    low, high = min(centroids), max(centroids)
    interpolation_rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        z_value = (1.0 - alpha) * centroids[low] + alpha * centroids[high]
        z = torch.from_numpy(z_value).to(device=device, dtype=batch["history"].dtype)[None]
        with torch.inference_mode():
            output = prediction_from_encoded(model, (*encoded[:-1], z), batch["actions"])
        effect = (output.future_by_action - output.natural_future[:, None]).cpu().numpy()[0]
        for action_index, action in enumerate(split.candidate_actions[source]):
            interpolation_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL,
                "model": "Phase4B_P2", "query_source": source,
                "low_profile": low, "high_profile": high,
                "alpha": alpha, "action": int(action),
                "predicted_response_amplitude": float(
                    np.linalg.norm(effect[action_index], axis=-1).mean()
                ),
            })
    model.to("cpu")
    return sensitivity_rows, interpolation_rows, recovery


def make_stage_a_figures(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    plt.figure(figsize=(6, 4))
    for split in ("train", "T3_unseen_person_profile", "T5_unseen_person_unseen_context"):
        values = [item["gt_action_sensitivity"] for item in rows if item["split"] == split and item["action"] == 1]
        plt.hist(values, bins=16, alpha=0.45, label=split)
    plt.xlabel("GT action sensitivity")
    plt.ylabel("count")
    plt.title(SYNTHETIC_LABEL)
    plt.legend(fontsize=7)
    path = figures / "sensitivity_distribution.png"
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); written.append(str(path))
    plt.figure(figsize=(7, 4))
    positions, labels, data = [], [], []
    position = 0
    for split in ("train", "T3_unseen_person_profile"):
        for action in range(1, 5):
            data.append([item["effect_magnitude"] for item in rows if item["split"] == split and item["action"] == action])
            positions.append(position); labels.append(f"{split[:2]}-A{action}"); position += 1
        position += 1
    plt.boxplot(data, positions=positions, widths=0.65, showfliers=False)
    plt.xticks(positions, labels, rotation=30)
    plt.ylabel("GT effect magnitude")
    plt.title(SYNTHETIC_LABEL)
    path = figures / "effect_distribution_by_action.png"
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); written.append(str(path))
    return written


def run_stage_a(args: argparse.Namespace, device: Any, torch: Any) -> dict[str, Any]:
    import scripts.run_phase4b_personalization as phase4b
    from src.evaluation.personal_response_metrics import personal_response_metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_args = phase4b_arguments(args)
    train, validation, splits = phase4b.build_corpora(phase_args)
    phase_summary = json.loads((args.phase4b_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = phase_summary["training"]

    profile_rows, correlations, sufficiency = profile_sufficiency_audit(args.seed)
    distribution, coverage = response_distribution_audit({
        "train": train, "validation": validation,
        "T3_unseen_person_profile": splits["T3_unseen_person_profile"],
        "T5_unseen_person_unseen_context": splits["T5_unseen_person_unseen_context"],
    })
    sensitivity_rows, interpolation_rows, recovery = old_p2_conditioning_audit(
        args, splits["T3_unseen_person_profile"], metadata, device, torch
    )
    o_effect_model, o_effect_training = train_or_load_oracle_effect(
        args, train, validation, metadata, device, torch
    )
    t3 = splits["T3_unseen_person_profile"]
    o_effect_output = predict_oracle_effect(o_effect_model, t3, args, device, torch)
    o_effect_metrics, _, _ = personal_response_metrics(
        o_effect_output["future"], o_effect_output["natural"],
        o_effect_output["log_std"], o_effect_output["effect_log_std"],
        t3.split, o_effect_output["source"],
    )
    source = t3.query_indices
    target = t3.split.future_by_action[source]
    natural = t3.split.natural_future[source]
    zero_log_std = np.zeros((*target.shape[:3], 3), dtype=np.float32)
    future_ceiling_metrics, _, _ = personal_response_metrics(
        natural[:, None] + t3.split.action_effect_by_action[source], natural,
        zero_log_std, zero_log_std, t3.split, source,
    )
    p0 = phase_summary["T3_unseen_person_profile"]["P0"]["0"]
    old_p3 = phase_summary["T3_unseen_person_profile"]["P3"]["0"]
    oracle_rows = []
    for model, metrics in (
        ("P0", p0), ("O-PROFILE_old_P3", old_p3),
        ("O-EFFECT", o_effect_metrics), ("O-FUTURE-EFFECT", future_ceiling_metrics),
    ):
        for metric in ("Global_MPJPE", "Local_MPJPE", "Root_ADE", "Root_FDE", "Action_Effect_Error", "Sensitivity_MAE"):
            oracle_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                "split": "T3_unseen_person_profile", "model": model,
                "metric": metric, "value": metrics.get(metric),
            })
    write_csv(args.output_dir / "oracle_audit.csv", oracle_rows)
    write_csv(args.output_dir / "profile_response_correlations.csv", correlations)
    write_csv(args.output_dir / "response_distribution.csv", distribution + coverage)
    write_csv(args.output_dir / "conditioning_sensitivity.csv", sensitivity_rows + [{
        "synthetic_interaction": SYNTHETIC_LABEL, "model": "Phase4B_P2", **recovery
    }])
    write_csv(args.output_dir / "latent_interpolation.csv", interpolation_rows)
    figures = make_stage_a_figures(args.output_dir, distribution)
    effect_information_valuable = bool(
        o_effect_metrics["Action_Effect_Error"] < p0["Action_Effect_Error"]
    )
    diagnostic = {
        "label": SYNTHETIC_LABEL,
        "seed": args.seed,
        "stage": "A",
        "phase4b_results_untouched": True,
        "oracle": {
            "P0": p0,
            "old_P3": old_p3,
            "O_EFFECT": o_effect_metrics,
            "O_FUTURE_EFFECT": future_ceiling_metrics,
            "O_EFFECT_training": o_effect_training,
        },
        "profile_sufficiency": sufficiency,
        "phase4b_conditioning_recovery": recovery,
        "response_information_valuable": effect_information_valuable,
        "stage_b_authorized_by_diagnostic": effect_information_valuable,
        "figures": figures,
    }
    write_json(args.output_dir / "diagnostic_summary.json", diagnostic)
    print(
        f"Stage A: P0 Effect={p0['Action_Effect_Error']:.6f} "
        f"O-EFFECT={o_effect_metrics['Action_Effect_Error']:.6f} "
        f"O-FUTURE Global={future_ceiling_metrics['Global_MPJPE']:.9f} "
        f"recovery={recovery['person_effect_recovery_ratio']:.4f}", flush=True,
    )
    return {"diagnostic": diagnostic, "train": train, "validation": validation, "splits": splits, "metadata": metadata}


def train_or_load_meta(
    name: str, args: argparse.Namespace, train: Any, validation: Any,
    phase4b_metadata: dict[str, Any], weights: Any, device: Any, torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from torch.utils.data import DataLoader
    from src.models.personalization_diagnostics import MetaPersonalizedWorldModel
    from src.training.train_meta_personalization import MetaEpisodeDataset, train_meta_model
    checkpoint = args.output_dir / "checkpoints" / f"{name.lower().replace('+', '_')}_best.pt"
    metadata_path = checkpoint.with_name(checkpoint.stem.replace("_best", "_training") + ".json")
    signature = {
        "seed": args.seed, "epochs": args.epochs, "protocol": 1,
        "train_split": train.split_label, "weights": weights.__dict__,
        "episodic_support_query": True,
    }
    model = MetaPersonalizedWorldModel()
    phase4b_state = torch.load(
        phase4b_metadata["P2"]["checkpoint"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    model.load_state_dict(phase4b_state)
    if checkpoint.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
            return model, metadata
    train_dataset = MetaEpisodeDataset(train, (0, 1, 3, 5, 10), "random", args.seed)
    validation_dataset = MetaEpisodeDataset(validation, 5, "earliest", args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    result = train_meta_model(
        model, train_loader, validation_loader, device, args.epochs, checkpoint,
        weights, args.learning_rate,
    )
    metadata = {"signature": signature, "checkpoint": str(checkpoint), **result}
    write_json(metadata_path, metadata)
    return model, metadata


def predict_meta(
    model: Any, corpus: Any, k: int, args: argparse.Namespace,
    device: Any, torch: Any,
) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from src.training.train_meta_personalization import MetaEpisodeDataset
    loader = DataLoader(
        MetaEpisodeDataset(corpus, k, "earliest", args.seed),
        batch_size=args.batch_size,
    )
    values: dict[str, list[Any]] = {
        name: [] for name in (
            "future", "natural", "log_std", "effect_log_std", "z", "source", "profile"
        )
    }
    model.to(device).eval()
    with torch.inference_mode():
        for raw in loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            personalized, _ = model.paired_forward(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"],
                batch["support_features"], batch["support_mask"],
            )
            values["future"].append(personalized.future_by_action.cpu())
            values["natural"].append(personalized.natural_future.cpu())
            values["log_std"].append(personalized.root_log_std_by_action.cpu())
            values["effect_log_std"].append(personalized.action_effect_root_log_std_by_action.cpu())
            values["z"].append(personalized.z_person.cpu())
            values["source"].append(batch["source_index"].cpu())
            values["profile"].append(batch["profile_id"].cpu())
    model.to("cpu")
    return {name: torch.cat(items).numpy() for name, items in values.items()}


def train_or_load_response_oracle(
    args: argparse.Namespace, train: Any, validation: Any,
    phase4b_metadata: dict[str, Any], device: Any, torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from torch.utils.data import DataLoader
    from src.models.personalization_diagnostics import ResponseOracleWorldModel
    from src.training.train_meta_personalization import MetaEpisodeDataset
    from src.training.train_personal_response import personal_interaction_loss
    checkpoint = args.output_dir / "checkpoints" / "p3_response_oracle_best.pt"
    metadata_path = args.output_dir / "checkpoints" / "p3_response_oracle_training.json"
    signature = {"seed": args.seed, "epochs": args.epochs, "protocol": 1,
                 "explicit_response_statistics_supervision": True}
    model = ResponseOracleWorldModel()
    p0_state = torch.load(
        phase4b_metadata["P0"]["checkpoint"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    model.backbone.load_state_dict(p0_state)
    if checkpoint.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
            return model, metadata
    train_loader = DataLoader(
        MetaEpisodeDataset(
            train, (0, 1, 3, 5, 10), "random", args.seed,
            oracle_access=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        MetaEpisodeDataset(
            validation, 5, "earliest", args.seed, oracle_access=True
        ),
        batch_size=args.batch_size,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        model.train()
        for raw in train_loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"], batch["profile_parameters"],
            )
            absolute, _ = personal_interaction_loss(
                output, batch["target"], batch["natural"], batch["supervision"],
                batch["actions"],
            )
            statistics_loss = (
                output.predicted_response_statistics - batch["response_statistics"]
            ).square().mean()
            loss = absolute + statistics_loss
            loss.backward(); optimizer.step()
        model.eval(); total = count = 0.0
        with torch.inference_mode():
            for raw in validation_loader:
                batch = {name: value.to(device) for name, value in raw.items()}
                output = model(
                    batch["history"], batch["robot"], batch["actions"],
                    batch["confidence"], batch["visibility"], batch["profile_parameters"],
                )
                error = torch.linalg.vector_norm(
                    output.future_by_action - batch["target"], dim=-1
                ).mean(dim=(-1, -2))
                mask = batch["supervision"].to(error.dtype)
                total += float((error * mask).sum().item()); count += float(mask.sum().item())
        validation_metric = total / count
        if validation_metric < best:
            best, best_epoch = validation_metric, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best_state}, checkpoint)
    model.load_state_dict(best_state)
    metadata = {
        "signature": signature, "checkpoint": str(checkpoint),
        "best_epoch": best_epoch, "best_validation_Global_MPJPE": best,
        "training_time_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(metadata_path, metadata)
    return model, metadata


def predict_response_oracle(
    model: Any, corpus: Any, args: argparse.Namespace, device: Any, torch: Any,
) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from src.training.train_meta_personalization import MetaEpisodeDataset
    loader = DataLoader(
        MetaEpisodeDataset(corpus, 0, oracle_access=True),
        batch_size=args.batch_size,
    )
    values: dict[str, list[Any]] = {name: [] for name in ("future", "natural", "log_std", "effect_log_std", "source")}
    model.to(device).eval()
    with torch.inference_mode():
        for raw in loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            output = model(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"], batch["profile_parameters"],
            )
            values["future"].append(output.future_by_action.cpu())
            values["natural"].append(output.natural_future.cpu())
            values["log_std"].append(output.root_log_std_by_action.cpu())
            values["effect_log_std"].append(output.action_effect_root_log_std_by_action.cpu())
            values["source"].append(batch["source_index"].cpu())
    model.to("cpu")
    return {name: torch.cat(items).numpy() for name, items in values.items()}


def meta_conditioning_audit(
    name: str, model: Any, corpus: Any, args: argparse.Namespace,
    device: Any, torch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    from torch.utils.data import DataLoader
    from src.data.synthetic_interaction import PROFILE_BY_ID, simulate_interaction_future
    from src.evaluation.personalization_diagnostics import person_effect_recovery_ratio
    from src.training.train_meta_personalization import MetaEpisodeDataset
    predictions = predict_meta(model, corpus, 10, args, device, torch)
    centroids = {
        int(profile): predictions["z"][predictions["profile"] == profile].mean(axis=0)
        for profile in np.unique(predictions["profile"])
    }
    raw = next(iter(DataLoader(MetaEpisodeDataset(corpus, 10, "earliest"), batch_size=1)))
    batch = {key: value.to(device) for key, value in raw.items()}
    model.to(device).eval()
    with torch.inference_mode():
        encoded = model.encode_context(
            batch["history"], batch["robot"], batch["confidence"], batch["visibility"],
            support_features=batch["support_features"], support_mask=batch["support_mask"],
        )
    source = int(batch["source_index"].item())
    split = corpus.split
    predicted_effects, expected_effects, sensitivity_rows = [], [], []
    for profile_id in sorted(centroids):
        z = torch.from_numpy(centroids[profile_id]).to(
            device=device, dtype=batch["history"].dtype
        )[None]
        with torch.inference_mode():
            output = model.interpolate_forward(encoded[:-1], z, batch["actions"])
        predicted_effect = (
            output.future_by_action - output.natural_future[:, None]
        ).cpu().numpy()[0]
        expected_effect = np.stack([
            simulate_interaction_future(
                split.human_history[source], split.natural_future[source],
                split.robot_history[source], int(action), PROFILE_BY_ID[profile_id],
            ).action_effect
            for action in split.candidate_actions[source]
        ])
        predicted_effects.append(predicted_effect); expected_effects.append(expected_effect)
        sensitivity_rows.append({
            "synthetic_interaction": SYNTHETIC_LABEL, "model": name,
            "query_source": source, "profile": profile_id,
            "person_conditioning_sensitivity": float(np.linalg.norm(predicted_effect, axis=-1).mean()),
            "gt_effect_magnitude": float(np.linalg.norm(expected_effect, axis=-1).mean()),
        })
    recovery = person_effect_recovery_ratio(
        np.stack(predicted_effects), np.stack(expected_effects)
    )
    response_by_profile = {
        profile: float(np.linalg.norm(expected_effects[index], axis=-1).mean())
        for index, profile in enumerate(sorted(centroids))
    }
    low = min(response_by_profile, key=response_by_profile.get)
    high = max(response_by_profile, key=response_by_profile.get)
    interpolation_rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        z_value = (1 - alpha) * centroids[low] + alpha * centroids[high]
        z = torch.from_numpy(z_value).to(device=device, dtype=batch["history"].dtype)[None]
        with torch.inference_mode():
            output = model.interpolate_forward(encoded[:-1], z, batch["actions"])
        effect = (output.future_by_action - output.natural_future[:, None]).cpu().numpy()[0]
        for action_index, action in enumerate(split.candidate_actions[source]):
            interpolation_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "model": name,
                "query_source": source, "low_profile": low, "high_profile": high,
                "alpha": alpha, "action": int(action),
                "predicted_response_amplitude": float(
                    np.linalg.norm(effect[action_index], axis=-1).mean()
                ),
            })
    model.to("cpu")
    return sensitivity_rows, interpolation_rows, recovery


def uncertainty_calibration_audit(
    args: argparse.Namespace, train: Any, validation: Any, t3: Any,
    metadata: dict[str, Any], device: Any, torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import scripts.run_phase4b_personalization as phase4b
    from src.data.skeleton_schema import compute_root
    from src.evaluation.personalization_diagnostics import (
        calibrated_uncertainty_metrics, fit_uncertainty_scale,
    )
    phase_args = phase4b_arguments(args)
    person_indices = phase4b.person_index_for(train)
    model = phase4b.load_model("P2", phase_args, len(person_indices), metadata["P2"], torch)
    validation_output = phase4b.predict(
        model, validation, 10, "earliest", person_indices, args.batch_size,
        device, torch, args.seed,
    )
    test_output = phase4b.predict(
        model, t3, 10, "earliest", person_indices, args.batch_size,
        device, torch, args.seed,
    )

    def arrays(corpus: Any, output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        source = output["source"]
        target = corpus.split.future_by_action[source]
        natural = corpus.split.natural_future[source]
        predicted_root = compute_root(output["future"])
        target_root = compute_root(target)
        predicted_effect_root = compute_root(output["future"] - output["natural"][:, None])
        target_effect_root = compute_root(target - natural[:, None])
        nonkeep = corpus.split.candidate_actions[source] != 0
        return {
            "root_error": target_root - predicted_root,
            "root_sigma": np.exp(output["log_std"]),
            "effect_error": (target_effect_root - predicted_effect_root)[nonkeep],
            "effect_sigma": np.exp(output["effect_log_std"])[nonkeep],
        }
    val = arrays(validation, validation_output)
    test = arrays(t3, test_output)
    root_scale = fit_uncertainty_scale(
        val["root_error"], val["root_sigma"], "validation"
    )
    effect_scale = fit_uncertainty_scale(
        val["effect_error"], val["effect_sigma"], "validation"
    )
    rows = []
    summary = {"fit_split": "validation", "test_split": t3.split_label}
    for component, error_key, sigma_key, scale in (
        ("root", "root_error", "root_sigma", root_scale),
        ("action_effect", "effect_error", "effect_sigma", effect_scale),
    ):
        summary[component] = {"scale": scale}
        for state, applied_scale in (("before", 1.0), ("after", scale)):
            metrics = calibrated_uncertainty_metrics(
                test[error_key], test[sigma_key], applied_scale
            )
            summary[component][state] = metrics
            for metric, value in metrics.items():
                rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "fit_split": "validation", "evaluation_split": t3.split_label,
                    "model": "Phase4B_P2_K10", "component": component,
                    "calibration": state, "metric": metric, "value": value,
                })
    return rows, summary


def make_stage_c_figures(
    output_dir: Path, metric_lookup: dict[tuple[str, int], dict[str, Any]],
    conditioning_rows: list[dict[str, Any]], interpolation_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    written = []
    models = sorted({model for model, _ in metric_lookup})
    for metric, filename in (
        ("Global_MPJPE", "meta_global_vs_k.png"),
        ("Action_Effect_Error", "meta_effect_vs_k.png"),
        ("Sensitivity_MAE", "meta_sensitivity_mae_vs_k.png"),
    ):
        plt.figure(figsize=(6.5, 4.2))
        for model in models:
            keys = sorted(k for candidate, k in metric_lookup if candidate == model)
            if keys:
                plt.plot(keys, [metric_lookup[(model, k)][metric] for k in keys], marker="o", label=model)
        plt.xlabel("K"); plt.ylabel(metric); plt.title(SYNTHETIC_LABEL); plt.legend(fontsize=7)
        path = figures / filename; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); written.append(str(path))
    recovery = [item for item in conditioning_rows if "person_effect_recovery_ratio" in item]
    plt.figure(figsize=(6, 4))
    plt.bar([item["model"] for item in recovery], [item["person_effect_recovery_ratio"] for item in recovery])
    plt.ylabel("Person Effect Recovery Ratio"); plt.xticks(rotation=25); plt.title(SYNTHETIC_LABEL)
    path = figures / "conditioning_recovery.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); written.append(str(path))
    plt.figure(figsize=(6, 4))
    for model in sorted({item["model"] for item in interpolation_rows}):
        selected = [item for item in interpolation_rows if item["model"] == model and item["action"] == 4]
        plt.plot([item["alpha"] for item in selected], [item["predicted_response_amplitude"] for item in selected], marker="o", label=model)
    plt.xlabel("latent interpolation alpha"); plt.ylabel("A4 response amplitude"); plt.title(SYNTHETIC_LABEL); plt.legend(fontsize=7)
    path = figures / "latent_intervention_amplitude.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); written.append(str(path))
    coverage = [item for item in calibration_rows if item["metric"] in ("Coverage_50", "Coverage_80", "Coverage_90")]
    plt.figure(figsize=(6, 4))
    x = np.arange(3); width = 0.18
    index = 0
    for component in ("root", "action_effect"):
        for state in ("before", "after"):
            selected = [item for item in coverage if item["component"] == component and item["calibration"] == state]
            plt.bar(x + index * width, [item["value"] for item in selected], width, label=f"{component}-{state}")
            index += 1
    plt.plot([-0.2, 2.8], [0.5, 0.8], alpha=0)
    plt.xticks(x + 1.5 * width, ("50%", "80%", "90%")); plt.ylabel("coverage"); plt.title(SYNTHETIC_LABEL); plt.legend(fontsize=7)
    path = figures / "uncertainty_calibration.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); written.append(str(path))
    return written


def run_stage_bc(
    args: argparse.Namespace, stage_a: dict[str, Any], device: Any, torch: Any,
) -> dict[str, Any]:
    from dataclasses import replace
    from src.data.personalization_diagnostics import generate_covered_personal_corpus
    from src.evaluation.personal_response_metrics import oracle_gap, personal_response_metrics
    from src.training.train_meta_personalization import MetaLossWeights
    train, validation, splits, metadata = (
        stage_a["train"], stage_a["validation"], stage_a["splits"], stage_a["metadata"]
    )
    t3 = splits["T3_unseen_person_profile"]
    phase_summary = json.loads((args.phase4b_dir / "summary.json").read_text(encoding="utf-8"))
    variants = {
        "P2-META-base": MetaLossWeights(),
        "P2-META-amplitude": MetaLossWeights(amplitude=1.0),
        "P2-META-personal-gain": MetaLossWeights(personal_gain=1.0),
        "P2-META-amplitude-gain": MetaLossWeights(amplitude=1.0, personal_gain=1.0),
    }
    trained: dict[str, Any] = {}
    training_metadata: dict[str, Any] = {}
    for name, weights in variants.items():
        print(f"TRAIN {name} - {SYNTHETIC_LABEL}", flush=True)
        model, info = train_or_load_meta(
            name, args, train, validation, metadata, weights, device, torch
        )
        trained[name], training_metadata[name] = model, info
        print(
            f"DONE {name} val={info['best_validation_Global_MPJPE']:.6f} "
            f"seconds={info['training_time_seconds']:.1f}", flush=True,
        )
    validation_selection: dict[str, dict[str, Any]] = {}
    for name, model in trained.items():
        validation_output = predict_meta(model, validation, 10, args, device, torch)
        validation_metrics, _, _ = personal_response_metrics(
            validation_output["future"], validation_output["natural"],
            validation_output["log_std"], validation_output["effect_log_std"],
            validation.split, validation_output["source"],
        )
        validation_selection[name] = {
            "Action_Effect_Error": validation_metrics["Action_Effect_Error"],
            "Sensitivity_MAE": validation_metrics["Sensitivity_MAE"],
            "Global_MPJPE": validation_metrics["Global_MPJPE"],
        }
        training_metadata[name]["validation_selection_metrics"] = validation_selection[name]
    selected_name = min(
        validation_selection,
        key=lambda name: (
            validation_selection[name]["Action_Effect_Error"],
            validation_selection[name]["Sensitivity_MAE"],
        ),
    )

    print(f"TRAIN P3-response-oracle - {SYNTHETIC_LABEL}", flush=True)
    response_oracle, response_oracle_training = train_or_load_response_oracle(
        args, train, validation, metadata, device, torch
    )
    oracle_output = predict_response_oracle(response_oracle, t3, args, device, torch)
    oracle_metrics, _, _ = personal_response_metrics(
        oracle_output["future"], oracle_output["natural"], oracle_output["log_std"],
        oracle_output["effect_log_std"], t3.split, oracle_output["source"],
    )

    metric_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    meta_rows: list[dict[str, Any]] = []
    for baseline in ("P0", "P2"):
        label = "P0" if baseline == "P0" else "Phase4B-P2"
        for k in (0, 1, 3, 5, 10):
            metrics = phase_summary["T3_unseen_person_profile"][baseline][str(k)]
            metric_lookup[(label, k)] = metrics
            for metric, value in metrics.items():
                meta_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "split": t3.split_label, "person": "ALL", "profile": "ALL",
                    "K": k, "model": label, "metric": metric, "value": value,
                })
    meta_predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for name, model in trained.items():
        for k in (0, 1, 3, 5, 10):
            output = predict_meta(model, t3, k, args, device, torch)
            metrics, _, _ = personal_response_metrics(
                output["future"], output["natural"], output["log_std"],
                output["effect_log_std"], t3.split, output["source"],
            )
            metric_lookup[(name, k)] = metrics
            meta_predictions[(name, k)] = output
            for metric, value in metrics.items():
                meta_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "split": t3.split_label, "person": "ALL", "profile": "ALL",
                    "K": k, "model": name, "metric": metric, "value": value,
                })
    for k in (0, 1, 3, 5, 10):
        metric_lookup[("P3-response-oracle", k)] = oracle_metrics
        for metric, value in oracle_metrics.items():
            meta_rows.append({
                "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                "split": t3.split_label, "person": "ALL", "profile": "ALL",
                "K": k, "model": "P3-response-oracle", "metric": metric,
                "value": value,
            })

    # Response-envelope expansion is a separate training experiment.
    covered_full = generate_covered_personal_corpus(seed=args.seed + 50_000)
    covered_train = replace(
        covered_full,
        query_indices=np.flatnonzero(np.isin(covered_full.order_indices, range(10, 20))),
        split_label="response_covered_train",
    )
    covered_validation = replace(
        covered_full,
        query_indices=np.flatnonzero(np.isin(covered_full.order_indices, range(20, 23))),
        split_label="response_covered_validation",
    )
    covered_name = "P2-META-amplitude-response-covered"
    print(f"TRAIN {covered_name} - {SYNTHETIC_LABEL}", flush=True)
    covered_model, covered_training = train_or_load_meta(
        covered_name, args, covered_train, covered_validation, metadata,
        MetaLossWeights(amplitude=1.0), device, torch,
    )
    coverage_rows = []
    for model_name, model in (
        ("original-profile-training", trained["P2-META-amplitude"]),
        ("response-covered-training", covered_model),
    ):
        for k in (0, 1, 3, 5, 10):
            output = predict_meta(model, t3, k, args, device, torch)
            metrics, _, _ = personal_response_metrics(
                output["future"], output["natural"], output["log_std"],
                output["effect_log_std"], t3.split, output["source"],
            )
            for metric in (
                "Global_MPJPE", "Action_Effect_Error", "Sensitivity_MAE",
                "Action_Sensitivity", "Human_Response_Ranking_Accuracy",
            ):
                coverage_rows.append({
                    "synthetic_interaction": SYNTHETIC_LABEL, "seed": args.seed,
                    "split": t3.split_label, "person": "ALL", "profile": "ALL",
                    "K": k, "model": model_name, "metric": metric,
                    "value": metrics[metric],
                    "test_profile_ids_used_in_training": False,
                })

    conditioning_rows, interpolation_rows = [], []
    old_sensitivity, old_interpolation, old_recovery = old_p2_conditioning_audit(
        args, t3, metadata, device, torch
    )
    conditioning_rows.extend(old_sensitivity)
    conditioning_rows.append({
        "synthetic_interaction": SYNTHETIC_LABEL, "model": "Phase4B-P2", **old_recovery
    })
    interpolation_rows.extend(old_interpolation)
    recovery_by_model = {"Phase4B-P2": old_recovery}
    for name, model in trained.items():
        sensitivity, interpolation, recovery = meta_conditioning_audit(
            name, model, t3, args, device, torch
        )
        conditioning_rows.extend(sensitivity)
        conditioning_rows.append({
            "synthetic_interaction": SYNTHETIC_LABEL, "model": name, **recovery
        })
        interpolation_rows.extend(interpolation)
        recovery_by_model[name] = recovery
    write_csv(args.output_dir / "conditioning_sensitivity.csv", conditioning_rows)
    write_csv(args.output_dir / "latent_interpolation.csv", interpolation_rows)

    calibration_rows, calibration_summary = uncertainty_calibration_audit(
        args, train, validation, t3, metadata, device, torch
    )
    write_csv(args.output_dir / "uncertainty_calibration.csv", calibration_rows)
    write_csv(args.output_dir / "meta_fewshot.csv", meta_rows)
    write_csv(args.output_dir / "profile_coverage_experiment.csv", coverage_rows)

    selected_metrics = {
        str(k): metric_lookup[(selected_name, k)] for k in (0, 1, 3, 5, 10)
    }
    p0 = metric_lookup[("P0", 0)]
    oracle_gaps = {
        metric: oracle_gap(
            float(p0[metric]), float(selected_metrics["10"][metric]),
            float(oracle_metrics[metric]),
        )
        for metric in ("Global_MPJPE", "Action_Effect_Error", "Sensitivity_MAE")
    }
    k_values = np.asarray((0, 1, 3, 5, 10), dtype=np.float64)
    effect_values = np.asarray([selected_metrics[str(k)]["Action_Effect_Error"] for k in k_values.astype(int)])
    sensitivity_values = np.asarray([selected_metrics[str(k)]["Sensitivity_MAE"] for k in k_values.astype(int)])

    def stable_improvement(values: np.ndarray) -> bool:
        descending_steps = int(np.sum(np.diff(values) <= 0.0))
        return bool(values[-1] < values[0] and np.polyfit(k_values, values, 1)[0] < 0.0 and descending_steps >= 3)

    def per_sample_response_errors(output: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        source = output["source"]
        target = t3.split.future_by_action[source]
        natural = t3.split.natural_future[source]
        predicted_effect = output["future"] - output["natural"][:, None]
        target_effect = target - natural[:, None]
        nonkeep = t3.split.candidate_actions[source] != 0
        branch_error = np.linalg.norm(predicted_effect - target_effect, axis=-1).mean(axis=(-1, -2))
        effect_error = (branch_error * nonkeep).sum(axis=1) / nonkeep.sum(axis=1)
        predicted_sensitivity = (
            np.linalg.norm(predicted_effect, axis=-1).mean(axis=(-1, -2)) * nonkeep
        ).sum(axis=1) / nonkeep.sum(axis=1)
        target_sensitivity = (
            np.linalg.norm(target_effect, axis=-1).mean(axis=(-1, -2)) * nonkeep
        ).sum(axis=1) / nonkeep.sum(axis=1)
        return effect_error, np.abs(predicted_sensitivity - target_sensitivity)

    def paired_bootstrap(improvement: np.ndarray) -> dict[str, float]:
        values = np.asarray(improvement, dtype=np.float64)
        rng = np.random.default_rng(args.seed + 45_500)
        indices = rng.integers(0, len(values), size=(5000, len(values)))
        means = values[indices].mean(axis=1)
        return {
            "mean_improvement": float(values.mean()),
            "ci95_lower": float(np.quantile(means, 0.025)),
            "ci95_upper": float(np.quantile(means, 0.975)),
            "sample_count": int(len(values)),
        }

    paired_evidence_by_variant = {}
    for name in variants:
        effect_k0, sensitivity_k0 = per_sample_response_errors(
            meta_predictions[(name, 0)]
        )
        effect_k10, sensitivity_k10 = per_sample_response_errors(
            meta_predictions[(name, 10)]
        )
        paired_evidence_by_variant[name] = {
            "Action_Effect_Error_K0_minus_K10": paired_bootstrap(
                effect_k0 - effect_k10
            ),
            "Sensitivity_MAE_K0_minus_K10": paired_bootstrap(
                sensitivity_k0 - sensitivity_k10
            ),
        }
    paired_evidence = paired_evidence_by_variant[selected_name]
    credible_k_improvement = bool(
        paired_evidence["Action_Effect_Error_K0_minus_K10"]["ci95_lower"] > 0.0
        or paired_evidence["Sensitivity_MAE_K0_minus_K10"]["ci95_lower"] > 0.0
    )

    selected_recovery = recovery_by_model[selected_name]["person_effect_recovery_ratio"]
    selected_interpolation = [
        item for item in interpolation_rows
        if item["model"] == selected_name and item["action"] == 4
    ]
    amplitudes = np.asarray([item["predicted_response_amplitude"] for item in selected_interpolation])
    interpolation_effective = bool(
        (np.all(np.diff(amplitudes) >= 0) or np.all(np.diff(amplitudes) <= 0))
        and abs(float(amplitudes[-1] - amplitudes[0]))
        / max(float(np.mean(np.abs(amplitudes))), 1e-12) > 0.10
    )
    target_coverage = {"Coverage_50": 0.5, "Coverage_80": 0.8, "Coverage_90": 0.9}
    calibration_reasonable = True
    for component in ("root", "action_effect"):
        after = calibration_summary[component]["after"]
        calibration_reasonable &= all(
            abs(float(after[name]) - expected) <= 0.10
            for name, expected in target_coverage.items()
        )
    new_p3_beats_generic = bool(
            oracle_metrics["Global_MPJPE"] < p0["Global_MPJPE"]
            and oracle_metrics["Action_Effect_Error"] < p0["Action_Effect_Error"]
    )
    strict_upper_bound = bool(
        new_p3_beats_generic
        and oracle_metrics["Global_MPJPE"] <= selected_metrics["10"]["Global_MPJPE"]
        and oracle_metrics["Action_Effect_Error"] <= selected_metrics["10"]["Action_Effect_Error"]
    )
    criteria = {
        "new_p3_beats_generic": new_p3_beats_generic,
        "new_p3_is_strict_upper_bound": strict_upper_bound,
        "oracle_condition_satisfied": new_p3_beats_generic,
        "stable_K_trend_effect_or_sensitivity": bool(
            credible_k_improvement
            and (stable_improvement(effect_values) or stable_improvement(sensitivity_values))
        ),
        "person_effect_recovery_clearly_above_phase4b": bool(
            selected_recovery > max(0.05, old_recovery["person_effect_recovery_ratio"] * 2.0)
        ),
        "latent_interpolation_effective": interpolation_effective,
        "human_response_metrics_better_than_P0": bool(
            selected_metrics["10"]["Action_Effect_Error"] < p0["Action_Effect_Error"]
            and selected_metrics["10"]["Sensitivity_MAE"] < p0["Sensitivity_MAE"]
        ),
        "uncertainty_calibration_reasonable": bool(calibration_reasonable),
    }
    required_for_phase4c = (
        "oracle_condition_satisfied",
        "stable_K_trend_effect_or_sensitivity",
        "person_effect_recovery_clearly_above_phase4b",
        "latent_interpolation_effective",
        "human_response_metrics_better_than_P0",
        "uncertainty_calibration_reasonable",
    )
    criteria["ready_for_phase4c"] = bool(
        all(criteria[name] for name in required_for_phase4c)
    )
    figures = make_stage_c_figures(
        args.output_dir, metric_lookup, conditioning_rows,
        interpolation_rows, calibration_rows,
    )
    summary = {
        "label": SYNTHETIC_LABEL, "seed": args.seed,
        "stage": "C_seed42", "five_seed_started": False,
        "selected_variant_by_validation_only": selected_name,
        "validation_ablation_selection": validation_selection,
        "training": {
            **training_metadata,
            "P3-response-oracle": response_oracle_training,
            covered_name: covered_training,
        },
        "selected_variant_metrics": selected_metrics,
        "P0": p0,
        "new_P3_response_oracle": oracle_metrics,
        "new_oracle_gap": oracle_gaps,
        "paired_K0_K10_evidence": paired_evidence,
        "paired_K0_K10_evidence_by_variant": paired_evidence_by_variant,
        "conditioning_recovery": recovery_by_model,
        "uncertainty_calibration": calibration_summary,
        "success_criteria": criteria,
        "figures": figures,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"Stage C selected={selected_name} "
        f"Effect K0={selected_metrics['0']['Action_Effect_Error']:.6f} "
        f"K10={selected_metrics['10']['Action_Effect_Error']:.6f} "
        f"recovery={selected_recovery:.4f} ready_phase4c={criteria['ready_for_phase4c']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(torch, args.seed)
    device = torch.device(args.device)
    print(SYNTHETIC_LABEL, flush=True)
    stage_a = run_stage_a(args, device, torch)
    if args.stage == "all":
        if not stage_a["diagnostic"]["stage_b_authorized_by_diagnostic"]:
            print("Stage B stopped: O-EFFECT did not improve over P0.", flush=True)
            return
        run_stage_bc(args, stage_a, device, torch)


if __name__ == "__main__":
    main()
