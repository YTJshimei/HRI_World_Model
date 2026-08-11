"""Train/evaluate the synthetic-interaction Phase 4A action-conditioned baselines."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from dataclasses import asdict
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
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--validation-size", type=int, default=180)
    parser.add_argument("--test-size", type=int, default=180)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--occlusion-rate", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results_dev" / "phase4a",
    )
    return parser.parse_args()


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, args: argparse.Namespace) -> Any:
    from src.models.action_conditioned_world_models import (
        ActionAgnosticHumanModel,
        ActionConditionedLSTM,
        ActionConditionedResidualModel,
        ActionConditionedRootPoseModel,
    )

    if name == "W0":
        return ActionAgnosticHumanModel(future_frames=args.future_frames)
    if name == "W1":
        return ActionConditionedLSTM(future_frames=args.future_frames)
    if name == "W2":
        return ActionConditionedRootPoseModel(
            history_frames=args.history_frames, future_frames=args.future_frames
        )
    if name == "W3":
        return ActionConditionedResidualModel(future_frames=args.future_frames)
    raise ValueError(name)


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_or_resume(
    name: str,
    args: argparse.Namespace,
    splits: Any,
    weights: Any,
    torch: Any,
    DataLoader: Any,
    device: Any,
) -> dict[str, Any]:
    from src.data.synthetic_interaction import as_interaction_tensor_dataset
    from src.training.train_interaction import train_interaction_model

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{name.lower()}_best.pt"
    metadata_path = checkpoint_dir / f"{name.lower()}_training.json"
    signature = {
        "synthetic_interaction": True,
        "synthetic_interaction_simulator_version": 2,
        "model": name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "noise_std": args.noise_std,
        "occlusion_rate": args.occlusion_rate,
        "learning_rate": args.learning_rate,
        "loss_weights": asdict(weights),
    }
    if checkpoint.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            print(f"CACHE {name}", flush=True)
            return cached
    print(f"TRAIN {name} — SYNTHETIC INTERACTION ONLY", flush=True)
    set_seed(torch, args.seed)
    model = build_model(name, args)
    train_loader = DataLoader(
        as_interaction_tensor_dataset(splits.train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        as_interaction_tensor_dataset(splits.val), batch_size=args.batch_size
    )
    result = train_interaction_model(
        model, train_loader, validation_loader, device, args.epochs, checkpoint,
        args.learning_rate, weights, verbose=False,
    )
    metadata = {
        "signature": signature,
        "checkpoint": str(checkpoint),
        "best_epoch": result.best_epoch,
        "best_validation_Global_MPJPE": result.best_validation_global_mpjpe,
        "training_time_seconds": result.training_time_seconds,
        "parameters": parameter_count(model),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"DONE {name} val={result.best_validation_global_mpjpe:.6f} "
        f"seconds={result.training_time_seconds:.1f}", flush=True,
    )
    model.to("cpu")
    del model, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def load_model(name: str, metadata: dict[str, Any], args: argparse.Namespace, torch: Any) -> Any:
    model = build_model(name, args)
    checkpoint = torch.load(metadata["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict(
    model: Any,
    loader: Any,
    device: Any,
    torch: Any,
    action_conditioning: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    futures, naturals = [], []
    model.to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            history, _, _, robot, actions, confidence, visibility, profiles, *_ = (
                value.to(device) for value in batch
            )
            output = model(
                history, robot, actions, confidence, visibility, profiles,
                action_conditioning=action_conditioning,
            )
            futures.append(output.future_by_action.cpu())
            naturals.append(output.natural_future.cpu())
    model.to("cpu")
    return torch.cat(futures).numpy(), torch.cat(naturals).numpy()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def per_sample_global_mpjpe(
    prediction: np.ndarray, target: np.ndarray, action_mask: np.ndarray
) -> np.ndarray:
    errors = np.linalg.norm(prediction - target, axis=-1).mean(axis=(-1, -2))
    selected = action_mask.astype(np.float64)
    denominator = selected.sum(axis=1)
    result = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = (errors[valid] * selected[valid]).sum(axis=1) / denominator[valid]
    return result


def paired_bootstrap_improvement(
    baseline_error: np.ndarray,
    model_error: np.ndarray,
    seed: int,
    repetitions: int = 5000,
) -> dict[str, float]:
    valid = np.isfinite(baseline_error) & np.isfinite(model_error)
    difference = baseline_error[valid] - model_error[valid]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        difference, size=(repetitions, len(difference)), replace=True
    ).mean(axis=1)
    mean = float(difference.mean())
    return {
        "mean_mpjpe_improvement_m": mean,
        "mean_relative_improvement": float(mean / baseline_error[valid].mean()),
        "bootstrap_95_ci_low": float(np.percentile(sampled, 2.5)),
        "bootstrap_95_ci_high": float(np.percentile(sampled, 97.5)),
        "bootstrap_probability_improvement": float(np.mean(sampled > 0.0)),
        "bootstrap_repetitions": repetitions,
    }


def main() -> int:
    args = parse_args()
    if args.seed != 42 or args.history_frames != 20 or args.future_frames != 10:
        raise SystemExit("Phase 4A first run is fixed to seed=42, history=20, future=10")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("Error: existing PyTorch is required; no dependency is installed.", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but unavailable.", file=sys.stderr)
        return 2

    from src.data.robot_action_schema import PHASE4A_ACTIONS
    from src.data.synthetic_interaction import (
        PROFILE_BY_ID,
        as_interaction_tensor_dataset,
        create_interaction_splits,
        simulate_interaction_future,
    )
    from src.evaluation.interaction_metrics import (
        counterfactual_ranking_per_sample,
        interaction_metrics,
    )
    from src.evaluation.interaction_plots import (
        plot_action_effect_vectors,
        plot_counterfactual_roots,
        plot_human_robot_distance,
        plot_profile_responses,
    )
    from src.training.train_interaction import InteractionLossWeights

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_seed(torch, args.seed)
    print("Generating SYNTHETIC INTERACTION counterfactual dataset", flush=True)
    splits = create_interaction_splits(
        args.train_size, args.validation_size, args.test_size, args.seed,
        history_frames=args.history_frames, future_frames=args.future_frames,
        sample_rate_hz=args.sample_rate, noise_std=args.noise_std,
        occlusion_rate=args.occlusion_rate,
    )
    # The effect weight is fixed before validation/test to prevent action collapse.
    weights = InteractionLossWeights(action_effect=5.0)
    device = torch.device(args.device)
    training = {
        name: train_or_resume(
            name, args, splits, weights, torch, DataLoader, device
        )
        for name in ("W0", "W1", "W2", "W3")
    }
    best_model_name = min(
        training,
        key=lambda name: training[name]["best_validation_Global_MPJPE"],
    )
    print(
        f"Validation-selected model: {best_model_name}; test access starts now",
        flush=True,
    )

    split_map = {
        "seen_person_seen_context": splits.test_seen_person_seen_context,
        "unseen_interaction_state": splits.test_unseen_interaction_state,
        "unseen_person_profile": splits.test_unseen_person_profile,
        "unseen_action_context_combination": splits.test_unseen_action_context,
    }
    overall, by_action, by_person, ranking_rows = {}, [], [], []
    prediction_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for model_name, metadata in training.items():
        model = load_model(model_name, metadata, args, torch)
        overall[model_name] = {}
        for split_name, split in split_map.items():
            loader = DataLoader(
                as_interaction_tensor_dataset(split), batch_size=args.batch_size
            )
            prediction, natural = predict(model, loader, device, torch)
            prediction_cache[(model_name, split_name)] = (prediction, natural)
            evaluation_mask = None
            if split_name == "unseen_action_context_combination":
                evaluation_mask = ~split.action_supervision_mask
            overall[model_name][split_name] = interaction_metrics(
                prediction, natural, split, args.sample_rate, evaluation_mask
            )
            base_mask = (
                np.ones(split.candidate_actions.shape, dtype=bool)
                if evaluation_mask is None else evaluation_mask
            )
            for action_index, action in enumerate(PHASE4A_ACTIONS):
                mask = base_mask & (split.candidate_actions == int(action))
                if not mask.any():
                    continue
                by_action.append({
                    "synthetic_interaction": True,
                    "model": model_name,
                    "split": split_name,
                    "action_id": int(action),
                    "action": action.name,
                    **interaction_metrics(
                        prediction, natural, split, args.sample_rate, mask
                    ),
                })
            for profile_id in np.unique(split.person_profile_id):
                sample_selected = split.person_profile_id == profile_id
                mask = base_mask & sample_selected[:, None]
                if not mask.any():
                    continue
                by_person.append({
                    "synthetic_interaction": True,
                    "model": model_name,
                    "split": split_name,
                    "person_profile_id": int(profile_id),
                    "person_profile": PROFILE_BY_ID[int(profile_id)].name,
                    **interaction_metrics(
                        prediction, natural, split, args.sample_rate, mask
                    ),
                })
            ranking = counterfactual_ranking_per_sample(
                prediction, split, evaluation_mask
            )
            for index, score in enumerate(ranking):
                if np.isfinite(score):
                    ranking_rows.append({
                        "synthetic_interaction": True,
                        "model": model_name,
                        "split": split_name,
                        "initial_state_id": str(split.initial_state_id[index]),
                        "person_profile_id": int(split.person_profile_id[index]),
                        "ranking_accuracy": float(score),
                    })
        del model

    paired_improvement_vs_w0 = {}
    for split_name, split in split_map.items():
        action_mask = (
            ~split.action_supervision_mask
            if split_name == "unseen_action_context_combination"
            else np.ones(split.candidate_actions.shape, dtype=bool)
        )
        baseline_error = per_sample_global_mpjpe(
            prediction_cache[("W0", split_name)][0], split.future_by_action, action_mask
        )
        paired_improvement_vs_w0[split_name] = {}
        for model_name in ("W1", "W2", "W3"):
            model_error = per_sample_global_mpjpe(
                prediction_cache[(model_name, split_name)][0],
                split.future_by_action,
                action_mask,
            )
            paired_improvement_vs_w0[split_name][model_name] = paired_bootstrap_improvement(
                baseline_error, model_error, args.seed + len(model_name)
            )

    seen = splits.test_seen_person_seen_context
    seen_loader = DataLoader(
        as_interaction_tensor_dataset(seen), batch_size=args.batch_size
    )
    conditioning_ablation = {}
    for model_name in ("W1", "W2", "W3"):
        model = load_model(model_name, training[model_name], args, torch)
        disabled_prediction, disabled_natural = predict(
            model, seen_loader, device, torch, action_conditioning=False
        )
        conditioning_ablation[model_name] = {
            "enabled": overall[model_name]["seen_person_seen_context"],
            "disabled": interaction_metrics(
                disabled_prediction, disabled_natural, seen, args.sample_rate
            ),
        }
        del model

    # Explicit run-time sanity audit on the validation-selected model.
    selected_model = load_model(best_model_name, training[best_model_name], args, torch).to(device).eval()
    sanity_loader = DataLoader(as_interaction_tensor_dataset(seen), batch_size=min(16, len(seen)))
    batch = next(iter(sanity_loader))
    history, _, _, robot, actions, confidence, visibility, profiles, *_ = (
        value.to(device) for value in batch
    )
    permutation = torch.tensor([4, 2, 0, 3, 1], device=device)
    with torch.inference_mode():
        enabled = selected_model(history, robot, actions, confidence, visibility, profiles, True)
        disabled = selected_model(history, robot, actions, confidence, visibility, profiles, False)
        permuted = selected_model(
            history, robot, actions[:, permutation], confidence, visibility, profiles, True
        )
    inverse = torch.argsort(permutation)
    delay_errors = []
    for sample in range(len(seen)):
        profile = PROFILE_BY_ID[int(seen.person_profile_id[sample])]
        delay = int(np.ceil(profile.response_delay * args.sample_rate - 1e-9))
        delay_errors.append(float(np.max(np.abs(seen.action_effect_by_action[sample, :, :delay]))))
    profile_first = simulate_interaction_future(
        seen.human_history[0], seen.natural_future[0], seen.robot_history[0],
        PHASE4A_ACTIONS[4], PROFILE_BY_ID[0], args.sample_rate,
    )
    profile_second = simulate_interaction_future(
        seen.human_history[0], seen.natural_future[0], seen.robot_history[0],
        PHASE4A_ACTIONS[4], PROFILE_BY_ID[4], args.sample_rate,
    )
    sanity = {
        "conditioning_off_max_action_difference": float(
            (disabled.future_by_action - disabled.future_by_action[:, :1]).abs().max().item()
        ),
        "conditioning_on_max_action_difference": float(
            (enabled.future_by_action - enabled.future_by_action[:, :1]).abs().max().item()
        ),
        "action_permutation_restoration_max_error": float(
            (enabled.future_by_action - permuted.future_by_action[:, inverse]).abs().max().item()
        ),
        "A0_keep_simulator_natural_max_error": float(
            np.max(np.abs(seen.future_by_action[:, 0] - seen.natural_future))
        ),
        "response_delay_pre_effect_max_error": max(delay_errors),
        "same_state_profile_response_max_difference": float(
            np.max(np.abs(profile_first.action_effect - profile_second.action_effect))
        ),
    }
    selected_model.to("cpu")

    selected_prediction, _ = prediction_cache[
        (best_model_name, "seen_person_seen_context")
    ]
    plot_counterfactual_roots(
        seen, 0, selected_prediction, best_model_name,
        figures_dir / "synthetic_counterfactual_root_trajectories.png",
    )
    plot_action_effect_vectors(
        seen, 0, figures_dir / "synthetic_action_effect_vectors.png"
    )
    plot_human_robot_distance(
        seen, 0, selected_prediction, args.sample_rate,
        figures_dir / "synthetic_human_robot_distance.png",
    )
    plot_profile_responses(
        seen, 0, 4, args.sample_rate,
        figures_dir / "synthetic_virtual_person_responses.png",
    )

    write_csv(args.output_dir / "by_action.csv", by_action)
    write_csv(args.output_dir / "by_person.csv", by_person)
    write_csv(args.output_dir / "counterfactual_ranking.csv", ranking_rows)
    payload = {
        "artifact_label": "SYNTHETIC INTERACTION — NOT REAL HUMAN DATA",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_actions": [action.name for action in PHASE4A_ACTIONS],
        "loss_weights_fixed_before_test": asdict(weights),
        "validation_selected_model": best_model_name,
        "test_used_for_selection": False,
        "split_protocol": {
            "counterfactual_branches_are_atomic": True,
            "seen_person_seen_context": len(splits.test_seen_person_seen_context),
            "unseen_interaction_state": len(splits.test_unseen_interaction_state),
            "unseen_person_profile": len(splits.test_unseen_person_profile),
            "unseen_action_context_combination": len(splits.test_unseen_action_context),
        },
        "training": training,
        "metrics": overall,
        "paired_bootstrap_improvement_vs_W0": paired_improvement_vs_w0,
        "conditioning_ablation_seen_context": conditioning_ablation,
        "sanity": sanity,
    }
    result_path = args.output_dir / "phase4_baseline.json"
    result_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"SYNTHETIC INTERACTION results: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
