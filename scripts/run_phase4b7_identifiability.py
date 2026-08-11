"""Phase 4B.7 offline synthetic functional-response identifiability study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
STRATEGIES = (
    "Recent", "Random", "Naive Diverse", "Greedy Uncertainty",
    "Greedy Observability", "Oracle Informative",
)
K_REPORT = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b7")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--max-episodes", type=int, default=60)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def clean(value: Any) -> Any:
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field, "")) for field in fields})


def phase4b_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        stage="full", seed=args.seed, epochs=20, batch_size=32,
        history_frames=20, future_frames=10, sample_rate=10.0,
        learning_rate=1e-3, noise_std=0.005, occlusion_rate=0.10,
        persons_per_profile=2, interactions_per_person=30,
        benchmark_batch_size=32, benchmark_warmup=50,
        benchmark_repetitions=200, output_dir=args.phase4b6_dir.parent / "phase4b",
        device=args.device,
    )


def load_f2_prior(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from src.data.response_statistics import RESPONSE_STATISTIC_DIM
    from src.models.functional_response_decoder import FunctionalResponseWorldModel

    checkpoint = args.phase4b6_dir / "checkpoints" / "f2_original_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"frozen Phase 4B.6 checkpoint not found: {checkpoint}")
    model = FunctionalResponseWorldModel().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(state)
    with torch.inference_mode():
        estimate = model.estimator(
            torch.zeros(1, 10, RESPONSE_STATISTIC_DIM),
            torch.zeros(1, 10, dtype=torch.bool),
            torch.zeros(1, 10, 6, dtype=torch.bool),
        )
    return estimate.theta_mean[0].numpy(), estimate.theta_log_std[0].exp().numpy()


def profile_theta(profile_id: int) -> np.ndarray:
    from src.data.functional_response_state import functional_state_from_profile
    from src.data.synthetic_interaction import PROFILE_BY_ID
    return functional_state_from_profile(PROFILE_BY_ID[int(profile_id)]).astype(np.float64)


def context(corpus: Any, record: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = int(record.source_row)
    return (
        corpus.split.human_history[source], corpus.split.natural_future[source],
        corpus.split.robot_history[source],
    )


def calibrate_probe_tradeoff(
    validation: Any, prior_mean: np.ndarray, prior_std: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    from src.data.response_probe_schema import PROBE_CATALOG
    from src.evaluation.response_identifiability import (
        disturbance_score, information_score, response_jacobian,
        simulate_functional_probe,
    )
    records = validation.query_records()[:10]
    aggregate: dict[str, list[tuple[float, float]]] = {probe.probe_id: [] for probe in PROBE_CATALOG}
    for record in records:
        history, natural, robot = context(validation, record)
        for probe in PROBE_CATALOG:
            jacobian = response_jacobian(history, natural, robot, probe, prior_mean)
            information = information_score(prior_std, (jacobian,))
            simulation = simulate_functional_probe(history, natural, robot, probe, prior_mean)
            aggregate[probe.probe_id].append((information, disturbance_score(simulation)))
    info_values = [np.mean([item[0] for item in values]) for values in aggregate.values()]
    disturbance_values = [np.mean([item[1] for item in values]) for values in aggregate.values()]
    positive_disturbance = [value for value in disturbance_values if value > 1e-9]
    penalty_lambda = float(
        0.25 * np.median(info_values) / max(np.median(positive_disturbance), 1e-8)
    )
    rows = []
    for probe, information, disturbance in zip(PROBE_CATALOG, info_values, disturbance_values):
        rows.append({
            "synthetic_interaction": LABEL, "split": "validation",
            "probe": probe.probe_id, "InformationScore": information,
            "DisturbanceScore": disturbance,
            "ProbeScore": information - penalty_lambda * disturbance,
            "lambda_validation_selected": penalty_lambda,
        })
    return penalty_lambda, rows


def build_observability_outputs(
    validation: Any, prior_mean: np.ndarray, prior_std: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from src.data.functional_response_state import RESPONSE_STATE_NAMES
    from src.data.response_probe_schema import PROBE_CATALOG
    from src.evaluation.response_identifiability import (
        PROBE_OBSERVABLE_NAMES, local_observability_diagnostics,
        posterior_std, response_jacobian,
    )
    records = validation.query_records()[:20]
    jacobian_rows, matrix_rows, summary = [], [], {}
    for probe in PROBE_CATALOG:
        jacobians = []
        for record in records:
            history, natural, robot = context(validation, record)
            jacobians.append(response_jacobian(history, natural, robot, probe, prior_mean))
        stack = np.stack(jacobians)
        mean_abs = np.abs(stack).mean(axis=0)
        std_abs = np.abs(stack).std(axis=0)
        for observable_index, observable in enumerate(PROBE_OBSERVABLE_NAMES):
            for dimension, state_name in enumerate(RESPONSE_STATE_NAMES):
                jacobian_rows.append({
                    "synthetic_interaction": LABEL, "probe": probe.probe_id,
                    "observable": observable, "response_dimension": state_name,
                    "absolute_sensitivity_mean": mean_abs[observable_index, dimension],
                    "absolute_sensitivity_std": std_abs[observable_index, dimension],
                })
        representative = stack.mean(axis=0)
        after_std = posterior_std(prior_std, (representative,))
        state_spread = np.asarray((0.32, 0.35, 0.34, 0.22, 0.37, 0.72))
        for dimension, state_name in enumerate(RESPONSE_STATE_NAMES):
            magnitude = float(np.linalg.norm(representative[:, dimension]))
            separation = magnitude * state_spread[dimension]
            reduction = float(max(prior_std[dimension] - after_std[dimension], 0.0))
            matrix_rows.append({
                "synthetic_interaction": LABEL, "probe": probe.probe_id,
                "response_dimension": state_name,
                "empirical_response_separation": separation,
                "estimator_uncertainty_reduction": reduction,
                "local_response_jacobian_magnitude": magnitude,
                "combined_observability": separation + reduction + magnitude,
            })
        diagnostic = local_observability_diagnostics((representative,))
        summary[probe.probe_id] = {
            "rank": diagnostic.rank,
            "effective_rank": diagnostic.effective_rank,
            "condition_number": diagnostic.condition_number,
            "smallest_singular_value": diagnostic.smallest_singular_value,
            "singular_values": diagnostic.singular_values,
        }
    return matrix_rows, jacobian_rows, summary


def response_scalars(
    history: np.ndarray, natural: np.ndarray, robot: np.ndarray,
    theta: np.ndarray,
) -> dict[str, float]:
    from src.data.response_probe_schema import PROBE_BY_ID
    from src.evaluation.response_identifiability import simulate_functional_probe
    simulations = {
        name: simulate_functional_probe(history, natural, robot, PROBE_BY_ID[name], theta)
        for name in (
            "SPEED_DOWN_10", "SPEED_UP_10", "DISTANCE_PLUS_0_2",
            "DISTANCE_MINUS_0_2", "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
        )
    }
    speed = np.mean([
        abs(float(item.response_statistics[0]))
        for name, item in simulations.items() if name.startswith("SPEED")
    ])
    distance = np.mean([
        abs(float(item.response_statistics[1]))
        for name, item in simulations.items() if name.startswith("DISTANCE")
    ])
    lateral = np.mean([
        abs(float(item.response_statistics[2]))
        for name, item in simulations.items() if name.startswith("DISTANCE")
    ])
    turn = np.mean([
        abs(float(item.response_statistics[3]))
        for name, item in simulations.items() if name.startswith("TURN")
    ])
    effect = np.mean([
        float(np.linalg.norm(item.action_effect, axis=-1).mean())
        for item in simulations.values()
    ])
    return {"speed": speed, "distance": distance, "lateral": lateral, "turn": turn, "effect": effect}


def recovery_ratio(predicted: dict[int, list[dict[str, float]]], expected: dict[int, list[dict[str, float]]], key: str) -> float:
    profiles = sorted(expected)
    if len(profiles) < 2:
        return 0.0
    pred_mean = [np.mean([item[key] for item in predicted[profile]]) for profile in profiles]
    gt_mean = [np.mean([item[key] for item in expected[profile]]) for profile in profiles]
    pred_difference = float(np.mean(np.abs(np.subtract.outer(pred_mean, pred_mean))))
    gt_difference = float(np.mean(np.abs(np.subtract.outer(gt_mean, gt_mean))))
    return pred_difference / max(gt_difference, 1e-12)


def strategy_key(strategy: str) -> str:
    return {
        "Random": "random", "Naive Diverse": "naive_diverse",
        "Greedy Uncertainty": "greedy_uncertainty",
        "Greedy Observability": "greedy_observability",
    }[strategy]


def run_support_selection(
    corpus: Any, prior_mean: np.ndarray, prior_std: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from src.data.functional_response_state import RESPONSE_STATE_NAMES
    from src.data.personal_interaction_memory import PersonalInteractionMemory
    from src.data.response_probe_schema import PROBE_CATALOG, probe_state_mask
    from src.evaluation.response_identifiability import (
        FunctionalBelief, classic_probe_for_action, disturbance_score,
        functional_belief_update, information_score, local_observability_diagnostics,
        response_jacobian, select_oracle_probe, select_probe_without_oracle,
        simulate_functional_probe, uncertainty_specificity_score,
    )
    memory = PersonalInteractionMemory(corpus.records)
    query_records = corpus.query_records()[: args.max_episodes]
    sequential_rows: list[dict[str, Any]] = []
    specificity_rows: list[dict[str, Any]] = []
    episode_results: dict[tuple[str, int], list[dict[str, Any]]] = {
        (strategy, k): [] for strategy in STRATEGIES for k in K_REPORT
    }
    for episode, query in enumerate(query_records):
        theta_true = profile_theta(query.person_profile_id)
        available = list(memory.available_before(query))
        support_states = available[-10:]
        if len(support_states) < 10:
            raise RuntimeError("Phase 4B.7 requires ten strictly prior interactions")
        query_history, query_natural, query_robot = context(corpus, query)
        for strategy in STRATEGIES:
            belief = FunctionalBelief(prior_mean.copy(), prior_std.copy())
            cumulative_jacobians: list[np.ndarray] = []
            sequence: list[str] = []
            for step, support_record in enumerate(support_states, start=1):
                history, natural, robot = context(corpus, support_record)
                if strategy == "Recent":
                    probe = classic_probe_for_action(support_record.executed_action)
                elif strategy == "Oracle Informative":
                    probe = select_oracle_probe(
                        PROBE_CATALOG, belief, theta_true, history, natural, robot
                    )
                else:
                    probe = select_probe_without_oracle(
                        strategy_key(strategy), PROBE_CATALOG, belief,
                        history, natural, robot, cumulative_jacobians,
                        seed=args.seed + episode * 1009, step=step,
                    )
                observed = simulate_functional_probe(
                    history, natural, robot, probe, theta_true
                )
                predicted = simulate_functional_probe(
                    history, natural, robot, probe, belief.mean
                )
                jacobian = response_jacobian(
                    history, natural, robot, probe, belief.mean
                )
                before = belief
                belief = functional_belief_update(
                    belief, observed.response_statistics,
                    predicted.response_statistics, jacobian,
                )
                cumulative_jacobians.append(jacobian)
                sequence.append(probe.probe_id)
                information = information_score(before.std, (jacobian,))
                specificity = uncertainty_specificity_score(
                    before.std, belief.std, probe_state_mask(probe)
                )
                actual_reduction = float(
                    np.mean(np.abs(before.mean - theta_true))
                    - np.mean(np.abs(belief.mean - theta_true))
                )
                future_error_reduction: float | None = None
                if strategy in ("Greedy Uncertainty", "Greedy Observability"):
                    true_response = response_scalars(history, natural, robot, theta_true)
                    before_response = response_scalars(history, natural, robot, before.mean)
                    after_response = response_scalars(history, natural, robot, belief.mean)
                    response_keys = ("speed", "distance", "lateral", "turn", "effect")
                    before_error = np.mean([
                        abs(before_response[key] - true_response[key]) for key in response_keys
                    ])
                    after_error = np.mean([
                        abs(after_response[key] - true_response[key]) for key in response_keys
                    ])
                    future_error_reduction = float(before_error - after_error)
                diagnostic = local_observability_diagnostics(cumulative_jacobians)
                sequential_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "split": corpus.split_label, "episode": episode,
                    "person": query.person_instance_id,
                    "profile": int(query.person_profile_id), "strategy": strategy,
                    "step": step, "selected_probe": probe.probe_id,
                    "theta_MAE": float(np.mean(np.abs(belief.mean - theta_true))),
                    "mean_uncertainty": float(belief.std.mean()),
                    "InformationScore": information,
                    "actual_theta_error_reduction": actual_reduction,
                    "actual_future_response_error_reduction": future_error_reduction,
                    "DisturbanceScore": disturbance_score(observed),
                    "observability_rank": diagnostic.rank,
                    "effective_rank": diagnostic.effective_rank,
                    "condition_number": diagnostic.condition_number,
                    "smallest_singular_value": diagnostic.smallest_singular_value,
                    "selected_sequence": "|".join(sequence),
                })
                for dimension, name in enumerate(RESPONSE_STATE_NAMES):
                    specificity_rows.append({
                        "synthetic_interaction": LABEL, "seed": args.seed,
                        "episode": episode, "strategy": strategy, "step": step,
                        "probe": probe.probe_id, "response_dimension": name,
                        "observable_for_probe": bool(probe_state_mask(probe)[dimension]),
                        "uncertainty_before": before.std[dimension],
                        "uncertainty_after": belief.std[dimension],
                        "uncertainty_reduction": before.std[dimension] - belief.std[dimension],
                        "specificity_score": specificity["specificity_score"],
                    })
                if step in K_REPORT:
                    predicted_scalars = response_scalars(
                        query_history, query_natural, query_robot, belief.mean
                    )
                    expected_scalars = response_scalars(
                        query_history, query_natural, query_robot, theta_true
                    )
                    errors = np.abs(belief.mean - theta_true)
                    episode_results[(strategy, step)].append({
                        "episode": episode, "profile": int(query.person_profile_id),
                        "theta": belief.mean.copy(), "theta_true": theta_true.copy(),
                        "std": belief.std.copy(), "errors": errors,
                        "predicted_scalars": predicted_scalars,
                        "expected_scalars": expected_scalars,
                        "Action_Effect_Error": abs(predicted_scalars["effect"] - expected_scalars["effect"]),
                        "Sensitivity_MAE": float(np.mean([
                            abs(predicted_scalars[key] - expected_scalars[key])
                            for key in ("speed", "distance", "lateral", "turn")
                        ])),
                        "sequence": tuple(sequence),
                    })

    selection_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        summary[strategy] = {}
        for k in K_REPORT:
            items = episode_results[(strategy, k)]
            predicted_by_profile: dict[int, list[dict[str, float]]] = {}
            expected_by_profile: dict[int, list[dict[str, float]]] = {}
            for item in items:
                predicted_by_profile.setdefault(item["profile"], []).append(item["predicted_scalars"])
                expected_by_profile.setdefault(item["profile"], []).append(item["expected_scalars"])
            metrics = {
                "Functional_State_MAE": float(np.mean([item["errors"].mean() for item in items])),
                "Action_Effect_Error": float(np.mean([item["Action_Effect_Error"] for item in items])),
                "Sensitivity_MAE": float(np.mean([item["Sensitivity_MAE"] for item in items])),
                "Person_Effect_Recovery_Ratio": recovery_ratio(predicted_by_profile, expected_by_profile, "effect"),
                "Speed_Effect_Recovery_Ratio": recovery_ratio(predicted_by_profile, expected_by_profile, "speed"),
                "Distance_Effect_Recovery_Ratio": recovery_ratio(predicted_by_profile, expected_by_profile, "distance"),
                "Lateral_Effect_Recovery_Ratio": recovery_ratio(predicted_by_profile, expected_by_profile, "lateral"),
                "Turn_Effect_Recovery_Ratio": recovery_ratio(predicted_by_profile, expected_by_profile, "turn"),
                "Mean_Uncertainty": float(np.mean([item["std"].mean() for item in items])),
            }
            for dimension, name in enumerate(RESPONSE_STATE_NAMES):
                metrics[f"{name}_MAE"] = float(np.mean([item["errors"][dimension] for item in items]))
            summary[strategy][str(k)] = metrics
            for metric, value in metrics.items():
                selection_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "split": corpus.split_label, "strategy": strategy,
                    "K": k, "metric": metric, "value": value,
                })
    return selection_rows, sequential_rows, specificity_rows, summary


def speed_horizon_diagnostics(args: argparse.Namespace, prior_mean: np.ndarray, prior_std: np.ndarray) -> list[dict[str, Any]]:
    from src.data.response_probe_schema import PROBE_BY_ID
    from src.data.synthetic_interaction import generate_interaction_split
    from src.evaluation.response_identifiability import (
        FunctionalBelief, functional_belief_update, information_score,
        response_jacobian, simulate_functional_probe,
    )
    rows = []
    for horizon in (5, 10, 15, 20, 30):
        split = generate_interaction_split(
            12, args.seed + 4700 + horizon, f"speed_horizon_{horizon}",
            profile_ids=(5, 6), future_frames=horizon,
            noise_std=0.005, occlusion_rate=0.10,
        )
        for probe_name in (
            "SPEED_DOWN_5", "SPEED_DOWN_10", "SPEED_DOWN_15",
            "SPEED_UP_5", "SPEED_UP_10", "SPEED_UP_15",
        ):
            probe = PROBE_BY_ID[probe_name]
            sensitivities, signals, errors, information_values = [], [], [], []
            for index in range(len(split)):
                theta_true = profile_theta(int(split.person_profile_id[index]))
                history, natural, robot = (
                    split.human_history[index], split.natural_future[index], split.robot_history[index]
                )
                jacobian = response_jacobian(history, natural, robot, probe, prior_mean)
                observed = simulate_functional_probe(history, natural, robot, probe, theta_true)
                predicted = simulate_functional_probe(history, natural, robot, probe, prior_mean)
                updated = functional_belief_update(
                    FunctionalBelief(prior_mean, prior_std),
                    observed.response_statistics, predicted.response_statistics, jacobian,
                )
                sensitivities.append(np.linalg.norm(jacobian[:, 0]))
                signals.append(abs(float(observed.response_statistics[0])))
                errors.append(abs(updated.mean[0] - theta_true[0]))
                information_values.append(information_score(prior_std, (jacobian,)))
            rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed,
                "horizon_seconds": horizon / 10.0, "future_frames": horizon,
                "probe": probe_name,
                "speed_response_magnitude": float(np.mean(signals)),
                "speed_jacobian_magnitude": float(np.mean(sensitivities)),
                "speed_gain_MAE_after_one_probe": float(np.mean(errors)),
                "InformationScore": float(np.mean(information_values)),
            })
    return rows


def turn_diagnostics(validation: Any, prior_mean: np.ndarray, prior_std: np.ndarray) -> list[dict[str, Any]]:
    from src.data.response_probe_schema import PROBE_BY_ID
    from src.evaluation.response_identifiability import (
        PROBE_OBSERVABLE_NAMES, FunctionalBelief, functional_belief_update,
        information_score, response_jacobian, simulate_functional_probe,
    )
    rows = []
    names = (
        "KEEP", "SPEED_DOWN_10", "SPEED_UP_10", "DISTANCE_PLUS_0_2",
        "DISTANCE_MINUS_0_2", "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
    )
    for name in names:
        probe = PROBE_BY_ID[name]
        per_observable: dict[str, list[float]] = {item: [] for item in PROBE_OBSERVABLE_NAMES}
        errors, information_values = [], []
        for record in validation.query_records()[:20]:
            history, natural, robot = context(validation, record)
            theta_true = profile_theta(record.person_profile_id)
            jacobian = response_jacobian(history, natural, robot, probe, prior_mean)
            observed = simulate_functional_probe(history, natural, robot, probe, theta_true)
            predicted = simulate_functional_probe(history, natural, robot, probe, prior_mean)
            updated = functional_belief_update(
                FunctionalBelief(prior_mean, prior_std),
                observed.response_statistics, predicted.response_statistics, jacobian,
            )
            for index, observable in enumerate(PROBE_OBSERVABLE_NAMES):
                per_observable[observable].append(abs(float(jacobian[index, 4])))
            errors.append(abs(updated.mean[4] - theta_true[4]))
            information_values.append(information_score(prior_std, (jacobian,)))
        for observable, values in per_observable.items():
            rows.append({
                "synthetic_interaction": LABEL, "probe": name,
                "observable": observable,
                "turn_jacobian_magnitude": float(np.mean(values)),
                "turn_gain_MAE_after_one_probe": float(np.mean(errors)),
                "InformationScore": float(np.mean(information_values)),
            })
    return rows


def safe_correlation(x: list[float], y: list[float]) -> float | None:
    first, second = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def aggregate_multiseed(output_dir: Path) -> None:
    seeds = (42, 123, 3407, 2026, 7777)
    paths = {
        42: output_dir / "summary.json",
        **{
            seed: output_dir / "multiseed_runs" / f"seed{seed}" / "summary.json"
            for seed in seeds[1:]
        },
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing multiseed summaries: {missing}")
    summaries = {
        seed: json.loads(path.read_text(encoding="utf-8"))
        for seed, path in paths.items()
    }
    metrics = (
        "Functional_State_MAE", "speed_response_gain_MAE",
        "distance_response_gain_MAE", "lateral_response_gain_MAE",
        "response_delay_MAE", "turn_response_gain_MAE",
        "adaptation_response_gain_MAE", "Action_Effect_Error",
        "Sensitivity_MAE", "Person_Effect_Recovery_Ratio",
        "Speed_Effect_Recovery_Ratio", "Distance_Effect_Recovery_Ratio",
        "Lateral_Effect_Recovery_Ratio", "Turn_Effect_Recovery_Ratio",
    )
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for strategy in STRATEGIES:
        aggregate[strategy] = {}
        for k in K_REPORT:
            aggregate[strategy][str(k)] = {}
            for metric in metrics:
                values = np.asarray([
                    summaries[seed]["support_selection"][strategy][str(k)][metric]
                    for seed in seeds
                ], dtype=np.float64)
                for seed, value in zip(seeds, values):
                    rows.append({
                        "synthetic_interaction": LABEL, "seed": seed,
                        "strategy": strategy, "K": k, "metric": metric,
                        "value": value, "statistic": "seed_value",
                    })
                mean, std = float(values.mean()), float(values.std(ddof=1))
                rows.extend((
                    {
                        "synthetic_interaction": LABEL, "seed": "ALL",
                        "strategy": strategy, "K": k, "metric": metric,
                        "value": mean, "statistic": "mean",
                    },
                    {
                        "synthetic_interaction": LABEL, "seed": "ALL",
                        "strategy": strategy, "K": k, "metric": metric,
                        "value": std, "statistic": "std",
                    },
                ))
                aggregate[strategy][str(k)][metric] = {"mean": mean, "std": std}
    write_csv(output_dir / "multiseed.csv", rows)
    root = summaries[42]
    all_gates = all(
        bool(summary["success_criteria"]["five_seed_gate_passed"])
        for summary in summaries.values()
    )
    root["five_seed_started"] = True
    root["five_seed_completed"] = True
    root["multiseed_seeds"] = list(seeds)
    root["all_seed_gates_passed"] = all_gates
    root["multiseed_summary"] = aggregate
    root["success_criteria"]["five_seed_gate_passed"] = all_gates
    root["success_criteria"]["ready_for_phase4c"] = all_gates
    write_json(output_dir / "summary.json", root)
    print(f"aggregated five seeds; all_gates_passed={all_gates}", flush=True)


def make_figures(
    output_dir: Path, matrix_rows: list[dict[str, Any]],
    observability_summary: dict[str, Any], probe_rows: list[dict[str, Any]],
    selection_summary: dict[str, Any], sequential_rows: list[dict[str, Any]],
    speed_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]],
    specificity_rows: list[dict[str, Any]],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.functional_response_state import RESPONSE_STATE_NAMES
    from src.data.response_probe_schema import PROBE_CATALOG

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = figure_dir / name
        plt.title(LABEL, fontsize=8)
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
        paths.append(str(path))

    probes = [probe.probe_id for probe in PROBE_CATALOG]
    heatmap = np.zeros((len(probes), len(RESPONSE_STATE_NAMES)))
    for row in matrix_rows:
        heatmap[probes.index(row["probe"]), RESPONSE_STATE_NAMES.index(row["response_dimension"])] = row["combined_observability"]
    plt.figure(figsize=(10, 6)); plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(label="combined local observability"); plt.yticks(range(len(probes)), probes, fontsize=6)
    plt.xticks(range(6), RESPONSE_STATE_NAMES, rotation=30, ha="right", fontsize=7)
    save("action_state_observability_heatmap.png")

    plt.figure(figsize=(9, 5))
    for probe in probes:
        values = observability_summary[probe]["singular_values"]
        plt.plot(range(1, 7), values, marker=".", alpha=0.65, label=probe)
    plt.yscale("symlog", linthresh=1e-4); plt.xlabel("singular-value index"); plt.ylabel("value")
    plt.legend(fontsize=5, ncol=2); save("singular_values_identifiability.png")

    plt.figure(figsize=(8, 5))
    for name in sorted(set(row["probe"] for row in speed_rows)):
        rows = [row for row in speed_rows if row["probe"] == name]
        plt.plot([row["horizon_seconds"] for row in rows], [row["InformationScore"] for row in rows], marker="o", label=name)
    plt.xlabel("observation horizon (s)"); plt.ylabel("InformationScore"); plt.legend(fontsize=6, ncol=2)
    save("speed_probe_information_vs_magnitude.png")

    shoulder_rows = [row for row in turn_rows if row["observable"] == "shoulder_yaw_response"]
    plt.figure(figsize=(9, 4)); plt.bar([row["probe"] for row in shoulder_rows], [row["turn_jacobian_magnitude"] for row in shoulder_rows])
    plt.xticks(rotation=30, ha="right"); plt.ylabel("turn Jacobian magnitude"); save("turn_probe_information.png")

    plt.figure(figsize=(7, 5))
    for row in probe_rows:
        plt.scatter(row["DisturbanceScore"], row["InformationScore"])
        plt.annotate(row["probe"], (row["DisturbanceScore"], row["InformationScore"]), fontsize=5)
    plt.xlabel("DisturbanceScore (synthetic proxy)"); plt.ylabel("InformationScore")
    save("information_disturbance_pareto.png")

    plt.figure(figsize=(8, 5))
    for strategy in STRATEGIES:
        plt.plot(K_REPORT, [selection_summary[strategy][str(k)]["Functional_State_MAE"] for k in K_REPORT], marker="o", label=strategy)
    plt.xlabel("K"); plt.ylabel("Functional State MAE"); plt.legend(fontsize=6)
    save("theta_mae_vs_k.png")

    plt.figure(figsize=(9, 4))
    values = [selection_summary[strategy]["3"]["Functional_State_MAE"] for strategy in STRATEGIES]
    plt.bar(STRATEGIES, values); plt.xticks(rotation=25, ha="right"); plt.ylabel("K=3 state MAE")
    save("support_strategy_comparison.png")

    plt.figure(figsize=(10, 4))
    sample = [row for row in sequential_rows if row["episode"] == 0 and row["strategy"] in ("Greedy Uncertainty", "Greedy Observability")]
    for strategy in ("Greedy Uncertainty", "Greedy Observability"):
        rows = [row for row in sample if row["strategy"] == strategy]
        plt.plot([row["step"] for row in rows], [row["theta_MAE"] for row in rows], marker="o", label=strategy)
        for row in rows:
            plt.annotate(row["selected_probe"], (row["step"], row["theta_MAE"]), fontsize=5, rotation=30)
    plt.xlabel("offline probe step"); plt.ylabel("theta MAE"); plt.legend(fontsize=7)
    save("selected_probe_sequence.png")

    plt.figure(figsize=(8, 5))
    greedy = [row for row in specificity_rows if row["strategy"] == "Greedy Uncertainty"]
    before = [np.mean([float(row["uncertainty_before"]) for row in greedy if row["response_dimension"] == name and row["step"] == 1]) for name in RESPONSE_STATE_NAMES]
    after = [np.mean([float(row["uncertainty_after"]) for row in greedy if row["response_dimension"] == name and row["step"] == 3]) for name in RESPONSE_STATE_NAMES]
    x = np.arange(6); plt.bar(x - 0.18, before, 0.36, label="before"); plt.bar(x + 0.18, after, 0.36, label="after K=3")
    plt.xticks(x, RESPONSE_STATE_NAMES, rotation=30, ha="right", fontsize=7); plt.ylabel("theta uncertainty"); plt.legend()
    save("uncertainty_before_after_probes.png")

    plt.figure(figsize=(9, 5))
    recovery_keys = ("Speed_Effect_Recovery_Ratio", "Distance_Effect_Recovery_Ratio", "Lateral_Effect_Recovery_Ratio", "Turn_Effect_Recovery_Ratio")
    x = np.arange(len(recovery_keys)); width = 0.12
    for index, strategy in enumerate(STRATEGIES):
        plt.bar(x + (index - 2.5) * width, [selection_summary[strategy]["3"][key] for key in recovery_keys], width, label=strategy)
    plt.xticks(x, ("speed", "distance", "lateral", "turn")); plt.ylabel("K=3 recovery ratio"); plt.legend(fontsize=5)
    save("per_dimension_recovery.png")
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_multiseed(args.output_dir)
        return
    random.seed(args.seed); np.random.seed(args.seed)
    print(LABEL, flush=True)

    import scripts.run_phase4b_personalization as phase4b
    from src.data.response_probe_schema import PROBE_CATALOG, probe_state_mask

    _, validation, splits = phase4b.build_corpora(phase4b_args(args))
    test = splits["T3_unseen_person_profile"]
    prior_mean, prior_std = load_f2_prior(args)

    penalty_lambda, probe_information_rows = calibrate_probe_tradeoff(
        validation, prior_mean, prior_std
    )
    observability_rows, jacobian_rows, observability_summary = build_observability_outputs(
        validation, prior_mean, prior_std
    )
    selection_rows, sequential_rows, specificity_rows, selection_summary = run_support_selection(
        test, prior_mean, prior_std, args
    )
    speed_rows = speed_horizon_diagnostics(args, prior_mean, prior_std)
    turn_rows = turn_diagnostics(validation, prior_mean, prior_std)

    probe_catalog_rows = []
    for probe in PROBE_CATALOG:
        probe_catalog_rows.append({
            "synthetic_interaction": LABEL, "probe": probe.probe_id,
            "speed_scale_delta": probe.speed_scale_delta,
            "distance_offset_m": probe.distance_offset_m,
            "lateral_offset_m": probe.lateral_offset_m,
            "turn_offset_rad": probe.turn_offset_rad,
            "synthetic_only": probe.synthetic_only,
            "high_level_action": probe.high_level_action,
            "response_state_mask": "|".join("1" if value else "0" for value in probe_state_mask(probe)),
            "contains_cmd_vel_sequence": False,
            "disturbance_definition": "0.20*|speed|/0.15 + 0.20*|distance|/0.30 + 0.15*|lateral|/0.20 + 0.15*|turn|/0.12 + 0.30*human_effect/0.05",
        })

    information_values = [float(row["InformationScore"]) for row in sequential_rows if row["strategy"] in ("Greedy Uncertainty", "Greedy Observability")]
    error_reductions = [float(row["actual_theta_error_reduction"]) for row in sequential_rows if row["strategy"] in ("Greedy Uncertainty", "Greedy Observability")]
    information_error_correlation = safe_correlation(information_values, error_reductions)
    future_error_reductions = [
        float(row["actual_future_response_error_reduction"])
        for row in sequential_rows
        if row["strategy"] in ("Greedy Uncertainty", "Greedy Observability")
        and row["actual_future_response_error_reduction"] is not None
    ]
    information_future_error_correlation = safe_correlation(
        information_values, future_error_reductions
    )
    greedy_specificity = [
        float(row["specificity_score"]) for row in specificity_rows
        if row["strategy"] in ("Greedy Uncertainty", "Greedy Observability")
        and row["response_dimension"] == "speed_response_gain"
    ]
    specificity_mean = float(np.mean(greedy_specificity))
    greedy_candidates = ("Greedy Uncertainty", "Greedy Observability")
    best_greedy = min(
        greedy_candidates,
        key=lambda name: selection_summary[name]["3"]["Functional_State_MAE"],
    )
    best_metrics = selection_summary[best_greedy]["3"]
    random_metrics = selection_summary["Random"]["3"]
    recent_metrics = selection_summary["Recent"]["3"]
    oracle_metrics = selection_summary["Oracle Informative"]["3"]
    old_speed, old_turn = 0.15372028946876526, 0.6019243597984314
    criteria = {
        "speed_gain_better_than_phase4b6": bool(best_metrics["speed_response_gain_MAE"] < 0.85 * old_speed),
        "turn_gain_better_than_phase4b6": bool(best_metrics["turn_response_gain_MAE"] < 0.85 * old_turn),
        "greedy_beats_random_and_recent": bool(
            best_metrics["Functional_State_MAE"] < min(random_metrics["Functional_State_MAE"], recent_metrics["Functional_State_MAE"])
            and best_metrics["Action_Effect_Error"] < min(random_metrics["Action_Effect_Error"], recent_metrics["Action_Effect_Error"])
            and best_metrics["Sensitivity_MAE"] < min(random_metrics["Sensitivity_MAE"], recent_metrics["Sensitivity_MAE"])
        ),
        "information_predicts_error_reduction": bool(
            information_error_correlation is not None and information_error_correlation > 0.20
        ),
        "uncertainty_reduction_dimension_specific": bool(specificity_mean > 0.60),
        "person_effect_recovery_preserved": bool(best_metrics["Person_Effect_Recovery_Ratio"] > 0.50),
        "oracle_is_upper_bound": bool(
            oracle_metrics["Functional_State_MAE"] <= best_metrics["Functional_State_MAE"]
            and oracle_metrics["Action_Effect_Error"] <= best_metrics["Action_Effect_Error"]
        ),
    }
    criteria["five_seed_gate_passed"] = bool(all(criteria.values()))
    criteria["ready_for_phase4c"] = False

    multiseed_rows = []
    for strategy in STRATEGIES:
        for metric in ("Functional_State_MAE", "Action_Effect_Error", "Sensitivity_MAE"):
            multiseed_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed,
                "strategy": strategy, "K": 3, "metric": metric,
                "value": selection_summary[strategy]["3"][metric],
                "detail": "seed42_gate_run_only; five-seed not started",
            })

    write_csv(args.output_dir / "observability_matrix.csv", observability_rows)
    write_csv(args.output_dir / "response_jacobian.csv", jacobian_rows)
    write_csv(args.output_dir / "probe_catalog.csv", probe_catalog_rows)
    write_csv(args.output_dir / "probe_information.csv", probe_information_rows)
    write_csv(args.output_dir / "support_selection.csv", selection_rows)
    write_csv(args.output_dir / "sequential_identification.csv", sequential_rows)
    write_csv(args.output_dir / "speed_identifiability.csv", speed_rows)
    write_csv(args.output_dir / "turn_identifiability.csv", turn_rows)
    write_csv(args.output_dir / "uncertainty_specificity.csv", specificity_rows)
    write_csv(args.output_dir / "multiseed.csv", multiseed_rows)
    figures = make_figures(
        args.output_dir, observability_rows, observability_summary,
        probe_information_rows, selection_summary, sequential_rows,
        speed_rows, turn_rows, specificity_rows,
    )
    summary = {
        "label": LABEL, "seed": args.seed, "five_seed_started": False,
        "phase4a_through_phase4b6_results_untouched": True,
        "experiment_scope": "offline synthetic replay/planning; no robot control",
        "frozen_f2_prior": {"mean": prior_mean, "std": prior_std},
        "probe_score_lambda": penalty_lambda,
        "lambda_selection_split": "validation",
        "observability": observability_summary,
        "support_selection": selection_summary,
        "best_greedy_strategy_at_K3": best_greedy,
        "information_error_reduction_correlation": information_error_correlation,
        "information_future_error_reduction_correlation": information_future_error_correlation,
        "uncertainty_specificity_mean": specificity_mean,
        "phase4b6_reference": {"speed_gain_MAE_K10": old_speed, "turn_gain_MAE_K10": old_turn},
        "success_criteria": criteria,
        "figures": figures,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"best={best_greedy} K3_MAE={best_metrics['Functional_State_MAE']:.6f} "
        f"speed={best_metrics['speed_response_gain_MAE']:.6f} "
        f"turn={best_metrics['turn_response_gain_MAE']:.6f} "
        f"gate={criteria['five_seed_gate_passed']}", flush=True,
    )


if __name__ == "__main__":
    main()
