"""Phase 4C uncertainty-aware counterfactual action selection (offline synthetic)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
MODELS = ("D0 Rule", "D1 Generic", "D2 Personalized", "D3 No Uncertainty", "D4 Oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-scenario", type=int, default=8)
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def clean(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({field: clean(row.get(field, "")) for field in fields})


SCENARIOS = (
    ("S1_too_close", 0, 0.92, 1.50, None, 3),
    ("S2_too_far", 0, 2.15, 1.50, None, 3),
    ("S3_human_accelerating", 2, 1.55, 1.45, "acceleration", 3),
    ("S4_human_decelerating", 3, 1.55, 1.45, "deceleration", 3),
    ("S5_human_turning", 5, 1.55, 1.45, "left_turn", 3),
    ("S6_high_distance_sensitive", 1, 1.25, 1.50, None, 3),
    ("S7_high_speed_sensitive", 2, 1.80, 1.50, None, 3),
    ("S8_high_turn_sensitive", 5, 1.72, 1.48, "right_turn", 3),
    ("S9_uncertain_new_person", 6, 1.78, 1.50, None, 0),
    ("S10_action_conflict", 1, 1.72, 1.45, None, 1),
)


def load_prior(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from src.data.response_statistics import RESPONSE_STATISTIC_DIM
    from src.models.functional_response_decoder import FunctionalResponseWorldModel
    path = args.phase4b6_dir / "checkpoints" / "f2_original_best.pt"
    model = FunctionalResponseWorldModel().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    with torch.inference_mode():
        estimate = model.estimator(
            torch.zeros(1, 10, RESPONSE_STATISTIC_DIM),
            torch.zeros(1, 10, dtype=torch.bool),
            torch.zeros(1, 10, 6, dtype=torch.bool),
        )
    return estimate.theta_mean[0].numpy(), estimate.theta_log_std[0].exp().numpy()


def set_robot_distance(history: np.ndarray, robot: np.ndarray, distance: float) -> np.ndarray:
    from src.data.skeleton_schema import compute_root
    result = robot.copy(); roots = compute_root(history)
    relative = roots[-1, :2] - result[-1, :2]
    norm = float(np.linalg.norm(relative))
    direction = relative / norm if norm > 1e-8 else np.asarray((1.0, 0.0))
    result[:, :2] = roots[:, :2] - distance * direction[None]
    result[:, 5] = distance
    angle = np.arctan2(direction[1], direction[0]) - result[:, 2]
    result[:, 6] = np.arctan2(np.sin(angle), np.cos(angle))
    return result.astype(np.float32)


def scenario_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    from src.data.synthetic_interaction import generate_interaction_split
    samples: list[dict[str, Any]] = []
    for scenario_index, (name, profile, distance, target, action_type, support_k) in enumerate(SCENARIOS):
        split = generate_interaction_split(
            120, args.seed + 10_000 + scenario_index * 103, name,
            profile_ids=(profile,), noise_std=0.005, occlusion_rate=0.10,
        )
        indices = np.arange(len(split))
        if action_type is not None:
            matched = np.flatnonzero(split.action_type == action_type)
            if len(matched) >= args.samples_per_scenario: indices = matched
        for local_index, source in enumerate(indices[: args.samples_per_scenario]):
            samples.append({
                "scenario": name, "scenario_index": scenario_index,
                "sample": local_index, "profile": profile,
                "support_k": support_k, "target_distance": target,
                "history": split.human_history[source],
                "natural": split.natural_future[source],
                "robot": set_robot_distance(split.human_history[source], split.robot_history[source], distance),
                "confidence": split.confidence[source],
                "visibility": split.visibility_mask[source],
                "action_type": str(split.action_type[source]),
            })
    return samples


def estimate_personal_belief(
    sample: dict[str, Any], theta_true: np.ndarray,
    prior_mean: np.ndarray, prior_std: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    from src.data.response_probe_schema import PROBE_BY_ID
    from src.data.synthetic_interaction import generate_interaction_split
    from src.evaluation.response_identifiability import (
        FunctionalBelief, functional_belief_update, response_jacobian,
        simulate_functional_probe,
    )
    k = int(sample["support_k"])
    if k == 0: return prior_mean.copy(), prior_std.copy(), ()
    support = generate_interaction_split(
        max(k, 3), seed, "phase4c_past_support",
        profile_ids=(int(sample["profile"]),), noise_std=0.005, occlusion_rate=0.10,
    )
    order = (
        "DISTANCE_PLUS_0_2", "SPEED_UP_10", "DISTANCE_MINUS_0_2",
        "TURN_LEFT_SMALL", "SPEED_DOWN_10",
    )
    belief = FunctionalBelief(prior_mean.copy(), prior_std.copy())
    used = []
    for index in range(k):
        probe = PROBE_BY_ID[order[index % len(order)]]
        history, natural, robot = (
            support.human_history[index], support.natural_future[index], support.robot_history[index]
        )
        observed = simulate_functional_probe(history, natural, robot, probe, theta_true)
        predicted = simulate_functional_probe(history, natural, robot, probe, belief.mean)
        jacobian = response_jacobian(history, natural, robot, probe, belief.mean)
        belief = functional_belief_update(
            belief, observed.response_statistics, predicted.response_statistics, jacobian
        )
        used.append(probe.probe_id)
    return belief.mean.astype(np.float32), belief.std.astype(np.float32), tuple(used)


def make_candidates(scenario: str):
    from src.data.robot_action_schema import RobotAction
    from src.decision.candidate_action import CandidateAction, TASK_SAFE_CANDIDATES
    values = list(TASK_SAFE_CANDIDATES)
    if scenario == "S1_too_close":
        index = next(i for i, item in enumerate(values) if item.action == RobotAction.DISTANCE_MINUS_0_2)
        values[index] = CandidateAction(RobotAction.DISTANCE_MINUS_0_2, feasible=False)
    return tuple(values)


def make_state(sample: dict[str, Any], theta: np.ndarray, uncertainty: np.ndarray):
    from src.decision.decision_state import DecisionState, FunctionalResponseBelief
    return DecisionState(
        human_history=sample["history"], robot_history=sample["robot"],
        confidence=sample["confidence"], visibility_mask=sample["visibility"],
        belief=FunctionalResponseBelief(theta, uncertainty),
        candidates=make_candidates(sample["scenario"]),
        target_follow_distance=float(sample["target_distance"]),
        too_close_distance=0.80, scenario_id=sample["scenario"],
    )


def ground_truth_rollout(sample: dict[str, Any], state: Any, theta_true: np.ndarray):
    from src.data.skeleton_schema import compute_root
    from src.decision.counterfactual_rollout import CounterfactualRollout, _robot_future
    from src.evaluation.response_identifiability import (
        classic_probe_for_action, simulate_functional_probe,
    )
    action_ids = np.asarray([int(item.action) for item in state.candidates], dtype=np.int64)
    simulations = [
        simulate_functional_probe(
            sample["history"], sample["natural"], sample["robot"],
            classic_probe_for_action(int(action_id)), theta_true,
        )
        for action_id in action_ids
    ]
    global_future = np.stack([item.future_global for item in simulations])
    effects = np.stack([item.action_effect for item in simulations])
    roots = compute_root(global_future)
    local = global_future - roots[:, :, None]
    robot_future = _robot_future(
        sample["history"], sample["robot"], action_ids,
        global_future.shape[1],
    )
    distances = np.linalg.norm(roots[..., :2] - robot_future, axis=-1)
    return CounterfactualRollout(
        action_ids, sample["natural"], roots, local, global_future,
        robot_future, distances.astype(np.float32), effects,
        np.zeros_like(effects), 0,
    )


def rank_fidelity(predicted: Any, expected: Any) -> dict[str, float]:
    pred = np.linalg.norm(predicted.predicted_action_effect, axis=-1).mean(axis=(1, 2))[1:]
    truth = np.linalg.norm(expected.predicted_action_effect, axis=-1).mean(axis=(1, 2))[1:]
    correct = total = 0
    for left in range(len(pred)):
        for right in range(left + 1, len(pred)):
            correct += int(np.sign(pred[left] - pred[right]) == np.sign(truth[left] - truth[right]))
            total += 1
    pred_rank = np.argsort(np.argsort(pred)).astype(float)
    true_rank = np.argsort(np.argsort(truth)).astype(float)
    spearman = float(np.corrcoef(pred_rank, true_rank)[0, 1]) if np.std(pred_rank) else 0.0
    return {
        "pairwise_ranking_accuracy": correct / max(total, 1),
        "top1_action_agreement": float(np.argmax(pred) == np.argmax(truth)),
        "spearman_ranking_correlation": spearman,
        "response_effect_MAE": float(np.mean(np.abs(pred - truth))),
    }


def selected_metrics(
    model: str, sample: dict[str, Any], selected_action: int,
    predicted_result: Any | None, gt_costs: Any, oracle_action: int,
    gt_optimal_action: int,
) -> dict[str, Any]:
    action_ids = gt_costs.action_ids.tolist()
    index = action_ids.index(int(selected_action))
    oracle_index = action_ids.index(int(oracle_action))
    initial_error = abs(float(sample["robot"][-1, 5]) - float(sample["target_distance"]))
    final_error = abs(float(gt_costs.minimum_distance[index]) - float(sample["target_distance"]))
    return {
        "synthetic_interaction": LABEL, "seed": sample["seed"],
        "scenario": sample["scenario"], "sample": sample["sample"],
        "profile": sample["profile"], "model": model,
        "selected_action": int(selected_action),
        "oracle_action": int(oracle_action), "gt_optimal_action": int(gt_optimal_action),
        "GT_Task_Cost": gt_costs.task[index],
        "GT_Safety_Cost": gt_costs.safety[index],
        "GT_Human_Response_Cost": gt_costs.human_response[index],
        "GT_Synthetic_Disturbance_Cost": gt_costs.disturbance[index],
        "GT_Total_Cost": gt_costs.total[index],
        "Minimum_Human_Robot_Distance": gt_costs.minimum_distance[index],
        "Safety_Violation": bool(gt_costs.unsafe_duration[index] > 0),
        "Task_Progress": initial_error - final_error,
        "Action_Selection_Accuracy": float(int(selected_action) == int(gt_optimal_action)),
        "Oracle_Regret": float(gt_costs.total[index] - gt_costs.total[oracle_index]),
        "Fallback": bool(predicted_result.safety_gate.abstained) if predicted_result is not None else False,
        "Fallback_Reason": predicted_result.safety_gate.fallback_reason if predicted_result is not None else "",
        "predicted_total_cost": (
            predicted_result.costs.total[predicted_result.selected_index]
            if predicted_result is not None else None
        ),
    }


def run_seed(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from src.data.synthetic_interaction import PROFILE_BY_ID
    from src.data.functional_response_state import functional_state_from_profile
    from src.decision.action_selector import (
        rule_based_select, select_model_action, select_oracle_action,
    )
    from src.decision.counterfactual_rollout import CounterfactualRolloutEngine
    from src.decision.decision_cost import DecisionCostWeights, compute_decision_costs

    checkpoint = args.phase4b6_dir / "checkpoints" / "f2_original_best.pt"
    engine = CounterfactualRolloutEngine.from_phase4b6_checkpoint(checkpoint, args.device)
    prior_mean, prior_std = load_prior(args)
    weights = DecisionCostWeights()
    selected_rows: list[dict[str, Any]] = []
    candidate_cost_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    scenario_examples: list[dict[str, Any]] = []

    for episode, sample in enumerate(scenario_samples(args)):
        sample["seed"] = args.seed
        theta_true = functional_state_from_profile(PROFILE_BY_ID[int(sample["profile"])]).astype(np.float32)
        theta_hat, theta_std, support_sequence = estimate_personal_belief(
            sample, theta_true, prior_mean, prior_std,
            args.seed + 70_000 + episode * 31,
        )
        generic_state = make_state(sample, prior_mean.astype(np.float32), prior_std.astype(np.float32))
        personal_state = make_state(sample, theta_hat, theta_std)
        oracle_state = make_state(sample, theta_true, np.zeros(6, dtype=np.float32))

        generic_rollout = engine.rollout(generic_state, uncertainty_aware=True)
        personal_rollout = engine.rollout(personal_state, uncertainty_aware=True)
        oracle_rollout = engine.rollout(oracle_state, uncertainty_aware=False)
        gt_rollout = ground_truth_rollout(sample, oracle_state, theta_true)
        gt_costs = compute_decision_costs(
            oracle_state, gt_rollout, weights, include_uncertainty=False
        )
        generic = select_model_action(generic_state, generic_rollout, weights, use_uncertainty=True)
        personal = select_model_action(personal_state, personal_rollout, weights, use_uncertainty=True)
        no_uncertainty = select_model_action(
            personal_state, personal_rollout, weights, use_uncertainty=False
        )
        oracle = select_oracle_action(oracle_state, oracle_rollout, theta_true, weights)
        rule_action = rule_based_select(generic_state)
        gt_optimal_index = int(np.argmin(gt_costs.total))
        gt_optimal_action = int(gt_costs.action_ids[gt_optimal_index])
        oracle_action = oracle.selected_action

        results = (
            ("D0 Rule", rule_action, None),
            ("D1 Generic", generic.selected_action, generic),
            ("D2 Personalized", personal.selected_action, personal),
            ("D3 No Uncertainty", no_uncertainty.selected_action, no_uncertainty),
            ("D4 Oracle", oracle.selected_action, oracle),
        )
        for model, action, result in results:
            row = selected_metrics(
                model, sample, action, result, gt_costs,
                oracle_action, gt_optimal_action,
            )
            row.update({
                "support_K": sample["support_k"],
                "support_sequence": "|".join(support_sequence),
                "theta_uncertainty_mean": float(theta_std.mean()),
                "action_type": sample["action_type"],
            })
            selected_rows.append(row)
            if result is None:
                candidate_cost_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "scenario": sample["scenario"], "sample": sample["sample"],
                    "profile": sample["profile"], "model": model,
                    "candidate_action": int(action), "selected": True,
                    "selector_uses_world_model": False,
                    "potential_information_gain": 0.0,
                    "information_gain_in_objective": False,
                })
            else:
                for candidate_index, candidate_action in enumerate(result.costs.action_ids):
                    gt_index = gt_costs.action_ids.tolist().index(int(candidate_action))
                    candidate_cost_rows.append({
                        "synthetic_interaction": LABEL, "seed": args.seed,
                        "scenario": sample["scenario"], "sample": sample["sample"],
                        "profile": sample["profile"], "model": model,
                        "candidate_action": int(candidate_action),
                        "selected": int(candidate_action) == int(action),
                        "allowed_by_safety_gate": bool(result.safety_gate.allowed_mask[candidate_index]),
                        "J_task": result.costs.task[candidate_index],
                        "J_safety": result.costs.safety[candidate_index],
                        "J_human_response": result.costs.human_response[candidate_index],
                        "J_disturbance": result.costs.disturbance[candidate_index],
                        "J_uncertainty": result.costs.uncertainty[candidate_index],
                        "J_total": result.costs.total[candidate_index],
                        "GT_J_task": gt_costs.task[gt_index],
                        "GT_J_safety": gt_costs.safety[gt_index],
                        "GT_J_human_response": gt_costs.human_response[gt_index],
                        "GT_J_disturbance": gt_costs.disturbance[gt_index],
                        "GT_J_total": gt_costs.total[gt_index],
                        "potential_information_gain": result.costs.potential_information_gain[candidate_index],
                        "information_gain_in_objective": False,
                        "selector_uses_world_model": True,
                    })

        for model, rollout in (
            ("D1 Generic", generic_rollout),
            ("D2 Personalized", personal_rollout),
            ("D4 Oracle", oracle_rollout),
        ):
            fidelity_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed,
                "scenario": sample["scenario"], "sample": sample["sample"],
                "profile": sample["profile"], "model": model,
                **rank_fidelity(rollout, gt_rollout),
            })

        for index, candidate in enumerate(personal_state.candidates):
            safety_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed,
                "scenario": sample["scenario"], "sample": sample["sample"],
                "action": int(candidate.action),
                "feasible": candidate.feasible,
                "predicted_min_distance": personal_rollout.predicted_human_robot_distance[index].min(),
                "gt_min_distance": gt_rollout.predicted_human_robot_distance[index].min(),
                "gt_unsafe": bool(gt_costs.unsafe_duration[index] > 0),
                "allowed": bool(personal.safety_gate.allowed_mask[index]),
                "rejection_reason": personal.safety_gate.rejection_reasons[index],
            })

        ablations = (
            ("Full_D2", personal),
            ("remove_personalization", generic),
            ("remove_uncertainty", no_uncertainty),
            ("remove_human_response", select_model_action(
                personal_state, personal_rollout, weights,
                include_human_response=False,
            )),
            ("remove_disturbance", select_model_action(
                personal_state, personal_rollout, weights,
                include_disturbance=False,
            )),
            ("remove_safety_hard_gate_OFFLINE_ONLY", select_model_action(
                personal_state, personal_rollout, weights,
                hard_safety=False,
            )),
        )
        for name, result in ablations:
            action_index = gt_costs.action_ids.tolist().index(result.selected_action)
            ablation_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed,
                "scenario": sample["scenario"], "sample": sample["sample"],
                "ablation": name, "selected_action": result.selected_action,
                "GT_Total_Cost": gt_costs.total[action_index],
                "GT_Task_Cost": gt_costs.task[action_index],
                "GT_Safety_Violation": bool(gt_costs.unsafe_duration[action_index] > 0),
                "Oracle_Regret": gt_costs.total[action_index] - gt_costs.total[gt_costs.action_ids.tolist().index(oracle_action)],
            })
        if sample["sample"] == 0:
            scenario_examples.append({
                "scenario": sample["scenario"], "history": sample["history"],
                "gt_rollout": gt_rollout, "personal": personal,
                "personal_rollout": personal_rollout,
                "generic": generic, "generic_rollout": generic_rollout,
                "oracle": oracle,
            })

    return summarize_seed(
        args, selected_rows, candidate_cost_rows, fidelity_rows, safety_rows,
        ablation_rows, scenario_examples,
    )


def mean_rows(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0


def summarize_seed(
    args: argparse.Namespace, selected_rows: list[dict[str, Any]],
    candidate_cost_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]], safety_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]], examples: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "GT_Task_Cost", "GT_Safety_Cost", "GT_Human_Response_Cost",
        "GT_Synthetic_Disturbance_Cost", "GT_Total_Cost",
        "Minimum_Human_Robot_Distance", "Safety_Violation", "Task_Progress",
        "Action_Selection_Accuracy", "Oracle_Regret", "Fallback",
    )
    scenario_rows, person_rows = [], []
    summary: dict[str, Any] = {model: {} for model in MODELS}
    for model in MODELS:
        model_rows = [row for row in selected_rows if row["model"] == model]
        summary[model] = {
            metric: mean_rows(model_rows, metric) for metric in metric_names
        }
        actions = [int(row["selected_action"]) for row in model_rows]
        summary[model]["KEEP_Rate"] = float(np.mean(np.asarray(actions) == 0))
        summary[model]["Action_Switch_Rate"] = float(np.mean(np.diff(actions) != 0)) if len(actions) > 1 else 0.0
        for scenario in sorted(set(row["scenario"] for row in model_rows)):
            rows = [row for row in model_rows if row["scenario"] == scenario]
            for metric in metric_names:
                scenario_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "scenario": scenario, "model": model,
                    "metric": metric, "value": mean_rows(rows, metric),
                })
        for profile in sorted(set(int(row["profile"]) for row in model_rows)):
            rows = [row for row in model_rows if int(row["profile"]) == profile]
            for metric in metric_names:
                person_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "profile": profile, "model": model,
                    "metric": metric, "value": mean_rows(rows, metric),
                })

    regret_rows = [
        {
            "synthetic_interaction": LABEL, "seed": args.seed,
            "scenario": row["scenario"], "sample": row["sample"],
            "profile": row["profile"], "model": row["model"],
            "selected_action": row["selected_action"],
            "oracle_action": row["oracle_action"],
            "Oracle_Regret": row["Oracle_Regret"],
            "Personalization_Regret": (
                row["GT_Total_Cost"]
                - next(item["GT_Total_Cost"] for item in selected_rows if item["scenario"] == row["scenario"] and item["sample"] == row["sample"] and item["model"] == "D2 Personalized")
            ),
        }
        for row in selected_rows
    ]
    uncertain_scenarios = ("S9_uncertain_new_person", "S10_action_conflict")
    uncertainty_rows = [
        row for row in selected_rows
        if row["scenario"] in uncertain_scenarios
        and row["model"] in ("D2 Personalized", "D3 No Uncertainty")
    ]
    unsafe_truth = [row for row in safety_rows if row["gt_unsafe"]]
    unsafe_rejection_rate = float(np.mean([not bool(row["allowed"]) for row in unsafe_truth])) if unsafe_truth else 1.0
    false_rejection = [row for row in safety_rows if not row["gt_unsafe"]]
    false_rejection_rate = float(np.mean([not bool(row["allowed"]) for row in false_rejection])) if false_rejection else 0.0

    sensitive = ("S6_high_distance_sensitive", "S7_high_speed_sensitive", "S8_high_turn_sensitive")
    def selected(model: str, scenarios: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        return [
            row for row in selected_rows
            if row["model"] == model and (scenarios is None or row["scenario"] in scenarios)
        ]
    d1_sensitive, d2_sensitive = selected("D1 Generic", sensitive), selected("D2 Personalized", sensitive)
    d2_uncertain, d3_uncertain = selected("D2 Personalized", uncertain_scenarios), selected("D3 No Uncertainty", uncertain_scenarios)
    high_regret_threshold = 0.08
    d2_bad = mean_rows([
        {"bad": float(row["Safety_Violation"] or row["Oracle_Regret"] > high_regret_threshold)}
        for row in d2_uncertain
    ], "bad")
    d3_bad = mean_rows([
        {"bad": float(row["Safety_Violation"] or row["Oracle_Regret"] > high_regret_threshold)}
        for row in d3_uncertain
    ], "bad")
    d2_fidelity = [row for row in fidelity_rows if row["model"] == "D2 Personalized"]
    d2_regret = selected("D2 Personalized")
    fidelity_lookup = {(row["scenario"], row["sample"]): row for row in d2_fidelity}
    fidelity_values, quality_values = [], []
    for row in d2_regret:
        fidelity = fidelity_lookup[(row["scenario"], row["sample"])]
        fidelity_values.append(float(fidelity["pairwise_ranking_accuracy"]))
        quality_values.append(-float(row["Oracle_Regret"]))
    fidelity_quality_correlation = (
        float(np.corrcoef(fidelity_values, quality_values)[0, 1])
        if np.std(fidelity_values) > 1e-12 and np.std(quality_values) > 1e-12 else None
    )
    different_actions = float(np.mean([
        int(left["selected_action"]) != int(right["selected_action"])
        for left, right in zip(d1_sensitive, d2_sensitive)
    ]))
    criteria = {
        "D2_sensitive_GT_cost_better_than_D1": bool(
            mean_rows(d2_sensitive, "GT_Total_Cost") < mean_rows(d1_sensitive, "GT_Total_Cost")
        ),
        "D2_oracle_regret_below_D1": bool(
            mean_rows(selected("D2 Personalized"), "Oracle_Regret")
            < mean_rows(selected("D1 Generic"), "Oracle_Regret")
        ),
        "D2_uncertainty_reduces_bad_choices": bool(d2_bad < d3_bad),
        "hard_safety_gate_reliable": bool(
            unsafe_rejection_rate >= 0.95
            and summary["D2 Personalized"]["Safety_Violation"] <= 0.01
        ),
        "D2_not_keep_only_and_task_preserved": bool(
            summary["D2 Personalized"]["KEEP_Rate"] < 0.80
            and summary["D2 Personalized"]["GT_Task_Cost"]
            <= 1.15 * summary["D1 Generic"]["GT_Task_Cost"]
        ),
        "counterfactual_ranking_above_random": bool(
            mean_rows(d2_fidelity, "pairwise_ranking_accuracy") > 0.60
        ),
        "D4_reasonable_upper_bound": bool(
            summary["D4 Oracle"]["Oracle_Regret"]
            <= summary["D2 Personalized"]["Oracle_Regret"]
        ),
        "selection_uses_prediction_not_GT": True,
    }
    criteria["five_seed_gate_passed"] = bool(all(criteria.values()))
    criteria["ready_to_freeze_phase4c"] = False

    write_csv(args.output_dir / "decision_by_scenario.csv", scenario_rows)
    write_csv(args.output_dir / "decision_by_person.csv", person_rows)
    write_csv(args.output_dir / "selected_actions.csv", candidate_cost_rows)
    write_csv(args.output_dir / "counterfactual_fidelity.csv", fidelity_rows)
    write_csv(args.output_dir / "regret.csv", regret_rows)
    write_csv(args.output_dir / "uncertainty_ablation.csv", uncertainty_rows)
    write_csv(args.output_dir / "decision_ablation.csv", ablation_rows)
    write_csv(args.output_dir / "safety_gate.csv", safety_rows)
    multiseed_rows = [
        {
            "synthetic_interaction": LABEL, "seed": args.seed,
            "model": model, "metric": metric, "value": value,
            "statistic": "seed_value", "detail": "seed42 gate run only",
        }
        for model, values in summary.items() for metric, value in values.items()
    ]
    write_csv(args.output_dir / "multiseed.csv", multiseed_rows)
    figures = make_figures(
        args.output_dir, selected_rows, fidelity_rows, safety_rows,
        ablation_rows, examples, summary,
    )
    result = {
        "label": LABEL, "seed": args.seed, "five_seed_started": False,
        "scope": "one-step offline synthetic high-level decision; no robot control",
        "information_gain_in_main_objective": False,
        "decision_weights": {
            "selection_source": "fixed before test evaluation / validation protocol",
            "task": 1.0, "safety": 3.0, "human_response": 1.4,
            "disturbance": 0.55, "uncertainty": 0.85,
        },
        "models": summary,
        "person_sensitive": {
            "D1_GT_Total_Cost": mean_rows(d1_sensitive, "GT_Total_Cost"),
            "D2_GT_Total_Cost": mean_rows(d2_sensitive, "GT_Total_Cost"),
            "different_action_rate": different_actions,
        },
        "uncertain_scenarios": {
            "D2_bad_choice_rate": d2_bad, "D3_bad_choice_rate": d3_bad,
            "D2_task_cost": mean_rows(d2_uncertain, "GT_Task_Cost"),
            "D3_task_cost": mean_rows(d3_uncertain, "GT_Task_Cost"),
        },
        "safety": {
            "unsafe_action_rejection_rate": unsafe_rejection_rate,
            "safe_action_false_rejection_rate": false_rejection_rate,
        },
        "counterfactual": {
            "D2_pairwise_ranking_accuracy": mean_rows(d2_fidelity, "pairwise_ranking_accuracy"),
            "D2_top1_agreement": mean_rows(d2_fidelity, "top1_action_agreement"),
            "D2_spearman": mean_rows(d2_fidelity, "spearman_ranking_correlation"),
            "D2_effect_MAE": mean_rows(d2_fidelity, "response_effect_MAE"),
            "fidelity_decision_quality_correlation": fidelity_quality_correlation,
        },
        "success_criteria": criteria,
        "figures": figures,
    }
    write_json(args.output_dir / "summary.json", result)
    print(
        f"D1_cost={summary['D1 Generic']['GT_Total_Cost']:.5f} "
        f"D2_cost={summary['D2 Personalized']['GT_Total_Cost']:.5f} "
        f"D2_regret={summary['D2 Personalized']['Oracle_Regret']:.5f} "
        f"gate={criteria['five_seed_gate_passed']}", flush=True,
    )
    return result


def make_figures(
    output_dir: Path, selected_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]], safety_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]], examples: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.skeleton_schema import compute_root

    figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    def save(name: str) -> None:
        path = figure_dir / name; plt.title(LABEL, fontsize=8)
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(); paths.append(str(path))

    example = examples[0]
    history_root = compute_root(example["history"])
    plt.figure(figsize=(7, 5)); plt.plot(history_root[:, 0], history_root[:, 1], "k.-", label="history")
    for index, action in enumerate(example["personal_rollout"].action_ids):
        root = example["personal_rollout"].predicted_root[index]
        plt.plot(root[:, 0], root[:, 1], marker=".", label=f"A{action}")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(fontsize=7)
    save("candidate_counterfactual_trajectories.png")

    plt.figure(figsize=(7, 4)); costs = example["personal"].costs
    plt.bar([f"A{x}" for x in costs.action_ids], costs.total)
    plt.ylabel("predicted total cost"); save("candidate_total_cost_comparison.png")

    plt.figure(figsize=(8, 4)); plt.bar(MODELS, [summary[model]["Oracle_Regret"] for model in MODELS])
    plt.xticks(rotation=20, ha="right"); plt.ylabel("mean oracle regret")
    save("decision_model_regret.png")

    d1 = [row for row in selected_rows if row["model"] == "D1 Generic"]
    d2 = [row for row in selected_rows if row["model"] == "D2 Personalized"]
    matrix = np.zeros((5, 5), dtype=int)
    for left, right in zip(d1, d2): matrix[int(left["selected_action"]), int(right["selected_action"])] += 1
    plt.figure(figsize=(6, 5)); plt.imshow(matrix, cmap="Blues"); plt.colorbar(label="count")
    plt.xlabel("D2 action"); plt.ylabel("D1 action")
    save("generic_vs_personalized_actions.png")

    aggressiveness = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.6, 4: 1.0}
    plt.figure(figsize=(7, 5)); plt.scatter(
        [row["theta_uncertainty_mean"] for row in d2],
        [aggressiveness[int(row["selected_action"])] for row in d2], alpha=0.65,
    )
    plt.xlabel("theta uncertainty"); plt.ylabel("selected action aggressiveness")
    save("uncertainty_vs_action_aggressiveness.png")

    sample_safety = safety_rows[:25]
    colors = ["tab:red" if not row["allowed"] else "tab:green" for row in sample_safety]
    plt.figure(figsize=(9, 4)); plt.bar(range(len(sample_safety)), [row["predicted_min_distance"] for row in sample_safety], color=colors)
    plt.axhline(0.8, color="k", linestyle="--"); plt.ylabel("predicted minimum distance")
    save("safety_rejection_examples.png")

    plt.figure(figsize=(7, 5))
    for model in ("D0 Rule", "D1 Generic", "D2 Personalized", "D4 Oracle"):
        rows = [row for row in selected_rows if row["model"] == model]
        plt.scatter([row["GT_Task_Cost"] for row in rows], [row["GT_Human_Response_Cost"] for row in rows], alpha=0.45, label=model)
    plt.xlabel("GT task cost"); plt.ylabel("GT human-response cost"); plt.legend(fontsize=6)
    save("task_human_response_pareto.png")

    profiles = sorted(set(int(row["profile"]) for row in d2))
    plt.figure(figsize=(7, 4)); plt.bar(
        [str(profile) for profile in profiles],
        [mean_rows([row for row in d2 if int(row["profile"]) == profile], "Oracle_Regret") for profile in profiles],
    )
    plt.xlabel("virtual profile"); plt.ylabel("D2 oracle regret"); save("per_person_regret.png")

    plt.figure(figsize=(7, 4)); plt.bar(MODELS, [summary[model]["Fallback"] for model in MODELS])
    plt.xticks(rotation=20, ha="right"); plt.ylabel("fallback rate"); save("fallback_rate.png")

    plt.figure(figsize=(7, 4))
    fidelity_models = ("D1 Generic", "D2 Personalized", "D4 Oracle")
    plt.bar(fidelity_models, [mean_rows([row for row in fidelity_rows if row["model"] == model], "pairwise_ranking_accuracy") for model in fidelity_models])
    plt.ylabel("counterfactual pairwise ranking accuracy"); save("counterfactual_ranking_accuracy.png")
    return paths


def aggregate_multiseed(output_dir: Path) -> None:
    seeds = (42, 123, 3407, 2026, 7777)
    paths = {
        42: output_dir / "summary.json",
        **{seed: output_dir / "multiseed_runs" / f"seed{seed}" / "summary.json" for seed in seeds[1:]},
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing: raise FileNotFoundError(f"missing multiseed summaries: {missing}")
    summaries = {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in paths.items()}
    metrics = (
        "GT_Task_Cost", "GT_Safety_Cost", "GT_Human_Response_Cost",
        "GT_Synthetic_Disturbance_Cost", "GT_Total_Cost",
        "Minimum_Human_Robot_Distance", "Safety_Violation", "Task_Progress",
        "Action_Selection_Accuracy", "Oracle_Regret", "Fallback", "KEEP_Rate",
        "Action_Switch_Rate",
    )
    rows, aggregate = [], {}
    for model in MODELS:
        aggregate[model] = {}
        for metric in metrics:
            values = np.asarray([summaries[seed]["models"][model][metric] for seed in seeds], dtype=float)
            for seed, value in zip(seeds, values):
                rows.append({"synthetic_interaction": LABEL, "seed": seed, "model": model, "metric": metric, "value": value, "statistic": "seed_value"})
            aggregate[model][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
            rows.extend((
                {"synthetic_interaction": LABEL, "seed": "ALL", "model": model, "metric": metric, "value": values.mean(), "statistic": "mean"},
                {"synthetic_interaction": LABEL, "seed": "ALL", "model": model, "metric": metric, "value": values.std(ddof=1), "statistic": "std"},
            ))
    write_csv(output_dir / "multiseed.csv", rows)
    root = summaries[42]
    all_gates = all(summary["success_criteria"]["five_seed_gate_passed"] for summary in summaries.values())
    root.update({
        "five_seed_started": True, "five_seed_completed": True,
        "multiseed_seeds": list(seeds), "all_seed_gates_passed": all_gates,
        "multiseed_summary": aggregate,
    })
    root["success_criteria"]["five_seed_gate_passed"] = all_gates
    root["success_criteria"]["ready_to_freeze_phase4c"] = all_gates
    root["ready_for_phase5_research"] = all_gates
    write_json(output_dir / "summary.json", root)
    print(f"aggregated five seeds; all_gates_passed={all_gates}", flush=True)


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_multiseed(args.output_dir); return
    random.seed(args.seed); np.random.seed(args.seed)
    print(LABEL, flush=True)
    run_seed(args)


if __name__ == "__main__":
    main()
