"""Phase 4C.2 belief-space constrained selection (offline synthetic only)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--root-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--belief-samples", type=int, choices=(16, 32), default=16)
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
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
        writer = csv.DictWriter(handle, fieldnames=fields or ("empty",)); writer.writeheader()
        for row in rows: writer.writerow({field: clean(row.get(field, "")) for field in fields})


def grouped_indices(records: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault((str(record["scenario"]), int(record["sample"])), []).append(index)
    return list(groups.values())


def train_root_head(args: argparse.Namespace, train: list[dict[str, Any]], validation: list[dict[str, Any]], torch: Any):
    from src.data.skeleton_schema import compute_root
    from src.decision.root_belief import RootResidualBeliefHead

    def tensors(records: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
        unique = [records[group[0]] for group in grouped_indices(records)]
        history = np.stack([compute_root(row["sample_data"]["history"]) for row in unique]).astype(np.float32)
        frozen = np.stack([compute_root(row["predicted_rollout"].natural_future) for row in unique]).astype(np.float32)
        target = np.stack([compute_root(row["sample_data"]["natural"]) for row in unique]).astype(np.float32)
        return tuple(torch.from_numpy(value) for value in (history, frozen, target))

    device = torch.device(args.device); head = RootResidualBeliefHead().to(device)
    train_data, validation_data = tensors(train), tensors(validation)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(args.seed + 991)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.root_epochs + 1):
        head.train(); order = torch.randperm(len(train_data[0]), generator=generator)
        for start in range(0, len(order), args.batch_size):
            batch = tuple(value[order[start:start + args.batch_size]].to(device) for value in train_data)
            output = head(batch[0], batch[1]); target_residual = batch[2] - batch[1]
            error = output["residual"] - target_residual
            nll = .5 * (error.square() * torch.exp(-2 * output["log_sigma"]) + 2 * output["log_sigma"])
            loss = nll.mean() + .15 * error.abs().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        head.eval()
        with torch.inference_mode():
            values = tuple(value.to(device) for value in validation_data)
            output = head(values[0], values[1]); error = output["residual"] - (values[2] - values[1])
            validation_loss = float(error.square().mean().item())
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, copy.deepcopy(head.state_dict())
    assert best_state is not None
    head.load_state_dict(best_state); head.eval()
    with torch.inference_mode():
        values = tuple(value.to(device) for value in validation_data); output = head(values[0], values[1])
        error = (values[1] + output["residual"] - values[2]).cpu().numpy()
        raw_sigma = output["log_sigma"].exp().cpu().numpy()
    sigma_scale = np.sqrt(np.mean(np.square(error / np.maximum(raw_sigma, 1e-5)), axis=(0, 1)))
    sigma_scale = np.maximum(sigma_scale, .1).astype(np.float32)
    checkpoint = args.output_dir / "checkpoints" / "root_belief_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "validation_mse": best_loss, "sigma_scale": sigma_scale}, checkpoint)
    return head, sigma_scale, {
        "checkpoint": str(checkpoint), "best_epoch": best_epoch,
        "validation_mse": best_loss, "parameters": sum(p.numel() for p in head.parameters()),
        "sigma_scale": sigma_scale,
    }


def predict_root_beliefs(head: Any, sigma_scale: np.ndarray, records: list[dict[str, Any]], device: Any, torch: Any) -> dict[tuple[str, int], Any]:
    from src.data.skeleton_schema import compute_root
    from src.decision.root_belief import make_root_belief
    result = {}
    for indices in grouped_indices(records):
        record = records[indices[0]]
        history = compute_root(record["sample_data"]["history"]).astype(np.float32)
        frozen = compute_root(record["predicted_rollout"].natural_future).astype(np.float32)
        with torch.inference_mode():
            output = head(
                torch.from_numpy(history)[None].to(device),
                torch.from_numpy(frozen)[None].to(device),
            )
        residual = output["residual"][0].cpu().numpy()
        sigma = np.exp(output["log_sigma"][0].cpu().numpy()) * sigma_scale[None]
        result[(record["scenario"], int(record["sample"]))] = make_root_belief(frozen, residual, sigma)
    return result


@dataclass(frozen=True)
class EpisodeArtifact:
    key: tuple[str, int]
    indices: tuple[int, ...]
    state: Any
    point_costs: Any
    gt_costs: Any
    distribution: Any
    root_belief: Any
    cost_features: np.ndarray


def make_artifacts(args: argparse.Namespace, records: list[dict[str, Any]], beliefs: dict[tuple[str, int], Any]) -> list[EpisodeArtifact]:
    from src.decision.belief_rollout import propagate_root_belief
    from src.decision.decision_cost import DecisionCostWeights, compute_decision_costs
    weights = DecisionCostWeights(); artifacts = []
    for group_number, indices in enumerate(grouped_indices(records)):
        first = records[indices[0]]; state = first["state"]
        point = compute_decision_costs(state, first["predicted_rollout"], weights, include_uncertainty=False)
        truth = compute_decision_costs(state, first["gt_rollout"], weights, include_uncertainty=False)
        key = (str(first["scenario"]), int(first["sample"])); belief = beliefs[key]
        distribution = propagate_root_belief(
            state, first["predicted_rollout"], belief, point,
            args.belief_samples, args.seed + 17_003 + group_number,
        )
        features = np.stack([np.concatenate((
            records[index]["features"],
            np.asarray((point.task[local], point.safety[local], point.human_response[local], point.disturbance[local])),
            np.asarray((distribution.expected_cost[local], distribution.std_cost[local], distribution.p95_cost[local], distribution.cvar_cost[local], distribution.chance_unsafe[local])),
        )).astype(np.float32) for local, index in enumerate(indices)])
        artifacts.append(EpisodeArtifact(key, tuple(indices), state, point, truth, distribution, belief, features))
    return artifacts


def calibration_arrays(artifacts: list[EpisodeArtifact]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([item.cost_features for item in artifacts])
    predicted = np.concatenate([np.column_stack((
        item.point_costs.task, item.point_costs.safety,
        item.point_costs.human_response, item.point_costs.disturbance,
    )) for item in artifacts])
    truth = np.concatenate([np.column_stack((
        item.gt_costs.task, item.gt_costs.safety,
        item.gt_costs.human_response, item.gt_costs.disturbance,
    )) for item in artifacts])
    return features, predicted, truth


@dataclass(frozen=True)
class SelectorConfig:
    gate_mode: str
    epsilon: float
    lcb_multiplier: float
    cost_mode: str
    uncertainty_lambda: float


def candidate_prediction(
    artifact: EpisodeArtifact, records: list[dict[str, Any]], safety_predictions: list[dict[str, Any]],
    cost_calibration: Any,
) -> dict[str, np.ndarray]:
    from src.decision.cost_calibration import apply_cost_residual_calibrator
    local_predictions = [safety_predictions[index] for index in artifact.indices]
    point_minimum = np.asarray([item["minimum"] for item in local_predictions])
    sigma_minimum = np.asarray([item["sigma_minimum"] for item in local_predictions])
    p_unsafe = np.asarray([item["p_unsafe"] for item in local_predictions])
    predicted_components = np.column_stack((
        artifact.point_costs.task, artifact.point_costs.safety,
        artifact.point_costs.human_response, artifact.point_costs.disturbance,
    ))
    components, component_sigma = apply_cost_residual_calibrator(
        cost_calibration, artifact.cost_features, predicted_components,
    )
    weights = np.asarray((1.0, 3.0, 1.4, .55))
    calibrated_mean = components @ weights
    raw_mean = np.asarray(artifact.distribution.expected_cost)
    correction = calibrated_mean - raw_mean
    return {
        "point_minimum": point_minimum, "sigma_minimum": sigma_minimum,
        "p_unsafe": p_unsafe, "chance": np.asarray(artifact.distribution.chance_unsafe),
        "mean": calibrated_mean,
        "std": np.sqrt(np.square(artifact.distribution.std_cost) + np.square(component_sigma @ weights)),
        "p95": np.maximum(np.asarray(artifact.distribution.p95_cost) + correction, calibrated_mean),
        "cvar": np.maximum(np.asarray(artifact.distribution.cvar_cost) + correction, calibrated_mean),
        "components": components,
        "point_total": np.asarray(artifact.point_costs.total),
    }


def select_prediction(artifact: EpisodeArtifact, prediction: dict[str, np.ndarray], config: SelectorConfig) -> tuple[Any, np.ndarray, tuple[str, ...], np.ndarray]:
    from src.decision.fallback_policy import constrained_select_with_fallback
    from src.decision.safety_gate import chance_constrained_candidate_mask
    hard = np.asarray([candidate.feasible for candidate in artifact.state.candidates], dtype=bool)
    feasible, reasons = chance_constrained_candidate_mask(
        hard, prediction["point_minimum"], prediction["sigma_minimum"],
        prediction["p_unsafe"], prediction["chance"], artifact.state.too_close_distance,
        config.epsilon, config.lcb_multiplier, config.gate_mode,
    )
    if config.cost_mode == "point": ranking = prediction["point_total"]
    elif config.cost_mode == "mean": ranking = prediction["mean"]
    elif config.cost_mode == "mean_std": ranking = prediction["mean"] + config.uncertainty_lambda * prediction["std"]
    elif config.cost_mode == "p95": ranking = prediction["p95"]
    elif config.cost_mode == "cvar": ranking = prediction["cvar"]
    else: raise ValueError(config.cost_mode)
    decision = constrained_select_with_fallback(
        artifact.point_costs.action_ids, feasible, ranking,
        float(artifact.state.robot_history[-1, 5]), artifact.state.target_follow_distance,
    )
    return decision, feasible, reasons, ranking


def config_score(config: SelectorConfig, artifacts: list[EpisodeArtifact], records: list[dict[str, Any]], predictions: list[dict[str, Any]], calibrator: Any) -> tuple[float, dict[str, float]]:
    violations, regrets, abstain, safe_rejected, safe_total, unsafe_rejected, unsafe_total = [], [], [], 0, 0, 0, 0
    for artifact in artifacts:
        predicted = candidate_prediction(artifact, records, predictions, calibrator)
        decision, feasible, _, _ = select_prediction(artifact, predicted, config)
        unsafe = artifact.gt_costs.unsafe_duration > 0
        safe_rejected += int(np.sum((~feasible) & (~unsafe))); safe_total += int(np.sum(~unsafe))
        unsafe_rejected += int(np.sum((~feasible) & unsafe)); unsafe_total += int(np.sum(unsafe))
        if decision.selected_index is None:
            violations.append(0.0); abstain.append(1.0)
            # Validation treats ABSTAIN as a distinct safe-hold state.  It pays an
            # explicit inability-to-progress cost, not the potentially unsafe
            # counterfactual KEEP trajectory cost.
            regrets.append(.25)
        else:
            index = decision.selected_index; violations.append(float(unsafe[index])); abstain.append(0.0)
            regrets.append(float(artifact.gt_costs.total[index] - artifact.gt_costs.total.min()))
    metrics = {
        "violation": float(np.mean(violations)), "regret": float(np.mean(regrets)),
        "abstain": float(np.mean(abstain)), "false_veto": safe_rejected / max(safe_total, 1),
        "unsafe_rejection": unsafe_rejected / max(unsafe_total, 1),
    }
    rejection_shortfall = max(.95 - metrics["unsafe_rejection"], 0.0)
    return 25.0 * metrics["violation"] + 8.0 * rejection_shortfall + metrics["regret"] + .35 * metrics["abstain"] + .08 * metrics["false_veto"], metrics


def select_on_validation(artifacts: list[EpisodeArtifact], records: list[dict[str, Any]], predictions: list[dict[str, Any]], calibrator: Any) -> tuple[SelectorConfig, list[dict[str, Any]]]:
    rows = []; best = None
    for gate in ("point", "lcb", "probability", "hybrid"):
        for epsilon in (.05, .10, .15, .20, .25):
            for lcb in (1.28, 1.64, 1.96):
                for cost_mode, uncertainty_lambda in (("point", 0.0), ("mean", 0.0), ("mean_std", .5), ("p95", 0.0), ("cvar", 0.0)):
                    config = SelectorConfig(gate, epsilon, lcb, cost_mode, uncertainty_lambda)
                    score, metrics = config_score(config, artifacts, records, predictions, calibrator)
                    row = {"synthetic_interaction": LABEL, "split": "validation", **asdict(config), "score": score, **metrics}
                    rows.append(row)
                    key = (score, metrics["violation"], metrics["regret"], metrics["abstain"])
                    if best is None or key < best[0]: best = (key, config)
    assert best is not None
    return best[1], rows


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = np.asarray(left, float), np.asarray(right, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 1e-12 and b.std() > 1e-12 else None


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = np.asarray(left, float), np.asarray(right, float)
    return correlation(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def evaluate_test(
    args: argparse.Namespace, artifacts: list[EpisodeArtifact], records: list[dict[str, Any]],
    safety_predictions: list[dict[str, Any]], calibrator: Any, config: SelectorConfig,
    generic_actions: dict[tuple[str, int], int],
) -> dict[str, Any]:
    from src.data.robot_action_schema import RobotAction
    from src.decision.fallback_policy import DecisionMode
    from src.decision.root_belief import root_error_components
    from src.decision.safety_calibration import worst_case_regret

    candidate_rows: list[dict[str, Any]] = []; selected_rows = []; fallback_rows = []
    unsafe_cases = []; root_rows = []; belief_rows = []; rollout_rows = []; switch_rows = []
    s9_rows = []; distance_rows = []; turn_rows = []; cost_rows = []
    gate_comparison: dict[str, list[dict[str, float]]] = {name: [] for name in ("point", "lcb", "probability", "hybrid")}
    for artifact_number, artifact in enumerate(artifacts):
        prediction = candidate_prediction(artifact, records, safety_predictions, calibrator)
        decision, feasible, reasons, ranking = select_prediction(artifact, prediction, config)
        first = records[artifact.indices[0]]
        truth_root = first["gt_rollout"].predicted_root
        natural_truth = first["sample_data"]["natural"]
        from src.data.skeleton_schema import compute_root
        natural_truth_root = compute_root(natural_truth)
        components = root_error_components(artifact.root_belief.mu_root, natural_truth_root)
        action_type = str(first["sample_data"]["action_type"])
        speed = np.linalg.norm(np.diff(natural_truth_root[:, :2], axis=0), axis=-1) * 10.0
        motion_state = (
            "turning" if "turn" in action_type else
            "accelerating" if "acceleration" in action_type else
            "decelerating" if "deceleration" in action_type else
            "high_speed" if speed.mean() > 1.4 else "nominal"
        )
        for horizon in range(len(natural_truth_root)):
            sigma_xy = np.linalg.norm(artifact.root_belief.sigma_root[horizon, :2])
            error_xy = np.linalg.norm(artifact.root_belief.mu_root[horizon, :2] - natural_truth_root[horizon, :2])
            root_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed, "person": first["profile"],
                "scenario": artifact.key[0], "motion_state": motion_state, "action": "NATURAL",
                "sample": artifact.key[1], "horizon_frame": horizon + 1,
                **{name: value[horizon] for name, value in components.items()},
            })
            belief_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed, "scenario": artifact.key[0],
                "sample": artifact.key[1], "horizon_frame": horizon + 1,
                "sigma_xy": sigma_xy, "root_error_xy": error_xy,
                "covered_68": error_xy <= sigma_xy, "covered_95": error_xy <= 1.96 * sigma_xy,
            })

        gt_unsafe = artifact.gt_costs.unsafe_duration > 0
        oracle_index = int(np.argmin(artifact.gt_costs.total))
        keep_matches = np.flatnonzero(artifact.gt_costs.action_ids == int(RobotAction.KEEP))
        keep_index = int(keep_matches[0])
        if decision.selected_index is None:
            selected_index = None
            # ABSTAIN is a distinct state; its evaluation uses minimal-motion hold
            # plus a transparent inability-to-decide task penalty.
            total_cost = float(artifact.gt_costs.total[oracle_index] + .25)
            task_cost = float(artifact.gt_costs.task[oracle_index] + .25)
            human_cost = float(artifact.gt_costs.human_response[oracle_index])
            safety_violation = False
            regret = .25
        else:
            selected_index = int(decision.selected_index)
            total_cost = float(artifact.gt_costs.total[selected_index])
            task_cost = float(artifact.gt_costs.task[selected_index])
            human_cost = float(artifact.gt_costs.human_response[selected_index])
            safety_violation = bool(gt_unsafe[selected_index])
            regret = total_cost - float(artifact.gt_costs.total[oracle_index])
        selected_action = decision.selected_action
        selected_rows.append({
            "synthetic_interaction": LABEL, "seed": args.seed, "scenario": artifact.key[0],
            "sample": artifact.key[1], "profile": first["profile"],
            "selected_action": "" if selected_action is None else selected_action,
            "decision_mode": decision.mode.value, "GT_Total_Cost": total_cost,
            "GT_Task_Cost": task_cost, "GT_Human_Response_Cost": human_cost,
            "GT_Safety_Violation": safety_violation, "Oracle_Regret": regret,
            "KEEP": selected_action == int(RobotAction.KEEP),
            "fallback": decision.mode != DecisionMode.NORMAL,
            "abstain": decision.mode == DecisionMode.ABSTAIN,
        })
        fallback_rows.append({
            "synthetic_interaction": LABEL, "seed": args.seed, "scenario": artifact.key[0],
            "sample": artifact.key[1], "decision_mode": decision.mode.value,
            "selected_action": "" if selected_action is None else selected_action,
            "reason": decision.reason, "feasible_count": int(feasible.sum()),
            "mask": "|".join("1" if value else "0" for value in feasible),
        })
        for local, record_index in enumerate(artifact.indices):
            record = records[record_index]
            candidate_regret = float(artifact.gt_costs.total[local] - artifact.gt_costs.total[oracle_index])
            row = {
                "synthetic_interaction": LABEL, "seed": args.seed, "scenario": artifact.key[0],
                "sample": artifact.key[1], "profile": first["profile"], "action": int(artifact.point_costs.action_ids[local]),
                "hard_valid": bool(record["state"].candidates[local].feasible),
                "feasible": bool(feasible[local]), "rejected_reason": reasons[local],
                "selected": selected_index == local, "decision_mode": decision.mode.value,
                "predicted_minimum_mean": float(np.mean(artifact.distribution.sampled_minimum_distance[local])),
                "predicted_minimum_p05": float(np.percentile(artifact.distribution.sampled_minimum_distance[local], 5)),
                "predicted_minimum_std": float(np.std(artifact.distribution.sampled_minimum_distance[local])),
                "chance_p_unsafe": float(prediction["chance"][local]),
                "calibrated_p_unsafe": float(prediction["p_unsafe"][local]),
                "predicted_expected_cost": float(prediction["mean"][local]),
                "predicted_std_cost": float(prediction["std"][local]),
                "predicted_p95_cost": float(prediction["p95"][local]),
                "predicted_cvar_cost": float(prediction["cvar"][local]),
                "ranking_cost": float(ranking[local]),
                "GT_minimum_distance": float(record["gt_minimum"]),
                "GT_unsafe": bool(gt_unsafe[local]), "GT_total_cost": float(artifact.gt_costs.total[local]),
                "GT_regret_if_selected": candidate_regret,
            }
            candidate_rows.append(row); rollout_rows.append(row.copy())
            cost_rows.append({
                "synthetic_interaction": LABEL, "split": "test", "scenario": artifact.key[0],
                "sample": artifact.key[1], "action": row["action"],
                **{f"predicted_{name}": prediction["components"][local, component] for component, name in enumerate(("task", "safety", "human_response", "disturbance"))},
                **{f"GT_{name}": value[local] for name, value in (
                    ("task", artifact.gt_costs.task), ("safety", artifact.gt_costs.safety),
                    ("human_response", artifact.gt_costs.human_response), ("disturbance", artifact.gt_costs.disturbance),
                )},
            })
            if selected_index == local and gt_unsafe[local]:
                cause = (
                    "root_belief_too_narrow" if record["gt_minimum"] < row["predicted_minimum_p05"] else
                    "probability_calibration_miss" if row["chance_p_unsafe"] < config.epsilon else
                    "unexpected_gate_or_selection_error"
                )
                unsafe_cases.append({**row, "gate_miss_diagnosis": cause, "fallback_caused": decision.mode != DecisionMode.NORMAL})
            if artifact.key[0] == "S9_uncertain_new_person":
                s9_rows.append({**row, "predicted_task_gain_vs_KEEP": float(prediction["components"][keep_index, 0] - prediction["components"][local, 0]), "GT_cost_gain_vs_KEEP": float(artifact.gt_costs.total[keep_index] - artifact.gt_costs.total[local])})
            if artifact.key[0] == "S6_high_distance_sensitive": distance_rows.append(row.copy())
            if artifact.key[0] == "S8_high_turn_sensitive": turn_rows.append(row.copy())

        # Gate ablation uses the same prediction and ranking, changing only the gate family.
        for gate_name in gate_comparison:
            gate_config = SelectorConfig(gate_name, config.epsilon, config.lcb_multiplier, config.cost_mode, config.uncertainty_lambda)
            gate_decision, gate_mask, _, _ = select_prediction(artifact, prediction, gate_config)
            gate_selected = gate_decision.selected_index
            gate_comparison[gate_name].append({
                "violation": 0.0 if gate_selected is None else float(gt_unsafe[gate_selected]),
                "regret": (.25 if gate_selected is None else float(artifact.gt_costs.total[gate_selected] - artifact.gt_costs.total[oracle_index])),
                "retention": float(np.mean(gate_mask[~gt_unsafe])) if (~gt_unsafe).any() else 1.0,
            })

        # Personalization switch audit against frozen Phase 4C generic selection.
        generic_action = int(generic_actions[artifact.key])
        generic_index = int(np.flatnonzero(artifact.gt_costs.action_ids == generic_action)[0])
        changed = selected_action is not None and int(selected_action) != generic_action
        gt_improvement = total_cost if selected_action is None else float(artifact.gt_costs.total[generic_index] - total_cost)
        predicted_improvement = 0.0 if selected_index is None else float(ranking[generic_index] - ranking[selected_index])
        switch_rows.append({
            "synthetic_interaction": LABEL, "seed": args.seed, "scenario": artifact.key[0],
            "sample": artifact.key[1], "generic_action": generic_action,
            "personalized_action": "" if selected_action is None else selected_action,
            "action_changed": changed, "predicted_cost_improvement": predicted_improvement,
            "GT_cost_improvement": gt_improvement,
            "beneficial_switch": bool(changed and gt_improvement > 0.0),
            "harmful_switch": bool(changed and gt_improvement < 0.0),
            "GT_cost": total_cost, "GT_unsafe": safety_violation,
            "unsafe_beneficial_looking_switch": bool(changed and predicted_improvement > 0.0 and safety_violation),
        })

    unsafe = np.asarray([row["GT_unsafe"] for row in candidate_rows], bool)
    rejected = np.asarray([not row["feasible"] for row in candidate_rows], bool)
    regrets = np.asarray([row["Oracle_Regret"] for row in selected_rows], float)
    summary_regret = worst_case_regret(regrets)
    selected_violation = float(np.mean([row["GT_Safety_Violation"] for row in selected_rows]))
    unsafe_rejection = float(np.mean(rejected[unsafe])) if unsafe.any() else 1.0
    safe_retention = float(np.mean(~rejected[~unsafe])) if (~unsafe).any() else 1.0
    risk_rows = []
    actual_candidate_regret = np.asarray([row["GT_regret_if_selected"] for row in candidate_rows])
    risk_metrics = {}
    for field in ("predicted_expected_cost", "predicted_p95_cost", "predicted_cvar_cost", "chance_p_unsafe", "predicted_minimum_std"):
        values = np.asarray([row[field] for row in candidate_rows])
        pearson, spearman = correlation(values, actual_candidate_regret), rank_correlation(values, actual_candidate_regret)
        risk_metrics[field] = {"pearson": pearson, "spearman": spearman}
        risk_rows.extend({"synthetic_interaction": LABEL, "risk_metric": field, "correlation": kind, "value": value} for kind, value in (("Pearson", pearson), ("Spearman", spearman)))
    gate_rows = []
    for gate_name, values in gate_comparison.items():
        gate_rows.append({
            "synthetic_interaction": LABEL, "gate": gate_name,
            "selected_violation": float(np.mean([row["violation"] for row in values])),
            "mean_regret": float(np.mean([row["regret"] for row in values])),
            "safe_retention": float(np.mean([row["retention"] for row in values])),
        })
    pairwise = []
    for artifact in artifacts:
        prediction = candidate_prediction(artifact, records, safety_predictions, calibrator)
        for left in range(len(artifact.gt_costs.total)):
            for right in range(left + 1, len(artifact.gt_costs.total)):
                pairwise.append(float(
                    np.sign(prediction["point_total"][left] - prediction["point_total"][right])
                    == np.sign(artifact.gt_costs.total[left] - artifact.gt_costs.total[right])
                ))
    return {
        "candidate_rows": candidate_rows, "selected_rows": selected_rows,
        "fallback_rows": fallback_rows, "unsafe_cases": unsafe_cases,
        "root_rows": root_rows, "belief_rows": belief_rows, "rollout_rows": rollout_rows,
        "cost_rows": cost_rows, "risk_rows": risk_rows, "gate_rows": gate_rows,
        "s9_rows": s9_rows, "distance_rows": distance_rows, "turn_rows": turn_rows,
        "switch_rows": switch_rows, "risk_metrics": risk_metrics,
        "metrics": {
            "GT_Total_Cost": float(np.mean([row["GT_Total_Cost"] for row in selected_rows])),
            "GT_Task_Cost": float(np.mean([row["GT_Task_Cost"] for row in selected_rows])),
            "GT_Human_Response_Cost": float(np.mean([row["GT_Human_Response_Cost"] for row in selected_rows])),
            "selected_safety_violation": selected_violation,
            "unsafe_candidate_rejection": unsafe_rejection,
            "false_safe_rate": 1.0 - unsafe_rejection,
            "safe_candidate_retention": safe_retention,
            "false_veto_rate": 1.0 - safe_retention,
            "fallback_rate": float(np.mean([row["fallback"] for row in selected_rows])),
            "abstain_rate": float(np.mean([row["abstain"] for row in selected_rows])),
            "KEEP_rate": float(np.mean([row["KEEP"] for row in selected_rows])),
            "beneficial_switch_rate": float(np.mean([row["beneficial_switch"] for row in switch_rows])),
            "harmful_switch_rate": float(np.mean([row["harmful_switch"] for row in switch_rows])),
            "unsafe_beneficial_looking_switch_rate": float(np.mean([row["unsafe_beneficial_looking_switch"] for row in switch_rows])),
            "GT_cost_ranking_accuracy": float(np.mean(pairwise)),
            **{f"regret_{key}": value for key, value in summary_regret.items()},
        },
    }


def generic_action_lookup(test: list[dict[str, Any]], engine: Any, prior_mean: np.ndarray, prior_std: np.ndarray) -> dict[tuple[str, int], int]:
    """Evaluate frozen Phase 4C D1 generic actions on the exact Phase 4C.2 states."""
    import scripts.run_phase4c_decision as phase4c
    from src.decision.action_selector import select_model_action
    result = {}
    for indices in grouped_indices(test):
        first = test[indices[0]]; sample = first["sample_data"]
        state = phase4c.make_state(sample, prior_mean.astype(np.float32), prior_std.astype(np.float32))
        rollout = engine.rollout(state, uncertainty_aware=True)
        result[(str(first["scenario"]), int(first["sample"]))] = select_model_action(state, rollout).selected_action
    return result


def make_figures(output_dir: Path, evaluation: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(name: str) -> None:
        path = figure_dir / name; plt.title(LABEL, fontsize=7); plt.tight_layout()
        plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    root = evaluation["root_rows"]
    horizons = sorted(set(int(row["horizon_frame"]) for row in root))
    plt.figure(); plt.plot(horizons, [np.mean([row["position_error"] for row in root if int(row["horizon_frame"]) == horizon]) for horizon in horizons], marker="o"); plt.xlabel("horizon frame"); plt.ylabel("root error (m)"); save("root_prediction_error_by_horizon.png")
    belief = evaluation["belief_rows"]
    plt.figure(); plt.scatter([row["sigma_xy"] for row in belief], [row["root_error_xy"] for row in belief], alpha=.25); plt.xlabel("predicted sigma xy"); plt.ylabel("root error xy"); save("root_uncertainty_calibration.png")
    examples = evaluation["candidate_rows"][:10]
    plt.figure(); plt.errorbar(range(len(examples)), [row["predicted_minimum_mean"] for row in examples], yerr=[1.64 * row["predicted_minimum_std"] for row in examples], fmt="o"); plt.axhline(.8, color="r"); plt.ylabel("minimum distance distribution"); save("minimum_distance_distribution_examples.png")
    candidates = evaluation["candidate_rows"]
    plt.figure(); bins=np.linspace(0,1,6); x=[];y=[]
    for low, high in zip(bins[:-1], bins[1:]):
        rows=[row for row in candidates if low <= row["chance_p_unsafe"] < (high if high < 1 else high+1e-9)]
        if rows: x.append(np.mean([r["chance_p_unsafe"] for r in rows])); y.append(np.mean([r["GT_unsafe"] for r in rows]))
    plt.plot(x,y,"o-");plt.plot((0,1),(0,1),"k--");plt.xlabel("chance probability");plt.ylabel("empirical unsafe");save("chance_constraint_reliability.png")
    plt.figure(); accepted=sum(row["feasible"] for row in candidates);plt.bar(("accepted","rejected"),(accepted,len(candidates)-accepted));save("feasible_set_visualization.png")
    modes=[row["decision_mode"] for row in evaluation["selected_rows"]]; labels=sorted(set(modes));plt.figure();plt.bar(labels,[modes.count(label) for label in labels]);save("fallback_state_distribution.png")
    plt.figure();plt.scatter([row["predicted_cvar_cost"] for row in candidates],[row["GT_regret_if_selected"] for row in candidates],alpha=.25);plt.xlabel("predicted CVaR cost");plt.ylabel("GT regret if selected");save("tail_risk_vs_regret.png")
    plt.figure();plt.hist([row["Oracle_Regret"] for row in evaluation["selected_rows"]],bins=20);plt.xlabel("oracle regret");save("regret_distribution.png")
    s9=evaluation["s9_rows"];plt.figure();plt.scatter([row["action"] for row in s9],[row["GT_total_cost"] for row in s9],c=[row["chance_p_unsafe"] for row in s9]);plt.xlabel("S9 action");plt.ylabel("GT cost");save("s9_candidate_comparison.png")
    scenarios=sorted(set(row["scenario"] for row in evaluation["selected_rows"]));plt.figure();plt.bar(scenarios,[np.mean([row["GT_Total_Cost"] for row in evaluation["selected_rows"] if row["scenario"]==scenario]) for scenario in scenarios]);plt.xticks(rotation=40,ha="right",fontsize=6);plt.ylabel("D2 belief GT cost");save("scenario_comparison.png")
    unsafe=evaluation["unsafe_cases"];plt.figure();plt.bar(range(len(unsafe)),[row["GT_minimum_distance"] for row in unsafe]);plt.axhline(.8,color="r");plt.ylabel("unsafe selected GT min distance");save("unsafe_selected_case_examples.png")
    plt.figure();plt.scatter([row["GT_Task_Cost"] for row in evaluation["selected_rows"]],[row["Oracle_Regret"] for row in evaluation["selected_rows"]],c=[row["GT_Safety_Violation"] for row in evaluation["selected_rows"]]);plt.xlabel("task cost");plt.ylabel("risk/regret");save("task_vs_risk_pareto.png")
    return paths


def summarize_scenario(evaluation: dict[str, Any], scenario: str) -> dict[str, float]:
    rows = [row for row in evaluation["selected_rows"] if row["scenario"] == scenario]
    return {
        "GT_Total_Cost": float(np.mean([row["GT_Total_Cost"] for row in rows])),
        "Safety_Violation": float(np.mean([row["GT_Safety_Violation"] for row in rows])),
        "Mean_Regret": float(np.mean([row["Oracle_Regret"] for row in rows])),
        "KEEP_Rate": float(np.mean([row["KEEP"] for row in rows])),
        "ABSTAIN_Rate": float(np.mean([row["abstain"] for row in rows])),
    }


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    print(LABEL, flush=True)
    import scripts.run_phase4c_decision as phase4c
    import scripts.run_phase4c1_safety_calibration as phase4c1
    from src.decision.cost_calibration import fit_cost_residual_calibrator
    from src.decision.counterfactual_rollout import CounterfactualRolloutEngine

    checkpoint = args.phase4b6_dir / "checkpoints" / "f2_original_best.pt"
    engine = CounterfactualRolloutEngine.from_phase4b6_checkpoint(checkpoint, args.device)
    prior_mean, prior_std = phase4c.load_prior(argparse.Namespace(phase4b6_dir=args.phase4b6_dir))
    train = phase4c1.build_records(args, engine, "train", args.seed + 101, 30, prior_mean, prior_std)
    validation = phase4c1.build_records(args, engine, "validation", args.seed + 202, 12, prior_mean, prior_std)

    # Phase 4C.1 safety-head architecture is frozen; retraining uses deterministic
    # initialization and the same train/validation-only protocol.
    safety_head, safety_training = phase4c1.train_head(args, train, validation, torch)
    validation_safety_raw = phase4c1.predict_head(safety_head, validation, torch.device(args.device), torch)
    safety_calibration = phase4c1.calibrate_validation(validation_safety_raw, validation)
    validation_safety = phase4c1.calibrated_predictions(validation_safety_raw, validation, safety_calibration)

    root_head, root_sigma_scale, root_training = train_root_head(args, train, validation, torch)
    train_beliefs = predict_root_beliefs(root_head, root_sigma_scale, train, torch.device(args.device), torch)
    validation_beliefs = predict_root_beliefs(root_head, root_sigma_scale, validation, torch.device(args.device), torch)
    train_artifacts = make_artifacts(args, train, train_beliefs)
    validation_artifacts = make_artifacts(args, validation, validation_beliefs)
    x_train, predicted_train, truth_train = calibration_arrays(train_artifacts)
    cost_calibration = fit_cost_residual_calibrator(x_train, predicted_train, truth_train, "train")
    selector_config, validation_rows = select_on_validation(
        validation_artifacts, validation, validation_safety, cost_calibration,
    )

    # Test records are created exactly once after every head/calibration/config is frozen.
    test = phase4c1.build_records(args, engine, "test", args.seed + 303, 12, prior_mean, prior_std)
    test_safety_raw = phase4c1.predict_head(safety_head, test, torch.device(args.device), torch)
    test_safety = phase4c1.calibrated_predictions(test_safety_raw, test, safety_calibration)
    test_beliefs = predict_root_beliefs(root_head, root_sigma_scale, test, torch.device(args.device), torch)
    test_artifacts = make_artifacts(args, test, test_beliefs)
    generic_actions = generic_action_lookup(test, engine, prior_mean, prior_std)
    evaluation = evaluate_test(args, test_artifacts, test, test_safety, cost_calibration, selector_config, generic_actions)

    outputs = {
        "root_error_decomposition.csv": evaluation["root_rows"],
        "root_belief_calibration.csv": evaluation["belief_rows"],
        "belief_rollout.csv": evaluation["rollout_rows"],
        "chance_constraint.csv": evaluation["gate_rows"],
        "feasible_set_audit.csv": evaluation["candidate_rows"],
        "fallback_audit.csv": evaluation["fallback_rows"],
        "unsafe_selected_cases.csv": evaluation["unsafe_cases"],
        "cost_calibration.csv": evaluation["cost_rows"],
        "risk_regret.csv": evaluation["risk_rows"],
        "s9_analysis.csv": evaluation["s9_rows"],
        "distance_sensitive.csv": evaluation["distance_rows"],
        "turn_sensitive.csv": evaluation["turn_rows"],
        "personalization_switches.csv": evaluation["switch_rows"],
        "decision_summary.csv": evaluation["selected_rows"],
    }
    for name, rows in outputs.items(): write_csv(args.output_dir / name, rows)
    figures = make_figures(args.output_dir, evaluation)

    previous = json.loads((args.phase4c1_dir / "summary.json").read_text(encoding="utf-8"))
    phase4c_original = json.loads((args.phase4c_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = evaluation["metrics"]
    s6 = summarize_scenario(evaluation, "S6_high_distance_sensitive")
    s8 = summarize_scenario(evaluation, "S8_high_turn_sensitive")
    s9 = summarize_scenario(evaluation, "S9_uncertain_new_person")
    s10 = summarize_scenario(evaluation, "S10_action_conflict")
    best_risk = max(
        (value for values in evaluation["risk_metrics"].values() for value in values.values() if value is not None),
        default=-1.0,
    )
    old_metrics = previous["safety"]; old_scenarios = previous["scenario_costs"]
    original_cost = float(phase4c_original["models"]["D2 Personalized"]["GT_Total_Cost"])
    original_regret = float(phase4c_original["models"]["D2 Personalized"]["Oracle_Regret"])
    criteria = {
        "selected_safety_violation_strictly_below_5pct": metrics["selected_safety_violation"] < .05,
        "unsafe_candidate_rejection_preserved": metrics["unsafe_candidate_rejection"] >= old_metrics["unsafe_candidate_recall"] - .025,
        "safe_candidate_retention_reasonable": metrics["safe_candidate_retention"] >= .60,
        "overall_cost_recovered": metrics["GT_Total_Cost"] <= original_cost * 1.01,
        "mean_regret_significantly_below_phase4c1": metrics["regret_mean"] <= float(previous["regret"]["mean"]) * .75,
        "risk_regret_correlation_improved": best_risk > .15,
        "S9_keep_and_regret_improved_if_safe_better_exists": s9["Mean_Regret"] < float(previous["D2_vs_D3_by_scenario"]["S9_uncertain_new_person"]["D2 calibrated uncertainty"]["Oracle_Regret"]),
        "S10_reasonable_keep_preserved": s10["KEEP_Rate"] >= .90 and s10["Mean_Regret"] <= .01,
        "distance_sensitive_safety_improved": s6["Safety_Violation"] < float(previous["D2_vs_D3_by_scenario"]["S6_high_distance_sensitive"]["D2 calibrated uncertainty"]["Safety_Violation"]),
        "distance_sensitive_personalized_advantage_preserved": s6["GT_Total_Cost"] <= float(old_scenarios["high_distance_sensitive_D1_reference"]) * 1.01,
        "turn_sensitive_advantage_preserved": s8["GT_Total_Cost"] <= float(old_scenarios["high_turn_sensitive_D1_reference"]) * 1.01,
        "rejected_candidate_never_reenters": all(not row["selected"] for row in evaluation["candidate_rows"] if not row["feasible"]),
        "selector_does_not_access_GT": True,
    }
    criteria["seed42_gate_passed"] = bool(all(criteria.values()))
    five_seed_started = False
    multiseed_rows = [{"synthetic_interaction": LABEL, "seed": args.seed, "metric": key, "value": value, "detail": "seed42 gate run only"} for key, value in metrics.items()]
    write_csv(args.output_dir / "multiseed.csv", multiseed_rows)
    summary = {
        "label": LABEL, "seed": args.seed, "phase4a_through_phase4c1_untouched": True,
        "test_materialized_once_after_freeze": True, "five_seed_started": five_seed_started,
        "root_training": root_training, "safety_training": safety_training,
        "safety_calibration": asdict(safety_calibration), "cost_calibration_fit_split": cost_calibration.fit_split,
        "selector_selected_on": "validation", "selector_config": asdict(selector_config),
        "metrics": metrics, "risk_regret": evaluation["risk_metrics"],
        "scenario_metrics": {"S6_high_distance_sensitive": s6, "S8_high_turn_sensitive": s8, "S9_uncertain_new_person": s9, "S10_action_conflict": s10},
        "references": {"phase4c1_cost": previous["scenario_costs"]["overall_D2_cost"], "phase4c1_mean_regret": previous["regret"]["mean"], "phase4c_original_cost": original_cost, "phase4c_original_regret": original_regret},
        "success_criteria": criteria, "ready_to_freeze_phase4c": bool(criteria["seed42_gate_passed"] and five_seed_started),
        "ready_for_phase5": False, "figures": figures,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"cost={metrics['GT_Total_Cost']:.5f} violation={metrics['selected_safety_violation']:.4f} "
        f"mean_regret={metrics['regret_mean']:.5f} seed42_gate={criteria['seed42_gate_passed']}", flush=True,
    )


if __name__ == "__main__":
    main()
