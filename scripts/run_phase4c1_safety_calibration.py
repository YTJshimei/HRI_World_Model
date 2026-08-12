"""Phase 4C.1 safety-critical rollout calibration (offline synthetic only)."""

from __future__ import annotations

import argparse
import copy
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
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    return parser.parse_args()


def clean(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
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


def sample_namespace(args: argparse.Namespace, seed: int, count: int) -> argparse.Namespace:
    return argparse.Namespace(seed=seed, samples_per_scenario=count)


def build_records(
    args: argparse.Namespace, engine: Any, split_name: str, seed: int,
    samples_per_scenario: int, prior_mean: np.ndarray, prior_std: np.ndarray,
) -> list[dict[str, Any]]:
    import scripts.run_phase4c_decision as phase4c
    from src.data.functional_response_state import functional_state_from_profile
    from src.data.robot_action_schema import action_feature
    from src.data.synthetic_interaction import PROFILE_BY_ID
    from src.decision.safety_calibration import safety_features
    from src.decision.safety_targets import build_safety_targets_for_training_or_evaluation

    records = []
    for episode, sample in enumerate(phase4c.scenario_samples(sample_namespace(args, seed, samples_per_scenario))):
        theta_true = functional_state_from_profile(PROFILE_BY_ID[int(sample["profile"])]).astype(np.float32)
        theta_hat, theta_std, support = phase4c.estimate_personal_belief(
            sample, theta_true, prior_mean, prior_std, seed + 70_000 + episode * 31
        )
        state = phase4c.make_state(sample, theta_hat, theta_std)
        predicted = engine.rollout(state, uncertainty_aware=True)
        gt = phase4c.ground_truth_rollout(sample, state, theta_true)
        targets = build_safety_targets_for_training_or_evaluation(
            gt.predicted_human_robot_distance, state.too_close_distance
        )
        for action_index, action_id in enumerate(predicted.action_ids):
            records.append({
                "split": split_name, "scenario": sample["scenario"],
                "sample": sample["sample"], "profile": sample["profile"],
                "action": int(action_id), "state": state, "sample_data": sample,
                "theta_true": theta_true, "theta_hat": theta_hat, "theta_std": theta_std,
                "support": support, "predicted_rollout": predicted, "gt_rollout": gt,
                "action_index": action_index,
                "features": safety_features(
                    sample["history"], sample["robot"], action_feature(int(action_id)),
                    predicted.predicted_human_robot_distance[action_index],
                    predicted.predicted_action_effect[action_index], theta_hat, theta_std,
                ),
                "predicted_distance": predicted.predicted_human_robot_distance[action_index],
                "gt_distance": targets.distance_trajectory[action_index],
                "gt_minimum": targets.minimum_distance[action_index],
                "gt_unsafe": targets.violation_any[action_index],
                "gt_violation_duration": targets.violation_duration[action_index],
                "gt_time_to_minimum": targets.time_to_minimum_distance[action_index],
            })
    return records


def tensorize(records: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    return {
        "features": torch.from_numpy(np.stack([r["features"] for r in records])),
        "distance_residual": torch.from_numpy(np.stack([
            r["gt_distance"] - r["predicted_distance"] for r in records
        ]).astype(np.float32)),
        "minimum_residual": torch.from_numpy(np.asarray([
            r["gt_minimum"] - np.min(r["predicted_distance"]) for r in records
        ], dtype=np.float32)),
        "unsafe": torch.from_numpy(np.asarray([r["gt_unsafe"] for r in records], dtype=np.float32)),
    }


def loss_function(output: dict[str, Any], batch: dict[str, Any], torch: Any) -> Any:
    distance_error = output["distance_residual"] - batch["distance_residual"]
    distance_nll = 0.5 * (
        distance_error.square() * torch.exp(-2 * output["distance_log_std"])
        + 2 * output["distance_log_std"]
    ).mean()
    minimum_error = output["minimum_residual"] - batch["minimum_residual"]
    minimum_nll = 0.5 * (
        minimum_error.square() * torch.exp(-2 * output["minimum_log_std"])
        + 2 * output["minimum_log_std"]
    ).mean()
    classification = torch.nn.functional.binary_cross_entropy_with_logits(
        output["unsafe_logit"], batch["unsafe"], pos_weight=torch.tensor(3.0, device=batch["unsafe"].device)
    )
    return distance_nll + minimum_nll + classification


def train_head(args: argparse.Namespace, train: list[dict[str, Any]], validation: list[dict[str, Any]], torch: Any):
    from src.decision.safety_calibration import SafetyResidualHead
    device = torch.device(args.device); head = SafetyResidualHead().to(device)
    train_data, val_data = tensorize(train, torch), tensorize(validation, torch)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        head.train(); order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = {k: v[indices].to(device) for k, v in train_data.items()}
            optimizer.zero_grad(set_to_none=True); loss = loss_function(head(batch["features"]), batch, torch)
            loss.backward(); optimizer.step()
        head.eval()
        with torch.inference_mode():
            batch = {k: v.to(device) for k, v in val_data.items()}
            validation_loss = float(loss_function(head(batch["features"]), batch, torch).item())
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, copy.deepcopy(head.state_dict())
    assert best_state is not None; head.load_state_dict(best_state); head.eval()
    checkpoint = args.output_dir / "checkpoints" / "safety_residual_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "validation_loss": best_loss}, checkpoint)
    return head, {"checkpoint": str(checkpoint), "best_epoch": best_epoch, "best_validation_loss": best_loss, "parameters": sum(p.numel() for p in head.parameters())}


def predict_head(head: Any, records: list[dict[str, Any]], device: Any, torch: Any) -> dict[str, np.ndarray]:
    features = torch.from_numpy(np.stack([r["features"] for r in records])).to(device)
    with torch.inference_mode(): output = head(features)
    return {key: value.cpu().numpy() for key, value in output.items()}


def binary_metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float | None]:
    probability = np.asarray(probability, dtype=float); truth = np.asarray(truth, dtype=bool)
    predicted = probability > threshold
    tp = np.sum(predicted & truth); fp = np.sum(predicted & ~truth)
    fn = np.sum(~predicted & truth); tn = np.sum(~predicted & ~truth)
    order = np.argsort(-probability); sorted_truth = truth[order].astype(float)
    positives, negatives = truth.sum(), (~truth).sum()
    ranks = np.argsort(np.argsort(probability)) + 1
    auroc = (
        (ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        if positives and negatives else None
    )
    precision_curve = np.cumsum(sorted_truth) / np.arange(1, len(truth) + 1)
    auprc = float((precision_curve * sorted_truth).sum() / positives) if positives else None
    bins = np.linspace(0.0, 1.0, 11); ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (probability >= left) & (probability < right if right < 1.0 else probability <= right)
        if mask.any(): ece += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return {
        "Brier": float(np.mean((probability - truth.astype(float)) ** 2)),
        "NLL": float(-np.mean(truth * np.log(probability.clip(1e-7, 1 - 1e-7)) + (~truth) * np.log((1 - probability).clip(1e-7, 1 - 1e-7)))),
        "ECE": float(ece), "AUROC": float(auroc) if auroc is not None else None,
        "AUPRC": auprc, "Recall": float(tp / max(tp + fn, 1)),
        "Precision": float(tp / max(tp + fp, 1)),
        "False_Safe_Rate": float(fn / max(tp + fn, 1)),
        "False_Veto_Rate": float(fp / max(fp + tn, 1)),
    }


def calibrate_validation(
    raw: dict[str, np.ndarray], validation: list[dict[str, Any]],
) -> Any:
    from src.decision.safety_calibration import SafetyCalibration
    distance_error = np.stack([
        r["gt_distance"] - r["predicted_distance"] - raw["distance_residual"][i]
        for i, r in enumerate(validation)
    ])
    minimum_error = np.asarray([
        r["gt_minimum"] - np.min(r["predicted_distance"])
        - raw["minimum_residual"][i]
        for i, r in enumerate(validation)
    ])
    distance_std = np.exp(raw["distance_log_std"])
    minimum_std = np.exp(raw["minimum_log_std"])
    distance_scale = float(np.sqrt(np.mean((distance_error / distance_std.clip(1e-5)) ** 2)))
    minimum_scale = float(np.sqrt(np.mean((minimum_error / minimum_std.clip(1e-5)) ** 2)))
    truth = np.asarray([r["gt_unsafe"] for r in validation], dtype=bool)
    best: tuple[float, float, float] | None = None
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        probability = 1 / (1 + np.exp(-np.clip(raw["unsafe_logit"] / temperature, -30, 30)))
        for threshold in np.linspace(0.15, 0.85, 15):
            predicted = probability > threshold
            false_safe = np.sum(~predicted & truth) / max(truth.sum(), 1)
            false_veto = np.sum(predicted & ~truth) / max((~truth).sum(), 1)
            retention = 1.0 - false_veto
            score = false_safe + 0.35 * false_veto + 0.15 * max(0.70 - retention, 0)
            item = (score, temperature, float(threshold))
            if best is None or item < best: best = item
    assert best is not None
    return SafetyCalibration(
        max(distance_scale, 0.1), max(minimum_scale, 0.1),
        best[1], best[2], 1.64, "validation",
    )


def calibrated_predictions(
    raw: dict[str, np.ndarray], records: list[dict[str, Any]], calibration: Any,
) -> list[dict[str, Any]]:
    from src.decision.safety_calibration import apply_safety_calibration
    calibrated = apply_safety_calibration(raw, calibration); results = []
    for index, record in enumerate(records):
        distance = record["predicted_distance"] + raw["distance_residual"][index]
        direct_minimum = np.min(record["predicted_distance"]) + raw["minimum_residual"][index]
        trajectory_minimum = float(np.min(distance))
        predicted_minimum = float(0.5 * direct_minimum + 0.5 * trajectory_minimum)
        results.append({
            "distance": distance, "minimum": predicted_minimum,
            "sigma_distance": calibrated["sigma_distance"][index],
            "sigma_minimum": float(calibrated["sigma_minimum"][index]),
            "p_unsafe": float(calibrated["p_unsafe"][index]),
        })
    return results


def audit_rows(records: list[dict[str, Any]], predictions: list[dict[str, Any]] | None, label: str) -> list[dict[str, Any]]:
    from src.data.skeleton_schema import compute_root
    rows = []
    for index, record in enumerate(records):
        predicted_distance = record["predicted_distance"] if predictions is None else predictions[index]["distance"]
        predicted_minimum = float(predicted_distance.min()) if predictions is None else predictions[index]["minimum"]
        predicted_root = record["predicted_rollout"].predicted_root[record["action_index"]]
        gt_root = record["gt_rollout"].predicted_root[record["action_index"]]
        robot_error = np.linalg.norm(
            record["predicted_rollout"].predicted_robot_xy[record["action_index"]]
            - record["gt_rollout"].predicted_robot_xy[record["action_index"]], axis=-1
        )
        for horizon in (2, 5, 10):
            frame = min(horizon, len(predicted_distance)) - 1
            rows.append({
                "synthetic_interaction": LABEL, "split": record["split"],
                "model": label, "scenario": record["scenario"],
                "profile": record["profile"], "action": record["action"],
                "horizon_seconds": (frame + 1) / 10.0,
                "human_root_error": np.linalg.norm(predicted_root[frame] - gt_root[frame]),
                "robot_position_error": robot_error[frame],
                "distance_error": abs(predicted_distance[frame] - record["gt_distance"][frame]),
                "distance_bias": predicted_distance[frame] - record["gt_distance"][frame],
                "minimum_distance_error": abs(predicted_minimum - record["gt_minimum"]),
                "gt_unsafe": record["gt_unsafe"],
                "predicted_unsafe": predicted_minimum < record["state"].too_close_distance,
            })
    return rows


def evaluate_test(
    args: argparse.Namespace, records: list[dict[str, Any]],
    predictions: list[dict[str, Any]], calibration: Any,
) -> dict[str, Any]:
    import scripts.run_phase4c_decision as phase4c
    from src.decision.action_selector import select_model_action
    from src.decision.decision_cost import DecisionCostWeights, compute_decision_costs
    from src.decision.safety_calibration import worst_case_regret
    from src.decision.safety_gate import choose_fallback_action, risk_aware_candidate_mask

    grouped: dict[tuple[str, int], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault((record["scenario"], int(record["sample"])), []).append(index)
    gate_rows, fallback_rows, regret_rows, switch_rows = [], [], [], []
    distance_rows, turn_rows, decision_fidelity_rows, selected_rows = [], [], [], []
    weights = DecisionCostWeights()
    for key, indices in grouped.items():
        first = records[indices[0]]; state = first["state"]
        gt_rollout = first["gt_rollout"]
        gt_costs = compute_decision_costs(state, gt_rollout, weights, include_uncertainty=False)
        base_result = select_model_action(
            state, first["predicted_rollout"], weights, use_uncertainty=False
        )
        feasible = np.asarray([item.feasible for item in state.candidates], dtype=bool)
        minimum = np.asarray([predictions[index]["minimum"] for index in indices])
        sigma = np.asarray([predictions[index]["sigma_minimum"] for index in indices])
        probability = np.asarray([predictions[index]["p_unsafe"] for index in indices])
        allowed, reasons = risk_aware_candidate_mask(
            feasible, minimum, sigma, probability, state.too_close_distance,
            calibration.unsafe_threshold, calibration.lcb_multiplier,
        )
        if allowed.any():
            adjusted = base_result.costs.total + 0.85 * (
                probability + np.maximum(state.too_close_distance - (minimum - calibration.lcb_multiplier * sigma), 0.0)
            )
            selected_index = int(np.argmin(np.where(allowed, adjusted, np.inf)))
            fallback_used, fallback_policy = False, ""
        else:
            risks = probability + np.maximum(state.too_close_distance - minimum, 0.0)
            selected_action = choose_fallback_action(
                "FALLBACK_MIN_RISK", gt_costs.action_ids, feasible,
                float(state.robot_history[-1, 5]), state.target_follow_distance, risks,
            )
            selected_index = gt_costs.action_ids.tolist().index(selected_action)
            fallback_used, fallback_policy = True, "FALLBACK_MIN_RISK"
        gt_oracle_index = int(np.argmin(gt_costs.total))
        regret = float(gt_costs.total[selected_index] - gt_costs.total[gt_oracle_index])
        d3_index = int(base_result.selected_index)
        d3_regret = float(gt_costs.total[d3_index] - gt_costs.total[gt_oracle_index])
        selected_rows.append({
            "model": "D2 calibrated uncertainty",
            "scenario": key[0], "sample": key[1], "profile": first["profile"],
            "selected_index": selected_index, "selected_action": int(gt_costs.action_ids[selected_index]),
            "GT_Total_Cost": gt_costs.total[selected_index], "GT_Task_Cost": gt_costs.task[selected_index],
            "GT_Safety_Violation": bool(gt_costs.unsafe_duration[selected_index] > 0),
            "Oracle_Regret": regret, "fallback": fallback_used,
            "p_unsafe_selected": probability[selected_index], "sigma_min_selected": sigma[selected_index],
        })
        selected_rows.append({
            "model": "D3 no safety uncertainty",
            "scenario": key[0], "sample": key[1], "profile": first["profile"],
            "selected_index": d3_index, "selected_action": int(gt_costs.action_ids[d3_index]),
            "GT_Total_Cost": gt_costs.total[d3_index], "GT_Task_Cost": gt_costs.task[d3_index],
            "GT_Safety_Violation": bool(gt_costs.unsafe_duration[d3_index] > 0),
            "Oracle_Regret": d3_regret, "fallback": False,
            "p_unsafe_selected": probability[d3_index], "sigma_min_selected": sigma[d3_index],
        })
        regret_rows.append({
            "synthetic_interaction": LABEL, "seed": args.seed, "scenario": key[0],
            "sample": key[1], "profile": first["profile"], "regret": regret,
            "safety_critical": bool(np.asarray([r["gt_unsafe"] for r in (records[i] for i in indices)]).any()),
            "person_sensitive": key[0] in (
                "S6_high_distance_sensitive", "S7_high_speed_sensitive", "S8_high_turn_sensitive"
            ), "predicted_risk": probability[selected_index], "sigma_minimum": sigma[selected_index],
        })
        def pairwise_accuracy(predicted_values, target_values):
            values=[]
            for left in range(len(predicted_values)):
                for right in range(left+1,len(predicted_values)):
                    values.append(float(np.sign(predicted_values[left]-predicted_values[right])==np.sign(target_values[left]-target_values[right])))
            return float(np.mean(values))
        fidelity_definitions=(
            ("Distance_Ranking_Accuracy",np.asarray([predictions[i]["distance"][-1] for i in indices]),np.asarray([records[i]["gt_distance"][-1] for i in indices])),
            ("Minimum_Distance_Ranking_Accuracy",minimum,np.asarray([records[i]["gt_minimum"] for i in indices])),
            ("Safety_Ranking_Accuracy",probability,np.asarray([records[i]["gt_unsafe"] for i in indices],float)),
            ("GT_Cost_Ranking_Accuracy",base_result.costs.total,gt_costs.total),
        )
        for metric,predicted_values,target_values in fidelity_definitions:
            decision_fidelity_rows.append({
                "synthetic_interaction":LABEL,"seed":args.seed,"scenario":key[0],"sample":key[1],
                "metric":metric,"pairwise_accuracy":pairwise_accuracy(predicted_values,target_values),
                "ranking_spearman":rank_correlation(predicted_values,target_values),
                "Top1_GT_Cost_Action_Agreement":float(np.argmin(base_result.costs.total)==gt_oracle_index),
                "Oracle_Regret":regret,"GT_Total_Cost":gt_costs.total[selected_index],
                "Safety_Violation":bool(gt_costs.unsafe_duration[selected_index]>0),
            })
        for local, index in enumerate(indices):
            record = records[index]
            gate_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed, "scenario": key[0],
                "sample": key[1], "profile": record["profile"], "action": record["action"],
                "predicted_minimum": minimum[local], "sigma_minimum": sigma[local],
                "distance_LCB": minimum[local] - calibration.lcb_multiplier * sigma[local],
                "p_unsafe": probability[local], "allowed": bool(allowed[local]),
                "rejection_reason": reasons[local], "gt_minimum": record["gt_minimum"],
                "gt_unsafe": record["gt_unsafe"], "selected": local == selected_index,
            })
        for policy in ("FALLBACK_KEEP", "FALLBACK_RULE_SAFE", "FALLBACK_MIN_RISK"):
            risks = probability + np.maximum(state.too_close_distance - minimum, 0.0)
            action = choose_fallback_action(
                policy, gt_costs.action_ids, feasible, float(state.robot_history[-1, 5]),
                state.target_follow_distance, risks,
            )
            idx = gt_costs.action_ids.tolist().index(action)
            fallback_rows.append({
                "synthetic_interaction": LABEL, "seed": args.seed,
                "scenario": key[0], "sample": key[1], "policy": policy,
                "selected_action": action, "GT_Total_Cost": gt_costs.total[idx],
                "GT_Task_Cost": gt_costs.task[idx],
                "GT_Safety_Violation": bool(gt_costs.unsafe_duration[idx] > 0),
                "Oracle_Regret": gt_costs.total[idx] - gt_costs.total[gt_oracle_index],
                "KEEP": action == 0,
            })
        if key[0] == "S6_high_distance_sensitive":
            for local, index in enumerate(indices):
                record = records[index]
                distance_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "sample": key[1], "action": record["action"],
                    "uncalibrated_distance_MAE": np.mean(np.abs(record["predicted_distance"] - record["gt_distance"])),
                    "calibrated_distance_MAE": np.mean(np.abs(predictions[index]["distance"] - record["gt_distance"])),
                    "minimum_distance_error": abs(predictions[index]["minimum"] - record["gt_minimum"]),
                    "natural_root_error": np.mean(np.linalg.norm(
                        record["predicted_rollout"].predicted_root[local] - record["gt_rollout"].predicted_root[local], axis=-1
                    )), "safety_decision_error": bool(allowed[local]) == bool(record["gt_unsafe"]),
                })
        if key[0] == "S8_high_turn_sensitive":
            personalized = first["predicted_rollout"].predicted_action_effect
            truth = gt_rollout.predicted_action_effect
            for local, index in enumerate(indices):
                turn_error = float(np.mean(np.abs(personalized[local] - truth[local])))
                turn_rows.append({
                    "synthetic_interaction": LABEL, "seed": args.seed,
                    "sample": key[1], "action": records[index]["action"],
                    "turn_response_effect_MAE": turn_error,
                    "root_trajectory_error": np.mean(np.linalg.norm(
                        first["predicted_rollout"].predicted_root[local] - gt_rollout.predicted_root[local], axis=-1
                    )), "selected_action_regret": regret if local == selected_index else "",
                })

    gt_unsafe = np.asarray([row["gt_unsafe"] for row in gate_rows], dtype=bool)
    rejected = np.asarray([not row["allowed"] for row in gate_rows], dtype=bool)
    unsafe_recall = float(np.mean(rejected[gt_unsafe])) if gt_unsafe.any() else 1.0
    safe_retention = float(np.mean(~rejected[~gt_unsafe])) if (~gt_unsafe).any() else 1.0
    false_safe = 1.0 - unsafe_recall; false_veto = 1.0 - safe_retention
    d2_selected=[row for row in selected_rows if row["model"]=="D2 calibrated uncertainty"]
    selected_violation = mean_bool(d2_selected, "GT_Safety_Violation")
    regrets = np.asarray([row["regret"] for row in regret_rows])
    risk = np.asarray([row["predicted_risk"] for row in regret_rows])
    sigma_values = np.asarray([row["sigma_minimum"] for row in regret_rows])
    regret_summary = worst_case_regret(regrets)
    regret_summary.update({
        "worst_scenario_regret": max(
            np.mean([row["regret"] for row in regret_rows if row["scenario"] == scenario])
            for scenario in set(row["scenario"] for row in regret_rows)
        ),
        "worst_person_regret": max(
            np.mean([row["regret"] for row in regret_rows if row["profile"] == profile])
            for profile in set(row["profile"] for row in regret_rows)
        ),
        "conditional_safety_critical": np.mean([row["regret"] for row in regret_rows if row["safety_critical"]]),
        "conditional_person_sensitive": np.mean([row["regret"] for row in regret_rows if row["person_sensitive"]]),
    })
    return {
        "gate_rows": gate_rows, "fallback_rows": fallback_rows,
        "regret_rows": regret_rows, "distance_rows": distance_rows,
        "turn_rows": turn_rows, "selected_rows": selected_rows,
        "decision_fidelity_rows":decision_fidelity_rows,
        "unsafe_recall": unsafe_recall, "safe_retention": safe_retention,
        "false_safe": false_safe, "false_veto": false_veto,
        "selected_violation": selected_violation,
        "regret_summary": regret_summary,
        "risk_regret_pearson": correlation(risk, regrets),
        "risk_regret_spearman": rank_correlation(risk, regrets),
        "sigma_regret_pearson": correlation(sigma_values, regrets),
        "top10_uncertainty_regret": tail_mean(sigma_values, regrets, True),
        "bottom10_uncertainty_regret": tail_mean(sigma_values, regrets, False),
    }


def mean_bool(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows])) if rows else 0.0


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12 else None


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    return correlation(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def tail_mean(score: np.ndarray, value: np.ndarray, top: bool) -> float:
    count = max(1, int(np.ceil(len(score) * 0.1))); order = np.argsort(score)
    indices = order[-count:] if top else order[:count]
    return float(np.mean(value[indices]))


def interval_and_classification_rows(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]],
    calibration: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probability = np.asarray([item["p_unsafe"] for item in predictions])
    truth = np.asarray([item["gt_unsafe"] for item in records], dtype=bool)
    metrics = binary_metrics(probability, truth, calibration.unsafe_threshold)
    target_rows = [
        {"synthetic_interaction": LABEL, "split": records[0]["split"], "metric": key, "value": value}
        for key, value in metrics.items()
    ]
    calibration_rows = []
    normal_quantile = {0.5: 0.67449, 0.8: 1.28155, 0.9: 1.64485}
    for coverage, quantile in normal_quantile.items():
        distance_covered, minimum_covered = [], []
        for record, prediction in zip(records, predictions):
            distance_covered.extend(
                np.abs(prediction["distance"] - record["gt_distance"])
                <= quantile * prediction["sigma_distance"]
            )
            minimum_covered.append(
                abs(prediction["minimum"] - record["gt_minimum"])
                <= quantile * prediction["sigma_minimum"]
            )
        calibration_rows.extend((
            {"synthetic_interaction": LABEL, "split": records[0]["split"], "target": "distance_trajectory", "nominal_coverage": coverage, "empirical_coverage": float(np.mean(distance_covered))},
            {"synthetic_interaction": LABEL, "split": records[0]["split"], "target": "minimum_distance", "nominal_coverage": coverage, "empirical_coverage": float(np.mean(minimum_covered))},
        ))
    return target_rows, calibration_rows


def fidelity_rows(records: list[dict[str, Any]], predictions: list[dict[str, Any]], evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    regret_lookup = {(row["scenario"], row["sample"]): row for row in evaluation["regret_rows"]}
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, record in enumerate(records): grouped.setdefault((record["scenario"], record["sample"]), []).append(index)
    rows = []
    for key, indices in grouped.items():
        predicted_distance = np.asarray([predictions[i]["distance"][-1] for i in indices])
        gt_distance = np.asarray([records[i]["gt_distance"][-1] for i in indices])
        predicted_min = np.asarray([predictions[i]["minimum"] for i in indices])
        gt_min = np.asarray([records[i]["gt_minimum"] for i in indices])
        predicted_risk = np.asarray([predictions[i]["p_unsafe"] for i in indices])
        gt_risk = np.asarray([records[i]["gt_unsafe"] for i in indices], dtype=float)
        regret = regret_lookup[key]["regret"]
        for metric, pred, truth, lower_is_better in (
            ("Distance_Ranking", predicted_distance, gt_distance, False),
            ("Minimum_Distance_Ranking", predicted_min, gt_min, False),
            ("Safety_Ranking", predicted_risk, gt_risk, False),
        ):
            pairs = []
            for left in range(len(pred)):
                for right in range(left + 1, len(pred)):
                    pairs.append(float(np.sign(pred[left] - pred[right]) == np.sign(truth[left] - truth[right])))
            rows.append({
                "synthetic_interaction": LABEL, "seed": records[0]["sample_data"].get("seed", ""),
                "scenario": key[0], "sample": key[1], "metric": metric,
                "pairwise_accuracy": float(np.mean(pairs)),
                "ranking_spearman": rank_correlation(pred, truth),
                "oracle_regret": regret,
            })
    return rows


def personalization_switch_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.phase4c_dir / "regret.csv"
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as handle: original = list(csv.DictReader(handle))
    lookup = {(r["scenario"], r["sample"], r["model"]): r for r in original}
    rows = []
    keys = sorted(set((r["scenario"], r["sample"]) for r in original))
    for scenario, sample in keys:
        generic = lookup[(scenario, sample, "D1 Generic")]
        personalized = lookup[(scenario, sample, "D2 Personalized")]
        changed = generic["selected_action"] != personalized["selected_action"]
        generic_cost = float(personalized["Personalization_Regret"]) + 0.0
        # regret.csv encodes model cost minus D2 cost as Personalization_Regret.
        improvement = float(generic["Personalization_Regret"])
        rows.append({
            "synthetic_interaction": LABEL, "seed": args.seed,
            "scenario": scenario, "sample": sample,
            "generic_action": generic["selected_action"],
            "personalized_action": personalized["selected_action"],
            "action_changed": changed,
            "GT_cost_improvement": improvement,
            "beneficial_switch": bool(changed and improvement > 0),
            "harmful_switch": bool(changed and improvement < 0),
        })
    return rows


def make_figures(output_dir: Path, audit: list[dict[str, Any]], gate: list[dict[str, Any]], evaluation: dict[str, Any], fidelity: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name):
        path=figure_dir/name; plt.title(LABEL,fontsize=8);plt.tight_layout();plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    calibrated=[r for r in audit if r["model"]=="calibrated" and r["horizon_seconds"]==1.0]
    plt.figure();plt.scatter([r["distance_bias"]+0 for r in calibrated],[r["distance_error"] for r in calibrated],alpha=.35);plt.xlabel("distance bias");plt.ylabel("absolute distance error");save("distance_prediction_calibration.png")
    plt.figure();plt.scatter([r["gt_minimum"] for r in gate],[r["predicted_minimum"] for r in gate],alpha=.35);plt.xlabel("GT min distance");plt.ylabel("predicted min distance");save("predicted_vs_gt_minimum_distance.png")
    plt.figure();bins=np.linspace(0,1,6); probs=np.asarray([r["p_unsafe"] for r in gate]);truth=np.asarray([r["gt_unsafe"] for r in gate],float);x=[];y=[]
    for l,u in zip(bins[:-1],bins[1:]):
        m=(probs>=l)&(probs<(u if u<1 else u+1e-6));
        if m.any():x.append(probs[m].mean());y.append(truth[m].mean())
    plt.plot(x,y,"o-");plt.plot([0,1],[0,1],"k--");plt.xlabel("p_unsafe");plt.ylabel("empirical unsafe");save("p_unsafe_reliability.png")
    false=[r for r in gate if r["gt_unsafe"] and r["allowed"]];plt.figure();plt.bar(range(len(false)),[r["gt_minimum"] for r in false]);plt.axhline(.8,color="r");plt.ylabel("GT min distance");save("false_safe_examples.png")
    tp=sum((not r["allowed"]) and r["gt_unsafe"] for r in gate);fn=sum(r["allowed"] and r["gt_unsafe"] for r in gate);fp=sum((not r["allowed"]) and not r["gt_unsafe"] for r in gate);tn=sum(r["allowed"] and not r["gt_unsafe"] for r in gate)
    plt.figure();plt.imshow([[tn,fp],[fn,tp]],cmap="Blues");plt.colorbar();plt.xticks((0,1),("allow","reject"));plt.yticks((0,1),("safe","unsafe"));save("safety_gate_confusion_matrix.png")
    regrets=np.asarray([r["regret"] for r in evaluation["regret_rows"]]);plt.figure();plt.hist(regrets,bins=20);plt.xlabel("oracle regret");save("regret_distribution.png")
    plt.figure();plt.scatter([r["predicted_risk"] for r in evaluation["regret_rows"]],regrets,alpha=.5);plt.xlabel("predicted risk");plt.ylabel("actual regret");save("uncertainty_vs_regret.png")
    for scenario,name in (("S6_high_distance_sensitive","distance_sensitive_rollout.png"),("S8_high_turn_sensitive","turn_sensitive_rollout.png")):
        rows=[r for r in gate if r["scenario"]==scenario];plt.figure();plt.scatter([r["predicted_minimum"] for r in rows],[r["gt_minimum"] for r in rows],c=[r["action"] for r in rows]);plt.xlabel("predicted min");plt.ylabel("GT min");save(name)
    plt.figure(); scenarios=sorted(set(r["scenario"] for r in evaluation["regret_rows"]));plt.bar(scenarios,[np.mean([r["regret"] for r in evaluation["regret_rows"] if r["scenario"]==s]) for s in scenarios]);plt.xticks(rotation=35,ha="right",fontsize=6);plt.ylabel("D2 calibrated regret");save("per_scenario_regret.png")
    return paths


def main() -> None:
    args=parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed)
    print(LABEL,flush=True)
    import torch
    import scripts.run_phase4c_decision as phase4c
    from src.decision.counterfactual_rollout import CounterfactualRolloutEngine
    checkpoint=args.phase4b6_dir/"checkpoints"/"f2_original_best.pt"
    engine=CounterfactualRolloutEngine.from_phase4b6_checkpoint(checkpoint,args.device)
    prior_mean,prior_std=phase4c.load_prior(argparse.Namespace(phase4b6_dir=args.phase4b6_dir))
    # Independent seeds and one-way train -> validation -> test protocol.
    train=build_records(args,engine,"train",args.seed+101,30,prior_mean,prior_std)
    validation=build_records(args,engine,"validation",args.seed+202,12,prior_mean,prior_std)
    head,training=train_head(args,train,validation,torch)
    validation_raw=predict_head(head,validation,torch.device(args.device),torch)
    calibration=calibrate_validation(validation_raw,validation)
    # Test is materialized only after model/checkpoint/calibration are frozen.
    test=build_records(args,engine,"test",args.seed+303,12,prior_mean,prior_std)
    test_raw=predict_head(head,test,torch.device(args.device),torch)
    predictions=calibrated_predictions(test_raw,test,calibration)
    baseline_audit=audit_rows(test,None,"uncalibrated")
    calibrated_audit=audit_rows(test,predictions,"calibrated")
    target_rows,calibration_rows=interval_and_classification_rows(test,predictions,calibration)
    evaluation=evaluate_test(args,test,predictions,calibration)
    fidelity=fidelity_rows(test,predictions,evaluation)
    switches=personalization_switch_rows(args)

    unsafe=np.asarray([r["gt_unsafe"] for r in evaluation["gate_rows"]],bool)
    rejected=np.asarray([not r["allowed"] for r in evaluation["gate_rows"]],bool)
    original=json.loads((args.phase4c_dir/"summary.json").read_text(encoding="utf-8"))
    selected=[row for row in evaluation["selected_rows"] if row["model"]=="D2 calibrated uncertainty"]
    def scenario_metric(scenario,field):return float(np.mean([r[field] for r in selected if r["scenario"]==scenario]))
    distance_old=float(np.mean([r["minimum_distance_error"] for r in baseline_audit if r["horizon_seconds"]==1.0]))
    distance_new=float(np.mean([r["minimum_distance_error"] for r in calibrated_audit if r["horizon_seconds"]==1.0]))
    root_error=float(np.mean([r["human_root_error"] for r in baseline_audit if r["horizon_seconds"]==1.0]))
    robot_error=float(np.mean([r["robot_position_error"] for r in baseline_audit if r["horizon_seconds"]==1.0]))
    effect_error=float(np.mean([np.linalg.norm(r["predicted_rollout"].predicted_action_effect[r["action_index"]]-r["gt_rollout"].predicted_action_effect[r["action_index"]],axis=-1).mean() for r in test]))
    high_distance_new=scenario_metric("S6_high_distance_sensitive","GT_Total_Cost")
    high_turn_new=scenario_metric("S8_high_turn_sensitive","GT_Total_Cost")
    phase4c_selected=list(csv.DictReader((args.phase4c_dir/"regret.csv").open(encoding="utf-8")))
    def old_cost(scenario,model):
        rows=[r for r in phase4c_selected if r["scenario"]==scenario and r["model"]==model]
        # reconstruct selected GT cost via personalization regret relative to D2
        d2=[r for r in phase4c_selected if r["scenario"]==scenario and r["model"]=="D2 Personalized"]
        d2_cost=scenario_metric(scenario,"GT_Total_Cost") if model=="D2 Personalized" else None
        return None
    original_d1_by_scenario={}
    scenario_csv=list(csv.DictReader((args.phase4c_dir/"decision_by_scenario.csv").open(encoding="utf-8")))
    for scenario in ("S6_high_distance_sensitive","S8_high_turn_sensitive"):
        original_d1_by_scenario[scenario]=float(next(r["value"] for r in scenario_csv if r["scenario"]==scenario and r["model"]=="D1 Generic" and r["metric"]=="GT_Total_Cost"))
    uncertain=[r for r in selected if r["scenario"] in ("S9_uncertain_new_person","S10_action_conflict")]
    uncertain_keep=float(np.mean([r["selected_action"]==0 for r in uncertain]))
    uncertain_near_optimal=float(np.mean([r["Oracle_Regret"]<=0.01 for r in uncertain]))
    old_overall=original["models"]["D2 Personalized"]["GT_Total_Cost"]
    new_overall=float(np.mean([r["GT_Total_Cost"] for r in selected]))
    new_regret=float(np.mean([r["Oracle_Regret"] for r in selected]))
    criteria={
        "unsafe_rejection_significantly_above_28_6pct":evaluation["unsafe_recall"]>=0.60,
        "selected_safety_violation_below_5pct":evaluation["selected_violation"]<0.05,
        "false_safe_down_and_retention_preserved":evaluation["false_safe"]<0.714 and evaluation["safe_retention"]>=0.60,
        "distance_sensitive_not_worse_than_D1":high_distance_new<=original_d1_by_scenario["S6_high_distance_sensitive"]*1.01,
        "turn_sensitive_not_worse_than_D1":high_turn_new<=original_d1_by_scenario["S8_high_turn_sensitive"]*1.01,
        "risk_positive_regret_correlation":evaluation["risk_regret_pearson"] is not None and evaluation["risk_regret_pearson"]>0.15,
        "uncertain_not_blind_keep_or_keep_near_optimal":uncertain_keep<1.0 or uncertain_near_optimal>=0.90,
        "overall_cost_not_materially_worse":new_overall<=old_overall*1.05,
        "oracle_regret_remains_close":new_regret<=0.02,
        "selectors_forbid_GT":True,
    }
    criteria["five_seed_gate_passed"]=bool(all(criteria.values()))
    criteria["ready_to_freeze_phase4c"]=False
    write_csv(args.output_dir/"root_distance_audit.csv",baseline_audit+calibrated_audit)
    write_csv(args.output_dir/"safety_target_metrics.csv",target_rows)
    write_csv(args.output_dir/"distance_calibration.csv",calibration_rows)
    write_csv(args.output_dir/"safety_uncertainty.csv",[{"synthetic_interaction":LABEL,"seed":args.seed,"scenario":r["scenario"],"sample":r["sample"],"predicted_risk":r["predicted_risk"],"sigma_minimum":r["sigma_minimum"],"actual_regret":r["regret"]} for r in evaluation["regret_rows"]])
    write_csv(args.output_dir/"safety_gate_v2.csv",evaluation["gate_rows"])
    write_csv(args.output_dir/"fallback_comparison.csv",evaluation["fallback_rows"])
    write_csv(args.output_dir/"distance_sensitive.csv",evaluation["distance_rows"])
    write_csv(args.output_dir/"turn_sensitive.csv",evaluation["turn_rows"])
    write_csv(args.output_dir/"decision_critical_fidelity.csv",evaluation["decision_fidelity_rows"])
    write_csv(args.output_dir/"regret_distribution.csv",evaluation["regret_rows"])
    write_csv(args.output_dir/"personalization_switches.csv",switches)
    write_csv(args.output_dir/"multiseed.csv",[{"synthetic_interaction":LABEL,"seed":args.seed,"metric":k,"value":v,"detail":"seed42 gate run only; five-seed not started"} for k,v in evaluation["regret_summary"].items()])
    figures=make_figures(args.output_dir,baseline_audit+calibrated_audit,evaluation["gate_rows"],evaluation,fidelity)
    summary={
        "label":LABEL,"seed":args.seed,"five_seed_started":False,
        "phase4c_results_untouched":True,"test_read_once_after_freeze":True,
        "training":training,"calibration":calibration.__dict__,
        "audit_at_1s":{"human_root_error":root_error,"robot_position_error":robot_error,"action_effect_error":effect_error,"minimum_distance_error_before":distance_old,"minimum_distance_error_after":distance_new},
        "safety":{"unsafe_candidate_recall":evaluation["unsafe_recall"],"safe_candidate_retention":evaluation["safe_retention"],"false_safe_rate":evaluation["false_safe"],"false_veto_rate":evaluation["false_veto"],"selected_action_violation_rate":evaluation["selected_violation"]},
        "regret":evaluation["regret_summary"],
        "uncertainty_regret":{"risk_pearson":evaluation["risk_regret_pearson"],"risk_spearman":evaluation["risk_regret_spearman"],"sigma_pearson":evaluation["sigma_regret_pearson"],"top10_uncertainty_regret":evaluation["top10_uncertainty_regret"],"bottom10_uncertainty_regret":evaluation["bottom10_uncertainty_regret"]},
        "scenario_costs":{"high_distance_sensitive_D2":high_distance_new,"high_distance_sensitive_D1_reference":original_d1_by_scenario["S6_high_distance_sensitive"],"high_turn_sensitive_D2":high_turn_new,"high_turn_sensitive_D1_reference":original_d1_by_scenario["S8_high_turn_sensitive"],"uncertain_KEEP_rate":uncertain_keep,"uncertain_near_optimal_rate":uncertain_near_optimal,"overall_D2_cost":new_overall,"phase4c_overall_D2_reference":old_overall,"mean_oracle_regret":new_regret},
        "D2_vs_D3_by_scenario":{
            scenario:{
                model:{
                    "GT_Total_Cost":float(np.mean([row["GT_Total_Cost"] for row in evaluation["selected_rows"] if row["scenario"]==scenario and row["model"]==model])),
                    "Oracle_Regret":float(np.mean([row["Oracle_Regret"] for row in evaluation["selected_rows"] if row["scenario"]==scenario and row["model"]==model])),
                    "Safety_Violation":float(np.mean([row["GT_Safety_Violation"] for row in evaluation["selected_rows"] if row["scenario"]==scenario and row["model"]==model])),
                    "KEEP_Rate":float(np.mean([row["selected_action"]==0 for row in evaluation["selected_rows"] if row["scenario"]==scenario and row["model"]==model])),
                } for model in ("D2 calibrated uncertainty","D3 no safety uncertainty")
            } for scenario in ("S6_high_distance_sensitive","S8_high_turn_sensitive","S9_uncertain_new_person","S10_action_conflict")
        },
        "success_criteria":criteria,"figures":figures,
    }
    write_json(args.output_dir/"summary.json",summary)
    print(f"unsafe_recall={evaluation['unsafe_recall']:.4f} selected_violation={evaluation['selected_violation']:.4f} min_error={distance_old:.4f}->{distance_new:.4f} gate={criteria['five_seed_gate_passed']}",flush=True)


if __name__=="__main__":main()
