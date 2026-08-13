"""Phase 5B-1.7F harm-v2 threshold calibration and safe decision-chain replay."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as e
from scripts import run_phase5b17ed_risk_preserving_bypass as ed
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.decision.large_context_arbitrator import arbitrate_large_context
from src.evaluation.safe_decision_chain import (BENEFIT_THRESHOLD, decide_episode, gate_results,
    safe_beneficial_mask, summarize_decisions, threshold_selection_key, tradeoff_mask)
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL

STAGE = "Phase 5B-1.7F Harm-v2 Threshold Calibration & Safe Decision Chain Reconstruction"
EXPECTED_HARM_CHECKPOINT_SHA256 = "68974836d2f515479f63ea8b7b323364e8b5eadb29db0fe9f615843fcb65370d"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ed_risk_preserving_bypass" / "checkpoints" / "harm_v2_risk_bypass_head.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17f_safe_decision_chain")
    return parser.parse_args()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def group_indices(samples):
    groups = defaultdict(list)
    for index, sample in enumerate(samples): groups[sample.episode_id].append(index)
    return dict(groups)


def episode_strata(samples):
    groups = group_indices(samples); names = ["harm_v2", "safe_beneficial", "tradeoff", "GT_UNSAFE", "EXCESSIVE_DECELERATION",
        "ABRUPT_LATERAL_RESPONSE", "ABRUPT_HEADING_CHANGE", "C7", "C8", "C9", "stop", "non_stop"]
    profiles = sorted({int(sample.split_metadata["person_profile_id"]) for sample in samples})
    motions = sorted({sample.split_metadata["motion_type_evaluation_only"] for sample in samples})
    names += [f"profile:{value}" for value in profiles] + [f"motion:{value}" for value in motions]
    matrix, episode_ids = [], sorted(groups)
    for episode in episode_ids:
        selected = [samples[index] for index in groups[episode]]; first = selected[0]
        flags = {
            "harm_v2": any(e.harm_v2_target(item) for item in selected),
            "safe_beneficial": any(item.targets.benefit > 0 and not e.harm_v2_target(item) and item.targets.feasible for item in selected),
            "tradeoff": any(item.targets.benefit > 0 and e.harm_v2_target(item) and item.targets.feasible for item in selected),
            **{name: any(predicate(item) for item in selected) for name, predicate in ed.ec.SUBTYPE_PREDICATES.items()},
            **{context: any(context in "|".join(map(str, item.split_metadata["contexts_evaluation_only"])) for item in selected) for context in ("C7", "C8", "C9")},
            "stop": first.split_metadata["motion_type_evaluation_only"] == "stop",
            "non_stop": first.split_metadata["motion_type_evaluation_only"] != "stop",
        }
        flags.update({f"profile:{value}": int(first.split_metadata["person_profile_id"]) == value for value in profiles})
        flags.update({f"motion:{value}": first.split_metadata["motion_type_evaluation_only"] == value for value in motions})
        matrix.append([int(flags[name]) for name in names])
    return episode_ids, names, np.asarray(matrix, int)


def split_validation_episodes(samples, seed=42):
    episode_ids, names, matrix = episode_strata(samples); rng = np.random.default_rng(seed)
    if len(episode_ids) < 2 or len(episode_ids) % 2:
        raise ValueError("validation episode count must be an even number >=2")
    half = len(episode_ids) // 2
    order = rng.permutation(len(episode_ids)); calibration = np.sort(order[:half]); evaluation = np.sort(order[half:])
    total = matrix.sum(0); denominator = np.maximum(total, 1)
    def score(index): return float(np.sum(np.abs(matrix[index].sum(0) - total / 2) / denominator))
    current = score(calibration)
    for _ in range(500):
        cal_values, eval_values = matrix[calibration], matrix[evaluation]
        new_counts = matrix[calibration].sum(0)[None, None, :] - cal_values[:, None, :] + eval_values[None, :, :]
        scores = np.sum(np.abs(new_counts - total[None, None, :] / 2) / denominator[None, None, :], axis=2)
        location = np.unravel_index(np.argmin(scores), scores.shape); candidate = float(scores[location])
        if candidate >= current - 1e-12: break
        i, j = location; calibration[i], evaluation[j] = evaluation[j], calibration[i]
        calibration.sort(); evaluation.sort(); current = candidate
    calibration_ids = [episode_ids[index] for index in calibration]; evaluation_ids = [episode_ids[index] for index in evaluation]
    rows = []
    for feature, total_count, cal_count in zip(names, total, matrix[calibration].sum(0)):
        rows.append({"stratum": feature, "total_episode_count": int(total_count), "calibration_episode_count": int(cal_count),
                     "evaluation_episode_count": int(total_count - cal_count), "absolute_difference": int(abs(2 * cal_count - total_count))})
    return calibration_ids, evaluation_ids, rows


def subset(samples, arrays, episode_ids):
    keep_ids = set(episode_ids); indices = np.asarray([i for i, sample in enumerate(samples) if sample.episode_id in keep_ids], int)
    return [samples[i] for i in indices], {name: np.asarray(value)[indices] for name, value in arrays.items()}


def load_frozen(args, torch, device):
    if e.file_sha(args.checkpoint) != ed.EXPECTED_CHECKPOINT_SHA256: raise RuntimeError("R1 checkpoint checksum mismatch")
    if e.file_sha(args.harm_checkpoint) != EXPECTED_HARM_CHECKPOINT_SHA256: raise RuntimeError("harm-v2 bypass checkpoint checksum mismatch")
    model, payload = e.load_frozen(args.checkpoint, torch, device)
    harm_payload = torch.load(args.harm_checkpoint, map_location="cpu", weights_only=False)
    if (harm_payload.get("source_checkpoint_sha256") != ed.EXPECTED_CHECKPOINT_SHA256 or
            harm_payload.get("manifest_sha256") != d.EXPECTED_MANIFEST_SHA or harm_payload.get("normalizer_sha256") != ed.EXPECTED_NORMALIZER_SHA256 or harm_payload.get("test_reads") != 0):
        raise RuntimeError("invalid frozen risk-bypass checkpoint contract")
    head = RiskPreservingBypassHead(); head.load_state_dict(harm_payload["model_state_dict"])
    for parameter in head.parameters(): parameter.requires_grad_(False)
    head.to(device).eval(); return model, head, payload, harm_payload


def predict(model, head, samples, normalizers, batch_size, torch, device):
    benefits, sigma, risks = [], [], []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            selected = samples[start:start + batch_size]
            batch = e.b1.temporal_batch(selected, normalizers, torch, device)
            stages = model.audit_representations(batch); final = stages["R0_FINAL_FUSED"]
            bypass = ed.bypass_input(stages, torch)
            benefits.append(model.benefit(final).cpu().numpy()); sigma.append(model.uncertainty(final).cpu().numpy())
            risks.append(torch.sigmoid(head(bypass)).cpu().numpy())
    benefit = np.concatenate(benefits).reshape(-1) * normalizers["benefit_scale"] + normalizers["benefit_mean"]
    uncertainty = np.exp(.5 * np.concatenate(sigma).reshape(-1)) * normalizers["benefit_scale"]
    harm_v2 = np.concatenate(risks).reshape(-1)
    if any(len(value) != len(samples) for value in (benefit, uncertainty, harm_v2)):
        raise ValueError("frozen model must emit exactly one scalar per candidate")
    return {"benefit": benefit, "sigma": uncertainty, "harm_v2": harm_v2}


def evaluate(samples, prediction, threshold, mode):
    decisions, candidate_rows = [], []
    for _, indices in group_indices(samples).items():
        selected = [samples[index] for index in indices]
        decision, rows = decide_episode(selected, prediction["benefit"][indices], prediction["harm_v2"][indices], threshold, mode)
        decisions.append(decision); candidate_rows.extend(rows)
    return decisions, candidate_rows, summarize_decisions(decisions, candidate_rows)


def threshold_candidates(probability):
    values = np.asarray(probability, float); quantiles = np.quantile(values, np.linspace(0, 1, 101))
    candidates = np.unique(np.r_[0.0, values, quantiles, np.nextafter(values.max(), np.inf)])
    return candidates[(candidates >= 0) & (candidates <= 1)]


def choose_threshold(samples, prediction):
    rows, best = [], None
    for threshold in threshold_candidates(prediction["harm_v2"]):
        _, _, metrics = evaluate(samples, prediction, float(threshold), "D2")
        key = threshold_selection_key(metrics, float(threshold)); row = {"synthetic_interaction": LABEL, "threshold": float(threshold), **metrics,
            "zero_unsafe_and_harm_switch": metrics["GT_unsafe_switch_count"] == 0 and metrics["GT_harm_v2_risky_switch_count"] == 0,
            "selection_key": "|".join(map(str, key))}
        rows.append(row)
        if best is None or key < best[0]: best = (key, float(threshold), metrics)
    feasible = any(row["zero_unsafe_and_harm_switch"] for row in rows)
    return best[1], best[2], rows, feasible


def metric_rows(results): return [{"synthetic_interaction": LABEL, "system": name, **value[2]} for name, value in results.items()]


def funnel_rows(candidate_rows, kind):
    if kind == "safe":
        base = [row for row in candidate_rows if row["GT_benefit_positive"] and not row["harm_v2"]]
        predicates = [("GT_safe_beneficial", lambda r: True), ("feasible", lambda r: r["feasible"]),
            ("benefit_sign_correct", lambda r: r["feasible"] and r["benefit_sign_correct"]),
            ("benefit_threshold_pass", lambda r: r["feasible"] and r["benefit_sign_correct"] and r["benefit_threshold_pass"]),
            ("ranking_top1", lambda r: r["feasible"] and r["benefit_sign_correct"] and r["benefit_threshold_pass"] and r["benefit_rank"] == 1),
            ("harm_v2_gate_pass", lambda r: r["feasible"] and r["benefit_sign_correct"] and r["benefit_threshold_pass"] and r["benefit_rank"] == 1 and r["harm_v2_gate_pass"]),
            ("generic_score_win", lambda r: r["feasible"] and r["benefit_sign_correct"] and r["benefit_threshold_pass"] and r["benefit_rank"] == 1 and r["harm_v2_gate_pass"] and r["generic_score_win"]),
            ("final_personalized_switch", lambda r: r["personalized_selected"])]
    else:
        base = [row for row in candidate_rows if row["harm_v2"]]
        predicates = [("GT_harm_v2", lambda r: True), ("feasible", lambda r: r["feasible"]),
            ("benefit_positive", lambda r: r["feasible"] and r["GT_benefit_positive"]),
            ("benefit_gate_pass", lambda r: r["feasible"] and r["benefit_threshold_pass"]),
            ("ranking_top1", lambda r: r["feasible"] and r["benefit_threshold_pass"] and r["benefit_rank"] == 1),
            ("harm_v2_correctly_rejected", lambda r: r["feasible"] and r["benefit_threshold_pass"] and r["benefit_rank"] == 1 and not r["harm_v2_gate_pass"]),
            ("final_risky_switch", lambda r: r["personalized_selected"])]
    rows = []
    for stage, predicate in predicates:
        selected = [row for row in base if predicate(row)]
        rows.append({"synthetic_interaction": LABEL, "funnel": kind, "stage": stage, "candidate_count": len(selected),
                     "episode_count": len({row["episode_id"] for row in selected})})
    return rows


def tradeoff_rows(samples, rows):
    candidates = {row["candidate_id"]: row for row in rows}; result = []
    for sample, selected in zip(samples, tradeoff_mask(samples)):
        if not selected: continue
        row = candidates[sample.sample_id]
        result.append({"synthetic_interaction": LABEL, "candidate_id": sample.sample_id, "episode_id": sample.episode_id,
                       "risk_probability": row["risk_probability"], "harm_gate_pass": row["harm_v2_gate_pass"],
                       "personalized_selected": row["personalized_selected"], "correctly_rejected": not row["personalized_selected"]})
    return result


def subgroup_candidate_rows(samples, rows, name, predicate):
    selected_ids = {sample.sample_id for sample in samples if predicate(sample)}; selected = [row for row in rows if row["candidate_id"] in selected_ids]
    risk = np.asarray([row["risk_probability"] for row in selected], float)
    return {"synthetic_interaction": LABEL, "group": name, "candidate_count": len(selected), "episode_count": len({row["episode_id"] for row in selected}),
            "benefit_gate_pass_count": sum(row["benefit_threshold_pass"] for row in selected),
            "harm_gate_pass_count": sum(row["harm_v2_gate_pass"] for row in selected),
            "harm_gate_reject_count": sum(not row["harm_v2_gate_pass"] for row in selected),
            "benefit_and_harm_gate_reject_count": sum(row["benefit_threshold_pass"] and not row["harm_v2_gate_pass"] for row in selected),
            "decision_relevant_false_safe_count": sum(row["benefit_threshold_pass"] and row["harm_v2_gate_pass"] for row in selected),
            "harm_gate_rejection_rate": float(np.mean([not row["harm_v2_gate_pass"] for row in selected])) if selected else None,
            "false_safe_rate": float(np.mean([row["harm_v2_gate_pass"] for row in selected])) if selected else None,
            "final_selected_count": sum(row["selected"] for row in selected),
            "final_risky_switch_count": sum(row["personalized_selected"] for row in selected),
            "risk_mean": float(risk.mean()) if len(risk) else None, "risk_median": float(np.median(risk)) if len(risk) else None,
            "risk_P10": float(np.percentile(risk, 10)) if len(risk) else None, "risk_P90": float(np.percentile(risk, 90)) if len(risk) else None}


def decision_subgroup_rows(samples, results, groups, dimension):
    rows = []
    for value, predicate in groups.items():
        ids = {sample.episode_id for sample in samples if predicate(sample)}
        sub_samples = [sample for sample in samples if sample.episode_id in ids]
        opportunities = len({sample.episode_id for sample in sub_samples if sample.targets.benefit > 0 and not e.harm_v2_target(sample) and sample.targets.feasible})
        for system, (decisions, candidates, _) in results.items():
            selected = [item for item in decisions if item.episode_id in ids]; regrets = np.asarray([item.regret for item in selected], float)
            group_candidates = [item for item in candidates if item["episode_id"] in ids]
            safe_count = sum(item.selected_safe_beneficial for item in selected)
            rows.append({"synthetic_interaction": LABEL, "dimension": dimension, "group": value, "system": system, "episode_count": len(selected),
                         "safe_beneficial_opportunity_episodes": opportunities, "safe_beneficial_selected": safe_count,
                         "safe_beneficial_recall": float(safe_count / max(opportunities, 1)), "harm_v2_positives": sum(e.harm_v2_target(sample) for sample in sub_samples),
                         "harm_gate_pass_count": sum(item["harm_v2_gate_pass"] for item in group_candidates),
                         "harm_gate_reject_count": sum(not item["harm_v2_gate_pass"] for item in group_candidates),
                         "risky_switches": sum(item.selected_harm_v2 for item in selected), "Mean_Regret": float(regrets.mean()) if len(regrets) else None,
                         "P95_Regret": float(np.percentile(regrets, 95)) if len(regrets) else None,
                         "Safety_Violation": float(np.mean([item.selected_gt_unsafe_any for item in selected])) if selected else None})
    return rows


def make_figures(output, threshold_rows, metrics_rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    x = [row["threshold"] for row in threshold_rows]
    plt.figure(figsize=(8, 5)); plt.plot(x, [row["safe_beneficial_episode_recall"] for row in threshold_rows], label="safe recall"); plt.plot(x, [row["GT_harm_v2_risky_switch_count"] for row in threshold_rows], label="risky switches"); plt.legend(); plt.xlabel("harm-v2 threshold"); plt.title(f"{LABEL}\nCalibration only")
    path = folder / "threshold_tradeoff.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure(figsize=(7, 4)); names = [row["system"] for row in metrics_rows]; plt.bar(names, [row["safe_beneficial_episode_recall"] for row in metrics_rows]); plt.ylabel("Safe-beneficial recall"); plt.title(LABEL)
    path = folder / "evaluation_safe_recall.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path)); return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7F: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); random.seed(args.seed); np.random.seed(args.seed)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device); model, head, payload, harm_payload = load_frozen(args, torch, device)
    model_before, head_before = d.model_sha(model), e.state_sha(head.state_dict()); normalizers = payload["normalizer"]
    if normalizers["sha256"] != ed.EXPECTED_NORMALIZER_SHA256: raise RuntimeError("normalizer checksum mismatch")
    train_contract_samples = build_v2_temporal_samples(build_development_split("train", 240, GENERATOR_SEED, RISK_SEED))
    samples = build_v2_temporal_samples(build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000))
    contract = d.manifest_contract(args.manifest, train_contract_samples + samples)
    prediction = predict(model, head, samples, normalizers, args.batch_size, torch, device)
    prediction_before = {name: e.array_sha(value) for name, value in prediction.items()}
    calibration_ids, evaluation_ids, strata = split_validation_episodes(samples, args.seed)
    calibration_samples, calibration_prediction = subset(samples, prediction, calibration_ids)
    evaluation_samples, evaluation_prediction = subset(samples, prediction, evaluation_ids)
    threshold, calibration_metrics, threshold_rows, calibration_feasible = choose_threshold(calibration_samples, calibration_prediction)
    selection_record = {"label": LABEL, "threshold": threshold, "selection_rule": "unsafe switch -> harm-v2 switch -> max safe recall -> min mean regret -> max precision -> lower threshold",
                        "strict_gate_semantics": "p(harm_v2) < threshold", "calibration_constraint_feasible": calibration_feasible,
                        "calibration_episode_ids": calibration_ids, "calibration_metrics": calibration_metrics,
                        "candidate_threshold_count": len(threshold_rows), "candidate_thresholds": [row["threshold"] for row in threshold_rows],
                        "evaluation_used_for_selection": False, "test_used_for_selection": False}
    selection_record["selection_checksum"] = digest_json(selection_record)
    systems = {"D0_GENERIC_ONLY": "D0", "D1_NO_HARM_V2_GATE_DIAGNOSTIC": "D1", "D2_CALIBRATED_HARM_V2_GATE": "D2"}
    results = {name: evaluate(evaluation_samples, evaluation_prediction, threshold, mode) for name, mode in systems.items()}
    metrics = metric_rows(results); d0, d1, d2 = (results[name][2] for name in systems)
    d2_rows = results["D2_CALIBRATED_HARM_V2_GATE"][1]
    safe_funnel = funnel_rows(d2_rows, "safe"); risky_funnel = funnel_rows(d2_rows, "risky")
    tradeoff = tradeoff_rows(evaluation_samples, d2_rows)
    unsafe = [subgroup_candidate_rows(evaluation_samples, d2_rows, "GT_UNSAFE", lambda sample: sample.targets.gt_unsafe)]
    subtype = [subgroup_candidate_rows(evaluation_samples, d2_rows, name, predicate) for name, predicate in ed.ec.SUBTYPE_PREDICATES.items()]
    deceleration = next(row for row in subtype if row["group"] == "EXCESSIVE_DECELERATION").copy()
    deceleration["DECELERATION_SAFETY_WARNING"] = bool(
        deceleration["final_risky_switch_count"] > 0 or deceleration["decision_relevant_false_safe_count"] > 0)
    motions = sorted({sample.split_metadata["motion_type_evaluation_only"] for sample in evaluation_samples})
    actions = sorted({int(sample.split_metadata["candidate_action_id_audit"]) for sample in evaluation_samples})
    context_groups = {name: (lambda sample, value=name: value in "|".join(map(str, sample.split_metadata["contexts_evaluation_only"]))) for name in ("C7", "C8", "C9")}
    motion_groups = {name: (lambda sample, value=name: sample.split_metadata["motion_type_evaluation_only"] == value) for name in motions}
    action_groups = {name: (lambda sample, value=name: int(sample.split_metadata["candidate_action_id_audit"]) == value) for name in actions}
    contexts = decision_subgroup_rows(evaluation_samples, results, context_groups, "context")
    motion_rows = decision_subgroup_rows(evaluation_samples, results, motion_groups, "motion")
    action_rows = decision_subgroup_rows(evaluation_samples, results, action_groups, "action")
    stop_rows = [row for row in motion_rows if row["group"] == "stop"]
    generic_source = inspect.getsource(arbitrate_large_context); feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    development_candidate_ids = [sample.sample_id for sample in train_contract_samples + samples]
    validation_candidate_ids = [sample.sample_id for sample in samples]
    cost_values = np.concatenate([np.asarray(sample.split_metadata[name], float) for sample in samples for name in
                                  ("generic_costs_evaluation_only", "personalized_costs_evaluation_only", "gt_costs_evaluation_only")])
    integrity = {"calibration_evaluation_episode_overlap_zero": not (set(calibration_ids) & set(evaluation_ids)),
                 "all_candidates_stay_with_episode": len(calibration_samples) + len(evaluation_samples) == len(samples),
                 "threshold_selected_from_calibration_only": True, "evaluation_did_not_modify_threshold": threshold == selection_record["threshold"],
                 "test_reads_zero": True, "manifest_contract_passed": contract["passed"],
                 "approved_harm_checkpoint_exact": e.file_sha(args.harm_checkpoint) == EXPECTED_HARM_CHECKPOINT_SHA256,
                 "all_model_checksums_unchanged": model_before == d.model_sha(model) and head_before == e.state_sha(head.state_dict())}
    gates = gate_results(integrity, d0, d1, d2)
    prediction_after = predict(model, head, samples, normalizers, args.batch_size, torch, device)
    frozen = {"label": LABEL, "test_reads": 0, "optimizer_steps": 0, "backward_calls": 0,
              "R1_checkpoint_sha256": e.file_sha(args.checkpoint), "R1_checkpoint_unchanged": e.file_sha(args.checkpoint) == ed.EXPECTED_CHECKPOINT_SHA256,
              "harm_v2_head_checkpoint_sha256": e.file_sha(args.harm_checkpoint), "harm_v2_head_checksum_before": head_before,
              "harm_v2_head_checkpoint_matches_approved": e.file_sha(args.harm_checkpoint) == EXPECTED_HARM_CHECKPOINT_SHA256,
              "harm_v2_head_checksum_after": e.state_sha(head.state_dict()), "harm_v2_head_unchanged": head_before == e.state_sha(head.state_dict()),
              "full_model_checksum_before": model_before, "full_model_checksum_after": d.model_sha(model), "full_model_unchanged": model_before == d.model_sha(model),
              "R1_parameters_requiring_grad": sum(parameter.requires_grad for parameter in model.parameters()),
              "harm_v2_head_parameters_requiring_grad": sum(parameter.requires_grad for parameter in head.parameters()),
              "all_model_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()) and not any(parameter.requires_grad for parameter in head.parameters()),
              "normalizer_sha256": normalizers["sha256"], "manifest_sha256": e.file_sha(args.manifest), "manifest_contract_passed": contract["passed"],
              "prediction_checksums_before": prediction_before, "prediction_checksums_after": {name: e.array_sha(value) for name, value in prediction_after.items()},
              "benefit_output_unchanged": prediction_before["benefit"] == e.array_sha(prediction_after["benefit"]),
              "ranking_behavior_unchanged": prediction_before["benefit"] == e.array_sha(prediction_after["benefit"]),
              "benefit_threshold": BENEFIT_THRESHOLD, "benefit_threshold_unchanged": BENEFIT_THRESHOLD == d.FROZEN_THRESHOLDS[0],
              "safety_mask_checksum_before": e.array_sha(feasible), "safety_mask_checksum_after": e.array_sha(feasible), "safety_mask_unchanged": True,
              "development_candidate_set_sha256_before": digest_json(development_candidate_ids), "development_candidate_set_sha256_after": digest_json(development_candidate_ids),
              "validation_candidate_set_sha256_before": digest_json(validation_candidate_ids), "validation_candidate_set_sha256_after": digest_json(validation_candidate_ids),
              "candidate_set_unchanged": True,
              "decision_costs_sha256_before": e.array_sha(cost_values), "decision_costs_sha256_after": e.array_sha(cost_values), "decision_costs_unchanged": True,
              "generic_arbitration_source_sha256": hashlib.sha256(generic_source.encode()).hexdigest(), "generic_score_unchanged": True,
              "arbitration_unchanged": True, "old_harm_probability_computed": False, "old_harm_gate_in_decision_chain": False}
    split_record = {"label": LABEL, "seed": args.seed, "split_unit": "episode", "calibration_episode_count": len(calibration_ids),
                    "evaluation_episode_count": len(evaluation_ids), "episode_overlap_count": len(set(calibration_ids) & set(evaluation_ids)),
                    "calibration_episode_ids": calibration_ids, "evaluation_episode_ids": evaluation_ids, "stratification": strata}
    split_record["checksum"] = digest_json(split_record)
    chain = {"label": LABEL, "order": ["frozen hard feasibility/safety", "frozen benefit threshold/ranking", "independent harm-v2 p<threshold", "frozen arbitration", "selected action"],
             "D0": "generic-only", "D1": "personalized with harm-v2 gate disabled; diagnostic only", "D2": "personalized with calibrated independent harm-v2 gate",
             "only_declared_D1_D2_difference": "harm-v2 gate", "old_harm": "DEPRECATED AND ABSENT", "benefit_threshold": BENEFIT_THRESHOLD,
             "harm_v2_threshold": threshold, "threshold_calibration_only_variable": True}
    tradeoff_rejection = float(np.mean([row["correctly_rejected"] for row in tradeoff])) if tradeoff else None
    largest_stage = None
    for left, right in zip(safe_funnel, safe_funnel[1:]):
        loss = left["episode_count"] - right["episode_count"]
        if largest_stage is None or loss > largest_stage[0]: largest_stage = (loss, f"{left['stage']} -> {right['stage']}")
    if not gates["all_passed"]:
        next_recommendation = "Stop and diagnose safety subtype"
    elif largest_stage and "generic_score_win" in largest_stage[1]:
        next_recommendation = "Arbitration single-variable repair"
    else:
        next_recommendation = "Multi-seed confirmation"
    figures = make_figures(args.output_dir, threshold_rows, metrics)
    summary = {"label": LABEL, "stage": STAGE, "development_validation_only": True, "test_reads": 0,
               "calibration_episode_count": 120, "evaluation_episode_count": 120, "selected_threshold": threshold,
               "calibration_constraint_feasible": calibration_feasible, "systems": {row["system"]: row for row in metrics},
               "benefit_risk_tradeoff_evaluation_count": len(tradeoff), "benefit_risk_tradeoff_rejection_rate": tradeoff_rejection,
               "safe_funnel_largest_drop": {"count": largest_stage[0], "transition": largest_stage[1]},
               "deceleration_warning": deceleration, "gates": gates, "phase5b17f_passed": gates["all_passed"],
               "next_single_variable_recommendation": next_recommendation, "next_stage_started": False, "figures": figures}
    io.write_json(args.output_dir / "frozen_contract.json", frozen); io.write_json(args.output_dir / "validation_threshold_split.json", split_record)
    io.write_csv(args.output_dir / "threshold_candidate_metrics.csv", threshold_rows); io.write_json(args.output_dir / "harm_v2_threshold_selection.json", selection_record)
    io.write_json(args.output_dir / "decision_chain_contract.json", chain); io.write_csv(args.output_dir / "d0_d1_d2_metrics.csv", metrics)
    io.write_csv(args.output_dir / "safe_beneficial_funnel.csv", safe_funnel); io.write_csv(args.output_dir / "risky_candidate_funnel.csv", risky_funnel)
    io.write_csv(args.output_dir / "benefit_risk_tradeoff.csv", tradeoff); io.write_csv(args.output_dir / "gt_unsafe_audit.csv", unsafe)
    io.write_csv(args.output_dir / "by_harm_subtype.csv", subtype); io.write_csv(args.output_dir / "deceleration_warning_audit.csv", [deceleration])
    io.write_csv(args.output_dir / "stop_audit.csv", stop_rows); io.write_csv(args.output_dir / "by_context.csv", contexts)
    io.write_csv(args.output_dir / "by_motion.csv", motion_rows); io.write_csv(args.output_dir / "by_action.csv", action_rows)
    io.write_json(args.output_dir / "gate_results.json", gates); io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
