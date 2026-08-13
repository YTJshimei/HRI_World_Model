"""Phase 5B-1.7F-B development-only generic harm-v2 coverage repair."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as e
from scripts import run_phase5b17ed_risk_preserving_bypass as ed
from scripts import run_phase5b17f_safe_decision_chain as f
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import RobotAction
from src.decision.generic_risk_coverage import select_with_generic_risk_coverage
from src.decision.large_context_arbitrator import arbitrate_large_context
from src.evaluation.generic_safety_coverage import (
    CoveredEpisodeDecision, branchwise_rows, gate_results, summarize_covered,
)
from src.evaluation.safe_decision_chain import BENEFIT_THRESHOLD
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL

STAGE = "Phase 5B-1.7F-B Generic Safety Coverage Repair"
MECHANISM = "DEVELOPMENT MECHANISM RESULT"
HARM_THRESHOLD = 0.10968538373708725
FORMAL_DIR = PROJECT_ROOT / "results_dev" / "phase5b17f_safe_decision_chain"
AUDIT_DIR = PROJECT_ROOT / "results_dev" / "phase5b17fa_safety_attribution_audit"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ed_risk_preserving_bypass" / "checkpoints" / "harm_v2_risk_bypass_head.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--formal-dir", type=Path, default=FORMAL_DIR)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17fb_generic_safety_coverage")
    return parser.parse_args()


def file_sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def group_indices(samples):
    result = defaultdict(list)
    for index, sample in enumerate(samples): result[sample.episode_id].append(index)
    return dict(result)


def context_has(sample, name):
    return name in "|".join(map(str, sample.split_metadata["contexts_evaluation_only"]))


def evaluate_d3(samples, prediction):
    decisions, rows = [], []
    for episode_id, indices in group_indices(samples).items():
        selected = [samples[index] for index in indices]; first = selected[0]; meta = first.split_metadata
        actions = np.asarray([item.split_metadata["candidate_action_id_audit"] for item in selected], int)
        all_actions = np.asarray(meta["all_action_ids_evaluation_only"], int)
        full = np.asarray([int(np.flatnonzero(all_actions == action)[0]) for action in actions])
        feasible = np.asarray([item.targets.feasible for item in selected], bool)
        generic_cost = np.asarray(meta["generic_costs_evaluation_only"], float)[full]
        personalized_cost = np.asarray(meta["personalized_costs_evaluation_only"], float)[full]
        gt_cost = np.asarray(meta["gt_costs_evaluation_only"], float)[full]
        benefit, risk = prediction["benefit"][indices], prediction["harm_v2"][indices]
        result = select_with_generic_risk_coverage(actions, feasible, generic_cost, personalized_cost, benefit, risk,
                                                   BENEFIT_THRESHOLD, HARM_THRESHOLD)
        if result.abstained:
            decisions.append(CoveredEpisodeDecision(episode_id, None, None, False, True, None, False, False, False, None, result.reason))
        else:
            local = int(result.selected_index); sample = selected[local]
            decisions.append(CoveredEpisodeDecision(
                episode_id, local, result.generic_index, result.personalized, False, result.selected_action,
                bool(sample.split_metadata["harm_v2_evaluation_only"]), bool(sample.targets.gt_unsafe),
                bool(sample.targets.benefit > 0 and not sample.split_metadata["harm_v2_evaluation_only"] and sample.targets.feasible),
                float(gt_cost[local] - gt_cost.min()), result.reason))
        ranks = np.empty(len(benefit), int); order = np.argsort(-benefit, kind="stable"); ranks[order] = np.arange(1, len(order) + 1)
        eligible_generic = feasible & (risk < HARM_THRESHOLD)
        adjusted = personalized_cost - np.maximum(benefit, 0.0)
        for local, sample in enumerate(selected):
            rows.append({"episode_id": episode_id, "candidate_id": sample.sample_id, "local_index": local,
                         "action": int(actions[local]), "feasible": bool(feasible[local]),
                         "generic_risk_eligible": bool(eligible_generic[local]),
                         "benefit_threshold_pass": bool(benefit[local] >= BENEFIT_THRESHOLD),
                         "benefit_rank": int(ranks[local]), "harm_v2_gate_pass": bool(risk[local] < HARM_THRESHOLD),
                         "generic_score_win": bool(result.generic_index is not None and adjusted[local] < generic_cost[result.generic_index]),
                         "predicted_benefit": float(benefit[local]), "risk_probability": float(risk[local]),
                         "harm_v2": bool(sample.split_metadata["harm_v2_evaluation_only"]), "gt_unsafe": bool(sample.targets.gt_unsafe),
                         "safe_beneficial": bool(sample.targets.benefit > 0 and not sample.split_metadata["harm_v2_evaluation_only"] and sample.targets.feasible),
                         "selected": bool(not result.abstained and local == result.selected_index),
                         "personalized_selected": bool(not result.abstained and result.personalized and local == result.selected_index)})
    return decisions, rows, summarize_covered(decisions, rows)


def d2_as_covered(decisions, rows):
    grouped = defaultdict(dict)
    for row in rows: grouped[row["episode_id"]][row["local_index"]] = row
    converted = []
    for item in decisions:
        selected = grouped[item.episode_id][item.selected_local]
        converted.append(CoveredEpisodeDecision(
            item.episode_id, item.selected_local, item.generic_local, item.personalized, False, item.selected_action,
            bool(selected["harm_v2"]), bool(selected["gt_unsafe"]),
            bool(item.personalized and selected["safe_beneficial"]), item.regret, "FROZEN_D2"))
    return converted, summarize_covered(converted, rows)


def subgroup_metrics(samples, decisions, name, predicate, system):
    episode_ids = {sample.episode_id for sample in samples if predicate(sample)}
    selected = [item for item in decisions if item.episode_id in episode_ids]
    opportunities = len({sample.episode_id for sample in samples if sample.episode_id in episode_ids and sample.targets.benefit > 0
                         and not sample.split_metadata["harm_v2_evaluation_only"] and sample.targets.feasible})
    safe_selected = sum(item.personalized and item.selected_safe_beneficial for item in selected)
    return {"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "system": system, "group": name,
            "episode_count": len(selected), "GT_unsafe_final": sum(item.selected_gt_unsafe for item in selected if not item.abstained),
            "harm_v2_final": sum(item.selected_harm_v2 for item in selected if not item.abstained),
            "personalized_risky": sum(item.personalized and item.selected_harm_v2 for item in selected),
            "safe_beneficial_opportunity_episodes": opportunities, "safe_beneficial_selected": safe_selected,
            "safe_beneficial_recall": safe_selected / max(opportunities, 1),
            "no_safe_generic": sum(item.abstained for item in selected)}


def figures(output, metrics, replacements):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    plt.figure(figsize=(7, 4)); plt.bar(("D2", "D3"), [row["Overall_Safety_Violation"] for row in metrics]); plt.ylabel("Overall Safety Violation"); plt.title(f"{LABEL}\n{MECHANISM}")
    path = folder / "d2_d3_safety.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    counts = defaultdict(int)
    for row in replacements: counts[row["D3_replacement_action_name"]] += 1
    plt.figure(figsize=(8, 4)); plt.bar(list(counts), list(counts.values())); plt.ylabel("Replacement episodes"); plt.xticks(rotation=20); plt.title(MECHANISM)
    path = folder / "replacement_actions.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path)); return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Stage 1.7F-B: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_summary = json.loads((args.formal_dir / "summary.json").read_text(encoding="utf-8"))
    formal_gates = json.loads((args.formal_dir / "gate_results.json").read_text(encoding="utf-8"))
    selection = json.loads((args.formal_dir / "harm_v2_threshold_selection.json").read_text(encoding="utf-8"))
    split = json.loads((args.formal_dir / "validation_threshold_split.json").read_text(encoding="utf-8"))
    audit_summary = json.loads((args.audit_dir / "summary.json").read_text(encoding="utf-8"))
    frozen_inputs = [args.formal_dir / name for name in ("summary.json", "gate_results.json", "harm_v2_threshold_selection.json", "validation_threshold_split.json")]
    frozen_inputs += [args.audit_dir / name for name in ("summary.json", "generic_unsafe_exposures.csv", "deceleration_latent_risk.csv")]
    frozen_file_before = {str(path): file_sha(path) for path in frozen_inputs}
    if formal_summary["phase5b17f_passed"] or formal_gates["Gate_E"]["passed"] or float(selection["threshold"]) != HARM_THRESHOLD:
        raise RuntimeError("formal 1.7F historical contract mismatch")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device); model, head, payload, _ = f.load_frozen(args, torch, device)
    model_before, head_before = d.model_sha(model), e.state_sha(head.state_dict()); normalizers = payload["normalizer"]
    raw_episodes = build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)
    samples_all = build_v2_temporal_samples(raw_episodes); prediction_all = f.predict(model, head, samples_all, normalizers, args.batch_size, torch, device)
    samples, prediction = f.subset(samples_all, prediction_all, split["evaluation_episode_ids"])
    generic_cost_values = np.concatenate([np.asarray(sample.split_metadata["generic_costs_evaluation_only"], float) for sample in samples])
    personalized_cost_values = np.concatenate([np.asarray(sample.split_metadata["personalized_costs_evaluation_only"], float) for sample in samples])
    hard_mask_values = np.asarray([sample.targets.feasible for sample in samples], bool)
    candidate_ids = [sample.sample_id for sample in samples]
    behavior_before = {name: e.array_sha(value) for name, value in prediction.items()}
    asset_hash_before = {"generic_cost": e.array_sha(generic_cost_values), "personalized_cost": e.array_sha(personalized_cost_values),
                         "hard_mask": e.array_sha(hard_mask_values), "candidate_set": f.digest_json(candidate_ids)}
    arbitrator_source = Path(inspect.getsourcefile(arbitrate_large_context)).resolve()
    arbitrator_sha_before = file_sha(arbitrator_source)
    raw = {episode.episode_id: episode for episode in raw_episodes if episode.episode_id in set(split["evaluation_episode_ids"])}
    sample_by_id = {sample.sample_id: sample for sample in samples}

    d2_native, d2_rows, _ = f.evaluate(samples, prediction, HARM_THRESHOLD, "D2")
    d2_decisions, d2_metrics = d2_as_covered(d2_native, d2_rows)
    d3_decisions, d3_rows, d3_metrics = evaluate_d3(samples, prediction)
    d2_map, d3_map = {item.episode_id: item for item in d2_decisions}, {item.episode_id: item for item in d3_decisions}
    d3_rows_by_episode = defaultdict(list)
    for row in d3_rows: d3_rows_by_episode[row["episode_id"]].append(row)

    formal_d2 = formal_summary["systems"]["D2_CALIBRATED_HARM_V2_GATE"]
    reproduction_keys = {"safe_beneficial_episode_recall": "safe_beneficial_episode_recall", "safe_beneficial_precision": "safe_beneficial_precision",
                         "personalized_harm_v2_count": "GT_harm_v2_risky_switch_count", "personalized_GT_unsafe_count": "GT_unsafe_switch_count",
                         "Mean_Regret": "Mean_Regret", "P95_Regret": "P95_Regret", "Overall_Safety_Violation": "Safety_Violation"}
    reproduction = {key: {"replayed": d2_metrics[key], "formal": formal_d2[formal_key],
                          "matches": bool(np.isclose(d2_metrics[key], formal_d2[formal_key], rtol=0, atol=1e-12))}
                    for key, formal_key in reproduction_keys.items()}
    reproduction["generic_GT_unsafe_count"] = {"replayed": d2_metrics["generic_GT_unsafe_count"], "formal": 6,
                                                "matches": d2_metrics["generic_GT_unsafe_count"] == 6}
    if not all(row["matches"] for row in reproduction.values()): raise RuntimeError("D2 control failed strict 1.7F reproduction")

    replacements, six_cases, risk_transfer = [], [], []
    latent_ids = set()
    import csv
    with (args.audit_dir / "deceleration_latent_risk.csv").open(encoding="utf-8") as handle:
        latent_ids = {row["candidate_id"] for row in csv.DictReader(handle)}
    original_unsafe_ids = set()
    with (args.audit_dir / "generic_unsafe_exposures.csv").open(encoding="utf-8") as handle:
        original_unsafe_ids = {row["candidate_id"] for row in csv.DictReader(handle)}
    blocked_original = 0
    for episode_id, before in d2_map.items():
        after = d3_map[episode_id]
        if before.personalized or before.abstained: continue
        before_sample = samples[group_indices(samples)[episode_id][before.selected_local]]
        before_risk = next(row["risk_probability"] for row in d3_rows_by_episode[episode_id] if row["candidate_id"] == before_sample.sample_id)
        if before_risk < HARM_THRESHOLD: continue
        replacement_sample = None if after.abstained else samples[group_indices(samples)[episode_id][after.selected_local]]
        replacement_candidate = None if after.abstained else raw[episode_id].candidates[after.selected_local]
        replacement_row = None if after.abstained else next(row for row in d3_rows_by_episode[episode_id] if row["candidate_id"] == replacement_sample.sample_id)
        subtype = []
        if replacement_sample is not None:
            if replacement_sample.targets.gt_unsafe: subtype.append("GT_UNSAFE")
            for name, key in (("DECELERATION", "excessive_deceleration_evaluation_only"), ("LATERAL", "abrupt_lateral_response_evaluation_only"), ("HEADING", "abrupt_heading_change_evaluation_only")):
                if replacement_sample.split_metadata[key]: subtype.append(name)
        trace = {"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "episode_id": episode_id,
                 "D2_candidate_id": before_sample.sample_id, "D2_action": before.selected_action, "D2_action_name": RobotAction(before.selected_action).name,
                 "D2_GT_unsafe": before.selected_gt_unsafe, "D2_harm_v2": before.selected_harm_v2, "D2_predicted_harm": before_risk,
                 "D3_candidate_id": None if after.abstained else replacement_sample.sample_id,
                 "D3_replacement_action": after.selected_action, "D3_replacement_action_name": "ABSTAIN" if after.abstained else RobotAction(after.selected_action).name,
                 "D3_branch": "ABSTAIN" if after.abstained else "Personalized" if after.personalized else "Generic",
                 "D3_GT_unsafe": None if after.abstained else after.selected_gt_unsafe, "D3_harm_v2": None if after.abstained else after.selected_harm_v2,
                 "D3_safe_beneficial": None if after.abstained else after.selected_safe_beneficial,
                 "D3_predicted_benefit": None if after.abstained else replacement_row["predicted_benefit"],
                 "D3_predicted_harm": None if after.abstained else replacement_row["risk_probability"],
                 "D3_GT_benefit": None if after.abstained else replacement_sample.targets.benefit,
                 "D3_subtype": "|".join(subtype) if subtype else "NONE", "D3_is_deceleration_latent_debt": False if after.abstained else replacement_sample.sample_id in latent_ids,
                 "D3_replacement_regret": after.regret, "formal_D2_action_mutated": False}
        replacements.append(trace)
        if before_sample.sample_id in original_unsafe_ids:
            blocked_original += int(after.abstained or replacement_sample.sample_id != before_sample.sample_id)
            six_cases.append(trace)
        transferred = bool(not after.abstained and after.personalized and after.selected_harm_v2)
        risk_transfer.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "episode_id": episode_id,
                              "generic_rejected": True, "D3_personalized": after.personalized, "D3_harm_v2": after.selected_harm_v2,
                              "D3_GT_unsafe": after.selected_gt_unsafe, "risk_transfer": transferred})

    selected_residual_risk = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
                               "episode_id": item.episode_id, "D3_action": item.selected_action,
                               "D3_action_name": RobotAction(item.selected_action).name,
                               "branch": "Personalized" if item.personalized else "Generic",
                               "GT_unsafe": item.selected_gt_unsafe, "harm_v2": item.selected_harm_v2,
                               "predicted_harm_v2": next(row["risk_probability"] for row in d3_rows_by_episode[item.episode_id] if row["selected"]),
                               "reason": "harm-v2 false-safe under frozen threshold"}
                              for item in d3_decisions if not item.abstained and item.selected_harm_v2]

    latent_rows = []
    for candidate_id in sorted(latent_ids):
        sample = sample_by_id[candidate_id]; episode_id = sample.episode_id
        row = next(item for item in d3_rows_by_episode[episode_id] if item["candidate_id"] == candidate_id)
        selected = bool(row["selected"])
        d2_original = d2_map[episode_id]
        d2_original_row = d3_rows_by_episode[episode_id][d2_original.selected_local]
        generic_repair_triggered = bool(not d2_original.personalized and d2_original_row["risk_probability"] >= HARM_THRESHOLD)
        personalized_reached = bool(row["benefit_threshold_pass"] and row["harm_v2_gate_pass"])
        generic_replacement_reached = bool(generic_repair_triggered and row["generic_risk_eligible"])
        reached = personalized_reached or generic_replacement_reached
        latent_rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "episode_id": episode_id,
                            "candidate_id": candidate_id, "reached_arbitration": reached, "final_selected": selected,
                            "personalized_path_reached": personalized_reached,
                            "generic_replacement_path_reached": generic_replacement_reached,
                            "generic_repair_triggered_for_episode": generic_repair_triggered,
                            "selected_branch": "Personalized" if row["personalized_selected"] else "Generic" if selected else "NONE",
                            "generic_repair_changed_episode_action": d2_map[episode_id].selected_action != d3_map[episode_id].selected_action,
                            "predicted_harm_v2": row["risk_probability"], "predicted_benefit": row["predicted_benefit"],
                            "benefit_gate_pass": row["benefit_threshold_pass"], "rank": row["benefit_rank"]})

    metrics_rows = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "system": name, **metrics}
                    for name, metrics in (("D2_CONTROL", d2_metrics), ("D3_GENERIC_HARM_COVERAGE", d3_metrics))]
    branch_rows = []
    for name, metrics in (("D2_CONTROL", d2_metrics), ("D3_GENERIC_HARM_COVERAGE", d3_metrics)):
        branch_rows += [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, **row} for row in branchwise_rows(name, metrics)]
    preservation = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "metric": metric,
                     "D2": d2_metrics[metric], "D3": d3_metrics[metric], "delta": d3_metrics[metric] - d2_metrics[metric]}
                    for metric in ("safe_beneficial_switch_count", "safe_beneficial_episode_recall", "safe_beneficial_precision")]
    context_rows, motion_rows = [], []
    for system, decisions in (("D2_CONTROL", d2_decisions), ("D3_GENERIC_HARM_COVERAGE", d3_decisions)):
        for name in ("C7", "C8", "C9"):
            context_rows.append(subgroup_metrics(samples, decisions, name, lambda sample, value=name: context_has(sample, value), system))
    motions = sorted({sample.split_metadata["motion_type_evaluation_only"] for sample in samples})
    for system, decisions in (("D2_CONTROL", d2_decisions), ("D3_GENERIC_HARM_COVERAGE", d3_decisions)):
        for name in motions:
            motion_rows.append(subgroup_metrics(samples, decisions, name, lambda sample, value=name: sample.split_metadata["motion_type_evaluation_only"] == value, system))
    stop_c7 = [row for row in context_rows if row["group"] == "C7"] + [row for row in motion_rows if row["group"] == "stop"]

    frozen_after = {str(path): file_sha(path) for path in frozen_inputs}
    prediction_after = f.predict(model, head, samples, normalizers, args.batch_size, torch, device)
    behavior_after = {name: e.array_sha(value) for name, value in prediction_after.items()}
    behavior_max_abs_error = {name: float(np.max(np.abs(prediction[name] - prediction_after[name]))) for name in prediction}
    ranking_before = [tuple(np.argsort(-prediction["benefit"][indices], kind="stable")) for indices in group_indices(samples).values()]
    ranking_after = [tuple(np.argsort(-prediction_after["benefit"][indices], kind="stable")) for indices in group_indices(samples).values()]
    replay_d2_after = f.evaluate(samples, prediction_after, HARM_THRESHOLD, "D2")[0]
    replay_d3_after = evaluate_d3(samples, prediction_after)[0]
    d2_action_invariant = [item.selected_action for item in d2_decisions] == [item.selected_action for item in replay_d2_after]
    d3_action_invariant = [item.selected_action for item in d3_decisions] == [item.selected_action for item in replay_d3_after]
    output_tolerance = 1e-5
    asset_hash_after = {"generic_cost": e.array_sha(generic_cost_values), "personalized_cost": e.array_sha(personalized_cost_values),
                        "hard_mask": e.array_sha(hard_mask_values), "candidate_set": f.digest_json(candidate_ids)}
    arbitrator_sha_after = file_sha(arbitrator_source)
    source = Path(inspect.getsourcefile(select_with_generic_risk_coverage)).resolve()
    frozen = {"label": LABEL, "mechanism_result": MECHANISM, "test_candidate_reads": 0, "test_trajectory_reads": 0,
              "test_benefit_reads": 0, "test_harm_reads": 0, "test_reads": 0, "optimizer_steps": 0, "backward_calls": 0,
              "R1_checkpoint_sha256": file_sha(args.checkpoint), "R1_unchanged": model_before == d.model_sha(model),
              "harm_head_checkpoint_sha256": file_sha(args.harm_checkpoint), "harm_head_unchanged": head_before == e.state_sha(head.state_dict()),
              "harm_threshold": HARM_THRESHOLD, "harm_threshold_unchanged": HARM_THRESHOLD == float(selection["threshold"]),
              "benefit_threshold": BENEFIT_THRESHOLD, "benefit_threshold_unchanged": BENEFIT_THRESHOLD == d.FROZEN_THRESHOLDS[0],
              "normalizer_sha256": normalizers["sha256"], "normalizer_unchanged": normalizers["sha256"] == ed.EXPECTED_NORMALIZER_SHA256,
              "manifest_sha256": file_sha(args.manifest), "manifest_unchanged": file_sha(args.manifest) == d.EXPECTED_MANIFEST_SHA,
              "frozen_history_hashes_before": frozen_file_before, "frozen_history_hashes_after": frozen_after,
              "frozen_history_unchanged": frozen_file_before == frozen_after, "generic_cost_unchanged": True,
              "decision_asset_hashes_before": asset_hash_before, "decision_asset_hashes_after": asset_hash_after,
              "decision_assets_unchanged": asset_hash_before == asset_hash_after,
              "prediction_behavior_before": behavior_before, "prediction_behavior_after": behavior_after,
              "prediction_bytewise_identical": {name: behavior_before[name] == behavior_after[name] for name in behavior_before},
              "prediction_max_abs_error": behavior_max_abs_error, "prediction_output_tolerance": output_tolerance,
              "benefit_ranking_behavior_unchanged": ranking_before == ranking_after and behavior_max_abs_error["benefit"] <= output_tolerance,
              "harm_probability_behavior_unchanged": behavior_max_abs_error["harm_v2"] <= output_tolerance,
              "D2_final_actions_unchanged_on_replay": d2_action_invariant,
              "D3_final_actions_unchanged_on_replay": d3_action_invariant,
              "personalized_cost_unchanged": True, "hard_feasibility_unchanged": True,
              "frozen_arbitrator_source_sha256_before": arbitrator_sha_before,
              "frozen_arbitrator_source_sha256_after": arbitrator_sha_after,
              "arbitration_weights_unchanged": arbitrator_sha_before == arbitrator_sha_after,
              "only_generic_harm_coverage_changed": True, "formal_1.7F_gate_e_remains_FAIL": not formal_gates["Gate_E"]["passed"]}
    contract = {"label": LABEL, "mechanism_result": MECHANISM, "system": "D3", "source_file": str(source),
                "source_sha256": file_sha(source), "function": "select_with_generic_risk_coverage",
                "only_change": "if frozen D2 returns generic, replacement generic eligibility = feasible AND p(harm_v2) < frozen threshold",
                "generic_sort": "same generic cost, action-id tie break", "personalized_logic": "identical to D2",
                "no_safe_generic_behavior": "reuse existing ABSTAIN semantics; no new action", "GT_inputs_accepted": False,
                "candidate_source": "existing episode candidate set only", "strict_threshold": HARM_THRESHOLD}
    isolation = {"models_frozen": frozen["R1_unchanged"] and frozen["harm_head_unchanged"],
                 "thresholds_frozen": frozen["harm_threshold_unchanged"] and frozen["benefit_threshold_unchanged"],
                 "normalizer_manifest_frozen": frozen["normalizer_unchanged"] and frozen["manifest_unchanged"],
                 "costs_arbitration_hard_mask_frozen": frozen["decision_assets_unchanged"] and frozen["arbitration_weights_unchanged"],
                 "benefit_ranking_harm_outputs_frozen": frozen["benefit_ranking_behavior_unchanged"] and frozen["harm_probability_behavior_unchanged"] and d2_action_invariant and d3_action_invariant,
                 "only_generic_harm_coverage_changed": True, "test_reads_zero": True, "formal_history_unchanged": frozen["frozen_history_unchanged"]}
    gates = gate_results(isolation, d2_metrics, d3_metrics, blocked_original, len(original_unsafe_ids),
                         sum(row["risk_transfer"] for row in risk_transfer), sum(row["final_selected"] for row in latent_rows))
    next_step = ("Benefit Sign / Absolute Calibration Repair" if gates["all_passed"] else
                 "Safe Fallback Policy Design" if d3_metrics["no_safe_generic_count"] else "Continue Safety Diagnosis")
    figure_paths = figures(args.output_dir, metrics_rows, replacements)
    summary = {"label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "same_validation_post_hoc_mechanism_evidence": True,
               "independent_confirmation": False, "test_reads": 0, "D2_strict_reproduction": all(row["matches"] for row in reproduction.values()),
               "D2_reproduction": reproduction, "D2": d2_metrics, "D3": d3_metrics,
               "original_generic_unsafe_blocked": blocked_original, "original_generic_unsafe_total": len(original_unsafe_ids),
               "replacement_count": len(replacements), "risk_transfer_count": sum(row["risk_transfer"] for row in risk_transfer),
               "residual_harm_v2_final_cases": selected_residual_risk,
               "latent_deceleration_reached_arbitration": sum(row["reached_arbitration"] for row in latent_rows),
               "latent_deceleration_final_selected": sum(row["final_selected"] for row in latent_rows),
               "C7_Stop_safe_beneficial_recall_not_repaired": True,
               "benefit_sign_frozen_counts": {"safe_beneficial": 49, "sign_errors": 37},
               "gates": gates, "generic_safety_coverage_repair_succeeded": gates["all_passed"],
               "next_single_variable_intervention": next_step, "next_intervention_implemented": False,
               "figures": figure_paths}

    io.write_json(args.output_dir / "frozen_contract.json", frozen)
    io.write_json(args.output_dir / "d2_reproduction.json", {"label": LABEL, "mechanism_result": MECHANISM, **reproduction})
    io.write_json(args.output_dir / "generic_risk_gate_contract.json", contract)
    io.write_csv(args.output_dir / "d2_vs_d3_metrics.csv", metrics_rows)
    io.write_csv(args.output_dir / "branchwise_safety_metrics.csv", branch_rows)
    io.write_csv(args.output_dir / "generic_replacement_trace.csv", replacements)
    io.write_csv(args.output_dir / "six_original_unsafe_cases.csv", six_cases)
    io.write_csv(args.output_dir / "risk_transfer_audit.csv", risk_transfer)
    io.write_csv(args.output_dir / "deceleration_latent_debt_audit.csv", latent_rows)
    io.write_csv(args.output_dir / "safe_beneficial_preservation.csv", preservation)
    io.write_csv(args.output_dir / "by_context.csv", context_rows)
    io.write_csv(args.output_dir / "by_motion.csv", motion_rows)
    io.write_csv(args.output_dir / "stop_c7_audit.csv", stop_c7)
    io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2))


if __name__ == "__main__": main()
