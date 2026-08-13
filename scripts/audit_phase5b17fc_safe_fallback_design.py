"""Phase 5B-1.7F-C safe-fallback semantics/support audit (development only)."""
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as e
from scripts import run_phase5b17ed_risk_preserving_bypass as ed
from scripts import run_phase5b17f_safe_decision_chain as f
from scripts import run_phase5b17fb_generic_safety_coverage as fb
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import ACTION_DEFINITIONS, PHASE4A_ACTIONS, RobotAction
from src.decision.action_selector import rule_based_select
from src.decision.fallback_policy import constrained_select_with_fallback
from src.decision.generic_risk_coverage import select_with_generic_risk_coverage
from src.decision.safety_gate import apply_safety_gate, choose_fallback_action
from src.evaluation.safe_fallback_design import (
    LABEL, MECHANISM, classify_abstain_semantics,
    oracle_safe_fallback_availability, safe_fallback_gates,
)
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples

STAGE = "Phase 5B-1.7F-C Safe Fallback Semantics & Support Design"
HARM_THRESHOLD = 0.10968538373708725
FB_DIR = PROJECT_ROOT / "results_dev" / "phase5b17fb_generic_safety_coverage"
FORMAL_DIR = PROJECT_ROOT / "results_dev" / "phase5b17f_safe_decision_chain"
ATTRIBUTION_DIR = PROJECT_ROOT / "results_dev" / "phase5b17fa_safety_attribution_audit"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ed_risk_preserving_bypass" / "checkpoints" / "harm_v2_risk_bypass_head.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--phase5b17fb-dir", type=Path, default=FB_DIR)
    parser.add_argument("--formal-dir", type=Path, default=FORMAL_DIR)
    parser.add_argument("--attribution-dir", type=Path, default=ATTRIBUTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17fc_safe_fallback_design")
    return parser.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_entry(name, obj, **values):
    path = Path(inspect.getsourcefile(obj)).resolve()
    return {"behavior": name, "source_file": str(path), "source_sha256": sha(path),
            "function": obj.__name__, **values}


def existing_registry():
    common = {"candidate_set": [action.name for action in PHASE4A_ACTIONS]}
    abstain = source_entry(
        "ABSTAIN", select_with_generic_risk_coverage, action_ID=None,
        control_semantics="selected_index=None and selected_action=None; reports that no eligible candidate exists",
        is_candidate=False, is_executable_robot_action=False, deterministic_robot_rollout=False,
        human_response_rollout=False, GT_cost_available=False, GT_unsafe_available=False,
        harm_v2_label_available=False, regret_computable=False,
        protocol_defined_safe_semantics=False, note="Evaluation placeholder; neither KEEP nor STOP/HOLD", **common,
    )
    keep = source_entry(
        "KEEP", ACTION_DEFINITIONS[RobotAction.KEEP].__class__, action_ID=int(RobotAction.KEEP),
        control_semantics="Keep the current high-level robot speed/yaw evolution; speed_scale_delta=0, distance_offset=0, lateral_offset=0",
        is_candidate=True, is_executable_robot_action=True, deterministic_robot_rollout=True,
        human_response_rollout=True, GT_cost_available=True, GT_unsafe_available=True,
        harm_v2_label_available=True, regret_computable=True,
        protocol_defined_safe_semantics=False,
        note="Robot rollout continues current linear/angular motion; KEEP is not zero velocity and is not guaranteed safe",
        speed_scale_delta=0.0, distance_offset_m=0.0, lateral_offset_m=0.0, **common,
    )
    rule = source_entry(
        "RULE_FALLBACK", constrained_select_with_fallback, action_ID=None,
        control_semantics="Routing policy over DISTANCE_PLUS_0_2/SPEED_DOWN_10/KEEP that must use its supplied feasible mask",
        is_candidate=False, is_executable_robot_action=False, deterministic_robot_rollout=False,
        human_response_rollout=False, GT_cost_available=False, GT_unsafe_available=False,
        harm_v2_label_available=False, regret_computable=False,
        protocol_defined_safe_semantics=False,
        note="Policy is not an action; selected candidates have rollout support, but no candidate has declared stop/hold safety semantics", **common,
    )
    legacy = source_entry(
        "LEGACY_GEOMETRY_CONSERVATIVE_FALLBACK", apply_safety_gate, action_ID=None,
        control_semantics="Routes to an existing feasible DISTANCE_PLUS_0_2/SPEED_DOWN_10/KEEP candidate when an older prediction gate is empty",
        is_candidate=False, is_executable_robot_action=False, deterministic_robot_rollout=False,
        human_response_rollout=False, GT_cost_available=False, GT_unsafe_available=False,
        harm_v2_label_available=False, regret_computable=False,
        protocol_defined_safe_semantics=False,
        note="Not the frozen D3 harm-v2 contract and not a STOP/HOLD action", **common,
    )
    min_risk = source_entry(
        "FALLBACK_MIN_RISK", choose_fallback_action, action_ID=None,
        control_semantics="Legacy policy selects the feasible candidate with minimum predicted risk",
        is_candidate=False, is_executable_robot_action=False, deterministic_robot_rollout=False,
        human_response_rollout=False, GT_cost_available=False, GT_unsafe_available=False,
        harm_v2_label_available=False, regret_computable=False,
        protocol_defined_safe_semantics=False, prohibited_in_stage=True,
        note="Minimum predicted risk is not verified safety and is explicitly forbidden in Phase 5B-1.7F-C", **common,
    )
    absent = []
    for name in ("STOP", "HOLD", "ZERO_VELOCITY_HOLD", "EMERGENCY_STOP"):
        absent.append({
            "behavior": name, "source_file": None, "source_sha256": None, "function": None,
            "action_ID": None, "control_semantics": None, "exists": False,
            "is_candidate": False, "is_executable_robot_action": False,
            "deterministic_robot_rollout": False, "human_response_rollout": False,
            "GT_cost_available": False, "GT_unsafe_available": False,
            "harm_v2_label_available": False, "regret_computable": False,
            "protocol_defined_safe_semantics": False,
            "note": "No corresponding action exists in RobotAction/PHASE4A_ACTIONS",
            **common,
        })
    for item in (abstain, keep, rule, legacy, min_risk):
        item["exists"] = True
    return [abstain, keep, rule, legacy, min_risk, *absent]


def group_indices(samples):
    result = defaultdict(list)
    for index, sample in enumerate(samples):
        result[sample.episode_id].append(index)
    return dict(result)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Stage 1.7F-C: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prior_paths = [
        args.phase5b17fb_dir / "summary.json", args.phase5b17fb_dir / "gate_results.json",
        args.phase5b17fb_dir / "frozen_contract.json",
        args.formal_dir / "harm_v2_threshold_selection.json",
        args.formal_dir / "validation_threshold_split.json",
        args.attribution_dir / "deceleration_latent_risk.csv",
    ]
    prior_sha_before = {str(path): sha(path) for path in prior_paths}
    prior_summary = json.loads(prior_paths[0].read_text(encoding="utf-8"))
    split = json.loads((args.formal_dir / "validation_threshold_split.json").read_text(encoding="utf-8"))
    threshold = json.loads((args.formal_dir / "harm_v2_threshold_selection.json").read_text(encoding="utf-8"))
    if float(threshold["threshold"]) != HARM_THRESHOLD or prior_summary["D3"]["no_safe_generic_count"] != 14:
        raise RuntimeError("frozen Phase 5B-1.7F-B contract mismatch")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model, head, payload, _ = f.load_frozen(args, torch, device)
    model_before, head_before = d.model_sha(model), e.state_sha(head.state_dict())
    normalizers = payload["normalizer"]
    raw_all = build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)
    samples_all = build_v2_temporal_samples(raw_all)
    prediction_all = f.predict(model, head, samples_all, normalizers, args.batch_size, torch, device)
    samples, prediction = f.subset(samples_all, prediction_all, split["evaluation_episode_ids"])
    raw = {episode.episode_id: episode for episode in raw_all if episode.episode_id in set(split["evaluation_episode_ids"])}

    d3_decisions, d3_rows, d3_metrics = fb.evaluate_d3(samples, prediction)
    if d3_metrics["episode_count"] != 120 or d3_metrics["no_safe_generic_count"] != 14:
        raise RuntimeError("D3 no-safe-generic episodes do not reproduce 120/14")
    decision_by_id = {item.episode_id: item for item in d3_decisions}
    no_safe_ids = {item.episode_id for item in d3_decisions if item.abstained}
    indices_by_episode = group_indices(samples)
    rows_by_episode = defaultdict(list)
    for row in d3_rows:
        rows_by_episode[row["episode_id"]].append(row)

    candidate_audit = []
    for episode_id in sorted(no_safe_ids):
        episode = raw[episode_id]
        sample_indices = indices_by_episode[episode_id]
        selected = [samples[index] for index in sample_indices]
        all_actions = np.asarray(selected[0].split_metadata["all_action_ids_evaluation_only"], int)
        gt_costs = np.asarray(selected[0].split_metadata["gt_costs_evaluation_only"], float)
        generic_costs = np.asarray(selected[0].split_metadata["generic_costs_evaluation_only"], float)
        oracle_cost = float(gt_costs.min())
        for local, sample in enumerate(selected):
            action_id = int(sample.split_metadata["candidate_action_id_audit"])
            full = int(np.flatnonzero(all_actions == action_id)[0])
            prediction_row = rows_by_episode[episode_id][local]
            candidate_audit.append({
                "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
                "episode_id": episode_id, "candidate_id": sample.sample_id,
                "candidate_action_id": action_id, "action": RobotAction(action_id).name,
                "feasible": bool(sample.targets.feasible),
                "predicted_harm": float(prediction_row["risk_probability"]),
                "harm_threshold": HARM_THRESHOLD,
                "runtime_harm_gate_pass": bool(prediction_row["risk_probability"] < HARM_THRESHOLD),
                "GT_harm_v2": bool(sample.split_metadata["harm_v2_evaluation_only"]),
                "GT_unsafe": bool(sample.targets.gt_unsafe), "GT_benefit": float(sample.targets.benefit),
                "generic_cost": float(generic_costs[full]), "GT_total_cost": float(gt_costs[full]),
                "regret": float(gt_costs[full] - oracle_cost),
                "motion": str(sample.split_metadata["motion_type_evaluation_only"]),
                "context": "|".join(map(str, sample.split_metadata["contexts_evaluation_only"])) or "NONE",
                "oracle_GT_safe_candidate": bool(sample.targets.feasible and not sample.targets.gt_unsafe and not sample.split_metadata["harm_v2_evaluation_only"]),
                "oracle_use_only_not_runtime": True,
            })

    oracle_rows, oracle_summary = oracle_safe_fallback_availability(candidate_audit)
    for row in oracle_rows:
        row.update({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM})
        episode_candidates = [item for item in candidate_audit if item["episode_id"] == row["episode_id"]]
        safe = [item for item in episode_candidates if item["oracle_GT_safe_candidate"]]
        row["safe_candidates_rejected_by_runtime_risk_count"] = sum(not item["runtime_harm_gate_pass"] for item in safe)
        row["all_oracle_safe_candidates_rejected_by_runtime_risk"] = bool(safe) and all(not item["runtime_harm_gate_pass"] for item in safe)

    registry = existing_registry()
    abstain_semantics = classify_abstain_semantics(
        selected_index=None, selected_action=None,
        candidate_rollout_available=False, candidate_cost_available=False,
    )
    abstain_semantics.update({
        "label": LABEL, "mechanism_result": MECHANISM,
        "D3_reason": "NO_SAFE_GENERIC_CANDIDATE", "episode_count": len(no_safe_ids),
        "KEEP_equivalent": False, "STOP_equivalent": False, "HOLD_equivalent": False,
        "legacy_phase4c2_penalty_is_rollout_cost": False,
        "legacy_phase4c2_note": "A prior development evaluation assigned a 0.25 inability-to-decide penalty, but no action/robot/human rollout backs it.",
    })
    fallback_contract = {
        "label": LABEL, "mechanism_result": MECHANISM, "contract_name": "SAFE_FALLBACK_V1",
        "status": "NOT_DEFINED", "safe_fallback_ready": False,
        "reason": "No existing STOP/HOLD/zero-motion candidate has executable semantics plus deterministic robot/human rollout and GT cost/labels.",
        "KEEP_rejected_reason": "KEEP continues current robot linear/angular motion and has no protocol-defined safety guarantee.",
        "ABSTAIN_rejected_reason": "ABSTAIN has no action ID, rollout, or GT cost.",
        "runtime_selection_implemented": False, "new_action_created": False,
        "GT_runtime_inputs": False, "min_risk_heuristic_used": False,
    }

    support_rows = []
    for episode_id in sorted(no_safe_ids):
        support_rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
            "episode_id": episode_id, "fallback_available": False,
            "fallback_action_ID": None, "fallback_action": None,
            "fallback_predicted_harm": None, "fallback_GT_harm_v2": None,
            "fallback_GT_unsafe": None, "fallback_GT_cost": None,
            "fallback_regret": None, "reason": "NO VERIFIED SAFE FALLBACK",
        })
    safety_rows = [{**row, "safety_claim_permitted": False} for row in support_rows]
    regret_rows = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
                    "episode_id": row["episode_id"], "defined_final_action": False,
                    "defined_GT_cost": False, "defined_regret": False,
                    "reason": "ABSTAIN is an evaluation placeholder; no cost was imputed"}
                   for row in support_rows]

    residual = next(item for item in d3_decisions if item.episode_id.endswith("000194"))
    residual_candidates = []
    for row in rows_by_episode[residual.episode_id]:
        sample = samples[indices_by_episode[residual.episode_id][row["local_index"]]]
        residual_candidates.append({
            "candidate_id": sample.sample_id, "action": RobotAction(row["action"]).name,
            "predicted_harm": row["risk_probability"], "harm_v2": row["harm_v2"],
            "GT_unsafe": row["gt_unsafe"], "generic_cost": float(sample.split_metadata["generic_costs_evaluation_only"][row["action"]]),
            "selected": row["selected"],
        })
    selected_residual = next(row for row in residual_candidates if row["selected"])
    residual_json = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "episode_id": residual.episode_id, "status": "RESIDUAL HARM-V2 FALSE-SAFE",
        "action": RobotAction(residual.selected_action).name,
        "motion": raw[residual.episode_id].motion_type,
        "subtype": "|".join(
            name for name, active in raw[residual.episode_id].candidates[residual.selected_local].events.__dict__.items()
            if isinstance(active, (bool, np.bool_)) and active
        ) or "NONE",
        "risk_probability": selected_residual["predicted_harm"], "harm_threshold": HARM_THRESHOLD,
        "GT_harm_v2": residual.selected_harm_v2, "GT_unsafe": residual.selected_gt_unsafe,
        "generic_cost": selected_residual["generic_cost"], "alternative_candidates": residual_candidates,
        "fallback_gate_scope": False,
        "warning": "This is a risk-head false-safe, not a missing-fallback event; Phase 5B-1.7F-C does not alter it.",
    }

    import csv
    with (args.attribution_dir / "deceleration_latent_risk.csv").open(encoding="utf-8") as handle:
        latent_source = list(csv.DictReader(handle))
    latent_rows = []
    for source in latent_source:
        candidate_id = source["candidate_id"]
        sample = next(sample for sample in samples if sample.sample_id == candidate_id)
        row = next(item for item in rows_by_episode[sample.episode_id] if item["candidate_id"] == candidate_id)
        latent_rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
            "episode_id": sample.episode_id, "candidate_id": candidate_id,
            "action": RobotAction(int(row["action"])).name, "predicted_harm": float(row["risk_probability"]),
            "harm_threshold": HARM_THRESHOLD, "latent_deceleration": True,
            "reached_arbitration": False, "final_selected": bool(row["selected"]),
            "fallback_design_changed_selection": False,
        })

    defined_existing = d3_metrics["selected_episode_count"]
    gates = safe_fallback_gates(
        semantics_valid=False, rollout_evaluation_supported=False,
        original_no_safe_count=len(no_safe_ids), remaining_undefined_count=len(no_safe_ids),
        fallback_gt_unsafe_count=None, fallback_harm_v2_count=None,
        personalized_risky_selected=d3_metrics["personalized_harm_v2_count"],
        latent_deceleration_selected=sum(row["final_selected"] for row in latent_rows),
        evaluation_episode_count=d3_metrics["episode_count"], defined_action_count=defined_existing,
        defined_gt_cost_count=defined_existing, defined_regret_count=defined_existing,
    )
    prior_sha_after = {str(path): sha(path) for path in prior_paths}
    frozen = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE,
        "test_candidate_reads": 0, "test_trajectory_reads": 0, "test_benefit_reads": 0,
        "test_harm_reads": 0, "test_reads": 0, "optimizer_steps": 0, "backward_calls": 0,
        "R1_checkpoint_sha256": sha(args.checkpoint), "R1_model_checksum_unchanged": model_before == d.model_sha(model),
        "harm_head_checkpoint_sha256": sha(args.harm_checkpoint), "harm_head_checksum_unchanged": head_before == e.state_sha(head.state_dict()),
        "harm_threshold": HARM_THRESHOLD, "threshold_unchanged": HARM_THRESHOLD == float(threshold["threshold"]),
        "benefit_threshold": f.BENEFIT_THRESHOLD, "benefit_logic_unchanged": True,
        "ranking_unchanged": True, "generic_cost_unchanged": True, "arbitration_unchanged": True,
        "hard_feasibility_unchanged": True, "manifest_sha256": sha(args.manifest),
        "manifest_unchanged": sha(args.manifest) == d.EXPECTED_MANIFEST_SHA,
        "normalizer_sha256": normalizers["sha256"], "normalizer_unchanged": normalizers["sha256"] == ed.EXPECTED_NORMALIZER_SHA256,
        "prior_artifact_hashes_before": prior_sha_before, "prior_artifact_hashes_after": prior_sha_after,
        "prior_artifacts_unchanged": prior_sha_before == prior_sha_after,
        "new_action_created": False, "runtime_policy_changed": False,
    }
    summary = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE,
        "result": "SAFE FALLBACK NOT READY", "safe_fallback_v1_defined": False,
        "D3_reproduced": True, "evaluation_episode_count": d3_metrics["episode_count"],
        "no_safe_generic_count": len(no_safe_ids), "remaining_undefined_count": len(no_safe_ids),
        "abstain_semantics": abstain_semantics["classification"],
        "oracle_safe_fallback_availability": oracle_summary,
        "oracle_safe_candidates_rejected_by_runtime_risk": sum(row["safe_candidates_rejected_by_runtime_risk_count"] for row in oracle_rows),
        "fallback_available_count": 0, "fallback_GT_unsafe_count": None,
        "fallback_harm_v2_count": None, "all_120_regret_defined": False,
        "Mean_Regret": None, "P95_Regret": None,
        "deceleration_latent_debt_count": len(latent_rows),
        "deceleration_latent_debt_final_selected": sum(row["final_selected"] for row in latent_rows),
        "residual_false_safe_episode": residual.episode_id,
        "residual_false_safe_unchanged": True, "gates": gates,
        "next_single_recommendation": "Action-space / Generator Safe-Hold Candidate Extension",
        "next_intervention_implemented": False,
    }

    io.write_json(args.output_dir / "frozen_contract.json", frozen)
    io.write_json(args.output_dir / "existing_fallback_registry.json", {"label": LABEL, "mechanism_result": MECHANISM, "entries": registry})
    io.write_json(args.output_dir / "abstain_semantics.json", abstain_semantics)
    io.write_csv(args.output_dir / "fourteen_episode_candidate_audit.csv", candidate_audit)
    io.write_csv(args.output_dir / "oracle_safe_fallback_availability.csv", oracle_rows)
    io.write_json(args.output_dir / "safe_fallback_v1_contract.json", fallback_contract)
    io.write_csv(args.output_dir / "fallback_support.csv", support_rows)
    io.write_csv(args.output_dir / "fallback_safety.csv", safety_rows)
    io.write_csv(args.output_dir / "fallback_regret.csv", regret_rows)
    io.write_json(args.output_dir / "residual_false_safe_000194.json", residual_json)
    io.write_csv(args.output_dir / "deceleration_latent_debt_audit.csv", latent_rows)
    io.write_json(args.output_dir / "gate_results.json", {
        "label": LABEL, "mechanism_result": MECHANISM, **gates,
    })
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2))


if __name__ == "__main__":
    main()
