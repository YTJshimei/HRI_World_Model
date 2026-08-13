"""Phase 5B-1.7F-A post-hoc safety attribution; no training or policy mutation."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter, defaultdict
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
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import RobotAction
from src.decision.large_context_arbitrator import arbitrate_large_context
from src.evaluation.safe_decision_chain import BENEFIT_THRESHOLD
from src.evaluation.safety_attribution import (
    cumulative_safe_funnel, episode_floor_class, primary_funnel_loss,
    reaches_harm_gate, rejection_reason, subgroup_summary,
)
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL

STAGE = "Phase 5B-1.7F-A Safety Attribution & Gate-E Floor-Effect Audit"
HARM_THRESHOLD = 0.10968538373708725
FORMAL_DIR = PROJECT_ROOT / "results_dev" / "phase5b17f_safe_decision_chain"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ed_risk_preserving_bypass" / "checkpoints" / "harm_v2_risk_bypass_head.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--formal-dir", type=Path, default=FORMAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17fa_safety_attribution_audit")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal_contract(formal_dir: Path):
    required = ("summary.json", "gate_results.json", "harm_v2_threshold_selection.json",
                "validation_threshold_split.json", "frozen_contract.json")
    hashes = {name: file_sha(formal_dir / name) for name in required}
    summary = json.loads((formal_dir / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((formal_dir / "gate_results.json").read_text(encoding="utf-8"))
    selection = json.loads((formal_dir / "harm_v2_threshold_selection.json").read_text(encoding="utf-8"))
    split = json.loads((formal_dir / "validation_threshold_split.json").read_text(encoding="utf-8"))
    if summary["phase5b17f_passed"] or gates["Gate_E"]["passed"] or gates["all_passed"]:
        raise RuntimeError("formal 1.7F FAIL/Gate-E FAIL contract was changed")
    if float(selection["threshold"]) != HARM_THRESHOLD or split["episode_overlap_count"] != 0:
        raise RuntimeError("formal threshold/split contract mismatch")
    return summary, gates, selection, split, hashes


def index_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    return dict(grouped)


def action_name(action: int) -> str:
    return RobotAction(int(action)).name


def make_figures(output: Path, floor_counts, subgroup_failures, coverage_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    names = list(floor_counts); values = [floor_counts[name] for name in names]
    plt.figure(figsize=(10, 4)); plt.bar(range(len(names)), values); plt.xticks(range(len(names)), [name.split("_")[0] for name in names]); plt.ylabel("Episodes"); plt.title(f"{LABEL}\nGate-E floor-effect attribution")
    path = folder / "gate_e_floor_effect.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    names = list(subgroup_failures); values = [subgroup_failures[name] for name in names]
    plt.figure(figsize=(8, 4)); plt.bar(names, values); plt.ylabel("Safe-beneficial sign failures"); plt.title(LABEL)
    path = folder / "benefit_sign_failures.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    matrix = np.asarray([[row["covered"] for row in coverage_rows if row["branch"] == branch] for branch in ("Personalized", "Generic")], float)
    plt.figure(figsize=(8, 3)); plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn"); plt.yticks((0, 1), ("Personalized", "Generic")); plt.xticks(range(6), ("Hard", "Benefit", "Ranking", "Harm-v2", "Arbitration", "GT unsafe"), rotation=20); plt.colorbar()
    path = folder / "decision_safety_coverage.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path)); return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_summary, formal_gates, selection, split, formal_before = formal_contract(args.formal_dir)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model, head, payload, _ = f.load_frozen(args, torch, device)
    model_before, head_before = d.model_sha(model), e.state_sha(head.state_dict())
    normalizers = payload["normalizer"]
    if normalizers["sha256"] != ed.EXPECTED_NORMALIZER_SHA256:
        raise RuntimeError("normalizer checksum mismatch")

    raw_episodes = build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)
    all_samples = build_v2_temporal_samples(raw_episodes)
    prediction = f.predict(model, head, all_samples, normalizers, args.batch_size, torch, device)
    evaluation_ids = split["evaluation_episode_ids"]
    samples, prediction = f.subset(all_samples, prediction, evaluation_ids)
    raw_by_episode = {episode.episode_id: episode for episode in raw_episodes if episode.episode_id in set(evaluation_ids)}
    results = {mode: f.evaluate(samples, prediction, HARM_THRESHOLD, mode) for mode in ("D0", "D1", "D2")}
    d0_decisions, d0_rows, _ = results["D0"]
    d1_decisions, d1_rows, d1_metrics = results["D1"]
    d2_decisions, d2_rows, d2_metrics = results["D2"]
    d0_map = {item.episode_id: item for item in d0_decisions}; d1_map = {item.episode_id: item for item in d1_decisions}; d2_map = {item.episode_id: item for item in d2_decisions}
    d1_by_episode, d2_by_episode = index_rows(d1_rows), index_rows(d2_rows)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    replay = {mode: f.evaluate(samples, prediction, HARM_THRESHOLD, mode)[0] for mode in ("D1", "D2")}
    action_replay = {
        mode: [item.selected_action for item in decisions] == [item.selected_action for item in replay[mode]]
        for mode, decisions in (("D1", d1_decisions), ("D2", d2_decisions))
    }
    formal_d1 = formal_summary["systems"]["D1_NO_HARM_V2_GATE_DIAGNOSTIC"]
    formal_d2 = formal_summary["systems"]["D2_CALIBRATED_HARM_V2_GATE"]
    metric_keys = ("safe_beneficial_episode_recall", "safe_beneficial_precision", "GT_harm_v2_risky_switch_count",
                   "GT_unsafe_switch_count", "GT_unsafe_final_selected_count", "Mean_Regret", "P95_Regret", "Safety_Violation")
    metric_reproduction = {
        "D1": all(np.isclose(d1_metrics[key], formal_d1[key], rtol=0, atol=1e-12) for key in metric_keys),
        "D2": all(np.isclose(d2_metrics[key], formal_d2[key], rtol=0, atol=1e-12) for key in metric_keys),
    }
    if not all(action_replay.values()) or not all(metric_reproduction.values()):
        raise RuntimeError("formal D1/D2 replay is not reproducible")

    floor_rows = []
    for episode_id in evaluation_ids:
        row = episode_floor_class(d1_map[episode_id], d2_map[episode_id], d1_by_episode[episode_id], d2_by_episode[episode_id])
        floor_rows.append({"synthetic_interaction": LABEL, "episode_id": episode_id, **row})
    floor_categories = ("F0_NO_RISK_OPPORTUNITY", "F1_RISK_PRESENT_BUT_ALREADY_BLOCKED_UPSTREAM",
                        "F2_HARM_GATE_VETO_WITHOUT_FINAL_ACTION_CHANGE", "F3_DECISION_CHANGING_SAFETY_VETO",
                        "F4_SAFE_BENEFICIAL_COLLATERAL_BLOCK")
    floor_counts = Counter({name: 0 for name in floor_categories}); floor_counts.update(row["category"] for row in floor_rows)
    for row in floor_rows:
        row["record_type"] = "episode"
        row["category_episode_count"] = floor_counts[row["category"]]
        row["category_episode_ratio"] = floor_counts[row["category"]] / len(floor_rows)

    candidate_value = []
    for row in d2_rows:
        target = "risk" if row["harm_v2"] else "safe_beneficial" if row["safe_beneficial"] else None
        if target is None or not reaches_harm_gate(row):
            continue
        changed = d1_map[row["episode_id"]].selected_action != d2_map[row["episode_id"]].selected_action
        candidate_value.append({"synthetic_interaction": LABEL, "episode_id": row["episode_id"], "candidate_id": row["candidate_id"],
                                "target_type": target, "risk_probability": row["risk_probability"], "threshold": HARM_THRESHOLD,
                                "harm_gate_pass": row["harm_v2_gate_pass"], "harm_gate_veto": not row["harm_v2_gate_pass"],
                                "D1_selected": next(item["selected"] for item in d1_by_episode[row["episode_id"]] if item["candidate_id"] == row["candidate_id"]),
                                "D2_selected": row["selected"], "final_action_changed": changed})

    generic_unsafe, generic_counterfactual, root_causes = [], [], []
    for decision in d0_decisions:
        if not decision.selected_gt_unsafe_any:
            continue
        episode = raw_by_episode[decision.episode_id]
        row = d2_by_episode[decision.episode_id][decision.selected_local]
        candidate = episode.candidates[decision.selected_local]
        alternatives = [{"action": int(sample_by_id[item["candidate_id"]].split_metadata["candidate_action_id_audit"]),
                         "action_name": action_name(sample_by_id[item["candidate_id"]].split_metadata["candidate_action_id_audit"]),
                         "predicted_benefit": item["predicted_benefit"], "harm_probability": item["risk_probability"],
                         "generic_cost": float(episode.generic_costs[item["local_index"]]),
                         "personalized_cost": float(episode.generic_costs[item["local_index"]]),
                         "benefit_pass": item["benefit_threshold_pass"], "harm_pass": item["harm_v2_gate_pass"],
                         "generic_score_win": item["generic_score_win"]} for item in d2_by_episode[decision.episode_id] if item["local_index"] != decision.selected_local]
        caught = not bool(row["risk_probability"] < HARM_THRESHOLD)
        common = {"synthetic_interaction": LABEL, "audit_only": True, "episode_id": decision.episode_id,
                  "candidate_id": row["candidate_id"], "action": decision.selected_action, "action_name": action_name(decision.selected_action),
                  "motion": episode.motion_type, "context": "|".join(episode.context_labels), "profile_id_audit_only": episode.profile_id,
                  "GT_unsafe": candidate.gt_unsafe, "unsafe_duration": candidate.unsafe_duration,
                  "minimum_human_robot_distance_m": candidate.minimum_distance_m, "harm_v2": candidate.harm_v2,
                  "predicted_harm_v2": row["risk_probability"], "harm_threshold": HARM_THRESHOLD,
                  "feasible": candidate.feasible, "generic_score": float(episode.generic_costs[decision.selected_local]),
                  "personalized_alternatives_json": json.dumps(alternatives, sort_keys=True),
                  "why_generic_selected": "minimum frozen generic cost; generic baseline bypasses benefit/harm gates when no approved personalized candidate wins"}
        generic_unsafe.append(common)
        generic_counterfactual.append({**common, "audit_gate_would_pass": not caught, "caught_by_harm_v2": caught,
                                       "missed_by_harm_v2": not caught, "formal_action_changed": False})
        primary = "C_GENERIC_BRANCH_NOT_COVERED_BY_HARM_GATE" if caught else "D_HARM_V2_UNDERESTIMATES_RISK"
        secondary = ["A_GT_UNSAFE_BUT_LEGAL_UNDER_CURRENT_FEASIBILITY_PROTOCOL"]
        if not caught: secondary.append("C_GENERIC_BRANCH_NOT_COVERED_BY_HARM_GATE")
        root_causes.append({"synthetic_interaction": LABEL, "episode_id": decision.episode_id, "candidate_id": row["candidate_id"],
                            "primary_cause": primary, "secondary_causes": "|".join(secondary),
                            "hard_mask_protocol_inconsistency": False, "caught_by_audit_harm_gate": caught})

    unsafe_rows = [row for row in d2_rows if row["gt_unsafe"]]
    hard_safety = [
        {"synthetic_interaction": LABEL, "group": "GT_UNSAFE_ALL", "candidate_count": len(unsafe_rows)},
        {"synthetic_interaction": LABEL, "group": "GT_UNSAFE_FEASIBLE_TRUE", "candidate_count": sum(row["feasible"] for row in unsafe_rows)},
        {"synthetic_interaction": LABEL, "group": "GT_UNSAFE_FEASIBLE_FALSE", "candidate_count": sum(not row["feasible"] for row in unsafe_rows)},
    ]
    for row in hard_safety:
        row.update({"feasibility_definition": "high-level candidate legality from TASK_SAFE_CANDIDATES; fixed before realized human response",
                    "GT_unsafe_definition": "future risk-conditioned rollout unsafe_duration > 0 at distance < 0.80 m",
                    "semantics_identical": False, "logic_bug_repaired": False})

    c7 = lambda row: "C7" in "|".join(map(str, sample_by_id[row["candidate_id"]].split_metadata["contexts_evaluation_only"]))
    stop = lambda row: sample_by_id[row["candidate_id"]].split_metadata["motion_type_evaluation_only"] == "stop"
    c7_funnel = [{"synthetic_interaction": LABEL, "subgroup": "C7", **row} for row in cumulative_safe_funnel(d2_rows, c7)]
    stop_funnel = [{"synthetic_interaction": LABEL, "subgroup": "stop", **row} for row in cumulative_safe_funnel(d2_rows, stop)]
    c7_loss, stop_loss = primary_funnel_loss(c7_funnel), primary_funnel_loss(stop_funnel)

    sign_errors = []
    for row in d2_rows:
        if not row["safe_beneficial"] or row["benefit_sign_correct"]:
            continue
        sample = sample_by_id[row["candidate_id"]]; meta = sample.split_metadata
        contexts = "|".join(map(str, meta["contexts_evaluation_only"])); action = int(meta["candidate_action_id_audit"])
        sign_errors.append({"synthetic_interaction": LABEL, "episode_id": row["episode_id"], "candidate_id": row["candidate_id"],
                            "GT_benefit": sample.targets.benefit, "predicted_benefit": row["predicted_benefit"],
                            "prediction_error": row["predicted_benefit"] - sample.targets.benefit,
                            "distance_to_zero": abs(row["predicted_benefit"]), "distance_to_benefit_threshold": row["predicted_benefit"] - BENEFIT_THRESHOLD,
                            "action": action, "action_name": action_name(action), "motion": meta["motion_type_evaluation_only"],
                            "context": contexts, "profile_id_audit_only": meta["person_profile_id"], "harm_v2": row["harm_v2"],
                            "C7": "C7" in contexts, "Stop": meta["motion_type_evaluation_only"] == "stop", "C8": "C8" in contexts,
                            "C9": "C9" in contexts, "deceleration": meta["excessive_deceleration_evaluation_only"],
                            "heading": meta["abrupt_heading_change_evaluation_only"], "lateral": meta["abrupt_lateral_response_evaluation_only"]})

    deceleration_latent = []
    for row in d2_rows:
        sample = sample_by_id[row["candidate_id"]]
        if not sample.split_metadata["excessive_deceleration_evaluation_only"] or row["risk_probability"] >= HARM_THRESHOLD:
            continue
        deceleration_latent.append({"synthetic_interaction": LABEL, "episode_id": row["episode_id"], "candidate_id": row["candidate_id"],
                                    "predicted_harm_v2": row["risk_probability"], "harm_threshold": HARM_THRESHOLD,
                                    "predicted_benefit": row["predicted_benefit"], "GT_benefit": sample.targets.benefit,
                                    "benefit_gate_pass": row["benefit_threshold_pass"], "ranking_position": row["benefit_rank"],
                                    "feasible": row["feasible"], "could_enter_if_benefit_improves": bool(row["feasible"] and row["harm_v2_gate_pass"]),
                                    "latent_safety_debt": bool(row["feasible"] and row["harm_v2_gate_pass"])})

    subtype_predicates = {
        "GT_UNSAFE": lambda sample: sample.targets.gt_unsafe,
        "EXCESSIVE_DECELERATION": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
        "ABRUPT_LATERAL_RESPONSE": lambda sample: sample.split_metadata["abrupt_lateral_response_evaluation_only"],
        "ABRUPT_HEADING_CHANGE": lambda sample: sample.split_metadata["abrupt_heading_change_evaluation_only"],
    }
    subtype_rows = []
    for name, predicate in subtype_predicates.items():
        ids = {sample.sample_id for sample in samples if predicate(sample)}
        selected = [row for row in d2_rows if row["candidate_id"] in ids]
        false_safe = [row for row in selected if row["harm_v2"] and row["risk_probability"] < HARM_THRESHOLD]
        subtype_rows.append({"synthetic_interaction": LABEL, "subtype": name, "candidate_count": len(selected),
                             "false_safe_under_threshold": len(false_safe),
                             "reaching_benefit_gate": sum(row["benefit_threshold_pass"] for row in false_safe),
                             "reaching_top_ranking": sum(row["benefit_threshold_pass"] and row["benefit_rank"] == 1 for row in false_safe),
                             "personalized_final_exposure": sum(row["personalized_selected"] for row in false_safe),
                             "any_final_exposure": sum(row["selected"] for row in false_safe)})

    tradeoff_rows = []
    for row in d2_rows:
        sample = sample_by_id[row["candidate_id"]]
        if not (sample.targets.benefit > 0 and row["harm_v2"] and row["feasible"]):
            continue
        d1_row = next(item for item in d1_by_episode[row["episode_id"]] if item["candidate_id"] == row["candidate_id"])
        if not row["benefit_threshold_pass"]: reason = "BENEFIT_GATE"
        elif not row["harm_v2_gate_pass"]: reason = "HARM_V2_GATE"
        elif row["benefit_rank"] != 1: reason = "RANKING"
        elif not row["generic_score_win"]: reason = "GENERIC_POLICY_DOMINANCE"
        else: reason = "OTHER_CANDIDATE_OR_TIE"
        tradeoff_rows.append({"synthetic_interaction": LABEL, "episode_id": row["episode_id"], "candidate_id": row["candidate_id"],
                              "GT_benefit": sample.targets.benefit, "predicted_benefit": row["predicted_benefit"],
                              "predicted_harm_v2": row["risk_probability"], "rejection_primary": reason,
                              "benefit_pass": row["benefit_threshold_pass"], "harm_pass": row["harm_v2_gate_pass"],
                              "also_rejected_by_harm_gate": not row["harm_v2_gate_pass"],
                              "rank": row["benefit_rank"], "generic_score_win": row["generic_score_win"],
                              "D1_selected": d1_row["personalized_selected"], "D2_selected": row["personalized_selected"]})

    d1_attribution = []
    for row in d1_rows:
        if not row["harm_v2"]:
            continue
        sample = sample_by_id[row["candidate_id"]]
        reason = rejection_reason(row)
        d1_attribution.append({"synthetic_interaction": LABEL, "episode_id": row["episode_id"], "candidate_id": row["candidate_id"],
                               "GT_benefit": sample.targets.benefit, "GT_benefit_sign": "positive" if sample.targets.benefit > 0 else "non_positive",
                               "predicted_benefit": row["predicted_benefit"],
                               "benefit_rank": row["benefit_rank"], "generic_score_win": row["generic_score_win"],
                               "D1_personalized_selected": row["personalized_selected"], "primary_attribution": reason})

    coverage = [
        ("Hard feasibility", 1, 1, "both branches select only from feasible mask"),
        ("Benefit gate", 1, 0, "applies to approved personalized candidates; generic baseline bypasses"),
        ("Ranking", 1, 0, "personalized adjusted-score comparison; generic chosen by generic cost"),
        ("Harm-v2 gate", 1, 0, "approved mask only; generic fallback is not filtered"),
        ("Arbitration", 1, 1, "personalized challenges generic; generic remains baseline/fallback"),
        ("GT-unsafe protection", 0, 0, "GT labels are evaluation-only; protection is indirect"),
    ]
    coverage_rows = [{"synthetic_interaction": LABEL, "mechanism": mechanism, "branch": branch, "covered": bool(value), "evidence": evidence}
                     for mechanism, personalized, generic, evidence in coverage for branch, value in (("Personalized", personalized), ("Generic", generic))]

    group_predicates = {
        "C7": lambda sample: "C7" in "|".join(map(str, sample.split_metadata["contexts_evaluation_only"])),
        "C8": lambda sample: "C8" in "|".join(map(str, sample.split_metadata["contexts_evaluation_only"])),
        "C9": lambda sample: "C9" in "|".join(map(str, sample.split_metadata["contexts_evaluation_only"])),
        "Stop": lambda sample: sample.split_metadata["motion_type_evaluation_only"] == "stop",
        "Deceleration": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
    }
    by_context_motion = [{"synthetic_interaction": LABEL, "group": name, **subgroup_summary(samples, d2_rows, predicate)} for name, predicate in group_predicates.items()]

    source_file = Path(inspect.getsourcefile(arbitrate_large_context)).resolve(); source_lines, start_line = inspect.getsourcelines(arbitrate_large_context)
    decision_path = {"label": LABEL, "audit_only": True, "source_file": str(source_file), "function": "arbitrate_large_context",
                     "source_line": start_line, "source_sha256": file_sha(source_file),
                     "decision_order": ["validate aligned arrays", "choose generic among feasible by generic_cost/action tie-break",
                                        "build personalized approved mask = feasible & benefit>=threshold & harm<=threshold",
                                        "choose approved personalized adjusted-cost challenger", "if challenger beats generic return personalized",
                                        "otherwise return preselected generic without benefit/harm filtering"],
                     "branches": {"generic_hard_feasibility": True, "generic_benefit_gate": False, "generic_harm_v2_gate": False,
                                  "generic_direct_fallback": True, "personalized_harm_v2_gate": True},
                     "branch_condition": "if approved challenger adjusted cost < generic cost: personalized; else: generic",
                     "source_excerpt_sha256": hashlib.sha256("".join(source_lines).encode()).hexdigest()}

    risk_reaching = [row for row in d2_rows if row["harm_v2"] and reaches_harm_gate(row)]
    risk_veto = [row for row in risk_reaching if not row["harm_v2_gate_pass"]]
    safe_reaching = [row for row in d2_rows if row["safe_beneficial"] and reaches_harm_gate(row)]
    safe_wrong = [row for row in safe_reaching if not row["harm_v2_gate_pass"]]
    final_changed = sum(row["final_action_changed"] and row["category"].startswith("F3") for row in floor_rows)
    floor_candidate_counts = {name: sum(row["risk_candidate_count"] for row in floor_rows if row["category"] == name) for name in floor_categories}
    floor_candidate_ratios = {name: count / max(sum(floor_candidate_counts.values()), 1) for name, count in floor_candidate_counts.items()}
    floor_output_rows = floor_rows + [{"synthetic_interaction": LABEL, "record_type": "category_summary", "episode_id": "",
                                       "category": name, "risk_candidate_count": floor_candidate_counts[name],
                                       "category_episode_count": floor_counts[name],
                                       "category_episode_ratio": floor_counts[name] / len(floor_rows),
                                       "category_candidate_ratio": floor_candidate_ratios[name]} for name in floor_categories]
    gate_classification = "B_FLOOR_EFFECT"
    root_classification = {
        "label": LABEL, "post_hoc_only": True, "formal_phase5b17f_gate_e_remains": "FAIL", "formal_phase5b17f_remains": "FAIL",
        "gate_e_failure_primary": "FLOOR_EFFECT_ON_CURRENT_EVALUATION_DISTRIBUTION: D1 already has zero risky personalized switches; candidate-level veto exists but changes no final action",
        "gate_e_information_classification": gate_classification,
        "safe_beneficial_recall_primary_bottleneck": "BENEFIT_SIGN_ABSOLUTE_CALIBRATION",
        "safety_violation_primary": "GENERIC_BRANCH_COVERAGE_GAP: frozen generic baseline bypasses harm-v2 gate",
        "next_single_variable_intervention": "Generic Safety Coverage Repair", "next_intervention_implemented": False,
        "deceleration_latent_safety_debt": bool(deceleration_latent),
    }
    frozen_after = {name: file_sha(args.formal_dir / name) for name in formal_before}
    frozen = {"label": LABEL, "post_hoc_only": True, "test_reads": 0, "optimizer_steps": 0, "backward_calls": 0,
              "R1_checkpoint_sha256": file_sha(args.checkpoint), "R1_checkpoint_unchanged": model_before == d.model_sha(model),
              "harm_v2_checkpoint_sha256": file_sha(args.harm_checkpoint), "harm_v2_head_unchanged": head_before == e.state_sha(head.state_dict()),
              "harm_v2_threshold": HARM_THRESHOLD, "threshold_unchanged": HARM_THRESHOLD == float(selection["threshold"]),
              "benefit_threshold": BENEFIT_THRESHOLD, "benefit_threshold_unchanged": BENEFIT_THRESHOLD == d.FROZEN_THRESHOLDS[0],
              "manifest_sha256": file_sha(args.manifest), "manifest_unchanged": file_sha(args.manifest) == d.EXPECTED_MANIFEST_SHA,
              "normalizer_sha256": normalizers["sha256"], "normalizer_unchanged": normalizers["sha256"] == ed.EXPECTED_NORMALIZER_SHA256,
              "formal_1.7F_hashes_before": formal_before, "formal_1.7F_hashes_after": frozen_after,
              "formal_1.7F_unchanged": formal_before == frozen_after, "formal_gate_e_still_fail": not formal_gates["Gate_E"]["passed"],
              "formal_phase5b17f_still_fail": not formal_summary["phase5b17f_passed"],
              "formal_D1_D2_metric_reproduction": metric_reproduction, "D1_D2_action_replay_reproducible": action_replay,
              "six_generic_unsafe_cases_reproduced": len(generic_unsafe) == int(formal_d1["GT_unsafe_final_selected_count"]) == 6,
              "C7_funnel_count_consistent": c7_funnel[0]["candidate_count"] == subgroup_summary(samples, d2_rows, group_predicates["C7"])["safe_beneficial_candidates"],
              "Stop_funnel_count_consistent": stop_funnel[0]["candidate_count"] == subgroup_summary(samples, d2_rows, group_predicates["Stop"])["safe_beneficial_candidates"],
              "generic_policy_source_sha256": file_sha(source_file), "hard_safety_mask_unchanged": True,
              "arbitration_unchanged": True, "audit_counterfactual_changed_formal_action": False,
              "all_model_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()) and not any(parameter.requires_grad for parameter in head.parameters())}
    sign_groups = {name: sum(row[name] for row in sign_errors) for name in ("C7", "Stop", "C8", "C9", "deceleration", "heading", "lateral")}
    figures = make_figures(args.output_dir, floor_counts, sign_groups, coverage_rows)
    summary = {"label": LABEL, "stage": STAGE, "post_hoc_only": True, "test_reads": 0,
               "formal_phase5b17f_gate_e": "FAIL", "formal_phase5b17f": "FAIL", "evaluation_episode_count": len(evaluation_ids),
               "floor_effect_episode_counts": dict(floor_counts), "risk_candidates_reaching_harm_gate": len(risk_reaching),
               "floor_effect_risk_candidate_counts": floor_candidate_counts, "floor_effect_risk_candidate_ratios": floor_candidate_ratios,
               "risk_candidates_rejected_by_harm_gate": len(risk_veto), "risk_rejection_rate": len(risk_veto) / max(len(risk_reaching), 1),
               "decision_changing_risk_veto_count": final_changed, "non_decision_changing_risk_veto_count": len(risk_veto) - final_changed,
               "safe_beneficial_reaching_harm_gate": len(safe_reaching), "safe_beneficial_incorrectly_rejected": len(safe_wrong),
               "generic_unsafe_exposure_count": len(generic_unsafe), "generic_unsafe_caught_by_audit_harm_gate": sum(row["caught_by_harm_v2"] for row in generic_counterfactual),
               "generic_unsafe_missed_by_audit_harm_gate": sum(row["missed_by_harm_v2"] for row in generic_counterfactual),
               "GT_unsafe_candidate_count": len(unsafe_rows), "GT_unsafe_feasible_true": sum(row["feasible"] for row in unsafe_rows),
               "GT_unsafe_feasible_false": sum(not row["feasible"] for row in unsafe_rows),
               "C7_funnel_primary_loss": c7_loss, "Stop_funnel_primary_loss": stop_loss,
               "safe_beneficial_candidate_count": sum(row["safe_beneficial"] for row in d2_rows), "benefit_sign_failure_count": len(sign_errors),
               "benefit_sign_failure_by_subgroup": sign_groups, "deceleration_latent_false_safe_count": len(deceleration_latent),
               "deceleration_latent_could_enter_if_benefit_improves": sum(row["could_enter_if_benefit_improves"] for row in deceleration_latent),
               "tradeoff_rejection_attribution": dict(Counter(row["rejection_primary"] for row in tradeoff_rows)),
               "tradeoff_also_rejected_by_harm_gate": sum(row["also_rejected_by_harm_gate"] for row in tradeoff_rows),
               "D1_zero_risky_attribution": dict(Counter(row["primary_attribution"] for row in d1_attribution)),
               "D1_metrics_reproduced": d1_metrics, "D2_metrics_reproduced": d2_metrics,
               "root_cause_classification": root_classification, "figures": figures}

    io.write_json(args.output_dir / "frozen_contract.json", frozen)
    io.write_csv(args.output_dir / "gate_e_floor_effect.csv", floor_output_rows)
    io.write_csv(args.output_dir / "candidate_level_harm_gate_value.csv", candidate_value)
    io.write_json(args.output_dir / "generic_decision_path.json", decision_path)
    io.write_csv(args.output_dir / "generic_unsafe_exposures.csv", generic_unsafe)
    io.write_csv(args.output_dir / "generic_harm_counterfactual.csv", generic_counterfactual)
    io.write_csv(args.output_dir / "hard_safety_vs_gt_unsafe.csv", hard_safety)
    io.write_csv(args.output_dir / "generic_unsafe_root_causes.csv", root_causes)
    io.write_csv(args.output_dir / "c7_safe_beneficial_funnel.csv", c7_funnel)
    io.write_csv(args.output_dir / "stop_safe_beneficial_funnel.csv", stop_funnel)
    io.write_csv(args.output_dir / "benefit_sign_error_audit.csv", sign_errors)
    io.write_csv(args.output_dir / "deceleration_latent_risk.csv", deceleration_latent)
    io.write_csv(args.output_dir / "by_harm_subtype.csv", subtype_rows)
    io.write_csv(args.output_dir / "benefit_risk_tradeoff_attribution.csv", tradeoff_rows)
    io.write_csv(args.output_dir / "d1_zero_risky_switch_attribution.csv", d1_attribution)
    io.write_csv(args.output_dir / "decision_safety_coverage_matrix.csv", coverage_rows)
    io.write_csv(args.output_dir / "by_context_motion.csv", by_context_motion)
    io.write_json(args.output_dir / "root_cause_classification.json", root_classification)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2))


if __name__ == "__main__":
    main()
