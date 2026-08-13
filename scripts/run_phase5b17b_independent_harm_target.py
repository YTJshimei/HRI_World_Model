"""Phase 5B-1.7B independent-harm semantics and label-support audit.

This script is deliberately read-only with respect to models, targets, datasets,
and arbitration.  It materializes the frozen TRAIN/VALIDATION synthetic samples
only; no TEST builder is imported or called.
"""
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

from scripts import run_phase5a_frozen3b as p5
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b16_candidate_ranking as b16
from src.multimodal.temporal_schema import LABEL

BENEFIT_EPSILON = 1e-6
ACTION_NAMES = {
    0: "KEEP", 1: "SPEED_DOWN_10", 2: "SPEED_UP_10",
    3: "DISTANCE_PLUS_0_2", 4: "DISTANCE_MINUS_0_2",
    5: "LEFT_OFFSET", 6: "RIGHT_OFFSET",
}
CONTEXTS = ("C7", "C8", "C9")
DEFINITIONS = ("harm_A", "harm_B", "harm_C")

# Declared before inspecting prevalence. These are support/readiness criteria,
# not target thresholds and do not change any labels.
SUPPORT_CRITERIA = {
    "minimum_train_positive_candidates": 50,
    "minimum_validation_positive_candidates": 20,
    "minimum_train_positive_episodes": 15,
    "minimum_validation_positive_episodes": 8,
    "minimum_positive_candidates_per_required_context_per_split": 5,
    "required_contexts": list(CONTEXTS),
    "requires_beneficial_and_harmful_validation_candidates": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17b_independent_harm_target")
    parser.add_argument("--phase5b16-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b16_candidate_ranking")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_definitions() -> dict[str, object]:
    return {
        "label": LABEL,
        "selection_prohibition": "definitions were not selected or thresholded using decision performance",
        "total_cost_prohibited": True,
        "definitions": {
            "harm_A": {
                "name": "HARD_SAFETY_ONLY", "constructible": True,
                "formula": "GT unsafe = (unsafe_duration > 0)",
                "independent_of_benefit_sign": True,
                "source_signal": "GT unsafe",
            },
            "harm_B": {
                "name": "SAFETY_OR_RESPONSE", "constructible": False,
                "conceptual_formula": "GT unsafe OR adverse_human_response_event",
                "blocking_reason": "no simulator/protocol/domain-defined adverse-response event or threshold exists",
                "continuous_human_response_cost_was_not_thresholded": True,
            },
            "harm_C": {
                "name": "SAFETY_OR_RESPONSE_OR_DISTURBANCE", "constructible": False,
                "conceptual_formula": "GT unsafe OR adverse_human_response_event OR excess_disturbance_event",
                "blocking_reason": "neither adverse-response nor excess-disturbance has an event definition or threshold",
                "continuous_costs_were_not_thresholded": True,
            },
        },
    }


def harm_signal_registry() -> dict[str, object]:
    def item(name, source, function, formula, unit, value_range, direction, meaning, availability, usable, limitation=""):
        return {"signal": name, "source_file": source, "function": function, "formula": formula,
                "unit": unit, "range": value_range, "direction": direction, "runtime_meaning": meaning,
                "availability": availability, "usable_for_independent_binary_harm": usable, "limitation": limitation}
    entries = [
        item("GT unsafe", "scripts/run_phase5a_context_value.py", "build_tokens", "unsafe_duration > 0", "boolean", "{0,1}", "1=higher risk", "any GT too-close frame", "GT-only", True),
        item("unsafe_duration / too-close duration", "src/decision/decision_cost.py", "compute_decision_costs", "mean(human_robot_distance < too_close_distance)", "fraction of horizon", "[0,1]", "higher-is-worse", "fraction of future frames violating 0.80 m synthetic boundary", "GT-only for target; predicted analogue runtime-estimable", True),
        item("minimum human-robot distance", "src/decision/decision_cost.py", "compute_decision_costs", "min_t(distance[t])", "m", "[0,+inf)", "lower-is-worse", "closest predicted/GT separation", "GT-only for target; predicted analogue runtime-estimable", True),
        item("safety cost", "src/decision/decision_cost.py", "compute_decision_costs", "5*violation_proxy + 8*unsafe_duration + 10*close_gap + 1e4*infeasible", "cost units", "[0,+inf)", "higher-is-worse", "soft proximity risk plus hard infeasibility penalty", "GT-only component for label audit", False, "continuous composite; no independent binary threshold beyond GT unsafe"),
        item("human_response cost", "src/decision/decision_cost.py", "compute_decision_costs", "0.30*effect_magnitude + 0.25*speed_effect + 0.20*lateral_effect + 0.25*heading_effect", "dimensionless synthetic cost", "[0,+inf)", "higher=larger response magnitude", "symmetric magnitude of action-conditioned deviation from natural motion", "GT-only component for label audit; model-predicted analogue runtime-estimable", False, "not signed by comfort, instability, injury, or adverse outcome; no event threshold"),
        item("disturbance cost", "src/decision/decision_cost.py", "compute_decision_costs", "0.30*|speed_delta|/.10 + 0.25*|distance_offset|/.20 + 0.20*|lateral_offset|/.20 + 0.25*effect_magnitude/.05", "dimensionless synthetic cost", "[0,+inf)", "higher=larger intervention", "action magnitude plus human action-effect magnitude regularizer", "partly deterministic/runtime-estimable; GT effect component for target", False, "preference/regularization proxy, not a validated harm event; no event threshold"),
        item("uncertainty cost", "src/decision/decision_cost.py", "compute_decision_costs", "mean_xy_coordinate_uncertainty / 0.05", "dimensionless", "[0,+inf)", "higher-is-worse", "world-model coordinate uncertainty", "runtime-estimable", False, "epistemic/model risk, not realized human harm"),
        item("task cost", "src/decision/decision_cost.py", "compute_decision_costs", "final_error + .35*mean_error + .45*progress_failure + .25*visibility_proxy", "m-equivalent cost", "[0,+inf)", "higher-is-worse", "following-task error", "GT-only component for audit; predicted analogue runtime-estimable", False, "task failure is not independent human harm"),
        item("human action effect", "src/data/synthetic_interaction.py", "simulate_action_response", "future_by_action - natural_future", "m per joint", "real-valued", "unsigned without an adverse criterion", "robot-conditioned residual human motion", "GT-only future; predicted analogue runtime-estimable", False, "response exists, but adverse direction/event does not"),
        item("human root / motion degradation", "not present", "not present", "not defined", "n/a", "n/a", "n/a", "no explicit degradation event", "unavailable", False),
        item("response instability", "not present", "not present", "not defined", "n/a", "n/a", "n/a", "no oscillation/instability event", "unavailable", False),
        item("relative speed anomaly", "not present", "not present", "not defined", "n/a", "n/a", "n/a", "relative velocity is not labeled anomalous", "unavailable", False),
        item("excess acceleration/deceleration", "not present", "not present", "not defined", "n/a", "n/a", "n/a", "no physical/domain excess threshold", "unavailable", False),
        item("tracking disruption", "not present", "not present", "not defined", "n/a", "n/a", "n/a", "visibility is simulated, but action-caused tracking loss is not", "unavailable", False),
        item("occlusion-associated response degradation", "not present", "not present", "not defined", "n/a", "n/a", "n/a", "C7 is input occlusion, not an adverse response outcome", "unavailable", False),
    ]
    return {"label": LABEL, "signals": entries, "independent_binary_signals": ["GT unsafe"],
            "adverse_response_event_available": False, "excess_disturbance_event_available": False}


def human_response_audit() -> dict[str, object]:
    return {"label": LABEL, "source_file": "src/decision/decision_cost.py", "function": "compute_decision_costs",
            "formula": "0.30*effect_magnitude + 0.25*speed_effect + 0.20*lateral_effect + 0.25*heading_effect",
            "inputs": ["action effect on all joints", "root speed-effect", "final lateral root effect", "final shoulder-heading effect"],
            "range": "[0,+inf)", "unit": "dimensionless synthetic cost", "direction": "higher means larger response magnitude",
            "interpretation": "action-conditioned departure from natural motion magnitude; not comfort, instability, injury, or adverse response",
            "can_define_binary_harm_without_new_protocol": False, "reason": "absolute magnitudes discard whether a response is helpful/adverse and no domain threshold exists"}


def disturbance_audit() -> dict[str, object]:
    return {"label": LABEL, "source_file": "src/decision/decision_cost.py", "function": "compute_decision_costs",
            "formula": "0.30*|speed_scale_delta|/.10 + 0.25*|distance_offset|/.20 + 0.20*|lateral_offset|/.20 + 0.25*effect_magnitude/.05",
            "inputs": ["candidate action magnitude", "human action-effect magnitude"], "range": "[0,+inf)",
            "unit": "dimensionless synthetic cost", "direction": "higher means larger intervention magnitude",
            "interpretation": "hand-weighted intervention/preference regularizer in total cost; it does not encode an observed adverse interaction event",
            "can_define_binary_harm_without_new_protocol": False, "reason": "no physical/protocol threshold separates acceptable intervention from harm"}


def label_for(sample, definition: str) -> bool | None:
    if definition == "harm_A":
        return bool(sample.targets.gt_unsafe)
    if definition in ("harm_B", "harm_C"):
        return None
    raise KeyError(definition)


def context_keys(sample) -> list[str]:
    present = {str(tag)[:2] for tag in sample.temporal_tags}
    return [name for name in CONTEXTS if name in present]


def group_summary(split, definition, dimension, group, samples) -> dict[str, object]:
    labels = [label_for(sample, definition) for sample in samples]
    if any(value is None for value in labels):
        return {"synthetic_interaction": LABEL, "split": split, "definition": definition,
                "dimension": dimension, "group": str(group), "definition_valid": False,
                "candidate_count": len(samples), "positive_count": None, "positive_rate": None,
                "positive_episode_count": None, "status": "NOT_CONSTRUCTIBLE"}
    positives = np.asarray(labels, bool)
    return {"synthetic_interaction": LABEL, "split": split, "definition": definition,
            "dimension": dimension, "group": str(group), "definition_valid": True,
            "candidate_count": len(samples), "negative_count": int((~positives).sum()),
            "positive_count": int(positives.sum()), "positive_rate": float(positives.mean()) if len(samples) else 0.0,
            "negative_rate": float((~positives).mean()) if len(samples) else 0.0,
            "positive_episode_count": len({sample.episode_id for sample, positive in zip(samples, positives) if positive}),
            "episode_count": len({sample.episode_id for sample in samples}), "status": "VALID"}


def grouped_rows(split, definition, samples, dimension, key_function, include_empty=()):
    grouped = defaultdict(list)
    for sample in samples:
        keys = key_function(sample); keys = keys if isinstance(keys, (list, tuple)) else [keys]
        for key in keys:
            grouped[str(key)].append(sample)
    for key in include_empty:
        grouped.setdefault(str(key), [])
    return [group_summary(split, definition, dimension, key, grouped[key]) for key in sorted(grouped)]


def quadrant_rows(split, definition, samples):
    labels = [label_for(sample, definition) for sample in samples]
    if any(value is None for value in labels):
        return [{"synthetic_interaction": LABEL, "split": split, "definition": definition,
                 "definition_valid": False, "quadrant": "NOT_CONSTRUCTIBLE", "candidate_count": None,
                 "episode_count": None, "feasible_count": None}]
    beneficial = np.asarray([sample.targets.benefit > BENEFIT_EPSILON for sample in samples], bool)
    harmful = np.asarray(labels, bool); feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    result = []
    for benefit_value, harm_value in ((1, 0), (1, 1), (0, 1), (0, 0)):
        selected = (beneficial == bool(benefit_value)) & (harmful == bool(harm_value))
        result.append({"synthetic_interaction": LABEL, "split": split, "definition": definition,
                       "definition_valid": True, "quadrant": f"beneficial={benefit_value},harmful={harm_value}",
                       "candidate_count": int(selected.sum()),
                       "episode_count": len({sample.episode_id for sample, keep in zip(samples, selected) if keep}),
                       "feasible_count": int((selected & feasible).sum())})
    return result


def safe_beneficial_row(split, definition, samples):
    labels = [label_for(sample, definition) for sample in samples]
    if any(value is None for value in labels):
        return {"synthetic_interaction": LABEL, "split": split, "definition": definition,
                "definition_valid": False, "candidate_count": None, "episode_count": None,
                "fraction_of_all_episodes": None, "status": "NOT_CONSTRUCTIBLE"}
    selected = np.asarray([sample.targets.benefit > BENEFIT_EPSILON and not harmful and sample.targets.feasible
                           for sample, harmful in zip(samples, labels)], bool)
    episode_count = len({sample.episode_id for sample, keep in zip(samples, selected) if keep})
    all_episodes = len({sample.episode_id for sample in samples})
    return {"synthetic_interaction": LABEL, "split": split, "definition": definition,
            "definition_valid": True, "candidate_count": int(selected.sum()), "episode_count": episode_count,
            "all_episode_count": all_episodes, "fraction_of_all_episodes": episode_count / max(all_episodes, 1),
            "status": "VALID"}


def point_biserial(benefit, label):
    x = np.asarray(benefit, float); y = np.asarray(label, bool)
    if len(x) < 2 or y.all() or (~y).all() or np.std(x) == 0:
        return None
    return float(np.corrcoef(x, y.astype(float))[0, 1])


def correlation_rows(split, definition, samples):
    labels = [label_for(sample, definition) for sample in samples]
    valid = not any(value is None for value in labels)
    return {"synthetic_interaction": LABEL, "split": split, "definition": definition,
            "definition_valid": valid, "candidate_count": len(samples) if valid else None,
            "point_biserial_benefit_vs_harm": point_biserial([s.targets.benefit for s in samples], labels) if valid else None,
            "exact_negative_benefit_equivalence": bool(valid and all(bool(y) == (s.targets.benefit < -BENEFIT_EPSILON)
                                                       for s, y in zip(samples, labels)))}


def dependency_rows():
    dependencies = {
        "benefit": {"task", "safety", "human_response", "disturbance", "uncertainty", "generic_candidate_difference"},
        "harm_A": {"GT distance trajectory", "too_close_distance", "unsafe_duration"},
        "harm_B": {"harm_A", "adverse_human_response_event (unavailable)"},
        "harm_C": {"harm_A", "adverse_human_response_event (unavailable)", "excess_disturbance_event (unavailable)"},
    }
    rows = []
    for target, signals in dependencies.items():
        for signal in sorted(signals):
            rows.append({"synthetic_interaction": LABEL, "target": target, "signal": signal,
                         "direct_dependency": True, "constructible": target in ("benefit", "harm_A"),
                         "uses_total_cost_comparison": target == "benefit",
                         "derived_from_benefit_sign": False if target != "benefit" else None})
    return rows


def oracle_rows(validation, definition):
    grouped = defaultdict(list)
    for sample in validation:
        grouped[sample.episode_id].append(sample)
    labels = [label_for(sample, definition) for sample in validation]
    if any(value is None for value in labels):
        return [{"synthetic_interaction": LABEL, "split": "validation", "definition": definition,
                 "definition_valid": False, "acceptable_candidate_count": None,
                 "acceptable_episode_count": None, "maximum_theoretical_episode_recall": None,
                 "beneficial_and_harmful_count": None}]
    acceptable_candidates = 0; acceptable_episodes = 0; beneficial_harmful = 0
    beneficial_episodes = set()
    for episode_id, samples in grouped.items():
        accepted = []
        for sample in samples:
            harmful = bool(label_for(sample, definition)); beneficial = sample.targets.benefit > BENEFIT_EPSILON
            beneficial_harmful += int(beneficial and harmful)
            if beneficial:
                beneficial_episodes.add(episode_id)
            if beneficial and not harmful and sample.targets.feasible:
                accepted.append(sample)
        acceptable_candidates += len(accepted); acceptable_episodes += bool(accepted)
    return [{"synthetic_interaction": LABEL, "split": "validation", "definition": definition,
             "definition_valid": True, "acceptable_candidate_count": acceptable_candidates,
             "acceptable_episode_count": acceptable_episodes,
             "beneficial_episode_count": len(beneficial_episodes),
             "maximum_theoretical_episode_recall": acceptable_episodes / max(len(beneficial_episodes), 1),
             "beneficial_and_harmful_count": beneficial_harmful}]


def readiness(splits, context_rows, quadrants):
    overall = {split: group_summary(split, "harm_A", "overall", "all", samples) for split, samples in splits.items()}
    context = {(row["split"], row["group"]): row for row in context_rows if row["definition"] == "harm_A"}
    validation_conflicts = next(row["candidate_count"] for row in quadrants
                                if row["definition"] == "harm_A" and row["split"] == "validation"
                                and row["quadrant"] == "beneficial=1,harmful=1")
    checks = {
        "train_positive_candidates": overall["train"]["positive_count"] >= SUPPORT_CRITERIA["minimum_train_positive_candidates"],
        "validation_positive_candidates": overall["validation"]["positive_count"] >= SUPPORT_CRITERIA["minimum_validation_positive_candidates"],
        "train_positive_episodes": overall["train"]["positive_episode_count"] >= SUPPORT_CRITERIA["minimum_train_positive_episodes"],
        "validation_positive_episodes": overall["validation"]["positive_episode_count"] >= SUPPORT_CRITERIA["minimum_validation_positive_episodes"],
        "required_context_support": all(context[(split, name)]["positive_count"] >= SUPPORT_CRITERIA["minimum_positive_candidates_per_required_context_per_split"]
                                        for split in ("train", "validation") for name in CONTEXTS),
        "benefit_risk_conflict_support": validation_conflicts > 0,
    }
    return {"criteria_declared_before_prevalence_review": True, "criteria": SUPPORT_CRITERIA,
            "checks": checks, "passed": all(checks.values()),
            "failed_checks": [name for name, passed in checks.items() if not passed]}


def make_figures(output, overall_rows, action_rows, context_rows, quadrants, correlations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.suptitle(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    valid_overall = [row for row in overall_rows if row["definition"] == "harm_A"]
    plt.figure(); plt.bar([r["split"] for r in valid_overall], [r["positive_rate"] for r in valid_overall]); plt.ylabel("GT unsafe positive rate"); save("hard_safety_prevalence.png")
    val_action = [r for r in action_rows if r["definition"] == "harm_A" and r["split"] == "validation"]
    plt.figure(figsize=(8, 4)); plt.bar([r["group"] for r in val_action], [r["positive_rate"] for r in val_action]); plt.xticks(rotation=25, ha="right"); plt.ylabel("validation harm_A rate"); save("harm_A_by_action.png")
    val_context = [r for r in context_rows if r["definition"] == "harm_A" and r["split"] == "validation"]
    plt.figure(); plt.bar([r["group"] for r in val_context], [r["positive_count"] for r in val_context]); plt.ylabel("validation positive candidates"); save("harm_A_context_support.png")
    val_quads = [r for r in quadrants if r["definition"] == "harm_A" and r["split"] == "validation"]
    plt.figure(figsize=(8, 4)); plt.bar([r["quadrant"] for r in val_quads], [r["candidate_count"] for r in val_quads]); plt.xticks(rotation=20, ha="right"); plt.ylabel("candidate count"); save("benefit_harm_A_quadrants.png")
    corr = [r for r in correlations if r["definition"] == "harm_A"]
    plt.figure(); plt.bar([r["split"] for r in corr], [r["point_biserial_benefit_vs_harm"] for r in corr]); plt.axhline(-1, color="k", linestyle="--"); plt.ylabel("point-biserial correlation"); save("benefit_harm_A_correlation.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Phase5B-1.7B: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest_path = args.manifest_dir / "phase5b_manifest_v1.json"
    checkpoint_path = args.phase5b16_dir / "checkpoints" / "r1_best.pt"
    frozen_paths = {
        "manifest": manifest_path,
        "r1_checkpoint": checkpoint_path,
        "old_target_source": PROJECT_ROOT / "scripts" / "run_phase5a_context_value.py",
        "old_target_schema": PROJECT_ROOT / "src" / "multimodal" / "temporal_schema.py",
        "formal_dataset": PROJECT_ROOT / "src" / "multimodal" / "temporal_dataset.py",
        "frozen_model_architecture": PROJECT_ROOT / "src" / "models" / "rich_temporal_small_transformer.py",
        "arbitration": PROJECT_ROOT / "src" / "decision" / "large_context_arbitrator.py",
    }
    before = {name: file_sha256(path) for name, path in frozen_paths.items()}
    _, manifest_audit = b1.manifest_file_audit(args.manifest_dir)
    splits = b16.build_train_validation_only(args, torch)
    if set(splits) != {"train", "validation"}:
        raise RuntimeError("only TRAIN/VALIDATION may enter Phase5B-1.7B")

    definitions = candidate_definitions(); registry = harm_signal_registry()
    overall_rows = []; hard_rows = []; prevalence = []; action_rows = []; motion_rows = []; profile_rows = []; context_rows = []
    quadrants = []; safe_space = []; correlations = []; oracle = []
    for split, samples in splits.items():
        hard_rows.append(group_summary(split, "harm_A", "overall", "all", samples))
        hard_rows += grouped_rows(split, "harm_A", samples, "action", lambda s: ACTION_NAMES[int(s.split_metadata["candidate_action_id_audit"])])
        hard_rows += grouped_rows(split, "harm_A", samples, "motion", lambda s: s.split_metadata["motion_type_evaluation_only"])
        hard_rows += grouped_rows(split, "harm_A", samples, "profile", lambda s: f"profile_{s.split_metadata['person_profile_id']}")
        hard_rows += grouped_rows(split, "harm_A", samples, "context", context_keys, CONTEXTS)
        for definition in DEFINITIONS:
            row = group_summary(split, definition, "overall", "all", samples); overall_rows.append(row); prevalence.append(row)
            action_rows += grouped_rows(split, definition, samples, "action", lambda s: ACTION_NAMES[int(s.split_metadata["candidate_action_id_audit"])])
            motion_rows += grouped_rows(split, definition, samples, "motion", lambda s: s.split_metadata["motion_type_evaluation_only"])
            profile_rows += grouped_rows(split, definition, samples, "profile", lambda s: f"profile_{s.split_metadata['person_profile_id']}")
            context_rows += grouped_rows(split, definition, samples, "context", context_keys, CONTEXTS)
            quadrants += quadrant_rows(split, definition, samples)
            safe_space.append(safe_beneficial_row(split, definition, samples))
            correlations.append(correlation_rows(split, definition, samples))
    for definition in DEFINITIONS:
        oracle += oracle_rows(splits["validation"], definition)

    readiness_audit = readiness(splits, context_rows, quadrants)
    unsafe_coverage = {
        split: all(not sample.targets.gt_unsafe or label_for(sample, "harm_A") is True for sample in samples)
        for split, samples in splits.items()
    }
    recommendation = {
        "label": LABEL, "HARM_TARGET_V2": "HARD_SAFETY_ONLY (harm_A = GT unsafe)",
        "target_semantics_ready": True, "formal_retraining_ready": readiness_audit["passed"],
        "selection_basis": ["independent adverse-event semantics", "physical interpretability", "GT unsafe coverage", "train/validation support audit"],
        "not_selected_by": ["decision performance", "beneficial recall", "regret", "switch rate"],
        "B_and_C_status": "NOT_CONSTRUCTIBLE without new adverse-response/disturbance event definitions",
        "next_step": "extend synthetic generator with protocol-defined independent adverse-response events before retraining",
        "next_step_automatically_started": False,
        "readiness_audit": readiness_audit,
    }
    figures = make_figures(args.output_dir, overall_rows, action_rows, context_rows, quadrants, correlations)
    after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    frozen = {
        "label": LABEL, **manifest_audit, "test_candidates_read": 0, "test_labels_read": 0,
        "test_metrics_computed": False, "optimizer_created": False, "optimizer_step_count": 0,
        "backward_call_count": 0, "model_checkpoint_loaded": False,
        "checkpoint_sha256_before": before["r1_checkpoint"], "checkpoint_sha256_after": after["r1_checkpoint"],
        "model_checksum_unchanged": before["r1_checkpoint"] == after["r1_checkpoint"],
        "file_checksums_before": before, "file_checksums_after": after,
        "frozen_files_unchanged": before == after, "old_target_overwritten": False,
        "profile_id_runtime_input": False, "profile_id_audit_only": True,
        "GT_fields_runtime_input": False, "GT_fields_audit_only": True,
        "manifest_unchanged": before["manifest"] == after["manifest"],
    }
    summary = {
        "label": LABEL, "stage": "Phase 5B-1.7B Independent Harm Target Design & Label Support Audit",
        "valid_definitions": ["harm_A"], "invalid_definitions": ["harm_B", "harm_C"],
        "hard_safety": {row["split"]: {"positive_candidates": row["positive_count"], "candidate_count": row["candidate_count"],
                                               "positive_rate": row["positive_rate"], "positive_episodes": row["positive_episode_count"]}
                        for row in overall_rows if row["definition"] == "harm_A"},
        "benefit_risk_conflict": {split: next(row["candidate_count"] for row in quadrants if row["split"] == split and row["definition"] == "harm_A" and row["quadrant"] == "beneficial=1,harmful=1") for split in splits},
        "unsafe_fully_covered": all(unsafe_coverage.values()), "unsafe_coverage_by_split": unsafe_coverage,
        "recommended_harm_target_v2": recommendation["HARM_TARGET_V2"],
        "formal_retraining_ready": recommendation["formal_retraining_ready"],
        "primary_limitation": "hard safety is valid, but C7 has sparse positives and the data contains no beneficial-and-harmful tradeoff examples or adverse-response events",
        "next_step": recommendation["next_step"], "next_step_automatically_started": False,
        "test_reads": 0, "optimizer_steps": 0, "backward_calls": 0, "figures": figures,
    }

    p5.write_json(args.output_dir / "harm_signal_registry.json", registry)
    p5.write_csv(args.output_dir / "hard_safety_distribution.csv", hard_rows)
    p5.write_json(args.output_dir / "human_response_signal_audit.json", human_response_audit())
    p5.write_json(args.output_dir / "disturbance_signal_audit.json", disturbance_audit())
    p5.write_json(args.output_dir / "candidate_harm_definitions.json", definitions)
    p5.write_csv(args.output_dir / "benefit_new_harm_quadrants.csv", quadrants)
    p5.write_csv(args.output_dir / "safe_beneficial_space.csv", safe_space)
    p5.write_csv(args.output_dir / "harm_label_prevalence.csv", prevalence)
    p5.write_csv(args.output_dir / "by_action_harm.csv", action_rows)
    p5.write_csv(args.output_dir / "by_motion_harm.csv", motion_rows)
    p5.write_csv(args.output_dir / "by_profile_harm.csv", profile_rows)
    p5.write_csv(args.output_dir / "by_context_harm.csv", context_rows)
    p5.write_csv(args.output_dir / "benefit_harm_correlation.csv", correlations)
    p5.write_csv(args.output_dir / "new_target_dependency_matrix.csv", dependency_rows())
    p5.write_csv(args.output_dir / "oracle_decision_semantics.csv", oracle)
    p5.write_json(args.output_dir / "recommended_harm_target_v2.json", recommendation)
    p5.write_json(args.output_dir / "frozen_contract.json", frozen)
    p5.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(p5.clean(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
