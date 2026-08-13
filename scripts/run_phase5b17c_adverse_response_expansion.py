"""Phase 5B-1.7C synthetic adverse-response expansion and readiness audit."""
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
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from src.data.adverse_response_dataset import (
    ACTION_IDS, DEVELOPMENT_PROFILE_IDS, GENERATOR_SEED, GENERATOR_VERSION,
    HELD_OUT_PROFILE_IDS, RISK_SEED, build_development_split, candidate_rows,
    episode_manifest_row, sealed_test_manifest_rows,
)
from src.data.adverse_response_protocol import LABEL, protocol_definition
from src.data.robot_action_schema import RobotAction

BENEFIT_EPSILON = 1e-6
TRAIN_SIZE = 240
VALIDATION_SIZE = 240
TEST_ID_COUNT = 120
ACTION_NAMES = {int(action): action.name for action in RobotAction}
CONTEXTS = ("C7", "C8", "C9")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion")
    parser.add_argument("--manifest-v1", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage" / "phase5b_manifest_v1.json")
    parser.add_argument("--phase5b17b-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17b_independent_harm_target")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(io.clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def risk_factor_registry():
    return {"label": LABEL, "generator_version": GENERATOR_VERSION, "label_only_risk_flags": False,
            "profile_id_directly_maps_to_label": False,
            "factors": [
                {"name": "braking_susceptibility", "support": "Uniform[0.15,1.35]", "unit": "dimensionless gain", "dynamics": "scales candidate-induced negative forward velocity pulse"},
                {"name": "lateral_startle_gain", "support": "Uniform[0.10,1.25]", "unit": "dimensionless gain", "dynamics": "scales candidate-induced lateral velocity pulse"},
                {"name": "heading_startle_gain", "support": "Uniform[0.10,1.20]", "unit": "dimensionless gain", "dynamics": "scales temporary local-pose/root heading rotation"},
                {"name": "approach_sensitivity", "support": "Uniform[0.35,1.30]", "unit": "dimensionless gain", "dynamics": "couples current separation and action approach pressure"},
                {"name": "onset_delay_seconds", "support": "Uniform[0.10,0.40]", "unit": "s", "dynamics": "delays onset of all added response dynamics"},
                {"name": "recovery_rate", "support": "Uniform[2.0,4.5]", "unit": "1/s", "dynamics": "controls decay after the startle pulse"},
            ], "sampling": "independent per initial-state episode; reused across that episode's counterfactual branches",
            "causal_path": "risk factors + action semantics + interaction state -> future trajectory -> physical event extraction -> harm_v2"}


def harm_definition():
    return {"label": LABEL, "name": "HARM_TARGET_V2", "formula": "GT unsafe OR GT adverse_human_kinematic_response",
            "GT_unsafe": "unsafe_duration > 0; frozen 0.80 m protocol unchanged",
            "adverse_human_kinematic_response": "EXCESSIVE_DECELERATION OR ABRUPT_LATERAL_RESPONSE OR ABRUPT_HEADING_CHANGE",
            "interaction_disruption_included": False, "disturbance_cost_included": False,
            "depends_on_benefit": False, "depends_on_total_cost_comparison": False,
            "depends_on_best_action": False, "depends_on_profile_id": False,
            "all_GT_unsafe_must_be_positive": True}


def summary_row(split, dimension, group, rows):
    harmful = [row for row in rows if row["harm_v2"]]; beneficial = [row for row in rows if row["benefit"] > BENEFIT_EPSILON]
    overlap = [row for row in rows if row["benefit"] > BENEFIT_EPSILON and row["harm_v2"]]
    safe = [row for row in rows if row["benefit"] > BENEFIT_EPSILON and not row["harm_v2"] and row["feasible"]]
    return {"synthetic_interaction": LABEL, "split": split, "dimension": dimension, "group": str(group),
            "candidate_count": len(rows), "episode_count": len({row["episode_id"] for row in rows}),
            "harm_positive_candidates": len(harmful), "harm_positive_episodes": len({row["episode_id"] for row in harmful}),
            "harm_positive_rate": len(harmful) / max(len(rows), 1),
            "unsafe_candidates": sum(row["gt_unsafe"] for row in rows), "adverse_response_candidates": sum(row["adverse_response"] for row in rows),
            "beneficial_candidates": len(beneficial), "beneficial_harm_candidates": len(overlap),
            "beneficial_harm_episodes": len({row["episode_id"] for row in overlap}),
            "safe_beneficial_candidates": len(safe), "safe_beneficial_episodes": len({row["episode_id"] for row in safe})}


def grouped(split, rows, dimension, key):
    groups = defaultdict(list)
    for row in rows:
        values = key(row); values = values if isinstance(values, (tuple, list)) else (values,)
        for value in values: groups[str(value)].append(row)
    if dimension == "context":
        for context in CONTEXTS: groups.setdefault(context, [])
    return [summary_row(split, dimension, group, groups[group]) for group in sorted(groups)]


def development_statistics(rows_by_split):
    result = {"label": LABEL, "generator_version": GENERATOR_VERSION, "test_labels_read": 0, "by_split": {}}
    for split, rows in rows_by_split.items():
        episodes = {row["episode_id"] for row in rows}
        result["by_split"][split] = {"episodes": len(episodes), "candidates": len(rows),
            "harm_positive_candidates": sum(row["harm_v2"] for row in rows),
            "harm_positive_episodes": len({row["episode_id"] for row in rows if row["harm_v2"]}),
            "unsafe_candidates": sum(row["gt_unsafe"] for row in rows),
            "adverse_response_candidates": sum(row["adverse_response"] for row in rows),
            "event_counts": {name: sum(row[name] for row in rows) for name in ("excessive_deceleration", "abrupt_lateral_response", "abrupt_heading_change")}}
    return result


def v1_distributions(manifest):
    # Development-only comparison: old TEST IDs are not used in readiness.
    episodes = [row for row in manifest["episodes"] if row["split"] in ("train", "validation")]
    def counts(field): return dict(Counter(str(row[field]) for row in episodes))
    contexts = Counter(context[:2] for row in episodes for context in row["context_labels"] if context[:2] in CONTEXTS)
    candidates = sum(len(row["candidate_ids"]) for row in episodes)
    old_summary = json.loads((PROJECT_ROOT / "results_dev" / "phase5b17b_independent_harm_target" / "summary.json").read_text(encoding="utf-8"))
    old_distribution = json.loads((PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage" / "dataset_statistics_old_vs_new.json").read_text(encoding="utf-8"))
    import csv
    with (PROJECT_ROOT / "results_dev" / "phase5b17b_independent_harm_target" / "by_action_harm.csv").open(encoding="utf-8") as handle:
        action_rows = [row for row in csv.DictReader(handle) if row["definition"] == "harm_A"]
    with (PROJECT_ROOT / "results_dev" / "phase5b17b_independent_harm_target" / "benefit_new_harm_quadrants.csv").open(encoding="utf-8") as handle:
        quadrants = [row for row in csv.DictReader(handle) if row["definition"] == "harm_A"]
    beneficial = sum(int(row["candidate_count"]) for row in quadrants if row["quadrant"].startswith("beneficial=1"))
    return {"development_episodes": len(episodes), "development_candidates": candidates, "split": dict(Counter(row["split"] for row in episodes)),
            "motion": counts("motion_type_evaluation_only"), "profile": counts("profile_id_split_only"), "contexts": dict(contexts),
            "action_candidates": {f"{row['split']}:{row['group']}": int(row["candidate_count"]) for row in action_rows},
            "development_unsafe_rate_from_frozen_1_7B": old_summary["hard_safety"],
            "beneficial_rate_development": beneficial / max(candidates, 1),
            "occlusion_rate_full_frozen_v1_from_existing_audit": old_distribution["new"]["occlusion_rate"],
            "test_labels_rematerialized": False}


def v2_distributions(episodes_by_split, rows_by_split, sealed_rows):
    all_episodes = sum(episodes_by_split.values(), []); all_rows = sum(rows_by_split.values(), [])
    context = Counter(name[:2] for episode in all_episodes for name in episode.context_labels)
    return {"episodes": len(all_episodes) + len(sealed_rows), "development_episodes": len(all_episodes),
            "sealed_test_episode_ids": len(sealed_rows), "development_candidates": len(all_rows),
            "split": {**{name: len(values) for name, values in episodes_by_split.items()}, "test": len(sealed_rows)},
            "motion_development": dict(Counter(episode.motion_type for episode in all_episodes)),
            "action_candidates_development": dict(Counter(ACTION_NAMES[row["action"]] for row in all_rows)),
            "profile_development": dict(Counter(str(episode.profile_id) for episode in all_episodes)),
            "contexts_development": dict(context),
            "beneficial_rate_development": float(np.mean([row["benefit"] > BENEFIT_EPSILON for row in all_rows])),
            "unsafe_rate_development": float(np.mean([row["gt_unsafe"] for row in all_rows])),
            "harm_v2_rate_development": float(np.mean([row["harm_v2"] for row in all_rows])),
            "occlusion_rate_development": float(np.mean([episode.history_occlusion_rate for episode in all_episodes]))}


def shortcut_audit(group_rows, dimension):
    rows = [row for row in group_rows if row["dimension"] == dimension]
    rates = {f"{row['split']}:{row['group']}": row["harm_positive_rate"] for row in rows if row["candidate_count"]}
    maximum = max(rates.values(), default=0.0); minimum = min(rates.values(), default=0.0)
    near_deterministic = [key for key, value in rates.items() if value > .95]
    return {"dimension": dimension, "rates": rates, "maximum_rate": maximum, "minimum_rate": minimum,
            "near_deterministic_groups_over_95_percent": near_deterministic,
            "passed": not near_deterministic, "warning": bool(maximum - minimum > .50)}


def leakage_audit(episodes_by_split, manifest, v1_before, v1_after):
    episode_splits = defaultdict(set); candidate_ids = set(); duplicate_candidates = []
    for row in manifest["episodes"]:
        episode_splits[row["episode_id"]].add(row["split"])
        for candidate in row["candidate_ids"]:
            if candidate in candidate_ids: duplicate_candidates.append(candidate)
            candidate_ids.add(candidate)
    train_profiles = {episode.profile_id for episode in episodes_by_split["train"]}
    return {"label": LABEL, "old_manifest_v1_sha256_before": v1_before, "old_manifest_v1_sha256_after": v1_after,
            "old_manifest_v1_unchanged": v1_before == v1_after,
            "same_episode_cross_split": sorted(key for key, values in episode_splits.items() if len(values) > 1),
            "duplicate_candidate_ids": duplicate_candidates, "held_out_profiles": list(HELD_OUT_PROFILE_IDS),
            "held_out_profiles_in_train": sorted(train_profiles & set(HELD_OUT_PROFILE_IDS)),
            "split_before_candidate_branching": True, "profile_id_runtime_input": False,
            "GT_labels_runtime_input": False, "test_episode_ids_assigned": TEST_ID_COUNT,
            "test_trajectories_materialized": 0, "test_benefit_labels_read": 0, "test_harm_labels_read": 0,
            "passed": v1_before == v1_after and not duplicate_candidates and not any(len(x) > 1 for x in episode_splits.values()) and not train_profiles.intersection(HELD_OUT_PROFILE_IDS)}


def readiness(overall, context, action_audit, profile_audit, leakage):
    o = {(row["split"]): row for row in overall}
    c = {(row["split"], row["group"]): row for row in context}
    semantic_source = inspect.getsource(__import__("src.data.adverse_response_protocol", fromlist=["derive_adverse_response_events"]).derive_adverse_response_events)
    gates = {
        "gate_1_independent_semantics": {"passed": all(term not in semantic_source for term in ("benefit", "total_cost", "best_action", "profile_id")) and all(row["unsafe_candidates"] <= row["harm_positive_candidates"] for row in overall)},
        "gate_2_positive_support": {"passed": o["train"]["harm_positive_episodes"] >= 20 and o["validation"]["harm_positive_episodes"] >= 10 and c[("train", "C7")]["harm_positive_episodes"] >= 5 and c[("validation", "C7")]["harm_positive_episodes"] >= 5,
                                     "requirements": {"train_overall": 20, "validation_overall": 10, "train_C7": 5, "validation_C7": 5}},
        "gate_3_benefit_risk_tradeoff": {"passed": o["train"]["beneficial_harm_episodes"] >= 5 and o["validation"]["beneficial_harm_episodes"] >= 3,
                                          "requirements": {"train": 5, "validation": 3}},
        "gate_4_safe_beneficial_remains": {"passed": o["train"]["safe_beneficial_episodes"] >= 5 and o["validation"]["safe_beneficial_episodes"] >= 3},
        "gate_5_not_action_shortcut": {"passed": action_audit["passed"], "audit": action_audit},
        "gate_6_not_profile_shortcut": {"passed": profile_audit["passed"], "audit": profile_audit},
    }
    return {"label": LABEL, "gates": gates, "leakage_required_and_passed": leakage["passed"],
            "ready_for_independent_harm_head_training": all(value["passed"] for value in gates.values()) and leakage["passed"],
            "model_training_performed": False, "next_stage_automatically_started": False}


def make_figures(output, overall, action, motion, context):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name): path=folder/name;plt.suptitle(LABEL,fontsize=7);plt.tight_layout();plt.savefig(path,dpi=150);plt.close();paths.append(str(path))
    for rows, name, title in ((action,"by_action_harm.png","P(harm_v2 | action)"),(motion,"by_motion_harm.png","P(harm_v2 | motion)"),(context,"by_context_harm.png","P(harm_v2 | context)")):
        val=[r for r in rows if r["split"]=="validation"];plt.figure(figsize=(9,4));plt.bar([r["group"] for r in val],[r["harm_positive_rate"] for r in val]);plt.xticks(rotation=25,ha="right");plt.ylabel(title);save(name)
    plt.figure();plt.bar([r["split"] for r in overall],[r["harm_positive_episodes"] for r in overall]);plt.ylabel("harm-positive episodes");save("overall_positive_episodes.png")
    plt.figure();plt.bar([r["split"] for r in overall],[r["beneficial_harm_episodes"] for r in overall]);plt.ylabel("beneficial & harm_v2 episodes");save("benefit_risk_tradeoff.png")
    return paths


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7C: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frozen_sources = {
        "manifest_v1": args.manifest_v1,
        "old_label_source": PROJECT_ROOT / "scripts" / "run_phase5a_context_value.py",
        "old_target_schema": PROJECT_ROOT / "src" / "multimodal" / "temporal_schema.py",
        "frozen_phase5b_model": PROJECT_ROOT / "src" / "models" / "rich_temporal_small_transformer.py",
        "frozen_arbitration": PROJECT_ROOT / "src" / "decision" / "large_context_arbitrator.py",
    }
    frozen_before={name:sha256(path) for name,path in frozen_sources.items()};v1_before=frozen_before["manifest_v1"]
    episodes={"train":build_development_split("train",TRAIN_SIZE,GENERATOR_SEED,RISK_SEED),
              "validation":build_development_split("validation",VALIDATION_SIZE,GENERATOR_SEED+1000,RISK_SEED+1000)}
    rows={split:candidate_rows(values) for split,values in episodes.items()}
    sealed=sealed_test_manifest_rows(TEST_ID_COUNT,GENERATOR_SEED+2000,RISK_SEED+2000)
    manifest={"label":LABEL,"version":"phase5b_manifest_v2","immutable_after_freeze":True,
              "generator":{"version":GENERATOR_VERSION,"generator_seed":GENERATOR_SEED,"risk_seed":RISK_SEED},
              "protocol":{"split_before_candidate_branching":True,"history_frames":20,"future_frames":10,"sample_rate_hz":10.0,
                          "development_profiles":list(DEVELOPMENT_PROFILE_IDS),"held_out_test_profiles":list(HELD_OUT_PROFILE_IDS),"test_labels_sealed":True},
              "episodes":[episode_manifest_row(item) for values in episodes.values() for item in values]+sealed}
    overall=[summary_row(split,"overall","all",values) for split,values in rows.items()]
    action=sum((grouped(split,values,"action",lambda r:ACTION_NAMES[r["action"]]) for split,values in rows.items()),[])
    motion=sum((grouped(split,values,"motion",lambda r:r["motion"]) for split,values in rows.items()),[])
    profile=sum((grouped(split,values,"profile",lambda r:f"profile_{r['profile']}") for split,values in rows.items()),[])
    context=sum((grouped(split,values,"context",lambda r:tuple(name[:2] for name in r["contexts"])) for split,values in rows.items()),[])
    action_shortcut=shortcut_audit(action,"action");profile_shortcut=shortcut_audit(profile,"profile")
    io.write_json(args.output_dir/"phase5b_manifest_v2.json",manifest)
    file_digest=sha256(args.output_dir/"phase5b_manifest_v2.json");content_digest=canonical_sha(manifest)
    frozen_after={name:sha256(path) for name,path in frozen_sources.items()};v1_after=frozen_after["manifest_v1"]
    leakage=leakage_audit(episodes,manifest,v1_before,v1_after)
    leakage["frozen_source_checksums_before"]=frozen_before;leakage["frozen_source_checksums_after"]=frozen_after
    leakage["all_frozen_sources_unchanged"]=frozen_before==frozen_after
    leakage["passed"]=bool(leakage["passed"] and frozen_before==frozen_after)
    gate=readiness(overall,context,action_shortcut,profile_shortcut,leakage)
    comparison={"label":LABEL,"manifest_v1":v1_distributions(json.loads(args.manifest_v1.read_text(encoding="utf-8"))),
                "manifest_v2":v2_distributions(episodes,rows,sealed),"comparison_limitations":["v1 occlusion ratio is the already-frozen full-manifest aggregate","old TEST labels were not rematerialized"]}
    figures=make_figures(args.output_dir,overall,action,motion,context)
    stats=development_statistics(rows)
    io.write_json(args.output_dir/"adverse_response_protocol.json",protocol_definition())
    io.write_json(args.output_dir/"generator_risk_factor_registry.json",risk_factor_registry())
    io.write_json(args.output_dir/"harm_v2_definition.json",harm_definition())
    io.write_json(args.output_dir/"new_episode_statistics.json",stats)
    io.write_csv(args.output_dir/"harm_v2_prevalence.csv",overall)
    io.write_csv(args.output_dir/"benefit_harm_v2_overlap.csv",overall)
    io.write_csv(args.output_dir/"safe_beneficial_space.csv",overall)
    io.write_csv(args.output_dir/"by_action_harm.csv",action);io.write_csv(args.output_dir/"by_motion_harm.csv",motion)
    io.write_csv(args.output_dir/"by_profile_harm.csv",profile);io.write_csv(args.output_dir/"by_context_harm.csv",context)
    io.write_json(args.output_dir/"manifest_v1_vs_v2.json",comparison);io.write_json(args.output_dir/"split_leakage_audit.json",leakage)
    io.write_json(args.output_dir/"manifest_v2_checksum.json",{"label":LABEL,"algorithm":"SHA256","file_sha256":file_digest,"canonical_content_sha256":content_digest})
    io.write_json(args.output_dir/"readiness_gate.json",gate)
    summary={"label":LABEL,"stage":"Phase 5B-1.7C Adverse-Response Generator Expansion & Harm-v2 Dataset Readiness",
             "overall":{row["split"]:row for row in overall},"C7":{row["split"]:row for row in context if row["group"]=="C7"},
             "action_shortcut":action_shortcut,"profile_shortcut":profile_shortcut,
             "manifest_v2":{"episodes":len(manifest["episodes"]),"development_episodes":TRAIN_SIZE+VALIDATION_SIZE,"sealed_test_episode_ids":TEST_ID_COUNT,"file_sha256":file_digest},
             "leakage_passed":leakage["passed"],"readiness":gate,"test_reads":0,"optimizer_steps":0,"backward_calls":0,
             "model_training_performed":False,"phase5b18_started":False,"phase5b2_started":False,"figures":figures}
    io.write_json(args.output_dir/"summary.json",summary);print(json.dumps(io.clean(summary),indent=2),flush=True)


if __name__=="__main__":main()
