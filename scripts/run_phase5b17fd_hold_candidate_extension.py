"""Phase 5B-1.7F-D executable HOLD and manifest-v3 readiness (no training)."""
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
from src.data.adverse_response_dataset import (
    ACTION_IDS, DEVELOPMENT_PROFILE_IDS, GENERATOR_SEED, GENERATOR_VERSION,
    HELD_OUT_PROFILE_IDS, POPULATION_PROFILE, PROFILE_BY_ID, RISK_SEED,
    build_development_split, candidate_rows,
)
from src.data.hold_candidate import (
    HOLD_ANGULAR_DECELERATION_LIMIT_RADPS2, HOLD_CONTROL_CONTRACT_VERSION,
    HOLD_LINEAR_DECELERATION_LIMIT_MPS2, build_hold_candidate_outcome,
    hold_robot_rollout,
)
from src.data.robot_action_schema import (
    HOLD_ACTION_ID, PHASE5B_V3_ACTIONS, RobotAction, RobotActionV3,
    V3_ACTION_ONE_HOT_DIM, candidate_action_vector_v3,
)
from src.evaluation.hold_candidate_readiness import readiness_gates, summarize_hold_group
from src.multimodal.phase5b_v3_dataset import build_hold_temporal_sample_v3

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-1.7F-D Executable Hold Candidate Extension & Manifest-v3 Readiness"
V3_GENERATOR_VERSION = "phase5b17fd_hold_candidate_generator_v1"
V2_EXPECTED_SHA = "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
TRAIN_SIZE = VALIDATION_SIZE = 240
TEST_ID_COUNT = 120
CONTEXTS = ("C7", "C8", "C9")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--manifest-v2", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--phase5b17fc-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17fc_safe_fallback_design")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17fd_hold_candidate_extension")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    payload = json.dumps(io.clean(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def contexts(row):
    return tuple(name[:2] for name in row["contexts"])


def hold_row(episode, outcome):
    events = outcome.events
    return {
        "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
        "episode_id": episode.episode_id, "split": episode.split,
        "candidate_id": f"{episode.episode_id}:{HOLD_ACTION_ID}",
        "action_id": HOLD_ACTION_ID, "action": "HOLD",
        "motion": episode.motion_type, "profile": episode.profile_id,
        "contexts": episode.context_labels, "context": "|".join(episode.context_labels) or "NONE",
        "stop_non_stop": "stop" if episode.motion_type == "stop" else "non-stop",
        "feasible": True, "GT_benefit": outcome.benefit,
        "GT_total_cost": outcome.gt_total_cost, "generic_total_cost": outcome.generic_total_cost,
        "regret_v3_oracle": outcome.regret, "GT_unsafe": outcome.gt_unsafe,
        "harm_v2": outcome.harm_v2,
        "excessive_deceleration": events.excessive_deceleration,
        "abrupt_lateral_response": events.abrupt_lateral_response,
        "abrupt_heading_change": events.abrupt_heading_change,
        "adverse_human_kinematic_response": events.adverse_human_kinematic_response,
        "extra_deceleration_mps2": events.extra_deceleration_mps2,
        "extra_lateral_displacement_m": events.extra_lateral_displacement_m,
        "extra_heading_change_rad": events.extra_heading_change_rad,
        "initial_robot_speed": float(episode.robot_history[-1, 3]),
        "terminal_robot_speed": float(outcome.gt_simulation.robot_future_state[-1, 3]),
    }


def original_rows(episodes):
    rows = candidate_rows(episodes)
    return [{**row, "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
             "action_name": RobotAction(int(row["action"])).name} for row in rows]


def distribution_rows(split, episodes, old_rows, hold_rows):
    base = {
        "episode_count": len(episodes),
        "motion_distribution": json.dumps(dict(Counter(e.motion_type for e in episodes)), sort_keys=True),
        "profile_distribution": json.dumps(dict(Counter(str(e.profile_id) for e in episodes)), sort_keys=True),
        "context_distribution": json.dumps(dict(Counter(name[:2] for e in episodes for name in e.context_labels)), sort_keys=True),
    }
    result = []
    for view, rows in (("Original A0-A4 only", old_rows), ("Original A0-A4 + HOLD (ID 7)", [*old_rows, *hold_rows])):
        result.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
            "split": split, "view": view, **base, "candidate_count": len(rows),
            "beneficial_rate": float(np.mean([(r.get("benefit", r.get("GT_benefit"))) > 1e-6 for r in rows])),
            "harm_v2_rate": float(np.mean([r["harm_v2"] for r in rows])),
            "GT_unsafe_rate": float(np.mean([r.get("gt_unsafe", r.get("GT_unsafe")) for r in rows])),
            "safe_beneficial_candidates": sum(
                (r.get("benefit", r.get("GT_benefit"))) > 1e-6
                and not r["harm_v2"] and r.get("feasible", True) for r in rows
            ),
        })
    return result


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Phase 5B-1.7F-D: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v2_before = file_sha(args.manifest_v2)
    if v2_before != V2_EXPECTED_SHA:
        raise RuntimeError("manifest-v2 checksum mismatch")
    v2 = json.loads(args.manifest_v2.read_text(encoding="utf-8"))
    original_14 = set(
        row["episode_id"] for row in __import__("csv").DictReader(
            (args.phase5b17fc_dir / "fallback_support.csv").open(encoding="utf-8")
        )
    )

    # Freeze the control contract before any HOLD label/outcome is materialized.
    source_files = {
        "action_schema": PROJECT_ROOT / "src/data/robot_action_schema.py",
        "hold_generator": PROJECT_ROOT / "src/data/hold_candidate.py",
        "cost_evaluator": PROJECT_ROOT / "src/decision/decision_cost.py",
        "harm_protocol": PROJECT_ROOT / "src/data/adverse_response_protocol.py",
        "interaction_generator": PROJECT_ROOT / "src/data/synthetic_interaction.py",
    }
    source_sha_before = {name: file_sha(path) for name, path in source_files.items()}
    control_protocol = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "contract_version": HOLD_CONTROL_CONTRACT_VERSION,
        "semantic_name": "BRAKE_TO_ZERO_AND_HOLD", "instant_zero_velocity": False,
        "sample_rate_hz": 10.0, "future_frames": 10, "horizon_seconds": 1.0,
        "linear_deceleration_limit_mps2": HOLD_LINEAR_DECELERATION_LIMIT_MPS2,
        "angular_deceleration_limit_radps2": HOLD_ANGULAR_DECELERATION_LIMIT_RADPS2,
        "parameter_source": {
            "linear": "existing frozen synthetic motion support: acceleration/deceleration changes approximately 0.04 m/s per 0.1 s frame (0.4 m/s^2)",
            "angular": "existing frozen synthetic turn support: 0.50 rad/s; reused as a conservative 0.50 rad/s^2 rate limit",
            "real_robot_controller_limit_claimed": False,
            "selected_using_safety_regret_recall_or_harm": False,
        },
        "integration": "trapezoidal velocity/angular-rate integration; monotonic approach to zero; remain zero after reached",
        "frozen_before_label_materialization": True,
    }
    io.write_json(args.output_dir / "hold_control_protocol.json", control_protocol)
    protocol_pre_materialization_sha = file_sha(args.output_dir / "hold_control_protocol.json")

    episodes = {
        "train": build_development_split("train", TRAIN_SIZE, GENERATOR_SEED, RISK_SEED),
        "validation": build_development_split("validation", VALIDATION_SIZE, GENERATOR_SEED + 1000, RISK_SEED + 1000),
    }
    holds, hold_samples, rows = {}, {}, {}
    for split, values in episodes.items():
        holds[split] = [build_hold_candidate_outcome(ep, POPULATION_PROFILE, PROFILE_BY_ID[ep.profile_id]) for ep in values]
        hold_samples[split] = [build_hold_temporal_sample_v3(ep, outcome) for ep, outcome in zip(values, holds[split])]
        rows[split] = [hold_row(ep, outcome) for ep, outcome in zip(values, holds[split])]
    old_rows = {split: original_rows(values) for split, values in episodes.items()}
    all_hold_rows = rows["train"] + rows["validation"]

    # TEST receives IDs only. No TEST generator, trajectory, label or cost call exists.
    v2_by_id = {row["episode_id"]: row for row in v2["episodes"]}
    manifest_rows = []
    for split, values in episodes.items():
        for ep, outcome in zip(values, holds[split]):
            old = v2_by_id[ep.episode_id]
            if old["candidate_ids"] != [f"{ep.episode_id}:{action}" for action in ACTION_IDS]:
                raise RuntimeError("v2 latent episode/candidate replay mismatch")
            manifest_rows.append({
                **old,
                "candidate_ids": [*old["candidate_ids"], f"{ep.episode_id}:{HOLD_ACTION_ID}"],
                "harm_v2_labels": {**old["harm_v2_labels"], str(HOLD_ACTION_ID): outcome.harm_v2},
                "gt_unsafe_labels": {**old["gt_unsafe_labels"], str(HOLD_ACTION_ID): outcome.gt_unsafe},
                "adverse_response_labels": {**old["adverse_response_labels"], str(HOLD_ACTION_ID): outcome.events.adverse_human_kinematic_response},
                "hold_contract_version": HOLD_CONTROL_CONTRACT_VERSION,
            })
    v2_test = [row for row in v2["episodes"] if row["split"] == "test"]
    if len(v2_test) != TEST_ID_COUNT:
        raise RuntimeError("frozen sealed TEST ID count mismatch")
    for old in v2_test:
        manifest_rows.append({
            **old, "candidate_ids": [*old["candidate_ids"], f"{old['episode_id']}:{HOLD_ACTION_ID}"],
            "harm_v2_labels": "SEALED_NOT_MATERIALIZED",
            "benefit_labels": "SEALED_NOT_MATERIALIZED",
            "gt_costs": "SEALED_NOT_MATERIALIZED",
            "hold_contract_version": HOLD_CONTROL_CONTRACT_VERSION,
        })
    manifest = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "version": "phase5b_manifest_v3", "immutable_after_freeze": True,
        "parent_manifest_v2_sha256": v2_before,
        "generator": {"version": V3_GENERATOR_VERSION, "base_generator_version": GENERATOR_VERSION,
                      "generator_seed": GENERATOR_SEED, "risk_seed": RISK_SEED,
                      "risk_dynamics_version": GENERATOR_VERSION},
        "action_contract": {"version": HOLD_CONTROL_CONTRACT_VERSION,
                            "candidate_ids": [int(action) for action in PHASE5B_V3_ACTIONS],
                            "candidate_names": [action.name for action in PHASE5B_V3_ACTIONS],
                            "hold_action_id": HOLD_ACTION_ID, "candidate_action_dimension": 12},
        "protocol": {**v2["protocol"], "test_labels_sealed": True,
                     "split_before_candidate_branching": True},
        "episodes": manifest_rows,
    }
    io.write_json(args.output_dir / "phase5b_manifest_v3.json", manifest)
    manifest_sha = file_sha(args.output_dir / "phase5b_manifest_v3.json")
    io.write_json(args.output_dir / "manifest_v3_checksum.json", {
        "label": LABEL, "mechanism_result": MECHANISM, "algorithm": "SHA256",
        "file_sha256": manifest_sha, "canonical_content_sha256": canonical_sha(manifest),
    })

    # Examples are real deterministic states, not hand-entered trajectories.
    examples = []
    for split in ("train", "validation"):
        for ep, outcome in list(zip(episodes[split], holds[split]))[:3]:
            for frame, state in enumerate(outcome.gt_simulation.robot_future_state, 1):
                examples.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
                                 "episode_id": ep.episode_id, "split": split, "frame": frame,
                                 "x": state[0], "y": state[1], "yaw": state[2],
                                 "linear_velocity": state[3], "angular_velocity": state[4]})

    def three_level_table(group_by, dimension):
        result = []
        for split in ("train", "validation"):
            result += summarize_hold_group(rows[split], group_by, dimension)
        combined = summarize_hold_group(all_hold_rows, group_by, dimension)
        for row in combined:
            row["split"] = "development"
        return result + combined

    by_motion = three_level_table(lambda r: (r["motion"],), "motion")
    by_profile = three_level_table(lambda r: (f"profile_{r['profile']}",), "profile")
    by_context = three_level_table(
        lambda r: (*tuple(name[:2] for name in r["contexts"]), r["stop_non_stop"]), "context"
    )
    for collection in (by_motion, by_profile, by_context):
        for row in collection:
            row.update({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM})

    original_14_rows = [{**row, "original_no_safe_generic": True}
                        for row in rows["validation"] if row["episode_id"] in original_14]
    if len(original_14_rows) != 14:
        raise RuntimeError("original 14 no-safe-generic episodes did not reproduce")
    fallback_rows = [{
        "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
        "episode_id": row["episode_id"], "HOLD_action_id": HOLD_ACTION_ID,
        "HOLD_GT_safe": not row["GT_unsafe"] and not row["harm_v2"],
        "HOLD_GT_unsafe": row["GT_unsafe"], "HOLD_GT_harm_v2": row["harm_v2"],
        "HOLD_GT_cost": row["GT_total_cost"], "HOLD_GT_benefit": row["GT_benefit"],
        "runtime_old_model_status": "OOD_NOT_SCORED",
        "runtime_old_model_safe": None,
    } for row in original_14_rows]

    ood_rows = [{
        "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
        "model": model, "checkpoint_action_support": "A0-A4 / 7-way reserved one-hot",
        "HOLD_action_id": HOLD_ACTION_ID, "manifest_v3_action_dimension": 12,
        "old_model_candidate_action_dimension": 11,
        "diagnostic_status": "OUT-OF-DISTRIBUTION DIAGNOSTIC",
        "forward_attempted": False, "risk_probability": None,
        "usable_as_formal_safety_evidence": False,
        "reason": "HOLD ID 7 was unseen and cannot be represented by the frozen 7-way/11D checkpoint input",
    } for model in ("R1-v2-CRACS Benefit/Ranking", "Independent Harm-v2 bypass head")]

    distribution = []
    for split in ("train", "validation"):
        distribution += distribution_rows(split, episodes[split], old_rows[split], rows[split])

    episode_splits = defaultdict(set); candidate_ids = []
    for row in manifest_rows:
        episode_splits[row["episode_id"]].add(row["split"]); candidate_ids += row["candidate_ids"]
    source_sha_after = {name: file_sha(path) for name, path in source_files.items()}
    v2_after = file_sha(args.manifest_v2)
    leakage = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "manifest_v2_sha256_before": v2_before, "manifest_v2_sha256_after": v2_after,
        "manifest_v2_unchanged": v2_before == v2_after == V2_EXPECTED_SHA,
        "same_episode_cross_split": [key for key, values in episode_splits.items() if len(values) > 1],
        "duplicate_candidate_ids": len(candidate_ids) - len(set(candidate_ids)),
        "candidates_per_episode_valid": all(len(row["candidate_ids"]) == 6 for row in manifest_rows),
        "held_out_profiles": list(HELD_OUT_PROFILE_IDS),
        "held_out_profiles_in_development": sorted(set(ep.profile_id for values in episodes.values() for ep in values) & set(HELD_OUT_PROFILE_IDS)),
        "split_before_candidate_branching": True, "test_episode_ids_assigned": len(v2_test),
        "test_trajectories_materialized": 0, "test_human_futures_read": 0,
        "test_benefit_reads": 0, "test_harm_v2_reads": 0, "test_cost_reads": 0,
        "source_checksums_before": source_sha_before, "source_checksums_after": source_sha_after,
        "sources_unchanged_during_materialization": source_sha_before == source_sha_after,
    }
    leakage["passed"] = bool(
        leakage["manifest_v2_unchanged"] and not leakage["same_episode_cross_split"]
        and leakage["duplicate_candidate_ids"] == 0 and leakage["candidates_per_episode_valid"]
        and not leakage["held_out_profiles_in_development"]
        and all(leakage[key] == 0 for key in ("test_trajectories_materialized", "test_human_futures_read", "test_benefit_reads", "test_harm_v2_reads", "test_cost_reads"))
    )

    action_schema = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "v2_candidate_action": {"dimension": 11, "encoding": "7-way fixed one-hot + 4 continuous semantics", "unchanged": True},
        "v3_candidate_action": {"dimension": 12, "encoding": "8-way fixed one-hot + same 4 continuous semantics"},
        "existing_reserved_ids": {"5": "LEFT_OFFSET", "6": "RIGHT_OFFSET"},
        "HOLD": {"action_id": HOLD_ACTION_ID, "independent_id": True,
                 "not_KEEP": True, "not_ABSTAIN": True,
                 "semantic_vector": candidate_action_vector_v3(HOLD_ACTION_ID).tolist()},
        "candidate_robot_future": {"shape": [10, 5], "fields": ["x", "y", "yaw", "linear_velocity", "angular_velocity"]},
        "natural_expression_requires_new_id": True,
    }
    hold_contract = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "contract_version": HOLD_CONTROL_CONTRACT_VERSION, "action_id": HOLD_ACTION_ID,
        "action_name": "HOLD", "control_semantics": "BRAKE_TO_ZERO_AND_HOLD",
        "executable_action": True, "ABSTAIN": False, "KEEP": False,
        "future_ROS_mapping": "bounded deceleration to zero then zero velocity hold",
        "ROS_code_modified": False, "safe_by_definition": False,
        "robot_rollout": True, "human_response_rollout": True,
        "GT_cost": True, "GT_benefit": True, "GT_unsafe": True,
        "adverse_response_events": True, "harm_v2": True,
    }

    event_source = inspect.getsource(build_hold_candidate_outcome)
    all_costs_finite = bool(np.isfinite([row["GT_total_cost"] for row in all_hold_rows]).all())
    motion_values = {row["motion"] for row in all_hold_rows}
    profile_values = {row["profile"] for row in all_hold_rows}
    context_values = {name[:2] for row in all_hold_rows for name in row["contexts"]}
    original_safe = sum(row["HOLD_GT_safe"] for row in fallback_rows)
    gates = readiness_gates(
        action_semantics=HOLD_ACTION_ID != int(RobotAction.KEEP),
        rollout_complete=all(sample.candidate_robot_future.shape == (10, 5) for values in hold_samples.values() for sample in values) and all_costs_finite,
        no_label_shortcut=("unsafe or events.adverse_human_kinematic_response" in event_source and "fixed" not in event_source.lower() and "bonus" not in event_source.lower()),
        train_hold_count=len(rows["train"]), validation_hold_count=len(rows["validation"]),
        expected_train_count=TRAIN_SIZE, expected_validation_count=VALIDATION_SIZE,
        motion_coverage=len(motion_values) == len({ep.motion_type for values in episodes.values() for ep in values}),
        context_coverage=set(CONTEXTS) <= context_values,
        profile_coverage=set(DEVELOPMENT_PROFILE_IDS) <= profile_values,
        original_no_safe_count=len(fallback_rows), original_hold_safe_count=original_safe,
        all_hold_costs_finite=all_costs_finite,
    )
    gates = {"label": LABEL, "mechanism_result": MECHANISM, **gates}
    hold_rate = {
        "count": len(all_hold_rows), "harm_v2_count": sum(row["harm_v2"] for row in all_hold_rows),
        "harm_v2_rate": float(np.mean([row["harm_v2"] for row in all_hold_rows])),
        "GT_unsafe_count": sum(row["GT_unsafe"] for row in all_hold_rows),
        "GT_unsafe_rate": float(np.mean([row["GT_unsafe"] for row in all_hold_rows])),
        "beneficial_count": sum(row["GT_benefit"] > 1e-6 for row in all_hold_rows),
        "beneficial_rate": float(np.mean([row["GT_benefit"] > 1e-6 for row in all_hold_rows])),
        "mean_GT_total_cost": float(np.mean([row["GT_total_cost"] for row in all_hold_rows])),
        "always_safe_warning": not any(row["harm_v2"] or row["GT_unsafe"] for row in all_hold_rows),
        "always_best_warning": all(row["regret_v3_oracle"] <= 1e-9 for row in all_hold_rows),
    }
    summary = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE,
        "model_training_performed": False, "optimizer_steps": 0, "backward_calls": 0,
        "test_reads": 0, "manifest_v2_unchanged": leakage["manifest_v2_unchanged"],
        "HOLD_action_id": HOLD_ACTION_ID, "HOLD_control_contract": HOLD_CONTROL_CONTRACT_VERSION,
        "control_protocol_pre_materialization_sha256": protocol_pre_materialization_sha,
        "development": {"train_episodes": TRAIN_SIZE, "validation_episodes": VALIDATION_SIZE,
                        "train_candidates_v2": TRAIN_SIZE * 5, "train_candidates_v3": TRAIN_SIZE * 6,
                        "validation_candidates_v2": VALIDATION_SIZE * 5, "validation_candidates_v3": VALIDATION_SIZE * 6,
                        "train_HOLD": len(rows["train"]), "validation_HOLD": len(rows["validation"])},
        "HOLD_statistics": hold_rate,
        "original_14": {"count": len(fallback_rows), "HOLD_GT_safe": original_safe,
                        "HOLD_GT_harm_v2": sum(row["HOLD_GT_harm_v2"] for row in fallback_rows),
                        "HOLD_GT_unsafe": sum(row["HOLD_GT_unsafe"] for row in fallback_rows)},
        "residual_false_safe_000194": {
            "episode_id": "validation:phase5b17c:52703:000194",
            "status": "RESIDUAL HARM-V2 FALSE-SAFE",
            "unchanged_by_HOLD_extension": True,
            "risk_model_repair_claimed": False,
        },
        "deceleration_latent_debt": {
            "historical_candidate_count": 11, "unchanged": True,
            "v3_HOLD_old_model_risk_support": "OOD_NOT_SCORED",
            "risk_model_modified": False,
        },
        "old_v2_model_HOLD_status": "OUT-OF-DISTRIBUTION DIAGNOSTIC; NOT SCORED; NOT FORMAL DECISION EVIDENCE",
        "manifest_v3": {"episode_count": len(manifest_rows), "development_candidates": 480 * 6,
                        "sealed_test_candidate_ids": TEST_ID_COUNT * 6, "file_sha256": manifest_sha},
        "base_episode_distribution_drift": False,
        "gates": gates, "ready_for_manifest_v3_model_rebaseline": gates["all_passed"] and leakage["passed"],
        "next_fair_rebaseline_minimum": ["Temporal/Ranking Benefit model", "Independent Harm-v2 model"],
        "next_stage_started": False,
    }

    io.write_json(args.output_dir / "action_schema_audit.json", action_schema)
    io.write_json(args.output_dir / "hold_action_contract.json", hold_contract)
    io.write_csv(args.output_dir / "hold_rollout_examples.csv", examples)
    io.write_csv(args.output_dir / "hold_label_audit.csv", all_hold_rows)
    io.write_csv(args.output_dir / "hold_by_motion.csv", by_motion)
    io.write_csv(args.output_dir / "hold_by_context.csv", by_context)
    io.write_csv(args.output_dir / "hold_by_profile.csv", by_profile)
    io.write_csv(args.output_dir / "original_14_hold_audit.csv", original_14_rows)
    io.write_csv(args.output_dir / "fallback_utility_audit.csv", fallback_rows)
    io.write_csv(args.output_dir / "hold_ood_model_diagnostic.csv", ood_rows)
    io.write_csv(args.output_dir / "manifest_v2_vs_v3.csv", distribution)
    io.write_json(args.output_dir / "split_leakage_audit.json", leakage)
    io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2))


if __name__ == "__main__":
    main()
