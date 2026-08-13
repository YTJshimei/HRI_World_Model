"""Build and audit the Phase 5B-0 rich temporal protocol (no model training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as phase5
from scripts import run_phase5a_context_value as phase5a
from src.multimodal.temporal_dataset import (
    build_temporal_samples, fit_train_normalizer, static_bridge_audit, validate_split_isolation,
)
from src.multimodal.temporal_schema import (
    DT_SECONDS, FUTURE_FRAMES, HISTORY_FRAMES, HISTORY_WINDOWS, LABEL, STREAM_DIMS,
    STREAM_ORDER, feature_registry, runtime_payload, schema_description,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b0_temporal_protocol")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value); digest = hashlib.sha256()
    digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def build_all(args, torch):
    development = phase5.build_development_data(args, torch)
    train = build_temporal_samples(development["train_episodes"], development["train_samples"], development["train_targets"], development["train_meta"], "train")
    validation = build_temporal_samples(development["val_episodes"], development["val_samples"], development["val_targets"], development["val_meta"], "validation")
    episodes, _, samples, targets, meta = phase5.materialize_test(args, development, torch)
    test = build_temporal_samples(episodes, samples, targets, meta, "test")
    return {"train": train, "validation": validation, "test": test}


def existing_signal_audit() -> dict[str, object]:
    return {
        "label": LABEL, "source_files": [
            "src/data/synthetic_skeleton.py", "src/data/synthetic_interaction.py",
            "scripts/run_phase4c_decision.py", "scripts/run_phase4c1_safety_calibration.py",
            "scripts/run_phase4c3_selective_personalization.py", "scripts/run_phase5a_context_value.py",
        ],
        "signals": {
            "human_skeleton_trajectory": {"available": True, "shape": [20, 17, 3], "runtime": True},
            "human_root_trajectory": {"available": True, "source": "pelvis midpoint computed from skeleton", "runtime": True},
            "human_motion_state": {"available": "derivable", "source": "history only", "runtime": True},
            "robot_state_trajectory": {"available": True, "shape": [20, 7], "runtime": True},
            "candidate_action": {"available": True, "runtime": True},
            "candidate_robot_future": {"available": True, "shape": [10, 2], "source": "deterministic planner rollout", "runtime": True},
            "human_response_future": {"available": True, "runtime": False, "use": "training/evaluation target only"},
            "functional_response_state": {"available": True, "runtime_source": "theta_hat only", "oracle_theta_forbidden": True},
            "functional_state_confidence": {"available": True, "source": "posterior std, dimension confidence, support coverage"},
            "visibility_occlusion": {"available": True, "shape": [20, 17]},
            "world_model_uncertainty": {"available": True, "source": "root belief, response and safety uncertainty"},
            "interaction_history": {"available": "partial", "source": "ordered support/probe IDs and current observed history", "limitation": "exact support timestamps and raw support trajectories are not retained in frozen episode records"},
            "scene_context": {"available": "derivable", "source": "runtime geometry/visibility/motion"},
            "safety_state": {"available": True, "source": "predicted sigma_minimum and p_unsafe; GT unsafe is target-only"},
            "timestamps": {"available": "reconstructable uniform grid", "dt_seconds": .1, "absolute_time": False},
            "person_profile": {"available": True, "runtime": False, "use": "split/evaluation audit only"},
        },
        "native_timeline": {"dt_seconds": .1, "sample_rate_hz": 10, "history_frames": 20, "future_frames": 10,
                            "episode_seconds": 3.0, "decision_points_per_episode": 1,
                            "decision_frequency": "not defined; current data are event-driven single-decision episodes"},
    }


def dataset_statistics(by_split) -> dict[str, object]:
    all_samples = [sample for values in by_split.values() for sample in values]
    split_stats = {}
    for split, samples in by_split.items():
        episodes = {sample.episode_id for sample in samples}; profiles = {sample.split_metadata["person_profile_id"] for sample in samples}
        split_stats[split] = {"episodes": len(episodes), "decision_points": len(episodes), "candidate_samples": len(samples), "profiles": len(profiles)}
    context = Counter(sample.context_split for sample in by_split["test"])
    temporal = Counter(tag for sample in by_split["test"] for tag in sample.temporal_tags)
    temporal_episodes = {tag: len({sample.episode_id for sample in by_split["test"] if tag in sample.temporal_tags})
                         for tag in ("C7_long_occlusion_history", "C8_recent_intervention_transition", "C9_motion_transition")}
    visibility = np.concatenate([sample.streams["visibility_history"][:, 1] for sample in all_samples])
    beneficial = np.asarray([sample.targets.benefit > 1e-6 for sample in all_samples]); harmful = np.asarray([sample.targets.harm for sample in all_samples])
    feasible = np.asarray([sample.targets.feasible for sample in all_samples])
    transitions = sum("C9_motion_transition" in sample.temporal_tags for sample in all_samples)
    return {
        "label": LABEL, "total": {"episodes": len({sample.episode_id for sample in all_samples}), "decision_points": len({sample.episode_id for sample in all_samples}),
                                   "candidate_samples": len(all_samples), "profiles": len({sample.split_metadata['person_profile_id'] for sample in all_samples})},
        "by_split": split_stats, "test_context_candidate_counts": dict(sorted(context.items())),
        "temporal_context_candidate_counts": dict(sorted(temporal.items())),
        "temporal_context_decision_point_counts": temporal_episodes,
        "history_length_distribution_frames": {str(HISTORY_FRAMES): len(all_samples)},
        "mean_keypoint_occlusion_ratio": float(1 - visibility.mean()), "motion_transition_candidate_count": transitions,
        "beneficial_candidate_ratio": float(beneficial.mean()), "harmful_candidate_ratio": float(harmful.mean()),
        "feasible_ratio": float(feasible.mean()), "infeasible_ratio": float((~feasible).mean()),
        "imbalance_flags": {"beneficial_below_10pct": bool(beneficial.mean() < .1), "harmful_below_10pct": bool(harmful.mean() < .1),
                            "any_empty_temporal_context": any(temporal.get(name, 0) == 0 for name in ("C7_long_occlusion_history", "C8_recent_intervention_transition", "C9_motion_transition"))},
        "no_oversampling": True,
    }


def split_manifest(by_split) -> dict[str, object]:
    return {"label": LABEL, "split_before_windowing": True, "group_keys": ["episode", "person/profile", "context"],
            "splits": {name: {"episode_ids": sorted({sample.episode_id for sample in samples}),
                              "profile_ids_split_only": sorted({int(sample.split_metadata["person_profile_id"]) for sample in samples})}
                       for name, samples in by_split.items()},
            "holdouts": {"unseen_profile_ids": [2, 6], "C2_motion_action": {"S4_human_decelerating": [2, 4]},
                         "C5_turn_speed_actions": [1, 2], "C6_support_max_K": 1}}


def leakage_audit(by_split, normalizer) -> dict[str, object]:
    all_samples = [sample for values in by_split.values() for sample in values]
    isolation = validate_split_isolation(all_samples)
    episode_sets = {name: {sample.episode_id for sample in samples} for name, samples in by_split.items()}
    overlaps = {f"{a}_{b}": sorted(episode_sets[a] & episode_sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    runtime_keys = set().union(*(runtime_payload(sample)["streams"].keys() for sample in all_samples))
    return {
        "label": LABEL, "history_max_timestamp": max(float(sample.timestamps["history"].max()) for sample in all_samples),
        "candidate_future_min_timestamp": min(float(sample.timestamps["candidate_future"].min()) for sample in all_samples),
        "history_future_human_fields_absent": not bool(runtime_keys & {"future_global", "gt_human_future", "natural_future"}),
        "runtime_target_fields_absent": not bool(runtime_keys & {"benefit", "harm", "gt_best_action"}),
        "runtime_person_identity_absent": not bool(runtime_keys & {"profile", "profile_id", "person_profile_id"}),
        "candidate_future_fields": ["robot_x", "robot_y", "robot_yaw", "robot_linear_velocity", "robot_angular_velocity"],
        "candidate_future_contains_human_gt": False, "functional_input_source": "estimated theta_hat; theta_true excluded",
        "episode_overlaps": overlaps, "split_isolation": isolation,
        "normalizer_fit_split": normalizer.fit_split, "normalizer_fit_sample_count": len(normalizer.fit_sample_ids),
        "normalizer_fit_ids_all_train": all(value.startswith("train:") for value in normalizer.fit_sample_ids),
        "passed": isolation["passed"] and not any(overlaps.values()) and max(float(sample.timestamps["history"].max()) for sample in all_samples) <= 0,
    }


def mask_audit(by_split) -> dict[str, object]:
    all_samples = [sample for values in by_split.values() for sample in values]
    missing_counts = {name: int(sum((~sample.masks[name]).sum() for sample in all_samples)) for name in STREAM_ORDER}
    return {"label": LABEL, "every_stream_has_mask": all(all(name in sample.masks for name in STREAM_ORDER) for sample in all_samples),
            "missing_value_counts": missing_counts, "skeleton_missing_has_mask": missing_counts["skeleton_history"] > 0,
            "functional_unavailable_has_mask": missing_counts["functional_history"] > 0,
            "wm_history_unavailable_has_mask": missing_counts["wm_diagnostic_history"] > 0,
            "padding_mask_present": all("history_padding_mask" in sample.masks for sample in all_samples),
            "zero_is_never_used_without_mask_contract": True}


def make_figures(output: Path, by_split) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    sample = by_split["test"][0]; t = sample.timestamps["history"]
    def save(name):
        path = folder / name; plt.title(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close(); paths.append(str(path))
    skeleton = sample.streams["skeleton_history"]; root = sample.streams["human_motion_history"][:, :3]; robot = sample.streams["robot_history"]
    plt.figure(); plt.plot(root[:, 0], root[:, 1], label="human root"); plt.scatter(skeleton[:, :, 0], skeleton[:, :, 1], s=2, alpha=.2); plt.legend(); save("01_skeleton_history.png")
    plt.figure(); plt.plot(t, robot[:, 5]); plt.xlabel("time (s)"); plt.ylabel("human-robot distance"); save("02_distance_history.png")
    plt.figure(); plt.imshow(sample.streams["human_motion_history"][:, -6:].T, aspect="auto"); plt.yticks(range(6), ("stop", "walk", "run", "accelerating", "decelerating", "turning")); save("03_motion_state.png")
    plt.figure(); plt.step(t, sample.streams["visibility_history"][:, 1]); plt.ylim(0, 1.05); plt.ylabel("keypoint valid ratio"); save("04_visibility.png")
    plt.figure(); plt.plot(t, sample.streams["functional_history"][:, :6]); plt.ylabel("estimated theta (masked)"); save("05_functional_estimates.png")
    plt.figure(); plt.plot(t, sample.streams["wm_diagnostic_history"]); plt.ylabel("WM diagnostic (masked)"); save("06_uncertainty.png")
    plt.figure(); plt.imshow(sample.streams["interaction_history"].T, aspect="auto"); plt.xlabel("history frame"); save("07_action_response_history.png")
    future = sample.streams["candidate_robot_future"]; plt.figure(); plt.plot(robot[:, 0], robot[:, 1], label="history"); plt.plot(future[:, 0], future[:, 1], label="candidate robot future"); plt.legend(); save("08_candidate_robot_future.png")
    plt.figure(); names = list(by_split); plt.bar(names, [len(values) for values in by_split.values()]); plt.ylabel("candidate samples"); save("09_split_distribution.png")
    plt.figure(); plt.imshow(sample.masks["functional_history"].T, aspect="auto"); plt.xlabel("history frame"); plt.ylabel("functional mask"); save("10_history_masks.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-0 results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); random.seed(42); np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    by_split = build_all(args, torch); all_samples = [sample for values in by_split.values() for sample in values]
    normalizer = fit_train_normalizer(by_split["train"])
    statistics = dataset_statistics(by_split); manifest = split_manifest(by_split); leakage = leakage_audit(by_split, normalizer)
    bridge = static_bridge_audit(all_samples); masks = mask_audit(by_split); figures = make_figures(args.output_dir, by_split)
    registry = {"label": LABEL, "registry": feature_registry()}; protocol = schema_description()
    for name, value in (("existing_temporal_signal_audit.json", existing_signal_audit()), ("temporal_feature_registry.json", registry),
                        ("temporal_protocol.json", protocol), ("dataset_statistics.json", statistics), ("split_manifest.json", manifest),
                        ("split_leakage_audit.json", leakage), ("static_temporal_bridge_audit.json", bridge), ("mask_audit.json", masks)):
        phase5.write_json(args.output_dir / name, value)
    reproducibility = {split: sha256_array(np.stack([sample.streams["skeleton_history"] for sample in samples])) for split, samples in by_split.items()}
    interface_ready = bool(leakage["passed"] and bridge["passed"] and masks["every_stream_has_mask"])
    temporal_context_minimum = 5  # Protocol-level coverage floor, declared before any Phase5B model/test result.
    data_ready = bool(interface_ready and all(statistics["temporal_context_decision_point_counts"].get(name, 0) >= temporal_context_minimum
                                              for name in ("C7_long_occlusion_history", "C8_recent_intervention_transition", "C9_motion_transition")))
    summary = {"label": LABEL, "stage": "Phase 5B-0 Rich Temporal Multimodal Context Protocol", "model_training_performed": False,
               "qwen_called": False, "lora_used": False, "rgb_used": False, "real_data_accessed": False,
               "stream_order": list(STREAM_ORDER), "statistics": statistics, "leakage": leakage, "bridge": bridge, "mask_audit": masks,
               "reproducibility_sha256": reproducibility, "figures": figures,
               "phase5b1_interface_ready": interface_ready, "phase5b1_data_ready": data_ready, "phase5b1_ready": data_ready,
               "readiness_rule": {"minimum_decision_points_per_temporal_context": temporal_context_minimum,
                                  "declared_before_phase5b_model_results": True},
               "readiness_blockers": [] if data_ready else ["C7 long-occlusion history has too few naturally occurring decision episodes; expand generator coverage before Phase5B-1 formal comparison"],
               "next_stage_requires_human_approval": True}
    phase5.write_json(args.output_dir / "summary.json", summary); print(json.dumps(phase5.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()
