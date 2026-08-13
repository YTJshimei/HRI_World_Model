"""Phase 5B-0.5: independent long-occlusion episodes and manifest-v1 freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_phase5b_temporal_dataset as b0
from scripts import run_phase5a_context_value as phase5a
from scripts import run_phase5a_frozen3b as phase5
from src.multimodal.temporal_dataset import (
    apply_continuous_occlusion, build_temporal_samples, fit_train_normalizer,
    is_c7_long_occlusion, longest_joint_occlusion_run, static_bridge_audit,
)
from src.multimodal.temporal_schema import DT_SECONDS, FUTURE_FRAMES, HISTORY_FRAMES, HISTORY_WINDOWS, LABEL

C7_MIN_FRAMES = 5
C7_MIN_SECONDS = C7_MIN_FRAMES * DT_SECONDS
C7_MIN_EPISODES_PER_SPLIT = 5
EXTENSION_SEED = 50_042

OCCLUSION_VARIANTS = (
    {"name": "left_leg", "joints": (11, 13, 15), "start": 2, "frames": 5},
    {"name": "right_leg", "joints": (12, 14, 16), "start": 5, "frames": 7},
    {"name": "left_arm", "joints": (5, 7, 9), "start": 8, "frames": 6},
    {"name": "right_arm", "joints": (6, 8, 10), "start": 10, "frames": 8},
    {"name": "lower_body", "joints": (11, 12, 13, 14, 15, 16), "start": 12, "frames": 5},
    {"name": "upper_body", "joints": (5, 6, 7, 8, 9, 10), "start": 11, "frames": 9},
)

# Existing synthetic scenarios/motions only; no model-targeted test-only type.
SPLIT_SPECS = {
    "train": (("S1_too_close", "walk"), ("S2_too_far", "run"), ("S4_human_decelerating", "deceleration"),
              ("S5_human_turning", "left_turn"), ("S6_high_distance_sensitive", "acceleration"), ("S8_high_turn_sensitive", "right_turn")),
    "validation": (("S2_too_far", "walk"), ("S1_too_close", "fast_walk"), ("S4_human_decelerating", "deceleration"),
                   ("S5_human_turning", "left_turn"), ("S6_high_distance_sensitive", "run"), ("S8_high_turn_sensitive", "right_turn")),
    "test": (("S3_human_accelerating", "acceleration"), ("S9_uncertain_new_person", "walk"),
             ("S4_human_decelerating", "deceleration"), ("S5_human_turning", "left_turn"),
             ("S6_high_distance_sensitive", "run"), ("S8_high_turn_sensitive", "right_turn")),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def c7_definition() -> dict[str, object]:
    return {"label": LABEL, "source": "frozen Phase5B-0 src.multimodal.temporal_dataset._temporal_tags",
            "minimum_consecutive_occlusion_frames": C7_MIN_FRAMES, "minimum_duration_seconds": C7_MIN_SECONDS,
            "visibility_threshold": "visibility_mask[joint] == False",
            "keypoint_valid_ratio_threshold": 16 / 17,
            "criterion": "at least one real COCO-17 joint is missing for >=5 consecutive history frames",
            "partial_visibility_allowed": True, "confidence_requirement": "occluded joint confidence == 0",
            "decision_time_relation": "occlusion interval lies fully in [t0-1.9s,t0]; onset varies and interval may end before or include t0",
            "scene_flag_is_neither_required_nor_sufficient": True, "definition_changed_from_phase5b0": False}


def generate_one_sample(scenario: str, motion: str, seed: int, unique_sample: int, variant: dict, split_name: str):
    import scripts.run_phase4c_decision as c0
    from src.data.synthetic_interaction import generate_interaction_split
    spec = next(item for item in c0.SCENARIOS if item[0] == scenario)
    _, profile, distance, target, _, support_k = spec
    generated = generate_interaction_split(160, seed, f"phase5b05_{split_name}_{unique_sample}", profile_ids=(profile,), noise_std=.005, occlusion_rate=.10)
    matches = np.flatnonzero(generated.action_type == motion); source = int(matches[0]) if len(matches) else 0
    visibility, confidence = apply_continuous_occlusion(generated.visibility_mask[source], generated.confidence[source],
                                                         start=variant["start"], frames=variant["frames"], joints=variant["joints"])
    return {"scenario": scenario, "scenario_index": next(i for i, item in enumerate(c0.SCENARIOS) if item[0] == scenario),
            "sample": unique_sample, "profile": profile, "support_k": support_k, "target_distance": target,
            "history": generated.human_history[source], "natural": generated.natural_future[source],
            "robot": c0.set_robot_distance(generated.human_history[source], generated.robot_history[source], distance),
            "confidence": confidence, "visibility": visibility, "action_type": str(generated.action_type[source]),
            "c7_generation": {**variant, "joints": list(variant["joints"]), "generator_seed": seed, "requested_motion": motion}}


def records_from_samples(args, engine, samples, prior_mean, prior_std):
    import scripts.run_phase4c_decision as c0
    from src.data.functional_response_state import functional_state_from_profile
    from src.data.robot_action_schema import action_feature
    from src.data.synthetic_interaction import PROFILE_BY_ID
    from src.decision.safety_calibration import safety_features
    from src.decision.safety_targets import build_safety_targets_for_training_or_evaluation
    records = []
    for episode, sample in enumerate(samples):
        theta_true = functional_state_from_profile(PROFILE_BY_ID[int(sample["profile"])]).astype(np.float32)
        theta_hat, theta_std, support = c0.estimate_personal_belief(sample, theta_true, prior_mean, prior_std, EXTENSION_SEED + episode * 31)
        state = c0.make_state(sample, theta_hat, theta_std); predicted = engine.rollout(state, uncertainty_aware=True)
        gt = c0.ground_truth_rollout(sample, state, theta_true)
        targets = build_safety_targets_for_training_or_evaluation(gt.predicted_human_robot_distance, state.too_close_distance)
        for action_index, action_id in enumerate(predicted.action_ids):
            records.append({"split": sample["c7_split"], "scenario": sample["scenario"], "sample": sample["sample"], "profile": sample["profile"],
                            "action": int(action_id), "state": state, "sample_data": sample, "theta_true": theta_true,
                            "theta_hat": theta_hat, "theta_std": theta_std, "support": support, "predicted_rollout": predicted,
                            "gt_rollout": gt, "action_index": action_index,
                            "features": safety_features(sample["history"], sample["robot"], action_feature(int(action_id)),
                                                        predicted.predicted_human_robot_distance[action_index], predicted.predicted_action_effect[action_index], theta_hat, theta_std),
                            "predicted_distance": predicted.predicted_human_robot_distance[action_index], "gt_distance": targets.distance_trajectory[action_index],
                            "gt_minimum": targets.minimum_distance[action_index], "gt_unsafe": targets.violation_any[action_index],
                            "gt_violation_duration": targets.violation_duration[action_index], "gt_time_to_minimum": targets.time_to_minimum_distance[action_index]})
    return records


def build_extension(args, development, split_name, torch):
    import scripts.run_phase4c2_belief_selection as c2
    import scripts.run_phase4c3_selective_personalization as c3
    samples = []
    offset = {"train": 9000, "validation": 9100, "test": 9200}[split_name]
    for index, ((scenario, motion), variant) in enumerate(zip(SPLIT_SPECS[split_name], OCCLUSION_VARIANTS)):
        sample = generate_one_sample(scenario, motion, EXTENSION_SEED + {"train": 0, "validation": 1000, "test": 2000}[split_name] + index * 97,
                                     offset + index, variant, split_name)
        sample["c7_split"] = split_name; samples.append(sample)
    records = records_from_samples(args, development["engine"], samples, development["prior_mean"], development["prior_std"])
    artifacts, predictions, _ = c3.build_base(args, records, development["engine"], development["prior_mean"], development["prior_std"],
                                               development["root"], development["scale"], development["safety"], development["calibration"], development["cost"], torch)
    episodes = c3.episode_data(args, records, artifacts, predictions, development["cost"], development["engine"], development["prior_mean"], development["prior_std"], development["selector"])
    _, static, targets, meta = phase5a.build_tokens(episodes, "test" if split_name == "test" else split_name, development["prior_mean"])
    if split_name != "test":
        keep = [phase5a.development_candidate_allowed(row) for row in meta]
        static = [value for value, allowed in zip(static, keep) if allowed]; targets = [value for value, allowed in zip(targets, keep) if allowed]
        meta = [value for value, allowed in zip(meta, keep) if allowed]
    temporal = build_temporal_samples(episodes, static, targets, meta, split_name)
    if len({sample.episode_id for sample in temporal if "C7_long_occlusion_history" in sample.temporal_tags}) != len(samples):
        raise RuntimeError(f"not every independent {split_name} extension episode satisfies C7")
    return temporal, samples


def episode_labels(sample) -> set[str]:
    labels = set(sample.temporal_tags)
    if sample.context_split.startswith(("C4_", "C5_", "C6_")): labels.add(sample.context_split)
    return labels


def overlap_rows(samples):
    names = ("C4", "C5", "C6", "C7", "C8", "C9"); episodes = {}
    for sample in samples: episodes.setdefault(sample.episode_id, {label[:2] for label in episode_labels(sample)})
    return [{"synthetic_interaction": LABEL, "context_a": a, "context_b": b,
             "population_scope": "full manifest; C4-C6 are test holdouts, C7-C9 are temporal tags in all splits",
             "overlap_decision_episodes": sum(a in labels and b in labels for labels in episodes.values())} for a in names for b in names]


def context_counts(samples):
    names = ("C4", "C5", "C6", "C7", "C8", "C9")
    result = {}
    for scope, scoped in [("all", samples)] + [
        (split, [sample for sample in samples if sample.split == split])
        for split in ("train", "validation", "test")
    ]:
        result[scope] = {}
        for name in names:
            matched = [sample for sample in scoped if name in {label[:2] for label in episode_labels(sample)}]
            result[scope][name] = {
                "decision_episodes": len({sample.episode_id for sample in matched}),
                "candidate_samples": len(matched),
            }
    return result


def distributions(samples):
    episode_samples = {sample.episode_id: sample for sample in samples}; episodes = list(episode_samples.values())
    return {"episodes": len(episodes), "motion": dict(Counter(str(s.split_metadata["motion_type_evaluation_only"]) for s in episodes)),
            "profile": dict(Counter(str(s.split_metadata["person_profile_id"]) for s in episodes)),
            "scene": dict(Counter(str(s.split_metadata["scenario"]) for s in episodes)),
            "beneficial_rate": float(np.mean([s.targets.benefit > 1e-6 for s in samples])), "harmful_rate": float(np.mean([s.targets.harm for s in samples])),
            "feasible_rate": float(np.mean([s.targets.feasible for s in samples])),
            "occlusion_rate": float(np.mean([1 - s.streams["visibility_history"][:, 1].mean() for s in episodes]))}


def manifest(by_split) -> dict[str, object]:
    rows = []
    for split, samples in by_split.items():
        grouped = {}
        for sample in samples: grouped.setdefault(sample.episode_id, []).append(sample)
        for episode_id, branches in sorted(grouped.items()):
            first = branches[0]
            rows.append({"episode_id": episode_id, "split": split, "profile_id_split_only": int(first.split_metadata["person_profile_id"]),
                         "scenario": first.split_metadata["scenario"], "motion_type_evaluation_only": first.split_metadata["motion_type_evaluation_only"],
                         "context_labels": sorted({label for branch in branches for label in episode_labels(branch)}),
                         "candidate_ids": sorted(branch.sample_id for branch in branches)})
    return {"label": LABEL, "version": "phase5b_manifest_v1", "immutable_after_freeze": True,
            "generator": {"base_seed": 42, "c7_extension_seed": EXTENSION_SEED, "c7_variants": [{**v, "joints": list(v["joints"])} for v in OCCLUSION_VARIANTS]},
            "protocol": {"dt_seconds": DT_SECONDS, "history_frames": HISTORY_FRAMES, "candidate_future_frames": FUTURE_FRAMES,
                         "history_windows": HISTORY_WINDOWS, "split_before_candidate_branching": True}, "episodes": rows}


def checksum(value) -> str:
    return hashlib.sha256(json.dumps(phase5.clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_c7_coverage(counts: dict[str, int]) -> None:
    """Refuse to freeze a manifest unless every split passes the episode gate."""
    missing = {split: int(counts.get(split, 0)) for split in ("train", "validation", "test")
               if int(counts.get(split, 0)) < C7_MIN_EPISODES_PER_SPLIT}
    if missing:
        raise RuntimeError(
            f"C7 coverage gate failed (need >= {C7_MIN_EPISODES_PER_SPLIT} independent decision episodes per split): {missing}; "
            "manifest was not frozen"
        )


def c7_audit_rows(by_split):
    rows = []
    for split, samples in by_split.items():
        seen = set()
        for sample in samples:
            if sample.episode_id in seen or "C7_long_occlusion_history" not in sample.temporal_tags: continue
            seen.add(sample.episode_id); visibility = sample.streams["visibility_history"][:, 1]; skeleton_valid = sample.masks["skeleton_history"].any(-1)
            run = longest_joint_occlusion_run(skeleton_valid)
            rows.append({"synthetic_interaction": LABEL, "split": split, "episode_id": sample.episode_id,
                         "longest_run_frames": run["frames"], "longest_run_seconds": run["frames"] * DT_SECONDS,
                         "joint": run["joint"], "start_frame": run["start"], "end_frame": run["end"],
                         "minimum_keypoint_valid_ratio": float(visibility.min()), "mask_verified": is_c7_long_occlusion(skeleton_valid),
                         "confidence_zero_on_missing": bool(np.all(np.asarray(sample.split_metadata["keypoint_confidence_audit"])[~skeleton_valid] == 0))})
    return rows


def make_c7_figures(output, by_split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    choices = []
    for split in ("train", "validation", "test"):
        for sample in by_split[split]:
            if "C7_long_occlusion_history" in sample.temporal_tags and sample.episode_id not in {item.episode_id for item in choices}: choices.append(sample)
    for index, sample in enumerate(choices[:5], 1):
        t = sample.timestamps["history"]; valid = sample.masks["skeleton_history"].any(-1); run = longest_joint_occlusion_run(valid)
        fig, axes = plt.subplots(3, 2, figsize=(10, 8)); root = sample.streams["human_motion_history"][:, :3]
        axes[0,0].imshow(valid.T, aspect="auto"); axes[0,0].set_title("skeleton joint validity")
        axes[0,1].plot(t, sample.streams["visibility_history"][:,1]); axes[0,1].axvspan(t[run["start"]], t[run["end"]], alpha=.25); axes[0,1].set_title("visibility / longest run")
        axes[1,0].plot(root[:,0], root[:,1]); axes[1,0].set_title("human root")
        axes[1,1].plot(t, sample.streams["robot_history"][:,5]); axes[1,1].set_title("robot-human distance")
        axes[2,0].imshow(sample.streams["human_motion_history"][:,-6:].T, aspect="auto"); axes[2,0].set_title("motion state")
        axes[2,1].imshow(sample.streams["interaction_history"].T, aspect="auto"); axes[2,1].axvline(HISTORY_FRAMES-1, color="r"); axes[2,1].set_title("actions / response; red=t0")
        fig.suptitle(LABEL, fontsize=8); fig.tight_layout(); path = folder / f"c7_episode_{index}.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-0.5: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); random.seed(42); np.random.seed(42)
    import torch
    torch.manual_seed(42); torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    development = phase5.build_development_data(args, torch)
    old_train = build_temporal_samples(development["train_episodes"], development["train_samples"], development["train_targets"], development["train_meta"], "train")
    old_val = build_temporal_samples(development["val_episodes"], development["val_samples"], development["val_targets"], development["val_meta"], "validation")
    test_episodes, _, test_static, test_targets, test_meta = phase5.materialize_test(args, development, torch)
    old_test = build_temporal_samples(test_episodes, test_static, test_targets, test_meta, "test")
    old = {"train": old_train, "validation": old_val, "test": old_test}; extensions = {}; generation = {}
    for split in old: extensions[split], generation[split] = build_extension(args, development, split, torch)
    new = {split: old[split] + extensions[split] for split in old}; all_old = sum(old.values(), []); all_new = sum(new.values(), [])
    normalizer = fit_train_normalizer(new["train"]); leakage = b0.leakage_audit(new, normalizer)
    statistics = b0.dataset_statistics(new); c7_rows = c7_audit_rows(new)
    c7_counts = {split: len({row["episode_id"] for row in c7_rows if row["split"] == split}) for split in new}
    # This is intentionally before manifest construction/writing. Coverage must
    # never be manufactured by candidate duplication or accepted post-hoc.
    validate_c7_coverage(c7_counts)
    bridge = static_bridge_audit(all_new); frozen_manifest = manifest(new); canonical_manifest_sha = checksum(frozen_manifest)
    split_info = b0.split_manifest(new); figures = make_c7_figures(args.output_dir, new)
    comparison = {"label": LABEL, "old": distributions(all_old), "new": distributions(all_new),
                  "context_counts_full_manifest": context_counts(all_new),
                  "new_by_split": {split: b0.dataset_statistics({split: samples, **{name: [] for name in new if name != split}})["by_split"][split] for split, samples in new.items()},
                  "added": {split: {"episodes": len({s.episode_id for s in extensions[split]}), "c7_episodes": len({s.episode_id for s in extensions[split] if "C7_long_occlusion_history" in s.temporal_tags}), "candidates": len(extensions[split])} for split in new},
                  "non_c7_original_episode_count": len({s.episode_id for s in all_old if "C7_long_occlusion_history" not in s.temporal_tags}),
                  "non_c7_original_episodes_preserved": True}
    class_balance = {"label": LABEL, "beneficial_candidate_ratio": distributions(all_new)["beneficial_rate"],
                     "harmful_candidate_ratio": distributions(all_new)["harmful_rate"], "feasible_candidate_ratio": distributions(all_new)["feasible_rate"],
                     "oversampling": False, "undersampling": False, "class_weighting": False}
    extension_audit = {"label": LABEL, "independent_generator_seeds": sorted(item["c7_generation"]["generator_seed"] for values in generation.values() for item in values),
                       "new_episode_ids_unique": len({s.episode_id for values in extensions.values() for s in values}) == sum(len({s.episode_id for s in values}) for values in extensions.values()),
                       "candidate_branches_are_not_episodes": True, "copied_original_episode_count": 0,
                       "variants": [{**v, "joints": list(v["joints"])} for v in OCCLUSION_VARIANTS], "all_c7_masks_verified": all(row["mask_verified"] for row in c7_rows)}
    checks = {"train_c7_at_least_5": c7_counts["train"] >= 5, "validation_c7_at_least_5": c7_counts["validation"] >= 5,
              "test_c7_at_least_5": c7_counts["test"] >= 5, "leakage_passed": leakage["passed"],
              "real_masks_verified": all(row["mask_verified"] and row["confidence_zero_on_missing"] for row in c7_rows), "manifest_frozen": len(canonical_manifest_sha) == 64,
              "static_bridge_available": bridge["passed"], "future_leakage_absent": leakage["history_future_human_fields_absent"] and leakage["candidate_future_contains_human_gt"] is False}
    ready = all(checks.values())
    outputs = {"c7_definition.json": c7_definition(), "generator_extension_audit.json": extension_audit,
               "dataset_statistics_old_vs_new.json": comparison, "split_manifest.json": split_info,
               "phase5b_manifest_v1.json": frozen_manifest,
               "split_leakage_audit.json": leakage, "static_temporal_bridge_v1.json": bridge, "class_balance.json": class_balance}
    for name, value in outputs.items(): phase5.write_json(args.output_dir / name, value)
    manifest_file_sha = hashlib.sha256((args.output_dir / "phase5b_manifest_v1.json").read_bytes()).hexdigest()
    phase5.write_json(args.output_dir / "manifest_checksum.json", {"label": LABEL, "algorithm": "SHA256",
                      "scope": "exact phase5b_manifest_v1.json file bytes", "sha256": manifest_file_sha,
                      "canonical_content_sha256": canonical_manifest_sha})
    phase5.write_csv(args.output_dir / "c7_episode_audit.csv", c7_rows); phase5.write_csv(args.output_dir / "context_overlap_matrix.csv", overlap_rows(all_new))
    summary = {"label": LABEL, "stage": "Phase 5B-0.5 Long-Occlusion Coverage Expansion + Manifest Freeze",
               "model_training_performed": False, "old_results_overwritten": False, "new_statistics": statistics,
               "context_counts_full_manifest": context_counts(all_new),
               "c7_decision_episode_counts": c7_counts, "readiness_checks": checks, "ready_for_phase5b1": ready,
               "manifest_v1_sha256": manifest_file_sha, "manifest_v1_canonical_content_sha256": canonical_manifest_sha,
               "figures": figures, "phase5b1_started": False, "next_step_requires_human_approval": True}
    phase5.write_json(args.output_dir / "summary.json", summary); print(json.dumps(phase5.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()
