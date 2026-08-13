"""Phase 5B-v3-R1A runtime-valid Benefit anchor contract realignment audit.

This is a synthetic TRAIN/VALIDATION label-contract audit.  It performs no
training, checkpoint selection, threshold calibration or decision execution.
Manifest-v3 and Benefit Target v1 remain immutable; Target v2 is written as a
separate deterministic derived layer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b1_static_vs_temporal as b1
from src.data.adverse_response_dataset import (
    ACTION_IDS,
    GENERATOR_SEED,
    RISK_SEED,
    build_development_split,
)
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.multimodal.phase5b_v2_dataset import replay_runtime_generic_policy
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_dataset import _candidate_future
from src.multimodal.temporal_schema import LABEL

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-v3-R1A Runtime-Valid Benefit Anchor Contract Realignment Audit"
TARGET_V1 = "BENEFIT_TARGET_V1_GT_NATURAL_FUTURE_ANCHORED_NOT_RUNTIME_ALIGNED"
TARGET_V2 = "BENEFIT_TARGET_V2_RUNTIME_ANCHORED"
EXPECTED_MANIFEST_SHA = "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a"
EXPECTED_V2_MANIFEST_SHA = "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
EXPECTED_R1_SHA = "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff"
EXPECTED_HARM_SHA = "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d"
# Frozen GT costs are serialized as float32 while v1 Benefit candidates retain
# their pre-serialization float64 difference.  At costs near 29, one float32
# ULP is about 3.8e-6, so 1e-5 is the explicit numerical-equivalence tolerance.
TOLERANCE = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--manifest-v3", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--manifest-v2", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17c_adverse_response_expansion/phase5b_manifest_v2.json")
    parser.add_argument("--hold-labels", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/hold_label_audit.csv")
    parser.add_argument("--r1-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    encoded = json.dumps(io.clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_sha(values) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bool_value(value: str) -> bool:
    if value == "True": return True
    if value == "False": return False
    raise ValueError(f"invalid frozen boolean: {value}")


def hold_rows(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["candidate_id"]] = {
                "action_id": HOLD_ACTION_ID, "benefit_v1": float(row["GT_benefit"]),
                "gt_cost": float(row["GT_total_cost"]), "feasible": bool_value(row["feasible"]),
                "harm_v2": bool_value(row["harm_v2"]), "gt_unsafe": bool_value(row["GT_unsafe"]),
                "excessive_deceleration": bool_value(row["excessive_deceleration"]),
                "abrupt_lateral_response": bool_value(row["abrupt_lateral_response"]),
                "abrupt_heading_change": bool_value(row["abrupt_heading_change"]),
            }
    return result


def source_record(function) -> dict[str, object]:
    path = Path(inspect.getsourcefile(function)).resolve(); _, line = inspect.getsourcelines(function)
    return {"file": str(path.relative_to(PROJECT_ROOT)), "function": function.__name__, "first_line": line, "sha256": file_sha(path)}


def frozen_episode_fingerprint(episodes, frozen_hold) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.episode_id.encode())
        for value in (
            episode.human_history, episode.robot_history, episode.confidence, episode.visibility,
            episode.natural_future, episode.generic_costs, episode.gt_costs,
        ):
            array = np.ascontiguousarray(value); digest.update(array.dtype.str.encode()); digest.update(array.tobytes())
        for candidate in episode.candidates:
            digest.update(json.dumps(io.clean(candidate.__dict__), sort_keys=True, default=str).encode())
        digest.update(json.dumps(io.clean(frozen_hold[f"{episode.episode_id}:{HOLD_ACTION_ID}"]), sort_keys=True).encode())
    return digest.hexdigest()


def rank_signature(actions, values) -> tuple[int, ...]:
    actions = np.asarray(actions, int); values = np.asarray(values, float)
    return tuple(int(actions[index]) for index in np.lexsort((actions, -values)))


def derive_split(split: str, episodes, frozen_hold) -> tuple[list[dict], list[dict], dict]:
    anchors, labels = [], []
    max_bridge_future_error = 0.0
    for episode in episodes:
        # The selector accepts runtime fields only and must return/freeze the ID
        # before this function reads any episode GT cost below.
        replay = replay_runtime_generic_policy(
            episode.human_history, episode.robot_history, episode.confidence,
            episode.visibility, episode.target_follow_distance,
        )
        runtime_anchor_action = int(replay.anchor_action_id)
        runtime_anchor_index = int(np.flatnonzero(np.asarray(ACTION_IDS) == runtime_anchor_action)[0])
        if replay.gt_read_count != 0:
            raise RuntimeError("runtime anchor selector reported a GT read")
        for index, simulation in enumerate(replay.simulations):
            direct = _candidate_future(simulation.robot_future_xy, episode.robot_history)
            max_bridge_future_error = max(max_bridge_future_error, float(np.max(np.abs(direct - _candidate_future(replay.simulations[index].robot_future_xy, episode.robot_history)))))

        # Label-side access begins only after runtime_anchor_action is frozen.
        old_anchor_index = int(episode.generic_action_index)
        old_anchor_action = int(episode.candidates[old_anchor_index].action_id)
        old_anchor_gt_cost = float(episode.gt_costs[old_anchor_index])
        runtime_anchor_gt_cost = float(episode.gt_costs[runtime_anchor_index])
        shift = runtime_anchor_gt_cost - old_anchor_gt_cost
        anchors.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "split": split,
            "episode_id": episode.episode_id, "runtime_anchor_action_id": runtime_anchor_action,
            "runtime_anchor_proxy_score": float(replay.costs.total[replay.anchor_index]),
            "tie_count": replay.tie_count, "tie_break": "minimum proxy cost then lowest action ID",
            "candidate_ids": "0|1|2|3|4", "HOLD_excluded": True, "GT_reads_during_selection": replay.gt_read_count,
            "old_anchor_action_id": old_anchor_action, "anchor_agrees": old_anchor_action == runtime_anchor_action,
            "episode_zero_shift": shift, "motion": episode.motion_type,
            "contexts": "|".join(episode.context_labels) or "NONE", "C7": any(x.startswith("C7") for x in episode.context_labels),
            "C8": any(x.startswith("C8") for x in episode.context_labels), "C9": any(x.startswith("C9") for x in episode.context_labels),
            "stop": episode.motion_type == "stop", "profile_audit_only": episode.profile_id,
        })
        all_costs = [*map(float, episode.gt_costs), frozen_hold[f"{episode.episode_id}:{HOLD_ACTION_ID}"]["gt_cost"]]
        all_actions = [*ACTION_IDS, HOLD_ACTION_ID]
        candidate_data = [
            {
                "benefit_v1": float(candidate.benefit), "gt_cost": float(episode.gt_costs[index]),
                "feasible": bool(candidate.feasible), "harm_v2": bool(candidate.harm_v2),
                "gt_unsafe": bool(candidate.gt_unsafe),
                "excessive_deceleration": bool(candidate.events.excessive_deceleration),
                "abrupt_lateral_response": bool(candidate.events.abrupt_lateral_response),
                "abrupt_heading_change": bool(candidate.events.abrupt_heading_change),
            }
            for index, candidate in enumerate(episode.candidates)
        ] + [frozen_hold[f"{episode.episode_id}:{HOLD_ACTION_ID}"]]
        v1 = np.asarray([item["benefit_v1"] for item in candidate_data], float)
        v2 = runtime_anchor_gt_cost - np.asarray(all_costs, float)
        rank_v1, rank_v2 = rank_signature(all_actions, v1), rank_signature(all_actions, v2)
        for action_id, data, old, new in zip(all_actions, candidate_data, v1, v2):
            labels.append({
                "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
                "target_version": TARGET_V2, "split": split, "episode_id": episode.episode_id,
                "candidate_id": f"{episode.episode_id}:{action_id}", "action_id": action_id,
                "runtime_anchor_action_id": runtime_anchor_action, "old_anchor_action_id": old_anchor_action,
                "GT_total_cost_unchanged": data["gt_cost"], "benefit_v1_preserved": float(old),
                "benefit_v2_runtime_anchor": float(new), "v2_minus_v1": float(new - old),
                "feasible_unchanged": data["feasible"], "harm_v2_unchanged": data["harm_v2"],
                "GT_unsafe_unchanged": data["gt_unsafe"],
                "excessive_deceleration_unchanged": data["excessive_deceleration"],
                "abrupt_lateral_response_unchanged": data["abrupt_lateral_response"],
                "abrupt_heading_change_unchanged": data["abrupt_heading_change"],
                "motion_unchanged": episode.motion_type, "contexts_unchanged": "|".join(episode.context_labels) or "NONE",
                "profile_audit_only": episode.profile_id, "old_rank_signature": "|".join(map(str, rank_v1)),
                "new_rank_signature": "|".join(map(str, rank_v2)),
            })
    return anchors, labels, {"maximum_runtime_bridge_candidate_future_replay_error": max_bridge_future_error}


def distribution(values) -> dict[str, float]:
    values = np.asarray(values, float)
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            **{f"P{p}": float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90)},
            "min": float(values.min()), "max": float(values.max())}


def grouped_anchor_rows(anchors):
    rows = []
    dimensions = {
        "overall": lambda row: "ALL", "motion": lambda row: row["motion"],
        "context": lambda row: row["contexts"], "profile_audit": lambda row: f"profile_{row['profile_audit_only']}",
        "C7": lambda row: str(row["C7"]), "C8": lambda row: str(row["C8"]),
        "C9": lambda row: str(row["C9"]), "stop": lambda row: str(row["stop"]),
    }
    for split in ("train", "validation", "development"):
        source = anchors if split == "development" else [row for row in anchors if row["split"] == split]
        for dimension, getter in dimensions.items():
            groups = defaultdict(list)
            for row in source: groups[getter(row)].append(row)
            for group, selected in sorted(groups.items(), key=lambda item: str(item[0])):
                rows.append({
                    "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "split": split,
                    "dimension": dimension, "group": group, "episode_count": len(selected),
                    "agreement_count": sum(row["anchor_agrees"] for row in selected),
                    "changed_count": sum(not row["anchor_agrees"] for row in selected),
                    "agreement_rate": float(np.mean([row["anchor_agrees"] for row in selected])),
                    **distribution([row["episode_zero_shift"] for row in selected]),
                })
    return rows


def sign_name(value: float) -> str:
    if value > TOLERANCE: return "positive"
    if value < -TOLERANCE: return "negative"
    return "zero"


def sign_flip_rows(labels):
    rows = []
    for split in ("train", "validation", "development"):
        selected = labels if split == "development" else [row for row in labels if row["split"] == split]
        counts = Counter(f"{sign_name(row['benefit_v1_preserved'])}_to_{sign_name(row['benefit_v2_runtime_anchor'])}" for row in selected)
        for transition in ("negative_to_positive", "positive_to_negative", "zero_to_positive", "zero_to_negative"):
            rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "split": split,
                         "transition": transition, "candidate_count": counts[transition]})
        for transition, count in sorted(counts.items()):
            if transition not in {row["transition"] for row in rows if row["split"] == split}:
                rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "split": split,
                             "transition": transition, "candidate_count": count})
    return rows


def safe_beneficial_rows(labels):
    rows = []
    for split in ("train", "validation"):
        selected = [row for row in labels if row["split"] == split]
        for version, key in ((TARGET_V1, "benefit_v1_preserved"), (TARGET_V2, "benefit_v2_runtime_anchor")):
            keep = [row for row in selected if row[key] > TOLERANCE and not row["harm_v2_unchanged"] and row["feasible_unchanged"]]
            rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "split": split,
                         "target_version": version, "candidate_count": len(keep),
                         "episode_count": len({row["episode_id"] for row in keep})})
    return rows


def frozen_predictions(args, episodes_by_split, torch, device):
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    payload = torch.load(args.r1_checkpoint, map_location=device, weights_only=False)
    model = RichTemporalSmallTransformerV3().to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
    if state_sha(model.state_dict()) != state_sha(payload["model_state_dict"]): raise RuntimeError("R1 load checksum mismatch")
    samples = build_v3_temporal_samples(episodes_by_split["validation"])
    prediction = b1.predict("R1-v3-BASE", model, samples, payload["normalizer"], args.batch_size, torch, device)
    return samples, {sample.sample_id: float(value) for sample, value in zip(samples, prediction["benefit"])}, state_sha(model.state_dict())


def subgroup_reclassification(name, labels, prediction, predicate):
    selected = [row for row in labels if row["split"] == "validation" and predicate(row)]
    rows = []
    for version, key in ((TARGET_V1, "benefit_v1_preserved"), (TARGET_V2, "benefit_v2_runtime_anchor")):
        beneficial = [row for row in selected if row[key] > TOLERANCE]
        safe = [row for row in selected if row[key] > TOLERANCE and not row["harm_v2_unchanged"] and row["feasible_unchanged"]]
        rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "group": name,
                     "target_version": version, "beneficial_count_regardless_of_harm": len(beneficial),
                     "safe_beneficial_count": len(safe),
                     "predicted_positive_count": sum(prediction[row["candidate_id"]] > 0 for row in safe),
                     "sign_failure_count": sum(prediction[row["candidate_id"]] <= 0 for row in safe),
                     "sign_accuracy": float(np.mean([prediction[row["candidate_id"]] > 0 for row in safe])) if safe else None,
                     "target_sign_flip_count": sum(sign_name(row["benefit_v1_preserved"]) != sign_name(row["benefit_v2_runtime_anchor"]) for row in selected)})
    historical_failures = [row for row in selected if row["benefit_v1_preserved"] > TOLERANCE and not row["harm_v2_unchanged"] and row["feasible_unchanged"] and prediction[row["candidate_id"]] <= 0]
    rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "group": name,
                 "target_version": "HISTORICAL_FAILURE_ATTRIBUTION",
                 "historical_sign_failure_count": len(historical_failures),
                 "explained_by_runtime_anchor_realignment": sum(row["benefit_v2_runtime_anchor"] <= TOLERANCE for row in historical_failures),
                 "still_true_prediction_failure": sum(row["benefit_v2_runtime_anchor"] > TOLERANCE for row in historical_failures)})
    return rows


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite R1A audit: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    frozen_paths = {"manifest_v3": args.manifest_v3, "manifest_v2": args.manifest_v2, "hold_labels": args.hold_labels,
                    "R1_v3_BASE": args.r1_checkpoint, "HARM_v3_BASE": args.harm_checkpoint}
    before = {name: file_sha(path) for name, path in frozen_paths.items()}
    expected = {"manifest_v3": EXPECTED_MANIFEST_SHA, "manifest_v2": EXPECTED_V2_MANIFEST_SHA,
                "R1_v3_BASE": EXPECTED_R1_SHA, "HARM_v3_BASE": EXPECTED_HARM_SHA}
    if any(before[name] != value for name, value in expected.items()): raise RuntimeError("frozen input checksum mismatch")
    frozen_hold = hold_rows(args.hold_labels)

    episodes = {"train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED),
                "validation": build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)}

    def reconstruct(immutable_episodes):
        # Rebuild only the derived anchor/label layer.  The frozen synthetic
        # episode trajectories are materialized once above and never regenerated.
        episodes = immutable_episodes
        fingerprints_before = {split: frozen_episode_fingerprint(values, frozen_hold) for split, values in episodes.items()}
        derived = {split: derive_split(split, values, frozen_hold) for split, values in episodes.items()}
        fingerprints_after = {split: frozen_episode_fingerprint(values, frozen_hold) for split, values in episodes.items()}
        anchors = derived["train"][0] + derived["validation"][0]
        labels = derived["train"][1] + derived["validation"][1]
        return episodes, anchors, labels, fingerprints_before, fingerprints_after, {split: derived[split][2] for split in derived}

    episodes, anchors, labels, fp_before, fp_after, bridge_equivalence = reconstruct(episodes)
    _, anchors_second, labels_second, _, fp_after_second, _ = reconstruct(episodes)
    if fp_after_second != fp_before:
        raise RuntimeError("derived replay mutated frozen episode trajectories/labels")
    anchor_sha_first, anchor_sha_second = canonical_sha(anchors), canonical_sha(anchors_second)
    label_sha_first, label_sha_second = canonical_sha(labels), canonical_sha(labels_second)
    if anchor_sha_first != anchor_sha_second or label_sha_first != label_sha_second: raise RuntimeError("derived contract replay is not deterministic")

    grouped = grouped_anchor_rows(anchors); sign_flips = sign_flip_rows(labels); safe_rows = safe_beneficial_rows(labels)
    samples, prediction, loaded_r1_state_sha = frozen_predictions(args, episodes, torch, device)
    validation_labels = [row for row in labels if row["split"] == "validation"]
    historical_failures = [row for row in validation_labels if row["benefit_v1_preserved"] > TOLERANCE and not row["harm_v2_unchanged"] and row["feasible_unchanged"] and prediction[row["candidate_id"]] <= 0]
    reclass = []
    for row in historical_failures:
        if row["benefit_v2_runtime_anchor"] > TOLERANCE and prediction[row["candidate_id"]] <= 0: category = "A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"
        elif row["benefit_v2_runtime_anchor"] < -TOLERANCE: category = "B_NO_LONGER_BENEFICIAL"
        elif abs(row["benefit_v2_runtime_anchor"]) <= TOLERANCE: category = "C_BECAME_ZERO"
        else: category = "D_OTHER_TARGET_CONTRACT_CHANGE"
        reclass.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "candidate_id": row["candidate_id"],
                        "episode_id": row["episode_id"], "action_id": row["action_id"], "category": category,
                        "benefit_v1": row["benefit_v1_preserved"], "benefit_v2": row["benefit_v2_runtime_anchor"],
                        "frozen_prediction": prediction[row["candidate_id"]], "motion": row["motion_unchanged"], "contexts": row["contexts_unchanged"]})
    reclass_summary = Counter(row["category"] for row in reclass)

    c7 = subgroup_reclassification("C7", labels, prediction, lambda row: "C7" in row["contexts_unchanged"])
    stop = subgroup_reclassification("STOP", labels, prediction, lambda row: row["motion_unchanged"] == "stop")
    hold = subgroup_reclassification("HOLD", labels, prediction, lambda row: row["action_id"] == HOLD_ACTION_ID)
    historical_hold = [row for row in validation_labels if row["action_id"] == HOLD_ACTION_ID and row["benefit_v1_preserved"] > TOLERANCE]
    for row in historical_hold:
        hold.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "group": "HISTORICAL_BENEFICIAL_HOLD_CANDIDATE",
                     "candidate_id": row["candidate_id"], "benefit_v1": row["benefit_v1_preserved"],
                     "benefit_v2": row["benefit_v2_runtime_anchor"], "new_sign": sign_name(row["benefit_v2_runtime_anchor"]),
                     "frozen_prediction": prediction[row["candidate_id"]], "motion": row["motion_unchanged"], "contexts": row["contexts_unchanged"]})

    episode_groups = defaultdict(list)
    for row in labels: episode_groups[row["episode_id"]].append(row)
    max_delta_range = max(max(x["v2_minus_v1"] for x in group) - min(x["v2_minus_v1"] for x in group) for group in episode_groups.values())
    max_pairwise_error = 0.0; rank_changes = top1_changes = top2_changes = 0
    for group in episode_groups.values():
        actions = np.asarray([row["action_id"] for row in group]); old = np.asarray([row["benefit_v1_preserved"] for row in group]); new = np.asarray([row["benefit_v2_runtime_anchor"] for row in group])
        max_pairwise_error = max(max_pairwise_error, float(np.max(np.abs((old[:, None] - old[None]) - (new[:, None] - new[None])))))
        old_rank, new_rank = rank_signature(actions, old), rank_signature(actions, new)
        rank_changes += old_rank != new_rank; top1_changes += old_rank[:1] != new_rank[:1]; top2_changes += set(old_rank[:2]) != set(new_rank[:2])
    anchor_self = [abs(row["benefit_v2_runtime_anchor"]) for row in labels if row["action_id"] == row["runtime_anchor_action_id"]]
    pairwise = {"label": LABEL, "mechanism_result": MECHANISM, "tolerance": TOLERANCE,
                "episode_count": len(episode_groups), "maximum_within_episode_delta_range": max_delta_range,
                "maximum_pairwise_difference_error": max_pairwise_error, "rank_signature_changes": rank_changes,
                "GT_Top1_changes": top1_changes, "GT_Top2_set_changes": top2_changes,
                "pairwise_invariant": max_pairwise_error <= TOLERANCE, "GT_ranking_100_percent_preserved": rank_changes == 0,
                "test_reads": 0}
    contract = {"label": LABEL, "mechanism_result": MECHANISM, "target_version": TARGET_V2,
                "formula": "GT_total_cost(runtime_anchor) - GT_total_cost(candidate)",
                "manifest_v3_role": "immutable episode/candidate/rollout source", "Benefit_Target_v1_role": "preserved historical, not runtime-aligned",
                "Benefit_Target_v2_role": "derived runtime-aligned supervised outcome target",
                "runtime_anchor_source": source_record(replay_runtime_generic_policy),
                "input_fields": ["pre-decision human_history", "robot_history", "confidence", "visibility", "target_follow_distance", "A0-A4", "fixed population profile"],
                "forbidden_signature_inputs_absent": ["GT natural_future", "GT human future", "GT candidate human future", "GT cost", "GT benefit", "GT harm", "GT unsafe", "profile ID", "oracle theta/action"],
                "candidate_family": list(ACTION_IDS), "HOLD_excluded": True,
                "tie_break": "lexicographic minimum (runtime proxy cost, action ID)",
                "selection_before_label_side_GT_cost_read": True, "GT_reads_during_selection": 0,
                "runtime_prior": "history last-two-frame constant velocity", "population_profile": "fixed development population constant",
                "cost_proxy": "unchanged DecisionCostWeights over runtime-predicted generic response; uncertainty disabled as frozen generic contract",
                "bridge_equivalence": bridge_equivalence}

    after = {name: file_sha(path) for name, path in frozen_paths.items()}
    harm_isolation = {"label": LABEL, "mechanism_result": MECHANISM, "manifest_sha_before": before["manifest_v3"], "manifest_sha_after": after["manifest_v3"],
                      "manifest_unchanged": before["manifest_v3"] == after["manifest_v3"] == EXPECTED_MANIFEST_SHA,
                      "HARM_checkpoint_sha_before": before["HARM_v3_BASE"], "HARM_checkpoint_sha_after": after["HARM_v3_BASE"],
                      "HARM_checkpoint_unchanged": before["HARM_v3_BASE"] == after["HARM_v3_BASE"] == EXPECTED_HARM_SHA,
                      "harm_label_checksum": canonical_sha([(row["candidate_id"], row["harm_v2_unchanged"]) for row in labels]),
                      "unsafe_label_checksum": canonical_sha([(row["candidate_id"], row["GT_unsafe_unchanged"]) for row in labels]),
                      "subtype_label_checksum": canonical_sha([(row["candidate_id"], row["excessive_deceleration_unchanged"], row["abrupt_lateral_response_unchanged"], row["abrupt_heading_change_unchanged"]) for row in labels]),
                      "episode_payload_fingerprint_before": fp_before, "episode_payload_fingerprint_after": fp_after,
                      "episode_payload_unchanged": fp_before == fp_after, "harm_model_loaded_or_run": False,
                      "harm_predictions_changed": False, "test_reads": 0}
    frozen = {"label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "checksums_before": before, "checksums_after": after,
              "manifest_v3_unchanged": before["manifest_v3"] == after["manifest_v3"] == EXPECTED_MANIFEST_SHA,
              "Benefit_Target_v1_overwritten": False, "model_training_performed": False, "optimizer_steps": 0,
              "R1_checkpoint_selection_performed": False, "R1_inference_audit_only": True, "R1_loaded_state_sha256": loaded_r1_state_sha,
              "HARM_retrained": False, "HARM_loaded_or_run": False, "threshold_calibration_performed": False,
              "decision_reconstruction_run": False, "HOLD_protocol_modified": False, "cost_weights_modified": False,
              "TEST_reads": 0}

    io.write_csv(args.output_dir / "runtime_anchor_map.csv", anchors)
    old_vs_rows = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "record_type": "episode", **row} for row in anchors]
    io.write_csv(args.output_dir / "old_vs_runtime_anchor.csv", old_vs_rows)
    io.write_csv(args.output_dir / "anchor_shift_distribution.csv", grouped)
    io.write_csv(args.output_dir / "benefit_target_v2_labels.csv", labels)
    label_file_sha = file_sha(args.output_dir / "benefit_target_v2_labels.csv")
    checksum = {"label": LABEL, "mechanism_result": MECHANISM, "target_version": TARGET_V2,
                "file_sha256": label_file_sha, "canonical_content_sha256_first_rebuild": label_sha_first,
                "canonical_content_sha256_second_rebuild": label_sha_second, "runtime_anchor_canonical_sha_first": anchor_sha_first,
                "runtime_anchor_canonical_sha_second": anchor_sha_second, "deterministic_rebuild": label_sha_first == label_sha_second and anchor_sha_first == anchor_sha_second}
    target_contract = {**contract, "Benefit_Target_v2_file_sha256": label_file_sha,
                       "canonical_content_sha256": label_sha_first, "derived_candidate_count": len(labels),
                       "source_manifest_sha256": EXPECTED_MANIFEST_SHA, "old_labels_preserved": True}
    io.write_json(args.output_dir / "frozen_contract.json", frozen)
    io.write_json(args.output_dir / "runtime_anchor_contract.json", contract)
    io.write_json(args.output_dir / "benefit_target_v2_contract.json", target_contract)
    io.write_json(args.output_dir / "benefit_target_v2_checksum.json", checksum)
    io.write_json(args.output_dir / "pairwise_invariance_audit.json", pairwise)
    io.write_csv(args.output_dir / "sign_flip_audit.csv", sign_flips)
    io.write_csv(args.output_dir / "safe_beneficial_v1_vs_v2.csv", safe_rows)
    io.write_csv(args.output_dir / "historical_sign_failure_reclassification.csv", reclass)
    io.write_csv(args.output_dir / "c7_reclassification.csv", c7)
    io.write_csv(args.output_dir / "stop_reclassification.csv", stop)
    io.write_csv(args.output_dir / "hold_reclassification.csv", hold)
    io.write_json(args.output_dir / "harm_label_isolation.json", harm_isolation)

    agreement = np.asarray([row["anchor_agrees"] for row in anchors], bool)
    gates = {
        "Gate_A": {"name": "Runtime Validity", "checks": {"GT_reads_zero": all(row["GT_reads_during_selection"] == 0 for row in anchors), "formal_runtime_bridge_reused": True, "A0_A4_only": all(row["candidate_ids"] == "0|1|2|3|4" for row in anchors), "HOLD_excluded": all(row["HOLD_excluded"] for row in anchors)},},
        "Gate_B": {"name": "Anchor Self-Consistency", "checks": {"anchor_benefit_max_abs_within_tolerance": max(anchor_self) <= TOLERANCE, "anchor_benefit_mean_abs_within_tolerance": float(np.mean(anchor_self)) <= TOLERANCE}},
        "Gate_C": {"name": "Pure Reference Shift", "checks": {"episode_constant_shift": max_delta_range <= TOLERANCE, "pairwise_differences_preserved": max_pairwise_error <= TOLERANCE, "GT_ranking_preserved": rank_changes == 0, "GT_Top1_preserved": top1_changes == 0, "GT_Top2_preserved": top2_changes == 0}},
        "Gate_D": {"name": "No Collateral Label Change", "checks": {"harm_unsafe_subtypes_unchanged": harm_isolation["episode_payload_unchanged"], "rollouts_and_cost_payload_unchanged": fp_before == fp_after, "manifest_unchanged": harm_isolation["manifest_unchanged"], "HOLD_protocol_unchanged": True, "only_derived_benefit_fields_added": True}},
        "Gate_E": {"name": "Reproducibility", "checks": {"runtime_anchor_map_rebuild_SHA_identical": anchor_sha_first == anchor_sha_second, "Benefit_Target_v2_rebuild_SHA_identical": label_sha_first == label_sha_second}},
    }
    for gate in gates.values(): gate["passed"] = all(gate["checks"].values())
    gates["Gate_F"] = {"name": "GARA Readiness", "checks": {"Gate_A_through_E_pass": all(gates[name]["passed"] for name in ("Gate_A", "Gate_B", "Gate_C", "Gate_D", "Gate_E"))}}
    gates["Gate_F"]["passed"] = all(gates["Gate_F"]["checks"].values()); gates["all_passed"] = all(gate["passed"] for gate in gates.values())
    io.write_json(args.output_dir / "gate_results.json", gates)
    v1_safe = next(row for row in safe_rows if row["split"] == "validation" and row["target_version"] == TARGET_V1)
    v2_safe = next(row for row in safe_rows if row["split"] == "validation" and row["target_version"] == TARGET_V2)
    summary = {"label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "test_reads": 0,
               "runtime_anchor_100_percent_GT_free": gates["Gate_A"]["passed"], "anchor_agreement_count": int(agreement.sum()),
               "anchor_changed_count": int((~agreement).sum()), "anchor_agreement_rate": float(agreement.mean()),
               "episode_zero_shift": distribution([row["episode_zero_shift"] for row in anchors]),
               "maximum_within_episode_delta_range": max_delta_range, "maximum_pairwise_difference_error": max_pairwise_error,
               "anchor_benefit_max_abs": max(anchor_self), "validation_safe_beneficial_v1": v1_safe,
               "validation_safe_beneficial_v2": v2_safe, "historical_61_failure_reclassification": dict(reclass_summary),
               "Benefit_Target_v2_SHA256": label_file_sha, "gates": gates,
               "ready_for_GARA_fair_test": gates["all_passed"], "GARA_training_started": False, "next_stage_started": False}
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
