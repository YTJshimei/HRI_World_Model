"""Build rich temporal samples from the frozen synthetic Phase 4/5 episodes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.data.robot_action_schema import action_feature
from src.data.skeleton_schema import compute_root
from src.multimodal.temporal_schema import (
    ACTION_ORDER, DT_SECONDS, FUTURE_FRAMES, HISTORY_FRAMES, MOTION_STATE_ORDER,
    RichTemporalSample, STREAM_ORDER, TemporalTargets,
)


def _motion(history: np.ndarray) -> np.ndarray:
    root = compute_root(history).astype(np.float32); velocity = np.gradient(root, DT_SECONDS, axis=0)
    speed = np.linalg.norm(velocity[:, :2], axis=-1); heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    turn = np.gradient(heading, DT_SECONDS); acceleration = np.gradient(speed, DT_SECONDS)
    states = np.zeros((len(root), len(MOTION_STATE_ORDER)), np.float32)
    for i in range(len(root)):
        state = 0 if speed[i] < .12 else 2 if speed[i] > 1.8 else 5 if abs(turn[i]) > .35 else 3 if acceleration[i] > .2 else 4 if acceleration[i] < -.2 else 1
        states[i, state] = 1
    return np.concatenate((root, velocity, speed[:, None], acceleration[:, None], heading[:, None], turn[:, None], states), axis=-1).astype(np.float32)


def _candidate_future(robot_xy: np.ndarray, robot_history: np.ndarray) -> np.ndarray:
    xy = np.asarray(robot_xy, np.float32); previous = np.vstack((robot_history[-1, :2], xy[:-1]))
    delta = xy - previous; yaw = np.arctan2(delta[:, 1], delta[:, 0]); speed = np.linalg.norm(delta, axis=-1) / DT_SECONDS
    angular = np.gradient(np.unwrap(yaw), DT_SECONDS)
    return np.column_stack((xy, yaw, speed, angular)).astype(np.float32)


def _support_action_id(name: str) -> int | None:
    if "SPEED_DOWN" in name: return 1
    if "SPEED_UP" in name: return 2
    if "DISTANCE_PLUS" in name: return 3
    if "DISTANCE_MINUS" in name: return 4
    return None


def _interaction_history(support: Iterable[str], root: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values = np.zeros((HISTORY_FRAMES, 13), np.float32); valid = np.zeros((HISTORY_FRAMES, 13), bool)
    action_ids = [value for item in support if (value := _support_action_id(str(item))) is not None]
    # Only ordering exists in the frozen Phase4 record. Place ordered events in
    # explicit event slots and mark their exact age/response fields unavailable.
    slots = np.linspace(max(0, HISTORY_FRAMES - 2 * max(len(action_ids), 1)), HISTORY_FRAMES - 2, max(len(action_ids), 1), dtype=int)
    for slot, action_id in zip(slots, action_ids):
        values[slot, action_id] = 1; valid[slot, :5] = True
        values[slot, 9] = 1; valid[slot, 9] = True  # probe/identification phase
        values[slot, 10] = len(set(action_ids)) / len(ACTION_ORDER); valid[slot, 10] = True
    # Current observed response proxy comes only from pre-decision root motion.
    values[-1, 5:8] = np.r_[root[-1, :2] - root[-2, :2], np.linalg.norm(root[-1, :2] - root[-2, :2])]
    valid[-1, 5:8] = True; values[-1, 11] = float(bool(action_ids)); values[-1, 12] = float(not action_ids); valid[-1, 11:13] = True
    return values, valid, {"support_event_count": len(action_ids), "exact_support_timestamps_available": False}


def _temporal_tags(history: np.ndarray, visibility: np.ndarray, support_count: int) -> tuple[str, ...]:
    tags = []
    hidden = ~np.asarray(visibility, bool)
    maximum = 0
    for joint in range(hidden.shape[1]):
        run = 0
        for value in hidden[:, joint]: run = run + 1 if value else 0; maximum = max(maximum, run)
    if maximum >= 5: tags.append("C7_long_occlusion_history")
    if support_count: tags.append("C8_recent_intervention_transition")
    motion = _motion(history); labels = motion[:, -len(MOTION_STATE_ORDER):].argmax(-1)
    if labels[0] != labels[-1] or len(np.unique(labels)) >= 3: tags.append("C9_motion_transition")
    return tuple(tags)


def longest_joint_occlusion_run(visibility: np.ndarray) -> dict[str, int]:
    """Return the longest consecutive missing run over real COCO-17 joints."""
    values = np.asarray(visibility, bool)
    if values.shape != (HISTORY_FRAMES, 17): raise ValueError("visibility must have shape [20,17]")
    best = {"frames": 0, "joint": -1, "start": -1, "end": -1}
    for joint in range(values.shape[1]):
        start = None
        for frame, visible in enumerate(np.r_[values[:, joint], True]):
            if not visible and start is None: start = frame
            if visible and start is not None:
                length = frame - start
                if length > best["frames"]: best = {"frames": length, "joint": joint, "start": start, "end": frame - 1}
                start = None
    return best


def is_c7_long_occlusion(visibility: np.ndarray, confidence: np.ndarray | None = None) -> bool:
    """Frozen Phase5B-0 C7 rule: any real joint missing for >=5 frames."""
    values = np.asarray(visibility, bool)
    run = longest_joint_occlusion_run(values)
    if confidence is not None:
        conf = np.asarray(confidence)
        if conf.shape != values.shape: raise ValueError("confidence/visibility shapes differ")
        hidden = ~values
        if np.any(conf[hidden] != 0): return False
    return run["frames"] >= 5


def apply_continuous_occlusion(
    visibility: np.ndarray, confidence: np.ndarray, *, start: int, frames: int, joints: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a real history mask/confidence outage; coordinates remain masked, not zero-imputed."""
    if frames < 5 or start < 0 or start + frames > HISTORY_FRAMES: raise ValueError("C7 interval must be >=5 frames inside history")
    if not joints or any(joint < 0 or joint >= 17 for joint in joints): raise ValueError("invalid occluded joint set")
    result_visibility = np.asarray(visibility, bool).copy(); result_confidence = np.asarray(confidence, np.float32).copy()
    result_visibility[start:start + frames, list(joints)] = False
    result_confidence[start:start + frames, list(joints)] = 0.0
    if not is_c7_long_occlusion(result_visibility, result_confidence): raise RuntimeError("structured occlusion did not satisfy frozen C7")
    return result_visibility, result_confidence


def build_temporal_samples(episodes, static_samples, targets, meta, split: str) -> list[RichTemporalSample]:
    """Build candidate samples after episodes/person/context have been split."""
    episode_map = {episode.key: episode for episode in episodes}; result = []
    for static, target, row in zip(static_samples, targets, meta):
        episode = episode_map[(row["scenario"], row["sample"])]; first = episode.first; data = first["sample_data"]
        history = np.asarray(data["history"], np.float32); confidence = np.asarray(data["confidence"], np.float32)
        visibility = np.asarray(data["visibility"], bool); robot = np.asarray(data["robot"], np.float32); root = compute_root(history)
        action_ids = np.asarray(first["predicted_rollout"].action_ids); action_index = int(np.flatnonzero(action_ids == int(row["action"]))[0])
        motion = _motion(history); functional = np.zeros((HISTORY_FRAMES, 18), np.float32); functional_valid = np.zeros_like(functional, bool)
        functional[-1] = np.concatenate((first["theta_hat"], first["theta_std"], episode.confidence.dimension_confidence)); functional_valid[-1] = True
        visible_ratio = visibility.mean(axis=1); mean_conf = confidence.mean(axis=1)
        visibility_stream = np.column_stack((visibility.any(axis=1), visible_ratio, mean_conf, mean_conf * visible_ratio)).astype(np.float32)
        wm = np.zeros((HISTORY_FRAMES, 8), np.float32); wm_valid = np.zeros_like(wm, bool)
        sigma = np.asarray(episode.artifact.root_belief.sigma_root, np.float32)
        wm[-1] = np.r_[sigma.mean(0), np.linalg.norm(sigma[..., :2], axis=-1).mean(), episode.safety_prediction["sigma_minimum"][action_index], episode.safety_prediction["p_unsafe"][action_index], 1/(1+np.linalg.norm(sigma)), episode.confidence.root_confidence]
        wm_valid[-1] = True
        interaction, interaction_valid, interaction_audit = _interaction_history(first["support"], root)
        action = int(row["action"]); candidate = np.r_[np.eye(7, dtype=np.float32)[action], action_feature(action)].astype(np.float32)
        robot_future = _candidate_future(first["predicted_rollout"].predicted_robot_xy[action_index], robot)
        speed = np.linalg.norm(motion[:, 3:5], axis=-1); distance = robot[:, 5]
        scene = np.asarray((visibility.mean(), confidence.mean(), distance[-1], distance[-1]-distance[0], speed.mean(), speed.std(), robot[-1, 6], len(first["support"])/5), np.float32)
        streams = {
            "skeleton_history": history, "human_motion_history": motion, "robot_history": robot,
            "functional_history": functional, "visibility_history": visibility_stream,
            "wm_diagnostic_history": wm, "interaction_history": interaction,
            "candidate_action": candidate, "candidate_robot_future": robot_future, "scene_context": scene,
        }
        masks = {
            "skeleton_history": visibility[..., None].repeat(3, axis=-1), "human_motion_history": np.ones_like(motion, bool),
            "robot_history": np.ones_like(robot, bool), "functional_history": functional_valid,
            "visibility_history": np.ones_like(visibility_stream, bool), "wm_diagnostic_history": wm_valid,
            "interaction_history": interaction_valid, "candidate_action": np.ones_like(candidate, bool),
            "candidate_robot_future": np.ones_like(robot_future, bool), "scene_context": np.ones_like(scene, bool),
            "history_valid_mask": np.ones(HISTORY_FRAMES, bool), "history_padding_mask": np.ones(HISTORY_FRAMES, bool),
            "candidate_future_valid_mask": np.ones(FUTURE_FRAMES, bool),
        }
        episode_id = f"{split}:{row['scenario']}:{row['sample']}"
        result.append(RichTemporalSample(
            streams=streams, masks=masks,
            timestamps={"history": np.arange(-HISTORY_FRAMES + 1, 1, dtype=np.float32) * DT_SECONDS,
                        "candidate_future": np.arange(1, FUTURE_FRAMES + 1, dtype=np.float32) * DT_SECONDS},
            targets=TemporalTargets(float(target.benefit), bool(target.harm), 0.0, False, bool(row["feasible"]), float(row["GT_cost"]), bool(row["GT_unsafe"])),
            sample_id=f"{episode_id}:{action}", episode_id=episode_id, split=split, context_split=str(row["context_split"]),
            temporal_tags=_temporal_tags(history, visibility, interaction_audit["support_event_count"]),
            split_metadata={"person_profile_id": int(row["profile"]), "scenario": row["scenario"], "motion_type_evaluation_only": str(data["action_type"]), "candidate_action_id_audit": action, **interaction_audit,
                            "keypoint_confidence_audit": confidence.copy(), "static_context_108": static.flattened().copy(),
                            "all_action_ids_evaluation_only": np.asarray(episode.personal_costs.action_ids, int).copy(),
                            "generic_costs_evaluation_only": np.asarray(episode.generic_costs.total, np.float32).copy(),
                            "personalized_costs_evaluation_only": np.asarray(episode.personal_costs.total, np.float32).copy(),
                            "gt_costs_evaluation_only": np.asarray(episode.gt_costs.total, np.float32).copy(),
                            "gt_unsafe_evaluation_only": (np.asarray(episode.gt_costs.unsafe_duration) > 0).copy()},
        ))
    return result


@dataclass(frozen=True)
class TemporalNormalizer:
    mean: dict[str, np.ndarray]; scale: dict[str, np.ndarray]; fit_sample_ids: tuple[str, ...]; fit_split: str = "train"


def fit_train_normalizer(samples: list[RichTemporalSample]) -> TemporalNormalizer:
    if not samples or any(sample.split != "train" for sample in samples): raise ValueError("temporal normalizer may only access train samples")
    mean, scale = {}, {}
    for name in STREAM_ORDER:
        values = np.stack([sample.streams[name] for sample in samples]); masks = np.stack([sample.masks[name] for sample in samples])
        selected = values[masks]
        mean[name] = np.asarray(selected.mean() if selected.size else 0, np.float32)
        scale[name] = np.asarray(max(float(selected.std()), 1e-5) if selected.size else 1, np.float32)
    return TemporalNormalizer(mean, scale, tuple(sample.sample_id for sample in samples))


def temporal_window(sample: RichTemporalSample, frames: int) -> dict[str, object]:
    """Select a declared suffix window without changing candidate/static streams."""
    if frames <= 0 or frames > HISTORY_FRAMES: raise ValueError("history window exceeds available pre-decision frames")
    temporal = {"skeleton_history", "human_motion_history", "robot_history", "functional_history",
                "visibility_history", "wm_diagnostic_history", "interaction_history"}
    return {
        "streams": {name: (np.asarray(value)[-frames:].copy() if name in temporal else np.asarray(value).copy()) for name, value in sample.streams.items()},
        "masks": {name: (np.asarray(value)[-frames:].copy() if name in temporal or name in ("history_valid_mask", "history_padding_mask") else np.asarray(value).copy()) for name, value in sample.masks.items()},
        "timestamps": {"history": np.asarray(sample.timestamps["history"])[-frames:].copy(),
                       "candidate_future": np.asarray(sample.timestamps["candidate_future"]).copy()},
    }


def validate_split_isolation(samples: list[RichTemporalSample], held_out_profiles=(2, 6)) -> dict[str, object]:
    episodes: dict[str, str] = {}; branch_violations = []
    for sample in samples:
        prior = episodes.setdefault(sample.episode_id, sample.split)
        if prior != sample.split: branch_violations.append(sample.episode_id)
    train_profiles = {int(sample.split_metadata["person_profile_id"]) for sample in samples if sample.split == "train"}
    return {"episode_branch_violations": sorted(set(branch_violations)), "held_out_profiles": list(held_out_profiles),
            "held_out_profiles_in_train": sorted(train_profiles & set(held_out_profiles)),
            "passed": not branch_violations and not (train_profiles & set(held_out_profiles))}


def static_bridge_audit(samples: list[RichTemporalSample]) -> dict[str, object]:
    exact = []
    for sample in samples:
        static = np.asarray(sample.split_metadata["static_context_108"], np.float32)
        exact.append(static.shape == (108,) and np.isfinite(static).all())
    return {"candidate_samples": len(samples), "exact_phase5a_108_available": int(sum(exact)),
            "exact_paired_export_rate": float(np.mean(exact)) if exact else 0.0,
            "bridge_method": "the exact frozen Phase5A StructuredContextTokens are paired as audit-only metadata with the same episode/candidate",
            "runtime_input": False,
            "exact_reconstruction_from_phase5b_runtime_streams": False,
            "unreconstructable_from_runtime_streams": [
                {"phase5a_group": "functional", "reason": "the frozen 108D group includes population theta; Phase5B runtime streams retain only the runtime personal estimate/uncertainty/confidence"},
                {"phase5a_group": "uncertainty", "reason": "the frozen group includes support-coverage details not preserved exactly in the canonical temporal runtime streams"},
                {"phase5a_group": "diagnostic", "reason": "the frozen group compresses generic/personalized predicted human futures; Phase5B candidate future deliberately contains deterministic robot rollout only"},
                {"phase5a_group": "interaction", "reason": "the frozen group includes future predicted human-distance changes, deliberately excluded from Phase5B runtime history"},
            ],
            "passed": bool(exact and all(exact))}


def export_phase5a_static_108(sample: RichTemporalSample) -> np.ndarray:
    """Export the exact same-sample frozen 108D baseline, never a runtime stream."""
    value = np.asarray(sample.split_metadata.get("static_context_108"), np.float32)
    if value.shape != (108,) or not np.isfinite(value).all():
        raise ValueError("exact Phase5A 108D audit bridge is unavailable for this sample")
    return value.copy()
