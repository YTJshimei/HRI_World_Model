"""Canonical, leakage-safe rich temporal context protocol for Phase 5B."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from src.data.skeleton_schema import NUM_JOINTS, joint_names

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
SAMPLE_RATE_HZ = 10.0
DT_SECONDS = 0.1
HISTORY_FRAMES = 20
FUTURE_FRAMES = 10
HISTORY_WINDOWS = {"short": 10, "medium": 15, "long": 20}

STREAM_ORDER = (
    "skeleton_history", "human_motion_history", "robot_history",
    "functional_history", "visibility_history", "wm_diagnostic_history",
    "interaction_history", "candidate_action", "candidate_robot_future",
    "scene_context",
)
MOTION_STATE_ORDER = ("stop", "walk", "run", "accelerating", "decelerating", "turning")
ACTION_ORDER = ("KEEP", "SPEED_DOWN_10", "SPEED_UP_10", "DISTANCE_PLUS_0_2", "DISTANCE_MINUS_0_2")


class Availability(str, Enum):
    OBSERVED = "OBSERVED"
    ESTIMATED = "ESTIMATED"
    DETERMINISTIC_CANDIDATE_FUTURE = "DETERMINISTIC_CANDIDATE_FUTURE"
    TRAINING_TARGET_ONLY = "TRAINING_TARGET_ONLY"
    SPLIT_ONLY = "SPLIT_ONLY"


FORBIDDEN_RUNTIME_KEYS = frozenset({
    "person_id", "profile", "profile_id", "person_profile_id", "theta_true", "gt_theta",
    "gt_human_future", "future_global", "natural_future", "future_by_action",
    "gt_benefit", "benefit", "harm", "gt_best_action", "oracle_action", "gt_unsafe",
})

STREAM_DIMS = {
    "skeleton_history": (HISTORY_FRAMES, NUM_JOINTS, 3),
    "human_motion_history": (HISTORY_FRAMES, 16),  # root/velocity/speed/heading/turn + 6-state one-hot
    "robot_history": (HISTORY_FRAMES, 7),
    "functional_history": (HISTORY_FRAMES, 18),  # theta_hat/std/per-dimension confidence
    "visibility_history": (HISTORY_FRAMES, 4),
    "wm_diagnostic_history": (HISTORY_FRAMES, 8),
    "interaction_history": (HISTORY_FRAMES, 13),
    "candidate_action": (11,),  # 7-way one-hot plus 4 semantic features
    "candidate_robot_future": (FUTURE_FRAMES, 5),
    "scene_context": (8,),
}


@dataclass(frozen=True)
class TemporalTargets:
    benefit: float
    harm: bool
    uncertainty: float
    uncertainty_valid: bool
    feasible: bool
    gt_cost: float
    gt_unsafe: bool


@dataclass(frozen=True)
class RichTemporalSample:
    streams: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    timestamps: Mapping[str, np.ndarray]
    targets: TemporalTargets
    sample_id: str
    episode_id: str
    split: str
    context_split: str
    temporal_tags: tuple[str, ...]
    split_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if tuple(self.streams) != STREAM_ORDER:
            raise ValueError("temporal streams must use the canonical STREAM_ORDER")
        if set(self.streams) & FORBIDDEN_RUNTIME_KEYS:
            raise ValueError("training targets, identity, oracle and future-human fields cannot enter runtime streams")
        for name, shape in STREAM_DIMS.items():
            value = np.asarray(self.streams[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            if name not in self.masks:
                raise ValueError(f"{name} requires an explicit validity mask")
        if "person_profile_id" not in self.split_metadata:
            raise ValueError("anonymous synthetic profile metadata is required for split auditing")
        if set(self.split_metadata) & set(self.streams):
            raise ValueError("split-only identity metadata cannot duplicate runtime streams")
        history_time = np.asarray(self.timestamps["history"])
        candidate_time = np.asarray(self.timestamps["candidate_future"])
        if history_time.shape != (HISTORY_FRAMES,) or candidate_time.shape != (FUTURE_FRAMES,):
            raise ValueError("timestamp shapes violate the canonical timeline")
        if history_time.max() > 1e-8 or candidate_time.min() <= 0:
            raise ValueError("human history may not cross decision time; candidate future must be strictly after it")


def runtime_payload(sample: RichTemporalSample) -> dict[str, object]:
    """Return only fields that a runtime model may consume."""
    return {"streams": sample.streams, "masks": sample.masks, "timestamps": sample.timestamps}


def feature_registry() -> list[dict[str, object]]:
    return [
        {"stream": "skeleton_history", "fields": ["COCO17 xyz"], "availability": Availability.OBSERVED.value, "confidence_visibility_stream": "visibility_history"},
        {"stream": "human_motion_history", "fields": ["root xyz", "root velocity", "speed", "heading", "heading change", "motion-state one-hot"], "availability": Availability.ESTIMATED.value},
        {"stream": "robot_history", "fields": ["x", "y", "yaw", "linear velocity", "angular velocity", "distance", "bearing"], "availability": Availability.OBSERVED.value},
        {"stream": "functional_history", "fields": ["estimated theta", "posterior std", "dimension confidence"], "availability": Availability.ESTIMATED.value, "oracle_theta_forbidden": True},
        {"stream": "visibility_history", "fields": ["target visible", "keypoint valid ratio", "target confidence", "tracking confidence"], "availability": Availability.OBSERVED.value},
        {"stream": "wm_diagnostic_history", "fields": ["root uncertainty", "minimum-distance uncertainty", "p_unsafe", "model confidence", "context confidence"], "availability": Availability.ESTIMATED.value, "future_gt_error_forbidden": True},
        {"stream": "interaction_history", "fields": ["executed action", "observed root response", "action age", "probe phase", "recent action diversity"], "availability": Availability.OBSERVED.value, "timestamp_quality": "support order available; exact support wall-clock time unavailable"},
        {"stream": "candidate_action", "fields": ["action id one-hot", "structured semantics"], "availability": Availability.OBSERVED.value},
        {"stream": "candidate_robot_future", "fields": ["robot x", "robot y", "yaw", "linear velocity", "angular velocity"], "availability": Availability.DETERMINISTIC_CANDIDATE_FUTURE.value, "gt_human_response_forbidden": True},
        {"stream": "scene_context", "fields": ["runtime observable synthetic scene state"], "availability": Availability.ESTIMATED.value},
        {"stream": "targets", "fields": ["benefit", "harm", "uncertainty supervision", "cost", "unsafe"], "availability": Availability.TRAINING_TARGET_ONLY.value},
        {"stream": "split_metadata", "fields": ["episode", "person/profile", "context", "motion-action combination"], "availability": Availability.SPLIT_ONLY.value, "runtime_input_forbidden": True},
    ]


def schema_description() -> dict[str, object]:
    return {
        "label": LABEL, "stream_order": list(STREAM_ORDER), "stream_shapes": {k: list(v) for k, v in STREAM_DIMS.items()},
        "joint_names": list(joint_names), "motion_state_order": list(MOTION_STATE_ORDER), "action_order": list(ACTION_ORDER),
        "dt_seconds": DT_SECONDS, "sample_rate_hz": SAMPLE_RATE_HZ, "decision_time": 0.0,
        "history_index": "[t0-H+1 ... t0]", "candidate_future_index": "[t0+1 ... t0+F]",
        "history_windows": {name: {"frames": frames, "seconds": frames * DT_SECONDS} for name, frames in HISTORY_WINDOWS.items()},
        "decision_frequency": "event-driven: one decision point per current synthetic episode; no fixed frequency is encoded",
        "padding_rule": "left pad with zeros only when accompanied by padding_mask=False; padded values are never semantically interpreted",
    }
