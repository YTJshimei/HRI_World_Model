"""Trajectory-derived adverse-response protocol for synthetic Phase 5B-1.7C."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.data.skeleton_schema import compute_root

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"

# Predeclared from the frozen synthetic motion support, not model performance:
# walk=0.70, fast-walk=1.20, run=2.00 m/s; acceleration/deceleration changes
# by about 0.04 m/s per 0.1 s frame; turn support is 0.50 rad/s.
EXTRA_DECELERATION_THRESHOLD_MPS2 = 1.50
EXTRA_LATERAL_DISPLACEMENT_THRESHOLD_M = 0.18
EXTRA_HEADING_CHANGE_THRESHOLD_RAD = 0.35
MIN_EVENT_DURATION_FRAMES = 2


@dataclass(frozen=True)
class AdverseResponseEvents:
    excessive_deceleration: bool
    abrupt_lateral_response: bool
    abrupt_heading_change: bool
    adverse_human_kinematic_response: bool
    extra_deceleration_mps2: float
    extra_lateral_displacement_m: float
    extra_heading_change_rad: float
    deceleration_duration_frames: int
    lateral_duration_frames: int
    heading_duration_frames: int


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.asarray((1.0, 0.0))


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _wrapped(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def trajectory_kinematics(
    human_history: np.ndarray,
    future_global: np.ndarray,
    sample_rate_hz: float = 10.0,
) -> dict[str, np.ndarray]:
    """Return physically dimensioned root/joint kinematics from real arrays."""
    history_root = compute_root(np.asarray(human_history, dtype=np.float64))
    future = np.asarray(future_global, dtype=np.float64)
    root = compute_root(future)
    joined_root = np.vstack((history_root[-1:], root))
    velocity = np.diff(joined_root[:, :2], axis=0) * sample_rate_hz
    acceleration = np.diff(np.vstack((((history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz)[None], velocity)), axis=0) * sample_rate_hz
    jerk = np.diff(np.vstack((acceleration[:1], acceleration)), axis=0) * sample_rate_hz
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    heading_change = np.r_[0.0, np.diff(heading)]
    joint_velocity = np.diff(np.concatenate((np.asarray(human_history[-1:], dtype=np.float64), future), axis=0), axis=0) * sample_rate_hz
    return {"root": root, "velocity": velocity, "acceleration": acceleration,
            "jerk": jerk, "heading": heading, "heading_change": heading_change,
            "joint_velocity": joint_velocity}


def derive_adverse_response_events(
    human_history: np.ndarray,
    natural_future: np.ndarray,
    candidate_future: np.ndarray,
    sample_rate_hz: float = 10.0,
) -> AdverseResponseEvents:
    """Label candidate-induced adverse motion relative to the natural baseline.

    The interface accepts trajectories and sample rate only; decision objectives,
    candidate identities, and identity metadata cannot enter event construction.
    """
    natural = trajectory_kinematics(human_history, natural_future, sample_rate_hz)
    candidate = trajectory_kinematics(human_history, candidate_future, sample_rate_hz)
    pre_velocity = (compute_root(human_history)[-1, :2] - compute_root(human_history)[-2, :2]) * sample_rate_hz
    forward = _unit(pre_velocity)
    lateral = np.asarray((-forward[1], forward[0]))

    natural_forward_accel = natural["acceleration"] @ forward
    candidate_forward_accel = candidate["acceleration"] @ forward
    extra_deceleration = natural_forward_accel - candidate_forward_accel
    deceleration_mask = extra_deceleration >= EXTRA_DECELERATION_THRESHOLD_MPS2

    root_effect = candidate["root"][:, :2] - natural["root"][:, :2]
    lateral_effect = np.abs(root_effect @ lateral)
    lateral_mask = lateral_effect >= EXTRA_LATERAL_DISPLACEMENT_THRESHOLD_M

    natural_heading_delta = _wrapped(natural["heading"] - natural["heading"][0])
    candidate_heading_delta = _wrapped(candidate["heading"] - candidate["heading"][0])
    extra_heading = np.abs(_wrapped(candidate_heading_delta - natural_heading_delta))
    heading_mask = extra_heading >= EXTRA_HEADING_CHANGE_THRESHOLD_RAD

    deceleration_frames = _longest_run(deceleration_mask)
    lateral_frames = _longest_run(lateral_mask)
    heading_frames = _longest_run(heading_mask)
    deceleration_event = deceleration_frames >= MIN_EVENT_DURATION_FRAMES
    lateral_event = lateral_frames >= MIN_EVENT_DURATION_FRAMES
    heading_event = heading_frames >= MIN_EVENT_DURATION_FRAMES
    return AdverseResponseEvents(
        deceleration_event, lateral_event, heading_event,
        deceleration_event or lateral_event or heading_event,
        float(np.max(extra_deceleration, initial=0.0)),
        float(np.max(lateral_effect, initial=0.0)),
        float(np.max(extra_heading, initial=0.0)),
        deceleration_frames, lateral_frames, heading_frames,
    )


def protocol_definition() -> dict[str, object]:
    return {
        "label": LABEL, "version": "phase5b_adverse_response_protocol_v1",
        "threshold_selection_used_model_performance": False,
        "threshold_source": "predeclared synthetic motion support/domain rules",
        "event_families": {
            "PHYSICAL_SAFETY_EVENT": {"included": True, "formula": "unsafe_duration > 0", "definition_changed": False},
            "ADVERSE_HUMAN_KINEMATIC_RESPONSE": {
                "included": True, "baseline": "natural human future from the same initial state",
                "events": {
                    "EXCESSIVE_DECELERATION": {"threshold": EXTRA_DECELERATION_THRESHOLD_MPS2, "unit": "m/s^2", "minimum_duration_frames": MIN_EVENT_DURATION_FRAMES},
                    "ABRUPT_LATERAL_RESPONSE": {"threshold": EXTRA_LATERAL_DISPLACEMENT_THRESHOLD_M, "unit": "m", "minimum_duration_frames": MIN_EVENT_DURATION_FRAMES},
                    "ABRUPT_HEADING_CHANGE": {"threshold": EXTRA_HEADING_CHANGE_THRESHOLD_RAD, "unit": "rad", "minimum_duration_frames": MIN_EVENT_DURATION_FRAMES},
                },
            },
            "INTERACTION_DISRUPTION_EVENT": {"included": False, "reason": "no trajectory-derived protocol event with an independently justified threshold exists"},
        },
        "harm_v2": "GT unsafe OR GT adverse_human_kinematic_response",
        "forbidden_inputs": ["benefit", "benefit sign", "total cost", "generic-vs-candidate cost", "best action", "profile ID"],
    }


def event_as_dict(event: AdverseResponseEvents) -> dict[str, object]:
    return asdict(event)
