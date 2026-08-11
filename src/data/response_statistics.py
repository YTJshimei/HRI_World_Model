"""Observable statistics extracted from completed interactions only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.functional_response_state import (
    RESPONSE_STATE_DIM,
    response_state_mask_for_action,
)
from src.data.personal_interaction_memory import PersonalInteractionRecord
from src.data.robot_action_schema import action_feature
from src.data.skeleton_schema import compute_root, shoulder_joints


RESPONSE_STATISTIC_NAMES = (
    "action_speed_delta",
    "action_distance_delta",
    "action_lateral_delta",
    "action_active",
    "pre_human_speed",
    "post_human_speed",
    "human_speed_delta",
    "distance_before",
    "distance_after",
    "distance_delta",
    "lateral_displacement",
    "heading_response",
    "observed_response_delay",
    "action_effect_magnitude",
)
RESPONSE_STATISTIC_DIM = len(RESPONSE_STATISTIC_NAMES)


@dataclass(frozen=True)
class ObservableResponseStatistics:
    values: np.ndarray
    validity_mask: np.ndarray
    response_state_mask: np.ndarray
    executed_action: int


def _angle(value: np.ndarray) -> float:
    return float(np.arctan2(value[1], value[0]))


def extract_response_statistics(
    record: PersonalInteractionRecord,
    sample_rate_hz: float = 10.0,
) -> ObservableResponseStatistics:
    """Extract r_i without consulting person/profile hidden parameters."""
    history_root = np.asarray(record.human_root_history)
    future_root = compute_root(record.human_future_response)
    pre_velocity = (history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz
    post_velocity = (future_root[-1, :2] - future_root[-2, :2]) * sample_rate_hz
    pre_speed, post_speed = np.linalg.norm(pre_velocity), np.linalg.norm(post_velocity)
    forward = pre_velocity / pre_speed if pre_speed > 1e-6 else np.asarray((1.0, 0.0))
    lateral = np.asarray((-forward[1], forward[0]))
    displacement = future_root[-1, :2] - history_root[-1, :2]
    left, right = shoulder_joints
    before_heading = _angle(
        record.human_local_pose_history[-1, right, :2]
        - record.human_local_pose_history[-1, left, :2]
    )
    future_root_full = compute_root(record.human_future_response)
    future_local = record.human_future_response - future_root_full[:, None, :]
    after_heading = _angle(future_local[-1, right, :2] - future_local[-1, left, :2])
    heading_response = float(
        np.arctan2(np.sin(after_heading - before_heading), np.cos(after_heading - before_heading))
    )
    feature = action_feature(record.executed_action)
    values = np.asarray(
        (
            feature[0], feature[1], feature[2], feature[3],
            pre_speed, post_speed, post_speed - pre_speed,
            record.human_robot_distance_before, record.human_robot_distance_after,
            record.human_robot_distance_after - record.human_robot_distance_before,
            float(np.dot(displacement, lateral)), heading_response,
            record.response_delay_observed,
            float(np.linalg.norm(record.action_effect, axis=-1).mean()),
        ),
        dtype=np.float32,
    )
    validity = np.isfinite(values)
    if not validity.all():
        values = np.nan_to_num(values)
    state_mask = response_state_mask_for_action(record.executed_action)
    if state_mask.shape != (RESPONSE_STATE_DIM,):
        raise RuntimeError("response-state mask shape regression")
    return ObservableResponseStatistics(
        values=values,
        validity_mask=validity,
        response_state_mask=state_mask,
        executed_action=int(record.executed_action),
    )


def pad_response_statistics(
    records: tuple[PersonalInteractionRecord, ...], max_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(records) > max_k:
        raise ValueError("support exceeds max_k")
    values = np.zeros((max_k, RESPONSE_STATISTIC_DIM), dtype=np.float32)
    support_mask = np.zeros(max_k, dtype=bool)
    state_mask = np.zeros((max_k, RESPONSE_STATE_DIM), dtype=bool)
    for index, record in enumerate(records):
        statistics = extract_response_statistics(record)
        values[index] = statistics.values
        support_mask[index] = True
        state_mask[index] = statistics.response_state_mask
    return values, support_mask, state_mask
