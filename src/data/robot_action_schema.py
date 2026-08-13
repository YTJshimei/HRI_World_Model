"""Structured high-level robot state/action schema for synthetic interaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


ROBOT_HISTORY_FIELDS = (
    "x",
    "y",
    "yaw",
    "linear_velocity",
    "angular_velocity",
    "human_robot_distance",
    "relative_bearing",
)


class RobotAction(IntEnum):
    KEEP = 0
    SPEED_DOWN_10 = 1
    SPEED_UP_10 = 2
    DISTANCE_PLUS_0_2 = 3
    DISTANCE_MINUS_0_2 = 4
    LEFT_OFFSET = 5
    RIGHT_OFFSET = 6


class RobotActionV3(IntEnum):
    """Manifest-v3 action IDs; v2 IDs remain immutable.

    IDs 5 and 6 were already reserved for lateral offsets, so HOLD receives
    the next unused ID instead of being disguised as KEEP or reusing A5.
    """

    KEEP = 0
    SPEED_DOWN_10 = 1
    SPEED_UP_10 = 2
    DISTANCE_PLUS_0_2 = 3
    DISTANCE_MINUS_0_2 = 4
    LEFT_OFFSET = 5
    RIGHT_OFFSET = 6
    HOLD = 7


PHASE4A_ACTIONS = (
    RobotAction.KEEP,
    RobotAction.SPEED_DOWN_10,
    RobotAction.SPEED_UP_10,
    RobotAction.DISTANCE_PLUS_0_2,
    RobotAction.DISTANCE_MINUS_0_2,
)

HOLD_ACTION_ID = int(RobotActionV3.HOLD)
PHASE5B_V3_ACTIONS = (*PHASE4A_ACTIONS, RobotActionV3.HOLD)
V3_ACTION_ONE_HOT_DIM = max(int(action) for action in RobotActionV3) + 1


@dataclass(frozen=True)
class StructuredRobotAction:
    action: RobotAction
    speed_scale_delta: float = 0.0
    distance_offset_m: float = 0.0
    lateral_offset_m: float = 0.0


ACTION_DEFINITIONS = {
    RobotAction.KEEP: StructuredRobotAction(RobotAction.KEEP),
    RobotAction.SPEED_DOWN_10: StructuredRobotAction(
        RobotAction.SPEED_DOWN_10, speed_scale_delta=-0.10
    ),
    RobotAction.SPEED_UP_10: StructuredRobotAction(
        RobotAction.SPEED_UP_10, speed_scale_delta=0.10
    ),
    RobotAction.DISTANCE_PLUS_0_2: StructuredRobotAction(
        RobotAction.DISTANCE_PLUS_0_2, distance_offset_m=0.20
    ),
    RobotAction.DISTANCE_MINUS_0_2: StructuredRobotAction(
        RobotAction.DISTANCE_MINUS_0_2, distance_offset_m=-0.20
    ),
    RobotAction.LEFT_OFFSET: StructuredRobotAction(
        RobotAction.LEFT_OFFSET, lateral_offset_m=0.20
    ),
    RobotAction.RIGHT_OFFSET: StructuredRobotAction(
        RobotAction.RIGHT_OFFSET, lateral_offset_m=-0.20
    ),
}

# HOLD's semantic delta is the requested terminal speed change.  Its actual
# trajectory is rate-limited by the separate frozen hold-control protocol.
ACTION_DEFINITIONS[HOLD_ACTION_ID] = StructuredRobotAction(
    HOLD_ACTION_ID, speed_scale_delta=-1.0
)


def action_feature(action: int | RobotAction) -> np.ndarray:
    """Compact semantics, never a high-frequency cmd_vel sequence."""
    definition = ACTION_DEFINITIONS[RobotAction(int(action))]
    return np.asarray(
        (
            definition.speed_scale_delta,
            definition.distance_offset_m,
            definition.lateral_offset_m,
            float(definition.action != RobotAction.KEEP),
        ),
        dtype=np.float32,
    )


def action_feature_v3(action: int | RobotActionV3) -> np.ndarray:
    """Four continuous semantics shared by the versioned v3 action contract."""
    action_id = int(action)
    if action_id not in ACTION_DEFINITIONS:
        raise ValueError(f"unknown manifest-v3 action ID: {action_id}")
    definition = ACTION_DEFINITIONS[action_id]
    return np.asarray(
        (
            definition.speed_scale_delta,
            definition.distance_offset_m,
            definition.lateral_offset_m,
            float(action_id != int(RobotActionV3.KEEP)),
        ),
        dtype=np.float32,
    )


def candidate_action_vector_v3(action: int | RobotActionV3) -> np.ndarray:
    """Return 8-way one-hot + 4 semantic values (12D) for manifest-v3."""
    action_id = int(action)
    if action_id < 0 or action_id >= V3_ACTION_ONE_HOT_DIM:
        raise ValueError(f"action ID outside v3 one-hot support: {action_id}")
    one_hot = np.eye(V3_ACTION_ONE_HOT_DIM, dtype=np.float32)[action_id]
    return np.concatenate((one_hot, action_feature_v3(action_id)))


def validate_robot_history(robot_history: np.ndarray) -> None:
    values = np.asarray(robot_history)
    if values.ndim < 2 or values.shape[-1] != len(ROBOT_HISTORY_FIELDS):
        raise ValueError(
            f"robot_history must end in {len(ROBOT_HISTORY_FIELDS)} fields"
        )
    if not np.isfinite(values).all():
        raise ValueError("robot_history contains non-finite values")
