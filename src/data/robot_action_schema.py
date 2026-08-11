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


PHASE4A_ACTIONS = (
    RobotAction.KEEP,
    RobotAction.SPEED_DOWN_10,
    RobotAction.SPEED_UP_10,
    RobotAction.DISTANCE_PLUS_0_2,
    RobotAction.DISTANCE_MINUS_0_2,
)


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


def validate_robot_history(robot_history: np.ndarray) -> None:
    values = np.asarray(robot_history)
    if values.ndim < 2 or values.shape[-1] != len(ROBOT_HISTORY_FIELDS):
        raise ValueError(
            f"robot_history must end in {len(ROBOT_HISTORY_FIELDS)} fields"
        )
    if not np.isfinite(values).all():
        raise ValueError("robot_history contains non-finite values")
