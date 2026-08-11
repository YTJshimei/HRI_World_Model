"""Canonical row schema for anonymized ROS-derived human trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any, ClassVar


NUMERIC_FIELDS = (
    "timestamp",
    "human_x",
    "human_y",
    "human_z",
    "human_vx",
    "human_vy",
    "robot_x",
    "robot_y",
    "robot_yaw",
    "robot_linear_velocity",
    "robot_angular_velocity",
    "cmd_vel_linear",
    "cmd_vel_angular",
    "human_robot_distance",
    "relative_bearing",
)


@dataclass(frozen=True)
class RosTrajectoryRecord:
    """One timestamped observation in a single common coordinate frame."""

    timestamp: float
    trial_id: str
    session_id: str
    person_id_anonymous: str
    scene_id: str
    track_id: str
    coordinate_frame: str
    human_x: float | None = None
    human_y: float | None = None
    human_z: float | None = None
    human_vx: float | None = None
    human_vy: float | None = None
    robot_x: float | None = None
    robot_y: float | None = None
    robot_yaw: float | None = None
    robot_linear_velocity: float | None = None
    robot_angular_velocity: float | None = None
    cmd_vel_linear: float | None = None
    cmd_vel_angular: float | None = None
    human_robot_distance: float | None = None
    relative_bearing: float | None = None
    requested_action: str | None = None
    validated_action: str | None = None
    executed_action: str | None = None

    CSV_FIELDS: ClassVar[tuple[str, ...]]

    @property
    def group_key(self) -> tuple[str, str, str]:
        """Leakage boundary used before window generation."""
        return self.trial_id, self.session_id, self.person_id_anonymous

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "RosTrajectoryRecord":
        missing = [name for name in cls.CSV_FIELDS if name not in row]
        if missing:
            raise ValueError(f"标准轨迹缺少字段：{', '.join(missing)}")
        values: dict[str, Any] = {}
        for name in cls.CSV_FIELDS:
            raw = row[name]
            if name in NUMERIC_FIELDS:
                if raw is None or str(raw).strip() == "":
                    values[name] = None
                else:
                    try:
                        values[name] = float(raw)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"字段 {name} 不是有效数值：{raw!r}") from exc
            else:
                text = "" if raw is None else str(raw).strip()
                values[name] = text or None
        if values["timestamp"] is None:
            raise ValueError("timestamp 不能为空")
        for required in (
            "trial_id",
            "session_id",
            "person_id_anonymous",
            "scene_id",
            "track_id",
            "coordinate_frame",
        ):
            if values[required] is None:
                raise ValueError(f"{required} 不能为空")
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


RosTrajectoryRecord.CSV_FIELDS = tuple(field.name for field in fields(RosTrajectoryRecord))


def required_position_is_finite(record: RosTrajectoryRecord) -> bool:
    return (
        record.human_x is not None
        and record.human_y is not None
        and isfinite(record.human_x)
        and isfinite(record.human_y)
    )
