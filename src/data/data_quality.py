"""Quality checks and descriptive statistics for standardized trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable

import numpy as np

from src.data.ros_trajectory_schema import NUMERIC_FIELDS, RosTrajectoryRecord
from src.data.trajectory_resampler import DEFAULT_TARGET_HZ


@dataclass(frozen=True)
class QualityIssue:
    kind: str
    trial_id: str
    timestamp: float | None
    detail: str


@dataclass(frozen=True)
class QualityReport:
    issues: tuple[QualityIssue, ...]
    statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [asdict(issue) for issue in self.issues],
            "statistics": self.statistics,
        }


def _identity_key(record: RosTrajectoryRecord) -> tuple[str, str, str]:
    return record.group_key


def inspect_data_quality(
    records: Iterable[RosTrajectoryRecord],
    target_hz: float = DEFAULT_TARGET_HZ,
    jump_speed_threshold: float = 3.0,
    valid_window_count: int = 0,
) -> QualityReport:
    if target_hz <= 0 or jump_speed_threshold <= 0:
        raise ValueError("target_hz 和 jump_speed_threshold 必须大于 0")
    rows = list(records)
    issues: list[QualityIssue] = []
    timestep = 1.0 / target_hz

    original_keys = [(_identity_key(row), row.timestamp) for row in rows]
    if original_keys != sorted(original_keys):
        issues.append(QualityIssue("unsorted_timestamp", "*", None, "输入记录未按 identity/timestamp 排序"))

    grouped: dict[tuple[str, str, str], list[RosTrajectoryRecord]] = {}
    for row in rows:
        grouped.setdefault(_identity_key(row), []).append(row)
        for field in NUMERIC_FIELDS:
            value = getattr(row, field)
            if value is not None and not isfinite(value):
                issues.append(QualityIssue("nan_or_inf", row.trial_id, row.timestamp, field))
        if not row.coordinate_frame:
            issues.append(QualityIssue("missing_coordinate_frame", row.trial_id, row.timestamp, "coordinate_frame 为空"))

    all_speeds: list[float] = []
    all_accelerations: list[float] = []
    all_heading_changes: list[float] = []
    missing_frames = 0
    track_switches = 0
    durations = []

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: row.timestamp)
        if ordered:
            durations.append(max(0.0, ordered[-1].timestamp - ordered[0].timestamp))
        frames = {row.coordinate_frame for row in ordered}
        if len(frames) > 1:
            issues.append(QualityIssue("inconsistent_coordinate_frame", key[0], None, str(sorted(frames))))
        seen_timestamps: set[float] = set()
        velocities = []
        for index, row in enumerate(ordered):
            if row.timestamp in seen_timestamps:
                issues.append(QualityIssue("duplicate_timestamp", row.trial_id, row.timestamp, row.track_id))
            seen_timestamps.add(row.timestamp)
            if index == 0:
                continue
            previous = ordered[index - 1]
            if row.track_id != previous.track_id:
                track_switches += 1
                issues.append(QualityIssue("track_id_switch", row.trial_id, row.timestamp, f"{previous.track_id} -> {row.track_id}"))
                continue
            delta = row.timestamp - previous.timestamp
            if delta <= 0:
                continue
            missing = max(0, int(round(delta / timestep)) - 1)
            missing_frames += missing
            if missing:
                issues.append(QualityIssue("missing_frames", row.trial_id, previous.timestamp, f"count={missing}, gap={delta:.6f}s"))
            if None in (previous.human_x, previous.human_y, row.human_x, row.human_y):
                continue
            velocity = np.array(
                [(row.human_x - previous.human_x) / delta, (row.human_y - previous.human_y) / delta],
                dtype=np.float64,
            )
            speed = float(np.linalg.norm(velocity))
            all_speeds.append(speed)
            velocities.append((row.timestamp, velocity))
            if speed > jump_speed_threshold:
                issues.append(QualityIssue("trajectory_jump", row.trial_id, row.timestamp, f"speed={speed:.6f}"))
        for (left_time, left_velocity), (right_time, right_velocity) in zip(velocities, velocities[1:]):
            delta = right_time - left_time
            if delta <= 0:
                continue
            all_accelerations.append(float(np.linalg.norm(right_velocity - left_velocity) / delta))
            left_heading = np.arctan2(left_velocity[1], left_velocity[0])
            right_heading = np.arctan2(right_velocity[1], right_velocity[0])
            difference = np.arctan2(np.sin(right_heading - left_heading), np.cos(right_heading - left_heading))
            all_heading_changes.append(float(abs(difference)))

    def distribution(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "p95": None, "max": None}
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    denominator = len(rows) + missing_frames
    statistics = {
        "duration_seconds": float(sum(durations)),
        "missing_ratio": float(missing_frames / denominator) if denominator else 0.0,
        "average_speed": float(np.mean(all_speeds)) if all_speeds else None,
        "max_speed": float(np.max(all_speeds)) if all_speeds else None,
        "acceleration_distribution": distribution(all_accelerations),
        "heading_change_distribution_rad": distribution(all_heading_changes),
        "track_switch_count": track_switches,
        "valid_window_count": int(valid_window_count),
        "record_count": len(rows),
        "detected_missing_frames": missing_frames,
    }
    return QualityReport(tuple(issues), statistics)
