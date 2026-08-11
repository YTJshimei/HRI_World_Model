"""Leakage-neutral resampling of ROS-derived trajectories."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from typing import Iterable

from src.data.ros_trajectory_schema import NUMERIC_FIELDS, RosTrajectoryRecord

DEFAULT_TARGET_HZ = 10.0
DEFAULT_MAX_INTERPOLATION_GAP_SECONDS = 0.3


def sort_records(records: Iterable[RosTrajectoryRecord]) -> list[RosTrajectoryRecord]:
    return sorted(records, key=lambda row: (row.group_key, row.timestamp, row.track_id))


def _stream_key(record: RosTrajectoryRecord) -> tuple[str, ...]:
    return (
        record.trial_id,
        record.session_id,
        record.person_id_anonymous,
        record.scene_id,
        record.track_id,
        record.coordinate_frame,
    )


def _interpolate(
    left: RosTrajectoryRecord, right: RosTrajectoryRecord, timestamp: float
) -> RosTrajectoryRecord:
    fraction = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
    updates = {"timestamp": timestamp}
    for name in NUMERIC_FIELDS:
        if name == "timestamp":
            continue
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        updates[name] = (
            None
            if left_value is None or right_value is None
            else float(left_value + fraction * (right_value - left_value))
        )
    # Actions are events, not continuous signals; never invent them by interpolation.
    updates.update(
        requested_action=None,
        validated_action=None,
        executed_action=None,
    )
    return replace(left, **updates)


def resample_records(
    records: Iterable[RosTrajectoryRecord],
    target_hz: float = DEFAULT_TARGET_HZ,
    max_interpolation_gap_seconds: float = DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
) -> list[list[RosTrajectoryRecord]]:
    """Return contiguous streams; large gaps and identity changes create boundaries."""
    if target_hz <= 0:
        raise ValueError("target_hz 必须大于 0")
    timestep = 1.0 / target_hz
    if max_interpolation_gap_seconds < timestep:
        raise ValueError("最大插值间隔不能小于目标 timestep")

    grouped: dict[tuple[str, ...], list[RosTrajectoryRecord]] = {}
    for record in records:
        grouped.setdefault(_stream_key(record), []).append(record)

    output: list[list[RosTrajectoryRecord]] = []
    for key in sorted(grouped):
        stream = sorted(grouped[key], key=lambda row: row.timestamp)
        timestamps = [row.timestamp for row in stream]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(f"检测到重复 timestamp，必须先解决：stream={key}")
        if not stream:
            continue
        segment = [stream[0]]
        for current in stream[1:]:
            previous = segment[-1]
            gap = current.timestamp - previous.timestamp
            if gap > max_interpolation_gap_seconds + 1e-9:
                output.append(segment)
                segment = [current]
                continue
            steps = max(1, int(round(gap / timestep)))
            if not isclose(gap, steps * timestep, abs_tol=timestep * 0.05):
                raise ValueError(
                    f"timestamp 无法对齐 {target_hz:g}Hz 网格：{previous.timestamp} -> {current.timestamp}"
                )
            for step in range(1, steps):
                segment.append(_interpolate(previous, current, previous.timestamp + step * timestep))
            segment.append(current)
        output.append(segment)
    return output
