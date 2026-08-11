from pathlib import Path

import pytest

from src.data.ros_trajectory_loader import (
    SplitManifest,
    load_trajectory_csv,
    validate_manifest_coverage,
)
from src.data.ros_trajectory_schema import RosTrajectoryRecord


def test_schema_contains_all_reserved_fields() -> None:
    required = {
        "timestamp", "trial_id", "session_id", "person_id_anonymous", "scene_id", "track_id",
        "human_x", "human_y", "human_z", "human_vx", "human_vy", "robot_x", "robot_y",
        "robot_yaw", "robot_linear_velocity", "robot_angular_velocity", "cmd_vel_linear",
        "cmd_vel_angular", "human_robot_distance", "relative_bearing", "requested_action",
        "validated_action", "executed_action", "coordinate_frame",
    }
    assert required <= set(RosTrajectoryRecord.CSV_FIELDS)


def test_csv_roundtrip_keeps_empty_optional_fields(mock_ros_trajectory_files: tuple[Path, Path]) -> None:
    csv_path, _ = mock_ros_trajectory_files
    records = load_trajectory_csv(csv_path)
    assert len(records) == 105
    assert records[0].cmd_vel_linear is None
    assert records[0].coordinate_frame == "map"


def test_manifest_rejects_trial_leakage() -> None:
    with pytest.raises(ValueError, match="跨 split"):
        SplitManifest(("trial_a",), ("trial_a",), ("trial_b",))


def test_manifest_requires_exact_coverage(mock_ros_trajectory_files: tuple[Path, Path]) -> None:
    csv_path, _ = mock_ros_trajectory_files
    records = load_trajectory_csv(csv_path)
    incomplete = SplitManifest(("mock_train",), ("mock_val",), ("unknown",))
    with pytest.raises(ValueError, match="未分配"):
        validate_manifest_coverage(incomplete, records)
