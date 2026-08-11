import math

from src.data.data_quality import inspect_data_quality
from src.data.ros_trajectory_schema import RosTrajectoryRecord


def row(timestamp: float, x: float, track: str = "a", frame: str = "map") -> RosTrajectoryRecord:
    return RosTrajectoryRecord(
        timestamp=timestamp,
        trial_id="trial",
        session_id="session",
        person_id_anonymous="person",
        scene_id="scene",
        track_id=track,
        coordinate_frame=frame,
        human_x=x,
        human_y=0.0,
    )


def test_quality_detects_required_failure_modes_and_statistics() -> None:
    records = [
        row(0.2, 2.0),
        row(0.0, 0.0),
        row(0.0, 0.0),
        row(0.4, 5.0, track="b", frame="odom"),
        row(0.5, math.nan, track="b", frame="odom"),
        row(0.6, 5.1, track="b", frame=""),
    ]
    report = inspect_data_quality(records, target_hz=10.0, jump_speed_threshold=1.0, valid_window_count=7)
    kinds = {issue.kind for issue in report.issues}
    assert {
        "unsorted_timestamp",
        "duplicate_timestamp",
        "missing_frames",
        "trajectory_jump",
        "track_id_switch",
        "nan_or_inf",
        "missing_coordinate_frame",
        "inconsistent_coordinate_frame",
    } <= kinds
    assert report.statistics["track_switch_count"] == 1
    assert report.statistics["valid_window_count"] == 7
    assert report.statistics["missing_ratio"] > 0
