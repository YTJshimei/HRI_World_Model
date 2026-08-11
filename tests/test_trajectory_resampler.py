import pytest

from src.data.ros_trajectory_schema import RosTrajectoryRecord
from src.data.trajectory_resampler import resample_records


def record(timestamp: float, x: float, track: str = "track_0") -> RosTrajectoryRecord:
    return RosTrajectoryRecord(
        timestamp=timestamp,
        trial_id="trial",
        session_id="session",
        person_id_anonymous="person",
        scene_id="scene",
        track_id=track,
        coordinate_frame="map",
        human_x=x,
        human_y=0.0,
    )


def test_sort_and_short_gap_interpolation() -> None:
    streams = resample_records([record(0.2, 0.2), record(0.0, 0.0)], target_hz=10.0)
    assert len(streams) == 1
    assert [row.timestamp for row in streams[0]] == pytest.approx([0.0, 0.1, 0.2])
    assert streams[0][1].human_x == pytest.approx(0.1)


def test_large_gap_is_never_interpolated() -> None:
    streams = resample_records(
        [record(0.0, 0.0), record(0.5, 0.5)],
        target_hz=10.0,
        max_interpolation_gap_seconds=0.3,
    )
    assert [len(stream) for stream in streams] == [1, 1]


def test_track_switch_is_never_interpolated() -> None:
    streams = resample_records([record(0.0, 0.0, "a"), record(0.1, 0.1, "b")])
    assert len(streams) == 2


def test_duplicate_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="重复 timestamp"):
        resample_records([record(0.0, 0.0), record(0.0, 0.1)])
