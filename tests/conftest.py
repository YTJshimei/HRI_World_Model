from pathlib import Path

import pytest

from src.data.ros_trajectory_loader import SplitManifest, write_trajectory_csv
from src.data.ros_trajectory_schema import RosTrajectoryRecord


@pytest.fixture
def mock_ros_trajectory_files(tmp_path: Path) -> tuple[Path, Path]:
    """Small synthetic ROS-like fixture; never a formal experiment dataset."""
    records = []
    for trial_index, trial_id in enumerate(("mock_train", "mock_val", "mock_test")):
        for frame in range(35):
            timestamp = frame * 0.1
            human_x = trial_index + 0.05 * frame
            human_y = 0.1 * trial_index
            records.append(
                RosTrajectoryRecord(
                    timestamp=timestamp,
                    trial_id=trial_id,
                    session_id=f"mock_session_{trial_index}",
                    person_id_anonymous=f"mock_person_{trial_index}",
                    scene_id="mock_scene",
                    track_id="mock_track_0",
                    coordinate_frame="map",
                    human_x=human_x,
                    human_y=human_y,
                    human_z=0.9,
                    human_vx=0.5,
                    human_vy=0.0,
                    robot_x=0.0,
                    robot_y=0.0,
                    robot_yaw=0.0,
                )
            )
    csv_path = tmp_path / "mock_only_not_real.csv"
    manifest_path = tmp_path / "mock_split.json"
    write_trajectory_csv(csv_path, records)
    SplitManifest(("mock_train",), ("mock_val",), ("mock_test",)).save(manifest_path)
    return csv_path, manifest_path
