import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.convert_ros1_bag_trajectory import (
    convert_samples,
    parse_human_message,
    protect_input_bag,
    read_ros1_bag,
)
from src.data.data_quality import inspect_data_quality
from src.data.ros_trajectory_loader import load_trajectory_csv, write_trajectory_csv
from src.data.trajectory_window_builder import build_windows


class Stamp:
    def __init__(self, value: float) -> None:
        self.value = value

    def to_sec(self) -> float:
        return self.value


def vector(x: float, y: float, z: float = 0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def human_target(timestamp: float, valid: bool = True):
    del timestamp
    return SimpleNamespace(
        track_id=17,
        position=vector(1.0, 2.0, 0.8),
        distance=2.5,
        bearing=0.25,
        confidence=0.9,
        locked=True,
        is_valid=valid,
    )


def converter_args(tmp_path: Path):
    return SimpleNamespace(
        bag=tmp_path / "mock.bag",
        output=tmp_path / "mock.csv",
        human_topic="/human/target",
        human_message_type="auto",
        timestamp_policy="auto",
        header_stamp_semantics="sensor_header",
        source_frame="map",
        coordinate_frame="map",
        robot_odom_topic=None,
        cmd_vel_topic=None,
        track_id_topic=None,
        requested_action_topic=None,
        validated_action_topic=None,
        executed_action_topic=None,
        trial_id="mock_trial",
        session_id="mock_session",
        person_id_anonymous="mock_person",
        scene_id="mock_scene",
        default_track_id="track_0",
        max_sync_gap=0.05,
    )


def test_headerless_human_target_uses_bag_timestamp_and_parses_fields() -> None:
    observation = parse_human_message(
        human_target(12.5), "follow_msgs/HumanTarget", 12.5, "base_link"
    )
    assert observation.timestamp == pytest.approx(12.5)
    assert observation.timestamp_source == "bag_message_timestamp"
    assert observation.track_id == "17"
    assert observation.distance == pytest.approx(2.5)
    assert observation.bearing == pytest.approx(0.25)
    assert observation.confidence == pytest.approx(0.9)
    assert observation.locked is True
    assert observation.is_valid is True


def test_headerless_human_target_rejects_missing_source_frame() -> None:
    with pytest.raises(ValueError, match="source_frame 缺失"):
        parse_human_message(human_target(1.0), "follow_msgs/HumanTarget", 1.0, "")


def test_pose_stamped_uses_real_header_stamp() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=Stamp(3.25), frame_id="map"),
        pose=SimpleNamespace(position=vector(1.0, 0.0, 0.5)),
    )
    observation = parse_human_message(
        message, "geometry_msgs/PoseStamped", 3.3, "map", "auto"
    )
    assert observation.timestamp == pytest.approx(3.25)
    assert observation.timestamp_source == "header_stamp"


def test_ros1_conversion_is_compatible_with_quality_resampler_and_windows(tmp_path: Path) -> None:
    args = converter_args(tmp_path)
    samples = {
        "/human/target": [
            (index * 0.1, human_target(index * 0.1)) for index in range(30)
        ]
    }
    records, metadata = convert_samples(
        args, {"/human/target": "follow_msgs/HumanTarget"}, samples
    )
    assert len(records) == 30
    assert metadata["timestamp_source"] == "bag_message_timestamp"
    assert metadata["parsed_human_target_fields"][-1] == "is_valid"
    write_trajectory_csv(args.output, records)
    loaded = load_trajectory_csv(args.output)
    report = inspect_data_quality(loaded, target_hz=10.0)
    windows = build_windows(loaded)
    assert report.statistics["record_count"] == 30
    assert windows.history.shape == (1, 20, 2)
    assert windows.future.shape == (1, 10, 2)


def test_invalid_human_target_is_not_exported(tmp_path: Path) -> None:
    args = converter_args(tmp_path)
    samples = {"/human/target": [(0.0, human_target(0.0, valid=False))]}
    records, metadata = convert_samples(
        args, {"/human/target": "follow_msgs/HumanTarget"}, samples
    )
    assert records == []
    assert metadata["invalid_human_target_count"] == 1


def test_missing_ros1_runtime_has_clear_error(tmp_path: Path) -> None:
    if importlib.util.find_spec("rosbag") is not None:
        pytest.skip("ROS1 runtime is installed in this test environment")
    with pytest.raises(RuntimeError, match="ROS1 rosbag Python 模块"):
        read_ros1_bag(converter_args(tmp_path))


def test_converter_never_overwrites_bag_or_csv_with_metadata(tmp_path: Path) -> None:
    args = converter_args(tmp_path)
    args.bag.write_bytes(b"mock; not opened")
    args.metadata_output = args.output
    args.overwrite = False
    with pytest.raises(ValueError, match="不同文件"):
        protect_input_bag(args)
