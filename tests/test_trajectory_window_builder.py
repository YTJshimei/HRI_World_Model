from pathlib import Path

from src.data.ros_trajectory_loader import SplitManifest, load_trajectory_csv
from src.data.trajectory_window_builder import build_split_windows


def test_split_before_windowing_and_default_shapes(mock_ros_trajectory_files: tuple[Path, Path]) -> None:
    csv_path, manifest_path = mock_ros_trajectory_files
    records = load_trajectory_csv(csv_path)
    windows = build_split_windows(records, SplitManifest.load(manifest_path))
    assert windows["train"].history.shape == (6, 20, 2)
    assert windows["train"].future.shape == (6, 10, 2)
    assert {key[0] for key in windows["train"].group_keys} == {"mock_train"}
    assert {key[0] for key in windows["validation"].group_keys} == {"mock_val"}
    assert {key[0] for key in windows["test"].group_keys} == {"mock_test"}
