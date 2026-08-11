from types import SimpleNamespace

import pytest

from scripts.convert_rosbag_trajectory import _protect_bag


def test_converter_refuses_output_inside_rosbag(tmp_path) -> None:
    bag = tmp_path / "mock_bag_directory"
    bag.mkdir()
    args = SimpleNamespace(bag=bag, output=bag / "derived.csv", overwrite=False)
    with pytest.raises(ValueError, match="rosbag 目录内部"):
        _protect_bag(args)
