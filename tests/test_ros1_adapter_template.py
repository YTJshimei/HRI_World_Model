from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

from ros1.phase2c_human_pose_adapter import AdapterStatistics, choose_timestamp


class Stamp:
    def __init__(self, value: float) -> None:
        self.value = value

    def to_sec(self) -> float:
        return self.value


def test_adapter_timestamp_is_explicit() -> None:
    receipt = Stamp(5.0)
    header_message = SimpleNamespace(
        header=SimpleNamespace(stamp=Stamp(4.0), frame_id="sensor")
    )
    header_stamp, header_source = choose_timestamp(header_message, receipt)
    assert header_stamp.to_sec() == 4.0
    assert header_source == "header_stamp"
    receipt_stamp, receipt_source = choose_timestamp(SimpleNamespace(), receipt)
    assert receipt_stamp is receipt
    assert receipt_source == "receipt_timestamp"


def test_adapter_statistics_include_tf_failure_reasons() -> None:
    summary = AdapterStatistics(tf_lookup_failure=2, invalid_target=1).summary()
    assert "tf_lookup_failure=2" in summary
    assert "invalid_target=1" in summary


def test_launch_is_read_only_and_contains_required_parameters() -> None:
    launch_path = Path(__file__).parents[1] / "ros1" / "phase2c_data_adapter.launch"
    root = ElementTree.parse(launch_path).getroot()
    text = launch_path.read_text(encoding="utf-8")
    assert root.tag == "launch"
    for name in (
        "source_frame", "target_frame", "input_topic", "pose_output_topic", "track_output_topic"
    ):
        assert f'name="{name}"' in text
    assert "/cmd_vel" not in text
