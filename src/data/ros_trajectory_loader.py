"""CSV persistence and leakage-safe split manifests for ROS trajectories."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data.ros_trajectory_schema import RosTrajectoryRecord


def load_trajectory_csv(path: str | Path) -> list[RosTrajectoryRecord]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"标准轨迹 CSV 不存在：{csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头：{csv_path}")
        missing = set(RosTrajectoryRecord.CSV_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV 缺少标准字段：{', '.join(sorted(missing))}")
        records = []
        for line_number, row in enumerate(reader, start=2):
            try:
                records.append(RosTrajectoryRecord.from_mapping(row))
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line_number}: {exc}") from exc
    return records


def write_trajectory_csv(path: str | Path, records: Iterable[RosTrajectoryRecord]) -> None:
    """Write derived data only; callers must never pass an original rosbag path."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RosTrajectoryRecord.CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {
                key: "" if value is None else value
                for key, value in record.to_mapping().items()
            }
            writer.writerow(row)


@dataclass(frozen=True)
class SplitManifest:
    train_trials: tuple[str, ...]
    validation_trials: tuple[str, ...]
    test_trials: tuple[str, ...]

    def __post_init__(self) -> None:
        named = {
            "train": self.train_trials,
            "validation": self.validation_trials,
            "test": self.test_trials,
        }
        for name, values in named.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name}_trials 内存在重复 trial")
        memberships: dict[str, list[str]] = {}
        for split_name, values in named.items():
            for trial in values:
                memberships.setdefault(trial, []).append(split_name)
        overlaps = {trial: splits for trial, splits in memberships.items() if len(splits) > 1}
        if overlaps:
            raise ValueError(f"trial 跨 split，禁止数据泄漏：{overlaps}")

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = ("train_trials", "validation_trials", "test_trials")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"split manifest 缺少字段：{', '.join(missing)}")
        return cls(*(tuple(str(item) for item in payload[key]) for key in required))

    def save(self, path: str | Path) -> None:
        payload = {
            "train_trials": list(self.train_trials),
            "validation_trials": list(self.validation_trials),
            "test_trials": list(self.test_trials),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def split_for_trial(self, trial_id: str) -> str:
        for name, trials in (
            ("train", self.train_trials),
            ("validation", self.validation_trials),
            ("test", self.test_trials),
        ):
            if trial_id in trials:
                return name
        raise ValueError(f"trial 未出现在 split manifest：{trial_id}")

    def partition_records(
        self, records: Iterable[RosTrajectoryRecord]
    ) -> dict[str, list[RosTrajectoryRecord]]:
        partitions = {"train": [], "validation": [], "test": []}
        for record in records:
            partitions[self.split_for_trial(record.trial_id)].append(record)
        return partitions


def validate_manifest_coverage(
    manifest: SplitManifest, records: Iterable[RosTrajectoryRecord]
) -> None:
    observed = {record.trial_id for record in records}
    declared = set(manifest.train_trials + manifest.validation_trials + manifest.test_trials)
    missing = observed - declared
    unknown = declared - observed
    if missing or unknown:
        raise ValueError(
            f"split manifest 与数据不一致：未分配={sorted(missing)}，无数据={sorted(unknown)}"
        )
