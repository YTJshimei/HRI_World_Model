"""Build model-compatible windows only after group-level split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.data.ros_trajectory_loader import SplitManifest, validate_manifest_coverage
from src.data.ros_trajectory_schema import RosTrajectoryRecord, required_position_is_finite
from src.data.trajectory_resampler import (
    DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
    DEFAULT_TARGET_HZ,
    resample_records,
)

DEFAULT_HISTORY_LENGTH = 20
DEFAULT_FUTURE_LENGTH = 10


@dataclass(frozen=True)
class TrajectoryWindows:
    history: np.ndarray
    future: np.ndarray
    group_keys: tuple[tuple[str, str, str], ...]
    start_timestamps: np.ndarray

    def __len__(self) -> int:
        return int(self.history.shape[0])


def build_windows(
    records: Iterable[RosTrajectoryRecord],
    history_length: int = DEFAULT_HISTORY_LENGTH,
    future_length: int = DEFAULT_FUTURE_LENGTH,
    target_hz: float = DEFAULT_TARGET_HZ,
    max_interpolation_gap_seconds: float = DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
    stride: int = 1,
) -> TrajectoryWindows:
    if history_length < 2 or future_length <= 0 or stride <= 0:
        raise ValueError("history >= 2、future > 0、stride > 0")
    total = history_length + future_length
    histories, futures, keys, timestamps = [], [], [], []
    streams = resample_records(records, target_hz, max_interpolation_gap_seconds)
    for stream in streams:
        for start in range(0, len(stream) - total + 1, stride):
            rows = stream[start : start + total]
            if not all(required_position_is_finite(row) for row in rows):
                continue
            positions = np.asarray([[row.human_x, row.human_y] for row in rows], dtype=np.float32)
            histories.append(positions[:history_length])
            futures.append(positions[history_length:])
            keys.append(rows[0].group_key)
            timestamps.append(rows[0].timestamp)
    return TrajectoryWindows(
        history=np.stack(histories) if histories else np.empty((0, history_length, 2), dtype=np.float32),
        future=np.stack(futures) if futures else np.empty((0, future_length, 2), dtype=np.float32),
        group_keys=tuple(keys),
        start_timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def build_split_windows(
    records: list[RosTrajectoryRecord],
    manifest: SplitManifest,
    **window_options: object,
) -> dict[str, TrajectoryWindows]:
    """Partition raw records first, then independently resample and window each split."""
    validate_manifest_coverage(manifest, records)
    partitions = manifest.partition_records(records)
    windows = {name: build_windows(rows, **window_options) for name, rows in partitions.items()}
    trial_sets = {
        name: {key[0] for key in value.group_keys} for name, value in windows.items()
    }
    if (
        trial_sets["train"] & trial_sets["validation"]
        or trial_sets["train"] & trial_sets["test"]
        or trial_sets["validation"] & trial_sets["test"]
    ):
        raise AssertionError("内部错误：生成窗口后检测到 trial 跨 split")
    return windows


def as_tensor_dataset(windows: TrajectoryWindows):
    import torch

    return torch.utils.data.TensorDataset(
        torch.from_numpy(windows.history), torch.from_numpy(windows.future)
    )
