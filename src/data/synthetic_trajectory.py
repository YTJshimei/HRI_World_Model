"""Deterministic synthetic 2-D trajectories for Phase 2 development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HISTORY_LENGTH = 20
FUTURE_LENGTH = 10
TRAJECTORY_TYPES = ("linear", "accelerate", "decelerate", "left_turn", "right_turn")


def _one_trajectory(rng: np.random.Generator, kind: str) -> np.ndarray:
    total = HISTORY_LENGTH + FUTURE_LENGTH
    position = rng.uniform(-5.0, 5.0, size=2)
    speed = rng.uniform(0.04, 0.16)
    angle = rng.uniform(-np.pi, np.pi)
    points = []
    for frame in range(total):
        points.append(position.copy())
        progress = frame / max(total - 1, 1)
        if kind == "accelerate":
            current_speed = speed * (0.5 + progress)
        elif kind == "decelerate":
            current_speed = speed * (1.5 - progress)
        else:
            current_speed = speed
        if kind == "left_turn":
            angle += 0.035
        elif kind == "right_turn":
            angle -= 0.035
        position = position + current_speed * np.array([np.cos(angle), np.sin(angle)])
    trajectory = np.asarray(points, dtype=np.float32)
    trajectory += rng.normal(0.0, 0.005, trajectory.shape).astype(np.float32)
    return trajectory


def generate_split(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one split; distinct seeds make train/val/test independent."""
    if size <= 0:
        raise ValueError("size 必须为正整数")
    rng = np.random.default_rng(seed)
    trajectories = np.stack(
        [_one_trajectory(rng, TRAJECTORY_TYPES[index % len(TRAJECTORY_TYPES)]) for index in range(size)]
    )
    rng.shuffle(trajectories)
    return trajectories[:, :HISTORY_LENGTH], trajectories[:, HISTORY_LENGTH:]


@dataclass(frozen=True)
class TrajectorySplits:
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]


def create_splits(train_size: int = 4000, val_size: int = 500, test_size: int = 500, seed: int = 42) -> TrajectorySplits:
    """Create reproducible, strictly separate splits using independent RNG streams."""
    sequence = np.random.SeedSequence(seed)
    split_seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(3)]
    return TrajectorySplits(
        train=generate_split(train_size, split_seeds[0]),
        val=generate_split(val_size, split_seeds[1]),
        test=generate_split(test_size, split_seeds[2]),
    )


def as_tensor_dataset(split: tuple[np.ndarray, np.ndarray]):
    """Convert a NumPy split to a PyTorch TensorDataset without hiding dependency errors."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("训练需要已有的 PyTorch 环境；代码不会自动安装依赖。") from exc
    return torch.utils.data.TensorDataset(torch.from_numpy(split[0]), torch.from_numpy(split[1]))
