"""Labeled and variable-horizon views of the unchanged v1 synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.synthetic_trajectory import HISTORY_LENGTH, TRAJECTORY_TYPES

TYPE_NAMES = {
    "linear": "straight",
    "accelerate": "acceleration",
    "decelerate": "deceleration",
    "left_turn": "left_turn",
    "right_turn": "right_turn",
}
BASE_SPEED_RANGE = (0.04, 0.16)
TURN_ANGULAR_VELOCITY = 0.035
NOISE_STD = 0.005
TIMESTEP_FRAMES = 1


@dataclass(frozen=True)
class LabeledSplit:
    history: np.ndarray
    future: np.ndarray
    trajectory_type: np.ndarray
    noise: np.ndarray


@dataclass(frozen=True)
class LabeledSplits:
    train: LabeledSplit
    val: LabeledSplit
    test: LabeledSplit


def _generate_one(
    rng: np.random.Generator, kind: str, history_length: int, future_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce v1 exactly for 20+10 while allowing diagnostic horizons."""
    total = history_length + future_length
    position = rng.uniform(-5.0, 5.0, size=2)
    speed = rng.uniform(*BASE_SPEED_RANGE)
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
            angle += TURN_ANGULAR_VELOCITY
        elif kind == "right_turn":
            angle -= TURN_ANGULAR_VELOCITY
        position = position + current_speed * np.array([np.cos(angle), np.sin(angle)])
    clean = np.asarray(points, dtype=np.float32)
    noise = rng.normal(0.0, NOISE_STD, clean.shape).astype(np.float32)
    return clean + noise, noise


def generate_labeled_split(
    size: int, seed: int, history_length: int = HISTORY_LENGTH, future_length: int = 10
) -> LabeledSplit:
    if size <= 0 or history_length < 2 or future_length <= 0:
        raise ValueError("size/future_length 必须为正，history_length 必须至少为 2")
    rng = np.random.default_rng(seed)
    trajectories, noises, labels = [], [], []
    for index in range(size):
        kind = TRAJECTORY_TYPES[index % len(TRAJECTORY_TYPES)]
        trajectory, noise = _generate_one(rng, kind, history_length, future_length)
        trajectories.append(trajectory)
        noises.append(noise)
        labels.append(TYPE_NAMES[kind])
    trajectories_array = np.stack(trajectories)
    noise_array = np.stack(noises)
    labels_array = np.asarray(labels)
    # np.random.Generator.shuffle on axis 0 uses the same permutation operation.
    order = rng.permutation(size)
    trajectories_array = trajectories_array[order]
    return LabeledSplit(
        history=trajectories_array[:, :history_length],
        future=trajectories_array[:, history_length:],
        trajectory_type=labels_array[order],
        noise=noise_array[order],
    )


def create_labeled_splits(
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
    history_length: int = HISTORY_LENGTH,
    future_length: int = 10,
) -> LabeledSplits:
    sequence = np.random.SeedSequence(seed)
    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(3)]
    return LabeledSplits(
        train=generate_labeled_split(train_size, seeds[0], history_length, future_length),
        val=generate_labeled_split(val_size, seeds[1], history_length, future_length),
        test=generate_labeled_split(test_size, seeds[2], history_length, future_length),
    )


def _range(values: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(values)), "max": float(np.max(values))}


def dataset_statistics(splits: LabeledSplits) -> dict[str, Any]:
    all_points = []
    all_noise = []
    counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split = getattr(splits, split_name)
        all_points.append(np.concatenate((split.history, split.future), axis=1))
        all_noise.append(split.noise)
        counts[split_name] = {
            kind: int(np.sum(split.trajectory_type == kind))
            for kind in TYPE_NAMES.values()
        }
    trajectories = np.concatenate(all_points)
    velocity = np.diff(trajectories, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    angles = np.unwrap(np.arctan2(velocity[..., 1], velocity[..., 0]), axis=1)
    angular_velocity = np.diff(angles, axis=1)
    curvature = angular_velocity / np.maximum(speed[:, 1:], 1e-8)
    noise = np.concatenate(all_noise)
    total = trajectories.shape[1]
    theoretical_acceleration = (
        BASE_SPEED_RANGE[0] / (total - 1),
        BASE_SPEED_RANGE[1] / (total - 1),
    )
    return {
        "trajectory_type_counts": counts,
        "history_frames": int(splits.train.history.shape[1]),
        "future_frames": int(splits.train.future.shape[1]),
        "timestep_frames": TIMESTEP_FRAMES,
        "generator_configuration": {
            "base_speed_range_units_per_frame": list(BASE_SPEED_RANGE),
            "acceleration_magnitude_range_units_per_frame2": list(theoretical_acceleration),
            "turn_angular_velocity_rad_per_frame": {
                "left_turn": TURN_ANGULAR_VELOCITY,
                "right_turn": -TURN_ANGULAR_VELOCITY,
            },
            "turn_curvature_range_rad_per_unit": [
                TURN_ANGULAR_VELOCITY / BASE_SPEED_RANGE[1],
                TURN_ANGULAR_VELOCITY / BASE_SPEED_RANGE[0],
            ],
            "gaussian_noise_std_per_coordinate": NOISE_STD,
            "gaussian_noise_theoretical_range": "unbounded",
        },
        "empirical_with_noise": {
            "speed_units_per_frame": _range(speed),
            "acceleration_norm_units_per_frame2": _range(acceleration_norm),
            "signed_angular_velocity_rad_per_frame": _range(angular_velocity),
            "signed_curvature_rad_per_unit": _range(curvature),
            "sampled_noise_per_coordinate": _range(noise),
        },
    }


def as_tensor_dataset(split: LabeledSplit):
    import torch

    return torch.utils.data.TensorDataset(
        torch.from_numpy(split.history), torch.from_numpy(split.future)
    )
