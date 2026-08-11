"""Deterministic synthetic COCO-17 3-D skeleton sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.skeleton_schema import (
    DEFAULT_FUTURE_FRAMES,
    DEFAULT_HISTORY_FRAMES,
    DEFAULT_SAMPLE_RATE_HZ,
    NUM_JOINTS,
    global_to_local,
    joint_ids,
)

ACTION_TYPES = (
    "walk",
    "fast_walk",
    "run",
    "acceleration",
    "deceleration",
    "left_turn",
    "right_turn",
    "stop",
    "lateral_motion",
)


@dataclass(frozen=True)
class SkeletonSplit:
    history_global: np.ndarray
    future_global: np.ndarray
    root_global: np.ndarray
    joint_local: np.ndarray
    confidence: np.ndarray
    visibility_mask: np.ndarray
    action_type: np.ndarray

    def __len__(self) -> int:
        return int(self.history_global.shape[0])


@dataclass(frozen=True)
class SkeletonSplits:
    train: SkeletonSplit
    val: SkeletonSplit
    test: SkeletonSplit


def _base_pose(scale: float) -> np.ndarray:
    pose = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    coordinates = {
        "left_hip": (0.0, 0.11, 0.0),
        "right_hip": (0.0, -0.11, 0.0),
        "left_shoulder": (0.0, 0.20, 0.52),
        "right_shoulder": (0.0, -0.20, 0.52),
        "nose": (0.025, 0.0, 0.76),
        "left_eye": (0.035, 0.035, 0.80),
        "right_eye": (0.035, -0.035, 0.80),
        "left_ear": (0.0, 0.085, 0.78),
        "right_ear": (0.0, -0.085, 0.78),
    }
    for name, xyz in coordinates.items():
        pose[joint_ids[name]] = xyz
    return pose * scale


def _motion_parameters(action: str, progress: float) -> tuple[float, float, float]:
    """Return speed (m/s), cadence (Hz), yaw rate (rad/s)."""
    if action == "walk":
        return 0.70, 1.6, 0.0
    if action == "fast_walk":
        return 1.20, 2.0, 0.0
    if action == "run":
        return 2.00, 2.8, 0.0
    if action == "acceleration":
        return 0.30 + 1.20 * progress, 1.2 + 1.1 * progress, 0.0
    if action == "deceleration":
        return 1.50 - 1.20 * progress, 2.3 - 1.1 * progress, 0.0
    if action == "left_turn":
        return 0.85, 1.7, 0.50
    if action == "right_turn":
        return 0.85, 1.7, -0.50
    if action == "stop":
        factor = max(0.0, 1.0 - 2.0 * progress)
        return 0.80 * factor, 1.5 * factor, 0.0
    return 0.65, 1.5, 0.0


def _articulated_local_pose(
    scale: float, phase: float, action: str, speed: float
) -> np.ndarray:
    pose = _base_pose(scale)
    amplitude = 0.18 if action in ("walk", "left_turn", "right_turn", "lateral_motion") else 0.27
    if action == "fast_walk":
        amplitude = 0.25
    elif action == "run":
        amplitude = 0.42
    elif action == "stop":
        amplitude *= min(1.0, speed / 0.4)
    elif action in ("acceleration", "deceleration"):
        amplitude = 0.16 + 0.12 * min(speed / 1.5, 1.0)
    upper_leg = 0.43 * scale
    lower_leg = 0.43 * scale
    upper_arm = 0.29 * scale
    lower_arm = 0.26 * scale
    for side, offset in (("left", 0.0), ("right", np.pi)):
        leg_angle = amplitude * np.sin(phase + offset)
        knee_bend = 0.18 * max(0.0, -np.sin(phase + offset))
        hip = pose[joint_ids[f"{side}_hip"]]
        knee = hip + np.array(
            [upper_leg * np.sin(leg_angle), 0.0, -upper_leg * np.cos(leg_angle)]
        )
        ankle_angle = leg_angle - knee_bend
        ankle = knee + np.array(
            [lower_leg * np.sin(ankle_angle), 0.0, -lower_leg * np.cos(ankle_angle)]
        )
        pose[joint_ids[f"{side}_knee"]] = knee
        pose[joint_ids[f"{side}_ankle"]] = ankle

        arm_angle = -0.75 * leg_angle
        shoulder = pose[joint_ids[f"{side}_shoulder"]]
        elbow = shoulder + np.array(
            [upper_arm * np.sin(arm_angle), 0.0, -upper_arm * np.cos(arm_angle)]
        )
        wrist = elbow + np.array(
            [lower_arm * np.sin(arm_angle), 0.0, -lower_arm * np.cos(arm_angle)]
        )
        pose[joint_ids[f"{side}_elbow"]] = elbow
        pose[joint_ids[f"{side}_wrist"]] = wrist
    return pose


def _generate_clean_sequence(
    rng: np.random.Generator,
    action: str,
    total_frames: int,
    sample_rate_hz: float,
) -> np.ndarray:
    timestep = 1.0 / sample_rate_hz
    scale = rng.uniform(0.90, 1.10)
    root = np.array(
        [rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(0.88, 1.02)],
        dtype=np.float64,
    )
    yaw = rng.uniform(-np.pi, np.pi)
    phase = rng.uniform(0.0, 2 * np.pi)
    frames = []
    for frame in range(total_frames):
        progress = frame / max(total_frames - 1, 1)
        speed, cadence, yaw_rate = _motion_parameters(action, progress)
        local = _articulated_local_pose(scale, phase, action, speed)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        world = local.copy()
        world[:, :2] = local[:, :2] @ rotation.T
        world += root
        frames.append(world)
        if action == "lateral_motion":
            direction = np.array([-np.sin(yaw), np.cos(yaw)])
        else:
            direction = np.array([np.cos(yaw), np.sin(yaw)])
        root[:2] += speed * timestep * direction
        yaw += yaw_rate * timestep
        phase += 2 * np.pi * cadence * timestep
    return np.asarray(frames, dtype=np.float32)


def generate_skeleton_split(
    size: int,
    seed: int,
    history_frames: int = DEFAULT_HISTORY_FRAMES,
    future_frames: int = DEFAULT_FUTURE_FRAMES,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    noise_std: float = 0.005,
    occlusion_rate: float = 0.10,
) -> SkeletonSplit:
    if size <= 0 or history_frames < 2 or future_frames <= 0 or sample_rate_hz <= 0:
        raise ValueError("size/future/sample_rate 必须为正，history 至少为 2")
    if noise_std < 0 or not 0.0 <= occlusion_rate < 1.0:
        raise ValueError("noise_std 必须非负，occlusion_rate 必须位于 [0, 1)")
    rng = np.random.default_rng(seed)
    total = history_frames + future_frames
    clean_sequences, action_types = [], []
    for index in range(size):
        action = ACTION_TYPES[index % len(ACTION_TYPES)]
        clean_sequences.append(
            _generate_clean_sequence(rng, action, total, sample_rate_hz)
        )
        action_types.append(action)
    clean = np.stack(clean_sequences)
    history = clean[:, :history_frames].copy()
    if noise_std:
        history += rng.normal(0.0, noise_std, history.shape).astype(np.float32)
    visibility = rng.random((size, history_frames, NUM_JOINTS)) >= occlusion_rate
    confidence = rng.uniform(0.80, 1.0, visibility.shape).astype(np.float32)
    confidence[~visibility] = 0.0
    root, local = global_to_local(history)
    order = rng.permutation(size)
    return SkeletonSplit(
        history_global=history[order].astype(np.float32),
        future_global=clean[order, history_frames:].astype(np.float32),
        root_global=root[order].astype(np.float32),
        joint_local=local[order].astype(np.float32),
        confidence=confidence[order],
        visibility_mask=visibility[order],
        action_type=np.asarray(action_types)[order],
    )


def create_skeleton_splits(
    train_size: int = 1800,
    val_size: int = 270,
    test_size: int = 270,
    seed: int = 42,
    **generator_options: float | int,
) -> SkeletonSplits:
    sequence = np.random.SeedSequence(seed)
    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(3)]
    return SkeletonSplits(
        train=generate_skeleton_split(train_size, seeds[0], **generator_options),
        val=generate_skeleton_split(val_size, seeds[1], **generator_options),
        test=generate_skeleton_split(test_size, seeds[2], **generator_options),
    )


def as_tensor_dataset(split: SkeletonSplit):
    import torch

    action_to_id = {name: index for index, name in enumerate(ACTION_TYPES)}
    action_ids = np.asarray([action_to_id[name] for name in split.action_type], dtype=np.int64)
    return torch.utils.data.TensorDataset(
        torch.from_numpy(split.history_global),
        torch.from_numpy(split.future_global),
        torch.from_numpy(split.confidence),
        torch.from_numpy(split.visibility_mask),
        torch.from_numpy(action_ids),
    )
