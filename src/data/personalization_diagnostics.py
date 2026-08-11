"""Synthetic-only profile/response diagnostics for Phase 4B.5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.personal_interaction_memory import (
    PersonalInteractionCorpus,
    PersonalInteractionRecord,
    _concatenate_splits,
)
from src.data.robot_action_schema import ACTION_DEFINITIONS, PHASE4A_ACTIONS, RobotAction
from src.data.skeleton_schema import compute_root, global_to_local
from src.data.synthetic_interaction import (
    InteractionSplit,
    VirtualPersonProfile,
    _robot_history_for_human,
    _supervision_mask,
    simulate_interaction_future,
)
from src.data.synthetic_skeleton import generate_skeleton_split


PROFILE_PARAMETER_NAMES = (
    "preferred_distance",
    "distance_sensitivity",
    "speed_response_gain",
    "response_delay",
    "lateral_avoidance_gain",
    "turn_sensitivity",
    "adaptation_rate",
)
TRUE_EFFECT_DESCRIPTOR_DIM = 10


RESPONSE_COVERED_PROFILES = (
    VirtualPersonProfile(100, "covered_low", 1.20, 0.25, 0.20, 0.65, 0.15, 0.18, 0.8),
    VirtualPersonProfile(101, "covered_medium_low", 1.85, 0.50, 0.40, 0.50, 0.35, 0.30, 1.3),
    VirtualPersonProfile(102, "covered_medium", 1.10, 0.78, 0.65, 0.42, 0.55, 0.48, 1.9),
    VirtualPersonProfile(103, "covered_medium_high", 1.95, 1.05, 0.95, 0.32, 0.85, 0.70, 2.6),
    VirtualPersonProfile(104, "covered_high", 1.25, 1.45, 1.30, 0.25, 1.10, 0.95, 3.1),
)


def profile_parameter_vector(profile: VirtualPersonProfile) -> np.ndarray:
    return np.asarray(
        [getattr(profile, name) for name in PROFILE_PARAMETER_NAMES], dtype=np.float32
    )


def true_effect_descriptor(
    human_history: np.ndarray,
    robot_history: np.ndarray,
    action: int,
    profile: VirtualPersonProfile,
    future_frames: int = 10,
    sample_rate_hz: float = 10.0,
) -> np.ndarray:
    """Context-specific simulator coefficients, never final future coordinates."""
    definition = ACTION_DEFINITIONS[RobotAction(int(action))]
    roots = compute_root(human_history)
    velocity = (roots[-1, :2] - roots[-2, :2]) * sample_rate_hz
    speed = float(np.linalg.norm(velocity))
    if speed > 1e-6:
        forward = velocity / speed
    else:
        forward = np.asarray(
            (np.cos(robot_history[-1, 2]), np.sin(robot_history[-1, 2])),
            dtype=np.float32,
        )
    robot_to_human = roots[-1, :2] - robot_history[-1, :2]
    norm = float(np.linalg.norm(robot_to_human))
    away = robot_to_human / norm if norm > 1e-6 else forward
    lateral = np.asarray((-away[1], away[0]), dtype=np.float32)
    bearing_sign = 1.0 if robot_history[-1, 6] >= 0.0 else -1.0
    pressure = -definition.distance_offset_m
    speed_scalar = profile.speed_response_gain * definition.speed_scale_delta * max(speed, 0.35)
    distance_scalar = profile.distance_sensitivity * pressure
    lateral_scalar = profile.lateral_avoidance_gain * pressure * bearing_sign * 0.45
    response_velocity = speed_scalar * forward + distance_scalar * away + lateral_scalar * lateral
    yaw_scalar = profile.turn_sensitivity * (definition.speed_scale_delta + pressure) * 0.20
    delay_frames = int(np.ceil(profile.response_delay * sample_rate_hz - 1e-9))
    active_time_end = max(0.0, (future_frames - delay_frames) / sample_rate_hz)
    adaptation_end = 1.0 - np.exp(-profile.adaptation_rate * active_time_end)
    if int(action) == int(RobotAction.KEEP):
        response_velocity = np.zeros(2, dtype=np.float32)
        speed_scalar = distance_scalar = lateral_scalar = yaw_scalar = 0.0
    return np.asarray(
        (
            response_velocity[0], response_velocity[1], speed_scalar,
            distance_scalar, lateral_scalar, yaw_scalar, profile.response_delay,
            adaptation_end, profile.adaptation_rate, float(int(action) != 0),
        ),
        dtype=np.float32,
    )


def descriptors_for_split(
    split: InteractionSplit,
    profiles: dict[int, VirtualPersonProfile],
    sample_rate_hz: float = 10.0,
) -> np.ndarray:
    descriptors = np.zeros(
        (len(split), split.candidate_actions.shape[1], TRUE_EFFECT_DESCRIPTOR_DIM),
        dtype=np.float32,
    )
    for sample in range(len(split)):
        profile = profiles[int(split.person_profile_id[sample])]
        for action_index, action in enumerate(split.candidate_actions[sample]):
            descriptors[sample, action_index] = true_effect_descriptor(
                split.human_history[sample], split.robot_history[sample], int(action),
                profile, split.future_by_action.shape[2], sample_rate_hz,
            )
    return descriptors


def _custom_interaction_split(
    size: int,
    seed: int,
    split_kind: str,
    profile: VirtualPersonProfile,
    history_frames: int,
    future_frames: int,
    sample_rate_hz: float,
    noise_std: float,
    occlusion_rate: float,
    state_mode: str,
    mask_unseen_combinations: bool,
) -> InteractionSplit:
    base = generate_skeleton_split(
        size, seed, history_frames, future_frames, sample_rate_hz,
        noise_std, occlusion_rate,
    )
    rng = np.random.default_rng(seed + 44_021)
    actions = np.asarray([int(action) for action in PHASE4A_ACTIONS], dtype=np.int64)
    robots, futures, effects, robot_futures, distances, masks = [], [], [], [], [], []
    for index in range(size):
        robot = _robot_history_for_human(
            base.history_global[index], rng, sample_rate_hz, state_mode
        )
        simulations = [
            simulate_interaction_future(
                base.history_global[index], base.future_global[index], robot,
                int(action), profile, sample_rate_hz,
            )
            for action in actions
        ]
        robots.append(robot)
        futures.append(np.stack([item.future_global for item in simulations]))
        effects.append(np.stack([item.action_effect for item in simulations]))
        robot_futures.append(np.stack([item.robot_future_xy for item in simulations]))
        distances.append(np.stack([item.future_human_robot_distance for item in simulations]))
        masks.append(
            _supervision_mask(str(base.action_type[index]))
            if mask_unseen_combinations else np.ones(len(actions), dtype=bool)
        )
    return InteractionSplit(
        human_history=base.history_global,
        confidence=base.confidence,
        visibility_mask=base.visibility_mask,
        robot_history=np.stack(robots),
        person_profile_id=np.full(size, profile.profile_id, dtype=np.int64),
        candidate_actions=np.broadcast_to(actions, (size, len(actions))).copy(),
        natural_future=base.future_global,
        future_by_action=np.stack(futures),
        action_effect_by_action=np.stack(effects),
        robot_future_xy_by_action=np.stack(robot_futures),
        future_human_robot_distance=np.stack(distances),
        action_supervision_mask=np.stack(masks),
        action_type=base.action_type,
        initial_state_id=np.asarray(
            [f"synthetic_{split_kind}_{seed}_{index:06d}" for index in range(size)]
        ),
        split_kind=split_kind,
    )


def generate_covered_personal_corpus(
    profiles: tuple[VirtualPersonProfile, ...] = RESPONSE_COVERED_PROFILES,
    persons_per_profile: int = 2,
    interactions_per_person: int = 30,
    query_start: int = 10,
    seed: int = 42,
    split_label: str = "response_covered_train",
    history_frames: int = 20,
    future_frames: int = 10,
    sample_rate_hz: float = 10.0,
    noise_std: float = 0.005,
    occlusion_rate: float = 0.10,
) -> PersonalInteractionCorpus:
    """Generate richer training profiles without using held-out profile IDs 5/6."""
    if any(profile.profile_id in (5, 6) for profile in profiles):
        raise ValueError("response-covered training must not include test profiles 5/6")
    sequence = iter(np.random.SeedSequence(seed).spawn(len(profiles) * persons_per_profile))
    parts, records, person_ids, orders = [], [], [], []
    global_row = 0
    for profile in profiles:
        for person_number in range(persons_per_profile):
            person_id = f"{split_label}_profile{profile.profile_id}_person{person_number}"
            part_seed = int(next(sequence).generate_state(1)[0])
            part = _custom_interaction_split(
                interactions_per_person, part_seed, person_id, profile,
                history_frames, future_frames, sample_rate_hz, noise_std,
                occlusion_rate, "seen", True,
            )
            parts.append(part)
            for order in range(interactions_per_person):
                action_index = order % len(PHASE4A_ACTIONS)
                root, local = global_to_local(part.human_history[order])
                state = np.concatenate(
                    (root[-1], (root[-1] - root[-2]) * sample_rate_hz)
                ).astype(np.float32)
                records.append(PersonalInteractionRecord(
                    person_profile_id=profile.profile_id,
                    person_instance_id=person_id,
                    interaction_id=f"{person_id}_interaction_{order:04d}",
                    timestamp=float(order * (history_frames + future_frames) / sample_rate_hz),
                    order_index=order,
                    human_state_before=state,
                    human_root_history=root.astype(np.float32),
                    human_local_pose_history=local.astype(np.float32),
                    robot_history=part.robot_history[order].astype(np.float32),
                    executed_action=int(part.candidate_actions[order, action_index]),
                    human_future_response=part.future_by_action[order, action_index].astype(np.float32),
                    action_effect=part.action_effect_by_action[order, action_index].astype(np.float32),
                    human_robot_distance_before=float(part.robot_history[order, -1, 5]),
                    human_robot_distance_after=float(part.future_human_robot_distance[order, action_index, -1]),
                    response_delay_observed=float(profile.response_delay),
                    split_kind=split_label,
                    source_row=global_row,
                ))
                person_ids.append(person_id)
                orders.append(order)
                global_row += 1
    merged = _concatenate_splits(parts, split_label)
    order_array = np.asarray(orders, dtype=np.int64)
    return PersonalInteractionCorpus(
        split=merged,
        records=tuple(records),
        person_instance_ids=np.asarray(person_ids),
        order_indices=order_array,
        query_indices=np.flatnonzero(order_array >= query_start),
        split_label=split_label,
    )
