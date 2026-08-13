"""Synthetic-only human response simulator and grouped counterfactual dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.robot_action_schema import (
    ACTION_DEFINITIONS, HOLD_ACTION_ID, PHASE4A_ACTIONS, RobotAction,
)
from src.data.skeleton_schema import compute_root, global_to_local
from src.data.synthetic_skeleton import SkeletonSplit, generate_skeleton_split


@dataclass(frozen=True)
class VirtualPersonProfile:
    profile_id: int
    name: str
    preferred_distance: float
    distance_sensitivity: float
    speed_response_gain: float
    response_delay: float
    lateral_avoidance_gain: float
    turn_sensitivity: float
    adaptation_rate: float


VIRTUAL_PERSON_PROFILES = (
    VirtualPersonProfile(0, "balanced", 1.35, 0.75, 0.55, 0.30, 0.35, 0.30, 2.0),
    VirtualPersonProfile(1, "cautious", 1.75, 1.25, 0.35, 0.20, 0.70, 0.45, 2.8),
    VirtualPersonProfile(2, "speed_follower", 1.30, 0.45, 1.15, 0.40, 0.25, 0.25, 1.8),
    VirtualPersonProfile(3, "slow_adapter", 1.50, 0.85, 0.50, 0.70, 0.45, 0.40, 0.9),
    VirtualPersonProfile(4, "lateral_avoider", 1.55, 0.95, 0.45, 0.30, 1.20, 0.55, 2.3),
    VirtualPersonProfile(5, "turn_sensitive", 1.40, 0.70, 0.65, 0.50, 0.55, 1.20, 1.7),
    VirtualPersonProfile(6, "highly_adaptive", 1.65, 1.10, 0.85, 0.20, 0.80, 0.75, 3.5),
)
PROFILE_BY_ID = {profile.profile_id: profile for profile in VIRTUAL_PERSON_PROFILES}
SEEN_PROFILE_IDS = (0, 1, 2, 3, 4)
UNSEEN_PROFILE_IDS = (5, 6)


@dataclass(frozen=True)
class InteractionSimulation:
    future_global: np.ndarray
    future_root: np.ndarray
    future_local: np.ndarray
    action_effect: np.ndarray
    robot_future_xy: np.ndarray
    future_human_robot_distance: np.ndarray
    response_delay_frames: int


@dataclass(frozen=True)
class InteractionSplit:
    human_history: np.ndarray
    confidence: np.ndarray
    visibility_mask: np.ndarray
    robot_history: np.ndarray
    person_profile_id: np.ndarray
    candidate_actions: np.ndarray
    natural_future: np.ndarray
    future_by_action: np.ndarray
    action_effect_by_action: np.ndarray
    robot_future_xy_by_action: np.ndarray
    future_human_robot_distance: np.ndarray
    action_supervision_mask: np.ndarray
    action_type: np.ndarray
    initial_state_id: np.ndarray
    split_kind: str

    def __len__(self) -> int:
        return int(self.human_history.shape[0])


@dataclass(frozen=True)
class InteractionSplits:
    train: InteractionSplit
    val: InteractionSplit
    test_seen_person_seen_context: InteractionSplit
    test_unseen_interaction_state: InteractionSplit
    test_unseen_person_profile: InteractionSplit
    test_unseen_action_context: InteractionSplit


@dataclass(frozen=True)
class AdverseResponseRiskFactors:
    """Continuous simulator dynamics; never a label or profile-ID lookup.

    Values are sampled independently for each initial state and modulate the
    actual future trajectory.  A downstream protocol may label only measured
    trajectory outcomes, never these factors themselves.
    """

    braking_susceptibility: float
    lateral_startle_gain: float
    heading_startle_gain: float
    approach_sensitivity: float
    onset_delay_seconds: float
    recovery_rate: float

    def __post_init__(self) -> None:
        values = np.asarray(tuple(self.__dict__.values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("adverse-response risk factors must be finite and non-negative")


@dataclass(frozen=True)
class RiskConditionedInteractionSimulation:
    future_global: np.ndarray
    future_root: np.ndarray
    future_local: np.ndarray
    action_effect: np.ndarray
    robot_future_xy: np.ndarray
    future_human_robot_distance: np.ndarray
    risk_factors: AdverseResponseRiskFactors
    risk_dynamics_applied: bool


RISK_FACTOR_NAMES = tuple(AdverseResponseRiskFactors.__dataclass_fields__)


def sample_adverse_response_risk_factors(
    rng: np.random.Generator,
) -> AdverseResponseRiskFactors:
    """Sample independent continuous susceptibility from declared support."""
    return AdverseResponseRiskFactors(
        braking_susceptibility=float(rng.uniform(0.15, 1.35)),
        lateral_startle_gain=float(rng.uniform(0.10, 1.25)),
        heading_startle_gain=float(rng.uniform(0.10, 1.20)),
        approach_sensitivity=float(rng.uniform(0.35, 1.30)),
        onset_delay_seconds=float(rng.uniform(0.10, 0.40)),
        recovery_rate=float(rng.uniform(2.0, 4.5)),
    )


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else fallback.copy()


def _rotate_local(local: np.ndarray, angles: np.ndarray) -> np.ndarray:
    result = local.copy()
    cosine, sine = np.cos(angles), np.sin(angles)
    x, y = local[..., 0], local[..., 1]
    result[..., 0] = cosine[:, None] * x - sine[:, None] * y
    result[..., 1] = sine[:, None] * x + cosine[:, None] * y
    return result


def simulate_interaction_future(
    human_history: np.ndarray,
    natural_future: np.ndarray,
    robot_history: np.ndarray,
    action: int | RobotAction,
    profile: VirtualPersonProfile,
    sample_rate_hz: float = 10.0,
) -> InteractionSimulation:
    """Apply a delayed residual response to the natural skeleton forecast."""
    if int(action) == HOLD_ACTION_ID:
        from src.data.hold_candidate import simulate_hold_interaction_future
        return simulate_hold_interaction_future(
            human_history, natural_future, robot_history, profile, None, sample_rate_hz
        )
    action_id = RobotAction(int(action))
    definition = ACTION_DEFINITIONS[action_id]
    history_root = compute_root(human_history)
    natural_root, natural_local = global_to_local(natural_future)
    future_frames = natural_future.shape[0]
    timestep = 1.0 / sample_rate_hz
    delay_frames = int(np.ceil(profile.response_delay * sample_rate_hz - 1e-9))

    velocity = (history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz
    speed = float(np.linalg.norm(velocity))
    forward = _unit(
        velocity,
        np.asarray((np.cos(robot_history[-1, 2]), np.sin(robot_history[-1, 2]))),
    )
    robot_to_human = history_root[-1, :2] - robot_history[-1, :2]
    away = _unit(robot_to_human, forward)
    lateral = np.asarray((-away[1], away[0]))
    bearing_sign = 1.0 if robot_history[-1, 6] >= 0.0 else -1.0

    pressure = -definition.distance_offset_m
    speed_response = (
        profile.speed_response_gain
        * definition.speed_scale_delta
        * max(speed, 0.35)
        * forward
    )
    distance_response = profile.distance_sensitivity * pressure * away
    lateral_response = (
        profile.lateral_avoidance_gain * pressure * bearing_sign * lateral * 0.45
    )
    response_velocity = speed_response + distance_response + lateral_response

    active_time = np.maximum(
        0.0,
        (np.arange(future_frames, dtype=np.float32) - delay_frames + 1.0) * timestep,
    )
    adaptation = 1.0 - np.exp(-profile.adaptation_rate * active_time)
    root_offset_xy = active_time[:, None] * adaptation[:, None] * response_velocity[None]
    root_offset = np.zeros((future_frames, 3), dtype=np.float32)
    root_offset[:, :2] = root_offset_xy
    yaw_response = (
        profile.turn_sensitivity
        * (definition.speed_scale_delta + pressure)
        * 0.20
        * adaptation
    ).astype(np.float32)

    if action_id == RobotAction.KEEP:
        root_offset.fill(0.0)
        yaw_response.fill(0.0)
    response_local = _rotate_local(natural_local, yaw_response)
    future_root = natural_root + root_offset
    future_global = future_root[:, None, :] + response_local

    robot_position = robot_history[-1, :2].astype(np.float32).copy()
    robot_yaw = float(robot_history[-1, 2])
    robot_speed = float(robot_history[-1, 3]) * (1.0 + definition.speed_scale_delta)
    robot_angular = float(robot_history[-1, 4])
    robot_future = []
    for frame in range(future_frames):
        robot_yaw += robot_angular * timestep
        robot_position = robot_position + robot_speed * timestep * np.asarray(
            (np.cos(robot_yaw), np.sin(robot_yaw)), dtype=np.float32
        )
        progress = (frame + 1) / future_frames
        shifted = (
            robot_position
            - away * definition.distance_offset_m * progress
            + lateral * definition.lateral_offset_m * progress
        )
        robot_future.append(shifted)
    robot_future_xy = np.asarray(robot_future, dtype=np.float32)
    future_distance = np.linalg.norm(
        future_root[:, :2] - robot_future_xy, axis=-1
    ).astype(np.float32)
    return InteractionSimulation(
        future_global=future_global.astype(np.float32),
        future_root=future_root.astype(np.float32),
        future_local=response_local.astype(np.float32),
        action_effect=(future_global - natural_future).astype(np.float32),
        robot_future_xy=robot_future_xy,
        future_human_robot_distance=future_distance,
        response_delay_frames=delay_frames,
    )


def simulate_risk_conditioned_interaction_future(
    human_history: np.ndarray,
    natural_future: np.ndarray,
    robot_history: np.ndarray,
    action: int | RobotAction,
    profile: VirtualPersonProfile,
    risk_factors: AdverseResponseRiskFactors,
    sample_rate_hz: float = 10.0,
) -> RiskConditionedInteractionSimulation:
    """Apply independent susceptibility dynamics to a real future trajectory.

    This wraps, but does not change, the frozen Phase-4 interaction response.
    The added residual is driven by continuous risk factors, action semantics,
    and the current approach geometry. Labels are deliberately absent here.
    """
    if int(action) == HOLD_ACTION_ID:
        from src.data.hold_candidate import simulate_hold_interaction_future
        return simulate_hold_interaction_future(
            human_history, natural_future, robot_history, profile,
            risk_factors, sample_rate_hz,
        )
    base = simulate_interaction_future(
        human_history, natural_future, robot_history, action, profile, sample_rate_hz
    )
    action_id = RobotAction(int(action))
    definition = ACTION_DEFINITIONS[action_id]
    if action_id == RobotAction.KEEP:
        return RiskConditionedInteractionSimulation(
            base.future_global, base.future_root, base.future_local,
            base.action_effect, base.robot_future_xy,
            base.future_human_robot_distance, risk_factors, False,
        )

    history_root = compute_root(np.asarray(human_history))
    velocity = (history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz
    speed = float(np.linalg.norm(velocity))
    fallback = np.asarray((np.cos(robot_history[-1, 2]), np.sin(robot_history[-1, 2])))
    forward = _unit(velocity, fallback)
    robot_to_human = history_root[-1, :2] - np.asarray(robot_history)[-1, :2]
    away = _unit(robot_to_human, forward)
    lateral = np.asarray((-forward[1], forward[0]))
    current_distance = float(np.linalg.norm(robot_to_human))
    current_bearing = float(np.asarray(robot_history)[-1, 6])

    # Approach pressure is continuous and depends on state + candidate action.
    approach = max(0.0, 1.65 - current_distance) / 0.85
    action_pressure = (
        max(definition.speed_scale_delta, 0.0)
        + max(-definition.distance_offset_m, 0.0) / 0.20
        + 0.35 * abs(definition.lateral_offset_m) / 0.20
    )
    pressure = risk_factors.approach_sensitivity * approach * action_pressure
    speed_excitation = abs(definition.speed_scale_delta) / 0.10
    distance_excitation = max(-definition.distance_offset_m, 0.0) / 0.20
    bearing_excitation = min(abs(current_bearing) / 0.55, 1.0)

    frames = len(base.future_root); dt = 1.0 / sample_rate_hz
    time = (np.arange(frames, dtype=np.float64) + 1.0) * dt
    active = np.maximum(time - risk_factors.onset_delay_seconds, 0.0)
    onset = 1.0 - np.exp(-7.0 * active)
    recovery = np.exp(-risk_factors.recovery_rate * np.maximum(active - 0.35, 0.0))
    pulse = onset * recovery

    braking_strength = (
        risk_factors.braking_susceptibility
        * (0.55 * pressure + 0.45 * speed_excitation)
        * min(max(speed, 0.35), 2.0)
    )
    lateral_strength = (
        risk_factors.lateral_startle_gain
        * (0.65 * pressure + 0.35 * distance_excitation)
        * (0.55 + 0.45 * bearing_excitation)
    )
    heading_strength = (
        risk_factors.heading_startle_gain
        * (0.55 * pressure + 0.30 * distance_excitation + 0.15 * speed_excitation)
    )
    bearing_sign = 1.0 if current_bearing >= 0.0 else -1.0
    velocity_residual = (
        -braking_strength * pulse[:, None] * forward[None]
        + bearing_sign * lateral_strength * pulse[:, None] * lateral[None]
    )
    root_residual_xy = np.cumsum(velocity_residual, axis=0) * dt
    root_residual = np.zeros((frames, 3), dtype=np.float64)
    root_residual[:, :2] = root_residual_xy
    yaw_residual = bearing_sign * heading_strength * pulse

    risk_local = _rotate_local(base.future_local.astype(np.float64), yaw_residual)
    risk_root = base.future_root.astype(np.float64) + root_residual
    future = risk_root[:, None] + risk_local
    distance = np.linalg.norm(risk_root[:, :2] - base.robot_future_xy, axis=-1)
    return RiskConditionedInteractionSimulation(
        future.astype(np.float32), risk_root.astype(np.float32),
        risk_local.astype(np.float32), (future - natural_future).astype(np.float32),
        base.robot_future_xy.copy(), distance.astype(np.float32), risk_factors, True,
    )


def _robot_history_for_human(
    history: np.ndarray,
    rng: np.random.Generator,
    sample_rate_hz: float,
    state_mode: str,
) -> np.ndarray:
    roots = compute_root(history)
    mean_velocity = (
        (roots[-1, :2] - roots[0, :2])
        * sample_rate_hz
        / max(len(roots) - 1, 1)
    )
    mean_speed = float(np.linalg.norm(mean_velocity))
    stable_heading = (
        float(np.arctan2(mean_velocity[1], mean_velocity[0]))
        if mean_speed > 0.05
        else float(rng.uniform(-np.pi, np.pi))
    )
    if state_mode == "unseen":
        distance = rng.choice((rng.uniform(0.65, 0.90), rng.uniform(2.20, 2.70)))
        bearing = rng.choice((-1.0, 1.0)) * rng.uniform(0.75, 1.20)
    else:
        distance = rng.uniform(1.10, 1.85)
        bearing = rng.uniform(-0.55, 0.55)
    direction = np.asarray(
        (np.cos(stable_heading + bearing), np.sin(stable_heading + bearing))
    )
    robot_xy = roots[:, :2] - distance * direction[None]
    robot_speed = np.full(len(roots), mean_speed, dtype=np.float32)
    heading = np.full(len(roots), stable_heading, dtype=np.float32)
    angular = np.zeros(len(roots), dtype=np.float32)
    relative = roots[:, :2] - robot_xy
    measured_distance = np.linalg.norm(relative, axis=-1)
    relative_angle = np.arctan2(relative[:, 1], relative[:, 0]) - heading
    relative_bearing = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
    return np.stack(
        (
            robot_xy[:, 0], robot_xy[:, 1], heading, robot_speed, angular,
            measured_distance, relative_bearing,
        ),
        axis=-1,
    ).astype(np.float32)


def _supervision_mask(action_type: str) -> np.ndarray:
    mask = np.ones(len(PHASE4A_ACTIONS), dtype=bool)
    if action_type in ("left_turn", "right_turn"):
        mask[3:5] = False
    if action_type in ("acceleration", "deceleration"):
        mask[1:3] = False
    mask[0] = True
    return mask


def generate_interaction_split(
    size: int,
    seed: int,
    split_kind: str,
    profile_ids: tuple[int, ...] = SEEN_PROFILE_IDS,
    history_frames: int = 20,
    future_frames: int = 10,
    sample_rate_hz: float = 10.0,
    noise_std: float = 0.005,
    occlusion_rate: float = 0.10,
    state_mode: str = "seen",
    mask_unseen_combinations: bool = False,
) -> InteractionSplit:
    if not profile_ids or any(profile_id not in PROFILE_BY_ID for profile_id in profile_ids):
        raise ValueError("profile_ids must reference declared virtual profiles")
    base: SkeletonSplit = generate_skeleton_split(
        size,
        seed,
        history_frames,
        future_frames,
        sample_rate_hz,
        noise_std,
        occlusion_rate,
    )
    rng = np.random.default_rng(seed + 44_021)
    profile_assignment = rng.choice(profile_ids, size=size, replace=True)
    actions = np.asarray([int(action) for action in PHASE4A_ACTIONS], dtype=np.int64)
    robot_histories, futures, effects, robot_futures, distances, masks = [], [], [], [], [], []
    for index in range(size):
        robot_history = _robot_history_for_human(
            base.history_global[index], rng, sample_rate_hz, state_mode
        )
        profile = PROFILE_BY_ID[int(profile_assignment[index])]
        simulations = [
            simulate_interaction_future(
                base.history_global[index], base.future_global[index], robot_history,
                action, profile, sample_rate_hz,
            )
            for action in actions
        ]
        robot_histories.append(robot_history)
        futures.append(np.stack([simulation.future_global for simulation in simulations]))
        effects.append(np.stack([simulation.action_effect for simulation in simulations]))
        robot_futures.append(np.stack([simulation.robot_future_xy for simulation in simulations]))
        distances.append(
            np.stack([simulation.future_human_robot_distance for simulation in simulations])
        )
        masks.append(
            _supervision_mask(str(base.action_type[index]))
            if mask_unseen_combinations
            else np.ones(len(actions), dtype=bool)
        )
    return InteractionSplit(
        human_history=base.history_global,
        confidence=base.confidence,
        visibility_mask=base.visibility_mask,
        robot_history=np.stack(robot_histories),
        person_profile_id=profile_assignment.astype(np.int64),
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


def create_interaction_splits(
    train_size: int = 900,
    validation_size: int = 180,
    test_size: int = 180,
    seed: int = 42,
    **options: float | int,
) -> InteractionSplits:
    sequence = np.random.SeedSequence(seed)
    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(6)]
    return InteractionSplits(
        train=generate_interaction_split(
            train_size, seeds[0], "train", SEEN_PROFILE_IDS,
            mask_unseen_combinations=True, **options,
        ),
        val=generate_interaction_split(
            validation_size, seeds[1], "validation", SEEN_PROFILE_IDS,
            mask_unseen_combinations=True, **options,
        ),
        test_seen_person_seen_context=generate_interaction_split(
            test_size, seeds[2], "seen_person_seen_context", SEEN_PROFILE_IDS, **options,
        ),
        test_unseen_interaction_state=generate_interaction_split(
            test_size, seeds[3], "unseen_interaction_state", SEEN_PROFILE_IDS,
            state_mode="unseen", **options,
        ),
        test_unseen_person_profile=generate_interaction_split(
            test_size, seeds[4], "unseen_person_profile", UNSEEN_PROFILE_IDS, **options,
        ),
        test_unseen_action_context=generate_interaction_split(
            test_size, seeds[5], "unseen_action_context", SEEN_PROFILE_IDS,
            mask_unseen_combinations=True, **options,
        ),
    )


def as_interaction_tensor_dataset(split: InteractionSplit):
    import torch

    return torch.utils.data.TensorDataset(
        torch.from_numpy(split.human_history),
        torch.from_numpy(split.natural_future),
        torch.from_numpy(split.future_by_action),
        torch.from_numpy(split.robot_history),
        torch.from_numpy(split.candidate_actions),
        torch.from_numpy(split.confidence),
        torch.from_numpy(split.visibility_mask),
        torch.from_numpy(split.person_profile_id),
        torch.from_numpy(split.action_supervision_mask),
        torch.from_numpy(split.robot_future_xy_by_action),
        torch.from_numpy(split.future_human_robot_distance),
    )
