"""Executable synthetic HOLD candidate for manifest-v3 readiness.

HOLD means rate-limited braking to zero followed by zero-motion hold.  It is
not declared safe: robot/human trajectories, costs, and harm labels are all
computed through the same Phase-5B development protocols as other actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.adverse_response_protocol import AdverseResponseEvents, derive_adverse_response_events
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.data.skeleton_schema import compute_root, global_to_local
from src.data.synthetic_interaction import (
    AdverseResponseRiskFactors, VirtualPersonProfile, _rotate_local, _unit,
)
from src.decision.candidate_action import CandidateAction
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_cost import DecisionCostWeights, compute_decision_costs
from src.decision.decision_state import DecisionState, FunctionalResponseBelief


HOLD_CONTROL_CONTRACT_VERSION = "hold_brake_to_zero_v1"
HOLD_LINEAR_DECELERATION_LIMIT_MPS2 = 0.40
HOLD_ANGULAR_DECELERATION_LIMIT_RADPS2 = 0.50
HOLD_SAMPLE_RATE_HZ = 10.0


@dataclass(frozen=True)
class HoldRobotRollout:
    states: np.ndarray  # [H,5]: x, y, yaw, linear velocity, angular velocity

    @property
    def xy(self) -> np.ndarray:
        return self.states[:, :2]


@dataclass(frozen=True)
class HoldInteractionSimulation:
    future_global: np.ndarray
    future_root: np.ndarray
    future_local: np.ndarray
    action_effect: np.ndarray
    robot_future_xy: np.ndarray
    robot_future_state: np.ndarray
    future_human_robot_distance: np.ndarray
    response_delay_frames: int
    risk_dynamics_applied: bool


@dataclass(frozen=True)
class HoldCandidateOutcome:
    action_id: int
    feasible: bool
    generic_simulation: HoldInteractionSimulation
    gt_simulation: HoldInteractionSimulation
    generic_total_cost: float
    gt_total_cost: float
    benefit: float
    gt_unsafe: bool
    events: AdverseResponseEvents
    harm_v2: bool
    regret: float


@dataclass(frozen=True)
class HoldCandidateAction(CandidateAction):
    action: int = HOLD_ACTION_ID


def _approach_zero(value: float, maximum_change: float) -> float:
    if value > 0.0:
        return max(0.0, value - maximum_change)
    if value < 0.0:
        return min(0.0, value + maximum_change)
    return 0.0


def hold_robot_rollout(
    robot_history: np.ndarray, future_frames: int = 10,
    sample_rate_hz: float = HOLD_SAMPLE_RATE_HZ,
    linear_deceleration_limit: float = HOLD_LINEAR_DECELERATION_LIMIT_MPS2,
    angular_deceleration_limit: float = HOLD_ANGULAR_DECELERATION_LIMIT_RADPS2,
) -> HoldRobotRollout:
    """Integrate a rate-limited brake-to-zero trajectory deterministically."""
    history = np.asarray(robot_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] != 7 or future_frames <= 0 or sample_rate_hz <= 0:
        raise ValueError("robot_history must be [T,7] and rollout settings positive")
    if linear_deceleration_limit <= 0 or angular_deceleration_limit <= 0:
        raise ValueError("HOLD deceleration limits must be positive")
    dt = 1.0 / float(sample_rate_hz)
    x, y, yaw, velocity, angular = map(float, history[-1, :5])
    rows = []
    for _ in range(future_frames):
        next_velocity = _approach_zero(velocity, linear_deceleration_limit * dt)
        next_angular = _approach_zero(angular, angular_deceleration_limit * dt)
        mean_velocity = 0.5 * (velocity + next_velocity)
        mean_angular = 0.5 * (angular + next_angular)
        mid_yaw = yaw + 0.5 * mean_angular * dt
        x += mean_velocity * dt * np.cos(mid_yaw)
        y += mean_velocity * dt * np.sin(mid_yaw)
        yaw = float(np.arctan2(np.sin(yaw + mean_angular * dt), np.cos(yaw + mean_angular * dt)))
        velocity, angular = next_velocity, next_angular
        rows.append((x, y, yaw, velocity, angular))
    return HoldRobotRollout(np.asarray(rows, dtype=np.float32))


def simulate_hold_interaction_future(
    human_history: np.ndarray, natural_future: np.ndarray, robot_history: np.ndarray,
    profile: VirtualPersonProfile,
    risk_factors: AdverseResponseRiskFactors | None = None,
    sample_rate_hz: float = HOLD_SAMPLE_RATE_HZ,
) -> HoldInteractionSimulation:
    """Apply delayed response residuals driven by the actual HOLD rollout."""
    history = np.asarray(human_history, dtype=np.float64)
    natural = np.asarray(natural_future, dtype=np.float64)
    robot = np.asarray(robot_history, dtype=np.float64)
    frames = natural.shape[0]; dt = 1.0 / sample_rate_hz
    hold = hold_robot_rollout(robot, frames, sample_rate_hz)
    history_root = compute_root(history)
    natural_root, natural_local = global_to_local(natural)
    human_velocity = (history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz
    human_speed = float(np.linalg.norm(human_velocity))
    forward = _unit(human_velocity, np.asarray((np.cos(robot[-1, 2]), np.sin(robot[-1, 2]))))
    robot_to_human = history_root[-1, :2] - robot[-1, :2]
    away = _unit(robot_to_human, forward)
    lateral = np.asarray((-away[1], away[0]))
    initial_robot_speed = abs(float(robot[-1, 3]))
    remaining = np.abs(hold.states[:, 3]) / max(initial_robot_speed, 1e-6)
    braking_fraction = np.clip(1.0 - remaining, 0.0, 1.0)
    delay_frames = int(np.ceil(profile.response_delay * sample_rate_hz - 1e-9))
    active_time = np.maximum(0.0, (np.arange(frames) - delay_frames + 1.0) * dt)
    adaptation = 1.0 - np.exp(-profile.adaptation_rate * active_time)
    delayed_braking = np.where(np.arange(frames) >= delay_frames, braking_fraction, 0.0)

    response_velocity = (
        -profile.speed_response_gain * delayed_braking[:, None]
        * max(human_speed, 0.35) * forward[None]
    )
    natural_distance = np.linalg.norm(natural_root[:, :2] - hold.xy, axis=-1)
    close_pressure = np.maximum(profile.preferred_distance - natural_distance, 0.0)
    response_velocity += (
        profile.distance_sensitivity * close_pressure[:, None] * away[None] * 0.25
    )
    yaw_response = np.zeros(frames, dtype=np.float64)

    if risk_factors is not None:
        onset_active = np.maximum(
            (np.arange(frames) + 1.0) * dt - risk_factors.onset_delay_seconds, 0.0
        )
        onset = 1.0 - np.exp(-7.0 * onset_active)
        recovery = np.exp(-risk_factors.recovery_rate * np.maximum(onset_active - 0.35, 0.0))
        pulse = onset * recovery
        approach = np.maximum(1.65 - natural_distance, 0.0) / 0.85
        response_velocity += (
            -risk_factors.braking_susceptibility * delayed_braking[:, None]
            * max(human_speed, 0.35) * pulse[:, None] * forward[None]
            + risk_factors.lateral_startle_gain * approach[:, None] * pulse[:, None]
            * 0.25 * np.sign(robot[-1, 6] if robot[-1, 6] != 0 else 1.0)
            * lateral[None]
        )
        yaw_response = (
            risk_factors.heading_startle_gain * approach * pulse * 0.20
            * np.sign(robot[-1, 6] if robot[-1, 6] != 0 else 1.0)
        )

    root_offset_xy = np.cumsum(adaptation[:, None] * response_velocity, axis=0) * dt
    root_offset = np.zeros((frames, 3), dtype=np.float64); root_offset[:, :2] = root_offset_xy
    local = _rotate_local(natural_local, yaw_response)
    future_root = natural_root + root_offset
    future = future_root[:, None, :] + local
    distance = np.linalg.norm(future_root[:, :2] - hold.xy, axis=-1)
    return HoldInteractionSimulation(
        future.astype(np.float32), future_root.astype(np.float32), local.astype(np.float32),
        (future - natural).astype(np.float32), hold.xy.copy(), hold.states.copy(),
        distance.astype(np.float32), delay_frames, risk_factors is not None,
    )


def _cost(episode, simulation: HoldInteractionSimulation) -> float:
    candidate = HoldCandidateAction()
    state = DecisionState(
        episode.human_history, episode.robot_history, episode.confidence,
        episode.visibility,
        FunctionalResponseBelief(np.ones(6, np.float32), np.zeros(6, np.float32)),
        (candidate,), episode.target_follow_distance, .80, episode.split,
    )
    future = simulation.future_global[None]
    root = compute_root(future); local = future - root[:, :, None]
    effect = simulation.action_effect[None]
    rollout = CounterfactualRollout(
        np.asarray((HOLD_ACTION_ID,), np.int64), episode.natural_future,
        root.astype(np.float32), local.astype(np.float32), future.astype(np.float32),
        simulation.robot_future_xy[None], simulation.future_human_robot_distance[None],
        effect.astype(np.float32), np.zeros_like(effect, np.float32), 0,
    )
    return float(compute_decision_costs(
        state, rollout, DecisionCostWeights(), include_uncertainty=False
    ).total[0])


def build_hold_candidate_outcome(episode, population_profile: VirtualPersonProfile,
                                 person_profile: VirtualPersonProfile) -> HoldCandidateOutcome:
    """Materialize HOLD for one already-split development episode."""
    # Use the same public action-conditioned generator entry points as A0--A4.
    from src.data.synthetic_interaction import (
        simulate_interaction_future, simulate_risk_conditioned_interaction_future,
    )
    generic = simulate_interaction_future(
        episode.human_history, episode.natural_future, episode.robot_history,
        HOLD_ACTION_ID, population_profile,
    )
    truth = simulate_risk_conditioned_interaction_future(
        episode.human_history, episode.natural_future, episode.robot_history,
        HOLD_ACTION_ID, person_profile, episode.risk_factors,
    )
    generic_cost, gt_cost = _cost(episode, generic), _cost(episode, truth)
    events = derive_adverse_response_events(
        episode.human_history, episode.natural_future, truth.future_global
    )
    unsafe = bool(np.mean(truth.future_human_robot_distance < .80) > 0.0)
    harm = bool(unsafe or events.adverse_human_kinematic_response)
    baseline = float(episode.gt_costs[episode.generic_action_index])
    oracle = min(float(np.min(episode.gt_costs)), gt_cost)
    return HoldCandidateOutcome(
        HOLD_ACTION_ID, True, generic, truth, generic_cost, gt_cost,
        baseline - gt_cost, unsafe, events, harm, gt_cost - oracle,
    )
