"""Offline identifiability tools for synthetic functional human response state.

The matrices here are local/empirical observability matrices. They are not
called Fisher information because no calibrated likelihood is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.data.functional_response_state import RESPONSE_STATE_DIM, RESPONSE_STATE_SCALE
from src.data.response_probe_schema import FunctionalProbe, PROBE_BY_ID, probe_state_mask
from src.data.skeleton_schema import compute_root, hip_joints, shoulder_joints


PROBE_OBSERVABLE_NAMES = (
    "root_speed_response",
    "away_displacement",
    "lateral_displacement",
    "shoulder_yaw_response",
    "hip_yaw_response",
    "root_heading_response",
    "observed_onset_seconds",
    "adaptation_shape",
    "human_perturbation_magnitude",
)
PROBE_OBSERVABLE_DIM = len(PROBE_OBSERVABLE_NAMES)
DEFAULT_OBSERVATION_NOISE = np.asarray(
    (0.015, 0.008, 0.008, 0.006, 0.006, 0.010, 0.055, 0.06, 0.006),
    dtype=np.float64,
)
THETA_LOWER = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.1), dtype=np.float64)
THETA_UPPER = np.asarray((2.0, 2.0, 2.0, 1.5, 2.0, 5.0), dtype=np.float64)


@dataclass(frozen=True)
class ProbeSimulation:
    future_global: np.ndarray
    action_effect: np.ndarray
    response_statistics: np.ndarray
    disturbance_components: np.ndarray


@dataclass(frozen=True)
class ObservabilityDiagnostics:
    rank: int
    effective_rank: float
    condition_number: float
    smallest_singular_value: float
    singular_values: np.ndarray
    information_matrix: np.ndarray


@dataclass(frozen=True)
class FunctionalBelief:
    mean: np.ndarray
    std: np.ndarray


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else fallback.copy()


def _wrapped_angle(vector: np.ndarray) -> float:
    return float(np.arctan2(vector[1], vector[0]))


def _angle_delta(after: float, before: float) -> float:
    return float(np.arctan2(np.sin(after - before), np.cos(after - before)))


def _rotate_local(local: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    result = local.copy()
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x, y = local[..., 0].copy(), local[..., 1].copy()
    result[..., 0] = cosine[:, None] * x - sine[:, None] * y
    result[..., 1] = sine[:, None] * x + cosine[:, None] * y
    return result


def simulate_functional_probe(
    human_history: np.ndarray,
    natural_future: np.ndarray,
    robot_history: np.ndarray,
    probe: FunctionalProbe,
    theta_response: np.ndarray,
    sample_rate_hz: float = 10.0,
) -> ProbeSimulation:
    """Independent generalized probe wrapper; the frozen simulator is untouched."""
    theta = np.asarray(theta_response, dtype=np.float64)
    if theta.shape != (RESPONSE_STATE_DIM,) or not np.isfinite(theta).all():
        raise ValueError("theta_response must be finite with shape [6]")
    history = np.asarray(human_history, dtype=np.float64)
    natural = np.asarray(natural_future, dtype=np.float64)
    robot = np.asarray(robot_history, dtype=np.float64)
    history_root = compute_root(history)
    natural_root = compute_root(natural)
    natural_local = natural - natural_root[:, None]
    dt = 1.0 / float(sample_rate_hz)

    velocity = (history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz
    speed = float(np.linalg.norm(velocity))
    fallback = np.asarray((np.cos(robot[-1, 2]), np.sin(robot[-1, 2])))
    forward = _unit(velocity, fallback)
    robot_to_human = history_root[-1, :2] - robot[-1, :2]
    away = _unit(robot_to_human, forward)
    lateral = np.asarray((-away[1], away[0]))
    bearing_sign = 1.0 if robot[-1, 6] >= 0.0 else -1.0

    pressure = -probe.distance_offset_m
    response_velocity = (
        theta[0] * probe.speed_scale_delta * max(speed, 0.35) * forward
        + theta[1] * pressure * away
        + theta[2] * pressure * bearing_sign * lateral * 0.45
        - theta[2] * probe.lateral_offset_m * lateral * 0.45
    )
    frame_time = (np.arange(len(natural), dtype=np.float64) + 1.0) * dt
    active_time = np.maximum(0.0, frame_time - theta[3])
    adaptation = 1.0 - np.exp(-theta[5] * active_time)
    root_offset = np.zeros((len(natural), 3), dtype=np.float64)
    root_offset[:, :2] = active_time[:, None] * adaptation[:, None] * response_velocity
    turn_excitation = (
        probe.turn_offset_rad
        + 0.20 * (probe.speed_scale_delta + pressure)
        + 0.20 * probe.lateral_offset_m
    )
    yaw_response = theta[4] * turn_excitation * adaptation
    if not probe.active:
        root_offset.fill(0.0)
        yaw_response.fill(0.0)
    future_local = _rotate_local(natural_local, yaw_response)
    future = natural_root[:, None] + root_offset[:, None] + future_local
    effect = future - natural
    statistics = extract_probe_response_statistics(
        history, natural, future, robot, effect, sample_rate_hz
    )
    disturbance = np.asarray(
        (
            abs(probe.speed_scale_delta), abs(probe.distance_offset_m),
            abs(probe.lateral_offset_m), abs(probe.turn_offset_rad),
            float(np.linalg.norm(effect, axis=-1).mean()),
        ),
        dtype=np.float64,
    )
    return ProbeSimulation(
        future.astype(np.float32), effect.astype(np.float32),
        statistics.astype(np.float32), disturbance.astype(np.float32),
    )


def extract_probe_response_statistics(
    human_history: np.ndarray,
    natural_future: np.ndarray,
    observed_future: np.ndarray,
    robot_history: np.ndarray,
    action_effect: np.ndarray,
    sample_rate_hz: float = 10.0,
) -> np.ndarray:
    """Extract only directly observable skeleton/robot response quantities."""
    history_root = compute_root(np.asarray(human_history))
    natural_root = compute_root(np.asarray(natural_future))
    future_root = compute_root(np.asarray(observed_future))
    effect_root = compute_root(np.asarray(action_effect))
    pre_velocity = (history_root[-1, :2] - history_root[-2, :2]) * sample_rate_hz
    forward = _unit(pre_velocity, np.asarray((1.0, 0.0)))
    away = _unit(
        history_root[-1, :2] - np.asarray(robot_history)[-1, :2], forward
    )
    lateral = np.asarray((-away[1], away[0]))
    velocity_effect = (
        (future_root[-1, :2] - future_root[-2, :2])
        - (natural_root[-1, :2] - natural_root[-2, :2])
    ) * sample_rate_hz

    natural_local = np.asarray(natural_future) - natural_root[:, None]
    future_local = np.asarray(observed_future) - future_root[:, None]
    ls, rs = shoulder_joints
    lh, rh = hip_joints
    shoulder = _angle_delta(
        _wrapped_angle(future_local[-1, rs, :2] - future_local[-1, ls, :2]),
        _wrapped_angle(natural_local[-1, rs, :2] - natural_local[-1, ls, :2]),
    )
    hip = _angle_delta(
        _wrapped_angle(future_local[-1, rh, :2] - future_local[-1, lh, :2]),
        _wrapped_angle(natural_local[-1, rh, :2] - natural_local[-1, lh, :2]),
    )
    natural_motion = natural_root[-1, :2] - natural_root[0, :2]
    observed_motion = future_root[-1, :2] - future_root[0, :2]
    root_heading = _angle_delta(_wrapped_angle(observed_motion), _wrapped_angle(natural_motion))
    magnitude_by_frame = np.linalg.norm(action_effect, axis=-1).mean(axis=-1)
    threshold = max(1e-5, float(magnitude_by_frame.max()) * 0.03)
    active = np.flatnonzero(magnitude_by_frame > threshold)
    onset = float((active[0] + 1) / sample_rate_hz) if len(active) else float(len(magnitude_by_frame) / sample_rate_hz)
    early_index = max(0, len(magnitude_by_frame) // 3 - 1)
    adaptation_shape = float(
        magnitude_by_frame[early_index] / max(float(magnitude_by_frame[-1]), 1e-8)
    )
    values = np.asarray(
        (
            float(np.dot(velocity_effect, forward)),
            float(np.dot(effect_root[-1, :2], away)),
            float(np.dot(effect_root[-1, :2], lateral)),
            shoulder, hip, root_heading, onset, adaptation_shape,
            float(np.linalg.norm(action_effect, axis=-1).mean()),
        ), dtype=np.float64,
    )
    if values.shape != (PROBE_OBSERVABLE_DIM,) or not np.isfinite(values).all():
        raise ValueError("non-finite probe response statistics")
    return values


def response_jacobian(
    human_history: np.ndarray,
    natural_future: np.ndarray,
    robot_history: np.ndarray,
    probe: FunctionalProbe,
    theta_response: np.ndarray,
    sample_rate_hz: float = 10.0,
    relative_step: float = 0.03,
) -> np.ndarray:
    """Stable central finite-difference d(observable statistics)/d(theta)."""
    theta = np.asarray(theta_response, dtype=np.float64)
    jacobian = np.zeros((PROBE_OBSERVABLE_DIM, RESPONSE_STATE_DIM), dtype=np.float64)
    for dimension in range(RESPONSE_STATE_DIM):
        step = max(float(RESPONSE_STATE_SCALE[dimension]) * relative_step, 1e-4)
        low, high = theta.copy(), theta.copy()
        low[dimension] = max(THETA_LOWER[dimension], low[dimension] - step)
        high[dimension] = min(THETA_UPPER[dimension], high[dimension] + step)
        denominator = high[dimension] - low[dimension]
        if denominator <= 0.0:
            continue
        low_stat = simulate_functional_probe(
            human_history, natural_future, robot_history, probe, low, sample_rate_hz
        ).response_statistics
        high_stat = simulate_functional_probe(
            human_history, natural_future, robot_history, probe, high, sample_rate_hz
        ).response_statistics
        jacobian[:, dimension] = (high_stat - low_stat) / denominator
    if not np.isfinite(jacobian).all():
        raise ValueError("response Jacobian contains non-finite values")
    return jacobian


def local_observability_diagnostics(
    jacobians: Sequence[np.ndarray],
    observation_noise: np.ndarray = DEFAULT_OBSERVATION_NOISE,
) -> ObservabilityDiagnostics:
    if not jacobians:
        matrix = np.zeros((0, RESPONSE_STATE_DIM), dtype=np.float64)
    else:
        matrix = np.concatenate(
            [np.asarray(item, dtype=np.float64) / observation_noise[:, None] for item in jacobians],
            axis=0,
        )
    singular = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.zeros(RESPONSE_STATE_DIM)
    tolerance = max(matrix.shape, default=1) * np.finfo(float).eps * (singular[0] if len(singular) else 0.0)
    rank = int(np.sum(singular > tolerance))
    positive = singular[singular > tolerance]
    if positive.size:
        probability = positive / positive.sum()
        effective_rank = float(np.exp(-(probability * np.log(probability)).sum()))
        condition = float(positive[0] / positive[-1]) if rank == RESPONSE_STATE_DIM else float("inf")
        smallest = float(positive[-1]) if rank == RESPONSE_STATE_DIM else 0.0
    else:
        effective_rank, condition, smallest = 0.0, float("inf"), 0.0
    return ObservabilityDiagnostics(
        rank, effective_rank, condition, smallest, singular,
        matrix.T @ matrix,
    )


def posterior_std(
    prior_std: np.ndarray,
    jacobians: Sequence[np.ndarray],
    observation_noise: np.ndarray = DEFAULT_OBSERVATION_NOISE,
) -> np.ndarray:
    prior = np.asarray(prior_std, dtype=np.float64).clip(1e-4)
    precision = np.diag(1.0 / np.square(prior))
    for jacobian in jacobians:
        normalized = np.asarray(jacobian) / observation_noise[:, None]
        precision += normalized.T @ normalized
    covariance = np.linalg.pinv(precision, rcond=1e-10)
    return np.sqrt(np.diag(covariance).clip(1e-12))


def uncertainty_specificity_score(
    before_std: np.ndarray, after_std: np.ndarray, state_mask: Sequence[bool]
) -> dict[str, float]:
    before, after = np.asarray(before_std), np.asarray(after_std)
    mask = np.asarray(state_mask, dtype=bool)
    reduction = np.maximum(before - after, 0.0)
    total = float(reduction.sum())
    relevant = float(reduction[mask].sum())
    irrelevant = float(reduction[~mask].sum())
    return {
        "specificity_score": relevant / max(total, 1e-12) if mask.any() else 0.0,
        "relevant_reduction": relevant,
        "irrelevant_reduction": irrelevant,
        "total_reduction": total,
    }


def disturbance_score(simulation: ProbeSimulation) -> float:
    speed, distance, lateral, turn, human = simulation.disturbance_components
    # Transparent dimensionless proxy, not a real-human comfort model.
    return float(
        0.20 * speed / 0.15 + 0.20 * distance / 0.30
        + 0.15 * lateral / 0.20 + 0.15 * turn / 0.12
        + 0.30 * human / 0.05
    )


def information_score(prior_std: np.ndarray, jacobians: Sequence[np.ndarray]) -> float:
    after = posterior_std(prior_std, jacobians)
    scale = np.asarray(RESPONSE_STATE_SCALE, dtype=np.float64)
    return float(np.sum((np.asarray(prior_std) - after).clip(0.0) / scale))


def functional_belief_update(
    belief: FunctionalBelief,
    observed_statistics: np.ndarray,
    predicted_statistics: np.ndarray,
    jacobian: np.ndarray,
    observation_noise: np.ndarray = DEFAULT_OBSERVATION_NOISE,
) -> FunctionalBelief:
    prior_precision = np.diag(
        1.0 / np.square(np.asarray(belief.std).clip(1e-4))
    )
    normalized_j = np.asarray(jacobian) / observation_noise[:, None]
    residual = (np.asarray(observed_statistics) - np.asarray(predicted_statistics)) / observation_noise
    precision = prior_precision + normalized_j.T @ normalized_j
    delta = np.linalg.pinv(precision, rcond=1e-10) @ normalized_j.T @ residual
    mean = np.clip(np.asarray(belief.mean) + delta, THETA_LOWER, THETA_UPPER)
    std = np.sqrt(np.diag(np.linalg.pinv(precision, rcond=1e-10)).clip(1e-12))
    return FunctionalBelief(mean.astype(np.float64), std.astype(np.float64))


def select_probe_without_oracle(
    strategy: str,
    candidates: Sequence[FunctionalProbe],
    belief: FunctionalBelief,
    human_history: np.ndarray,
    natural_future: np.ndarray,
    robot_history: np.ndarray,
    cumulative_jacobians: Sequence[np.ndarray] = (),
    seed: int = 0,
    step: int = 0,
) -> FunctionalProbe:
    """Select without any person ID, hidden profile, or test theta input."""
    if not candidates:
        raise ValueError("at least one candidate probe is required")
    if strategy == "random":
        return candidates[int(np.random.default_rng(seed + step * 7919).integers(len(candidates)))]
    if strategy == "recent":
        return candidates[step % len(candidates)]
    if strategy == "naive_diverse":
        order = ("SPEED_DOWN_10", "DISTANCE_PLUS_0_2", "SPEED_UP_10", "DISTANCE_MINUS_0_2", "KEEP")
        return PROBE_BY_ID[order[step % len(order)]]
    scored: list[tuple[float, str, FunctionalProbe]] = []
    for probe in candidates:
        jacobian = response_jacobian(
            human_history, natural_future, robot_history, probe, belief.mean
        )
        if strategy == "greedy_uncertainty":
            before = posterior_std(belief.std, ())
            after = posterior_std(belief.std, (jacobian,))
            score = float(np.sum((before - after) / RESPONSE_STATE_SCALE))
        elif strategy == "greedy_observability":
            diagnostic = local_observability_diagnostics((*cumulative_jacobians, jacobian))
            score = diagnostic.effective_rank + 0.05 * np.log1p(
                np.trace(diagnostic.information_matrix)
            )
        else:
            raise ValueError(f"unknown non-oracle selection strategy: {strategy}")
        scored.append((score, probe.probe_id, probe))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def select_oracle_probe(
    candidates: Sequence[FunctionalProbe],
    belief: FunctionalBelief,
    theta_true: np.ndarray,
    human_history: np.ndarray,
    natural_future: np.ndarray,
    robot_history: np.ndarray,
) -> FunctionalProbe:
    """The only selector allowed to consult GT theta; theoretical upper bound."""
    best: tuple[float, str, FunctionalProbe] | None = None
    reference_probes = tuple(
        PROBE_BY_ID[name] for name in (
            "SPEED_DOWN_10", "SPEED_UP_10", "DISTANCE_PLUS_0_2",
            "DISTANCE_MINUS_0_2", "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL",
        )
    )
    for probe in candidates:
        observed = simulate_functional_probe(
            human_history, natural_future, robot_history, probe, theta_true
        ).response_statistics
        predicted = simulate_functional_probe(
            human_history, natural_future, robot_history, probe, belief.mean
        ).response_statistics
        jacobian = response_jacobian(
            human_history, natural_future, robot_history, probe, belief.mean
        )
        updated = functional_belief_update(belief, observed, predicted, jacobian)
        response_error = np.mean([
            np.linalg.norm(
                simulate_functional_probe(
                    human_history, natural_future, robot_history,
                    reference, updated.mean,
                ).action_effect
                - simulate_functional_probe(
                    human_history, natural_future, robot_history,
                    reference, theta_true,
                ).action_effect,
                axis=-1,
            ).mean()
            for reference in reference_probes
        ])
        normalized_state_error = np.mean(
            np.abs(updated.mean - theta_true) / RESPONSE_STATE_SCALE
        )
        # GT is deliberately allowed only in B5. Prioritize response-function
        # fidelity while retaining a small full-state identifiability term.
        score = -float(response_error + 0.02 * normalized_state_error)
        item = (score, probe.probe_id, probe)
        if best is None or item[:2] > best[:2]:
            best = item
    assert best is not None
    return best[2]


def classic_probe_for_action(action: int) -> FunctionalProbe:
    mapping = {
        0: "KEEP", 1: "SPEED_DOWN_10", 2: "SPEED_UP_10",
        3: "DISTANCE_PLUS_0_2", 4: "DISTANCE_MINUS_0_2",
        5: "TURN_LEFT_SMALL", 6: "TURN_RIGHT_SMALL",
    }
    return PROBE_BY_ID[mapping[int(action)]]
