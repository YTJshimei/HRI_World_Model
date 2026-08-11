"""Canonical observable response-function state for synthetic Phase 4B.6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.data.robot_action_schema import RobotAction


RESPONSE_STATE_NAMES = (
    "speed_response_gain",
    "distance_response_gain",
    "lateral_response_gain",
    "response_delay",
    "turn_response_gain",
    "adaptation_response_gain",
)
RESPONSE_STATE_DIM = len(RESPONSE_STATE_NAMES)
RESPONSE_STATE_SCALE = np.asarray((1.5, 1.5, 1.2, 0.8, 1.2, 3.5), dtype=np.float32)


@dataclass(frozen=True)
class FunctionalResponseState:
    values: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.values).shape != (RESPONSE_STATE_DIM,):
            raise ValueError(f"response state must have {RESPONSE_STATE_DIM} values")
        if np.asarray(self.mask).shape != (RESPONSE_STATE_DIM,):
            raise ValueError(f"response state mask must have {RESPONSE_STATE_DIM} values")
        if not np.isfinite(self.values).all():
            raise ValueError("response state contains non-finite values")


def functional_state_from_profile(profile: Any) -> np.ndarray:
    """Explicit simulator-profile -> functional-response mapping.

    Identity fields and preferred distance are deliberately excluded.
    """
    return np.asarray(
        (
            profile.speed_response_gain,
            profile.distance_sensitivity,
            profile.lateral_avoidance_gain,
            profile.response_delay,
            profile.turn_sensitivity,
            profile.adaptation_rate,
        ),
        dtype=np.float32,
    )


def response_state_mask_for_action(action: int | RobotAction) -> np.ndarray:
    """Dimensions made observable by one executed high-level action."""
    action = RobotAction(int(action))
    mask = np.zeros(RESPONSE_STATE_DIM, dtype=bool)
    if action in (RobotAction.SPEED_DOWN_10, RobotAction.SPEED_UP_10):
        mask[[0, 3, 4, 5]] = True
    elif action in (RobotAction.DISTANCE_PLUS_0_2, RobotAction.DISTANCE_MINUS_0_2):
        mask[[1, 2, 3, 4, 5]] = True
    return mask


def aggregate_response_state_mask(actions: Iterable[int]) -> np.ndarray:
    result = np.zeros(RESPONSE_STATE_DIM, dtype=bool)
    for action in actions:
        result |= response_state_mask_for_action(int(action))
    return result


def population_mean_response_state(profiles: Iterable[Any]) -> np.ndarray:
    values = np.stack([functional_state_from_profile(profile) for profile in profiles])
    if len(values) == 0:
        raise ValueError("population mean needs at least one profile")
    return values.mean(axis=0).astype(np.float32)

