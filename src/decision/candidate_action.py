"""Task-safe high-level candidate actions for synthetic Phase 4C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data.robot_action_schema import RobotAction


@dataclass(frozen=True)
class CandidateAction:
    action: RobotAction
    task_action: bool = True
    identification_probe: bool = False
    feasible: bool = True

    def __post_init__(self) -> None:
        if self.task_action and self.identification_probe:
            raise ValueError("task action and identification probe must remain distinct")


TASK_SAFE_CANDIDATES = (
    CandidateAction(RobotAction.KEEP),
    CandidateAction(RobotAction.SPEED_DOWN_10),
    CandidateAction(RobotAction.SPEED_UP_10),
    CandidateAction(RobotAction.DISTANCE_PLUS_0_2),
    CandidateAction(RobotAction.DISTANCE_MINUS_0_2),
)


def validate_candidate_actions(candidates: Iterable[CandidateAction]) -> tuple[CandidateAction, ...]:
    values = tuple(candidates)
    if not values:
        raise ValueError("candidate set cannot be empty")
    ids = [int(item.action) for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate action IDs must be unique")
    if any(not item.task_action or item.identification_probe for item in values):
        raise ValueError("Phase 4C main decision accepts task actions only")
    return values
