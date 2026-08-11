"""Synthetic-only high-level probes for Phase 4B.7 offline identification."""

from __future__ import annotations

from dataclasses import dataclass


SYNTHETIC_INTERACTION_LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


@dataclass(frozen=True)
class FunctionalProbe:
    probe_id: str
    speed_scale_delta: float = 0.0
    distance_offset_m: float = 0.0
    lateral_offset_m: float = 0.0
    turn_offset_rad: float = 0.0
    synthetic_only: bool = True
    high_level_action: bool = True

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id cannot be empty")
        if not self.synthetic_only:
            raise ValueError("Phase 4B.7 probes must be synthetic-only")
        if not self.high_level_action:
            raise ValueError("probe must remain a structured high-level action")

    @property
    def active(self) -> bool:
        return any(
            abs(value) > 0.0 for value in (
                self.speed_scale_delta,
                self.distance_offset_m,
                self.lateral_offset_m,
                self.turn_offset_rad,
            )
        )


PROBE_CATALOG = (
    FunctionalProbe("KEEP"),
    FunctionalProbe("SPEED_DOWN_5", speed_scale_delta=-0.05),
    FunctionalProbe("SPEED_DOWN_10", speed_scale_delta=-0.10),
    FunctionalProbe("SPEED_DOWN_15", speed_scale_delta=-0.15),
    FunctionalProbe("SPEED_UP_5", speed_scale_delta=0.05),
    FunctionalProbe("SPEED_UP_10", speed_scale_delta=0.10),
    FunctionalProbe("SPEED_UP_15", speed_scale_delta=0.15),
    FunctionalProbe("DISTANCE_PLUS_0_1", distance_offset_m=0.10),
    FunctionalProbe("DISTANCE_PLUS_0_2", distance_offset_m=0.20),
    FunctionalProbe("DISTANCE_PLUS_0_3", distance_offset_m=0.30),
    FunctionalProbe("DISTANCE_MINUS_0_1", distance_offset_m=-0.10),
    FunctionalProbe("DISTANCE_MINUS_0_2", distance_offset_m=-0.20),
    FunctionalProbe("TURN_LEFT_SMALL", turn_offset_rad=0.12),
    FunctionalProbe("TURN_RIGHT_SMALL", turn_offset_rad=-0.12),
)
PROBE_BY_ID = {probe.probe_id: probe for probe in PROBE_CATALOG}


def probe_state_mask(probe: FunctionalProbe) -> tuple[bool, ...]:
    """Dimensions that a probe can physically excite, not learned shortcuts."""
    speed = abs(probe.speed_scale_delta) > 0.0
    distance = abs(probe.distance_offset_m) > 0.0
    lateral = abs(probe.lateral_offset_m) > 0.0
    turn = abs(probe.turn_offset_rad) > 0.0
    active = probe.active
    return (
        speed,
        distance,
        distance or lateral,
        active,
        turn or speed or distance or lateral,
        active,
    )


def validate_probe_catalog() -> None:
    ids = [probe.probe_id for probe in PROBE_CATALOG]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate functional probe IDs")
    if any(hasattr(probe, "cmd_vel") for probe in PROBE_CATALOG):
        raise ValueError("probe schema must not contain cmd_vel sequences")


validate_probe_catalog()
