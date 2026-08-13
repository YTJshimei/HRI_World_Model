"""Leakage-safe manifest-v3 temporal bridge for A0--A4 plus HOLD.

This module deliberately leaves the frozen v2 schema untouched.  It reuses
the exact v2 streams for shared candidates, changes only their action encoding
from 11D to 12D, and constructs HOLD from the same runtime-observable history.
Only TRAIN/VALIDATION episodes may be materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.data.adverse_response_dataset import DEVELOPMENT_PROFILE_IDS, POPULATION_PROFILE, HarmV2Episode
from src.data.functional_response_state import (
    aggregate_response_state_mask,
    functional_state_from_profile,
    population_mean_response_state,
)
from src.data.hold_candidate import HoldCandidateOutcome, build_hold_candidate_outcome
from src.data.robot_action_schema import HOLD_ACTION_ID, action_feature_v3, candidate_action_vector_v3
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import PROFILE_BY_ID, simulate_interaction_future
from src.multimodal.context_schema import build_context_tokens
from src.multimodal.phase5b_v2_dataset import (
    DEPRECATED_HARM_TARGET,
    _motion_observable,
    _support,
    build_v2_temporal_samples,
)
from src.multimodal.temporal_dataset import _candidate_future
from src.multimodal.temporal_schema import (
    DT_SECONDS,
    FORBIDDEN_RUNTIME_KEYS,
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    STREAM_ORDER,
    TemporalTargets,
)

V3_CANDIDATE_ACTION_DIM = 12  # 8-way action ID + frozen four semantic values
V3_CANDIDATE_ROBOT_FUTURE_SHAPE = (10, 5)
V3_STREAM_DIMS = {
    "skeleton_history": (HISTORY_FRAMES, 17, 3),
    "human_motion_history": (HISTORY_FRAMES, 16),
    "robot_history": (HISTORY_FRAMES, 7),
    "functional_history": (HISTORY_FRAMES, 18),
    "visibility_history": (HISTORY_FRAMES, 4),
    "wm_diagnostic_history": (HISTORY_FRAMES, 8),
    "interaction_history": (HISTORY_FRAMES, 13),
    "candidate_action": (V3_CANDIDATE_ACTION_DIM,),
    "candidate_robot_future": V3_CANDIDATE_ROBOT_FUTURE_SHAPE,
    "scene_context": (8,),
}


@dataclass(frozen=True)
class RichTemporalSampleV3:
    streams: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    timestamps: Mapping[str, np.ndarray]
    targets: TemporalTargets
    sample_id: str
    episode_id: str
    split: str
    context_split: str
    temporal_tags: tuple[str, ...]
    split_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation"):
            raise ValueError("manifest-v3 runtime bridge exposes TRAIN/VALIDATION only")
        if tuple(self.streams) != STREAM_ORDER:
            raise ValueError("v3 temporal streams must use the canonical STREAM_ORDER")
        if set(self.streams) & FORBIDDEN_RUNTIME_KEYS:
            raise ValueError("target/identity/future-human fields cannot enter v3 runtime streams")
        for name, shape in V3_STREAM_DIMS.items():
            value = np.asarray(self.streams[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            if name not in self.masks or np.asarray(self.masks[name]).shape != shape:
                raise ValueError(f"{name} requires a shape-matched validity mask")
        if "person_profile_id" not in self.split_metadata:
            raise ValueError("anonymous profile metadata is required for split auditing")
        history_time = np.asarray(self.timestamps["history"])
        future_time = np.asarray(self.timestamps["candidate_future"])
        if history_time.shape != (HISTORY_FRAMES,) or future_time.shape != (FUTURE_FRAMES,):
            raise ValueError("v3 timestamp shapes violate the canonical timeline")
        if history_time.max() > 1e-8 or future_time.min() <= 0:
            raise ValueError("v3 history/future crosses the decision boundary")


@dataclass(frozen=True)
class HoldTemporalSampleV3:
    """Small HOLD-only readiness view retained for Phase 5B-1.7F-D."""

    episode_id: str
    split: str
    candidate_action: np.ndarray
    candidate_robot_future: np.ndarray
    gt_cost: float
    benefit: float
    gt_unsafe: bool
    harm_v2: bool

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation"):
            raise ValueError("HOLD readiness materializes TRAIN/VALIDATION only")
        if np.asarray(self.candidate_action).shape != (V3_CANDIDATE_ACTION_DIM,):
            raise ValueError("manifest-v3 candidate action must have shape [12]")
        if np.asarray(self.candidate_robot_future).shape != V3_CANDIDATE_ROBOT_FUTURE_SHAPE:
            raise ValueError("candidate robot future must have shape [10,5]")
        if not np.isfinite(self.candidate_action).all() or not np.isfinite(self.candidate_robot_future).all():
            raise ValueError("HOLD runtime inputs must be finite")


def build_hold_temporal_sample_v3(episode, outcome) -> HoldTemporalSampleV3:
    return HoldTemporalSampleV3(
        episode.episode_id,
        episode.split,
        candidate_action_vector_v3(outcome.action_id),
        outcome.gt_simulation.robot_future_state.copy(),
        outcome.gt_total_cost,
        outcome.benefit,
        outcome.gt_unsafe,
        outcome.harm_v2,
    )


def _shared_metadata(episode: HarmV2Episode, hold: HoldCandidateOutcome) -> dict[str, object]:
    return {
        "all_action_ids_evaluation_only": np.asarray(
            [*[item.action_id for item in episode.candidates], HOLD_ACTION_ID], dtype=int
        ),
        "generic_costs_evaluation_only": np.r_[episode.generic_costs, hold.generic_total_cost].astype(np.float32),
        "personalized_costs_evaluation_only": np.r_[episode.generic_costs, hold.generic_total_cost].astype(np.float32),
        "gt_costs_evaluation_only": np.r_[episode.gt_costs, hold.gt_total_cost].astype(np.float32),
        "gt_unsafe_evaluation_only": np.asarray(
            [*[item.gt_unsafe for item in episode.candidates], hold.gt_unsafe], dtype=bool
        ),
    }


def _hold_static_token(episode: HarmV2Episode, runtime_natural: np.ndarray, simulation) -> np.ndarray:
    population_theta = population_mean_response_state([PROFILE_BY_ID[index] for index in DEVELOPMENT_PROFILE_IDS])
    population_values = np.stack([functional_state_from_profile(PROFILE_BY_ID[index]) for index in DEVELOPMENT_PROFILE_IDS])
    theta_std = population_values.std(axis=0).astype(np.float32)
    support_actions = _support(episode)
    response_mask = aggregate_response_state_mask(support_actions)
    support_feature = np.asarray(
        (len(support_actions) / 5, len(set(support_actions)) / 5, response_mask.mean()), np.float32
    )
    distance = np.asarray(simulation.future_human_robot_distance, np.float32)
    effect = np.asarray(simulation.action_effect, np.float32)
    sigma = np.full((FUTURE_FRAMES, 3), .03, np.float32)
    motion = np.linalg.norm(np.diff(compute_root(episode.human_history)[:, :2], axis=0) / DT_SECONDS, axis=-1)
    scene = np.asarray(
        (
            episode.visibility.mean(), episode.confidence.mean(), episode.robot_history[-1, 5],
            episode.robot_history[-1, 5] - episode.robot_history[0, 5], motion.mean(), motion.std(),
            episode.robot_history[-1, 6], len(support_actions) / 5,
        ), np.float32,
    )
    token = build_context_tokens(
        human_history=episode.human_history,
        robot_history=episode.robot_history,
        confidence=episode.confidence,
        visibility=episode.visibility,
        theta_person=population_theta,
        theta_population=population_theta,
        theta_uncertainty=theta_std,
        response_state_mask=response_mask,
        support_coverage=response_mask.astype(np.float32),
        support_action_features=support_feature,
        candidate_action=HOLD_ACTION_ID,
        candidate_feature=action_feature_v3(HOLD_ACTION_ID),
        predicted_robot_future=simulation.robot_future_xy,
        generic_effect=effect,
        personalized_effect=effect,
        generic_distance=distance,
        personalized_distance=distance,
        root_sigma=sigma,
        minimum_sigma=.03,
        p_unsafe=float(np.mean(distance < .80)),
        motion_state_observable=_motion_observable(episode.human_history),
        scene_observable=scene,
        context_id=f"{episode.episode_id}:{HOLD_ACTION_ID}",
        initial_state_id=episode.episode_id,
        context_split=episode.split,
    )
    return token.flattened().copy()


def _convert_shared(sample, episode: HarmV2Episode, hold: HoldCandidateOutcome) -> RichTemporalSampleV3:
    action_id = int(sample.split_metadata["candidate_action_id_audit"])
    action = candidate_action_vector_v3(action_id)
    streams = {name: np.asarray(value).copy() for name, value in sample.streams.items()}
    streams["candidate_action"] = action
    masks = {name: np.asarray(value).copy() for name, value in sample.masks.items()}
    masks["candidate_action"] = np.ones_like(action, bool)
    metadata = {**sample.split_metadata, **_shared_metadata(episode, hold)}
    return RichTemporalSampleV3(
        streams, masks, {name: np.asarray(value).copy() for name, value in sample.timestamps.items()},
        sample.targets, sample.sample_id, sample.episode_id, sample.split, sample.context_split,
        sample.temporal_tags, metadata,
    )


def _build_hold(episode: HarmV2Episode, template, hold: HoldCandidateOutcome) -> RichTemporalSampleV3:
    history = episode.human_history
    runtime_natural = history[-1][None] + np.arange(1, FUTURE_FRAMES + 1, dtype=np.float32)[:, None, None] * (
        history[-1] - history[-2]
    )[None]
    simulation = simulate_interaction_future(
        history, runtime_natural, episode.robot_history, HOLD_ACTION_ID, POPULATION_PROFILE
    )
    streams = {name: np.asarray(value).copy() for name, value in template.streams.items()}
    action = candidate_action_vector_v3(HOLD_ACTION_ID)
    streams["candidate_action"] = action
    streams["candidate_robot_future"] = _candidate_future(simulation.robot_future_xy, episode.robot_history)
    distance = np.asarray(simulation.future_human_robot_distance, np.float32)
    streams["wm_diagnostic_history"][-1] = np.asarray(
        (.03, .03, .03, np.sqrt(2) * .03, .03, np.mean(distance < .80),
         1.0 / (1.0 + np.linalg.norm(np.full((FUTURE_FRAMES, 3), .03))), episode.confidence.mean()),
        np.float32,
    )
    masks = {name: np.asarray(value).copy() for name, value in template.masks.items()}
    masks["candidate_action"] = np.ones_like(action, bool)
    masks["candidate_robot_future"] = np.ones_like(streams["candidate_robot_future"], bool)
    event = hold.events
    metadata = {
        **template.split_metadata,
        **_shared_metadata(episode, hold),
        "candidate_action_id_audit": HOLD_ACTION_ID,
        "harm_v2_evaluation_only": hold.harm_v2,
        "safe_beneficial_evaluation_only": bool(hold.benefit > 1e-6 and not hold.harm_v2 and hold.feasible),
        "benefit_risk_tradeoff_evaluation_only": bool(hold.benefit > 1e-6 and hold.harm_v2),
        "excessive_deceleration_evaluation_only": event.excessive_deceleration,
        "abrupt_lateral_response_evaluation_only": event.abrupt_lateral_response,
        "abrupt_heading_change_evaluation_only": event.abrupt_heading_change,
        "static_context_108": _hold_static_token(episode, runtime_natural, simulation),
        "old_harm_semantics": DEPRECATED_HARM_TARGET,
    }
    targets = TemporalTargets(
        hold.benefit, hold.benefit < -1e-6, 0.0, False, hold.feasible, hold.gt_total_cost, hold.gt_unsafe
    )
    return RichTemporalSampleV3(
        streams, masks, {name: np.asarray(value).copy() for name, value in template.timestamps.items()},
        targets, f"{episode.episode_id}:{HOLD_ACTION_ID}", episode.episode_id, episode.split,
        episode.split, template.temporal_tags, metadata,
    )


def build_v3_temporal_samples(episodes: Sequence[HarmV2Episode]) -> list[RichTemporalSampleV3]:
    """Build the complete six-candidate v3 development set without TEST access."""
    if any(episode.split not in ("train", "validation") for episode in episodes):
        raise ValueError("manifest-v3 builder refuses TEST episodes")
    v2 = build_v2_temporal_samples(episodes)
    grouped = {episode.episode_id: [] for episode in episodes}
    for sample in v2:
        grouped[sample.episode_id].append(sample)
    result: list[RichTemporalSampleV3] = []
    for episode in episodes:
        hold = build_hold_candidate_outcome(episode, POPULATION_PROFILE, PROFILE_BY_ID[episode.profile_id])
        shared = grouped[episode.episode_id]
        if len(shared) != 5:
            raise RuntimeError("v3 expects exactly five frozen shared candidates per episode")
        result.extend(_convert_shared(sample, episode, hold) for sample in shared)
        result.append(_build_hold(episode, shared[0], hold))
    return result


def v3_runtime_contract_audit(samples: Sequence[RichTemporalSampleV3]) -> dict[str, object]:
    by_episode: dict[str, set[int]] = {}
    for sample in samples:
        by_episode.setdefault(sample.episode_id, set()).add(int(sample.split_metadata["candidate_action_id_audit"]))
    expected = {0, 1, 2, 3, 4, HOLD_ACTION_ID}
    return {
        "sample_count": len(samples),
        "candidate_action_dimension": V3_CANDIDATE_ACTION_DIM,
        "all_action_shapes_valid": all(sample.streams["candidate_action"].shape == (12,) for sample in samples),
        "all_episodes_have_hold": all(actions == expected for actions in by_episode.values()),
        "hold_sample_count": sum(sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for sample in samples),
        "test_samples": sum(sample.split == "test" for sample in samples),
        "test_reads": 0,
        "profile_id_runtime_input": any("person_profile_id" in sample.streams for sample in samples),
        "passed": bool(samples) and all(actions == expected for actions in by_episode.values()),
    }
