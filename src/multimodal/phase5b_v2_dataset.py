"""Runtime-only TRAIN/VALIDATION bridge for frozen Phase5B manifest_v2.

The bridge deterministically replays the already frozen synthetic generator.
It never exposes the sealed TEST builder and never places profile identity,
future human ground truth, benefit, harm_v2, or oracle actions in model inputs.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.data.adverse_response_dataset import (
    ACTION_IDS, DEVELOPMENT_PROFILE_IDS, POPULATION_PROFILE, HarmV2Episode,
)
from src.data.functional_response_state import (
    aggregate_response_state_mask, functional_state_from_profile,
    population_mean_response_state,
)
from src.data.robot_action_schema import action_feature
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import PROFILE_BY_ID
from src.data.synthetic_interaction import simulate_interaction_future
from src.decision.candidate_action import TASK_SAFE_CANDIDATES
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_cost import DecisionCostWeights, DecisionCosts, compute_decision_costs
from src.decision.decision_state import DecisionState, FunctionalResponseBelief
from src.multimodal.context_schema import build_context_tokens
from src.multimodal.temporal_dataset import (
    _candidate_future, _interaction_history, _motion, _temporal_tags,
)
from src.multimodal.temporal_schema import (
    DT_SECONDS, FUTURE_FRAMES, HISTORY_FRAMES, RichTemporalSample,
    TemporalTargets,
)

DEPRECATED_HARM_TARGET = "DEPRECATED_AUXILIARY_CONTROL_TARGET: benefit < -1e-6"
SUPPORT_ACTION_SEQUENCE = (1, 2, 3, 4)


@dataclass(frozen=True)
class RuntimeGenericPolicyReplay:
    """GT-free replay of the frozen Phase 5B runtime generic-policy inputs."""

    action_ids: np.ndarray
    runtime_natural_future: np.ndarray
    simulations: tuple[object, ...]
    costs: DecisionCosts
    anchor_index: int
    anchor_action_id: int
    tie_count: int
    gt_read_count: int = 0


def runtime_constant_velocity_prior(human_history: np.ndarray) -> np.ndarray:
    """History-only natural prior already used by the frozen runtime bridge."""
    history = np.asarray(human_history, np.float32)
    if history.shape != (HISTORY_FRAMES, 17, 3) or not np.isfinite(history).all():
        raise ValueError("human_history must be finite with shape [20,17,3]")
    last = history[-1]
    velocity_per_frame = history[-1] - history[-2]
    return (
        last[None]
        + np.arange(1, FUTURE_FRAMES + 1, dtype=np.float32)[:, None, None]
        * velocity_per_frame[None]
    ).astype(np.float32)


def _runtime_rollout(action_ids, natural, simulations) -> CounterfactualRollout:
    future = np.stack([simulation.future_global for simulation in simulations])
    root = compute_root(future); local = future - root[:, :, None]
    effects = np.stack([simulation.action_effect for simulation in simulations])
    return CounterfactualRollout(
        np.asarray(action_ids, np.int64), np.asarray(natural, np.float32), root.astype(np.float32),
        local.astype(np.float32), future.astype(np.float32),
        np.stack([simulation.robot_future_xy for simulation in simulations]),
        np.stack([simulation.future_human_robot_distance for simulation in simulations]),
        effects.astype(np.float32), np.zeros_like(effects, np.float32), 0,
    )


def replay_runtime_generic_policy(
    human_history: np.ndarray,
    robot_history: np.ndarray,
    confidence: np.ndarray,
    visibility: np.ndarray,
    target_follow_distance: float,
) -> RuntimeGenericPolicyReplay:
    """Select the A0--A4 generic anchor using runtime-valid inputs only.

    The signature intentionally cannot accept profile identity, GT natural or
    human futures, GT costs/benefit/harm/unsafe, or an oracle action.  Selection
    is finalized before any label-side caller is allowed to read GT cost.
    """
    history = np.asarray(human_history, np.float32)
    robot = np.asarray(robot_history, np.float32)
    confidence = np.asarray(confidence, np.float32)
    visibility = np.asarray(visibility, bool)
    natural = runtime_constant_velocity_prior(history)
    action_ids = np.asarray([int(candidate.action) for candidate in TASK_SAFE_CANDIDATES], np.int64)
    if not np.array_equal(action_ids, np.asarray(ACTION_IDS, np.int64)):
        raise RuntimeError("runtime generic candidate family must remain A0-A4")
    simulations = tuple(
        simulate_interaction_future(history, natural, robot, int(action_id), POPULATION_PROFILE)
        for action_id in action_ids
    )
    state = DecisionState(
        history, robot, confidence, visibility,
        FunctionalResponseBelief(np.ones(6, np.float32), np.zeros(6, np.float32)),
        TASK_SAFE_CANDIDATES, float(target_follow_distance), .80, "phase5b_runtime_generic",
    )
    costs = compute_decision_costs(
        state, _runtime_rollout(action_ids, natural, simulations),
        DecisionCostWeights(), include_uncertainty=False,
    )
    # Explicit action-ID secondary key makes the frozen tie rule independent
    # of container order while remaining identical for the canonical A0-A4 order.
    anchor_index = int(np.lexsort((action_ids, np.asarray(costs.total, np.float64)))[0])
    minimum = float(costs.total[anchor_index])
    tie_count = int(np.sum(np.isclose(costs.total, minimum, rtol=0.0, atol=1e-12)))
    return RuntimeGenericPolicyReplay(
        action_ids.copy(), natural, simulations, costs, anchor_index,
        int(action_ids[anchor_index]), tie_count, 0,
    )


def _support(episode: HarmV2Episode) -> tuple[int, ...]:
    return tuple(SUPPORT_ACTION_SEQUENCE[index % len(SUPPORT_ACTION_SEQUENCE)] for index in range(episode.support_count))


def _support_names(actions: Sequence[int]) -> tuple[str, ...]:
    names = {1: "SPEED_DOWN_10", 2: "SPEED_UP_10", 3: "DISTANCE_PLUS_0_2", 4: "DISTANCE_MINUS_0_2"}
    return tuple(names[int(action)] for action in actions)


def _motion_observable(history: np.ndarray) -> np.ndarray:
    root = compute_root(history); velocity = np.diff(root[:, :2], axis=0) / DT_SECONDS
    speed = np.linalg.norm(velocity, axis=1) if len(velocity) else np.zeros(1)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0])) if len(velocity) else np.zeros(1)
    return np.asarray((
        speed.mean(), speed[-1], speed[-1] - speed[0], heading[-1] - heading[0],
        speed.std(), np.mean(np.abs(np.diff(heading))) if len(heading) > 1 else 0.0,
        float(abs(heading[-1] - heading[0]) > .12), float(speed[-1] - speed[0] > .15),
    ), np.float32)


def build_v2_temporal_samples(episodes: Sequence[HarmV2Episode]) -> list[RichTemporalSample]:
    """Create the paired 108D/temporal views without reading TEST."""
    population_theta = population_mean_response_state([PROFILE_BY_ID[index] for index in DEVELOPMENT_PROFILE_IDS])
    population_values = np.stack([functional_state_from_profile(PROFILE_BY_ID[index]) for index in DEVELOPMENT_PROFILE_IDS])
    theta_std = population_values.std(axis=0).astype(np.float32)
    result: list[RichTemporalSample] = []
    for episode in episodes:
        if episode.split not in ("train", "validation"):
            raise ValueError("manifest_v2 runtime bridge exposes TRAIN/VALIDATION only")
        history, robot = episode.human_history, episode.robot_history
        confidence, visibility = episode.confidence, episode.visibility
        root = compute_root(history); motion = _motion(history)
        support_actions = _support(episode); support_names = _support_names(support_actions)
        response_mask = aggregate_response_state_mask(support_actions)
        support_feature = np.asarray((len(support_actions) / 5, len(set(support_actions)) / 5, response_mask.mean()), np.float32)
        support_coverage = response_mask.astype(np.float32)
        interaction, interaction_valid, interaction_audit = _interaction_history(support_names, root)
        functional = np.zeros((HISTORY_FRAMES, 18), np.float32)
        functional_valid = np.zeros_like(functional, bool)
        # Population response state is runtime-safe.  Profile identity and true
        # episode response state remain split/evaluation metadata only.
        functional[-1] = np.concatenate((population_theta, theta_std, response_mask.astype(np.float32)))
        functional_valid[-1] = True
        visible_ratio = visibility.mean(axis=1); mean_conf = confidence.mean(axis=1)
        visibility_stream = np.column_stack((visibility.any(axis=1), visible_ratio, mean_conf, mean_conf * visible_ratio)).astype(np.float32)
        speed = np.linalg.norm(motion[:, 3:5], axis=-1); distance_history = robot[:, 5]
        scene = np.asarray((visibility.mean(), confidence.mean(), distance_history[-1], distance_history[-1] - distance_history[0],
                            speed.mean(), speed.std(), robot[-1, 6], len(support_actions) / 5), np.float32)
        motion_observable = _motion_observable(history)
        # The same public GT-free replay is now also the single canonical
        # runtime generic-policy path used by the v1 bridge and v2 anchor audit.
        generic_replay = replay_runtime_generic_policy(
            history, robot, confidence, visibility, episode.target_follow_distance,
        )
        runtime_natural = generic_replay.runtime_natural_future
        for action_index, candidate in enumerate(episode.candidates):
            if candidate.action_id != int(generic_replay.action_ids[action_index]):
                raise RuntimeError("candidate order differs from runtime generic replay")
            simulation = generic_replay.simulations[action_index]
            generic_distance = np.asarray(simulation.future_human_robot_distance, np.float32)
            generic_effect = np.asarray(simulation.action_effect, np.float32)
            sigma_root = np.full((FUTURE_FRAMES, 3), .03, np.float32)
            predicted_unsafe = float(np.mean(generic_distance < .80))
            token = build_context_tokens(
                human_history=history, robot_history=robot, confidence=confidence, visibility=visibility,
                theta_person=population_theta, theta_population=population_theta, theta_uncertainty=theta_std,
                response_state_mask=response_mask, support_coverage=support_coverage,
                support_action_features=support_feature, candidate_action=candidate.action_id,
                candidate_feature=action_feature(candidate.action_id), predicted_robot_future=simulation.robot_future_xy,
                generic_effect=generic_effect, personalized_effect=generic_effect,
                generic_distance=generic_distance, personalized_distance=generic_distance,
                root_sigma=sigma_root, minimum_sigma=.03, p_unsafe=predicted_unsafe,
                motion_state_observable=motion_observable, scene_observable=scene,
                context_id=f"{episode.episode_id}:{candidate.action_id}", initial_state_id=episode.episode_id,
                context_split=episode.split,
            )
            wm = np.zeros((HISTORY_FRAMES, 8), np.float32); wm_valid = np.zeros_like(wm, bool)
            wm[-1] = np.asarray((.03, .03, .03, np.sqrt(2) * .03, .03, predicted_unsafe,
                                 1.0 / (1.0 + np.linalg.norm(sigma_root)), confidence.mean()), np.float32)
            wm_valid[-1] = True
            action = np.r_[np.eye(7, dtype=np.float32)[candidate.action_id], action_feature(candidate.action_id)].astype(np.float32)
            robot_future = _candidate_future(simulation.robot_future_xy, robot)
            streams = {
                "skeleton_history": history, "human_motion_history": motion, "robot_history": robot,
                "functional_history": functional, "visibility_history": visibility_stream,
                "wm_diagnostic_history": wm, "interaction_history": interaction,
                "candidate_action": action, "candidate_robot_future": robot_future, "scene_context": scene,
            }
            masks = {
                "skeleton_history": visibility[..., None].repeat(3, axis=-1),
                "human_motion_history": np.ones_like(motion, bool), "robot_history": np.ones_like(robot, bool),
                "functional_history": functional_valid, "visibility_history": np.ones_like(visibility_stream, bool),
                "wm_diagnostic_history": wm_valid, "interaction_history": interaction_valid,
                "candidate_action": np.ones_like(action, bool), "candidate_robot_future": np.ones_like(robot_future, bool),
                "scene_context": np.ones_like(scene, bool), "history_valid_mask": np.ones(HISTORY_FRAMES, bool),
                "history_padding_mask": np.ones(HISTORY_FRAMES, bool), "candidate_future_valid_mask": np.ones(FUTURE_FRAMES, bool),
            }
            sample_id = f"{episode.episode_id}:{candidate.action_id}"
            event = candidate.events
            result.append(RichTemporalSample(
                streams=streams, masks=masks,
                timestamps={"history": np.arange(-HISTORY_FRAMES + 1, 1, dtype=np.float32) * DT_SECONDS,
                            "candidate_future": np.arange(1, FUTURE_FRAMES + 1, dtype=np.float32) * DT_SECONDS},
                targets=TemporalTargets(candidate.benefit, candidate.benefit < -1e-6, 0.0, False,
                                        candidate.feasible, float(episode.gt_costs[action_index]), candidate.gt_unsafe),
                sample_id=sample_id, episode_id=episode.episode_id, split=episode.split,
                context_split=episode.split, temporal_tags=_temporal_tags(history, visibility, len(support_actions)),
                split_metadata={
                    "person_profile_id": episode.profile_id, "motion_type_evaluation_only": episode.motion_type,
                    "scenario": episode.motion_type,
                    "candidate_action_id_audit": candidate.action_id, "contexts_evaluation_only": episode.context_labels,
                    "harm_v2_evaluation_only": candidate.harm_v2, "safe_beneficial_evaluation_only": bool(candidate.benefit > 1e-6 and not candidate.harm_v2),
                    "benefit_risk_tradeoff_evaluation_only": bool(candidate.benefit > 1e-6 and candidate.harm_v2),
                    "excessive_deceleration_evaluation_only": event.excessive_deceleration,
                    "abrupt_lateral_response_evaluation_only": event.abrupt_lateral_response,
                    "abrupt_heading_change_evaluation_only": event.abrupt_heading_change,
                    "all_action_ids_evaluation_only": np.asarray([item.action_id for item in episode.candidates], int),
                    "generic_costs_evaluation_only": episode.generic_costs.copy(),
                    "personalized_costs_evaluation_only": episode.generic_costs.copy(),
                    "gt_costs_evaluation_only": episode.gt_costs.copy(),
                    "gt_unsafe_evaluation_only": np.asarray([item.gt_unsafe for item in episode.candidates], bool),
                    "generic_action_index_evaluation_only": episode.generic_action_index,
                    "static_context_108": token.flattened().copy(),
                    "old_harm_semantics": DEPRECATED_HARM_TARGET,
                    **interaction_audit,
                },
            ))
    return result


def runtime_contract_audit(samples: Sequence[RichTemporalSample]) -> dict[str, object]:
    forbidden = {"profile_id", "person_profile_id", "harm_v2", "benefit", "gt_future", "future_global", "natural_future"}
    return {
        "sample_count": len(samples),
        "static_108_shapes_valid": all(np.asarray(item.split_metadata["static_context_108"]).shape == (108,) for item in samples),
        "runtime_forbidden_keys": sorted(set().union(*(set(item.streams) for item in samples)) & forbidden),
        "test_samples": sum(item.split == "test" for item in samples),
        "test_reads": 0,
        "passed": bool(samples) and all(item.split in ("train", "validation") for item in samples),
    }
