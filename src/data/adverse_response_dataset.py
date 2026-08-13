"""Leakage-safe synthetic adverse-response episode protocol for Phase 5B-1.7C."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.data.adverse_response_protocol import AdverseResponseEvents, derive_adverse_response_events
from src.data.robot_action_schema import PHASE4A_ACTIONS
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import (
    PROFILE_BY_ID, VIRTUAL_PERSON_PROFILES, AdverseResponseRiskFactors,
    VirtualPersonProfile, generate_interaction_split,
    sample_adverse_response_risk_factors, simulate_interaction_future,
    simulate_risk_conditioned_interaction_future,
)
from src.decision.candidate_action import TASK_SAFE_CANDIDATES
from src.decision.counterfactual_rollout import CounterfactualRollout
from src.decision.decision_cost import DecisionCostWeights, compute_decision_costs
from src.decision.decision_state import DecisionState, FunctionalResponseBelief
from src.multimodal.temporal_dataset import apply_continuous_occlusion

LABEL = "SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
GENERATOR_VERSION = "phase5b17c_adverse_response_generator_v1"
GENERATOR_SEED = 51_703
RISK_SEED = 77_117
DEVELOPMENT_PROFILE_IDS = (0, 1, 3, 5)
HELD_OUT_PROFILE_IDS = (2, 6)
ACTION_IDS = tuple(int(action) for action in PHASE4A_ACTIONS)


@dataclass(frozen=True)
class HarmV2Candidate:
    action_id: int
    feasible: bool
    benefit: float
    gt_unsafe: bool
    adverse_response: bool
    harm_v2: bool
    unsafe_duration: float
    minimum_distance_m: float
    events: AdverseResponseEvents


@dataclass(frozen=True)
class HarmV2Episode:
    episode_id: str
    split: str
    profile_id: int
    motion_type: str
    context_labels: tuple[str, ...]
    risk_factors: AdverseResponseRiskFactors
    generation_seed: int
    target_follow_distance: float
    history_occlusion_rate: float
    candidates: tuple[HarmV2Candidate, ...]
    # Development-only deterministic replay payload.  These values are never
    # serialized into manifest_v2 and TEST never calls this builder.
    human_history: np.ndarray
    robot_history: np.ndarray
    confidence: np.ndarray
    visibility: np.ndarray
    natural_future: np.ndarray
    generic_simulations: tuple[object, ...]
    generic_costs: np.ndarray
    gt_costs: np.ndarray
    generic_action_index: int
    support_count: int


def _population_profile() -> VirtualPersonProfile:
    profiles = [PROFILE_BY_ID[index] for index in DEVELOPMENT_PROFILE_IDS]
    fields = ("preferred_distance", "distance_sensitivity", "speed_response_gain", "response_delay",
              "lateral_avoidance_gain", "turn_sensitivity", "adaptation_rate")
    values = [float(np.mean([getattr(profile, name) for profile in profiles])) for name in fields]
    return VirtualPersonProfile(-1, "development_population", *values)


POPULATION_PROFILE = _population_profile()


def _set_robot_distance(history: np.ndarray, robot: np.ndarray, distance: float) -> np.ndarray:
    result = np.asarray(robot, dtype=np.float32).copy(); roots = compute_root(history)
    direction = roots[-1, :2] - result[-1, :2]
    norm = float(np.linalg.norm(direction)); direction = direction / norm if norm > 1e-8 else np.asarray((1.0, 0.0))
    result[:, :2] = roots[:, :2] - distance * direction[None]
    result[:, 5] = distance
    relative = np.arctan2(direction[1], direction[0]) - result[:, 2]
    result[:, 6] = np.arctan2(np.sin(relative), np.cos(relative))
    return result


def _rollout(action_ids, natural, simulations) -> CounterfactualRollout:
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


def _contexts(motion: str, support_count: int, c7: bool) -> tuple[str, ...]:
    result = []
    if c7: result.append("C7_long_occlusion_history")
    if support_count > 0: result.append("C8_recent_intervention_transition")
    if motion in ("acceleration", "deceleration", "left_turn", "right_turn", "stop"):
        result.append("C9_motion_transition")
    return tuple(result)


def _c7_mask(visibility, confidence, episode_index):
    joint_groups = ((11, 13, 15), (12, 14, 16), (5, 7, 9), (6, 8, 10), (11, 12, 13, 14, 15, 16))
    joints = joint_groups[episode_index % len(joint_groups)]
    frames = 5 + episode_index % 4; start = 2 + episode_index % max(1, 20 - frames - 2)
    return apply_continuous_occlusion(visibility, confidence, start=start, frames=frames, joints=joints)


def build_development_split(
    split: str, size: int, seed: int, risk_seed: int,
    profile_ids: tuple[int, ...] = DEVELOPMENT_PROFILE_IDS,
) -> list[HarmV2Episode]:
    """Generate independent episodes, then branch candidates within each split."""
    if split not in ("train", "validation"):
        raise ValueError("readiness builder exposes TRAIN/VALIDATION only")
    if set(profile_ids) & set(HELD_OUT_PROFILE_IDS):
        raise ValueError("held-out profiles cannot enter development splits")
    base = generate_interaction_split(size, seed, f"phase5b17c_{split}", profile_ids=profile_ids,
                                      noise_std=.005, occlusion_rate=.10)
    rng = np.random.default_rng(risk_seed); episodes = []
    for index in range(size):
        # Independent state/risk sampling; no episode or candidate duplication.
        distance = float(rng.uniform(.88, 2.25)); target = float(rng.uniform(1.40, 1.60))
        robot = _set_robot_distance(base.human_history[index], base.robot_history[index], distance)
        risk = sample_adverse_response_risk_factors(rng); profile = PROFILE_BY_ID[int(base.person_profile_id[index])]
        support_count = int(rng.integers(0, 5)); c7 = index % 3 == 0
        visibility, confidence = base.visibility_mask[index], base.confidence[index]
        if c7: visibility, confidence = _c7_mask(visibility, confidence, index)
        motion = str(base.action_type[index]); contexts = _contexts(motion, support_count, c7)
        state = DecisionState(
            base.human_history[index], robot, confidence, visibility,
            FunctionalResponseBelief(np.ones(6, np.float32), np.zeros(6, np.float32)),
            TASK_SAFE_CANDIDATES, target, .80, f"phase5b17c_{split}",
        )
        generic_simulations = [simulate_interaction_future(base.human_history[index], base.natural_future[index], robot, action, POPULATION_PROFILE) for action in ACTION_IDS]
        gt_simulations = [simulate_risk_conditioned_interaction_future(base.human_history[index], base.natural_future[index], robot, action, profile, risk) for action in ACTION_IDS]
        generic_costs = compute_decision_costs(state, _rollout(ACTION_IDS, base.natural_future[index], generic_simulations), DecisionCostWeights(), include_uncertainty=False)
        gt_costs = compute_decision_costs(state, _rollout(ACTION_IDS, base.natural_future[index], gt_simulations), DecisionCostWeights(), include_uncertainty=False)
        generic_index = int(np.argmin(generic_costs.total)); candidates = []
        for action_index, (action_id, simulation) in enumerate(zip(ACTION_IDS, gt_simulations)):
            event = derive_adverse_response_events(base.human_history[index], base.natural_future[index], simulation.future_global)
            unsafe = bool(gt_costs.unsafe_duration[action_index] > 0.0)
            adverse = bool(event.adverse_human_kinematic_response)
            candidates.append(HarmV2Candidate(
                action_id, True, float(gt_costs.total[generic_index] - gt_costs.total[action_index]),
                unsafe, adverse, unsafe or adverse, float(gt_costs.unsafe_duration[action_index]),
                float(gt_costs.minimum_distance[action_index]), event,
            ))
        episodes.append(HarmV2Episode(
            f"{split}:phase5b17c:{seed}:{index:06d}", split, int(base.person_profile_id[index]), motion,
            contexts, risk, seed, target, float(1.0 - visibility.mean()), tuple(candidates),
            np.asarray(base.human_history[index], np.float32).copy(), np.asarray(robot, np.float32).copy(),
            np.asarray(confidence, np.float32).copy(), np.asarray(visibility, bool).copy(),
            np.asarray(base.natural_future[index], np.float32).copy(), tuple(generic_simulations),
            np.asarray(generic_costs.total, np.float32).copy(), np.asarray(gt_costs.total, np.float32).copy(),
            generic_index, support_count,
        ))
    return episodes


def sealed_test_manifest_rows(size: int, seed: int, risk_seed: int) -> list[dict[str, object]]:
    """Assign sealed TEST IDs without materializing trajectories, benefit or labels."""
    rng = np.random.default_rng(risk_seed); motions = ("walk", "run", "acceleration", "deceleration", "left_turn", "right_turn")
    rows = []
    for index in range(size):
        profile = HELD_OUT_PROFILE_IDS[index % len(HELD_OUT_PROFILE_IDS)]
        risk = sample_adverse_response_risk_factors(rng); c7 = index % 3 == 0
        motion = motions[index % len(motions)]; contexts = _contexts(motion, index % 5, c7)
        episode_id = f"test:phase5b17c:{seed}:{index:06d}"
        rows.append({"episode_id": episode_id, "split": "test", "profile_id_split_only": profile,
                     "motion_type_evaluation_only": motion, "context_labels": list(contexts),
                     "risk_factor_metadata": asdict(risk), "generation_seed": seed,
                     "candidate_ids": [f"{episode_id}:{action}" for action in ACTION_IDS],
                     "harm_v2_labels": "SEALED_NOT_MATERIALIZED", "benefit_labels": "SEALED_NOT_MATERIALIZED"})
    return rows


def episode_manifest_row(episode: HarmV2Episode) -> dict[str, object]:
    return {"episode_id": episode.episode_id, "split": episode.split,
            "profile_id_split_only": episode.profile_id, "motion_type_evaluation_only": episode.motion_type,
            "context_labels": list(episode.context_labels), "risk_factor_metadata": asdict(episode.risk_factors),
            "generation_seed": episode.generation_seed,
            "candidate_ids": [f"{episode.episode_id}:{candidate.action_id}" for candidate in episode.candidates],
            "harm_v2_labels": {str(candidate.action_id): candidate.harm_v2 for candidate in episode.candidates},
            "gt_unsafe_labels": {str(candidate.action_id): candidate.gt_unsafe for candidate in episode.candidates},
            "adverse_response_labels": {str(candidate.action_id): candidate.adverse_response for candidate in episode.candidates}}


def candidate_rows(episodes: list[HarmV2Episode]) -> list[dict[str, object]]:
    rows = []
    for episode in episodes:
        for candidate in episode.candidates:
            rows.append({"episode_id": episode.episode_id, "split": episode.split, "profile": episode.profile_id,
                         "motion": episode.motion_type, "contexts": episode.context_labels, "action": candidate.action_id,
                         "feasible": candidate.feasible, "benefit": candidate.benefit, "gt_unsafe": candidate.gt_unsafe,
                         "adverse_response": candidate.adverse_response, "harm_v2": candidate.harm_v2,
                         "unsafe_duration": candidate.unsafe_duration, "minimum_distance_m": candidate.minimum_distance_m,
                         **asdict(candidate.events)})
    return rows
