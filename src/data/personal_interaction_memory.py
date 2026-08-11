"""Leakage-safe observable interaction memory for synthetic Phase 4B.

The records in this module deliberately contain no simulator profile parameters.
They are observations that would be available after a completed interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from src.data.robot_action_schema import action_feature
from src.data.skeleton_schema import compute_root, global_to_local
from src.data.synthetic_interaction import (
    InteractionSplit,
    PROFILE_BY_ID,
    generate_interaction_split,
)


SupportStrategy = Literal["earliest", "recent", "random", "diverse_action"]
SUPPORTED_K = (0, 1, 3, 5, 10)
OBSERVABLE_INTERACTION_FEATURE_DIM = 28


@dataclass(frozen=True)
class PersonalInteractionRecord:
    person_profile_id: int
    person_instance_id: str
    interaction_id: str
    timestamp: float
    order_index: int
    human_state_before: np.ndarray
    human_root_history: np.ndarray
    human_local_pose_history: np.ndarray
    robot_history: np.ndarray
    executed_action: int
    human_future_response: np.ndarray
    action_effect: np.ndarray
    human_robot_distance_before: float
    human_robot_distance_after: float
    response_delay_observed: float
    split_kind: str
    source_row: int


@dataclass(frozen=True)
class PersonalInteractionCorpus:
    split: InteractionSplit
    records: tuple[PersonalInteractionRecord, ...]
    person_instance_ids: np.ndarray
    order_indices: np.ndarray
    query_indices: np.ndarray
    split_label: str

    def query_records(self) -> tuple[PersonalInteractionRecord, ...]:
        return tuple(self.records[int(index)] for index in self.query_indices)


def _validate_record(record: PersonalInteractionRecord) -> None:
    if record.order_index < 0 or record.timestamp < 0:
        raise ValueError("interaction order and timestamp must be non-negative")
    if record.human_root_history.ndim != 2 or record.human_root_history.shape[-1] != 3:
        raise ValueError("human_root_history must have shape [T,3]")
    if record.human_local_pose_history.ndim != 3 or record.human_local_pose_history.shape[-2:] != (17, 3):
        raise ValueError("human_local_pose_history must have shape [T,17,3]")
    if record.robot_history.ndim != 2 or record.robot_history.shape[-1] != 7:
        raise ValueError("robot_history must have shape [T,7]")
    arrays = (
        record.human_state_before,
        record.human_root_history,
        record.human_local_pose_history,
        record.robot_history,
        record.human_future_response,
        record.action_effect,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("interaction record contains non-finite observations")


class PersonalInteractionMemory:
    """Chronological memory with strict same-person/past-only support selection."""

    def __init__(self, records: Iterable[PersonalInteractionRecord]) -> None:
        ordered = tuple(sorted(records, key=lambda item: (item.person_instance_id, item.order_index)))
        for record in ordered:
            _validate_record(record)
        seen_keys: set[tuple[str, int]] = set()
        for record in ordered:
            key = (record.person_instance_id, record.order_index)
            if key in seen_keys:
                raise ValueError(f"duplicate person/order interaction: {key}")
            seen_keys.add(key)
        self.records = ordered

    def available_before(self, query: PersonalInteractionRecord) -> tuple[PersonalInteractionRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.person_instance_id == query.person_instance_id
            and record.order_index < query.order_index
            and record.timestamp < query.timestamp
            and record.interaction_id != query.interaction_id
        )

    def select_support(
        self,
        query: PersonalInteractionRecord,
        k: int,
        strategy: SupportStrategy = "earliest",
        seed: int = 0,
    ) -> tuple[PersonalInteractionRecord, ...]:
        if k not in SUPPORTED_K:
            raise ValueError(f"k must be one of {SUPPORTED_K}")
        if k == 0:
            return ()
        available = list(self.available_before(query))
        if len(available) < k:
            raise ValueError(
                f"query {query.interaction_id} has {len(available)} prior interactions, needs {k}"
            )
        if strategy == "earliest":
            chosen = available[:k]
        elif strategy == "recent":
            chosen = available[-k:]
        elif strategy == "random":
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(available), size=k, replace=False))
            chosen = [available[int(index)] for index in indices]
        elif strategy == "diverse_action":
            chosen = []
            used_actions: set[int] = set()
            for record in available:
                if record.executed_action not in used_actions:
                    chosen.append(record)
                    used_actions.add(record.executed_action)
                    if len(chosen) == k:
                        break
            if len(chosen) < k:
                chosen_ids = {record.interaction_id for record in chosen}
                chosen.extend(
                    record for record in available if record.interaction_id not in chosen_ids
                )
                chosen = chosen[:k]
            chosen.sort(key=lambda item: item.order_index)
        else:
            raise ValueError(f"unknown support strategy: {strategy}")
        validate_support_query(chosen, query)
        return tuple(chosen)


def validate_support_query(
    support: Iterable[PersonalInteractionRecord], query: PersonalInteractionRecord
) -> None:
    records = tuple(support)
    if any(record.interaction_id == query.interaction_id for record in records):
        raise ValueError("query interaction cannot appear in its own support")
    if any(record.person_instance_id != query.person_instance_id for record in records):
        raise ValueError("support records mix persons or do not match the query person")
    if any(record.order_index >= query.order_index for record in records):
        raise ValueError("support must be strictly earlier than the query")
    if any(record.timestamp >= query.timestamp for record in records):
        raise ValueError("support timestamp must be strictly earlier than query timestamp")


def record_to_observable_feature(record: PersonalInteractionRecord) -> np.ndarray:
    """Encode observations only; profile ID and latent profile parameters are excluded."""
    root_future = compute_root(record.human_future_response)
    effect_root = compute_root(record.action_effect)
    _, response_local = global_to_local(record.human_future_response)
    history_local_last = record.human_local_pose_history[-1]
    feature = np.concatenate(
        (
            action_feature(record.executed_action),                         # 4
            record.human_state_before[:6],                                 # 6
            root_future[-1] - record.human_root_history[-1],               # 3
            effect_root[-1],                                               # 3
            effect_root.mean(axis=0),                                      # 3
            np.asarray((np.linalg.norm(record.action_effect, axis=-1).mean(),)),
            np.asarray((np.linalg.norm(response_local[-1] - history_local_last, axis=-1).mean(),)),
            np.asarray(
                (
                    record.human_robot_distance_before,
                    record.human_robot_distance_after,
                    record.human_robot_distance_after - record.human_robot_distance_before,
                    record.response_delay_observed,
                    float(np.linalg.norm(effect_root[-1, :2])),
                    float(np.linalg.norm(root_future[-1, :2] - root_future[0, :2])),
                    float(record.robot_history[-1, 3]),
                )
            ),                                                             # 7
        )
    ).astype(np.float32)
    if feature.shape != (OBSERVABLE_INTERACTION_FEATURE_DIM,):
        raise RuntimeError(f"observable feature shape regression: {feature.shape}")
    return feature


def support_to_padded_features(
    support: Iterable[PersonalInteractionRecord], max_k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    records = tuple(support)
    if len(records) > max_k:
        raise ValueError("support exceeds max_k")
    features = np.zeros((max_k, OBSERVABLE_INTERACTION_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_k, dtype=bool)
    for index, record in enumerate(records):
        features[index] = record_to_observable_feature(record)
        mask[index] = True
    return features, mask


def _concatenate_splits(parts: list[InteractionSplit], split_kind: str) -> InteractionSplit:
    fields = {}
    for name in InteractionSplit.__dataclass_fields__:
        if name == "split_kind":
            fields[name] = split_kind
        else:
            fields[name] = np.concatenate([getattr(part, name) for part in parts], axis=0)
    return InteractionSplit(**fields)


def subset_interaction_split(split: InteractionSplit, indices: np.ndarray, split_kind: str | None = None) -> InteractionSplit:
    selected = np.asarray(indices, dtype=np.int64)
    fields = {}
    for name in InteractionSplit.__dataclass_fields__:
        fields[name] = (split_kind or split.split_kind) if name == "split_kind" else getattr(split, name)[selected]
    return InteractionSplit(**fields)


def generate_personal_interaction_corpus(
    profile_ids: tuple[int, ...],
    persons_per_profile: int,
    interactions_per_person: int,
    query_start: int,
    seed: int,
    split_label: str,
    state_mode: str = "seen",
    history_frames: int = 20,
    future_frames: int = 10,
    sample_rate_hz: float = 10.0,
    noise_std: float = 0.005,
    occlusion_rate: float = 0.10,
    mask_unseen_combinations: bool = False,
) -> PersonalInteractionCorpus:
    """Generate chronological virtual-person timelines with atomic counterfactual rows."""
    if query_start < 0 or query_start >= interactions_per_person:
        raise ValueError("query_start must leave at least one later query")
    if any(profile_id not in PROFILE_BY_ID for profile_id in profile_ids):
        raise ValueError("unknown profile ID")
    parts: list[InteractionSplit] = []
    person_ids: list[str] = []
    orders: list[int] = []
    records: list[PersonalInteractionRecord] = []
    sequence = np.random.SeedSequence(seed)
    child_seeds = iter(sequence.spawn(len(profile_ids) * persons_per_profile))
    global_row = 0
    for profile_id in profile_ids:
        for person_number in range(persons_per_profile):
            person_id = f"{split_label}_profile{profile_id}_person{person_number}"
            person_seed = int(next(child_seeds).generate_state(1)[0])
            part = generate_interaction_split(
                interactions_per_person,
                person_seed,
                f"{split_label}_{person_id}",
                profile_ids=(profile_id,),
                history_frames=history_frames,
                future_frames=future_frames,
                sample_rate_hz=sample_rate_hz,
                noise_std=noise_std,
                occlusion_rate=occlusion_rate,
                state_mode=state_mode,
                mask_unseen_combinations=mask_unseen_combinations,
            )
            parts.append(part)
            profile = PROFILE_BY_ID[profile_id]
            for order in range(interactions_per_person):
                executed_index = order % part.candidate_actions.shape[1]
                executed_action = int(part.candidate_actions[order, executed_index])
                root_history, local_history = global_to_local(part.human_history[order])
                velocity = (root_history[-1] - root_history[-2]) * sample_rate_hz
                state_before = np.concatenate((root_history[-1], velocity)).astype(np.float32)
                record = PersonalInteractionRecord(
                    person_profile_id=profile_id,
                    person_instance_id=person_id,
                    interaction_id=f"{person_id}_interaction_{order:04d}",
                    timestamp=float(order * (history_frames + future_frames) / sample_rate_hz),
                    order_index=order,
                    human_state_before=state_before,
                    human_root_history=root_history.astype(np.float32),
                    human_local_pose_history=local_history.astype(np.float32),
                    robot_history=part.robot_history[order].astype(np.float32),
                    executed_action=executed_action,
                    human_future_response=part.future_by_action[order, executed_index].astype(np.float32),
                    action_effect=part.action_effect_by_action[order, executed_index].astype(np.float32),
                    human_robot_distance_before=float(part.robot_history[order, -1, 5]),
                    human_robot_distance_after=float(part.future_human_robot_distance[order, executed_index, -1]),
                    response_delay_observed=float(profile.response_delay),
                    split_kind=split_label,
                    source_row=global_row,
                )
                records.append(record)
                person_ids.append(person_id)
                orders.append(order)
                global_row += 1
    merged = _concatenate_splits(parts, split_label)
    order_array = np.asarray(orders, dtype=np.int64)
    query_indices = np.flatnonzero(order_array >= query_start)
    return PersonalInteractionCorpus(
        split=merged,
        records=tuple(records),
        person_instance_ids=np.asarray(person_ids),
        order_indices=order_array,
        query_indices=query_indices,
        split_label=split_label,
    )
