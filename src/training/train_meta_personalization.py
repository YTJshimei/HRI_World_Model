"""Episodic support-query training for synthetic Phase 4B.5."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.personal_interaction_memory import (
    PersonalInteractionCorpus,
    PersonalInteractionMemory,
    SUPPORTED_K,
    support_to_padded_features,
)
from src.data.personalization_diagnostics import (
    RESPONSE_COVERED_PROFILES,
    descriptors_for_split,
    profile_parameter_vector,
)
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import PROFILE_BY_ID
from src.models.personalization_diagnostics import MetaPersonalizedWorldModel
from src.training.train_personal_response import personal_interaction_loss


RESPONSE_STATISTIC_SCALE = np.asarray(
    (1.5, 1.5, 1.2, 1.2, 0.8, 3.5, 0.05), dtype=np.float32
)


class MetaEpisodeDataset(Dataset):
    """Each item is one same-person, past-support -> later-query episode."""

    def __init__(
        self,
        corpus: PersonalInteractionCorpus,
        k_values: int | Sequence[int] = SUPPORTED_K,
        strategy: str = "random",
        seed: int = 42,
        profiles: dict[int, Any] | None = None,
        oracle_access: bool = False,
    ) -> None:
        values = (k_values,) if isinstance(k_values, int) else tuple(k_values)
        if not values or any(value not in SUPPORTED_K for value in values):
            raise ValueError(f"invalid K values: {values}")
        self.corpus = corpus
        self.k_values = values
        self.strategy = strategy
        self.seed = seed
        self.epoch = 0
        self.oracle_access = bool(oracle_access)
        self.memory = PersonalInteractionMemory(corpus.records)
        covered = {profile.profile_id: profile for profile in RESPONSE_COVERED_PROFILES}
        self.profiles = profiles or {**PROFILE_BY_ID, **covered}
        self.effect_descriptors = (
            descriptors_for_split(corpus.split, self.profiles)
            if self.oracle_access else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.corpus.query_indices) * len(self.k_values)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        query_position = item // len(self.k_values)
        k = int(self.k_values[item % len(self.k_values)])
        source = int(self.corpus.query_indices[query_position])
        record = self.corpus.records[source]
        support = self.memory.select_support(
            record, k, self.strategy,
            seed=self.seed + self.epoch * 1_000_003 + source * 97 + k,
        )
        if any(item.person_instance_id != record.person_instance_id for item in support):
            raise RuntimeError("meta episode support/query person mismatch")
        support_features, support_mask = support_to_padded_features(support)
        split = self.corpus.split
        result = {
            "history": torch.from_numpy(split.human_history[source]),
            "natural": torch.from_numpy(split.natural_future[source]),
            "target": torch.from_numpy(split.future_by_action[source]),
            "robot": torch.from_numpy(split.robot_history[source]),
            "actions": torch.from_numpy(split.candidate_actions[source]),
            "confidence": torch.from_numpy(split.confidence[source]),
            "visibility": torch.from_numpy(split.visibility_mask[source]),
            "supervision": torch.from_numpy(split.action_supervision_mask[source]),
            "support_features": torch.from_numpy(support_features),
            "support_mask": torch.from_numpy(support_mask),
            "source_index": torch.tensor(source, dtype=torch.long),
            "profile_id": torch.tensor(record.person_profile_id, dtype=torch.long),
            "k": torch.tensor(k, dtype=torch.long),
        }
        if self.oracle_access:
            profile = self.profiles[int(record.person_profile_id)]
            effect = split.action_effect_by_action[source]
            nonkeep = split.candidate_actions[source] != 0
            sensitivity = float(np.linalg.norm(effect[nonkeep], axis=-1).mean())
            response_statistics = np.asarray(
                (
                    profile.speed_response_gain,
                    profile.distance_sensitivity,
                    profile.lateral_avoidance_gain,
                    profile.turn_sensitivity,
                    profile.response_delay,
                    profile.adaptation_rate,
                    sensitivity,
                ), dtype=np.float32,
            ) / RESPONSE_STATISTIC_SCALE
            result.update({
                "profile_parameters": torch.from_numpy(profile_parameter_vector(profile)),
                "effect_descriptors": torch.from_numpy(self.effect_descriptors[source]),
                "response_statistics": torch.from_numpy(response_statistics),
            })
        return result


@dataclass(frozen=True)
class MetaLossWeights:
    amplitude: float = 0.0
    personal_gain: float = 0.0
    personal_gain_margin: float = 0.001


def amplitude_loss(
    prediction: torch.Tensor,
    natural_prediction: torch.Tensor,
    target: torch.Tensor,
    natural_target: torch.Tensor,
    action_ids: torch.Tensor,
    supervision: torch.Tensor,
) -> torch.Tensor:
    predicted_effect = prediction - natural_prediction[:, None]
    target_effect = target - natural_target[:, None]
    predicted_magnitude = torch.linalg.vector_norm(predicted_effect, dim=-1).mean(dim=(-1, -2))
    target_magnitude = torch.linalg.vector_norm(target_effect, dim=-1).mean(dim=(-1, -2))
    mask = supervision & (action_ids != 0)
    weights = mask.to(prediction.dtype)
    return (((predicted_magnitude - target_magnitude).square()) * weights).sum() / weights.sum().clamp_min(1.0)


def personalization_gain_loss(
    personalized: torch.Tensor,
    generic: torch.Tensor,
    target: torch.Tensor,
    support_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    personal_error = torch.linalg.vector_norm(personalized - target, dim=-1).mean(dim=(-1, -2, -3))
    generic_error = torch.linalg.vector_norm(generic - target, dim=-1).mean(dim=(-1, -2, -3))
    valid = support_mask.any(dim=1)
    if not bool(valid.any()):
        return personalized.sum() * 0.0
    return torch.relu(personal_error[valid] - generic_error[valid] + margin).mean()


def meta_query_loss(
    personalized: Any,
    generic: Any,
    batch: dict[str, torch.Tensor],
    weights: MetaLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    absolute, components = personal_interaction_loss(
        personalized, batch["target"], batch["natural"], batch["supervision"],
        batch["actions"],
    )
    amplitude = amplitude_loss(
        personalized.future_by_action, personalized.natural_future,
        batch["target"], batch["natural"], batch["actions"], batch["supervision"],
    )
    gain = personalization_gain_loss(
        personalized.future_by_action, generic.future_by_action, batch["target"],
        batch["support_mask"], weights.personal_gain_margin,
    )
    total = absolute + weights.amplitude * amplitude + weights.personal_gain * gain
    return total, {**components, "amplitude": amplitude, "personal_gain": gain}


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.inference_mode()
def meta_validation_mpjpe(model: MetaPersonalizedWorldModel, loader: Any, device: torch.device) -> float:
    model.eval()
    total = count = 0.0
    for raw in loader:
        batch = _move(raw, device)
        personalized, _ = model.paired_forward(
            batch["history"], batch["robot"], batch["actions"],
            batch["confidence"], batch["visibility"],
            batch["support_features"], batch["support_mask"],
        )
        error = torch.linalg.vector_norm(
            personalized.future_by_action - batch["target"], dim=-1
        ).mean(dim=(-1, -2))
        mask = batch["supervision"].to(error.dtype)
        total += float((error * mask).sum().item())
        count += float(mask.sum().item())
    return total / max(count, 1.0)


def train_meta_model(
    model: MetaPersonalizedWorldModel,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    weights: MetaLossWeights,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        for raw in train_loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            personalized, generic = model.paired_forward(
                batch["history"], batch["robot"], batch["actions"],
                batch["confidence"], batch["visibility"],
                batch["support_features"], batch["support_mask"],
            )
            loss, _ = meta_query_loss(personalized, generic, batch, weights)
            loss.backward()
            optimizer.step()
        validation = meta_validation_mpjpe(model, validation_loader, device)
        if validation < best:
            best, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_Global_MPJPE": best,
                "weights": weights.__dict__,
            }, path)
    if best_state is None:
        raise RuntimeError("meta training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_Global_MPJPE": best,
        "training_time_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
