"""Leakage-safe episodic training for functional response identification."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.functional_response_state import (
    RESPONSE_STATE_SCALE, functional_state_from_profile,
)
from src.data.personal_interaction_memory import (
    PersonalInteractionCorpus, PersonalInteractionMemory, SUPPORTED_K,
    validate_support_query,
)
from src.data.personalization_diagnostics import RESPONSE_COVERED_PROFILES
from src.data.response_statistics import pad_response_statistics
from src.data.synthetic_interaction import PROFILE_BY_ID


class FunctionalEpisodeDataset(Dataset):
    def __init__(
        self,
        corpus: PersonalInteractionCorpus,
        k_values: int | Sequence[int],
        support_type: str = "random",
        seed: int = 42,
        profiles: dict[int, Any] | None = None,
    ) -> None:
        values = (k_values,) if isinstance(k_values, int) else tuple(k_values)
        if not values or any(value not in SUPPORTED_K for value in values):
            raise ValueError(f"invalid K values: {values}")
        if support_type not in ("random", "earliest", "speed_only", "distance_only", "diverse_action"):
            raise ValueError("unknown support type")
        self.corpus = corpus
        self.k_values = values
        self.support_type = support_type
        self.seed = seed
        self.epoch = 0
        self.memory = PersonalInteractionMemory(corpus.records)
        covered = {profile.profile_id: profile for profile in RESPONSE_COVERED_PROFILES}
        self._oracle_targets = profiles or {**PROFILE_BY_ID, **covered}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.corpus.query_indices) * len(self.k_values)

    def _support(self, query: Any, k: int, source: int) -> tuple[Any, ...]:
        if k == 0:
            return ()
        available = list(self.memory.available_before(query))
        if self.support_type == "speed_only":
            available = [item for item in available if item.executed_action in (1, 2)]
        elif self.support_type == "distance_only":
            available = [item for item in available if item.executed_action in (3, 4)]
        if len(available) < k:
            raise ValueError(f"insufficient {self.support_type} support for K={k}")
        if self.support_type == "earliest" or self.support_type.endswith("_only"):
            chosen = available[:k]
        elif self.support_type == "random":
            rng = np.random.default_rng(
                self.seed + self.epoch * 1_000_003 + source * 97 + k
            )
            indices = np.sort(rng.choice(len(available), size=k, replace=False))
            chosen = [available[int(index)] for index in indices]
        else:
            chosen, used = [], set()
            for item in available:
                if item.executed_action != 0 and item.executed_action not in used:
                    chosen.append(item); used.add(item.executed_action)
                    if len(chosen) == k:
                        break
            chosen_ids = {item.interaction_id for item in chosen}
            chosen.extend(
                item for item in available if item.interaction_id not in chosen_ids
            )
            chosen = chosen[:k]
        validate_support_query(chosen, query)
        return tuple(chosen)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        query_position = item // len(self.k_values)
        k = int(self.k_values[item % len(self.k_values)])
        source = int(self.corpus.query_indices[query_position])
        query = self.corpus.records[source]
        support = self._support(query, k, source)
        statistics, support_mask, state_mask = pad_response_statistics(support)
        split = self.corpus.split
        # GT state is a supervision target only; it is never an F2 model input.
        theta_target = functional_state_from_profile(
            self._oracle_targets[int(query.person_profile_id)]
        )
        return {
            "history": torch.from_numpy(split.human_history[source]),
            "natural": torch.from_numpy(split.natural_future[source]),
            "target": torch.from_numpy(split.future_by_action[source]),
            "robot": torch.from_numpy(split.robot_history[source]),
            "actions": torch.from_numpy(split.candidate_actions[source]),
            "confidence": torch.from_numpy(split.confidence[source]),
            "visibility": torch.from_numpy(split.visibility_mask[source]),
            "supervision": torch.from_numpy(split.action_supervision_mask[source]),
            "statistics": torch.from_numpy(statistics),
            "support_mask": torch.from_numpy(support_mask),
            "response_state_mask": torch.from_numpy(state_mask),
            "theta_target": torch.from_numpy(theta_target),
            "profile_id": torch.tensor(query.person_profile_id, dtype=torch.long),
            "source_index": torch.tensor(source, dtype=torch.long),
            "k": torch.tensor(k, dtype=torch.long),
        }


@dataclass(frozen=True)
class FunctionalTrainingResult:
    best_epoch: int
    best_validation_effect_error: float
    training_time_seconds: float
    parameters: int


def functional_query_loss(
    output: Any, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output.future_by_action
    target = batch["target"]
    predicted_effect = prediction - output.natural_future[:, None]
    target_effect = target - batch["natural"][:, None]
    action_mask = batch["supervision"].to(prediction.dtype)
    global_per_action = (prediction - target).square().flatten(start_dim=2).mean(dim=-1)
    effect_per_action = (predicted_effect - target_effect).square().flatten(start_dim=2).mean(dim=-1)
    global_loss = (global_per_action * action_mask).sum() / action_mask.sum().clamp_min(1)
    effect_mask = action_mask * (batch["actions"] != 0).to(prediction.dtype)
    effect_loss = (effect_per_action * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    aggregate_mask = batch["response_state_mask"].any(dim=1)
    scale = torch.tensor(RESPONSE_STATE_SCALE, device=prediction.device)
    state_error = (output.theta_response - batch["theta_target"]) / scale
    state_weights = aggregate_mask.to(prediction.dtype)
    state_loss = (state_error.square() * state_weights).sum() / state_weights.sum().clamp_min(1)
    if output.theta_log_std is not None:
        normalized_log_std = output.theta_log_std - torch.log(scale)
        state_nll_values = 0.5 * (
            state_error.square() * torch.exp(-2.0 * normalized_log_std)
            + 2.0 * normalized_log_std
        )
        state_nll = (state_nll_values * state_weights).sum() / state_weights.sum().clamp_min(1)
    else:
        state_nll = state_loss * 0.0
    total = global_loss + effect_loss + state_loss + 0.05 * state_nll
    return total, {
        "global": global_loss, "effect": effect_loss,
        "state": state_loss, "state_nll": state_nll,
    }


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _forward(model: Any, batch: dict[str, torch.Tensor]) -> Any:
    return model(
        batch["history"], batch["robot"], batch["actions"],
        batch["confidence"], batch["visibility"], batch["statistics"],
        batch["support_mask"], batch["response_state_mask"],
    )


@torch.inference_mode()
def validation_effect_error(model: Any, loader: Any, device: torch.device) -> float:
    model.eval(); total = count = 0.0
    for raw in loader:
        batch = _move(raw, device); output = _forward(model, batch)
        predicted_effect = output.future_by_action - output.natural_future[:, None]
        target_effect = batch["target"] - batch["natural"][:, None]
        error = torch.linalg.vector_norm(predicted_effect - target_effect, dim=-1).mean(dim=(-1, -2))
        mask = (batch["supervision"] & (batch["actions"] != 0)).to(error.dtype)
        total += float((error * mask).sum().item()); count += float(mask.sum().item())
    return total / max(count, 1.0)


def train_functional_model(
    model: Any, train_loader: Any, validation_loader: Any,
    device: torch.device, epochs: int, checkpoint_path: str | Path,
    learning_rate: float = 1e-3,
) -> FunctionalTrainingResult:
    path = Path(checkpoint_path); path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        # The Phase 4B natural-motion backbone is a frozen prior. Keep its
        # dropout/inference behaviour frozen as well while the functional
        # response estimator and decoder are trained.
        model.decoder.natural_backbone.eval()
        for raw in train_loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss, _ = functional_query_loss(output, batch)
            loss.backward(); optimizer.step()
        validation = validation_effect_error(model, validation_loader, device)
        if validation < best:
            best, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state, "best_epoch": best_epoch,
                "best_validation_Action_Effect_Error": best,
            }, path)
    if best_state is None:
        raise RuntimeError("functional training produced no checkpoint")
    model.load_state_dict(best_state)
    return FunctionalTrainingResult(
        best_epoch, best, time.perf_counter() - started,
        sum(parameter.numel() for parameter in model.parameters()),
    )
