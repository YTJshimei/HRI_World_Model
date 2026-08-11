"""Leakage-safe datasets and validation-selected training for Phase 4B."""

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
    SupportStrategy,
    support_to_padded_features,
)
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import PROFILE_BY_ID


ORACLE_PARAMETER_NAMES = (
    "preferred_distance",
    "distance_sensitivity",
    "speed_response_gain",
    "response_delay",
    "lateral_avoidance_gain",
    "turn_sensitivity",
    "adaptation_rate",
)


def oracle_parameter_vector(profile_id: int) -> np.ndarray:
    profile = PROFILE_BY_ID[int(profile_id)]
    return np.asarray(
        [getattr(profile, name) for name in ORACLE_PARAMETER_NAMES], dtype=np.float32
    )


class PersonalInteractionQueryDataset(Dataset):
    def __init__(
        self,
        corpus: PersonalInteractionCorpus,
        k_values: int | Sequence[int],
        strategy: SupportStrategy = "earliest",
        person_index: dict[str, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.corpus = corpus
        self.memory = PersonalInteractionMemory(corpus.records)
        values = (k_values,) if isinstance(k_values, int) else tuple(k_values)
        if not values or any(value not in SUPPORTED_K for value in values):
            raise ValueError(f"k_values must use {SUPPORTED_K}")
        self.k_values = values
        self.strategy = strategy
        self.person_index = person_index or {}
        self.seed = seed

    def __len__(self) -> int:
        return len(self.corpus.query_indices) * len(self.k_values)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        query_position = item // len(self.k_values)
        k = self.k_values[item % len(self.k_values)]
        source = int(self.corpus.query_indices[query_position])
        record = self.corpus.records[source]
        support = self.memory.select_support(
            record, k, self.strategy, seed=self.seed + source * 97 + k
        )
        support_features, support_mask = support_to_padded_features(support)
        split = self.corpus.split
        person_value = self.person_index.get(record.person_instance_id, -1)
        return {
            "history": torch.from_numpy(split.human_history[source]),
            "natural": torch.from_numpy(split.natural_future[source]),
            "target": torch.from_numpy(split.future_by_action[source]),
            "robot": torch.from_numpy(split.robot_history[source]),
            "actions": torch.from_numpy(split.candidate_actions[source]),
            "confidence": torch.from_numpy(split.confidence[source]),
            "visibility": torch.from_numpy(split.visibility_mask[source]),
            "supervision": torch.from_numpy(split.action_supervision_mask[source]),
            "robot_future": torch.from_numpy(split.robot_future_xy_by_action[source]),
            "distance": torch.from_numpy(split.future_human_robot_distance[source]),
            "support_features": torch.from_numpy(support_features),
            "support_mask": torch.from_numpy(support_mask),
            "person_index": torch.tensor(person_value, dtype=torch.long),
            "oracle_parameters": torch.from_numpy(
                oracle_parameter_vector(record.person_profile_id)
            ),
            "profile_id": torch.tensor(record.person_profile_id, dtype=torch.long),
            "source_index": torch.tensor(source, dtype=torch.long),
            "k": torch.tensor(k, dtype=torch.long),
        }


@dataclass(frozen=True)
class PersonalTrainingResult:
    best_epoch: int
    best_validation_global_mpjpe: float
    training_time_seconds: float
    history: tuple[dict[str, float], ...]


def _masked_action_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_action = values.flatten(start_dim=2).mean(dim=-1)
    weights = mask.to(values.dtype)
    return (per_action * weights).sum() / weights.sum().clamp_min(1.0)


def personal_interaction_loss(
    output: Any,
    target: torch.Tensor,
    natural: torch.Tensor,
    supervision: torch.Tensor,
    action_ids: torch.Tensor | None = None,
    uncertainty_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output.future_by_action
    pred_root, target_root = compute_root(prediction), compute_root(target)
    pred_local = prediction - pred_root[..., None, :]
    target_local = target - target_root[..., None, :]
    pred_effect = prediction - output.natural_future[:, None]
    target_effect = target - natural[:, None]
    log_std = output.root_log_std_by_action
    root_nll = 0.5 * (
        (pred_root - target_root).square() * torch.exp(-2.0 * log_std) + 2.0 * log_std
    )
    pred_effect_root = compute_root(pred_effect)
    target_effect_root = compute_root(target_effect)
    effect_log_std = output.action_effect_root_log_std_by_action
    effect_root_nll = 0.5 * (
        (pred_effect_root - target_effect_root).square()
        * torch.exp(-2.0 * effect_log_std)
        + 2.0 * effect_log_std
    )
    nonkeep_supervision = supervision & (
        action_ids != 0 if action_ids is not None else
        torch.arange(supervision.shape[1], device=supervision.device)[None] != 0
    )
    components = {
        "future_global": _masked_action_mean((prediction - target).square(), supervision),
        "root": _masked_action_mean((pred_root - target_root).square(), supervision),
        "local": _masked_action_mean((pred_local - target_local).square(), supervision),
        "action_effect": _masked_action_mean((pred_effect - target_effect).square(), supervision),
        "natural": (output.natural_future - natural).square().mean(),
        "root_nll": _masked_action_mean(root_nll, supervision),
        "action_effect_root_nll": _masked_action_mean(
            effect_root_nll, nonkeep_supervision
        ),
    }
    total = sum(components[name] for name in ("future_global", "root", "local", "action_effect", "natural"))
    total = total + uncertainty_weight * (
        components["root_nll"] + components["action_effect_root_nll"]
    )
    return total, components


def model_forward(model: nn.Module, batch: dict[str, torch.Tensor]) -> Any:
    return model(
        batch["history"], batch["robot"], batch["actions"],
        batch["confidence"], batch["visibility"],
        support_features=batch["support_features"],
        support_mask=batch["support_mask"],
        person_indices=batch["person_index"],
        oracle_parameters=batch["oracle_parameters"],
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.inference_mode()
def validation_global_mpjpe(model: nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0.0
    for raw in loader:
        batch = move_batch(raw, device)
        output = model_forward(model, batch)
        errors = (output.future_by_action - batch["target"]).square().sum(dim=-1).sqrt().mean(dim=(-1, -2))
        weights = batch["supervision"].to(errors.dtype)
        total += float((errors * weights).sum().item())
        count += float(weights.sum().item())
    if count == 0:
        raise ValueError("validation has no supervised action branches")
    return total / count


def train_personal_response_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    learning_rate: float = 1e-3,
    verbose: bool = True,
) -> PersonalTrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best, best_epoch, best_state = float("inf"), 0, None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for raw in train_loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            output = model_forward(model, batch)
            loss, _ = personal_interaction_loss(
                output, batch["target"], batch["natural"], batch["supervision"],
                batch["actions"],
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["history"].shape[0]
            count += int(batch["history"].shape[0])
        validation = validation_global_mpjpe(model, validation_loader, device)
        row = {"epoch": float(epoch), "train_loss": total / count, "validation_Global_MPJPE": validation}
        history.append(row)
        if validation < best:
            best, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model_state_dict": best_state, "best_epoch": best_epoch,
                 "best_validation_Global_MPJPE": best}, path,
            )
        if verbose:
            print(f"epoch={epoch:03d} train={total / count:.6f} val={validation:.6f}")
    if best_state is None:
        raise RuntimeError("no validation checkpoint produced")
    model.load_state_dict(best_state)
    return PersonalTrainingResult(
        best_epoch, best, time.perf_counter() - started, tuple(history)
    )
