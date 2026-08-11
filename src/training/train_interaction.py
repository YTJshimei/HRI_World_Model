"""Validation-only checkpoint protocol and Phase 4A multi-objective losses."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.skeleton_schema import compute_root


@dataclass(frozen=True)
class InteractionLossWeights:
    future_global: float = 1.0
    root: float = 1.0
    local: float = 1.0
    action_effect: float = 1.0
    natural: float = 1.0


@dataclass(frozen=True)
class InteractionTrainingResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_global_mpjpe: float
    checkpoint_path: str
    training_time_seconds: float


def _masked_mean(values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    per_action = values.flatten(start_dim=2).mean(dim=-1)
    weights = action_mask.to(values.dtype)
    return (per_action * weights).sum() / weights.sum().clamp_min(1.0)


def interaction_loss(
    output: Any,
    target_future: torch.Tensor,
    natural_target: torch.Tensor,
    supervision_mask: torch.Tensor,
    weights: InteractionLossWeights = InteractionLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output.future_by_action
    predicted_root = compute_root(prediction)
    target_root = compute_root(target_future)
    predicted_local = prediction - predicted_root[..., None, :]
    target_local = target_future - target_root[..., None, :]
    predicted_effect = prediction - output.natural_future[:, None]
    target_effect = target_future - natural_target[:, None]
    components = {
        "future_global": _masked_mean((prediction - target_future).square(), supervision_mask),
        "root": _masked_mean((predicted_root - target_root).square(), supervision_mask),
        "local": _masked_mean((predicted_local - target_local).square(), supervision_mask),
        "action_effect": _masked_mean(
            (predicted_effect - target_effect).square(), supervision_mask
        ),
        "natural": (output.natural_future - natural_target).square().mean(),
    }
    total = sum(getattr(weights, name) * value for name, value in components.items())
    return total, components


def _move_batch(batch: tuple[Any, ...], device: torch.device) -> tuple[Any, ...]:
    return tuple(value.to(device) for value in batch)


def _model_inputs(batch: tuple[Any, ...]) -> tuple[Any, ...]:
    history, _, _, robot, actions, confidence, visibility, profiles, *_ = batch
    return history, robot, actions, confidence, visibility, profiles


@torch.inference_mode()
def validation_global_mpjpe(model: nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0.0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        target, supervision = batch[2], batch[8]
        prediction = model(*_model_inputs(batch)).future_by_action
        errors = (prediction - target).square().sum(dim=-1).sqrt().mean(dim=(-1, -2))
        weights = supervision.to(errors.dtype)
        total += float((errors * weights).sum().item())
        count += float(weights.sum().item())
    if count == 0:
        raise ValueError("validation loader has no supervised action branches")
    return total / count


def train_interaction_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    learning_rate: float = 1e-3,
    weights: InteractionLossWeights = InteractionLossWeights(),
    verbose: bool = True,
) -> InteractionTrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_metric, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            natural, target, supervision = batch[1], batch[2], batch[8]
            optimizer.zero_grad(set_to_none=True)
            output = model(*_model_inputs(batch))
            loss, _ = interaction_loss(output, target, natural, supervision, weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch[0].shape[0]
            count += int(batch[0].shape[0])
        validation = validation_global_mpjpe(model, validation_loader, device)
        row = {
            "epoch": float(epoch),
            "train_weighted_loss": total_loss / count,
            "validation_Global_MPJPE": validation,
        }
        history.append(row)
        if validation < best_metric:
            best_metric, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_Global_MPJPE": best_metric,
                },
                path,
            )
        if verbose:
            print(
                f"epoch={epoch:03d} train_loss={row['train_weighted_loss']:.6f} "
                f"val_Global_MPJPE={validation:.6f}"
            )
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    return InteractionTrainingResult(
        tuple(history), best_epoch, best_metric, str(path),
        time.perf_counter() - started,
    )
