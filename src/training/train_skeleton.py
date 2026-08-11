"""Validation-MPJPE checkpoint protocol for Phase 3 skeleton models."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.skeleton_schema import compute_root, skeleton_edges


def torch_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum(dim=-1).sqrt().mean()


@dataclass(frozen=True)
class SkeletonTrainingResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_mpjpe: float
    checkpoint_path: str
    training_time_seconds: float


@dataclass(frozen=True)
class SkeletonLossWeights:
    """Fixed before test access; zero-valued terms are disabled."""

    global_loss: float = 1.0
    root_loss: float = 0.0
    local_loss: float = 0.0
    bone_loss: float = 0.0
    velocity_loss: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.global_loss,
            self.root_loss,
            self.local_loss,
            self.bone_loss,
            self.velocity_loss,
        )
        if any(value < 0 for value in values) or not any(value > 0 for value in values):
            raise ValueError("loss weights must be non-negative with at least one active term")


def skeleton_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate_hz: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Differentiable global/root/local/bone/velocity training objectives."""
    predicted_root = compute_root(prediction)
    target_root = compute_root(target)
    predicted_local = prediction - predicted_root[..., None, :]
    target_local = target - target_root[..., None, :]
    predicted_bones = torch.stack(
        [
            (prediction[..., first, :] - prediction[..., second, :]).norm(dim=-1)
            for first, second in skeleton_edges
        ],
        dim=-1,
    )
    target_bones = torch.stack(
        [
            (target[..., first, :] - target[..., second, :]).norm(dim=-1)
            for first, second in skeleton_edges
        ],
        dim=-1,
    )
    if prediction.shape[1] >= 2:
        predicted_velocity = torch.diff(prediction, dim=1) * sample_rate_hz
        target_velocity = torch.diff(target, dim=1) * sample_rate_hz
        velocity = (predicted_velocity - target_velocity).square().mean()
    else:
        velocity = prediction.new_zeros(())
    return {
        "global": (prediction - target).square().mean(),
        "root": (predicted_root - target_root).square().mean(),
        "local": (predicted_local - target_local).square().mean(),
        "bone": (predicted_bones - target_bones).square().mean(),
        "velocity": velocity,
    }


def weighted_skeleton_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: SkeletonLossWeights,
    sample_rate_hz: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = skeleton_loss_components(prediction, target, sample_rate_hz)
    total = (
        weights.global_loss * components["global"]
        + weights.root_loss * components["root"]
        + weights.local_loss * components["local"]
        + weights.bone_loss * components["bone"]
        + weights.velocity_loss * components["velocity"]
    )
    return total, components


@torch.inference_mode()
def validation_mpjpe(model: nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for history, future, confidence, visibility, _ in loader:
        prediction = model(
            history.to(device), confidence.to(device), visibility.to(device)
        )
        batch = history.shape[0]
        total += torch_mpjpe(prediction, future.to(device)).item() * batch
        count += batch
    if count == 0:
        raise ValueError("validation loader 为空")
    return total / count


def train_skeleton_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    learning_rate: float = 1e-3,
    loss_weights: SkeletonLossWeights | None = None,
    sample_rate_hz: float = 10.0,
    verbose: bool = True,
) -> SkeletonTrainingResult:
    """Train without a test loader and restore the lowest-validation-MPJPE state."""
    if epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = loss_weights or SkeletonLossWeights()
    best_mpjpe, best_epoch, best_state = float("inf"), 0, None
    rows = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_global, count = 0.0, 0.0, 0
        for history, future, confidence, visibility, _ in train_loader:
            history = history.to(device)
            future = future.to(device)
            confidence = confidence.to(device)
            visibility = visibility.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, confidence, visibility)
            loss, components = weighted_skeleton_loss(
                prediction, future, weights, sample_rate_hz
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * history.shape[0]
            total_global += components["global"].item() * history.shape[0]
            count += history.shape[0]
        if count == 0:
            raise ValueError("train loader 为空")
        current_mpjpe = validation_mpjpe(model, validation_loader, device)
        row = {
            "epoch": float(epoch),
            "train_mse": total_global / count,
            "train_weighted_loss": total_loss / count,
            "validation_MPJPE": current_mpjpe,
        }
        rows.append(row)
        if current_mpjpe < best_mpjpe:
            best_mpjpe = current_mpjpe
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_MPJPE": best_mpjpe,
                },
                path,
            )
        if verbose:
            print(
                f"epoch={epoch:03d} train_loss={row['train_weighted_loss']:.6f} "
                f"val_MPJPE={current_mpjpe:.6f}"
            )
    training_time = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("未产生 skeleton validation checkpoint")
    model.load_state_dict(best_state)
    return SkeletonTrainingResult(
        tuple(rows), best_epoch, best_mpjpe, str(path), training_time
    )
