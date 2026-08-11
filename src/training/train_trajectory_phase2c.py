"""Phase 2C training protocol with validation-only checkpoint selection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.evaluation.trajectory_metrics import ade_fde


@dataclass(frozen=True)
class BestValidationResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_ade: float
    checkpoint_path: str


@torch.inference_mode()
def validation_metrics(model: nn.Module, loader: Any, device: torch.device) -> tuple[float, float]:
    model.eval()
    predictions, targets = [], []
    for history, future in loader:
        predictions.append(model(history.to(device)).cpu())
        targets.append(future)
    if not predictions:
        raise ValueError("validation loader 为空")
    return ade_fde(torch.cat(predictions), torch.cat(targets))


def train_with_best_validation_checkpoint(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    learning_rate: float = 1e-3,
) -> BestValidationResult:
    """Train without any test-set input and restore the lowest-validation-ADE state."""
    if epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_ade = float("inf")
    best_epoch = 0
    best_state = None
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for history, future in train_loader:
            history, future = history.to(device), future.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(history), future)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * history.shape[0]
            count += history.shape[0]
        if count == 0:
            raise ValueError("train loader 为空")
        validation_ade, validation_fde = validation_metrics(model, validation_loader, device)
        row = {
            "epoch": float(epoch),
            "train_mse": total_loss / count,
            "validation_ADE": validation_ade,
            "validation_FDE": validation_fde,
        }
        rows.append(row)
        if validation_ade < best_ade:
            best_ade = validation_ade
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_ADE": best_ade,
                },
                path,
            )
        print(
            f"epoch={epoch:03d} train_mse={row['train_mse']:.6f} "
            f"val_ADE={validation_ade:.6f} val_FDE={validation_fde:.6f}"
        )
    if best_state is None:
        raise RuntimeError("未能产生 validation checkpoint")
    model.load_state_dict(best_state)
    return BestValidationResult(tuple(rows), best_epoch, best_ade, str(path))
