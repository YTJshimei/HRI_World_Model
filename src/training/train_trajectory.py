"""Shared training and evaluation utilities for trajectory baselines."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from src.evaluation.trajectory_metrics import ade_fde


def train_model(model: nn.Module, train_loader: Any, val_loader: Any, device: torch.device, epochs: int, learning_rate: float = 1e-3) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for observed, future in train_loader:
            observed, future = observed.to(device), future.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(observed), future)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * observed.shape[0]
            total_items += observed.shape[0]
        val_ade, val_fde, _ = evaluate_model(model, val_loader, device)
        row = {"epoch": float(epoch), "train_mse": total_loss / total_items, "val_ade": val_ade, "val_fde": val_fde}
        history.append(row)
        print(f"epoch={epoch:03d} train_mse={row['train_mse']:.6f} val_ADE={val_ade:.6f} val_FDE={val_fde:.6f}")
    return history


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: Any, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    predictions, targets = [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for observed, future in loader:
        predictions.append(model(observed.to(device)).cpu())
        targets.append(future)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    ade, fde = ade_fde(prediction, target)
    return ade, fde, elapsed * 1000 / len(target)
