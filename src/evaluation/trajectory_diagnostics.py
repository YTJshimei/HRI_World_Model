"""Grouped evaluation and reproducible inference benchmarking."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from src.evaluation.trajectory_metrics import ade_fde, parameter_count


@torch.inference_mode()
def predict_batches(model: nn.Module, loader: Any, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat([model(history.to(device)).cpu() for history, _ in loader])


def metrics_by_type(
    prediction: Any, target: Any, labels: np.ndarray
) -> dict[str, dict[str, float | int]]:
    rows = {}
    for kind in np.unique(labels):
        mask = labels == kind
        ade, fde = ade_fde(prediction[mask], target[mask])
        rows[str(kind)] = {"count": int(mask.sum()), "ADE": ade, "FDE": fde}
    return rows


class ConstantVelocityModule(nn.Module):
    def __init__(self, future_length: int) -> None:
        super().__init__()
        self.future_length = future_length

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        velocity = history[:, -1] - history[:, -2]
        steps = torch.arange(
            1,
            self.future_length + 1,
            device=history.device,
            dtype=history.dtype,
        ).view(1, -1, 1)
        return history[:, -1:, :] + steps * velocity[:, None, :]


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int = 50,
    repetitions: int = 200,
) -> dict[str, float | int | None]:
    if warmup < 50 or repetitions < 200:
        raise ValueError("GPU benchmark 要求 warmup >= 50 且 repetitions >= 200")
    model = model.to(device).eval()
    sample = sample.to(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000)

    peak_memory = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    timings = np.asarray(timings_ms)
    return {
        "batch_size": int(sample.shape[0]),
        "warmup": warmup,
        "repetitions": repetitions,
        "mean_ms_per_batch": float(timings.mean()),
        "median_ms_per_batch": float(np.median(timings)),
        "p95_ms_per_batch": float(np.percentile(timings, 95)),
        "mean_ms_per_sample": float(timings.mean() / sample.shape[0]),
        "parameters": parameter_count(model),
        "peak_cuda_memory_mib": peak_memory,
    }


def evaluate_predictor(
    predictor: Callable[[np.ndarray], np.ndarray], history: np.ndarray, future: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    prediction = predictor(history)
    ade, fde = ade_fde(prediction, future)
    return prediction, {"ADE": ade, "FDE": fde}
