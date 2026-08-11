"""Strict synchronized CUDA inference benchmark shared by Phase 3 experiments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def benchmark_skeleton_model(
    model: Any,
    sample: tuple[Any, Any, Any],
    device: Any,
    torch: Any,
    warmup: int = 50,
    repetitions: int = 200,
) -> dict[str, float | int | None]:
    if warmup < 50 or repetitions < 200:
        raise ValueError("CUDA benchmark requires warmup >= 50 and repetitions >= 200")
    history, confidence, visibility = (value.to(device) for value in sample)
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(history, confidence, visibility)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(history, confidence, visibility)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(timings)
    peak = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "batch_size": int(history.shape[0]),
        "warmup": warmup,
        "repetitions": repetitions,
        "mean_ms_per_batch": float(values.mean()),
        "median_ms_per_batch": float(np.median(values)),
        "p95_ms_per_batch": float(np.percentile(values, 95)),
        "mean_ms_per_sample": float(values.mean() / history.shape[0]),
        "peak_cuda_memory_mib": peak,
    }
