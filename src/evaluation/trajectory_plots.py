"""Non-interactive plots for Phase 2B diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_examples(
    output_dir: Path,
    history: np.ndarray,
    future: np.ndarray,
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    samples_per_type: int = 5,
    filename_prefix: str = "trajectories",
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    styles = {
        "ground_truth": ("black", "-"),
        "M0": ("tab:gray", "--"),
        "LSTM": ("tab:blue", "--"),
        "Transformer": ("tab:orange", "--"),
        "M3": ("tab:green", "--"),
    }
    for kind in np.unique(labels):
        indices = np.flatnonzero(labels == kind)[:samples_per_type]
        if len(indices) < samples_per_type:
            raise ValueError(f"{kind} 测试样本少于 {samples_per_type} 个")
        figure, axes = plt.subplots(1, samples_per_type, figsize=(4 * samples_per_type, 4))
        for axis, index in zip(np.atleast_1d(axes), indices):
            axis.plot(history[index, :, 0], history[index, :, 1], color="tab:purple", label="history")
            axis.plot(future[index, :, 0], future[index, :, 1], color="black", label="ground truth")
            for name, values in predictions.items():
                color, linestyle = styles.get(name, (None, "--"))
                axis.plot(values[index, :, 0], values[index, :, 1], color=color, linestyle=linestyle, label=name)
            axis.scatter(history[index, -1, 0], history[index, -1, 1], s=20, color="tab:purple")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.25)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels))
        figure.suptitle(f"{kind}: {samples_per_type} test trajectories")
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        path = output_dir / f"{filename_prefix}_{kind}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def plot_training_curves(output_dir: Path, curves: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    fields = (("train_mse", "Train loss"), ("val_ade", "Validation ADE"), ("val_fde", "Validation FDE"))
    for axis, (field, title) in zip(axes, fields):
        for name, rows in curves.items():
            axis.plot([row["epoch"] for row in rows], [row[field] for row in rows], label=name)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    path = output_dir / "training_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)
