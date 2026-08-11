"""Publication-style aggregate figures for Phase 3B diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_grouped_lines(
    aggregate_rows: Iterable[dict],
    x_key: str,
    metric: str,
    output_path: str | Path,
    title: str,
    x_label: str,
) -> None:
    plt = _plt()
    rows = list(aggregate_rows)
    figure, axis = plt.subplots(figsize=(8, 5))
    for model in dict.fromkeys(row["model"] for row in rows):
        selected = sorted((row for row in rows if row["model"] == model), key=lambda row: float(row[x_key]))
        x = np.asarray([float(row[x_key]) for row in selected])
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
        std = np.asarray([float(row[f"{metric}_std"]) for row in selected])
        axis.plot(x, mean, marker="o", label=model)
        axis.fill_between(x, mean - std, mean + std, alpha=0.15)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(metric.replace("_", " "))
    axis.grid(alpha=0.3)
    axis.legend()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_oracle_decomposition(rows: Iterable[dict], output_path: str | Path) -> None:
    plt = _plt()
    selected = list(rows)
    labels = [row["model"] for row in selected]
    values = [row["Global_MPJPE_mean"] for row in selected]
    errors = [row["Global_MPJPE_std"] for row in selected]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, values, yerr=errors, capsize=3)
    axis.set_ylabel("Global MPJPE (m)")
    axis.set_title("Oracle root/local recombination")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_contributions(rows: Iterable[dict], output_path: str | Path) -> None:
    plt = _plt()
    selected = list(rows)
    labels = [row["model"] for row in selected]
    root = np.asarray([row["root_contribution_mean"] for row in selected])
    local = np.asarray([row["local_contribution_mean"] for row in selected])
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(labels, root, label="root contribution")
    axis.bar(labels, local, bottom=root, label="local contribution")
    axis.set_ylabel("Shapley-attributed Global MPJPE (m)")
    axis.set_title("Root/local contribution to global error")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
