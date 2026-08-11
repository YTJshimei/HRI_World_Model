"""Root-only and root-aligned pose figures for Phase 3A.5 diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from src.data.skeleton_schema import compute_root, skeleton_edges


def plot_root_trajectories(
    history: np.ndarray,
    future: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    action_types: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot one GT/predicted root trajectory per action."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions = list(dict.fromkeys(action_types.tolist()))
    figure, axes = plt.subplots(3, 3, figsize=(15, 14), squeeze=False)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for axis, action in zip(axes.flat, actions):
        index = int(np.flatnonzero(action_types == action)[0])
        history_root = compute_root(history[index])
        target_root = compute_root(future[index])
        axis.plot(history_root[:, 0], history_root[:, 1], "k.-", label="history")
        axis.plot(target_root[:, 0], target_root[:, 1], "k--", label="GT future")
        for color, (name, prediction) in zip(colors, predictions.items()):
            root = compute_root(prediction[index])
            axis.plot(root[:, 0], root[:, 1], color=color, label=name)
        axis.set_title(action)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Root-only trajectories: GT versus prediction")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_skeleton(axis, skeleton: np.ndarray, color: str, label: str) -> None:
    for edge_index, (first, second) in enumerate(skeleton_edges):
        points = skeleton[[first, second]]
        axis.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=color, linewidth=1.8, label=label if edge_index == 0 else None,
        )
    axis.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], color=color, s=10)


def plot_local_pose_comparison(
    future: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    action_types: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Plot final-frame skeletons after independent pelvis alignment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for action in dict.fromkeys(action_types.tolist()):
        index = int(np.flatnonzero(action_types == action)[0])
        target = future[index, -1]
        target = target - compute_root(target)[None]
        figure = plt.figure(figsize=(5 * len(predictions), 5))
        for panel, (name, prediction) in enumerate(predictions.items(), start=1):
            axis = figure.add_subplot(1, len(predictions), panel, projection="3d")
            predicted = prediction[index, -1]
            predicted = predicted - compute_root(predicted)[None]
            _draw_skeleton(axis, target, "black", "GT local")
            _draw_skeleton(axis, predicted, "tab:red", f"{name} local")
            combined = np.concatenate((target, predicted))
            center = combined.mean(axis=0)
            half_range = max(float(np.ptp(combined, axis=0).max()) / 2, 0.45)
            axis.set_xlim(center[0] - half_range, center[0] + half_range)
            axis.set_ylim(center[1] - half_range, center[1] + half_range)
            axis.set_zlim(center[2] - half_range, center[2] + half_range)
            axis.set_title(name)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            axis.legend(fontsize=8)
        figure.suptitle(f"Root-aligned final pose: {action}")
        figure.tight_layout()
        figure.savefig(directory / f"local_pose_{action}.png", dpi=160)
        plt.close(figure)
