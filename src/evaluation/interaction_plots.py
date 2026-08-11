"""Clearly labelled synthetic-interaction figures for Phase 4A."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from src.data.robot_action_schema import PHASE4A_ACTIONS
from src.data.skeleton_schema import compute_root
from src.data.synthetic_interaction import VIRTUAL_PERSON_PROFILES, simulate_interaction_future


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_counterfactual_roots(
    split,
    sample_index: int,
    model_prediction: np.ndarray,
    model_name: str,
    output_path: str | Path,
) -> None:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(8, 6))
    history_root = compute_root(split.human_history[sample_index])
    axis.plot(history_root[:, 0], history_root[:, 1], "k.-", label="human history")
    for branch, action in enumerate(PHASE4A_ACTIONS):
        target_root = compute_root(split.future_by_action[sample_index, branch])
        predicted_root = compute_root(model_prediction[sample_index, branch])
        line = axis.plot(target_root[:, 0], target_root[:, 1], label=f"{action.name} GT")[0]
        axis.plot(
            predicted_root[:, 0], predicted_root[:, 1], "--",
            color=line.get_color(), label=f"{action.name} {model_name}",
        )
    axis.set_title("SYNTHETIC INTERACTION — same state, five candidate actions")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_action_effect_vectors(split, sample_index: int, output_path: str | Path) -> None:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7, 6))
    natural_root = compute_root(split.natural_future[sample_index])
    axis.plot(natural_root[:, 0], natural_root[:, 1], "k--", label="natural future")
    for branch, action in enumerate(PHASE4A_ACTIONS):
        root = compute_root(split.future_by_action[sample_index, branch])
        delta = root[-1, :2] - natural_root[-1, :2]
        axis.arrow(
            natural_root[-1, 0], natural_root[-1, 1], delta[0], delta[1],
            width=0.002, length_includes_head=True, label=action.name,
        )
        axis.text(root[-1, 0], root[-1, 1], action.name, fontsize=8)
    axis.set_title("SYNTHETIC INTERACTION — final root action-effect vectors")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_human_robot_distance(
    split,
    sample_index: int,
    model_prediction: np.ndarray,
    sample_rate_hz: float,
    output_path: str | Path,
) -> None:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(8, 5))
    time_axis = (np.arange(split.future_by_action.shape[2]) + 1) / sample_rate_hz
    predicted_root = compute_root(model_prediction[sample_index])
    predicted_distance = np.linalg.norm(
        predicted_root[..., :2] - split.robot_future_xy_by_action[sample_index], axis=-1
    )
    for branch, action in enumerate(PHASE4A_ACTIONS):
        line = axis.plot(
            time_axis, split.future_human_robot_distance[sample_index, branch],
            label=f"{action.name} GT",
        )[0]
        axis.plot(
            time_axis, predicted_distance[branch], "--", color=line.get_color(),
            label=f"{action.name} predicted",
        )
    axis.set_title("SYNTHETIC INTERACTION — future human–robot distance")
    axis.set_xlabel("future time (s)")
    axis.set_ylabel("distance (m)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_profile_responses(
    split,
    sample_index: int,
    action_index: int,
    sample_rate_hz: float,
    output_path: str | Path,
) -> None:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(8, 6))
    natural_root = compute_root(split.natural_future[sample_index])
    axis.plot(natural_root[:, 0], natural_root[:, 1], "k--", label="natural future")
    action = PHASE4A_ACTIONS[action_index]
    for profile in VIRTUAL_PERSON_PROFILES:
        simulation = simulate_interaction_future(
            split.human_history[sample_index], split.natural_future[sample_index],
            split.robot_history[sample_index], action, profile, sample_rate_hz,
        )
        root = simulation.future_root
        axis.plot(root[:, 0], root[:, 1], label=profile.name)
    axis.set_title(
        f"SYNTHETIC INTERACTION — virtual-person responses to {action.name}"
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
