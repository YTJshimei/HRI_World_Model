"""Pure diagnostic utilities for Phase 5B-1.7E-C representation auditing."""
from __future__ import annotations

from itertools import combinations

import numpy as np


SUBTYPE_ORDER = ("GT_UNSAFE", "EXCESSIVE_DECELERATION", "ABRUPT_LATERAL_RESPONSE", "ABRUPT_HEADING_CHANGE")


def fixed_noisy_or(probabilities) -> np.ndarray:
    """Combine fixed subtype probabilities without learned weights."""
    value = np.asarray(probabilities, np.float64)
    if value.ndim != 2 or value.shape[1] != len(SUBTYPE_ORDER):
        raise ValueError("probabilities must have shape [N,4] in frozen subtype order")
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    return 1.0 - np.prod(1.0 - value, axis=1)


def cosine_similarity_rows(vectors: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for left, right in combinations(vectors, 2):
        a, b = np.asarray(vectors[left], np.float64), np.asarray(vectors[right], np.float64)
        similarity = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        rows.append({"left": left, "right": right, "cosine_similarity": similarity})
    return rows


def group_geometry(embedding, masks: dict[str, np.ndarray]) -> dict[str, object]:
    x = np.asarray(embedding, np.float64)
    if x.ndim != 2: raise ValueError("embedding must be [N,D]")
    centroids, groups = {}, {}
    for name, original in masks.items():
        mask = np.asarray(original, bool)
        if mask.shape != (len(x),) or not mask.any(): raise ValueError(f"invalid/nonpositive group mask: {name}")
        centroid = x[mask].mean(0); centroids[name] = centroid
        groups[name] = {"count": int(mask.sum()), "within_group_squared_euclidean_dispersion": float(np.mean(np.sum((x[mask] - centroid) ** 2, axis=1)))}
    distances = []
    for left, right in combinations(masks, 2):
        a, b = centroids[left], centroids[right]
        distances.append({"left": left, "right": right, "euclidean_distance": float(np.linalg.norm(a - b)),
                          "cosine_distance": float(1 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))})
    return {"embedding_dimension": int(x.shape[1]), "groups": groups, "centroid_distances": distances}


def pairwise_discrimination(scores, labels, episode_ids) -> dict[str, float | int]:
    scores, labels, episode_ids = np.asarray(scores, np.float64), np.asarray(labels, bool), np.asarray(episode_ids)
    correct, pairs, episodes = 0.0, 0, 0
    for episode in np.unique(episode_ids):
        index = np.flatnonzero(episode_ids == episode); positive = index[labels[index]]; negative = index[~labels[index]]
        if not len(positive) or not len(negative): continue
        episodes += 1
        for pos in positive:
            for neg in negative:
                correct += float(scores[pos] > scores[neg]) + .5 * float(scores[pos] == scores[neg]); pairs += 1
    return {"mixed_episode_count": episodes, "positive_negative_pair_count": pairs,
            "pairwise_discrimination_accuracy": float(correct / pairs) if pairs else float("nan")}


def candidate_conditioning_distances(embedding, labels, episode_ids) -> dict[str, float | int]:
    x, labels, episode_ids = np.asarray(embedding, np.float64), np.asarray(labels, bool), np.asarray(episode_ids)
    same, different, varying, total = [], [], 0, 0
    for episode in np.unique(episode_ids):
        index = np.flatnonzero(episode_ids == episode)
        if len(np.unique(labels[index])) > 1: varying += 1
        total += 1
        for i, j in combinations(index, 2):
            (different if labels[i] != labels[j] else same).append(float(np.linalg.norm(x[i] - x[j])))
    return {"episode_count": total, "episodes_with_candidate_dependent_harm_label": varying,
            "label_variation_episode_fraction": float(varying / total), "same_label_pair_count": len(same),
            "different_label_pair_count": len(different), "same_label_mean_distance": float(np.mean(same)),
            "different_label_mean_distance": float(np.mean(different)),
            "different_to_same_distance_ratio": float(np.mean(different) / max(np.mean(same), 1e-12))}
