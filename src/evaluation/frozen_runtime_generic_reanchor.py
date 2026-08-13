"""Parameter-free frozen runtime-generic Benefit re-anchoring."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FRGR_ALPHA = 1.0


def frozen_runtime_generic_reanchor(
    prediction: np.ndarray,
    episode_ids: Sequence[str],
    generic_index_by_episode: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``mu_i - mu_g`` and the per-candidate frozen generic offset.

    This function has no learned state and exposes no scale, temperature or
    clipping option.  Every episode must provide exactly one already-frozen
    runtime-generic candidate index.
    """
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(episode_ids):
        raise ValueError("prediction and episode_ids must be aligned 1-D values")
    if not np.isfinite(values).all():
        raise ValueError("prediction contains NaN or Inf")
    offsets = np.empty_like(values)
    for index, episode_id in enumerate(episode_ids):
        if episode_id not in generic_index_by_episode:
            raise ValueError(f"missing frozen runtime-generic index for {episode_id}")
        generic_index = int(generic_index_by_episode[episode_id])
        if generic_index < 0 or generic_index >= len(values):
            raise ValueError(f"invalid frozen runtime-generic index for {episode_id}")
        if episode_ids[generic_index] != episode_id:
            raise ValueError(f"runtime-generic index crosses episode boundary for {episode_id}")
        offsets[index] = values[generic_index]
    return values - FRGR_ALPHA * offsets, offsets
