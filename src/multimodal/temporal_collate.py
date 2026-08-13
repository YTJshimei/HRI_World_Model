"""Structure-preserving NumPy/PyTorch collate for Phase 5B samples."""
from __future__ import annotations

import numpy as np

from src.multimodal.temporal_schema import RichTemporalSample, STREAM_ORDER, runtime_payload


def collate_temporal(samples: list[RichTemporalSample], as_torch: bool = False) -> dict[str, object]:
    if not samples: raise ValueError("cannot collate an empty batch")
    streams = {name: np.stack([np.asarray(runtime_payload(sample)["streams"][name]) for sample in samples]) for name in STREAM_ORDER}
    mask_names = tuple(samples[0].masks)
    if any(tuple(sample.masks) != mask_names for sample in samples): raise ValueError("mask order/schema differs inside batch")
    masks = {name: np.stack([np.asarray(sample.masks[name]) for sample in samples]) for name in mask_names}
    result = {"streams": streams, "masks": masks,
              "timestamps": {name: np.stack([sample.timestamps[name] for sample in samples]) for name in samples[0].timestamps},
              "targets": {"benefit": np.asarray([sample.targets.benefit for sample in samples], np.float32),
                          "harm": np.asarray([sample.targets.harm for sample in samples], bool)}}
    if as_torch:
        import torch
        result["streams"] = {name: torch.from_numpy(value) for name, value in streams.items()}
        result["masks"] = {name: torch.from_numpy(value) for name, value in masks.items()}
        result["timestamps"] = {name: torch.from_numpy(value) for name, value in result["timestamps"].items()}
        result["targets"] = {name: torch.from_numpy(value) for name, value in result["targets"].items()}
    return result
