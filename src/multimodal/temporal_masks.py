"""Mask and padding utilities for rich temporal streams."""
from __future__ import annotations

import numpy as np


def left_pad(values: np.ndarray, valid: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values); valid = np.asarray(valid, dtype=bool)
    if values.shape[0] != valid.shape[0]: raise ValueError("value/mask time dimensions differ")
    if length <= 0: raise ValueError("length must be positive")
    kept = min(length, values.shape[0]); result = np.zeros((length, *values.shape[1:]), dtype=values.dtype)
    result_valid = np.zeros((length, *valid.shape[1:]), dtype=bool); padding = np.zeros(length, dtype=bool)
    result[-kept:] = values[-kept:]; result_valid[-kept:] = valid[-kept:]; padding[-kept:] = True
    return result, result_valid, padding


def masked_values_equal(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> bool:
    mask = np.asarray(valid, bool)
    while mask.ndim < np.asarray(a).ndim: mask = mask[..., None]
    return bool(np.array_equal(np.asarray(a)[np.broadcast_to(mask, np.asarray(a).shape)], np.asarray(b)[np.broadcast_to(mask, np.asarray(b).shape)]))
