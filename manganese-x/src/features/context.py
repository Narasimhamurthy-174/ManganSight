"""Neighbourhood ("context") versions of per-cell features.

Why: MRDS coordinates have 'highly variable' accuracy (see mrds.met), so a record may really sit
a kilometre or more from its recorded point. Averaging features over a ~1-2.5 km neighbourhood
tests whether blurry labels, not missing signal, are what hurts the 250 m models.
focal_mean ignores NaN cells (outside the district) and returns NaN where too little data exists.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def focal_mean(arr: np.ndarray, radius_cells: int, min_valid_frac: float = 0.5) -> np.ndarray:
    """Mean over a square window of (2*radius+1)^2 cells, using only valid (finite) cells."""
    if radius_cells < 1:
        raise ValueError("radius_cells must be >= 1")
    size = 2 * radius_cells + 1
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float64)
    num = ndimage.uniform_filter(filled, size=size, mode="constant", cval=0.0)
    den = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="constant", cval=0.0)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    ok = valid & (den >= min_valid_frac)
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def tpi(arr: np.ndarray, radius_cells: int) -> np.ndarray:
    """Topographic position index: elevation minus the mean elevation around it
    (positive = higher than surroundings, e.g. ridge; negative = valley or low ground)."""
    return (arr - focal_mean(arr, radius_cells)).astype(np.float32)
