"""Terrain derivatives from a DEM (numpy arrays, NaN = no data).

Deliberately small feature set (project rule: no feature inflation):
  slope_degrees  - Horn (1981) 3x3 method
  tri            - mean absolute elevation difference to the 8 neighbours (ruggedness)
  relief         - max - min elevation inside a square window (local relative relief)
Aspect is NOT computed: it is circular and has no clear link to manganese host rocks.
Pixels touching NaN neighbours become NaN for slope/tri (conservative).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _neighbours(a: np.ndarray) -> dict[str, np.ndarray]:
    h, w = a.shape
    p = np.pad(a, 1, mode="constant", constant_values=np.nan)
    return {
        "nw": p[0:h, 0:w], "n": p[0:h, 1:w + 1], "ne": p[0:h, 2:w + 2],
        "w": p[1:h + 1, 0:w], "e": p[1:h + 1, 2:w + 2],
        "sw": p[2:h + 2, 0:w], "s": p[2:h + 2, 1:w + 1], "se": p[2:h + 2, 2:w + 2],
    }


def slope_degrees(dem: np.ndarray, res: float) -> np.ndarray:
    if res <= 0:
        raise ValueError("res must be positive")
    n = _neighbours(dem.astype(np.float32, copy=False))
    dzdx = ((n["ne"] + 2 * n["e"] + n["se"]) - (n["nw"] + 2 * n["w"] + n["sw"])) / (8 * res)
    dzdy = ((n["sw"] + 2 * n["s"] + n["se"]) - (n["nw"] + 2 * n["n"] + n["ne"])) / (8 * res)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def tri(dem: np.ndarray) -> np.ndarray:
    d = dem.astype(np.float32, copy=False)
    n = _neighbours(d)
    total = np.zeros_like(d)
    for v in n.values():
        total += np.abs(v - d)
    return (total / 8.0).astype(np.float32)


def relief(dem: np.ndarray, window_px: int) -> np.ndarray:
    """Max-min elevation in a window_px x window_px square (window_px must be odd)."""
    if window_px < 3 or window_px % 2 == 0:
        raise ValueError("window_px must be an odd integer >= 3")
    d = dem.astype(np.float32, copy=False)
    nan = np.isnan(d)
    mx = ndimage.maximum_filter(np.where(nan, -np.inf, d), size=window_px, mode="nearest")
    mn = ndimage.minimum_filter(np.where(nan, np.inf, d), size=window_px, mode="nearest")
    out = (mx - mn).astype(np.float32)
    out[nan | ~np.isfinite(out)] = np.nan
    return out
