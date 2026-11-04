"""Sentinel-2 surface-reflectance indices (inputs: reflectance 0-1, NaN = no data).

Each index is a documented, literature-style proxy -- NOT a manganese detector:
  ndvi            (B8-B4)/(B8+B4)   vegetation cover; also tells us how much bare surface is visible
  clay_ratio      B11/B12           SWIR ratio, commonly used for clay/hydroxyl-bearing surfaces
  iron_ratio      B4/B2             red/blue ratio, commonly used as a ferric-iron-oxide proxy
  vis_brightness  (B2+B3+B4)/3      overall visible brightness (Mn/Fe oxides tend to be dark)
  swir1           B11               raw SWIR1 reflectance
dense_veg_flag marks pixels with NDVI above DENSE_VEG_NDVI (an ASSUMED threshold, to be
tested in sensitivity analysis). Its cell average = fraction of the cell under dense
vegetation = where surface spectral evidence is least trustworthy.
"""
from __future__ import annotations

import numpy as np

DENSE_VEG_NDVI = 0.6
_EPS = 1e-6


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > _EPS)
    return out


def ndvi(b8: np.ndarray, b4: np.ndarray) -> np.ndarray:
    return safe_ratio(b8 - b4, b8 + b4)


def clay_ratio(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    return safe_ratio(b11, b12)


def iron_ratio(b4: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return safe_ratio(b4, b2)


def vis_brightness(b2: np.ndarray, b3: np.ndarray, b4: np.ndarray) -> np.ndarray:
    return ((b2 + b3 + b4) / 3.0).astype(np.float32)


def dense_veg_flag(ndvi_arr: np.ndarray, threshold: float = DENSE_VEG_NDVI) -> np.ndarray:
    """1.0 where NDVI > threshold, 0.0 elsewhere, NaN where NDVI is NaN."""
    flag = (ndvi_arr > threshold).astype(np.float32)
    flag[np.isnan(ndvi_arr)] = np.nan
    return flag
