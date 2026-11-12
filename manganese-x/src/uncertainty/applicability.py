"""Area of applicability (AOA), adapted from Meyer & Pebesma (2021).

Original idea: a prediction is only trustworthy where a cell's predictors resemble the training data.
Dissimilarity index DI = distance (in standardised predictor space) to the nearest reference point,
divided by the mean pairwise distance among reference points. Threshold = upper whisker (Q3 + 1.5 IQR)
of the reference points' own DI, each measured to the nearest reference point in a DIFFERENT spatial
block (their cross-validation excludes same-fold neighbours; we exclude same-block neighbours).
Our adaptation: the reference points are the KNOWN-RECORD cells, because those define the conditions
under which manganese has been reported. Equal variable weights (no importance weighting).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


def standardize(X: np.ndarray, ref: np.ndarray | None = None):
    ref = X if ref is None else ref
    mean = np.nanmean(ref, axis=0)
    std = np.nanstd(ref, axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def dissimilarity_index(Z_cells: np.ndarray, Z_ref: np.ndarray) -> np.ndarray:
    if len(Z_ref) < 2:
        raise ValueError("need at least 2 reference points")
    d, _ = cKDTree(Z_ref).query(Z_cells)
    return d / pdist(Z_ref).mean()


def aoa_threshold(Z_ref: np.ndarray, blocks_ref: np.ndarray, whisker: float = 1.5):
    """Return (threshold, DI of each reference point to its nearest reference point in another block)."""
    norm = pdist(Z_ref).mean()
    di = np.empty(len(Z_ref))
    for i in range(len(Z_ref)):
        other = blocks_ref != blocks_ref[i]
        if not other.any():                              # all in one block: fall back to leave-one-out
            other = np.arange(len(Z_ref)) != i
        di[i] = np.linalg.norm(Z_ref[other] - Z_ref[i], axis=1).min() / norm
    q1, q3 = np.percentile(di, [25, 75])
    return float(q3 + whisker * (q3 - q1)), di
