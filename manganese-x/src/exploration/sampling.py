"""Background (pseudo-absence) sampling.

'No known occurrence' does NOT mean 'no manganese', so these cells are called BACKGROUND, not
negatives. Two strategies are compared because results may depend on this choice:
  uniform    - anywhere in the training area (contrast: belt vs the whole district)
  near_belt  - only between `buffer_m` and `ring_max_m` of a training positive (contrast: inside
               the mining belt, a harder and less geography-driven question)
Background is drawn ONLY from training cells, never from the test fold.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

STRATEGIES = ("uniform", "near_belt")


def draw_background(xy: np.ndarray, is_train: np.ndarray, is_pos: np.ndarray, strategy: str,
                    n_bg: int, rng: np.random.Generator, buffer_m: float = 1000.0,
                    ring_max_m: float = 15000.0) -> np.ndarray:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    train_pos = np.flatnonzero(is_train & is_pos)
    if len(train_pos) == 0:
        raise ValueError("no training positives")
    cand = np.flatnonzero(is_train & ~is_pos)
    d, _ = cKDTree(xy[train_pos]).query(xy[cand])
    keep = d > buffer_m
    if strategy == "near_belt":
        keep &= d <= ring_max_m
    cand = cand[keep]
    if len(cand) == 0:
        raise ValueError("no candidate background cells satisfy the strategy constraints")
    return rng.choice(cand, size=min(n_bg, len(cand)), replace=False)
