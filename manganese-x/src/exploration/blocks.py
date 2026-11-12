"""Spatial blocks and fold assignment.

Blocks are square tiles of `block_m` metres. WHOLE blocks go to the same fold, so test cells are
never immediate neighbours of training cells (when block_m >= the autocorrelation range).
block_m equal to the cell size makes every cell its own block, i.e. ordinary random CV.
"""
from __future__ import annotations

import numpy as np


def block_ids(xy: np.ndarray, block_m: float, offset=(0.0, 0.0)) -> np.ndarray:
    """Integer id of the block each point falls in (xy in metres, offset shifts the block grid)."""
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    bx = np.floor((xy[:, 0] + offset[0]) / block_m).astype(np.int64)
    by = np.floor((xy[:, 1] + offset[1]) / block_m).astype(np.int64)
    return bx * 1_000_003 + by


def make_folds(ids: np.ndarray, is_pos: np.ndarray, k: int, rng: np.random.Generator):
    """Assign blocks to folds, balancing positives first, then cells.

    Returns (fold_per_cell, k_effective). k_effective < k when fewer than k blocks hold positives.
    """
    blocks, inv = np.unique(ids, return_inverse=True)
    npos = np.bincount(inv, weights=is_pos.astype(float), minlength=len(blocks))
    ncell = np.bincount(inv, minlength=len(blocks)).astype(float)
    pos_blocks = np.flatnonzero(npos > 0)
    k_eff = int(min(k, len(pos_blocks)))
    if k_eff < 2:
        raise ValueError("need at least 2 blocks that contain positives; use smaller blocks")
    fold_of = np.full(len(blocks), -1, dtype=int)
    load_pos = np.zeros(k_eff)
    order = pos_blocks[np.lexsort((rng.random(len(pos_blocks)), -npos[pos_blocks]))]
    for b in order:                                   # biggest positive blocks first
        f = int(np.argmin(load_pos))
        fold_of[b] = f
        load_pos[f] += npos[b]
    load_cells = np.bincount(fold_of[pos_blocks], weights=ncell[pos_blocks], minlength=k_eff).astype(float)
    others = np.flatnonzero(fold_of < 0)
    rng.shuffle(others)
    for b in others:
        f = int(np.argmin(load_cells))
        fold_of[b] = f
        load_cells[f] += ncell[b]
    return fold_of[inv], k_eff
