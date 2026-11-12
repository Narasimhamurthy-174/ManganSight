"""Turn a map of 'top priority' cells into contiguous zones."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def label_zones(row: np.ndarray, col: np.ndarray, mask: np.ndarray, nrow: int, ncol: int,
                min_cells: int = 8) -> np.ndarray:
    """Zone id per cell (0 = not in a zone). Zones = 8-connected groups of top cells with >= min_cells cells,
    numbered from the largest."""
    grid = np.zeros((nrow, ncol), bool)
    grid[row[mask], col[mask]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sizes >= min_cells)
    keep = keep[keep > 0]
    order = keep[np.argsort(-sizes[keep], kind="stable")]
    remap = np.zeros(n + 1, int)
    remap[order] = np.arange(1, len(order) + 1)
    return remap[lab[row, col]]
