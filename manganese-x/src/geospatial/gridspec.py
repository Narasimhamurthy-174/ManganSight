"""Grid geometry bookkeeping: maps (row, col) cells <-> projected x/y <-> raster rows.

Must stay consistent with src/geospatial/grid.py::cell_centers (same snapping rule):
cell (row, col) is counted from the snapped south-west origin (x0, y0); row grows NORTH.
Raster arrays (north-up) have row 0 at the north, so raster_row = nrow - 1 - row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    x0: float      # west edge of column 0 (projected metres)
    y0: float      # south edge of row 0
    size: float    # cell size (m)
    nrow: int
    ncol: int

    @property
    def west(self) -> float:
        return self.x0

    @property
    def north(self) -> float:
        return self.y0 + self.nrow * self.size


def grid_spec(minx: float, miny: float, maxx: float, maxy: float, size: float) -> GridSpec:
    if size <= 0:
        raise ValueError("size must be positive")
    if maxx <= minx or maxy <= miny:
        raise ValueError("bbox is empty or inverted")
    x0 = float(np.floor(minx / size) * size)
    y0 = float(np.floor(miny / size) * size)
    return GridSpec(x0, y0, float(size),
                    nrow=int(np.ceil((maxy - y0) / size)),
                    ncol=int(np.ceil((maxx - x0) / size)))


def cell_center_xy(row, col, spec: GridSpec):
    row, col = np.asarray(row), np.asarray(col)
    return spec.x0 + (col + 0.5) * spec.size, spec.y0 + (row + 0.5) * spec.size


def xy_to_rowcol(x, y, spec: GridSpec):
    """Projected coordinates -> integer (row, col). May fall outside the grid; check with in_grid()."""
    col = np.floor((np.asarray(x) - spec.x0) / spec.size).astype(int)
    row = np.floor((np.asarray(y) - spec.y0) / spec.size).astype(int)
    return row, col


def in_grid(row, col, spec: GridSpec):
    row, col = np.asarray(row), np.asarray(col)
    return (row >= 0) & (row < spec.nrow) & (col >= 0) & (col < spec.ncol)


def raster_row(row, spec: GridSpec):
    return spec.nrow - 1 - np.asarray(row)


def make_cell_id(row, col) -> list[str]:
    return [f"r{int(r):04d}c{int(c):04d}" for r, c in zip(row, col)]
