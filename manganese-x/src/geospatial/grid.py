"""Analysis grid. Every model row in MANGANESE-X is one grid cell.

The pure-numpy part (cell_centers) is unit-tested without GIS libraries.
build_grid() needs geopandas/shapely/pyproj.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cell_centers(minx: float, miny: float, maxx: float, maxy: float,
                 size: float) -> pd.DataFrame:
    """Regular grid of cell centres covering a bbox (projected metres).

    Origin is snapped to a multiple of `size` so the grid is identical every run.
    cell_id is deterministic: r{row}c{col} counted from the snapped origin.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if maxx <= minx or maxy <= miny:
        raise ValueError("bbox is empty or inverted")
    x0 = np.floor(minx / size) * size
    y0 = np.floor(miny / size) * size
    ncol = int(np.ceil((maxx - x0) / size))
    nrow = int(np.ceil((maxy - y0) / size))
    cols, rows = np.meshgrid(np.arange(ncol), np.arange(nrow))
    cols, rows = cols.ravel(), rows.ravel()
    return pd.DataFrame({
        "cell_id": [f"r{r:04d}c{c:04d}" for r, c in zip(rows, cols)],
        "row": rows, "col": cols,
        "x": x0 + (cols + 0.5) * size,
        "y": y0 + (rows + 0.5) * size,
    })


def build_grid(boundary_path: str | Path, cell_size_m: float, crs_projected: str,
               grid_version: str):
    """Clip a regular grid to the district boundary. Returns a GeoDataFrame (EPSG:4326)
    with cell_id, latitude, longitude, geometry (cell polygon)."""
    import geopandas as gpd
    import shapely

    boundary = gpd.read_file(boundary_path).to_crs(crs_projected)
    union = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    minx, miny, maxx, maxy = union.bounds
    centers = cell_centers(minx, miny, maxx, maxy, cell_size_m)
    half = cell_size_m / 2
    polys = shapely.box(centers.x - half, centers.y - half, centers.x + half, centers.y + half)
    grid = gpd.GeoDataFrame(centers, geometry=polys, crs=crs_projected)
    grid = grid[grid.intersects(union)].reset_index(drop=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(grid.x, grid.y), crs=crs_projected).to_crs("EPSG:4326")
    grid["latitude"], grid["longitude"] = pts.y.values, pts.x.values
    grid["grid_version"] = grid_version
    return grid.to_crs("EPSG:4326")[["cell_id", "row", "col", "latitude", "longitude",
                                     "grid_version", "geometry"]]
