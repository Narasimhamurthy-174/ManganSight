"""Tests use small synthetic arrays (fabricated for code testing only)."""
import numpy as np
import pytest

from src.features import spectral as sp
from src.features import terrain as tr
from src.geospatial.grid import cell_centers
from src.geospatial.gridspec import (cell_center_xy, grid_spec, in_grid, make_cell_id,
                                     raster_row, xy_to_rowcol)


def plane(h=12, w=12, res=30.0, slope=0.1):
    j = np.arange(w, dtype=np.float32)
    return np.tile(slope * res * j, (h, 1))            # rises 3 m per pixel eastwards


def test_slope_of_plane_matches_theory():
    s = tr.slope_degrees(plane(), 30.0)
    assert np.allclose(s[2:-2, 2:-2], np.degrees(np.arctan(0.1)), atol=1e-3)
    assert np.isnan(s[0, 0])                            # border has missing neighbours


def test_tri_of_plane():
    t = tr.tri(plane())
    assert np.allclose(t[2:-2, 2:-2], 2.25, atol=1e-4)  # 18 m summed difference / 8


def test_relief_window():
    r = tr.relief(plane(), 3)
    assert np.allclose(r[2:-2, 2:-2], 6.0)


def test_nodata_propagates_and_does_not_crash():
    d = plane()
    d[4:7, 4:7] = np.nan
    s, r = tr.slope_degrees(d, 30.0), tr.relief(d, 3)
    assert np.isnan(s[5, 5]) and np.isnan(r[5, 5])
    assert np.isfinite(s[9, 9]) and np.isfinite(r[9, 9])


def test_bad_arguments():
    with pytest.raises(ValueError):
        tr.relief(plane(), 4)
    with pytest.raises(ValueError):
        tr.slope_degrees(plane(), 0)


def test_ndvi_and_safe_division():
    b8, b4 = np.array([0.4, 0.0, np.nan], np.float32), np.array([0.1, 0.0, 0.1], np.float32)
    out = sp.ndvi(b8, b4)
    assert np.isclose(out[0], 0.6) and np.isnan(out[1]) and np.isnan(out[2])


def test_ratios_and_brightness():
    a = np.array([0.3], np.float32)
    assert np.isclose(sp.clay_ratio(a, np.array([0.2], np.float32))[0], 1.5)
    assert np.isclose(sp.iron_ratio(np.array([0.1], np.float32), np.array([0.05], np.float32))[0], 2.0)
    assert np.isclose(sp.vis_brightness(a, a, a)[0], 0.3)


def test_dense_veg_flag_keeps_nan():
    f = sp.dense_veg_flag(np.array([0.7, 0.5, np.nan], np.float32))
    assert f[0] == 1 and f[1] == 0 and np.isnan(f[2])


def test_gridspec_matches_cell_centers():
    minx, miny, maxx, maxy = 1010.0, 2020.0, 1600.0, 2600.0
    spec = grid_spec(minx, miny, maxx, maxy, 250.0)
    c = cell_centers(minx, miny, maxx, maxy, 250.0)
    assert len(c) == spec.nrow * spec.ncol
    x, y = cell_center_xy(c.row.values, c.col.values, spec)
    assert np.allclose(x, c.x.values) and np.allclose(y, c.y.values)
    assert make_cell_id(c.row, c.col) == list(c.cell_id)


def test_xy_roundtrip_and_raster_row():
    spec = grid_spec(0, 0, 1000, 500, 250.0)
    row, col = xy_to_rowcol([125.0, 990.0], [125.0, 490.0], spec)
    assert list(row) == [0, 1] and list(col) == [0, 3]
    assert list(raster_row(row, spec)) == [1, 0]        # south row 0 -> bottom raster row
    assert list(in_grid([0, 5, -1], [0, 0, 0], spec)) == [True, False, False]
