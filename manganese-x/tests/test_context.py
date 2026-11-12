"""Synthetic arrays (fabricated for code testing only)."""
import numpy as np
import pytest

from src.features.context import focal_mean, tpi


def test_constant_field_unchanged_inside():
    a = np.full((40, 40), 7.0, np.float32)
    out = focal_mean(a, 3)
    assert np.allclose(out[5:-5, 5:-5], 7.0)


def test_nan_cells_ignored_and_stay_nan():
    a = np.full((40, 40), 5.0, np.float32)
    a[:, 20:] = np.nan                                  # outside the district
    out = focal_mean(a, 2)
    assert np.isnan(out[10, 30])                         # stays NaN outside
    assert np.isclose(out[10, 19], 5.0)                  # edge cell averages only valid neighbours


def test_sparse_neighbourhood_gives_nan():
    a = np.full((30, 30), np.nan, np.float32)
    a[15, 15] = 1.0                                      # a single valid cell
    assert np.isnan(focal_mean(a, 4)[15, 15])


def test_tpi_zero_on_plane_and_signed_on_bump():
    plane = np.tile(np.arange(60, dtype=np.float32) * 3.0, (60, 1))
    assert np.allclose(tpi(plane, 5)[10:-10, 10:-10], 0.0, atol=1e-3)
    bump = np.zeros((41, 41), np.float32)
    bump[20, 20] = 10.0
    assert tpi(bump, 5)[20, 20] > 5 and tpi(bump, 5)[20, 22] < 0.5


def test_bad_radius():
    with pytest.raises(ValueError):
        focal_mean(np.ones((5, 5)), 0)
