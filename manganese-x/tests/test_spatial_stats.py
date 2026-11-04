"""Synthetic fields (fabricated for code testing only)."""
import numpy as np
import pandas as pd
from scipy import ndimage

from src.features.spatial_stats import plot_variograms, practical_range, semivariogram, to_raster


def smooth_field(size=300, window=15, seed=0):
    rng = np.random.default_rng(seed)
    f = ndimage.uniform_filter(rng.normal(size=(size, size)), size=window, mode="wrap")
    return ((f - f.mean()) / f.std()).astype(np.float32)


def test_white_noise_has_no_spatial_structure():
    rng = np.random.default_rng(1)
    vg = semivariogram(rng.normal(size=(300, 300)).astype(np.float32), [1, 5, 20])
    assert (vg["gamma"].between(0.9, 1.1)).all()
    assert practical_range(vg) == 1.0


def test_smooth_field_range_close_to_window():
    vg = semivariogram(smooth_field(), [1, 2, 4, 8, 12, 15, 20, 40])
    g = vg.set_index("lag_cells")["gamma"]
    assert g[1] < 0.2 and g[40] > 0.9
    assert 8 <= practical_range(vg) <= 20


def test_nan_cells_are_ignored():
    f = smooth_field()
    f[:, 150:] = np.nan                       # half the raster is outside the 'district'
    vg = semivariogram(f, [1, 10])
    assert np.isfinite(vg["gamma"]).all() and (vg["n_pairs"] > 1000).all()


def test_too_few_pairs_gives_nan_and_no_range():
    f = np.full((50, 50), np.nan, np.float32)
    f[:5, :5] = np.random.default_rng(2).normal(size=(5, 5))
    vg = semivariogram(f, [1, 2])
    assert vg["gamma"].isna().all() and np.isnan(practical_range(vg))


def test_to_raster_places_values_and_leaves_nan():
    df = pd.DataFrame({"row": [0, 1], "col": [1, 0], "v": [5.0, 7.0]})
    a = to_raster(df, "v", 3, 3)
    assert a[0, 1] == 5 and a[1, 0] == 7 and np.isnan(a[2, 2])


def test_plot_writes_file(tmp_path):
    vg = semivariogram(smooth_field(), [1, 4, 16])
    plot_variograms({"a": vg}, 250, tmp_path / "v.png", highlight=["a"])
    assert (tmp_path / "v.png").stat().st_size > 1000
