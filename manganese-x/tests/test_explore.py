"""Synthetic data (fabricated for code testing only)."""
import numpy as np
import pandas as pd

from src.features.explore import high_corr_pairs, plot_distributions, plot_feature_maps


def toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.01, size=n),
                         "c": rng.normal(size=n), "latitude": rng.uniform(21, 22, n),
                         "longitude": rng.uniform(80, 81, n)})


def test_high_corr_pairs_finds_only_the_redundant_pair():
    p = high_corr_pairs(toy()[["a", "b", "c"]], 0.8)
    assert len(p) == 1 and set(p.iloc[0][["feature_a", "feature_b"]]) == {"a", "b"}


def test_no_pairs_when_threshold_not_met():
    assert high_corr_pairs(toy()[["a", "c"]], 0.8).empty


def test_plots_write_files(tmp_path):
    df = toy()
    pos = df.head(5)
    plot_feature_maps(df, pos, ["a", "c"], tmp_path / "m.png",
                      boundary_lines=[np.array([[80, 21], [81, 21], [81, 22], [80, 22], [80, 21]])])
    plot_distributions(df, pos, ["a", "b", "c"], tmp_path / "d.png")
    assert (tmp_path / "m.png").stat().st_size > 1000 and (tmp_path / "d.png").stat().st_size > 1000
