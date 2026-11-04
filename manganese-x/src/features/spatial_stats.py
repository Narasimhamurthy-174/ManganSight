"""Spatial autocorrelation of raster-like feature grids (label-free).

Why: spatial cross-validation blocks must be at least as large as the distance over which
neighbouring cells stay similar; otherwise 'test' cells sit next to 'training' cells and the
score is optimistic (Roberts et al., 2017). We measure that distance instead of guessing.

Because our cells form a regular grid, semivariance at a lag of k cells can be computed
exactly with array shifts along rows and columns (no pair sampling needed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 240]


def to_raster(df: pd.DataFrame, value_col: str, nrow: int | None = None,
              ncol: int | None = None) -> np.ndarray:
    """Place per-cell values into a (row, col) array; cells not in the district stay NaN."""
    nrow = int(df["row"].max()) + 1 if nrow is None else nrow
    ncol = int(df["col"].max()) + 1 if ncol is None else ncol
    arr = np.full((nrow, ncol), np.nan, dtype=np.float32)
    arr[df["row"].to_numpy(), df["col"].to_numpy()] = df[value_col].to_numpy()
    return arr


def semivariogram(arr: np.ndarray, lags_cells=None, min_pairs: int = 1000) -> pd.DataFrame:
    """Omnidirectional (E-W and N-S pooled) semivariance of the z-scored array.

    gamma(h) = 0.5 * mean[(z(x) - z(x+h))^2]. For a trend-free field gamma rises to ~1
    (the variance, the 'sill') once h exceeds the autocorrelation range.
    """
    lags = DEFAULT_LAGS if lags_cells is None else list(lags_cells)
    z = (arr - np.nanmean(arr)) / np.nanstd(arr)
    rows = []
    for k in lags:
        num, cnt = 0.0, 0
        if k < z.shape[1]:
            d = z[:, :-k] - z[:, k:]
            m = np.isfinite(d)
            num += float(np.sum(d[m] ** 2, dtype=np.float64))
            cnt += int(m.sum())
        if k < z.shape[0]:
            d = z[:-k, :] - z[k:, :]
            m = np.isfinite(d)
            num += float(np.sum(d[m] ** 2, dtype=np.float64))
            cnt += int(m.sum())
        rows.append((k, 0.5 * num / cnt if cnt >= min_pairs else np.nan, cnt))
    return pd.DataFrame(rows, columns=["lag_cells", "gamma", "n_pairs"])


def practical_range(vg: pd.DataFrame, sill: float = 1.0, frac: float = 0.95) -> float:
    """First lag (in cells) where gamma reaches frac*sill; NaN if never reached (long-range trend)."""
    hit = vg[vg["gamma"] >= frac * sill]
    return float(hit["lag_cells"].iloc[0]) if len(hit) else float("nan")


def plot_variograms(curves: dict[str, pd.DataFrame], cell_size_m: float, path, highlight=()) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vg in curves.items():
        km = vg["lag_cells"] * cell_size_m / 1000
        ax.plot(km, vg["gamma"], marker="o", ms=3, lw=2.2 if name in highlight else 1,
                alpha=1 if name in highlight else 0.5, label=name)
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 0.97, " 95% of sill", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Lag distance (km, log scale)")
    ax.set_ylabel("Semivariance of z-scored feature")
    ax.set_title("How far do neighbouring cells stay similar? (label-free, all cells)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
