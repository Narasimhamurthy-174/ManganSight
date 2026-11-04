"""Exploratory helpers: redundancy check and plots for the per-cell feature table."""
from __future__ import annotations

import numpy as np
import pandas as pd


def high_corr_pairs(df: pd.DataFrame, threshold: float = 0.8, method: str = "spearman") -> pd.DataFrame:
    """Feature pairs whose absolute rank correlation is >= threshold (candidates for dropping one)."""
    corr = df.corr(method=method)
    cols = list(corr.columns)
    rows = [(a, b, float(corr.loc[a, b]))
            for i, a in enumerate(cols) for b in cols[i + 1:]
            if abs(corr.loc[a, b]) >= threshold]
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "rho"])
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_feature_maps(df, pos, names, path, boundary_lines=None, ncols=3):
    """Scatter-map each feature (cell centres) with known records on top. Saves a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), squeeze=False)
    aspect = 1.0 / np.cos(np.radians(df["latitude"].mean()))
    for ax, name in zip(axes.ravel(), names):
        v = df[name].to_numpy()
        lo, hi = np.nanpercentile(v, [1, 99])
        sc = ax.scatter(df["longitude"], df["latitude"], c=v, s=0.8, marker="s", linewidths=0,
                        vmin=lo, vmax=hi, cmap="viridis", rasterized=True)
        for line in boundary_lines or []:
            ax.plot(line[:, 0], line[:, 1], color="black", linewidth=0.7)
        ax.scatter(pos["longitude"], pos["latitude"], s=9, facecolor="none", edgecolor="red", linewidths=0.7)
        ax.set_title(name, fontsize=11)
        ax.set_aspect(aspect)
        ax.tick_params(labelsize=7)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Derived features per 250 m cell (red rings = known USGS MRDS records). "
                 "Source data / derived features, not model output.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_distributions(df, pos, names, path, ncols=4):
    """Histogram of all cells vs known-record cells for every feature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        lo, hi = np.nanpercentile(df[name], [0.5, 99.5])
        ax.hist(df[name].dropna(), bins=50, range=(lo, hi), density=True, alpha=0.5, label="all cells")
        ax.hist(pos[name].dropna(), bins=15, range=(lo, hi), density=True, histtype="step",
                linewidth=2, color="red", label=f"known records (n={pos[name].notna().sum()})")
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Do known-record cells differ from the district? (descriptive, 59 clustered points)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
