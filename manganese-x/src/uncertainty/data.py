"""Loading the per-cell table used by the uncertainty scripts (shared by scripts 12 and 13)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import resolve
from src.features.context import focal_mean
from src.features.spatial_stats import to_raster


def load_cells(cfg: dict, U: dict) -> pd.DataFrame:
    """Features + context features + known-record flag + 5 km dense-vegetation share (column dense_veg_window)."""
    gv, size = cfg["grid"]["version"], cfg["grid"]["cell_size_m"]
    df = pd.read_csv(resolve(f"data/processed/features_{gv}.csv")).merge(
        pd.read_csv(resolve(f"data/processed/features_{gv}_context.csv")), on="cell_id", how="left", suffixes=("", "_ctx"))
    df["is_pos"] = df["cell_id"].isin(pd.read_csv(resolve("data/processed/positives_cells.csv"))["cell_id"])
    nrow, ncol = int(df["row"].max()) + 1, int(df["col"].max()) + 1
    rad = max(1, round(U["quality"]["dense_veg_window_km"] * 1000 / size))
    df["dense_veg_window"] = focal_mean(to_raster(df, "dense_veg_frac", nrow, ncol), rad)[
        df["row"].to_numpy(), df["col"].to_numpy()]
    return df


def valid_mask(df: pd.DataFrame, U: dict, sets: dict) -> np.ndarray:
    need = sorted({c for s in U["members"]["feature_sets"] for c in sets[s]} | {"dense_veg_window"})
    return df[need].notna().all(axis=1).to_numpy()
