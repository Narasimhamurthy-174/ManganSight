"""Phase 3: sanity-check the downloaded rasters and weather CSV.
Run: python scripts/04_check_downloads.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio

from src.common.config import resolve


def check_raster(path: Path) -> None:
    with rasterio.open(path) as ds:
        print(f"\n{path.name}\n  CRS: {ds.crs} | size: {ds.width}x{ds.height} | "
              f"bands: {ds.count} | pixel size: {ds.res}")
        for i in range(1, ds.count + 1):
            # read a 1/10-resolution preview so this is fast
            arr = ds.read(i, out_shape=(max(1, ds.height // 10), max(1, ds.width // 10)),
                          masked=True).astype(float).filled(np.nan)
            valid = np.isfinite(arr)
            zero = (arr == 0).sum() / max(1, valid.sum()) * 100
            if valid.sum() == 0:
                print(f"  band {i}: ALL EMPTY")
                continue
            print(f"  band {i} [{ds.descriptions[i-1]}]: valid {100*valid.mean():.0f}% | "
                  f"min {np.nanmin(arr):.3f} mean {np.nanmean(arr):.3f} max {np.nanmax(arr):.3f} | "
                  f"exact zeros {zero:.1f}%")


for pattern, folder in [("*.tif", "data/raw/sentinel2"), ("*.tif", "data/raw/dem")]:
    files = sorted(resolve(folder).glob(pattern))
    print(f"\n=== {folder}: {len(files)} file(s) ===")
    for f in files:
        check_raster(f)

csvs = list(resolve("data/raw/weather").glob("*.csv"))
print(f"\n=== weather: {len(csvs)} file(s) ===")
for f in csvs:
    w = pd.read_csv(f)
    w["date"] = pd.to_datetime(w["date"])
    print(w.head(3).to_string())
    print(f"rows: {len(w)} | {w.date.min().date()} to {w.date.max().date()} | "
          f"missing values: {int(w.isna().sum().sum())}")
    print("Annual precipitation totals (mm):")
    print(w.groupby(w.date.dt.year)["precip_mm"].sum().round(0).to_string())