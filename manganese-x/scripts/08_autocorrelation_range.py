"""Phase 6/9: how far does spatial autocorrelation reach? (decides spatial-CV block size)

Run (project root):  python scripts/08_autocorrelation_range.py
Needs: data/processed/features_<grid_version>.csv, config/features.yaml
Outputs: outputs/metrics/autocorrelation_ranges.csv, outputs/figures/variograms.png
Uses NO labels: only the feature grids.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.features.spatial_stats import (DEFAULT_LAGS, plot_variograms, practical_range,
                                        semivariogram, to_raster)

log = get_logger("autocorr")
ID_COLS = {"cell_id", "row", "col", "latitude", "longitude", "grid_version"}


def main() -> None:
    cfg = load_config()
    size = cfg["grid"]["cell_size_m"]
    df = pd.read_csv(resolve(f"data/processed/features_{cfg['grid']['version']}.csv"))
    fs = yaml.safe_load(resolve("config/features.yaml").read_text(encoding="utf-8"))
    names = [c for c in df.columns if c not in ID_COLS]
    log.info("%d cells; computing semivariograms for %d features", len(df), len(names))

    curves, rows = {}, []
    for name in names:
        vg = semivariogram(to_raster(df, name))
        curves[name] = vg
        pr = practical_range(vg)
        role = ("predictor" if name in fs["predictors"] else
                "quality" if name in fs["quality_indicators"] else "dropped")
        rows.append({"feature": name, "role": role,
                     "range_km": round(pr * size / 1000, 2) if pr == pr else float("nan"),
                     "gamma_at_1km": vg.loc[vg.lag_cells == 4, "gamma"].iloc[0],
                     "gamma_at_5km": vg.loc[vg.lag_cells == 24, "gamma"].iloc[0],
                     "gamma_at_20km": vg.loc[vg.lag_cells == 96, "gamma"].iloc[0]})
    out = pd.DataFrame(rows).round(3)
    max_km = max(DEFAULT_LAGS) * size / 1000
    print(f"\nPractical range = distance where semivariance reaches 95% of the variance "
          f"(NaN = not reached within {max_km:.0f} km, i.e. a long-range regional trend)\n")
    print(out.sort_values(["role", "range_km"]).to_string(index=False))
    out.to_csv(resolve("outputs/metrics/autocorrelation_ranges.csv"), index=False)
    plot_variograms(curves, size, resolve("outputs/figures/variograms.png"), highlight=fs["predictors"])
    log.info("Saved outputs/metrics/autocorrelation_ranges.csv and outputs/figures/variograms.png")


if __name__ == "__main__":
    main()
