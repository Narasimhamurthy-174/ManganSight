"""Phase 6: neighbourhood-averaged features and a topographic position index.

Run (project root):  python scripts/10_build_context_features.py
Needs: data/processed/features_<grid_version>.csv, config/features.yaml
Outputs: data/processed/features_<grid_version>_context.csv (cell_id + new columns)
         data/metadata/features_<grid_version>_context.json
No labels are used. Radii are fixed a priori: 1, 2.5 and 5 km windows (label-position tolerance / regional context) and a 5 km TPI.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.features.context import focal_mean, tpi
from src.features.spatial_stats import to_raster

log = get_logger("context_features")
SCALES = {"r1km": 1.0, "r2p5km": 2.5, "r5km": 5.0}      # name -> radius in km
TPI_KM = 5.0


def main() -> None:
    cfg = load_config()
    size, gv = cfg["grid"]["cell_size_m"], cfg["grid"]["version"]
    df = pd.read_csv(resolve(f"data/processed/features_{gv}.csv"))
    base = yaml.safe_load(resolve("config/features.yaml").read_text(encoding="utf-8"))["predictors"]
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    out = df[["cell_id"]].copy()
    for f in base:
        arr = to_raster(df, f)
        for tag, km in SCALES.items():
            rad = max(1, round(km * 1000 / size))
            out[f"{f}_{tag}"] = focal_mean(arr, rad)[r, c]
            log.info("%s %s: radius %d cells (~%.2f km)", f, tag, rad, rad * size / 1000)
    rad = max(1, round(TPI_KM * 1000 / size))
    out["tpi_5km"] = tpi(to_raster(df, "elev_m"), rad)[r, c]
    new = [c_ for c_ in out.columns if c_ != "cell_id"]
    print("\nMissing share per new column (%):")
    print((out[new].isna().mean() * 100).round(2).to_string())
    print("\nSpearman correlation of tpi_5km with the base predictors (want |rho| < 0.8 to add information):")
    print(pd.concat([df[base], out["tpi_5km"]], axis=1).corr(method="spearman")["tpi_5km"].drop("tpi_5km").round(3).to_string())
    print("\nSpearman correlation of each 1 km mean with its 250 m original:")
    for f in base:
        print(f"  {f}: {df[f].corr(out[f + '_r1km'], method='spearman'):.3f}")
    path = resolve(f"data/processed/features_{gv}_context.csv")
    out.to_csv(path, index=False)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "grid_version": gv,
            "scales_km": SCALES, "tpi_km": TPI_KM, "columns": new, "data_class": "DERIVED",
            "method": "square-window mean ignoring NaN; tpi = elevation minus window mean"}
    resolve(f"data/metadata/features_{gv}_context.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Saved %s", path)


if __name__ == "__main__":
    main()
