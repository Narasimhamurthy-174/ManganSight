"""Phase 5: terrain + spectral features for every 250 m grid cell.

Run (from the project root):
    python scripts/06_extract_features.py
Optional:
    python scripts/06_extract_features.py --s2 data/raw/sentinel2/s2_dry_balaghat.tif

Inputs : data/processed/<grid_version>.gpkg, Sentinel-2 composite, SRTM DEM,
         data/interim/occurrences_mrds_usable.csv, data/raw/occurrences/mrds.csv
Outputs: data/processed/features_<grid_version>.csv
         data/processed/positives_cells.csv        (which cell each known record falls in)
         data/metadata/features_<grid_version>.json (what was used, for provenance)

Method: indices are computed at native resolution (20 m / 30 m), then AVERAGED into each
250 m cell with GDAL average resampling onto a raster aligned exactly with our grid.
NOTE: first version, UNTESTED on real data in the authoring sandbox (no rasterio there).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.features import spectral as sp
from src.features import terrain as tr
from src.geospatial.gridspec import (GridSpec, cell_center_xy, grid_spec, in_grid,
                                     make_cell_id, raster_row, xy_to_rowcol)

log = get_logger("extract_features")
RELIEF_WINDOW_M = 1000          # local relative relief window (assumption, documented)


def make_spec(cfg) -> GridSpec:
    """Rebuild the exact grid geometry used by scripts/02_build_grid.py."""
    b = gpd.read_file(resolve(cfg["study_area"]["boundary_file"])).to_crs(cfg["crs"]["projected"])
    union = b.union_all() if hasattr(b, "union_all") else b.unary_union
    minx, miny, maxx, maxy = union.bounds
    return grid_spec(minx, miny, maxx, maxy, cfg["grid"]["cell_size_m"])


def to_grid(arr: np.ndarray, src_transform, crs, spec: GridSpec) -> np.ndarray:
    """Average a (north-up) raster into our grid. NaN pixels are ignored by the average."""
    dst = np.full((spec.nrow, spec.ncol), np.nan, dtype=np.float32)
    reproject(source=arr.astype(np.float32, copy=False), destination=dst,
              src_transform=src_transform, src_crs=crs,
              dst_transform=from_origin(spec.west, spec.north, spec.size, spec.size), dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    return dst


def spectral_features(s2_path: Path, spec: GridSpec) -> dict[str, np.ndarray]:
    need = ["B2", "B3", "B4", "B8", "B11", "B12", "n_valid_obs"]
    with rasterio.open(s2_path) as ds:
        names = list(ds.descriptions)
        missing = [n for n in need if n not in names]
        if missing:
            raise SystemExit(f"{s2_path.name}: bands {missing} not found. Band names: {names}")
        b = {n: ds.read(names.index(n) + 1, masked=True).astype(np.float32).filled(np.nan)
             for n in need}
        tf, crs = ds.transform, ds.crs
    if str(crs) != "EPSG:32644":
        raise SystemExit(f"Expected EPSG:32644 raster, got {crs}")
    log.info("Sentinel-2 read: %s pixels", b["B4"].size)
    out: dict[str, np.ndarray] = {}
    nd = sp.ndvi(b["B8"], b["B4"])
    out["ndvi"] = to_grid(nd, tf, crs, spec)
    out["dense_veg_frac"] = to_grid(sp.dense_veg_flag(nd), tf, crs, spec)
    del nd
    out["clay_ratio"] = to_grid(sp.clay_ratio(b["B11"], b["B12"]), tf, crs, spec)
    out["iron_ratio"] = to_grid(sp.iron_ratio(b["B4"], b["B2"]), tf, crs, spec)
    out["vis_brightness"] = to_grid(sp.vis_brightness(b["B2"], b["B3"], b["B4"]), tf, crs, spec)
    out["swir1"] = to_grid(b["B11"], tf, crs, spec)
    out["s2_nobs"] = to_grid(b["n_valid_obs"], tf, crs, spec)
    return out


def terrain_features(dem_path: Path, spec: GridSpec) -> dict[str, np.ndarray]:
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        res, tf, crs = float(ds.res[0]), ds.transform, ds.crs
    zeros = float((dem == 0).mean())
    dem[dem == 0] = np.nan          # outside the district the export wrote 0 (not nodata)
    log.info("DEM read; %.1f%% zero pixels treated as no-data; elevation %.0f-%.0f m",
             100 * zeros, np.nanmin(dem), np.nanmax(dem))
    window = int(round(RELIEF_WINDOW_M / res)) | 1          # force odd
    return {
        "elev_m": to_grid(dem, tf, crs, spec),
        "slope_deg": to_grid(tr.slope_degrees(dem, res), tf, crs, spec),
        "tri_m": to_grid(tr.tri(dem), tf, crs, spec),
        "relief_1km_m": to_grid(tr.relief(dem, window), tf, crs, spec),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", default="data/raw/sentinel2/s2_dry_balaghat.tif")
    ap.add_argument("--dem", default="data/raw/dem/srtm_balaghat.tif")
    args = ap.parse_args()
    cfg = load_config()
    proj, gv = cfg["crs"]["projected"], cfg["grid"]["version"]

    spec = make_spec(cfg)
    log.info("Grid spec: %d rows x %d cols, origin (%.0f, %.0f)", spec.nrow, spec.ncol, spec.x0, spec.y0)

    g = gpd.read_file(resolve(f"data/processed/{gv}.gpkg"))
    cells = pd.DataFrame(g.drop(columns="geometry"))
    log.info("Grid file: %d cells", len(cells))

    # --- alignment check: our (row, col) maths must reproduce the lon/lat stored in the grid file
    x, y = cell_center_xy(cells.row.values, cells.col.values, spec)
    lon, lat = Transformer.from_crs(proj, "EPSG:4326", always_xy=True).transform(x, y)
    err = max(np.abs(lon - cells.longitude.values).max(), np.abs(lat - cells.latitude.values).max())
    if err > 1e-6 or not in_grid(cells.row.values, cells.col.values, spec).all():
        raise SystemExit(f"Grid alignment check FAILED (max error {err:.2e} deg). Stop and tell your mentor.")
    log.info("Grid alignment check passed (max error %.1e deg)", err)

    feats = {}
    feats.update(terrain_features(resolve(args.dem), spec))
    feats.update(spectral_features(resolve(args.s2), spec))

    rr, cc = raster_row(cells.row.values, spec), cells.col.values
    for name, arr in feats.items():
        cells[name] = arr[rr, cc]
    names = list(feats)

    print("\nMissing values per feature (share of cells):")
    print((cells[names].isna().mean() * 100).round(2).astype(str).add(" %").to_string())
    print("\nFeature summary over all cells:")
    print(cells[names].describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())

    out_csv = resolve(f"data/processed/features_{gv}.csv")
    cells.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    # --- which grid cell does each known record fall in?
    pos = pd.read_csv(resolve("data/interim/occurrences_mrds_usable.csv"))
    pos = pos[pos["in_district"]].copy()
    raw = pd.read_csv(resolve("data/raw/occurrences/mrds.csv"), low_memory=False, encoding="latin-1",
                      usecols=lambda c: c.lower() in ("dep_id", "dev_stat", "site_name"))
    raw.columns = [c.lower() for c in raw.columns]
    pos = pos.merge(raw, left_on="occurrence_id", right_on="dep_id", how="left")
    px, py = Transformer.from_crs("EPSG:4326", proj, always_xy=True).transform(
        pos.longitude.values, pos.latitude.values)
    prow, pcol = xy_to_rowcol(px, py, spec)
    pos["cell_id"] = make_cell_id(prow, pcol)
    pos["cell_in_grid"] = pos.cell_id.isin(cells.cell_id)
    log.info("%d positives; %d fall in a grid cell; %d distinct cells",
             len(pos), int(pos.cell_in_grid.sum()), pos.cell_id.nunique())
    pos[["occurrence_id", "site_name", "dev_stat", "latitude", "longitude", "cell_id",
         "cell_in_grid"]].to_csv(resolve("data/processed/positives_cells.csv"), index=False)

    # --- DESCRIPTIVE preview (not a model result): do positive cells look different?
    hit = cells[cells.cell_id.isin(pos.cell_id)]
    pct = cells[names].rank(pct=True)
    table = pd.DataFrame({"district_median": cells[names].median(),
                          "positives_median": hit[names].median(),
                          "positives_median_percentile": pct.loc[hit.index].median().mul(100)})
    print(f"\nDESCRIPTIVE PREVIEW ({len(hit)} positive cells vs all {len(cells)} cells). "
          "Not a model result; positives are spatially clustered, so do not over-read this.")
    print(table.round(3).to_string())

    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "grid_version": gv, "cell_size_m": spec.size, "n_cells": int(len(cells)),
            "s2_file": args.s2, "dem_file": args.dem, "features": names,
            "dense_veg_ndvi_threshold": sp.DENSE_VEG_NDVI, "relief_window_m": RELIEF_WINDOW_M,
            "aggregation": "GDAL average resampling of native-resolution values into grid cells",
            "dem_zero_pixels_treated_as_nodata": True, "data_class": "DERIVED"}
    resolve(f"data/metadata/features_{gv}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
