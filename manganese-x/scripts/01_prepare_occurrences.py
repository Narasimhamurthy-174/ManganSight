"""Phase 2/3: raw USGS MRDS -> clean manganese table with district flags.

Run:  python scripts/01_prepare_occurrences.py --mrds data/raw/occurrences/mrds.shp
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.data.occurrences import clean_occurrences, standardize_mrds, usable_positives

log = get_logger("prepare_occurrences")


def add_district_flags(df: pd.DataFrame, boundary_path: Path, crs_proj: str) -> pd.DataFrame:
    """Add in_district (bool) and km_from_district (0 inside, else distance to boundary)."""
    boundary = gpd.read_file(boundary_path).to_crs(crs_proj)
    union = boundary.union_all()
    out = df.copy()
    out["in_district"] = False
    out["km_from_district"] = np.nan
    ok = out["coord_valid"]
    if ok.any():
        pts = gpd.GeoSeries(
            gpd.points_from_xy(out.loc[ok, "longitude"], out.loc[ok, "latitude"]),
            index=out.index[ok], crs="EPSG:4326").to_crs(crs_proj)
        out.loc[ok, "in_district"] = pts.within(union).to_numpy()
        out.loc[ok, "km_from_district"] = (pts.distance(union) / 1000).to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrds", required=True, help="path to mrds.shp (or .csv)")
    args = ap.parse_args()
    cfg = load_config()
    b = cfg["study_area"]["sanity_bounds"]

    path = Path(args.mrds)
    if path.suffix == ".csv":
        raw = pd.read_csv(path, low_memory=False, encoding="latin-1")
    else:
        raw = pd.DataFrame(gpd.read_file(path).drop(columns="geometry"))
    log.info("Loaded %d raw MRDS rows; columns: %s", len(raw), list(raw.columns)[:15])

    raw.columns = [c.lower() for c in raw.columns]
    lat = pd.to_numeric(raw["latitude"], errors="coerce")
    lon = pd.to_numeric(raw["longitude"], errors="coerce")
    raw = raw[lat.between(b["lat_min"], b["lat_max"]) & lon.between(b["lon_min"], b["lon_max"])]
    log.info("%d rows inside the sanity box", len(raw))

    mn = standardize_mrds(raw)
    log.info("%d manganese records in the box", len(mn))

    cleaned = clean_occurrences(mn, b)
    cleaned = add_district_flags(cleaned, resolve(cfg["study_area"]["boundary_file"]),
                                 cfg["crs"]["projected"])
    cleaned.to_csv(resolve("data/interim/occurrences_mrds_all.csv"), index=False)

    pos = usable_positives(cleaned)
    pos.to_csv(resolve("data/interim/occurrences_mrds_usable.csv"), index=False)

    n_in = int(pos["in_district"].sum())
    n_near = int((pos["km_from_district"] <= 25).sum())
    log.info("Usable positives (valid coords, no duplicates): %d", len(pos))
    log.info("  inside Balaghat district: %d", n_in)
    log.info("  within 25 km of the district (incl. inside): %d", n_near)
    if n_in < 30:
        log.warning("Only %d usable positives inside Balaghat. Complex models will overfit; "
                    "supplement from GSI/NGDR/MOIL mine locations.", n_in)


if __name__ == "__main__":
    main()