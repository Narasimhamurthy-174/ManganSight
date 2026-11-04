"""Phase 6 EDA: redundancy check + feature maps + distributions.

Run (project root):  python scripts/07_explore_features.py
Needs: data/processed/features_<grid_version>.csv and positives_cells.csv (from script 06).
Outputs: outputs/metrics/feature_correlations.csv
         outputs/maps/feature_maps.png
         outputs/figures/feature_distributions.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.features.explore import high_corr_pairs, plot_distributions, plot_feature_maps

log = get_logger("explore_features")
ID_COLS = {"cell_id", "row", "col", "latitude", "longitude", "grid_version"}


def boundary_lines(path: Path) -> list[np.ndarray]:
    lines = []
    for geom in gpd.read_file(path).to_crs("EPSG:4326").geometry:
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        lines += [np.asarray(p.exterior.coords) for p in polys]
    return lines


def main() -> None:
    cfg = load_config()
    gv = cfg["grid"]["version"]
    df = pd.read_csv(resolve(f"data/processed/features_{gv}.csv"))
    pos_ids = pd.read_csv(resolve("data/processed/positives_cells.csv"))
    names = [c for c in df.columns if c not in ID_COLS]
    pos = df[df.cell_id.isin(pos_ids.cell_id)]
    log.info("%d cells, %d features, %d positive cells", len(df), len(names), len(pos))

    corr = df[names].corr(method="spearman")
    corr.round(3).to_csv(resolve("outputs/metrics/feature_correlations.csv"))
    for thr in (0.8, 0.6):
        pairs = high_corr_pairs(df[names], threshold=thr)
        print(f"\nFeature pairs with |Spearman rho| >= {thr}: {len(pairs)}")
        if len(pairs):
            print(pairs.round(3).to_string(index=False))

    lines = boundary_lines(resolve(cfg["study_area"]["boundary_file"]))
    maps = ["elev_m", "slope_deg", "ndvi", "dense_veg_frac", "clay_ratio", "iron_ratio"]
    plot_feature_maps(df, pos, maps, resolve("outputs/maps/feature_maps.png"), boundary_lines=lines)
    plot_distributions(df, pos, names, resolve("outputs/figures/feature_distributions.png"))
    log.info("Saved outputs/maps/feature_maps.png and outputs/figures/feature_distributions.png")


if __name__ == "__main__":
    main()
