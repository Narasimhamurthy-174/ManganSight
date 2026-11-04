"""Dry-season (Feb-May, 2024 and 2025) Sentinel-2 composite for Balaghat.
Run: python scripts/gee/export_sentinel2_dry.py
"""
import ee
from _common import init_ee, balaghat_geometry
from export_sentinel2 import mask_scl
from src.common.config import load_config


def main() -> None:
    cfg = load_config()["sentinel2"]
    init_ee()
    geom = balaghat_geometry()
    col = (ee.ImageCollection(cfg["collection"])
           .filterBounds(geom)
           .filterDate(cfg["date_start"], cfg["date_end"])
           .filter(ee.Filter.calendarRange(2, 5, "month"))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["max_scene_cloud_pct"]))
           .map(mask_scl))
    print("[info] dry-season scenes:", col.size().getInfo())
    median = col.select(cfg["bands"]).median().divide(10000)
    nobs = col.select("B4").count().rename("n_valid_obs")
    stack = median.addBands(nobs).toFloat().clip(geom)
    ee.batch.Export.image.toDrive(
        image=stack, description="s2_dry_balaghat", fileNamePrefix="s2_dry_balaghat",
        region=geom, scale=cfg["export_scale_m"], crs="EPSG:32644", maxPixels=1e10).start()
    print("Started export 's2_dry_balaghat'.")


if __name__ == "__main__":
    main()