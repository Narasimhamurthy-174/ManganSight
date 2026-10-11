"""Export DAILY district-mean precipitation and top-layer soil water from ERA5-Land.
Output: CSV with columns date, precip_mm, soil_water_l1.
[Verify] Band names in the Code Editor:
   total_precipitation_sum (metres of water), volumetric_soil_water_layer_1 (m3/m3)
Run: python scripts/gee/export_era5land.py
"""
import ee
from _common import init_ee, balaghat_geometry
from src.common.config import load_config


def main() -> None:
    cfg = load_config()["weather"]
    init_ee()
    geom = balaghat_geometry()
    col = (ee.ImageCollection(cfg["collection"])
           .filterDate(cfg["date_start"], cfg["date_end"])
           .select(["total_precipitation_sum", "volumetric_soil_water_layer_1"]))

    def per_day(img):
        stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=11132,
                                 maxPixels=1e9)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "precip_mm": ee.Number(stats.get("total_precipitation_sum")).multiply(1000),
            "soil_water_l1": stats.get("volumetric_soil_water_layer_1"),
        })

    fc = col.map(per_day)
    task = ee.batch.Export.table.toDrive(
        collection=fc, description="era5land_balaghat_daily",
        fileNamePrefix="era5land_balaghat_daily", fileFormat="CSV",
        selectors=["date", "precip_mm", "soil_water_l1"])
    task.start()
    print("Started export 'era5land_balaghat_daily'.")


if __name__ == "__main__":
    main()
