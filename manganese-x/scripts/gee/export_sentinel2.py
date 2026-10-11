"""Export a cloud-masked Sentinel-2 median composite + observation count for Balaghat.

Why median?      robust to residual cloud/shadow.
Why obs count?   becomes a DATA-QUALITY input to the uncertainty system (few clear
                 observations => less trustworthy spectral evidence).
Run: python scripts/gee/export_sentinel2.py
"""
import ee
from _common import init_ee, balaghat_geometry
from src.common.config import load_config


def mask_scl(img):
    """Mask cloud, cloud shadow, cirrus, snow using the Scene Classification Layer.
    SCL: 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus, 11=snow."""
    scl = img.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())


def main() -> None:
    cfg = load_config()["sentinel2"]
    init_ee()
    geom = balaghat_geometry()
    col = (ee.ImageCollection(cfg["collection"])
           .filterBounds(geom)
           .filterDate(cfg["date_start"], cfg["date_end"])
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["max_scene_cloud_pct"]))
           .map(mask_scl))
    print("[info] scenes after filtering:", col.size().getInfo())

    bands = cfg["bands"]
    median = col.select(bands).median().divide(10000)          # reflectance 0-1
    p10 = col.select(["B11"]).reduce(ee.Reducer.percentile([10])).divide(10000).rename("B11_p10")
    nobs = col.select("B4").count().rename("n_valid_obs")
    stack = median.addBands(nobs).addBands(p10).toFloat().clip(geom)

    task = ee.batch.Export.image.toDrive(
        image=stack, description="s2_composite_balaghat", fileNamePrefix="s2_composite_balaghat",
        region=geom, scale=cfg["export_scale_m"], crs="EPSG:32644", maxPixels=1e10)
    task.start()
    print("Started export 's2_composite_balaghat'. Output CRS EPSG:32644. "
          "Band order:", bands + ["n_valid_obs", "B11_p10"])


if __name__ == "__main__":
    main()
