"""Export SRTM elevation for Balaghat (slope/aspect/ruggedness are computed later in Python).
Run: python scripts/gee/export_dem.py
"""
import ee
from _common import init_ee, balaghat_geometry


def main() -> None:
    init_ee()
    geom = balaghat_geometry()
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(geom)
    task = ee.batch.Export.image.toDrive(
        image=dem, description="srtm_balaghat", fileNamePrefix="srtm_balaghat",
        region=geom, scale=30, crs="EPSG:32644", maxPixels=1e10)
    task.start()
    print("Started export 'srtm_balaghat'.")


if __name__ == "__main__":
    main()
