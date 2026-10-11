"""Export the Balaghat boundary as GeoJSON to Google Drive (small, quick).
Run: python scripts/gee/export_boundary.py   then download from Drive into data/raw/boundary/
"""
import ee
from _common import init_ee, balaghat_geometry


def main() -> None:
    init_ee()
    geom = balaghat_geometry()
    fc = ee.FeatureCollection([ee.Feature(geom, {"name": "Balaghat"})])
    task = ee.batch.Export.table.toDrive(
        collection=fc, description="balaghat_district", fileNamePrefix="balaghat_district",
        fileFormat="GeoJSON")
    task.start()
    print("Started export task 'balaghat_district'. Watch progress at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
