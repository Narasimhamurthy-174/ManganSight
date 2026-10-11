"""Phase 4: build the analysis grid from the district boundary.
Needs data/raw/boundary/balaghat_district.geojson (see docs/DATA_SOURCES.md D1).
Run:  python scripts/02_build_grid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.geospatial.grid import build_grid

log = get_logger("build_grid")


def main() -> None:
    cfg = load_config()
    boundary = resolve(cfg["study_area"]["boundary_file"])
    if not boundary.exists():
        raise SystemExit(f"Boundary not found: {boundary}\nSee docs/DATA_SOURCES.md item D1.")
    g = cfg["grid"]
    grid = build_grid(boundary, g["cell_size_m"], cfg["crs"]["projected"], g["version"])
    out = resolve(f"data/processed/{g['version']}.gpkg")
    grid.to_file(out, driver="GPKG")
    log.info("Grid written: %s  (%d cells, %d m)", out, len(grid), g["cell_size_m"])


if __name__ == "__main__":
    main()
