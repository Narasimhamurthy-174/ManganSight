"""Phase 1 check: are the libraries, config and provenance registry OK?
Run:  python scripts/00_check_environment.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, resolve
from src.common.provenance import load_registry, validate_registry

LIBS = ["numpy", "pandas", "yaml", "geopandas", "shapely", "pyproj", "rasterio",
        "sklearn", "xgboost", "shap", "flask", "matplotlib", "plotly"]


def main() -> int:
    bad = 0
    for lib in LIBS:
        try:
            m = importlib.import_module(lib)
            print(f"[OK]      {lib:12s} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[MISSING] {lib:12s} ({exc.__class__.__name__})")
            bad += 1
    cfg = load_config()
    print(f"[OK]      config loaded, projected CRS = {cfg['crs']['projected']}")
    problems = validate_registry(load_registry(resolve(cfg["paths"]["registry"])))
    print("[OK]      dataset registry valid" if not problems else "[WARN] registry problems:")
    for p in problems:
        print("   -", p)
    return 1 if (bad or problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
