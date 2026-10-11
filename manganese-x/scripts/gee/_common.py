"""Shared Earth Engine helpers. UNTESTED in the authoring sandbox (no network / no GEE login).
First-time setup on YOUR machine:
    pip install earthengine-api
    earthengine authenticate
Then put your GEE cloud project id in .env as GEE_PROJECT.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ee
from dotenv import load_dotenv

load_dotenv()


def init_ee() -> None:
    project = os.getenv("GEE_PROJECT")
    if not project:
        raise SystemExit("Set GEE_PROJECT in .env (your Earth Engine cloud project id).")
    ee.Initialize(project=project)


def balaghat_geometry() -> "ee.Geometry":
    """Balaghat district from FAO GAUL 2015 level 2.
    [Verify] Check the printed area (~9,000 km2) and consider replacing with an official boundary."""
    fc = (ee.FeatureCollection("FAO/GAUL/2015/level2")
          .filter(ee.Filter.eq("ADM2_NAME", "Balaghat"))
          .filter(ee.Filter.eq("ADM1_NAME", "Madhya Pradesh")))
    n = fc.size().getInfo()
    if n == 0:
        raise SystemExit("Balaghat not found in GAUL. Check ADM2_NAME spelling in the Code Editor "
                         "or upload your own boundary as an Earth Engine asset.")
    geom = fc.geometry()
    print(f"[info] Balaghat area from GAUL: {geom.area(1).divide(1e6).getInfo():.0f} km2")
    return geom
