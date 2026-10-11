"""Manganese occurrence cleaning.

Rules (from the project data policy):
  * never invent values -- missing fields stay NA
  * every row keeps its source
  * bad coordinates are FLAGGED and excluded from modelling, not silently 'fixed'
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

STANDARD_COLUMNS = [
    "occurrence_id", "latitude", "longitude", "district", "locality", "state",
    "host_rock", "formation", "stratigraphic_position",
    "mineralization_information", "source", "source_date",
]

# MRDS column name (lower-case) -> our standard column
_MRDS_MAP = {
    "dep_id": "occurrence_id", "latitude": "latitude", "longitude": "longitude",
    "county": "district", "site_name": "locality", "state": "state",
    "host_rock": "host_rock", "dep_type": "mineralization_information",
}


def validate_coordinates(df: pd.DataFrame, bounds: dict[str, float],
                         lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """Add `coord_valid` (bool) and `coord_issue` (str) columns. Does not drop rows."""
    out = df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    issue = pd.Series("", index=out.index, dtype="object")
    issue[lat.isna() | lon.isna()] = "missing_or_non_numeric"
    in_world = lat.between(-90, 90) & lon.between(-180, 180)
    issue[(issue == "") & ~in_world] = "impossible_coordinates"
    in_box = lat.between(bounds["lat_min"], bounds["lat_max"]) & \
        lon.between(bounds["lon_min"], bounds["lon_max"])
    issue[(issue == "") & ~in_box] = "outside_study_area_bounds"
    out[lat_col], out[lon_col] = lat, lon
    out["coord_valid"] = issue == ""
    out["coord_issue"] = issue
    return out


def standardize_mrds(df: pd.DataFrame, keywords: Iterable[str] = ("manganese",)) -> pd.DataFrame:
    """Keep manganese records from a USGS MRDS table and map to the standard schema."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    commod_cols = [c for c in d.columns if c.startswith("commod")]
    if not commod_cols:
        raise ValueError("No 'commod*' columns found - is this really an MRDS table? "
                         f"Columns present: {list(d.columns)[:20]}")
    kw = [k.lower() for k in keywords]
    text = d[commod_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    d = d[text.apply(lambda t: any(k in t for k in kw))]
    out = pd.DataFrame(index=d.index)
    for src_col, dst_col in _MRDS_MAP.items():
        out[dst_col] = d[src_col] if src_col in d.columns else pd.NA
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["source"] = "USGS MRDS"
    return out[STANDARD_COLUMNS].reset_index(drop=True)


def clean_occurrences(df: pd.DataFrame, bounds: dict[str, float],
                      decimals: int = 4) -> pd.DataFrame:
    """Validate coordinates and flag near-duplicate points (~11 m at 4 decimals)."""
    if df.empty:
        return df.assign(coord_valid=pd.Series(dtype=bool), coord_issue=pd.Series(dtype=object),
                         is_duplicate=pd.Series(dtype=bool))
    out = validate_coordinates(df, bounds)
    key = out["latitude"].round(decimals).astype(str) + "_" + out["longitude"].round(decimals).astype(str)
    out["is_duplicate"] = key.duplicated(keep="first") & out["coord_valid"]
    return out


def usable_positives(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that may be used as positive training evidence."""
    return df[df["coord_valid"] & ~df["is_duplicate"]].reset_index(drop=True)
