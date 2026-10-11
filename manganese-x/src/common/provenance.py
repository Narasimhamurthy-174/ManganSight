"""Dataset provenance registry.

Every dataset used in MANGANESE-X must be listed in data/metadata/dataset_registry.yaml
with the fields below. Unknown values are written as `unknown` -- never invented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = [
    "dataset_id", "dataset_name", "source", "publisher", "url", "download_date",
    "time_period", "spatial_resolution", "temporal_resolution", "crs", "license",
    "processing_steps", "limitations", "data_class", "status",
]
ALLOWED_DATA_CLASS = {"REAL", "SIMULATED", "DERIVED", "MODEL_OUTPUT"}
ALLOWED_STATUS = {"verified_exists", "to_verify", "acquired", "unavailable"}


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("datasets", []))


def validate_registry(entries: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable problems (empty list = registry is valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        label = e.get("dataset_id", f"entry #{i}")
        for field in REQUIRED_FIELDS:
            if field not in e or e[field] in (None, ""):
                problems.append(f"{label}: missing field '{field}' (write 'unknown' if unknown)")
        if e.get("data_class") not in ALLOWED_DATA_CLASS:
            problems.append(f"{label}: data_class must be one of {sorted(ALLOWED_DATA_CLASS)}")
        if e.get("status") not in ALLOWED_STATUS:
            problems.append(f"{label}: status must be one of {sorted(ALLOWED_STATUS)}")
        if label in seen:
            problems.append(f"{label}: duplicate dataset_id")
        seen.add(label)
    return problems
