"""Configuration loading. All tunable values live in config/config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Defaults to <project_root>/config/config.yaml."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is empty or malformed: {cfg_path}")
    return cfg


def resolve(rel_path: str | Path) -> Path:
    """Turn a project-relative path from the config into an absolute path."""
    return PROJECT_ROOT / rel_path
