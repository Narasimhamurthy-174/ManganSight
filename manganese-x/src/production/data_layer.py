"""Production data layer for real MOIL production and ERA5-Land weather integration.

Enforces strict separation of REAL vs DERIVED data and past-only feature construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from src.common.config import PROJECT_ROOT


def load_production_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "production.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_shortfall(actual: float | np.ndarray, baseline: float | np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Calculate shortfall amount and percentage.
    
    shortfall_amount = max(0, baseline - actual)
    shortfall_pct = max(0, (baseline - actual) / baseline) * 100
    """
    actual_arr = np.asarray(actual, dtype=float)
    baseline_arr = np.asarray(baseline, dtype=float)
    is_scalar = actual_arr.ndim == 0 and baseline_arr.ndim == 0
    shortfall_amount = np.maximum(0.0, baseline_arr - actual_arr)
    shortfall_pct = np.where(baseline_arr > 0, (shortfall_amount / baseline_arr) * 100.0, 0.0)
    if is_scalar:
        return float(shortfall_amount), float(shortfall_pct)
    return shortfall_amount, shortfall_pct


def assign_risk_level(shortfall_pct: float | np.ndarray, low_thresh: float = 5.0, high_thresh: float = 15.0) -> np.ndarray | str:
    """Assign risk level: LOW (<5%), MEDIUM (5%-15%), HIGH (>15%)."""
    arr = np.asarray(shortfall_pct, dtype=float)
    is_scalar = arr.ndim == 0
    arr_1d = np.atleast_1d(arr)
    risk = np.full(arr_1d.shape, "LOW", dtype=object)
    risk[(arr_1d >= low_thresh) & (arr_1d <= high_thresh)] = "MEDIUM"
    risk[arr_1d > high_thresh] = "HIGH"
    return str(risk[0]) if is_scalar else risk


def load_real_moil_production(path: str | Path | None = None) -> pd.DataFrame:
    """Load and process real MOIL company-wide monthly production records."""
    cfg = load_production_config()
    csv_path = Path(path) if path else PROJECT_ROOT / cfg["data_sources"]["real_production_csv"]
    df = pd.read_csv(csv_path)
    
    # Filter only monthly rows
    month_df = df[df["period_type"] == "month"].copy()
    month_df["period"] = pd.to_datetime(month_df["period"], format="%Y-%m")
    month_df = month_df.sort_values("period").reset_index(drop=True)
    
    target = cfg["targets"]["assumed_monthly_target_lakh_tonnes"]
    month_df["assumed_target_lakh_tonnes"] = target
    
    shortfall_amt, shortfall_pct = calculate_shortfall(
        month_df["production_lakh_tonnes"].to_numpy(),
        month_df["assumed_target_lakh_tonnes"].to_numpy()
    )
    month_df["shortfall_amount_lakh_tonnes"] = shortfall_amt
    month_df["shortfall_pct"] = shortfall_pct
    month_df["risk_level"] = assign_risk_level(shortfall_pct)
    month_df["data_class"] = "REAL"
    month_df["data_mode"] = "REAL"
    return month_df


def load_daily_weather(path: str | Path | None = None) -> pd.DataFrame:
    """Load daily ERA5-Land weather data and compute past-only rolling aggregates."""
    cfg = load_production_config()
    csv_path = Path(path) if path else PROJECT_ROOT / cfg["data_sources"]["weather_daily_csv"]
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Strict past-only rolling windows (excluding current day when shifted or strictly rolling backward)
    heavy_rain_th = cfg["weather_features"]["heavy_rain_threshold_mm"]
    
    for w in cfg["weather_features"]["rolling_windows_days"]:
        df[f"rain_sum_{w}d"] = df["precip_mm"].rolling(window=w, min_periods=1).sum()
        df[f"soil_water_mean_{w}d"] = df["soil_water_l1"].rolling(window=w, min_periods=1).mean()
        
    df["rainy_days_30d"] = (df["precip_mm"] >= 1.0).rolling(window=30, min_periods=1).sum()
    df["heavy_rain_days_30d"] = (df["precip_mm"] >= heavy_rain_th).rolling(window=30, min_periods=1).sum()
    
    # Historical monthly baseline to compute soil moisture and rain anomalies
    df["month"] = df["date"].dt.month
    monthly_clim = df.groupby("month")[["precip_mm", "soil_water_l1"]].transform("mean")
    df["rain_anomaly_daily"] = df["precip_mm"] - monthly_clim["precip_mm"]
    df["soil_water_anomaly_daily"] = df["soil_water_l1"] - monthly_clim["soil_water_l1"]
    
    df["data_class"] = "REAL"
    return df


def aggregate_weather_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather into monthly totals and indicators for macro analysis."""
    df = daily_df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    
    monthly = df.groupby("year_month").agg(
        total_precip_mm=("precip_mm", "sum"),
        mean_soil_water=("soil_water_l1", "mean"),
        max_daily_precip_mm=("precip_mm", "max"),
        rainy_days=("precip_mm", lambda s: (s >= 1.0).sum()),
        heavy_rain_days=("precip_mm", lambda s: (s >= 25.0).sum())
    ).reset_index()
    monthly["period"] = monthly["year_month"].dt.to_timestamp()
    monthly["data_class"] = "DERIVED"
    return monthly
