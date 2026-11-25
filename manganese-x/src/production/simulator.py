"""Weekly mine-level operational simulator.

Calibrated to real seasonal dips (MOIL 14.5% monsoon dip in July-August).
Generates synthetic mine operations with complete transparency and labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.production.data_layer import calculate_shortfall, assign_risk_level, load_daily_weather, load_production_config


def simulate_mine_operations(
    weather_daily_df: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
    start_year: int = 2023,
    end_year: int = 2025
) -> pd.DataFrame:
    """Simulate weekly operational logs for the 4 Balaghat district mines.
    
    All rows carry data_class = 'SIMULATED'.
    """
    if cfg is None:
        cfg = load_production_config()
    if weather_daily_df is None:
        weather_daily_df = load_daily_weather()
        
    rng = np.random.default_rng(seed)
    
    # Filter weather to requested window
    w_df = weather_daily_df[
        (weather_daily_df["date"].dt.year >= start_year) &
        (weather_daily_df["date"].dt.year <= end_year)
    ].copy()
    
    # Resample weather to weekly periods
    w_df.set_index("date", inplace=True)
    weekly_weather = w_df.resample("W-MON").agg(
        weekly_rain_mm=("precip_mm", "sum"),
        heavy_rain_days=("precip_mm", lambda s: (s >= 25.0).sum()),
        extreme_rain_days=("precip_mm", lambda s: (s >= 50.0).sum()),
        mean_soil_water=("soil_water_l1", "mean"),
        max_daily_rain=("precip_mm", "max")
    ).reset_index()
    weekly_weather.rename(columns={"date": "week_ending"}, inplace=True)
    
    mines = cfg["simulation"]["mines"]
    records = []
    
    for mine in mines:
        mine_id = mine["id"]
        mine_name = mine["name"]
        nominal_cap = mine["nominal_weekly_capacity_tonnes"]
        weather_sens = mine["weather_sensitivity"]
        base_avail = mine["base_availability_pct"]
        is_opencast = mine["type"] == "open_cast"
        
        for _, row in weekly_weather.iterrows():
            week_date = row["week_ending"]
            rain_mm = row["weekly_rain_mm"]
            heavy_days = row["heavy_rain_days"]
            soil_water = row["mean_soil_water"]
            
            # Weather penalty calibrated to rainfall and soil moisture
            # Open-cast mines suffer more from haul road softening & flooding
            rain_norm = min(1.0, rain_mm / 150.0)
            soil_norm = max(0.0, min(1.0, (soil_water - 0.25) / 0.15))
            weather_impact = (0.7 * rain_norm + 0.3 * soil_norm) * weather_sens * 0.35
            
            # Maintenance hours (scheduled: 8-12 hours)
            maint_hrs = max(4.0, rng.normal(10.0, 1.5))
            
            # Unscheduled downtime (stochastic equipment breakdown + weather-induced stoppages)
            downtime_weather_extra = (heavy_days * 3.5) if is_opencast else (heavy_days * 1.5)
            downtime_hrs = max(2.0, rng.normal(12.0, 3.5) + downtime_weather_extra)
            
            # Blasting delays (blasting halted if daily rain >= 15mm or electrical storms)
            blasting_delay_hrs = heavy_days * (4.0 if is_opencast else 1.0)
            
            # Total hours lost out of 168 hours in a week
            total_lost_hrs = maint_hrs + downtime_hrs + blasting_delay_hrs
            avail_pct = max(0.40, min(0.98, base_avail * (1.0 - (total_lost_hrs / 168.0))))
            
            # Operational production output
            actual_tonnes = nominal_cap * avail_pct * (1.0 - weather_impact)
            # Add small random noise
            actual_tonnes = max(100.0, actual_tonnes * rng.uniform(0.97, 1.03))
            
            # Baseline is nominal planned capacity
            planned_baseline = nominal_cap * base_avail
            
            s_amt, s_pct = calculate_shortfall(actual_tonnes, planned_baseline)
            r_level = assign_risk_level(s_pct)
            
            records.append({
                "week_ending": week_date,
                "mine_id": mine_id,
                "mine_name": mine_name,
                "mine_type": mine["type"],
                "planned_baseline_tonnes": round(planned_baseline, 1),
                "actual_production_tonnes": round(actual_tonnes, 1),
                "shortfall_amount_tonnes": round(s_amt, 1),
                "shortfall_pct": round(s_pct, 2),
                "risk_level": r_level,
                "scheduled_maintenance_hrs": round(maint_hrs, 1),
                "unplanned_downtime_hrs": round(downtime_hrs, 1),
                "blasting_delay_hrs": round(blasting_delay_hrs, 1),
                "effective_availability_pct": round(avail_pct * 100.0, 2),
                "weekly_rain_mm": round(rain_mm, 1),
                "heavy_rain_days": int(heavy_days),
                "mean_soil_water": round(soil_water, 4),
                "weather_impact_pct": round(weather_impact * 100.0, 2),
                "data_class": "SIMULATED",
                "data_mode": "SIMULATED"
            })
            
    sim_df = pd.DataFrame(records)
    sim_df["week_ending"] = pd.to_datetime(sim_df["week_ending"])
    return sim_df.sort_values(["mine_id", "week_ending"]).reset_index(drop=True)
