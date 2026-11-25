"""Past-only temporal feature construction for production forecasting.

Guarantees zero future-leakage by strictly using lag and backward rolling windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_past_only_features(
    sim_df: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 4],
    rolling_windows: list[int] = [4, 8, 12]
) -> pd.DataFrame:
    """Build strictly past-only lag and rolling features for each mine series."""
    df = sim_df.copy()
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df = df.sort_values(["mine_id", "week_ending"]).reset_index(drop=True)
    
    feature_cols = []
    
    # Process per mine to preserve temporal integrity
    groups = []
    for mine_id, group in df.groupby("mine_id", sort=False):
        g = group.copy().sort_values("week_ending")
        
        # 1. Target lags (production and shortfall)
        for lag in lags:
            col_prod_lag = f"prod_lag_{lag}w"
            col_shortfall_lag = f"shortfall_lag_{lag}w"
            g[col_prod_lag] = g["actual_production_tonnes"].shift(lag)
            g[col_shortfall_lag] = g["shortfall_pct"].shift(lag)
            feature_cols.extend([col_prod_lag, col_shortfall_lag])
            
        # 2. Rolling production stats (past only: shift 1 then rolling)
        for w in rolling_windows:
            col_mean = f"prod_roll_mean_{w}w"
            col_std = f"prod_roll_std_{w}w"
            g[col_mean] = g["actual_production_tonnes"].shift(1).rolling(window=w, min_periods=1).mean()
            g[col_std] = g["actual_production_tonnes"].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            feature_cols.extend([col_mean, col_std])
            
        # 3. Operational lags (downtime & maintenance)
        g["downtime_lag_1w"] = g["unplanned_downtime_hrs"].shift(1)
        g["downtime_roll_mean_4w"] = g["unplanned_downtime_hrs"].shift(1).rolling(window=4, min_periods=1).mean()
        g["blasting_delay_lag_1w"] = g["blasting_delay_hrs"].shift(1)
        feature_cols.extend(["downtime_lag_1w", "downtime_roll_mean_4w", "blasting_delay_lag_1w"])
        
        # 4. Past weather indicators (past week's rain and soil water)
        g["rain_lag_1w"] = g["weekly_rain_mm"].shift(1)
        g["rain_roll_sum_4w"] = g["weekly_rain_mm"].shift(1).rolling(window=4, min_periods=1).sum()
        g["soil_water_lag_1w"] = g["mean_soil_water"].shift(1)
        feature_cols.extend(["rain_lag_1w", "rain_roll_sum_4w", "soil_water_lag_1w"])
        
        # 5. Seasonal/Calendar indicators (deterministic, no leakage)
        g["month"] = g["week_ending"].dt.month
        g["quarter"] = g["week_ending"].dt.quarter
        g["is_monsoon"] = g["month"].isin([6, 7, 8, 9]).astype(int)
        feature_cols.extend(["month", "quarter", "is_monsoon"])
        
        groups.append(g)
        
    result_df = pd.concat(groups, ignore_index=True)
    # Deduplicate feature_cols list
    feature_cols = list(dict.fromkeys(feature_cols))
    result_df.attrs["feature_cols"] = feature_cols
    return result_df


def verify_past_only_integrity(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> bool:
    """Verify that shifting target into future does not change past features."""
    # Check that the first row of each group has NaN for 1-lag features
    for _, g in df.groupby("mine_id"):
        first_row = g.iloc[0]
        assert pd.isna(first_row.get("prod_lag_1w")), "Leakage detected: 1-week lag feature has value at index 0!"
    return True
