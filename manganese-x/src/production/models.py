"""Production forecasting models and expanding-window time-series CV.

Includes Seasonal Naive, Moving Average, Ridge, Random Forest, and Gradient Boosting.
Provides both operational forecasting (past-only) and conditional scenario forecasting.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard time-series evaluation metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 1e-6 else 0.0
    
    # MAPE only when actuals are comfortably away from zero
    non_zero = y_t > 1e-3
    mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0) if np.any(non_zero) else 0.0
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }


class BaselineSeasonalNaive:
    """Predicts value from the same week in the previous year or seasonal cycle."""
    def __init__(self, cycle_weeks: int = 52):
        self.cycle_weeks = cycle_weeks
        self.history: np.ndarray = np.array([])
        
    def fit(self, y: np.ndarray):
        self.history = np.asarray(y, dtype=float)
        return self
        
    def predict(self, steps: int = 1) -> np.ndarray:
        if len(self.history) == 0:
            return np.zeros(steps)
        if len(self.history) >= self.cycle_weeks:
            return self.history[-self.cycle_weeks : -self.cycle_weeks + steps]
        return np.full(steps, np.mean(self.history[-4:]))


class BaselineMovingAverage:
    """Predicts rolling mean of the recent past."""
    def __init__(self, window: int = 4):
        self.window = window
        self.last_mean: float = 0.0
        
    def fit(self, y: np.ndarray):
        self.last_mean = float(np.mean(y[-self.window:])) if len(y) >= self.window else float(np.mean(y))
        return self
        
    def predict(self, steps: int = 1) -> np.ndarray:
        return np.full(steps, self.last_mean)


def train_production_model(
    model_type: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    seed: int = 42
) -> Any:
    """Fit a forecasting model."""
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)
    elif model_type in ("gradient_boosting", "xgb"):
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model.fit(X_train, y_train)
    return model


def expanding_window_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "actual_production_tonnes",
    min_train_weeks: int = 26,
    step_weeks: int = 4,
    models: list[str] = ["moving_avg", "ridge", "random_forest"]
) -> pd.DataFrame:
    """Run expanding-window cross validation per mine to avoid temporal leakage."""
    results = []
    
    for mine_id, group in df.groupby("mine_id"):
        g = group.sort_values("week_ending").reset_index(drop=True)
        n = len(g)
        
        for split_point in range(min_train_weeks, n - step_weeks, step_weeks):
            train_g = g.iloc[:split_point]
            test_g = g.iloc[split_point : split_point + step_weeks]
            
            y_train = train_g[target_col].to_numpy()
            y_test = test_g[target_col].to_numpy()
            
            # Clean features: fill NaNs with column median from train
            X_train = train_g[feature_cols].copy()
            X_test = test_g[feature_cols].copy()
            train_medians = X_train.median()
            X_train.fillna(train_medians, inplace=True)
            X_test.fillna(train_medians, inplace=True)
            
            for m in models:
                if m == "moving_avg":
                    ma = BaselineMovingAverage(window=4).fit(y_train)
                    y_pred = ma.predict(len(y_test))
                elif m == "seasonal_naive":
                    sn = BaselineSeasonalNaive(cycle_weeks=52).fit(y_train)
                    y_pred = sn.predict(len(y_test))
                else:
                    reg = train_production_model(m, X_train, y_train)
                    y_pred = reg.predict(X_test)
                    
                metrics = evaluate_forecast_metrics(y_test, y_pred)
                results.append({
                    "mine_id": mine_id,
                    "split_week": split_point,
                    "model": m,
                    **metrics
                })
                
    return pd.DataFrame(results)
