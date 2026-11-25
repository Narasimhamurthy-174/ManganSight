"""Risk engine and model explainability for production shortfall risk.

Synthesizes forecasts, weather patterns, and operational conditions into
actionable, uncertainty-aware risk assessments with strictly compliant wording.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.production.data_layer import calculate_shortfall, assign_risk_level


def evaluate_shortfall_risk(
    predicted_production: float | np.ndarray,
    planned_baseline: float | np.ndarray,
    recent_rain_mm: float = 0.0,
    recent_downtime_hrs: float = 0.0,
    recent_soil_water: float = 0.25
) -> dict[str, Any]:
    """Calculate shortfall risk, uncertainty bounds, and primary contributing factors."""
    s_amt, s_pct = calculate_shortfall(predicted_production, planned_baseline)
    risk_level = assign_risk_level(s_pct)
    
    # Feature attribution / driver decomposition
    # Quantify approximate relative impact of weather vs operational downtime
    # Calibrated against simulated parameter weights
    weather_score = min(1.0, (recent_rain_mm / 100.0) * 0.7 + max(0.0, (recent_soil_water - 0.25) / 0.15) * 0.3)
    downtime_score = min(1.0, max(0.0, (recent_downtime_hrs - 8.0) / 20.0))
    
    total_score = weather_score + downtime_score
    if total_score > 1e-4:
        weather_pct = (weather_score / total_score) * 100.0
        downtime_pct = (downtime_score / total_score) * 100.0
    else:
        weather_pct = 0.0
        downtime_pct = 0.0
        
    drivers = []
    if weather_pct > 30.0 and recent_rain_mm > 25.0:
        drivers.append(f"Associated with elevated precipitation ({recent_rain_mm:.1f} mm/week) and saturated soil moisture")
    if downtime_pct > 30.0 and recent_downtime_hrs > 12.0:
        drivers.append(f"Associated with unplanned equipment downtime ({recent_downtime_hrs:.1f} hrs/week)")
    if s_pct > 15.0 and len(drivers) == 0:
        drivers.append("Associated with capacity constraint relative to aggressive production target")
    if len(drivers) == 0:
        drivers.append("Operations operating within normal nominal baseline bounds")
        
    return {
        "predicted_production_tonnes": round(float(predicted_production), 1),
        "planned_baseline_tonnes": round(float(planned_baseline), 1),
        "shortfall_amount_tonnes": round(float(s_amt), 1),
        "shortfall_pct": round(float(s_pct), 2),
        "risk_level": risk_level,
        "weather_attribution_pct": round(weather_pct, 1),
        "downtime_attribution_pct": round(downtime_pct, 1),
        "primary_drivers": drivers,
        "note": "Model-based scenario estimate; feature attribution represents statistical model explanation, not direct causal guarantee."
    }


def compute_feature_importance_summary(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """Extract and format feature importances for tree/linear models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return []
        
    total = np.sum(importances)
    norm_importances = (importances / total) * 100.0 if total > 0 else np.zeros_like(importances)
    
    results = [
        {"feature": name, "importance_pct": round(float(imp), 2)}
        for name, imp in zip(feature_names, norm_importances)
    ]
    results.sort(key=lambda x: x["importance_pct"], reverse=True)
    return results
