"""What-if scenario simulator for mining operations.

Simulates counterfactual operational conditions (maintenance rescheduling,
downtime reduction, extreme rainfall) and calculates expected production impacts,
shortfall risk, and estimated uncertainty ranges.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from src.production.data_layer import calculate_shortfall, assign_risk_level


def run_scenario(
    baseline_production: float,
    planned_baseline: float,
    scenario_type: str,
    parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute what-if operational scenario simulation.
    
    All outputs labelled 'model-based scenario estimate' with data_class = 'SIMULATED'.
    """
    params = parameters or {}
    b_prod = float(baseline_production)
    b_plan = float(planned_baseline)
    
    delta_tonnes = 0.0
    uncertainty_spread = 0.05  # Default 5% spread
    
    if scenario_type == "current_conditions":
        delta_tonnes = 0.0
        uncertainty_spread = 0.04
        description = "Operations continue under standard operating conditions."
        
    elif scenario_type == "downtime_reduced":
        # Reduction in unplanned downtime (e.g. 15% to 40% reduction)
        pct_reduction = float(params.get("reduction_pct", 25.0)) / 100.0
        # Recovered production: approx 8-15% gain on baseline
        recovered_gain = min(0.20, pct_reduction * 0.12)
        delta_tonnes = b_plan * recovered_gain
        uncertainty_spread = 0.06
        description = f"Unplanned downtime reduced by {pct_reduction*100:.0f}% through proactive reliability and fleet maintenance."
        
    elif scenario_type == "maintenance_rescheduled":
        # Rescheduling scheduled maintenance out of high-vulnerability weeks
        shift_pct = float(params.get("shift_pct", 50.0)) / 100.0
        delta_tonnes = b_plan * (shift_pct * 0.06)
        uncertainty_spread = 0.05
        description = f"Scheduled heavy maintenance shifted from peak monsoon weeks into dry operational windows."
        
    elif scenario_type == "higher_rainfall":
        # Severe rainfall anomaly (+20% to +60% rain or extended inundation)
        rain_increase_pct = float(params.get("rain_increase_pct", 30.0)) / 100.0
        loss_pct = rain_increase_pct * 0.15
        delta_tonnes = -1.0 * (b_prod * loss_pct)
        uncertainty_spread = 0.08
        description = f"Severe rainfall anomaly (+{rain_increase_pct*100:.0f}%) causing increased haul road deterioration and pit water accumulation."
        
    elif scenario_type == "alternative_schedule":
        # Combined scenario: downtime reduced + flexible shift staggering
        delta_tonnes = b_plan * 0.10
        uncertainty_spread = 0.07
        description = "Comprehensive operational pivot: fleet dispatch optimization combined with flexible shift rescheduling."
        
    else:
        raise ValueError(f"Unknown scenario_type: {scenario_type}")
        
    scenario_prod = max(0.0, b_prod + delta_tonnes)
    
    # Uncertainty range (lower and upper estimate)
    lower_bound = max(0.0, scenario_prod * (1.0 - uncertainty_spread))
    upper_bound = scenario_prod * (1.0 + uncertainty_spread)
    
    s_amt, s_pct = calculate_shortfall(scenario_prod, b_plan)
    r_level = assign_risk_level(s_pct)
    
    pct_change = ((scenario_prod - b_prod) / b_prod * 100.0) if b_prod > 0 else 0.0
    
    return {
        "scenario_type": scenario_type,
        "description": description,
        "baseline_production_tonnes": round(b_prod, 1),
        "planned_baseline_tonnes": round(b_plan, 1),
        "scenario_production_tonnes": round(scenario_prod, 1),
        "production_change_tonnes": round(delta_tonnes, 1),
        "production_change_pct": round(pct_change, 2),
        "scenario_shortfall_amount_tonnes": round(float(s_amt), 1),
        "scenario_shortfall_pct": round(float(s_pct), 2),
        "scenario_risk_level": r_level,
        "range_tonnes": {
            "lower_bound": round(lower_bound, 1),
            "upper_bound": round(upper_bound, 1)
        },
        "label": "model-based scenario estimate",
        "data_class": "SIMULATED",
        "data_mode": "SIMULATED"
    }
