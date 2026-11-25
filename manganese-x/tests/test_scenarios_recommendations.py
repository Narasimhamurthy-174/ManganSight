"""Tests for Phase 13 (what-if scenarios) and Phase 14 (recommendations)."""
import pytest
from src.scenarios.simulator import run_scenario
from src.recommendations.engine import generate_recommendations


def test_what_if_scenarios():
    baseline = 10000.0
    planned = 10000.0
    
    # 1. Current conditions
    res_curr = run_scenario(baseline, planned, "current_conditions")
    assert res_curr["scenario_production_tonnes"] == baseline
    assert res_curr["label"] == "model-based scenario estimate"
    assert res_curr["data_class"] == "SIMULATED"
    
    # 2. Downtime reduced
    res_dt = run_scenario(baseline, planned, "downtime_reduced", {"reduction_pct": 30.0})
    assert res_dt["scenario_production_tonnes"] > baseline
    assert res_dt["production_change_tonnes"] > 0
    assert res_dt["range_tonnes"]["lower_bound"] < res_dt["scenario_production_tonnes"] < res_dt["range_tonnes"]["upper_bound"]
    
    # 3. Higher rainfall
    res_rain = run_scenario(baseline, planned, "higher_rainfall", {"rain_increase_pct": 40.0})
    assert res_rain["scenario_production_tonnes"] < baseline
    assert res_rain["production_change_tonnes"] < 0
    assert res_rain["scenario_shortfall_amount_tonnes"] > 0
    
    # 4. Invalid scenario raises
    with pytest.raises(ValueError):
        run_scenario(baseline, planned, "unknown_scenario")


def test_recommendation_engine():
    # Production weather risk trigger
    prod_data = {
        "risk_level": "HIGH",
        "weather_attribution_pct": 65.0,
        "downtime_attribution_pct": 35.0,
        "shortfall_pct": 22.0
    }
    recs = generate_recommendations(production_risk_data=prod_data)
    assert len(recs) >= 1
    assert any(r["id"] == "REC_OP_WEATHER_HAULAGE" for r in recs)
    assert all("Potential action for consideration" in r["action"] for r in recs)
    
    # Exploration step-out trigger
    exp_data = {
        "zone_id": "SO-01",
        "confidence_tier": "Medium",
        "is_stepout": True,
        "dist_to_record_km": 12.5,
        "area_km2": 8.4
    }
    recs_exp = generate_recommendations(prospectivity_zone_data=exp_data)
    assert len(recs_exp) >= 1
    assert any(r["id"] == "REC_EXP_STEPOUT_SO-01" for r in recs_exp)
    assert recs_exp[0]["priority"] == "HIGH"
