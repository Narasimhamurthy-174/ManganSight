"""Unit tests for Phase 11: Production data layer, simulator, and past-only features."""
import numpy as np
import pandas as pd
import pytest

from src.production.data_layer import (
    load_production_config,
    calculate_shortfall,
    assign_risk_level,
    load_real_moil_production,
    load_daily_weather,
    aggregate_weather_monthly
)
from src.production.simulator import simulate_mine_operations
from src.production.features import build_past_only_features, verify_past_only_integrity


def test_production_config_loads_valid_fields():
    cfg = load_production_config()
    assert "targets" in cfg
    assert "simulation" in cfg
    assert len(cfg["simulation"]["mines"]) == 4
    mine_names = [m["name"] for m in cfg["simulation"]["mines"]]
    assert "Balaghat Mine" in mine_names
    assert "Tirodi Mine" in mine_names


def test_shortfall_calculation_and_risk_assignment():
    # Actuals vs baseline
    actual = np.array([100.0, 95.0, 85.0, 70.0])
    baseline = np.array([100.0, 100.0, 100.0, 100.0])
    s_amt, s_pct = calculate_shortfall(actual, baseline)
    
    np.testing.assert_allclose(s_amt, [0.0, 5.0, 15.0, 30.0])
    np.testing.assert_allclose(s_pct, [0.0, 5.0, 15.0, 30.0])
    
    risks = assign_risk_level(s_pct, low_thresh=5.0, high_thresh=15.0)
    assert list(risks) == ["LOW", "MEDIUM", "MEDIUM", "HIGH"]


def test_real_moil_monthly_loader():
    df = load_real_moil_production()
    assert not df.empty
    assert "shortfall_amount_lakh_tonnes" in df.columns
    assert "shortfall_pct" in df.columns
    assert "risk_level" in df.columns
    assert (df["data_class"] == "REAL").all()


def test_daily_weather_and_monthly_agg():
    daily = load_daily_weather()
    assert not daily.empty
    assert "rain_sum_7d" in daily.columns
    assert "rain_sum_30d" in daily.columns
    assert "soil_water_mean_30d" in daily.columns
    
    monthly = aggregate_weather_monthly(daily)
    assert not monthly.empty
    assert "total_precip_mm" in monthly.columns
    assert (monthly["data_class"] == "DERIVED").all()


def test_simulate_mine_operations_and_recovery():
    # Synthetic weather for quick run
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(0)
    precip = np.zeros(len(dates))
    # Add monsoon rain in July-August
    monsoon_mask = dates.month.isin([7, 8])
    precip[monsoon_mask] = rng.uniform(10.0, 40.0, size=monsoon_mask.sum())
    soil_water = np.where(monsoon_mask, 0.35, 0.20)
    
    synthetic_weather = pd.DataFrame({
        "date": dates,
        "precip_mm": precip,
        "soil_water_l1": soil_water
    })
    
    sim = simulate_mine_operations(
        weather_daily_df=synthetic_weather,
        seed=42,
        start_year=2024,
        end_year=2024
    )
    assert not sim.empty
    assert set(sim["mine_id"].unique()) == {"balaghat_mine", "tirodi_mine", "ukwa_mine", "sitapatore_mine"}
    assert (sim["data_class"] == "SIMULATED").all()
    
    # Check that monsoon production is lower than pre-monsoon (recovering seasonal dip)
    sim["month"] = sim["week_ending"].dt.month
    pre_monsoon = sim[sim["month"].isin([4, 5])]["actual_production_tonnes"].mean()
    monsoon = sim[sim["month"].isin([7, 8])]["actual_production_tonnes"].mean()
    assert monsoon < pre_monsoon, "Operational simulator failed to reflect seasonal monsoon dip!"


def test_past_only_features_leakage_free():
    # Create simple simulated input
    sim = simulate_mine_operations(seed=42, start_year=2024, end_year=2024)
    feat_df = build_past_only_features(sim)
    
    assert verify_past_only_integrity(feat_df, "actual_production_tonnes", feat_df.attrs["feature_cols"])
    assert "prod_lag_1w" in feat_df.columns
    assert "rain_roll_sum_4w" in feat_df.columns
    assert "is_monsoon" in feat_df.columns


from src.production.models import (
    BaselineSeasonalNaive,
    BaselineMovingAverage,
    train_production_model,
    evaluate_forecast_metrics,
    expanding_window_cv
)
from src.production.risk_engine import evaluate_shortfall_risk, compute_feature_importance_summary


def test_forecast_baselines_and_metrics():
    y_hist = np.array([100.0, 105.0, 98.0, 102.0, 95.0])
    ma = BaselineMovingAverage(window=3).fit(y_hist)
    pred_ma = ma.predict(2)
    assert len(pred_ma) == 2
    assert abs(pred_ma[0] - np.mean([98.0, 102.0, 95.0])) < 1e-4

    sn = BaselineSeasonalNaive(cycle_weeks=4).fit(y_hist)
    pred_sn = sn.predict(1)
    assert len(pred_sn) == 1
    assert abs(pred_sn[0] - 105.0) < 1e-4

    metrics = evaluate_forecast_metrics(np.array([100.0, 110.0]), np.array([98.0, 112.0]))
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics


def test_models_and_cv():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] * 2.0 + rng.normal(scale=0.1, size=50)

    model_rf = train_production_model("random_forest", X, y)
    preds = model_rf.predict(X)
    assert preds.shape == (50,)

    fi = compute_feature_importance_summary(model_rf, list(X.columns))
    assert len(fi) == 4
    assert fi[0]["feature"] == "f0"  # Strongest feature ranked first


def test_shortfall_risk_engine():
    # Low risk case
    res_low = evaluate_shortfall_risk(
        predicted_production=9800.0,
        planned_baseline=10000.0,
        recent_rain_mm=5.0,
        recent_downtime_hrs=8.0
    )
    assert res_low["risk_level"] == "LOW"
    assert res_low["shortfall_pct"] == 2.0

    # High risk with heavy rain and downtime
    res_high = evaluate_shortfall_risk(
        predicted_production=7500.0,
        planned_baseline=10000.0,
        recent_rain_mm=85.0,
        recent_downtime_hrs=22.0
    )
    assert res_high["risk_level"] == "HIGH"
    assert any("Associated with elevated precipitation" in d for d in res_high["primary_drivers"])
    assert any("Associated with unplanned equipment downtime" in d for d in res_high["primary_drivers"])
