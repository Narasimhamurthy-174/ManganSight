"""Tests for Flask REST backend endpoints (Phase 15)."""
import pytest
from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "scientific_disclaimer" in data
    assert data["data_mode"] == "MIXED"


def test_prospectivity_and_map_cells(client):
    res_summary = client.get("/api/prospectivity")
    assert res_summary.status_code == 200
    summary = res_summary.get_json()
    assert "total_cells" in summary
    assert "priority_a_cells" in summary

    res_cells = client.get("/api/prospectivity/map-cells")
    assert res_cells.status_code == 200
    cells_data = res_cells.get_json()
    assert "cells" in cells_data
    assert len(cells_data["cells"]) > 0


def test_zones_and_priority(client):
    res = client.get("/api/zones")
    assert res.status_code == 200
    data = res.get_json()
    assert "known_belt_zones" in data
    assert "stepout_candidate_zones" in data
    assert len(data["stepout_candidate_zones"]) == 17  # 17 stepout zones verified in brief

    res_p = client.get("/api/exploration-priority")
    assert res_p.status_code == 200
    p_data = res_p.get_json()
    assert p_data["stepout_candidates"] > 0


def test_production_endpoints(client):
    # History
    res_hist = client.get("/api/production/history")
    assert res_hist.status_code == 200
    hist = res_hist.get_json()
    assert len(hist["real_monthly"]) > 0
    assert len(hist["simulated_weekly_summary"]) > 0

    # Forecast
    res_fc = client.get("/api/production/forecast?horizon=8")
    assert res_fc.status_code == 200
    fc = res_fc.get_json()
    assert len(fc["forecast"]) == 8
    assert fc["label"] == "model-based scenario estimate"

    # Risk
    res_risk = client.get("/api/production/risk")
    assert res_risk.status_code == 200
    risk = res_risk.get_json()
    assert risk["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "primary_drivers" in risk


def test_scenario_and_recommendations(client):
    # Post scenario
    payload = {
        "scenario_type": "downtime_reduced",
        "parameters": {"reduction_pct": 20.0}
    }
    res_sc = client.post("/api/scenario", json=payload)
    assert res_sc.status_code == 200
    sc_data = res_sc.get_json()
    assert sc_data["scenario_production_tonnes"] > 0
    assert "range_tonnes" in sc_data

    # Recommendations
    res_rec = client.get("/api/recommendations")
    assert res_rec.status_code == 200
    recs = res_rec.get_json()
    assert len(recs["recommendations"]) > 0
    assert "Potential actions for consideration" in recs["framing"]


def test_explainability_and_data_sources(client):
    res_exp = client.get("/api/model/explainability")
    assert res_exp.status_code == 200
    exp = res_exp.get_json()
    assert "exploration_models" in exp
    assert "production_models" in exp

    res_ds = client.get("/api/data-sources")
    assert res_ds.status_code == 200
    ds = res_ds.get_json()
    assert len(ds["registry"]) > 0


def test_frontend_dashboard_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"ManganSight" in res.data
    assert b"Executive Decision Overview" in res.data

    res_css = client.get("/styles.css")
    assert res_css.status_code == 200
    assert b"--bg-primary" in res_css.data

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert b"initNavigation" in res_js.data
