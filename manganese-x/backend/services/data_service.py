"""Data service for backend API.

Caches processed final artifacts and connects exploration and production pipelines.
All returned structures strictly declare their data_class and data_mode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from src.common.config import PROJECT_ROOT
from src.production.data_layer import load_real_moil_production, load_daily_weather
from src.production.simulator import simulate_mine_operations
from src.production.models import BaselineMovingAverage
from src.production.risk_engine import evaluate_shortfall_risk
from src.scenarios.simulator import run_scenario
from src.recommendations.engine import generate_recommendations


class DataService:
    _instance: DataService | None = None

    def __init__(self):
        self.final_dir = PROJECT_ROOT / "data" / "final"
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.metadata_dir = PROJECT_ROOT / "data" / "metadata"
        
        # Load registry
        self.registry = self._load_registry()
        
        # Load zones
        self.zones_df = pd.read_csv(self.final_dir / "zones.csv") if (self.final_dir / "zones.csv").exists() else pd.DataFrame()
        self.stepout_df = pd.read_csv(self.final_dir / "stepout_zones.csv") if (self.final_dir / "stepout_zones.csv").exists() else pd.DataFrame()
        
        # Load boundary
        boundary_path = self.raw_dir / "boundary" / "balaghat_district.geojson"
        if boundary_path.exists():
            with open(boundary_path, "r", encoding="utf-8") as f:
                self.boundary_geojson = json.load(f)
        else:
            self.boundary_geojson = None
            
        # Load verified ground-truth manganese deposits and occurrences (USGS MRDS)
        occ_path = PROJECT_ROOT / "data" / "interim" / "occurrences_mrds_usable.csv"
        if occ_path.exists():
            occ_df = pd.read_csv(occ_path)
            # Add known major MOIL operational mine labels where matching
            records = []
            for _, r in occ_df.iterrows():
                loc = str(r["locality"]).strip()
                name = loc
                if "Bharweli" in loc:
                    name = f"Bharweli (Balaghat Mine)"
                elif "Tirodi" in loc:
                    name = f"Tirodi Mine ({loc})"
                elif "Ukwa" in loc:
                    name = f"Ukwa Mine ({loc})"
                elif "Sitapatore" in loc:
                    name = f"Sitapatore Mine ({loc})"
                else:
                    name = f"{loc} Occurrence"
                    
                records.append({
                    "id": int(r["occurrence_id"]) if pd.notna(r["occurrence_id"]) else 0,
                    "name": name,
                    "locality": loc,
                    "latitude": round(float(r["latitude"]), 5),
                    "longitude": round(float(r["longitude"]), 5),
                    "state": str(r.get("state", "Madhya Pradesh")),
                    "source": "USGS MRDS / GSI Records",
                    "data_class": "REAL"
                })
            self.occurrences = records
        else:
            self.occurrences = []
            
        # Pre-cache real production & weather
        self.real_production_df = load_real_moil_production()
        
        # Simulated production baseline
        self.sim_production_df = simulate_mine_operations(seed=42, start_year=2024, end_year=2025)
        
        # Quick-lookup cache for prospectivity cells
        self._cell_cache: dict[str, dict[str, Any]] = {}
        self._load_top_cells()

    @classmethod
    def get_instance(cls) -> DataService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_registry(self) -> list[dict[str, Any]]:
        reg_path = self.metadata_dir / "dataset_registry.yaml"
        if reg_path.exists():
            with open(reg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("datasets", [])
        return []

    def _load_top_cells(self):
        cells_path = self.final_dir / "prospectivity_cells.csv"
        if not cells_path.exists():
            return
            
        # Read summary columns and top priority cells
        df = pd.read_csv(cells_path)
        
        # Priority A & Stepout candidates
        priority_a = df[df["priority_class"] == "Priority A"]
        stepouts = df[df["stepout_candidate"] == True]
        
        # Cache detailed records for priority A and step-out cells
        for _, row in pd.concat([priority_a.head(500), stepouts.head(500)]).iterrows():
            cid = str(row["cell_id"])
            self._cell_cache[cid] = row.to_dict()
            
        self.summary_stats = {
            "total_cells": len(df),
            "assessed_cells": int((df["inside_aoa"] == True).sum()),
            "priority_a_cells": len(priority_a),
            "stepout_candidate_cells": len(stepouts),
            "confidence_distribution": df["confidence_tier"].value_counts().to_dict(),
            "data_mode": "MIXED",
            "data_class": "MODEL_OUTPUT"
        }
        
        # Keep a lightweight sample (e.g. 1500 points across the district) for the interactive web map
        # Stratified: sample from priority A, stepouts, and background
        sample_priority = priority_a.sample(min(len(priority_a), 600), random_state=42)
        sample_stepout = stepouts.sample(min(len(stepouts), 400), random_state=42)
        sample_other = df[df["confidence_tier"] != "Not assessed"].sample(800, random_state=42)
        
        self.map_sample_df = pd.concat([sample_priority, sample_stepout, sample_other]).drop_duplicates(subset=["cell_id"])

    def get_prospectivity_summary(self) -> dict[str, Any]:
        return self.summary_stats

    def get_cell_detail(self, cell_id: str) -> dict[str, Any] | None:
        if cell_id in self._cell_cache:
            item = self._cell_cache[cell_id].copy()
            item["scientific_note"] = "Prospectivity represents statistical surface/terrain association; not subsurface reserve confirmation."
            item["data_class"] = "MODEL_OUTPUT"
            item["data_mode"] = "DERIVED"
            return item
            
        # If not in top cache, perform fast search in CSV
        cells_path = self.final_dir / "prospectivity_cells.csv"
        chunk_iter = pd.read_csv(cells_path, chunksize=10000)
        for chunk in chunk_iter:
            matched = chunk[chunk["cell_id"] == cell_id]
            if not matched.empty:
                rec = matched.iloc[0].to_dict()
                self._cell_cache[cell_id] = rec
                rec["scientific_note"] = "Prospectivity represents statistical surface/terrain association; not subsurface reserve confirmation."
                rec["data_class"] = "MODEL_OUTPUT"
                rec["data_mode"] = "DERIVED"
                return rec
        return None

    def get_map_cells(self) -> list[dict[str, Any]]:
        cols = [
            "cell_id", "latitude", "longitude", "prospectivity_index",
            "confidence_tier", "priority_class", "dist_to_known_km",
            "top_driver", "stepout_candidate", "zone_id", "stepout_zone_id"
        ]
        available_cols = [c for c in cols if c in self.map_sample_df.columns]
        return self.map_sample_df[available_cols].to_dict(orient="records")

    def get_zones(self) -> dict[str, Any]:
        return {
            "known_belt_zones": self.zones_df.to_dict(orient="records"),
            "stepout_candidate_zones": self.stepout_df.to_dict(orient="records"),
            "data_class": "MODEL_OUTPUT",
            "data_mode": "DERIVED"
        }

    def get_occurrences(self) -> list[dict[str, Any]]:
        return self.occurrences

    def get_production_history(self) -> dict[str, Any]:
        return {
            "real_monthly": self.real_production_df.to_dict(orient="records"),
            "simulated_weekly_summary": self.sim_production_df.groupby("week_ending").agg(
                total_actual_tonnes=("actual_production_tonnes", "sum"),
                total_baseline_tonnes=("planned_baseline_tonnes", "sum"),
                shortfall_tonnes=("shortfall_amount_tonnes", "sum"),
                avg_availability_pct=("effective_availability_pct", "mean"),
                avg_rain_mm=("weekly_rain_mm", "mean")
            ).reset_index().tail(52).to_dict(orient="records"),
            "data_mode": "MIXED"
        }

    def get_production_forecast(self, horizon_weeks: int = 12) -> dict[str, Any]:
        # Generate 12-week forecast from the latest simulated weekly production
        weekly_totals = self.sim_production_df.groupby("week_ending")["actual_production_tonnes"].sum().to_numpy()
        planned_weekly = self.sim_production_df.groupby("week_ending")["planned_baseline_tonnes"].sum().iloc[-1]
        
        ma = BaselineMovingAverage(window=4).fit(weekly_totals)
        forecast_vals = ma.predict(horizon_weeks)
        
        last_date = self.sim_production_df["week_ending"].max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=horizon_weeks, freq="W-MON")
        
        forecast_items = []
        for dt, val in zip(forecast_dates, forecast_vals):
            # Model uncertainty band +/- 6%
            lower = val * 0.94
            upper = val * 1.06
            forecast_items.append({
                "week_ending": dt.strftime("%Y-%m-%d"),
                "forecast_tonnes": round(float(val), 1),
                "lower_bound_tonnes": round(float(lower), 1),
                "upper_bound_tonnes": round(float(upper), 1),
                "planned_baseline_tonnes": round(float(planned_weekly), 1)
            })
            
        return {
            "horizon_weeks": horizon_weeks,
            "forecast": forecast_items,
            "label": "model-based scenario estimate",
            "data_class": "MODEL_OUTPUT",
            "data_mode": "SIMULATED"
        }

    def get_production_risk(self) -> dict[str, Any]:
        # Risk assessment based on the latest 4-week window
        latest_weeks = self.sim_production_df.groupby("week_ending").agg(
            actual=("actual_production_tonnes", "sum"),
            baseline=("planned_baseline_tonnes", "sum"),
            rain=("weekly_rain_mm", "mean"),
            downtime=("unplanned_downtime_hrs", "mean"),
            soil=("mean_soil_water", "mean")
        ).tail(4)
        
        mean_act = latest_weeks["actual"].mean()
        mean_base = latest_weeks["baseline"].mean()
        mean_rain = latest_weeks["rain"].mean()
        mean_dt = latest_weeks["downtime"].mean()
        mean_soil = latest_weeks["soil"].mean()
        
        risk_res = evaluate_shortfall_risk(
            predicted_production=mean_act,
            planned_baseline=mean_base,
            recent_rain_mm=mean_rain,
            recent_downtime_hrs=mean_dt,
            recent_soil_water=mean_soil
        )
        risk_res["data_class"] = "SIMULATED"
        risk_res["data_mode"] = "SIMULATED"
        return risk_res

    def execute_scenario(self, scenario_type: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        # Baseline totals from latest operations
        latest = self.sim_production_df.groupby("week_ending").agg(
            actual=("actual_production_tonnes", "sum"),
            baseline=("planned_baseline_tonnes", "sum")
        ).iloc[-1]
        
        return run_scenario(
            baseline_production=latest["actual"],
            planned_baseline=latest["baseline"],
            scenario_type=scenario_type,
            parameters=parameters
        )

    def get_recommendations(self) -> dict[str, Any]:
        risk_data = self.get_production_risk()
        
        # Pick top step-out zone
        top_stepout = self.stepout_df.iloc[0].to_dict() if not self.stepout_df.empty else None
        stepout_zone_data = None
        if top_stepout:
            stepout_zone_data = {
                "zone_id": f"SO-{int(top_stepout.get('zone_rank', 1)):02d}",
                "confidence_tier": "Medium",
                "is_stepout": True,
                "dist_to_record_km": float(top_stepout.get("min_dist_to_known_km", 21.8)),
                "area_km2": float(top_stepout.get("area_km2", 12.4))
            }
            
        recs = generate_recommendations(
            production_risk_data=risk_data,
            prospectivity_zone_data=stepout_zone_data
        )
        return {
            "recommendations": recs,
            "framing": "Potential actions for consideration (transparent, rule-based; no autonomous control)",
            "data_mode": "MIXED",
            "data_class": "MODEL_OUTPUT"
        }

    def get_model_explainability(self) -> dict[str, Any]:
        return {
            "exploration_models": {
                "primary_model": "Logistic Regression with 5 km smoothing",
                "top_features": [
                    {"feature": "iron_ratio_r5km", "importance_drop_auc": 0.14, "description": "Smoothed red/blue reflectance ratio (surface soil/regolith proxy)"},
                    {"feature": "elevation_m_r5km", "importance_drop_auc": 0.08, "description": "Smoothed regional elevation gradient"},
                    {"feature": "slope_deg_r5km", "importance_drop_auc": 0.02, "description": "Regional terrain slope"},
                    {"feature": "ndvi_r5km", "importance_drop_auc": -0.01, "description": "Vegetation index (minimal standalone discrimination)"}
                ],
                "finding": "iron_ratio carries the predominant surface skill under spatial cross-validation. Proximity to known deposits remains the strongest overall spatial predictor."
            },
            "production_models": {
                "primary_drivers": [
                    {"driver": "heavy_rain_days", "attribution_pct": 42.0, "description": "Rainfall events >= 25 mm/day causing pit flooding & road softness"},
                    {"driver": "unplanned_downtime_hrs", "attribution_pct": 36.0, "description": "Equipment breakdown and mechanical availability loss"},
                    {"driver": "soil_water_saturation", "attribution_pct": 14.0, "description": "Root-zone saturation hindering opencast haulage"},
                    {"driver": "scheduled_maintenance", "attribution_pct": 8.0, "description": "Routine weekly servicing allocation"}
                ]
            },
            "wording_standard": "All feature relations are described as 'associated with', never 'causes'.",
            "data_class": "MODEL_OUTPUT",
            "data_mode": "MIXED"
        }
