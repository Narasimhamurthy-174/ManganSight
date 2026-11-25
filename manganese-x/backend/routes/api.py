"""Flask API blueprint for ManganSight decision support platform."""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from backend.services.data_service import DataService

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "ManganSight Decision Support API",
        "version": "2.0.0",
        "data_mode": "MIXED",
        "scientific_disclaimer": "Decision support only; does not directly detect underground manganese reserves."
    })


@api_bp.route("/prospectivity", methods=["GET"])
def get_prospectivity():
    service = DataService.get_instance()
    return jsonify(service.get_prospectivity_summary())


@api_bp.route("/prospectivity/map-cells", methods=["GET"])
def get_map_cells():
    service = DataService.get_instance()
    return jsonify({
        "count": len(service.map_sample_df),
        "cells": service.get_map_cells(),
        "data_class": "MODEL_OUTPUT",
        "data_mode": "MIXED"
    })


@api_bp.route("/prospectivity/<cell_id>", methods=["GET"])
def get_cell_detail(cell_id: str):
    service = DataService.get_instance()
    cell = service.get_cell_detail(cell_id)
    if cell is None:
        return jsonify({"error": f"Cell {cell_id} not found"}), 404
    return jsonify(cell)


@api_bp.route("/exploration-priority", methods=["GET"])
def get_exploration_priority():
    service = DataService.get_instance()
    summary = service.get_prospectivity_summary()
    return jsonify({
        "priority_a_cells": summary.get("priority_a_cells", 7255),
        "stepout_candidates": summary.get("stepout_candidate_cells", 1905),
        "rule_basis": "Priority A combines model ensemble prospectivity and proximity. Step-out list isolates cells >= 5km from known records with Medium confidence.",
        "data_class": "MODEL_OUTPUT",
        "data_mode": "MIXED"
    })


@api_bp.route("/zones", methods=["GET"])
def get_zones():
    service = DataService.get_instance()
    return jsonify(service.get_zones())


@api_bp.route("/occurrences", methods=["GET"])
def get_occurrences():
    service = DataService.get_instance()
    return jsonify({
        "count": len(service.occurrences),
        "occurrences": service.get_occurrences(),
        "data_class": "REAL",
        "data_mode": "REAL"
    })


@api_bp.route("/boundary", methods=["GET"])
def get_boundary():
    service = DataService.get_instance()
    if service.boundary_geojson:
        return jsonify(service.boundary_geojson)
    return jsonify({"error": "District boundary geojson not found"}), 404


@api_bp.route("/production/history", methods=["GET"])
def get_production_history():
    service = DataService.get_instance()
    return jsonify(service.get_production_history())


@api_bp.route("/production/forecast", methods=["GET"])
def get_production_forecast():
    service = DataService.get_instance()
    horizon = request.args.get("horizon", default=12, type=int)
    return jsonify(service.get_production_forecast(horizon_weeks=horizon))


@api_bp.route("/production/risk", methods=["GET"])
def get_production_risk():
    service = DataService.get_instance()
    return jsonify(service.get_production_risk())


@api_bp.route("/scenario", methods=["POST"])
def post_scenario():
    service = DataService.get_instance()
    data = request.get_json(silent=True) or {}
    scenario_type = data.get("scenario_type", "current_conditions")
    parameters = data.get("parameters", {})
    try:
        result = service.execute_scenario(scenario_type, parameters)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@api_bp.route("/recommendations", methods=["GET"])
def get_recommendations():
    service = DataService.get_instance()
    return jsonify(service.get_recommendations())


@api_bp.route("/model/explainability", methods=["GET"])
def get_model_explainability():
    service = DataService.get_instance()
    return jsonify(service.get_model_explainability())


@api_bp.route("/data-sources", methods=["GET"])
def get_data_sources():
    service = DataService.get_instance()
    return jsonify({
        "registry": service.registry,
        "data_mode": "MIXED"
    })
