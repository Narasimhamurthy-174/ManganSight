"""ManganSight Flask application entrypoint."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so running directly via `python backend/app.py` works seamlessly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, send_from_directory, jsonify
from backend.routes.api import api_bp

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
    app.config["JSON_SORT_KEYS"] = False
    
    # Register API blueprint
    app.register_blueprint(api_bp)
    
    # Serve single-page dashboard or fallback
    @app.route("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.route("/")
    def index():
        if (FRONTEND_DIST / "index.html").exists():
            return send_from_directory(str(FRONTEND_DIST), "index.html")
        return jsonify({
            "platform": "ManganSight Decision Support System",
            "message": "Backend API is running. Access dashboard at / or API documentation at /api/health.",
            "api_endpoints": [
                "/api/health",
                "/api/prospectivity",
                "/api/prospectivity/<cell_id>",
                "/api/exploration-priority",
                "/api/zones",
                "/api/boundary",
                "/api/production/history",
                "/api/production/forecast",
                "/api/production/risk",
                "/api/scenario",
                "/api/recommendations",
                "/api/model/explainability",
                "/api/data-sources"
            ]
        })
        
    @app.route("/<path:path>")
    def static_files(path: str):
        if (FRONTEND_DIST / path).exists():
            return send_from_directory(str(FRONTEND_DIST), path)
        if (FRONTEND_DIST / "index.html").exists():
            return send_from_directory(str(FRONTEND_DIST), "index.html")
        return jsonify({"error": f"Path /{path} not found"}), 404
        
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
