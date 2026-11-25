"""Transparent rule-based recommendation engine.

All recommendations are framed as 'potential actions for consideration'
with explicit evidence trails and without guarantees or unsafe operational commands.
"""
from __future__ import annotations

from typing import Any


def generate_recommendations(
    production_risk_data: dict[str, Any] | None = None,
    prospectivity_zone_data: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Synthesize exploration and production evidence into prioritized recommendations."""
    recommendations = []
    
    # 1. Operational Production Recommendations
    if production_risk_data:
        risk_level = production_risk_data.get("risk_level", "LOW")
        weather_attr = production_risk_data.get("weather_attribution_pct", 0.0)
        downtime_attr = production_risk_data.get("downtime_attribution_pct", 0.0)
        shortfall_pct = production_risk_data.get("shortfall_pct", 0.0)
        
        # Rule: High or Medium risk with heavy weather contribution
        if risk_level in ("HIGH", "MEDIUM") and weather_attr >= 35.0:
            recommendations.append({
                "id": "REC_OP_WEATHER_HAULAGE",
                "category": "OPERATIONAL",
                "priority": "HIGH" if risk_level == "HIGH" else "MEDIUM",
                "action": "Potential action for consideration: Review haulage scheduling and pit dewatering readiness ahead of anticipated peak weather periods.",
                "evidence": f"Model indicates {risk_level} shortfall risk ({shortfall_pct:.1f}%) with {weather_attr:.1f}% attribution to precipitation and ground saturation.",
                "evidence_sources": ["ECMWF/ERA5-Land daily", "Operational short-term forecast"]
            })
            
        # Rule: High or Medium risk with equipment downtime dominance
        if risk_level in ("HIGH", "MEDIUM") and downtime_attr >= 35.0:
            recommendations.append({
                "id": "REC_OP_EQUIPMENT_MAINT",
                "category": "OPERATIONAL",
                "priority": "HIGH" if risk_level == "HIGH" else "MEDIUM",
                "action": "Potential action for consideration: Review excavator and haul-truck allocation and accelerate scheduled preventive maintenance cycles.",
                "evidence": f"Model indicates {risk_level} shortfall risk with {downtime_attr:.1f}% attribution to unplanned equipment downtime and mechanical availability.",
                "evidence_sources": ["Mine operational logs", "Simulator equipment reliability curve"]
            })
            
        # Rule: Low risk operations
        if risk_level == "LOW":
            recommendations.append({
                "id": "REC_OP_NOMINAL_STABILITY",
                "category": "OPERATIONAL",
                "priority": "LOW",
                "action": "Potential action for consideration: Maintain current operational parameters while monitoring seasonal weather advisories.",
                "evidence": f"Operations are tracking within nominal baseline limits (shortfall {shortfall_pct:.1f}%).",
                "evidence_sources": ["Operational production monitoring"]
            })
            
    # 2. Exploration Prioritisation Recommendations
    if prospectivity_zone_data:
        confidence = prospectivity_zone_data.get("confidence_tier", "Low")
        is_stepout = prospectivity_zone_data.get("is_stepout", False)
        dist_km = prospectivity_zone_data.get("dist_to_record_km", 0.0)
        area_km2 = prospectivity_zone_data.get("area_km2", 0.0)
        zone_id = prospectivity_zone_data.get("zone_id", "Unknown")
        
        # Rule: Step-out exploration zone (Medium confidence, >5 km from known records)
        if is_stepout and confidence == "Medium":
            recommendations.append({
                "id": f"REC_EXP_STEPOUT_{zone_id}",
                "category": "EXPLORATION",
                "priority": "HIGH",
                "action": f"Potential action for consideration: Prioritise step-out Zone {zone_id} ({area_km2:.1f} km², {dist_km:.1f} km from known deposits) for preliminary non-invasive ground survey and lithological validation.",
                "evidence": "Zone demonstrates high spectral prospectivity in the features-only ensemble under Medium confidence, independently of distance to known mines.",
                "evidence_sources": ["COPERNICUS Sentinel-2", "SRTM DEM", "18-Model Exploration Ensemble"]
            })
            
        # Rule: Near-belt brownfield extension
        elif dist_km < 3.0:
            recommendations.append({
                "id": f"REC_EXP_BROWNFIELD_{zone_id}",
                "category": "EXPLORATION",
                "priority": "MEDIUM",
                "action": f"Potential action for consideration: Evaluate brownfield continuity of Zone {zone_id} in conjunction with active pit mine lease extension planning.",
                "evidence": f"Located within {dist_km:.1f} km of existing manganese occurrences with strong proximity and structural continuity.",
                "evidence_sources": ["USGS MRDS Occurrence Records", "Spatial proximity gradient"]
            })
            
        # Rule: Low confidence zone
        elif confidence == "Low":
            recommendations.append({
                "id": f"REC_EXP_LOW_CONF_{zone_id}",
                "category": "EXPLORATION",
                "priority": "LOW",
                "action": f"Potential action for consideration: Defer active exploration commitments in Zone {zone_id} until regional geological mapping (Sausar Group structure) can be integrated.",
                "evidence": "Low confidence assignment indicates environmental/spectral disparity from verified occurrence baseline.",
                "evidence_sources": ["Applicability domain mask", "Ensemble prediction variance"]
            })
            
    return recommendations
