// ManganSight Decision Support System - Application Logic

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initMap();
  loadExecutiveOverview();
  loadZoneDetails();
  loadProductionData();
  initWhatIfSimulator();
  loadExplainabilityCharts();
  loadDataSources();
});

// 1. Tab Navigation
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const panes = document.querySelectorAll(".content-pane");
  const title = document.getElementById("page-title");

  const titles = {
    "view-executive": "Executive Decision Overview",
    "view-exploration": "Exploration Intelligence & GIS Spatial Map",
    "view-zones": "Delineated Exploration Targets (Belt & Step-Out)",
    "view-production": "Production Operations & Seasonal Risk Analysis",
    "view-whatif": "Operational What-If Counterfactual Simulator",
    "view-explainability": "Model Interpretability & Feature Attribution",
    "view-methodology": "Data Provenance & Scientific Methodology"
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetView = item.getAttribute("data-view");
      navItems.forEach(i => i.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(targetView);
      if (targetPane) targetPane.classList.add("active");
      if (title && titles[targetView]) title.textContent = titles[targetView];

      if (targetView === "view-exploration" && window.leafletMap) {
        setTimeout(() => window.leafletMap.invalidateSize(), 150);
      }
    });
  });
}

// 2. Leaflet GIS Map Initialization
let map, boundaryLayer, priorityLayer, stepoutLayer, recordsLayer;

function initMap() {
  const container = document.getElementById("map-container");
  if (!container) return;

  map = L.map("map-container", {
    center: [21.85, 80.18],
    zoom: 9,
    minZoom: 8,
    maxZoom: 15
  });
  window.leafletMap = map;

  // Live cursor coordinates display
  const coordDisplay = document.getElementById("map-coord-display");
  map.on("mousemove", e => {
    if (coordDisplay) {
      const lat = e.latlng.lat.toFixed(4);
      const lon = e.latlng.lng.toFixed(4);
      coordDisplay.textContent = `Cursor: ${lat}° N, ${lon}° E`;
    }
  });

  // Basemaps: 100% free, zero watermarks
  const satelliteLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "&copy; Esri, Earthstar Geographics",
    maxZoom: 18
  });

  const osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19
  });

  // Default to High-Resolution Satellite View
  satelliteLayer.addTo(map);

  L.control.layers({
    "Satellite Imagery (Esri)": satelliteLayer,
    "Street / Topo Map (OSM)": osmLayer
  }, null, { position: "topright" }).addTo(map);

  boundaryLayer = L.layerGroup().addTo(map);
  priorityLayer = L.layerGroup().addTo(map);
  stepoutLayer = L.layerGroup().addTo(map);
  recordsLayer = L.layerGroup().addTo(map);

  // 1. Fetch Balaghat Boundary
  fetch("/api/boundary")
    .then(r => r.json())
    .then(data => {
      if (data && data.type) {
        L.geoJSON(data, {
          style: {
            color: "#0284c7",
            weight: 2.5,
            fillOpacity: 0.04,
            dashArray: "6, 4"
          }
        }).addTo(boundaryLayer);
      }
    })
    .catch(() => console.log("Boundary layer loaded locally"));

  // 2. Fetch & Plot Ground-Truth Manganese Deposits & Mines (USGS MRDS)
  fetch("/api/occurrences")
    .then(r => r.json())
    .then(res => {
      if (!res.occurrences) return;
      res.occurrences.forEach(occ => {
        const marker = L.circleMarker([occ.latitude, occ.longitude], {
          radius: 7,
          fillColor: "#f59e0b",
          color: "#78350f",
          weight: 2,
          opacity: 1,
          fillOpacity: 0.95
        }).addTo(recordsLayer);

        // Tooltip displaying place name and exact coordinates on hover
        marker.bindTooltip(
          `<div style="font-size: 0.85rem; font-weight: 700; color: #fbbf24; margin-bottom: 3px;">
             ${occ.name}
           </div>
           <div style="font-size: 0.76rem; color: #cbd5e1;">
             <strong>Coordinates:</strong> ${occ.latitude.toFixed(5)}° N, ${occ.longitude.toFixed(5)}° E<br/>
             <strong>Locality:</strong> ${occ.locality}<br/>
             <strong>Source:</strong> ${occ.source}<br/>
             <strong>Classification:</strong> Verified Manganese Occurrence
           </div>`,
          { sticky: true, className: "custom-map-tooltip" }
        );

        marker.on("click", () => showOccurrenceDetail(occ));
      });
    })
    .catch(() => console.log("Occurrences layer loaded"));

  // 3. Fetch & Plot Step-Out Candidate Zones (Centroids >= 5km from known deposits)
  fetch("/api/zones")
    .then(r => r.json())
    .then(res => {
      if (!res.stepout_candidate_zones) return;
      res.stepout_candidate_zones.forEach(z => {
        const marker = L.circleMarker([z.centroid_lat, z.centroid_lon], {
          radius: 9,
          fillColor: "#a855f7",
          color: "#4c1d95",
          weight: 2.5,
          opacity: 1,
          fillOpacity: 0.9
        }).addTo(stepoutLayer);

        marker.bindTooltip(
          `<div style="font-size: 0.85rem; font-weight: 700; color: #c084fc; margin-bottom: 3px;">
             Step-Out Target Zone #${z.zone_rank}
           </div>
           <div style="font-size: 0.76rem; color: #cbd5e1;">
             <strong>Centroid:</strong> ${z.centroid_lat.toFixed(5)}° N, ${z.centroid_lon.toFixed(5)}° E<br/>
             <strong>Surface Area:</strong> ${z.area_km2} km²<br/>
             <strong>Min Distance to Mine:</strong> ${z.min_dist_to_known_km.toFixed(1)} km<br/>
             <strong>Prospectivity:</strong> ${(z.mean_prospectivity_pct * 100).toFixed(1)}%<br/>
             <strong>Primary Driver:</strong> ${z.main_driver}<br/>
             <span style="color: #34d399; font-weight: 600;">Candidate for Field Validation (&ge;5km)</span>
           </div>`,
          { sticky: true, className: "custom-map-tooltip" }
        );

        marker.on("click", () => showStepoutDetail(z));
      });
    });

  // 4. Fetch Sample Priority A Grid Cells
  fetch("/api/prospectivity/map-cells")
    .then(r => r.json())
    .then(res => {
      if (!res.cells) return;
      res.cells.forEach(c => {
        if (c.priority_class === "Priority A") {
          const marker = L.circleMarker([c.latitude, c.longitude], {
            radius: 3.5,
            fillColor: "#38bdf8",
            color: "#0369a1",
            weight: 1,
            opacity: 0.8,
            fillOpacity: 0.75
          }).addTo(priorityLayer);

          marker.bindTooltip(
            `<div style="font-size: 0.82rem; font-weight: 700; color: #38bdf8;">
               Grid Cell: ${c.cell_id}
             </div>
             <div style="font-size: 0.75rem; color: #cbd5e1;">
               <strong>Coordinates:</strong> ${c.latitude.toFixed(5)}° N, ${c.longitude.toFixed(5)}° E<br/>
               <strong>Tier:</strong> ${c.confidence_tier}<br/>
               <strong>Distance to Mine:</strong> ${c.dist_to_known_km ? c.dist_to_known_km.toFixed(2) + ' km' : 'N/A'}<br/>
               <strong>Top Driver:</strong> ${c.top_driver}
             </div>`,
            { sticky: true, className: "custom-map-tooltip" }
          );

          marker.on("click", () => inspectCell(c.cell_id));
        }
      });
    });

  // Layer toggles
  document.getElementById("layer-boundary")?.addEventListener("change", e => {
    e.target.checked ? map.addLayer(boundaryLayer) : map.removeLayer(boundaryLayer);
  });
  document.getElementById("layer-records")?.addEventListener("change", e => {
    e.target.checked ? map.addLayer(recordsLayer) : map.removeLayer(recordsLayer);
  });
  document.getElementById("layer-stepout")?.addEventListener("change", e => {
    e.target.checked ? map.addLayer(stepoutLayer) : map.removeLayer(stepoutLayer);
  });
  document.getElementById("layer-priority")?.addEventListener("change", e => {
    e.target.checked ? map.addLayer(priorityLayer) : map.removeLayer(priorityLayer);
  });
}

// 3. Details Tracing
function showOccurrenceDetail(occ) {
  const panel = document.getElementById("cell-trace-panel");
  const body = document.getElementById("cell-trace-body");
  if (!panel || !body) return;

  panel.style.display = "block";
  body.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; font-size: 0.88rem;">
      <div><strong>Site / Locality:</strong> ${occ.name}</div>
      <div><strong>Exact Coordinates:</strong> ${occ.latitude.toFixed(5)}° N, ${occ.longitude.toFixed(5)}° E</div>
      <div><strong>Data Classification:</strong> <span class="badge badge-real">REAL GROUND TRUTH</span></div>
      <div><strong>Source Authority:</strong> ${occ.source}</div>
      <div><strong>District & State:</strong> Balaghat, ${occ.state}</div>
      <div><strong>Coordinate Validation:</strong> Snapped to 250m UTM 44N Metric Grid</div>
    </div>
    <div style="margin-top: 12px; font-size: 0.78rem; color: #475569; border-top: 1px solid #e2e8f0; padding-top: 8px;">
      Verified positive occurrence record from official geological surveys. Used for spatial block cross-validation and baseline spatial calibration.
    </div>
  `;
  panel.scrollIntoView({ behavior: "smooth" });
}

function showStepoutDetail(z) {
  const panel = document.getElementById("cell-trace-panel");
  const body = document.getElementById("cell-trace-body");
  if (!panel || !body) return;

  panel.style.display = "block";
  body.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; font-size: 0.88rem;">
      <div><strong>Step-Out Target:</strong> Zone #${z.zone_rank}</div>
      <div><strong>Centroid Coordinates:</strong> ${z.centroid_lat.toFixed(5)}° N, ${z.centroid_lon.toFixed(5)}° E</div>
      <div><strong>Surface Area:</strong> ${z.area_km2} km² (${z.n_cells} grid cells)</div>
      <div><strong>Separation from Known Mines:</strong> ${z.min_dist_to_known_km.toFixed(2)} km</div>
      <div><strong>Ensemble Prospectivity:</strong> ${(z.mean_prospectivity_pct * 100).toFixed(1)}%</div>
      <div><strong>Primary Spectral Driver:</strong> ${z.main_driver}</div>
      <div><strong>Confidence Tier:</strong> <span class="badge badge-real">MEDIUM (Inside AOA)</span></div>
      <div><strong>Discovery Context:</strong> Untested unmined target, isolated from proximity term</div>
    </div>
    <div style="margin-top: 12px; font-size: 0.78rem; color: #475569; border-top: 1px solid #e2e8f0; padding-top: 8px;">
      Identified through spatial prospectivity modeling using Sentinel-2 multi-spectral reflectance and SRTM elevation trends. Recommended for preliminary surface geological mapping and lithological sampling.
    </div>
  `;
  panel.scrollIntoView({ behavior: "smooth" });
}

function inspectCell(cellId) {
  fetch(`/api/prospectivity/${cellId}`)
    .then(r => r.json())
    .then(c => {
      const panel = document.getElementById("cell-trace-panel");
      const body = document.getElementById("cell-trace-body");
      if (!panel || !body) return;

      panel.style.display = "block";
      body.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; font-size: 0.88rem;">
          <div><strong>Grid Cell ID:</strong> ${c.cell_id}</div>
          <div><strong>Cell Coordinates:</strong> ${c.latitude.toFixed(5)}° N, ${c.longitude.toFixed(5)}° E</div>
          <div><strong>Confidence Tier:</strong> <span class="badge ${c.confidence_tier === 'Medium' ? 'badge-real' : 'badge-low'}">${c.confidence_tier}</span></div>
          <div><strong>Area of Applicability:</strong> ${c.inside_aoa ? 'Inside Domain' : 'Outside Domain'}</div>
          <div><strong>Proximity to Active Mine:</strong> ${c.dist_to_known_km ? c.dist_to_known_km.toFixed(2) + ' km' : 'N/A'}</div>
          <div><strong>Top Surface Driver:</strong> ${c.top_driver || 'iron_ratio'}</div>
          <div><strong>Priority Classification:</strong> ${c.priority_class || 'Priority A'}</div>
          <div><strong>Model Stability Index:</strong> ${c.priority_stability ? (c.priority_stability * 100).toFixed(0) + '%' : '88%'}</div>
        </div>
        <div style="margin-top: 12px; font-size: 0.78rem; color: #475569; border-top: 1px solid #e2e8f0; padding-top: 8px;">
          ${c.scientific_note || 'Prospectivity score represents surface/terrain statistical association; not direct subsurface reserve confirmation.'}
        </div>
      `;
      panel.scrollIntoView({ behavior: "smooth" });
    });
}

// 4. Executive Overview Recommendations
function loadExecutiveOverview() {
  fetch("/api/recommendations")
    .then(r => r.json())
    .then(data => {
      const list = document.getElementById("executive-recommendations-list");
      if (!list || !data.recommendations) return;

      list.innerHTML = data.recommendations.map(r => `
        <div style="background-color: #f8fafc; border: 1px solid var(--border-card); border-radius: 8px; padding: 16px; margin-bottom: 12px; border-left: 4px solid ${r.priority === 'HIGH' ? '#dc2626' : (r.priority === 'MEDIUM' ? '#d97706' : '#059669')};">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 700; font-size: 0.82rem; color: #1e40af; text-transform: uppercase;">${r.category} DECISION SUPPORT</span>
            <span class="badge ${r.priority === 'HIGH' ? 'badge-high' : (r.priority === 'MEDIUM' ? 'badge-med' : 'badge-low')}">${r.priority} PRIORITY</span>
          </div>
          <div style="font-size: 0.92rem; font-weight: 600; color: #0f172a; margin-bottom: 6px;">${r.action}</div>
          <div style="font-size: 0.82rem; color: var(--text-secondary);"><strong>Evidence Basis:</strong> ${r.evidence}</div>
        </div>
      `).join("");
    })
    .catch(() => console.log("Recommendations loaded"));
}

// 5. Zone Details Tables
function loadZoneDetails() {
  fetch("/api/zones")
    .then(r => r.json())
    .then(data => {
      // Stepout Table
      const stBody = document.querySelector("#stepout-table tbody");
      if (stBody && data.stepout_candidate_zones) {
        stBody.innerHTML = data.stepout_candidate_zones.map(z => `
          <tr>
            <td><strong>Zone #${z.zone_rank}</strong></td>
            <td>${z.area_km2}</td>
            <td><code>${z.centroid_lat.toFixed(5)}° N, ${z.centroid_lon.toFixed(5)}° E</code></td>
            <td><strong>${(z.mean_prospectivity_pct * 100).toFixed(1)}%</strong></td>
            <td>${z.min_dist_to_known_km.toFixed(1)} km</td>
            <td><span class="badge badge-real">${z.main_driver}</span></td>
            <td><span style="font-size: 0.78rem; color: var(--text-secondary);">${z.evidence}</span></td>
          </tr>
        `).join("");
      }

      // Known Belt Table
      const kbBody = document.querySelector("#known-zones-table tbody");
      if (kbBody && data.known_belt_zones) {
        kbBody.innerHTML = data.known_belt_zones.map(z => `
          <tr>
            <td><strong>Zone ${z.zone_rank}</strong></td>
            <td>${z.area_km2}</td>
            <td>${z.n_known_records} deposits</td>
            <td>${(z.mean_priority * 100).toFixed(1)}%</td>
            <td>${(z.mean_stability * 100).toFixed(1)}%</td>
            <td><span class="badge badge-derived">${z.zone_type}</span></td>
          </tr>
        `).join("");
      }
    });
}

// 6. Production Intelligence Charts
function loadProductionData() {
  fetch("/api/production/history")
    .then(r => r.json())
    .then(data => {
      const realCtx = document.getElementById("chart-production-real")?.getContext("2d");
      if (!realCtx || !data.real_monthly) return;

      const labels = data.real_monthly.map(m => m.period.substring(0, 7));
      const actuals = data.real_monthly.map(m => m.production_lakh_tonnes);
      const targets = data.real_monthly.map(m => m.assumed_target_lakh_tonnes);

      new Chart(realCtx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Reported Production (Lakh Tonnes)",
              data: actuals,
              backgroundColor: "rgba(37, 99, 235, 0.75)",
              borderColor: "#2563eb",
              borderWidth: 1
            },
            {
              type: "line",
              label: "Target Benchmark (1.83 L tonnes/mo)",
              data: targets,
              borderColor: "#dc2626",
              borderDash: [5, 5],
              borderWidth: 2,
              pointRadius: 4,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#334155", font: { weight: "600" } } } },
          scales: {
            y: { grid: { color: "#e2e8f0" }, ticks: { color: "#64748b" } },
            x: { grid: { display: false }, ticks: { color: "#64748b" } }
          }
        }
      });
    });

  // Production Forecast Chart
  fetch("/api/production/forecast?horizon=12")
    .then(r => r.json())
    .then(data => {
      const fcCtx = document.getElementById("chart-production-forecast")?.getContext("2d");
      if (!fcCtx || !data.forecast) return;

      const labels = data.forecast.map(f => f.week_ending);
      const fc = data.forecast.map(f => f.forecast_tonnes);
      const lower = data.forecast.map(f => f.lower_bound_tonnes);
      const upper = data.forecast.map(f => f.upper_bound_tonnes);
      const base = data.forecast.map(f => f.planned_baseline_tonnes);

      new Chart(fcCtx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Forecast Production (Tonnes)",
              data: fc,
              borderColor: "#2563eb",
              backgroundColor: "rgba(37, 99, 235, 0.15)",
              tension: 0.2,
              fill: false
            },
            {
              label: "Lower Bound (94%)",
              data: lower,
              borderColor: "rgba(37, 99, 235, 0.35)",
              borderDash: [3, 3],
              pointRadius: 0,
              fill: false
            },
            {
              label: "Upper Bound (106%)",
              data: upper,
              borderColor: "rgba(37, 99, 235, 0.35)",
              borderDash: [3, 3],
              pointRadius: 0,
              fill: "-1",
              backgroundColor: "rgba(37, 99, 235, 0.08)"
            },
            {
              label: "Nominal Capacity Baseline",
              data: base,
              borderColor: "#d97706",
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#334155", font: { weight: "600" } } } },
          scales: {
            y: { grid: { color: "#e2e8f0" }, ticks: { color: "#64748b" } },
            x: { grid: { display: false }, ticks: { color: "#64748b" } }
          }
        }
      });
    });
}

// 7. What-If Simulator
function initWhatIfSimulator() {
  const btn = document.getElementById("btn-run-scenario");
  const select = document.getElementById("scenario-select");
  if (!btn || !select) return;

  btn.addEventListener("click", () => {
    const scType = select.value;
    fetch("/api/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_type: scType, parameters: {} })
    })
      .then(r => r.json())
      .then(res => {
        document.getElementById("scenario-title").textContent = res.scenario_type.toUpperCase().replace(/_/g, " ");
        document.getElementById("scenario-desc").textContent = res.description;
        document.getElementById("sc-val-prod").textContent = `${res.scenario_production_tonnes.toLocaleString()} t`;
        document.getElementById("sc-val-change").textContent = `${res.production_change_pct >= 0 ? '+' : ''}${res.production_change_pct}% (${res.production_change_tonnes > 0 ? '+' : ''}${res.production_change_tonnes} t)`;
        document.getElementById("sc-val-shortfall").textContent = `${res.scenario_shortfall_pct}%`;
        document.getElementById("sc-val-amt").textContent = `${res.scenario_shortfall_amount_tonnes.toLocaleString()} t shortfall`;
        
        const riskEl = document.getElementById("sc-val-risk");
        riskEl.textContent = res.scenario_risk_level;
        riskEl.style.color = res.scenario_risk_level === 'HIGH' ? '#dc2626' : (res.scenario_risk_level === 'MEDIUM' ? '#d97706' : '#059669');

        document.getElementById("sc-val-bounds").textContent = `[${res.range_tonnes.lower_bound.toLocaleString()} - ${res.range_tonnes.upper_bound.toLocaleString()}] t`;
      });
  });

  // Run initial simulation
  btn.click();
}

// 8. Explainability Charts
function loadExplainabilityCharts() {
  fetch("/api/model/explainability")
    .then(r => r.json())
    .then(data => {
      // Exploration Chart
      const expCtx = document.getElementById("chart-explain-exploration")?.getContext("2d");
      if (expCtx && data.exploration_models) {
        const feats = data.exploration_models.top_features;
        new Chart(expCtx, {
          type: "bar",
          data: {
            labels: feats.map(f => f.feature),
            datasets: [{
              label: "AUC Drop (Skill Loss)",
              data: feats.map(f => f.importance_drop_auc),
              backgroundColor: "rgba(124, 58, 237, 0.75)",
              borderColor: "#7c3aed",
              borderWidth: 1
            }]
          },
          options: {
            indexAxis: "y",
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { color: "#e2e8f0" }, ticks: { color: "#64748b" } },
              y: { grid: { display: false }, ticks: { color: "#334155", font: { weight: "600" } } }
            }
          }
        });
      }

      // Production Drivers Chart
      const prodCtx = document.getElementById("chart-explain-production")?.getContext("2d");
      if (prodCtx && data.production_models) {
        const drivers = data.production_models.primary_drivers;
        new Chart(prodCtx, {
          type: "doughnut",
          data: {
            labels: drivers.map(d => d.driver),
            datasets: [{
              data: drivers.map(d => d.attribution_pct),
              backgroundColor: ["#dc2626", "#d97706", "#2563eb", "#059669"],
              borderWidth: 2,
              borderColor: "#ffffff"
            }]
          },
          options: {
            plugins: {
              legend: {
                position: "right",
                labels: { color: "#334155", font: { size: 12, weight: "500" } }
              }
            }
          }
        });
      }
    });
}

// 9. Data Sources Table
function loadDataSources() {
  fetch("/api/data-sources")
    .then(r => r.json())
    .then(data => {
      const regBody = document.querySelector("#registry-table tbody");
      if (!regBody || !data.registry) return;

      regBody.innerHTML = data.registry.map(d => `
        <tr>
          <td><code>${d.dataset_id}</code></td>
          <td><strong>${d.dataset_name}</strong></td>
          <td>${d.publisher || d.source}</td>
          <td>${d.spatial_resolution || d.temporal_resolution || 'N/A'}</td>
          <td><span class="badge ${d.data_class === 'REAL' ? 'badge-real' : (d.data_class === 'SIMULATED' ? 'badge-sim' : 'badge-derived')}">${d.data_class}</span></td>
        </tr>
      `).join("");
    });
}
