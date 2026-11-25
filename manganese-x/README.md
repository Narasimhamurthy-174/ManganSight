# ManganSight (formerly MANGANESE-X)
**AI-Powered Manganese Exploration Prioritisation and Production Forecasting Platform**  
*SIH Problem Statement 26009 — Ministry of Steel / MOIL Ltd.*  
*Study Area: Balaghat District, Madhya Pradesh (Metric CRS: EPSG:32644, UTM 44N)*

> **Scientific Notice:** This is an uncertainty-aware decision-support platform. It does **not** claim to directly detect underground manganese ore bodies from satellite imagery. All surface outputs represent mineral prospectivity, manganese-associated environmental signals, and exploration prioritisation. Production forecasts are scenario-based model estimates.

---

## System Architecture

ManganSight connects two pipelines into an integrated decision and GIS layer:
1. **Exploration Intelligence Pipeline**:
   - 250 m district grid (149,510 cells).
   - USGS MRDS occurrence records cleaned and verified.
   - Sentinel-2 multi-spectral composites & SRTM DEM terrain features with 5 km spatial smoothing.
   - Spatial block cross-validation (10–40 km blocks) to prevent autocorrelation leakage.
   - 18-model ensemble (Logistic Regression, Random Forest, Gradient Boosting across 2 windows and 3 background draws).
   - Applicability domain masking, confidence tiering (Medium / Low), and step-out target delineation (17 zones $\ge 5\text{ km}$ from known records).
2. **Production Forecasting & Operational Risk Pipeline**:
   - Company-wide MOIL monthly production tracking with seasonal rainfall dip modeling.
   - Weekly mine-level operational simulator (Balaghat, Tirodi, Ukwa, Sitapatore) calibrated to the 14.5% monsoon dip.
   - Past-only temporal features (weather lags, rolling downtime, availability) with verified zero-leakage.
   - Expanding-window time-series forecasting.
   - Operational what-if scenario simulator (downtime reduction, maintenance rescheduling, extreme rainfall).
   - Transparent rule-based recommendations framed as "potential actions for consideration".
3. **Interactive 7-Page GIS Dashboard**:
   - Executive Overview
   - Exploration Intelligence & Leaflet GIS Map
   - Zone Details (Known Belt & Step-Out Candidates)
   - Production Intelligence & Real vs Target Charts
   - What-If Scenario Simulator
   - Model Explainability & Driver Decomposition
   - Data Provenance & Scientific Methodology

---

## Quick Start

### 1. Environment Activation
```powershell
# From the project root
cd C:\Users\Administrator\Downloads\manganese-x\manganese-x
..\.venv\Scripts\Activate.ps1
```

### 2. Run Test Suite
```powershell
pytest -v
```
All unit tests across exploration, spatial CV, features, production data layer, operational simulator, scenario engine, recommendations, and backend REST endpoints are fully automated with synthetic and verified test fixtures.

### 3. Launch Dashboard & Backend API
```powershell
python backend/app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in any web browser. The dashboard runs fully offline in DEMO mode.

---

## REST API Endpoints

- `GET /api/health` — Service health and scientific disclaimer
- `GET /api/prospectivity` — Summary metrics of prospectivity and confidence tiers
- `GET /api/prospectivity/map-cells` — Sampled spatial grid points for web GIS mapping
- `GET /api/prospectivity/<cell_id>` — Per-cell evidence trace, spectral drivers, and confidence tier
- `GET /api/exploration-priority` — Priority A and step-out candidate counts
- `GET /api/zones` — Geo-referenced known belt zones and 17 step-out targets
- `GET /api/boundary` — Balaghat district GeoJSON boundary
- `GET /api/production/history` — Real MOIL monthly production & simulated weekly operations
- `GET /api/production/forecast` — 12-week operational forecast with uncertainty bounds
- `GET /api/production/risk` — Shortfall risk classification, weather vs equipment attribution
- `POST /api/scenario` — Real-time operational what-if simulation
- `GET /api/recommendations` — Evidence-linked operational & exploration recommendations
- `GET /api/model/explainability` — Permutation importance and driver attributions
- `GET /api/data-sources` — Verified dataset registry with explicit `data_class` tags

---

## Data Classification Policy
Every dataset and API response carries an explicit data classification:
- `REAL`: Ground-truth observed data (e.g., USGS MRDS records, ERA5-Land weather, reported MOIL monthly totals).
- `DERIVED`: Computed features, spatial aggregations, or calculated shortfalls.
- `SIMULATED`: Synthetic operational logs (labeled everywhere; never represented as actual MOIL internal telemetry).
- `MODEL_OUTPUT`: Model predictions, prospectivity indices, forecasts, and scenario estimates.
