# MANGANESE-X
Uncertainty-Aware Exploration-to-Production Intelligence System for Manganese Mining (Balaghat, MP)

SIH Problem Statement 26009 - Ministry of Steel / MOIL Ltd.

> Decision-support system. It does **not** detect underground manganese from satellite imagery.
> Outputs are *predicted prospectivity* and *exploration priority*, requiring geological field validation.

## Status
Phase 1 (environment/scaffold) + start of Phase 2 (data ingestion). Nothing has been evaluated yet: **Not yet evaluated.**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # fill GEE_PROJECT
python scripts/00_check_environment.py
pytest -q
```
Then follow `docs/ROADMAP.md` and `docs/DATA_SOURCES.md`.

## Data policy
REAL / SIMULATED / DERIVED / MODEL_OUTPUT are tracked in `data/metadata/dataset_registry.yaml`.
Simulated operational data is never presented as MOIL data.


ManganSight
python -m venv .venv 
.venv\Scripts\activate  
pip install -r requirements.txt  
C:\Users\Administrator\Downloads\manganese-x\.venv\Scripts\Activate.ps1
python scripts/gee/export_boundary.py
python scripts/gee/export_sentinel2.py
python scripts/gee/export_dem.py
