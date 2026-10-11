# Data Sources for MANGANESE-X (Balaghat)

Status legend: **FOUND** = a page for it was located during research. **VERIFY** = believed to exist, you must confirm.
**GAP** = no public source found. Exact downloadable layers for Balaghat were NOT confirmed for GSI portals.

| ID | What | Source | Status | How to get it |
|----|------|--------|--------|---------------|
| D1 | District boundary | FAO GAUL 2015 level-2 in Earth Engine (or Survey of India/Census) | VERIFY | `python scripts/gee/export_boundary.py`, download from Drive to `data/raw/boundary/balaghat_district.geojson`. Check area is ~9,000 km2 |
| D2 | Manganese occurrences (points) | USGS MRDS - https://mrdata.usgs.gov/mrds/ (shapefile listed at http://mrdata.usgs.gov/mrds/mrds.zip) | FOUND | Download, unzip to `data/raw/occurrences/`, run `scripts/01_prepare_occurrences.py`. USGS stopped systematic updates in 2011 and calls MRDS "large, complex, and somewhat problematic": treat as a *starting* list only |
| D2b | Better occurrences / mine locations | GSI Bhukosh http://bhukosh.gsi.gov.in/Bhukosh/MapViewer.aspx ; NGDR https://geodataindia.gov.in | FOUND (portals) / VERIFY (Balaghat layers) | Register, look for mineral-occurrence and geology layers for Balaghat. NGDR states exploration data is being standardised into GIS formats |
| D2c | Mine positions from documents | MOIL Annual Report 2024-25 https://moil.nic.in/userfiles/Annual_Report_2024_25.pdf ; NMET exploration proposals (nmet.gov.in, search "Balaghat manganese") | FOUND | Manually digitise mine/block locations with source + page number. Balaghat district mines named by MOIL: Balaghat, Ukwa, Tirodi, Sitapatore |
| D3 | Geology / lithology / structure | GSI Bhukosh (same as D2b). Fallback: digitise the geological sketch map of the Sausar Group of Balaghat-Seoni (Banerji et al. 2007, as reproduced in the Crimson Publishers paper) | VERIFY | If no vector geology can be downloaded, georeference a published map and digitise. Document it as a proxy |
| D4 | Sentinel-2 SR | GEE `COPERNICUS/S2_SR_HARMONIZED` | VERIFY | `python scripts/gee/export_sentinel2.py` |
| D5 | DEM | GEE `USGS/SRTMGL1_003` | VERIFY | `python scripts/gee/export_dem.py` |
| D6 | Rain + soil moisture | GEE `ECMWF/ERA5_LAND/DAILY_AGGR` (~9 km, daily, reanalysis) | FOUND | `python scripts/gee/export_era5land.py` |
| D7 | MOIL production | MOIL annual reports (annual). IBM Indian Minerals Yearbook - Manganese Ore 2022 https://ibm.gov.in/writereaddata/files/17125770456613da1576db0Manganese_Ore_2022.pdf | FOUND (annual only) | Transcribe tables into CSV with page refs |
| D8 | Monthly/daily mine production, equipment downtime, maintenance, blasting delays | Not public as far as found | **GAP** | Must be SIMULATED unless MOIL/mentor provides it |

## Key facts from the literature you should know (with where they came from)

* Balaghat manganese is hosted in the Precambrian **Sausar Group**, notably the **Mansar Formation**; a
  Balaghat-Ukwa ore belt is described in NMET exploration proposals. Some papers argue the deposits are shear/mylonite-zone
  controlled (epigenetic) in the Sukli-Sitapathore-Tirodi-Ramrama-Bharweli-Ukwa belt. => *Lithology and structure
  (shear zones/fold trends) are likely the strongest predictors; satellite spectral features are supporting evidence.*
  This is the hypothesis your ablation study (geology vs +terrain vs +spectral) will test.
* A directly relevant prior paper: "Earth observation approach for targeting stratiform deposit of manganese in
  central India" (ScienceDirect, 2023, pii S0273117723002454). Its abstract notes remote-sensing-based Mn mapping literature is limited.
  **Read it before writing any novelty claim.**
* MOIL: MOIL's own site describes ~1.8 Mt annual production; its Balaghat mine is described as its largest and deepest.

## Things I could NOT verify (do not present as fact)
* Number of Balaghat occurrences in MRDS (unknown until you run the script).
* Whether Bhukosh offers downloadable vector geology at a useful scale for Balaghat.
* Any monthly MOIL production series.
* GEE band names for ERA5-Land (check in the Code Editor Data Catalog page).
