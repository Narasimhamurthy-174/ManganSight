# Architecture (see Phase 0 specification in the chat; summary)
Offline pipelines (ingest -> clean -> grid -> features -> train/validate -> uncertainty -> production forecast/risk -> zone integration -> export artifacts)
-> Flask serving layer (precomputed artifacts + light live logic: scenarios, rules) -> Leaflet/Plotly dashboard.

## Correction to Phase 0
Phase 0 said the metric CRS is UTM 43N (EPSG:32643). That was WRONG. Balaghat (~80E) is in UTM zone 44N (EPSG:32644).
Zone number = floor((longitude + 180) / 6) + 1. config.yaml uses EPSG:32644.
