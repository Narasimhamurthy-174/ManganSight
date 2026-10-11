# Who does what

## Your tasks (need YOUR accounts / browser / machine)
1. Install Python 3.10+, create venv, `pip install -r requirements.txt` (on Windows, if rasterio/geopandas fail, use conda-forge).
2. Register a Google Earth Engine account (free for research/education) -> https://code.earthengine.google.com/register ; run `earthengine authenticate`.
3. Register on Bhukosh and NGDR; find out what Balaghat layers you can actually download. Screenshot what you find.
4. Download MRDS shapefile; download MOIL annual reports (FY2019-20 to FY2024-25) and IBM yearbook.
5. Ask your SIH mentor/faculty whether *any* real MOIL operational data exists. (Ask; don't assume.)
6. Run every command I give you and paste back errors/outputs verbatim.

## My tasks
* Write and test code, design experiments, review your outputs, write docs, explain concepts, debug from your pasted errors.
* I cannot download data or log in to GEE/Bhukosh/NGDR for you (my sandbox has no network).

## Sequence (each step = one message to me)
| Step | Goal | You send me |
|------|------|-------------|
| 1 | Environment works | output of `python scripts/00_check_environment.py` |
| 2 | Boundary + MRDS occurrences | count of usable positives printed by script 01 |
| 3 | GEE exports (boundary, S2, DEM, ERA5-Land) | file list in `data/raw/` |
| 4 | Geology layer (or digitised proxy) | what Bhukosh/NGDR offered |
| 5 | Grid + features | grid cell count |
| 6 | Exploration models + spatial CV | metrics tables |
| 7+ | Production, scenarios, backend, dashboard | ... |
