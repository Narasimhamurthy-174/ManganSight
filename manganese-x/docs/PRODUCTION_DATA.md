# Production data: what is real, what is not (updated)

## Correction
`docs/DATA_SOURCES.md` (item D7/D8) said monthly production was probably not public. That was too pessimistic:
**MOIL publishes company-wide monthly production** in press releases (moil.nic.in and stock-exchange filings; news sites repeat them).
What is still NOT public as far as found: mine-wise monthly production, equipment downtime, maintenance and blasting delays.
Also, MOIL's own documents list its Balaghat-district mines as Balaghat, Sitapatore, Tirodi and Ukwa (earlier I doubted Sitapatore: that doubt was unfounded).

## Files
`data/raw/production/moil_monthly_production.csv`: months found so far (Jan 2025, Apr-Aug 2025), cumulative checks, FY totals and a reported FY2025-26 target.
The monthly values add up to the reported cumulative figures (Apr-Jun 5.02, Apr-Jul 6.47, Apr-Aug 7.92 lakh tonnes).

## Limits you must remember
* Scope is the WHOLE company (Balaghat district mines plus the Maharashtra mines), not Balaghat alone.
* Months without a press release are missing. Releases are mostly about record months, so this series can be biased towards good months.
* The FY2025-26 target (22 lakh tonnes) comes from a press report; the monthly phasing of targets is not public. Any monthly "shortfall vs target" assumes even phasing (22 / 12 = 1.83 lakh tonnes a month) and must be labelled as an assumption.

## Your task: extend the series
Add one row per month for as many months as you can (target: April 2024 to the latest month), using the same columns.
Where to look: MOIL press releases (files under `moil.nic.in/upload_files/upload_news/`), the monthly notes on SteelOrbis, Business Standard, Angel One.
Use `value_type = reported` only when the month's tonnage is stated; use `derived` (and say how) otherwise. Put your initials in `checked_by_you` after opening the source.

## How it will be used
* REAL layer: company-wide monthly production vs real Balaghat rainfall (ERA5-Land): seasonal pattern, shortfall against the stated target (even-phasing assumption).
* SIMULATED layer: mine-level weekly operations for the what-if demo, calibrated to the real seasonal dip and labelled SIMULATED everywhere.

## Registry entry to add to data/metadata/dataset_registry.yaml
```
  - dataset_id: D_PROD_MOIL_MONTHLY
    dataset_name: MOIL company-wide monthly manganese ore production (press releases)
    source: MOIL press releases and stock-exchange filings
    publisher: MOIL Limited
    url: https://moil.nic.in
    download_date: unknown
    time_period: "months found so far: 2025-01, 2025-04 to 2025-08"
    spatial_resolution: company-wide (all mines, incl. Maharashtra)
    temporal_resolution: monthly
    crs: n/a
    license: public disclosure (check terms)
    processing_steps: "Transcribed by hand with source URL; derived months flagged"
    limitations: "Not Balaghat-only; months without releases missing; record-month bias possible"
    data_class: REAL
    status: acquired
```
