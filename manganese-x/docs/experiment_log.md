# Experiment log (living document)
Rule: every experiment is listed here BEFORE it is judged. Results are copied in as printed.
All numbers are cross-validated skill at ranking known records; none is a probability that a reserve exists.
Metric: belt-AUC = ranking of held-out known records vs other cells within 15 km of any record; chance = 0.5.
Fold-to-fold spread is 0.06-0.15, so differences below ~0.05 are noise. Runs share the same 59 records and similar folds,
so they are NOT independent comparisons.

## Experiment 1 - base features (terrain + spectral, 250 m), 47 occurrence-only records (2026-11-05)
- Location-only baseline (distance to nearest training record): belt AUC 0.70-0.79.
- Feature models: belt AUC 0.49-0.58 (chance within spread). Conclusion: no demonstrated skill at 250 m.

## Experiment 2 - mine sites included (base, all 59 records) (2026-11-06)
Belt AUC of feature models at 10/20/30 km is within 0.02 of Experiment 1. No evidence that producer / past-producer
footprints inflate skill.

## Experiments 3-4 - results (2026-11-06), mean belt AUC over 2 models x 2 background strategies
| feature set | random | 10 km | 20 km | 30 km | 40 km |
|---|---|---|---|---|---|
| location only (no features) | 0.81 | 0.77 | 0.73 | 0.69 | 0.72 |
| base (250 m) | 0.60 | 0.56 | 0.53 | 0.53 | 0.56 |
| base + TPI 5 km | 0.61 | 0.56 | 0.53 | 0.54 | 0.58 |
| scale_1km | 0.69 | 0.63 | 0.58 | 0.59 | 0.60 |
| scale_2p5km | 0.74 | 0.67 | 0.62 | 0.62 | 0.62 |
Rule applied (fixed in advance: every model x strategy cell >= 0.60 at 10, 20 and 30 km AND >= 0.07 above base):
- base + TPI: FAILS (mean margin +0.00). Landform position adds nothing.
- scale_1km: FAILS (0.56-0.60 at 20 km; margin >= 0.07 in only 6 of 12 cells).
- scale_2p5km: all 12 cells >= 0.60 (PASS); margin >= 0.07 in 11 of 12 cells (one cell +0.05; mean +0.10).
  Strictly read, the margin condition is missed by one cell, so the outcome is "promising, not confirmed".
  (The rule did not say whether the margin applies per cell or on average; we do not redefine it after seeing results.)
Observation: skill rises steadily with window size (250 m -> 1 km -> 2.5 km). Elevation barely changes when averaged (rho 0.993 with
the 250 m version), so the gain comes from smoothed slope, NDVI and iron_ratio. Two explanations are still open: (a) blurry label
positions, (b) coarse regional context carries the signal. Experiment 5 helps separate them.
Even the best feature set (0.62-0.67) stays below the location-only baseline (0.69-0.77).

## Experiments 5-6 - results (2026-11-08)
Mean belt AUC over logreg + rf x 2 background strategies (same convention as the table above):
| feature set | random | 10 km | 20 km | 30 km | 40 km |
|---|---|---|---|---|---|
| scale_2p5km | 0.74 | 0.67 | 0.61 | 0.62 | 0.62 |
| scale_5km | 0.75 | 0.69 | 0.64 | 0.66 | 0.66 |
- Experiment 5: skill still rises with window size but is FLATTENING (2.5 -> 5 km: +0.01 to +0.04; the earlier steps were larger).
  Rule applied to scale_5km: all 12 cells >= 0.60 and margin >= 0.07 in 12 of 12 cells (min +0.08, mean +0.12): PASS.
  scale_2p5km remains a near-miss (11 of 12). Four windows have now been tried (250 m, 1, 2.5, 5 km): window size is NOT tuned further.
- Experiment 6 (2.5 km): XGBoost 0.62-0.67 vs random forest 0.60-0.69 vs logistic regression 0.61-0.67 (10-40 km). XGBoost adds nothing; at 5 km
  logistic regression is as good or better (mean 10-30 km: logreg 0.67, rf 0.64, xgb 0.65). Simplest model wins; we keep logistic regression
  as the primary, interpretable model.
- Permutation importance (held-out 20 km blocks, drop in belt AUC): iron_ratio 0.10-0.16 in all three models; elevation 0.06-0.11 for
  logistic regression only; slope and NDVI about 0 (NDVI slightly negative for trees). Spread across folds is large (0.05-0.17).
  One feature (the smoothed red/blue ratio) appears to carry most of the skill.
- Location-only baseline stays higher than every feature model (mean 10-30 km: 0.73 vs 0.63-0.67).

## Experiment 7 (pre-registered BEFORE running) - is iron_ratio sufficient and necessary?
Sets: iron_only_5km, no_iron_5km (and the 2.5 km pair if time allows). Run with --models logreg rf.
Reading rule (fixed in advance), using mean belt AUC at 10-30 km over both models and strategies:
- iron_only within 0.03 of the full 5 km set (0.66) AND no_iron < 0.55  -> the skill is essentially ONE feature.
- no_iron >= 0.58 -> other features contribute meaningfully.
- anything in between -> inconclusive; report as such.

## Experiment 7 - result (2026-11-08)
Mean belt AUC at 10-30 km (both models and strategies): iron_only_5km 0.60 (logistic 0.65, random forest 0.53-0.57);
no_iron_5km 0.57 (0.53-0.59); full 5 km set 0.655.
Rule applied: iron-only was not within 0.03 of the full set (0.60 vs 0.66) and no-iron (0.57) was neither < 0.55 nor >= 0.58 -> INCONCLUSIVE.
Pattern: iron_ratio alone with logistic regression keeps almost all the skill; without it the skill is weak. Random vs spatial split:
no-iron random forest 0.73 (random) -> 0.57 (unseen areas); iron-only logistic 0.68 -> 0.65. The other features' skill mostly does not survive
spatial testing; iron_ratio's does.

## Experiment 8 (pre-registered BEFORE running) - does the uncertainty system carry information? (scripts 12-13)
All thresholds are in config/uncertainty.yaml and are NOT tuned on this test.
- Ensemble (3 algorithms x 2 windows x 3 background draws) vs its members: the ensemble is kept as primary if its mean belt AUC at
  20-30 km is within 0.02 of the best single member (robustness at no cost). Otherwise the best member is reported as primary.
- Confidence tiers (Medium / Low): the Medium tier counts as INFORMATIVE only if, pooled over block sizes, lift(Medium) >= 1.2 x lift(Low)
  AND mean AUC-within-tier(Medium) >= AUC(Low) + 0.03. Otherwise: "tiers have not demonstrated information", and the dashboard says so.
- 'High' confidence is never assigned until independent evidence (geology / lineaments) exists.
- Known records score high in the priority map by construction (training data + proximity term). Only zones WITHOUT a known record
  ("extension candidates") say anything new, and even they are untested until new records or field checks exist.

## Experiment 8 - result (2026-11-11), scripts 12-13
- Ensemble vs members, belt AUC on held-out blocks: ensemble 0.66 (20 km) / 0.65 (30 km); best single member logreg@5km 0.66 / 0.64;
  others 0.61-0.65. Rule (within 0.02 of the best member): PASS. The ensemble is kept: as good as the best member and less sensitive to one choice.
- Confidence tiers (belt, held-out blocks): Medium holds 62% of known records in 49-51% of cells (lift 1.22-1.27); Low holds 38% (lift 0.75-0.78).
  AUC within tier: Medium 0.67 / 0.64 vs Low 0.63 / 0.62 (+0.04 / +0.02, pooled about +0.03).
  Rule (lift ratio >= 1.2 AND AUC gap >= 0.03): PASS - the AUC part only just. Evidence is modest: 59 records, units not independent.
  Caveat: part of the tier is 'resembles known-record conditions', so it overlaps with the ranking itself; not tested separately yet.
- Priority map (script 12): Priority A (top 5%) = 7,255 cells; weight-sensitivity overlap 0.88; 79% of those cells stay in the top group in >= 90% of weight draws.
  PROBLEM FOUND: all top zones are driven by the proximity term; 6 of the top 12 zones contain known records and the 3 'extension candidates' lie only
  0.25-0.35 km from a known record. Priority A is therefore a map of the known belt, not a list of new targets.

## Experiment 9 (defined BEFORE running) - step-out candidates
Purpose: a separate list for NEW areas. Definition (config/uncertainty.yaml, 'stepout'): cells at least 5 km from every known record (the feature-window
radius, so their windows contain no known-record cell), Medium confidence, top 5% of those by the features-only prospectivity index. No proximity term.
Evidence available: the features-only ensemble's held-out belt AUC (about 0.65). No direct test of discovery far from known records is possible with 59
records; step-out zones stay candidates for field checking, not findings.

## Caution: selection optimism
The 2.5 km window was picked as the best of three windows tried on the same 59 records. Its skill is therefore optimistic by an unknown
amount. Confirm on independent data (new records, or a layer with independent evidence) before quoting it.

## Caution: data snooping
We saw the known records on the slope map and noticed they line up with a steep band. Any feature designed from that observation
was suggested by the labels it would be tested on, so cross-validation cannot protect against it. Report such features as EXPLORATORY only.
Independent evidence layers (geology, lineaments from an independent source) do not have this problem.
