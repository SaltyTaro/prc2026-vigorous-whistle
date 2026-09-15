# After v11 (248.4368, 4th; third is 247.0109): plan written 15 September 2026 ~09:20 UTC

Gap to third: 1.43 s (0.58%). Nothing below is declared or scored yet.

## Step 1 — v11b: residual with neighbour context (cheapest, in progress)

`src/prc2026_adsb/features.py` now has `neighbour_features(departures, matches)`
(11 `nb_*` columns: matched neighbours' ADS-B taxi-out medians/max/stopped time on
the same runway within ±10/±30 min, airport-wide ±10 min, previous/next runway
neighbour taxi). It covers ~85% of November rows (vs 71% matched), i.e. it also
reaches unmatched departures at covered airports. Timing: 81 s per month
single-threaded on the workstation (fine). Not yet in any config or tool.

To do: declare `configs/adsb_v11_v2.json` (copy of v1 with the `nb_*` features
added to `residual_model.features`, `applies_to` = rows with a match OR at least
one runway neighbour within 30 min, `prior_evidence` citing v11's official
248.4368 and the LOMO −2.9/−3.6/−8.9%), extend `tools/run_adsb_v11.py`
(`month_table`: merge `neighbour_features`; `predict_residual`: apply where
`adsb_matched == 1 or nb_runway_30m_count > 0`), rerun LOMO (Jan/Jul/Nov 2025),
export with `tools/export_adsb_v11.py --version 12` (same extension), verify,
upload only with approval.

## Step 2 — v12: full retrain with ADS-B features in the base models

Layer the 16 `adsb_*` + 11 `nb_*` label-free features onto the NM parent bundle
(pattern: `src/prc2026_arrivalctx/`, ID-aligned extra table), rerun the v7
recipe's development folds and refit on Azure (12 models; ~2–3 h). Needs pass-2
matches for every 2025 month (extraction on arcw3/arcw4/arcw5 was finishing at
09:20 UTC; then `tools/adsb_sync.py collect`, `stands` (relearn from all months),
`pass2 HOST MONTHS`, `collect2`).

## Step 3 — pushback detection for "dark" segments (62% of matches)

Gate share is 33–38%; the remaining segments start already moving. Ideas: use
the learned stand position to estimate the pushback distance already covered
(`adsb_first_dist_stand_m` exists), speed profile at the first points, and a
per-airport pushback-duration prior.

Housekeeping: arcw1–arcw5 are running (authorized); deallocate when the surface
files are collected (`az vm deallocate`). All surface parquet files are on the
workers under `data/external/adsblol/surface/` (git-ignored); workstation has
Mar 2025, Jan/Jul 2026 and the matches.
