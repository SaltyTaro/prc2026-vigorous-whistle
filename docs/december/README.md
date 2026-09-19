# December 2025, opened on 19 September 2026

`authorization_20260919.json`: the user lifted the Phase-1 reservation of December (`configs/validation_v1.json`,
`reserved_audit_months`) at 08:54 UTC, head-to-head at the top of the ranking. December had never been read for
any selection, score, early stopping, refit or ADS-B extraction; from here on it is a **fourth development
month** (adjacent to January 2026, winter, the same ADS-B coverage regime) and, later, refit rows. The Phase-1
"final audit on an untouched month" is forfeited and must not be claimed in any later document.

## Inputs (all label-free until the scorer's final evaluation)

- **Surface positions**: adsb.lol `globe_history` for 2025-12-01 … 12-31 on arcw5 with the unchanged
  `tools/adsb_extract_day.py` (31 December is published under `adsblol/globe_history_2026`; extracted with the
  same tool through a repository-year override script). 31 day files, part hashes in each day's manifest;
  copied to the workstation's `data/external/adsblol/surface/`.
- **Weather**: `tools/pull_december_weather.py` (the frozen tool's request/receipt code) → ten December station
  objects under `data/external/iem_december/`, inventory `weather_inventory.json`; the frozen 2025 inventory is
  untouched.
- **Feature table**: `src/prc2026_december/` runs the six frozen feature layers (phase 3, 4, 5R, weather, NM
  clock, timeline) on the Phase-1 bundle's December rows. Five frozen modules refuse December by a single `raise`;
  `patched.py` executes each from its unchanged source with that one statement replaced by `pass` (frozen files
  untouched, source hashes recorded in the build manifest). Context is every 2025 month through the month being
  built, because the frozen headway features look at the previous n same-runway departures however old.
  `tools/build_december_features.py --month 2025-11 --verify` reproduces the frozen timeline bundle's November
  rows: 331 columns exact, the four cyclic NM-clock float64 columns within 1.11e-16 (the documented Linux/Windows
  libm difference; float32 identical) — `data/december/7dc46218c228a5675e8f/manifest.json`. The December build is
  `data/december/d198dea10625b0a75977/` (165,677 rows × 335 columns, labels alongside).
- **ADS-B matches**: `tools/build_december_matches.py` (the frozen `match_departures`, separate tool because the
  frozen builder's December guard is a v22 parent) for both stand tables: 116,713 matched (70.4%), gate 39–40%;
  LIRF 79%, LFPG 24%, LTFM 0% — January 2026's regime (80% / 21% / 0%). v19 surface features:
  `tools/build_adsb_v19_features.py --months 2025-12`.

## Scoring with the shipped models (`tools/score_december.py`)

`scratch/december.parquet`: v7 from the surface-refit `pipeline.json` (211.80 s), v9 from the January–November
LIRF mixture classifier (209.36 s). Every leaf learner behind a declaration's tables is applied from its refit
model with the categorical vocabulary rebuilt from its fitting rows (`runs/december/learners/`), ensembles are
rebuilt from the recorded member lists, the v19 export's residual booster is applied on the declared matches
directory, and every candidate is scored against the declaration's baseline recipe. December is out-of-sample
for every model involved. First read (`score_adsb_v22_v1.json`): v18 recipe 178.41, v19 177.88, v22's selected
`lirf_q_85` 176.99 (−0.50%) — see `docs/adsb_v22/README.md`.
