# ADS-B v22: combination CatBoosts fitted with LIRF's ordinary rows (declared 19 September 2026)

Declaration `configs/adsb_v22_v1.json` (03:36 UTC, commit 819bebe), recorded before any v22 fit or
candidate score. Baseline: **v19's development predictions** (330.302 / 269.512 / 191.878 on Jan /
Jul / Nov 2025, `runs/adsb_v21_v2/selection.json`, candidate `lirf_ens3_70`). Rule, pre-registered:
not worse than v19 on any month, mean ratio at most 0.9998 (0.02%), the paired-day interval below
zero on every month the candidate touches (`interval_condition_touched_months_only`; January 2025
has no LIRF row in scope because LIRF was unmatched that month); lowest mean ratio wins; otherwise
nothing and v19 stays.

## Why

`docs/adsb_v21/README.md` and `docs/adsb_v21_v2/README.md`: three LightGBMs fitted on every airport
with only the LIRF fill population excluded, blended at 0.7 on matched LIRF non-population rows with
ordinary clocks (scope `lirf_nonpop_matched`), improved v18 by 0.74% (July 2025) and 0.16%
(November 2025) and shipped as v19 (official 241.1789, −0.095% vs v18). The gain grew with the blend
weight over 0.3 / 0.5 / 0.7, the NM-residual target was the strongest single learner, and no
CatBoost has ever been fitted with LIRF rows. In v15–v17 the LightGBM → combination-CatBoost step
(`max_ctr_complexity` 2 / 3, 20,000-round cap) was worth two to three times the LightGBM gain on the
non-LIRF rows. This round takes the same step on the LIRF scope.

## Learners (`runs/adsb_v22/`, one per worker, 16 threads, folds seasonal → forward → refit)

| variant | recipe | target | seed | worker |
| --- | --- | --- | --- | --- |
| cat_lirf_ctr2_s34 | timeline_v1 CatBoost, `max_ctr_complexity` 2, 20,000-round cap, all airports, fill population excluded | label | 20260934 | arcw1 |
| cat_lirf_ctr3_s35 | as above, `max_ctr_complexity` 3 | label | 20260935 | arcw2 |
| cat_lirf_ctr2_nm_s36 | as cat_lirf_ctr2_s34 | nm_residual (label − NM clock; no clock → NaN) | 20260936 | arcw3 |
| cat_lirf_ctr3_nm_s37 | as cat_lirf_ctr3_s35 | nm_residual | 20260937 | arcw4 |

Runner `tools/run_adsb_learner.py` (unchanged since v21; snapshot
`docs/adsb_v22/run_adsb_learner_declared.py`), `--exclude-population`, no excluded airport, the
`matches` tables and `stands.parquet` hash-bound in every run record. arcw1–arcw4 were started at
03:33 UTC; chains launched 03:37 (arcw1) and 03:49–03:51 UTC (arcw2–4).

Ensembles (`tools/ensemble_adsb_full.py`, mean of finite member predictions):
`ens_v22_cat_lirf_label` (s34 + s35), `ens_v22_cat_lirf_nm` (s36 + s37), `ens_v22_cat_lirf4` (all
four), `ens_v22_lirf_q` (¼ `ens_v21_lgb_lirf3` + ¾ `ens_v22_cat_lirf4`), `ens_v22_lirf_half` (½ + ½).

## Candidates (18)

`lirf_ens3_70` reproduces v19 (check, cannot qualify). Every other candidate is the v18 recipe outside
LIRF and, on `lirf_nonpop_matched`, `(1 − w) · v9 + w · table` — the table **replaces** v19's LightGBM
blend there, it is not stacked on it; every other LIRF row stays v9. Tables and weights:
`ens_v22_cat_lirf4`, `ens_v22_lirf_q`, `ens_v22_lirf_half` at 0.7 / 0.85 / 1.0;
`ens_v22_cat_lirf_label`, `ens_v22_cat_lirf_nm` at 0.7 / 0.85; each learner alone at 0.7 (records).

Ranking scope: 8,781 LIRF rows (8,669 in January 2026, 34 in July 2026), as for v19. No new data,
no December row, nothing uploaded automatically.
