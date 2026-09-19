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

## Fits (arcw1–arcw4, 03:37–09:47 UTC)

| learner | seasonal best / RMSE | forward best / RMSE | refit rounds | chain |
| --- | --- | --- | ---: | --- |
| cat_lirf_ctr2_s34 | 13,218 / 549.56 | 12,879 / 375.32 | 13,048 | 5.3 h |
| cat_lirf_ctr3_s35 | 14,685 / 549.45 | 11,516 / 374.72 | 13,100 | 6.0 h |
| cat_lirf_ctr2_nm_s36 | 14,284 / 209.00 | 11,837 / 168.41 | 13,060 | 5.6 h |
| cat_lirf_ctr3_nm_s37 | 13,106 / 208.78 | 13,182 / 168.13 | 13,144 | 5.8 h |

Whole-population fold RMSE (label learners include the LIRF fill population; NM-residual learners score their
clock-present rows), 1–3% under the v21 LightGBM counterparts (551.25 / 377.74 and 213.78 / 173.10). Every
model was fitted once; the workers were deallocated after mirroring. The two mixed ensembles were built as
flattened member lists that implement the declared weights exactly (`ens_v22_lirf_q`: each LightGBM four times
and each CatBoost nine times, 48 entries; `ens_v22_lirf_half`: four and three times, 24 entries).

## Selection (`runs/adsb_v22_v1/selection.json`, `development_results.json`)

`lirf_ens3_70` reproduces v19 exactly (ratio 1.0 on every month). Selected **`lirf_q_85`**: v19's recipe with
its LIRF blend replaced by `0.15 · v9 + 0.85 · ens_v22_lirf_q` on `lirf_nonpop_matched`:

| month | v19 | lirf_q_85 | change | paired-day interval (s) | days better | scope rows |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Jan 2025 | 330.302 | 330.302 | 0 | untouched | – | 0 |
| Jul 2025 | 269.512 | 268.424 | **−0.404%** | [−2.04, −0.40] | 22 / 31 | 13,992 |
| Nov 2025 | 191.878 | 191.662 | **−0.113%** | [−0.37, −0.08] | 22 / 30 | 7,401 |

Mean ratio 0.99828 (−0.172%). LIRF: 641.4 → 635.7 s (July), 449.8 → 448.5 s (November). Eight candidates are
eligible; the four CatBoosts alone (`lirf_cat4_70`, −0.134%) are slightly weaker than the v19 LightGBM blend
they replace, and the label pair / NM pair alone do not clear the November interval. The CatBoost step on the
LIRF scope is therefore additive to the LightGBM group, not the two-to-three-fold step seen on the non-LIRF
rows in v15–v17. No weight or member was chosen after the scores.

## December (opened 19 September; `docs/december/score_adsb_v22_v1.json`)

The same candidates scored on December 2025 with the shipped January–November models (every model
out-of-sample; `tools/score_december.py`): v9 209.36, v18 recipe 178.41, **v19 recipe 177.88**,
**lirf_q_85 176.99 (−0.501%)**, best `lirf_cat4_85` 176.92 (−0.540%); every v22 candidate −0.34 … −0.54%.
LIRF on December: 354.7 → 348.5 s (−1.74%) on 9,506 scope rows of 12,068. December's LIRF is 79% matched, the
January-2026 regime, which is why the read is three times the development mean; the pre-registered selection
is kept (`cat4_85` is 0.04% better on December, not a selection month).

## Export, verification, audit

`tools/export_adsb_v11.py --version 20` → **`runs/submissions/vigorous-whistle_v20.parquet`, SHA-256
`20516971c44a03aa658acf132259cb1e58643e468552d5371a9b6f9d5dbaf78a`**, 344,841 rows, min −12, max 120,549,
residual refit rounds 1,173 ([`export.json`](export.json)). LIRF ranking rows in scope: 8,781 of 26,899 (8,669 in
January 2026, 34 in July); identical to v19 elsewhere.

`tools/verify_adsb_v11.py --version 20` passed ([`verification.json`](verification.json)): receipts and parents
re-checked, v9 reproduced from scratch, ADS-B features reproduced from the saved matches, the residual recomputed
from the saved booster (difference 0.0), independent recomputation equal to the export, rows outside the scopes
equal to v9, 300 of 300 segment re-derivations agree. `tools/audit_export.py --version 20`
([`export_audit_v20.json`](export_audit_v20.json)): identical to v19 outside LIRF; on LIRF RMS difference 35.7 s
(one row moved by more than 1,800 s: a fill-hedge row where both models sit between the NM clock and the
schedule interval); extremes, negative-proxy rows, the LIRF population and the disruption days unchanged from v19.
