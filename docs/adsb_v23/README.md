# ADS-B v23: the v20 recipe refit on all twelve months of 2025 (declared 19 September 2026)

Declaration `configs/adsb_v23_v1.json` (11:16 UTC, commit 26c3a24), recorded before any twelve-month fit.
Incumbent: **v20, official 240.89** (`vigorous-whistle_v20.parquet`, SHA-256 `20516971…af78a`), the
`lirf_q_85` candidate of `configs/adsb_v22_v1.json`. Nothing in the recipe changes — not a weight, scope,
feature, member or v9. What changes is the fitting data of every learner behind the recipe's tables and of the
residual stage: December 2025, opened on 19 September (`docs/december/README.md`), joins the fit.

## Why

`docs/december/month_value.json`: on December 2025, out-of-sample for every model involved, the v20 recipe built
from the forward-fold learners (fitted January–September, October stopping) scores 180.13; built from the
January–November refit learners it scores 176.99 — the two adjacent months bought **−1.74%** (v19 recipe:
180.68 → 177.88, −1.55%). December is the month adjacent to January 2026 (the ranking's larger half), winter,
and the same ADS-B coverage regime (LIRF 79% matched, LFPG 24%, LTFM 0%; January 2026: 80% / 21% / 0%).
Adding it (+8.6% rows, one adjacent month) is expected to be worth a fraction of that on the ranking. Every
other lever measured on December in the same session was flat: adding the LIRF-inclusive CatBoosts to the
non-LIRF committee moves December by at most 0.01% (not declared).

## What is refit

- **29 leaf learners** (`runs/december/refit_leaves.json`: 12 LightGBM, 17 CatBoost) — every leaf behind
  `ens_v20_bag_q_all` (9 LightGBM + 13 combination CatBoosts), `ens_v21_lgb_lirf3` (3 LightGBM) and
  `ens_v22_cat_lirf4` (4 LIRF-inclusive CatBoosts). Each is refit **once** by `tools/run_adsb_learner_dec.py`
  (imports the frozen `tools/run_adsb_learner.py` for features, encoding and fitting) on the frozen
  January–November refit population (1,919,370 rows) plus December's 165,677 departures from the opened build
  (`data/december/d198dea10625b0a75977/`, hash-bound) = **2,085,047 rows**, at exactly its recorded rounds
  (median best iteration of its seasonal and forward folds — the frozen refit rule; no fold saw December), with
  its recorded kind, seed, parameters, exclusions, target and extra features. The categorical vocabulary is
  rebuilt from the twelve-month fitting rows (the frozen rule: fitting rows only).
- **Tables**: each declared table is rebuilt for the ranking as the per-row mean of finite twelve-month refit
  predictions over the flattened leaf list its frozen run record averaged (multiplicities preserved), written to
  `runs/adsb_v23/ens/<name>/refit/ranking_predictions.parquet`.
- **Residual stage** (v12's nb model, matched rows outside LIRF, single booster): refit on the in-scope rows of
  January, July, November **and December** 2025, rounds from the same day-grouped five-fold inner rule applied
  once to that four-month training set.
- **Unchanged**: v7, the LIRF mixture (v9), the −12 floor, the blend weights 0.7 / 0.85 and the scopes.

## Validation

By construction no held-out month exists for a twelve-month refit. The declared test is the official score
against v20 (240.89): if v21 is not better, v20 remains the incumbent and the slot is spent. No candidate search,
no weight, member or scope choice.

## Execution (arcw0–arcw7, 11:28–18:04 UTC on 19 September)

Inputs fanned out VM-to-VM from arcw0 (frozen bundles, raw files, code; an ephemeral key, removed afterwards);
the December build, the December match table, the December v19 features, the runner and the 29 leaf fold
records shipped from the workstation. One chain per worker (`tools/launch_december_refit.py`), LightGBMs first,
16 threads per fit. Three first attempts failed on missing inputs (arcw3 `ranking.parquet`, arcw4 December raw,
arcw6 `requirements-timeline.txt`) before any model was written and were relaunched after the sync; their
launch logs keep the tracebacks. Every leaf was fitted once.

| leaf | kind | rounds | fitted rows | seconds | worker |
| --- | --- | ---: | ---: | ---: | --- |
| adsb_full_nolirf | lightgbm | 1,306 | 1,924,343 | 490 | arcw1 |
| adsb_v15/lgb_clean | lightgbm | 1,606 | 1,924,340 | 244 | arcw3 |
| adsb_v15/lgb_deep | lightgbm | 2,373 | 1,924,343 | 597 | arcw3 |
| adsb_v15/lgb_s16 | lightgbm | 1,464 | 1,924,343 | 516 | arcw0 |
| adsb_v15/lgb_s17 | lightgbm | 1,472 | 1,924,343 | 545 | arcw2 |
| adsb_v15/lgb_s18 | lightgbm | 1,310 | 1,924,343 | 227 | arcw2 |
| adsb_v16/cat_ctr2 | catboost | 11,034 | 1,924,343 | 6,591 | arcw0 |
| adsb_v17/cat_ctr2_d10 | catboost | 9,286 | 1,924,343 | 10,095 | arcw0 (moved from arcw7) |
| adsb_v17/cat_ctr2_oh2 | catboost | 14,036 | 1,924,343 | 10,792 | arcw1 |
| adsb_v17/cat_ctr2_s17 | catboost | 10,088 | 1,924,343 | 6,032 | arcw0 |
| adsb_v17/cat_ctr3 | catboost | 11,169 | 1,924,343 | 7,162 | arcw1 |
| adsb_v18/cat_ctr2_s20 | catboost | 12,230 | 1,924,343 | 7,449 | arcw2 |
| adsb_v18/cat_ctr2_s21 | catboost | 12,290 | 1,924,343 | 7,474 | arcw2 |
| adsb_v18/cat_ctr3_s22 | catboost | 13,466 | 1,924,343 | 7,818 | arcw3 |
| adsb_v18/cat_ctr3_s23 | catboost | 12,672 | 1,924,343 | 7,322 | arcw3 |
| adsb_v20/cat_ctr2_x19_s27 | catboost | 13,646 | 1,924,343 | 8,666 | arcw4 |
| adsb_v20/cat_ctr2_x19_s28 | catboost | 11,308 | 1,924,343 | 7,192 | arcw4 |
| adsb_v20/cat_ctr2_x19_s30 | catboost | 13,188 | 1,924,343 | 7,755 | arcw5 |
| adsb_v20/cat_ctr3_x19_s29 | catboost | 12,685 | 1,924,343 | 7,575 | arcw5 |
| adsb_v20/lgb_x19_s24 | lightgbm | 1,388 | 1,924,343 | 321 | arcw4 |
| adsb_v20/lgb_x19_s25 | lightgbm | 1,348 | 1,924,343 | 253 | arcw4 |
| adsb_v20/lgb_x19_s26 | lightgbm | 1,384 | 1,924,343 | 489 | arcw5 |
| adsb_v21/lgb_lirf_cap_s32 | lightgbm | 1,767 | 2,083,556 | 534 | arcw6 |
| adsb_v21/lgb_lirf_nm_s33 | lightgbm | 1,744 | 2,062,577 | 276 | arcw6 |
| adsb_v21/lgb_lirf_s31 | lightgbm | 1,584 | 2,083,560 | 250 | arcw5 |
| adsb_v22/cat_lirf_ctr2_nm_s36 | catboost | 13,060 | 2,062,577 | 8,760 | arcw7 |
| adsb_v22/cat_lirf_ctr2_s34 | catboost | 13,048 | 2,083,560 | 8,461 | arcw6 |
| adsb_v22/cat_lirf_ctr3_nm_s37 | catboost | 13,144 | 2,062,577 | 8,793 | arcw7 |
| adsb_v22/cat_lirf_ctr3_s35 | catboost | 13,100 | 2,083,560 | 8,788 | arcw6 |

Total fit time 39.3 h across the eight workers (`refits.json`: per-leaf worker, rounds, rows, seconds and record hash).
`cat_ctr2_d10` (9,286 rounds at depth 10, the longest fit) was queued third on arcw7 and moved to arcw0 once arcw0's chain
had finished (arcw7's chain script was stopped while its second fit ran on as an orphan process); it was fitted once, on arcw0.
Each host's leaves were mirrored as they completed (`runs/adsb_v23/<dir>__<name>/refit/{model, ranking_predictions.parquet,
run.json}`, plus `launch.log`); every host was deallocated after mirroring and the ephemeral fan-out key was removed from
arcw1–7's `authorized_keys` and deleted from arcw0. All eight VMs were deallocated by 18:19 UTC.

## Export, verification, audit

`tools/export_december_refit.py --version 21` ([`export.json`](export.json)): the 11 declared tables rebuilt as flattened-member
means of the twelve-month refit predictions (`runs/adsb_v23/ens/<name>/refit/`); v9 reproduced from scratch (rounding equal to
the uploaded v9); the residual refit on the **378,270** in-scope rows of January / July / November / December 2025 at **1,676**
rounds (inner best iterations 902 / 2,672 / 1,879 / 1,508 / 1,676, median; `runs/adsb_v23/export_v21/`); blended and floored
exactly as v20 → **`runs/submissions/vigorous-whistle_v21.parquet`, SHA-256
`d3173e24628ccdc4ea26ea18393cf8673c874ba278486a1a05dc24c8af1ad1dc`**, 344,841 rows, min −12, max 120,549; rows outside every
scope equal the uploaded v9. Versus v20: 307,131 rows change, RMS 16.7 s (EDDF 13.0 … LSZH 22.2, LIRF 15.0), median |Δ| 5 s,
99.9th percentile 158 s, mean −0.50 s, no row moved by more than 1,800 s.

`tools/verify_december_refit.py --version 21` ([`verification.json`](verification.json)) passed at 18:32 UTC: parents and the
receipt re-checked; every refit record checked against its frozen fold records (twelve months, 2,085,047 rows, December's
165,677, rounds = median fold best iteration, recipe fields, code hashes, December identity, ranking match tables); every
mirrored model re-predicts 3,000 sampled ranking rows from an independently rebuilt feature frame with the twelve-month
fitting-row vocabularies — **maximum difference 0.0 on all 29 leaves**; every table equals the mean of finite leaf predictions
over its frozen flattened member list; v9 from scratch; the residual recomputed from the saved booster (difference 0.0); the
blend recomputed and equal to the export; rows outside the scopes equal to v9. The verifier's first run failed on its own
ranking frame (it had used the December-specific `scoring.adsb_table` for the ranking rows, leaving every row unmatched);
[`verifier_fix.json`](verifier_fix.json) records the failure, the cause and the verifier-only correction — no model, prediction,
table, residual, export or declaration changed.

`tools/audit_export.py --version 21 --compare 20 19 9` ([`export_audit_v21.json`](export_audit_v21.json)): IDs exact, no NaN,
20 negative predictions (none below −12); extremes 945 / 37 / 12 / 2 rows above 1 h / 4 h / 12 h / 24 h (v20: 937 / 37 / 12 / 2);
the LIRF fill population (383 rows) and the 169 negative-proxy rows unchanged (v9 rows); January-2026 airport means within 0.5%
of v20 (the same five airport-months sit >15% above the 2025 development means, as in v20); the EHAM 2–4 and LFPG 7 January
disruption days slightly lower (EHAM 3 January 2,045 s vs 2,074 s).

## Upload and official result

Uploaded with the user's approval at 18:36:18 UTC on 19 September 2026, processed at
18:37:01 UTC: **official 239.5766 on all 344,841 rows — −1.3134 s (−0.545%) vs v20's 240.89**
([`official_result.json`](official_result.json)). The declared test passes; **v21 (239.5766, SHA `d3173e24…1dc`) is the
incumbent.** Never re-upload v1–v21. The user had reported rank 1 at 240.7357 earlier that day, so v21 sits 1.16 s under that
figure (the assistant does not poll the standings). December's month-value read (two adjacent months worth −1.74% on December)
called the direction and the order of magnitude: one adjacent month for the January half and 8.6% more rows for everything else
bought 0.545% on the ranking, the largest single step since the CatBoost committee (v14 → v17, 1%). From here every future lever
is developed on the folds plus December (fold models never see December) and its learners are refit on twelve months before
shipping; December reads of the *shipped* stack are no longer out-of-sample.
