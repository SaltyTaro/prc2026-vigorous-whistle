# ADS-B v17: categorical-combination CatBoosts on top of the v15 ensemble (16 September 2026)

v16 (`docs/adsb_v16/README.md`) found that the CatBoost with two-way categorical
combinations (`max_ctr_complexity` 2) alone matched the whole v15 ensemble, and that the
five-CatBoost group mean was behind that single learner. This round adds four learners of
that kind and tests how to weight them. Declaration `configs/adsb_v17_v1.json` (10:29 UTC;
amended 10:45 UTC before any v17 prediction existed, see below). The baseline is now the
**v15 development prediction** (the incumbent's recipe on the same rows: 330.814 /
272.538 / 193.272 on Jan / Jul / Nov 2025, `runs/adsb_v16_v2/selection.json`); rule: not
worse than v15 on any month and mean ratio ≤ 0.9985 (a 0.15% mean improvement), lowest
mean ratio wins, otherwise nothing. The bar was set before any fit: 0.15% is about twice a
single month's paired-day interval width over the month RMSE, so a qualifying candidate
improves every month by more than the day-level noise explains.

## Declared learners (CatBoost, 20,000-round cap, without LIRF, timeline bundle + 27 ADS-B features)

| variant | seed | change versus the timeline_v1 CatBoost block | worker |
| --- | --- | --- | --- |
| cat_ctr2_s17 | 20260917 | `max_ctr_complexity` 2 (the v16 cat_ctr2 recipe, new seed) | arcw2 |
| cat_ctr3 | 20260915 | `max_ctr_complexity` 3 (three-way combinations) | arcw1 |
| cat_ctr2_d10 | 20260918 | `max_ctr_complexity` 2 and depth 10 | arcw4 |
| cat_ctr2_oh2 | 20260919 | `max_ctr_complexity` 2 and `one_hot_max_size` 2: the airport code (10 values, one-hot at the default and therefore outside CatBoost's combinations) enters the target statistics and the combinations; 49% of the runway codes and 53% of the stand codes of the November 2025 departures are shared by more than one airport, so the plain runway and stand statistics pool airports | arcw3 |

Ensembles (`tools/ensemble_adsb_full.py`, unchanged; nested means via member
multiplicities): `ens_v17_new` (the four), `ens_v17_cat` (the five earlier CatBoosts and
the four), `ens_v17_ctr` (cat_ctr2 and the four), and the bags with the v15 LightGBM group
`ens_v15_lgb`: `ens_v17_bag` (½ / ½ with ens_v17_cat), `ens_v17_bag_ctr` (½ / ½ with
ens_v17_ctr), `ens_v17_bag_c2` (⅓ / ⅔ with ens_v17_cat) and, by the amendment,
`ens_v17_bag_ctr_c2` (⅓ / ⅔ with ens_v17_ctr). Nine candidates: the four bags at
w = 0.6 / 0.7 on every row outside LIRF on top of v9 + the v12 residual, plus
`ens_v16_bag_60`, the incumbent recipe, as a reproduction check (ratio must be 1.0).

Amendment (10:45 UTC, chains 2–14 minutes into their first fold, no v17 prediction
written): a descriptive scan of the recorded v15/v16 fold tables (learner-alone RMSE on
the non-LIRF rows, no fitting) put the best LightGBM share at 0.3–0.4 rather than 0.5 —
a·ens_v15_lgb + (1−a)·cat_ctr2 is 267.84 / 157.78 (seasonal / forward) at a = 0.3 against
268.20 / 157.92 at a = 0.5; with ens_v16_cat 268.19 / 158.50 against 268.51 / 158.53 — so
`ens_v17_bag_ctr_c2` and its two candidates were added. Nothing else changed.

## Learner fits

Chains started 10:31 UTC (arcw1) and 10:43 UTC (arcw2, arcw3, arcw4); one variant per
worker, seasonal → forward → refit. (The launcher's ssh call blocks until its 600 s
timeout although the chain starts within seconds; tooling only.)

| variant | seasonal best iter | seasonal matched / unmatched | forward best iter | forward matched / unmatched | refit rounds | hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v16 cat_ctr2 (reference) | 9,610 | 417.99 / 697.42 | 12,457 | 202.73 / 645.49 | 11,034 | 4.5 |
| cat_ctr2_s17 | 9,891 | **415.30** / 696.58 | 10,285 | 203.33 / 646.45 | 10,088 | 4.1 |
| cat_ctr3 | 10,814 | 417.51 / 697.72 | 11,524 | 202.60 / 647.12 | 11,169 | 4.5 |
| cat_ctr2_d10 | 11,761 | 419.09 / 697.46 | 6,812 | 202.96 / 646.66 | 9,286 | 6.1 |
| cat_ctr2_oh2 | 13,332 | 421.78 / 701.43 | 14,741 | 204.09 / **643.58** | 14,036 | 6.9 |

Every fold record carries the declared runner hash. Individually the four sit in the
cat_ctr2 band; cat_ctr2_oh2 is the weakest on the seasonal months and the best on the
unmatched rows of November. Learner means alone on the non-LIRF rows (seasonal /
forward): the v15 committee `ens_v16_bag` 268.51 / 158.53; the combination group of five
`ens_v17_ctr` 267.74 / 158.45; the nine-CatBoost group `ens_v17_cat` 268.17 / 159.06;
`ens_v17_bag_ctr_c2` (⅓ LightGBM group, ⅔ combination group) **267.35 / 157.33** (the
four bags: 268.15 / 158.10, 267.92 / 157.75, 267.65 / 157.77, 267.35 / 157.33). The
nested means were built with the unchanged ensemble tool by listing members with
multiplicities; each equals its weighted group-mean formula to 1.2e-11 s.

## Development results and selection ([`development_results.json`](development_results.json))

`tools/run_adsb_v11.py --config configs/adsb_v17_v1.json` (the baseline reproduces v15:
330.814 / 272.538 / 193.272; the incumbent-recipe candidate `ens_v16_bag_60` scores ratio
1.00000 on every month):

| candidate | Jan | Jul | Nov | mean | paired-day 95% intervals (s) |
| --- | ---: | ---: | ---: | ---: | --- |
| **ens_v17_bag_ctr_c2_70** | −0.105% | −0.277% | −0.442% | **−0.275%** | [−0.81, −0.05] / [−0.96, −0.57] / [−1.04, −0.70] |
| ens_v17_bag_ctr_c2_60 | −0.128% | −0.278% | −0.322% | −0.243% | all below zero |
| ens_v17_bag_c2_70 | −0.061% | −0.189% | −0.310% | −0.187% | Jan interval touches zero |
| ens_v17_bag_c2_60 | −0.089% | −0.201% | −0.206% | −0.165% | all below zero |
| ens_v17_bag_ctr_70 / _60 | −0.037 / −0.069% | −0.101 / −0.124% | −0.287 / −0.178% | −0.142 / −0.124% | — |
| ens_v17_bag_70 / _60 | −0.003 / −0.039% | −0.034 / −0.066% | −0.186 / −0.089% | −0.074 / −0.065% | — |

Four candidates clear the 0.15% bar; **the rule selects `ens_v17_bag_ctr_c2_70`** (mean
ratio 0.99725), the amendment's ensemble at blend weight 0.7 — the first time 0.7 beats
0.6, as the committee is now strong enough to deserve more of the blend. Two structural
readings: the five combination CatBoosts alone beat all nine CatBoosts at every weight
(the older max_ctr_complexity-1 learners dilute), and the ⅓ / ⅔ split beats ½ / ½ in both
groups. Per airport the selected candidate improves 26 of the 27 airport-months outside
LIRF by 0.04–1.9% (LEMD −0.9 / −0.8 / −1.9%, LEBL −1.4 / −0.2 / −1.0%, LTFM −0.7 / −0.6 /
−0.7%, LFPG −0.0 / −0.8 / −0.7%); the exceptions are small (EDDF Jan +0.24%, LSZH Jan
+0.37%, EHAM Jul +0.87%, EDDM Nov +0.09%). No relaxation was needed.

## Export, verification and official result: 241.75

`tools/export_adsb_v11.py --config configs/adsb_v17_v1.json --selection
runs/adsb_v17_sel/selection.json --version 16`: v9 rebuilt from scratch, the v12 residual
refitted on the 271,115 in-scope rows (1,173 rounds), `ens_v17_bag_ctr_c2` blended at
w = 0.7 on every non-LIRF row (317,942 rows; the 26,899 LIRF rows keep v9 exactly). Output
`runs/submissions/vigorous-whistle_v16.parquet`, SHA-256
**`b6a7ef5e803da4ccd8e5bfbaf61011dd8a34e0e7f3aec8767068f4dfaed1ec23`**, 344,841 rows,
min −12, max 120,549 (inherited). Versus v15: 302,516 rows differ, RMS of the change
16.0 s, median absolute change 5 s, 99th percentile 55 s ([`export.json`](export.json)).
`tools/verify_adsb_v11.py --version 16 --run-root runs/adsb_v17_sel`
([`verification.json`](verification.json)): receipts and parents re-checked, v9 reproduced
from scratch, features and the ensemble reproduced, residual recomputed from the saved
booster (difference 0.0), the blend recomputed row by row, rows outside the scope equal
the uploaded v9, independent segment re-derivation 300/300.

Uploaded with the user's approval at **18:03 UTC on 16 September** (`prc_submit.py upload
--calendar-day`, the third upload of the UTC day) and processed at 18:04:06 UTC:
**241.75 seconds RMSE on all 344,841 rows, 0.83 s (0.34%) better than v15's 242.5783**
([`official_result.json`](official_result.json)) — beyond the development mean of
−0.275%, as v16's and v12_v5's gains also were. Never re-upload v1–v16. Per the user's
standing rule, no leaderboard standing is recorded here.

## What this round says about the next one

- Blend weight: 0.7 won; 0.8 was not declared and should be a candidate next time.
- Group weighting: the LightGBM group is now a minority partner (⅓); a ¼ share, or
  dropping the weaker non-combination CatBoosts from the CatBoost group entirely, are the
  obvious next candidates on the recorded tables (descriptive scans first, then declare).
- More combination CatBoosts (seeds of ctr2 / ctr3) keep paying at roughly the v16→v17
  rate; the depth-10 and one-hot variants were not better than a plain seed.
