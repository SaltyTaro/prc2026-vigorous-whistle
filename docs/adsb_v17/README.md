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
