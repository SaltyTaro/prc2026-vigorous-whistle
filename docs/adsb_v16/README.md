# ADS-B v16: learner structure on top of the v15 ensemble (16 September 2026)

v15 (`docs/adsb_v15/README.md`) showed that the learner family still moves every
airport but that seeds and round caps alone buy 0.1–0.3%. This campaign changes the
*structure* of the learners: airport specialists, CatBoost two-way categorical
combinations and CatBoost depth 10, stacked on the v15 ensemble. Declaration
`configs/adsb_v16_v1.json` (04:48 UTC; amended 05:12 UTC before any scoring, see
below). Baseline and rule as in v15: the v14 development predictions, not worse on any
month, mean ratio ≤ 0.997; the v15 gain counts toward the bar because v14 is still the
incumbent.

## Declared learners (all without LIRF, timeline bundle + 27 ADS-B features)

| variant | kind | change | worker |
| --- | --- | --- | --- |
| lgb_ap_EDDF … lgb_ap_LTFM (nine) | LightGBM | fitted, early-stopped and predicting one airport each (`--airports`); NaN elsewhere | arcw4 |
| cat_ctr2 | CatBoost | `max_ctr_complexity` 2 (two-way categorical combinations, e.g. stand × runway), 20,000-round cap | arcw2 |
| cat_d10 | CatBoost | depth 10, 20,000-round cap, seed 20260916 | arcw1 |

Ensembles: `ens_v16_lgb_ap` (the nine specialists; each non-LIRF row has exactly one),
`ens_v16_cat` (the three earlier CatBoosts plus the two new ones), `ens_v16_bag` (mean of
the v15 LightGBM group mean and `ens_v16_cat`), `ens_v16_bag_ap` (three group means),
`ens_v16_v15_ap` (the v15 bag and the specialists). Eight candidates: `ens_v15_bag`
(re-scored), `ens_v16_bag`, `ens_v16_bag_ap`, `ens_v16_v15_ap`, each at w = 0.6 / 0.7.

Amendment (05:12 UTC): the LTFM specialist's fold record failed on a NaN matched-row
RMSE (LTFM has no ADS-B matches); the runner now records None there and lgb_ap_LTFM
was refitted with that one-line change (both runner hashes are in the declaration; the
declared version is snapshotted at `run_adsb_learner_declared.py`). The launcher's chain
submission moved to a script file with a detached stdin (tooling only).

## Learner fits

_(filled in after the chains complete)_

## Development results and selection

_(filled in after `tools/run_adsb_v11.py --config configs/adsb_v16_v1.json`)_
