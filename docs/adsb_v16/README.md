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

## Learner fits (04:50–09:25 UTC; every worker deallocated afterwards)

| variant | seasonal best iter | seasonal matched / unmatched | forward best iter | forward matched / unmatched | refit rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| v15 cat_s16 (reference) | 13,870 | 416.34 / 696.64 | 15,284 | 202.23 / 645.88 | 14,577 |
| cat_ctr2 | 9,610 | 417.99 / 697.42 | 12,457 | 202.73 / 645.49 | 11,034 |
| cat_d10 | 8,749 | 418.39 / 697.03 | 13,231 | 203.96 / 646.05 | 10,990 |
| nine specialists (merged) | 206–1,361 | see below | 219–892 | see below | 212–953 |

Learner means alone on the non-LIRF rows (seasonal / forward): v15 bag 268.81 / 159.18;
cat_ctr2 alone **268.84 / 159.60** (as good as the whole v15 ensemble — the two-way
categorical combinations pay on the unmatched rows even though the matched-row numbers
are flat); cat_d10 270.49 / 161.97; the specialists 275.07 / 165.65 (per airport they
match the single global LightGBM: EHAM −7% on Nov but +7% on Jan/Jul, small-data
variance); `ens_v16_cat` 269.17 / 160.30; **`ens_v16_bag` 268.51 / 158.53**;
`ens_v16_bag_ap` 268.53 / 158.16; `ens_v16_v15_ap` 269.45 / 159.25. The nested means were
built with the unchanged ensemble tool by listing members with multiplicities; each
equals its group-mean definition to 7e-12 s.

## Development results and selection ([`development_results.json`](development_results.json))

`tools/run_adsb_v11.py --config configs/adsb_v16_v1.json` (the baseline reproduces v14:
331.309 / 273.169 / 194.212; the v15 candidates reproduce their v15 scores):

| candidate | Jan | Jul | Nov | mean | paired-day 95% intervals (s) |
| --- | ---: | ---: | ---: | ---: | --- |
| **ens_v16_bag_60** | −0.149% | −0.231% | −0.484% | **−0.288%** | [−0.88, −0.30] / [−0.73, −0.54] / [−1.11, −0.79] |
| ens_v16_bag_70 | −0.106% | −0.186% | −0.561% | −0.284% | [−0.68, −0.17] / [−0.64, −0.36] / [−1.31, −0.91] |
| ens_v16_bag_ap_70 | −0.037% | −0.165% | −0.569% | −0.257% | Jan interval includes zero |
| ens_v16_bag_ap_60 | −0.081% | −0.199% | −0.464% | −0.248% | Jan interval includes zero |
| ens_v15_bag_60 | −0.121% | −0.177% | −0.346% | −0.214% | all below zero |
| ens_v16_v15_ap_60 | +0.036% | −0.076% | −0.249% | −0.096% | — |

The airport specialists add nothing beyond the global ensembles (every `_ap` candidate
is behind its counterpart without them); the two new CatBoosts move the bag from
−0.214% to −0.288%. **`ens_v16_bag_60` has mean ratio 0.99712 — 0.012 percentage points
short of the declared 0.997 bar — so the rule selects nothing** and no export was made
under this declaration. Every month and every interval is negative.
