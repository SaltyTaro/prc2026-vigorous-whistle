# ADS-B v18: more combination-CatBoost seeds and re-weighting (16–17 September 2026)

Declared `configs/adsb_v18_v1.json` at 18:09 UTC on 16 September, right after v17 shipped
as `vigorous-whistle_v16` (official 241.75). Baseline: the **v16 development predictions**
(the incumbent's recipe on the same rows: 330.469 / 271.782 / 192.417 on Jan / Jul / Nov
2025, `runs/adsb_v17_sel/selection.json`); rule: not worse on any month and mean ratio
≤ 0.9985 (the v17 bar), lowest mean ratio wins, otherwise nothing. The declaration states
up front that the pure re-weighting candidates were expected to miss the bar (a
descriptive scan of the recorded tables put the LightGBM share of ¼ within 0.1 s of ⅓ at
the learner level); they were included so that the rule, not a hand choice, picks the
weight and share.

## Declared learners (CatBoost, 20,000-round cap, without LIRF)

| variant | recipe | worker | seasonal best / matched / unmatched | forward best / matched / unmatched | refit rounds | hours |
| --- | --- | --- | --- | --- | ---: | ---: |
| cat_ctr2_s20 | ctr2, seed 20260920 | arcw2 | 12,676 / **414.89** / 699.67 | 11,784 / **199.83** / 644.52 | 12,230 | 4.7 |
| cat_ctr2_s21 | ctr2, seed 20260921 | arcw3 | 12,646 / 419.70 / 700.54 | 11,935 / 203.46 / 646.31 | 12,290 | 4.8 |
| cat_ctr3_s22 | ctr3, seed 20260922 | arcw1 | 11,469 / 414.99 / 696.69 | 15,464 / 202.70 / 647.11 | 13,466 | 5.7 |
| cat_ctr3_s23 | ctr3, seed 20260923 | arcw4 | 11,194 / 415.88 / 698.83 | 14,150 / 200.75 / 644.70 | 12,672 | 5.3 |

Chains ran 18:10–23:55 UTC; every fold record carries the declared runner hash; all
workers deallocated afterwards. Three of the four are the strongest single learners
fitted so far on the matched rows (previous best 415.30 / 202.23).

Ensembles (`tools/ensemble_adsb_full.py`, unchanged; nested means via multiplicities,
each equal to its weighted-group formula to 1.6e-11 s): `ens_v18_ctr` (the five v17
combination CatBoosts and the four new ones), `ens_v18_bag_c2` (⅓ LightGBM group, ⅔
ens_v18_ctr), `ens_v18_bag_q` (¼ / ¾), `ens_v17_bag_ctr_q` (¼ / ¾ of the v17 committee;
built into `runs/adsb_v17/` as declared). Learner-level RMSE on the non-LIRF rows
(seasonal / forward): ens_v17_ctr 267.74 / 158.45 → ens_v18_ctr 267.57 / 158.24; the v16
committee ens_v17_bag_ctr_c2 267.35 / 157.33 → ens_v18_bag_q 267.15 / 157.19.

## Development results and selection ([`development_results.json`](development_results.json))

`tools/run_adsb_v11.py --config configs/adsb_v18_v1.json` (the baseline reproduces v16;
the incumbent-recipe candidate `ens_v17_bag_ctr_c2_70` scores ratio 1.00000):

| candidate | Jan | Jul | Nov | mean | paired-day 95% intervals (s) |
| --- | ---: | ---: | ---: | ---: | --- |
| ens_v18_bag_q_70 | −0.029% | −0.073% | −0.073% | **−0.059%** | [−0.23, −0.02] / [−0.26, −0.14] / [−0.19, −0.09] |
| ens_v17_bag_ctr_q_70 | −0.013% | −0.057% | −0.030% | −0.033% | Jan interval includes zero |
| ens_v18_bag_q_80 | +0.038% | −0.027% | −0.078% | −0.022% | worse on Jan |
| ens_v18_bag_c2_70 | −0.014% | −0.013% | −0.039% | −0.022% | — |
| ens_v17_bag_ctr_q_80 | +0.058% | −0.007% | −0.027% | +0.008% | worse on Jan |
| ens_v18_bag_c2_80 | +0.051% | +0.036% | −0.054% | +0.011% | worse on Jan and Jul |
| ens_v17_bag_ctr_c2_80 | +0.068% | +0.051% | −0.008% | +0.037% | worse on Jan and Jul |

**Nothing is eligible; no export, no upload.** The best candidate is 0.09 percentage
points short of the bar. Readings: (1) the committee lever is spent — four more learners of
the best kind moved the combination group a quarter as far as v17's four did, and the
blended candidate 0.06%; (2) blend weight 0.7 is the optimum (every 0.8 candidate is worse
on January, the month with the least ADS-B coverage); (3) ¼ against ⅓ LightGBM share is
within noise. The four learners are banked in `runs/adsb_v18/` for any future round that
changes features or targets rather than seeds. v16 (241.75) remains the incumbent.
