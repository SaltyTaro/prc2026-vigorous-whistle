# ADS-B v20: the v19 surface features inside the full-data learners (declared 18 September 2026)

Declaration `configs/adsb_v20_v1.json` (08:48 UTC, commit df9accb), written after v17 shipped
(official 241.5068) and before any v20 learner was fitted. Baseline: **v17's development
predictions** (330.372 / 271.583 / 192.276 on Jan / Jul / Nov 2025,
`runs/adsb_v18_v2/selection.json`). Rule, pre-registered this time: not worse than v17 on
any month, mean ratio ≤ 0.9995 (0.05%), every month's paired-day interval of the RMSE
difference below zero; lowest mean ratio wins; otherwise nothing.

## Why

The adsb_v19 screen (`docs/adsb_v19_screen/`) found that the eleven surface features move
a second-stage model on the incumbent's residual by −0.15% on July and November and by
nothing on January. A second-stage model is the weakest place for a feature to act; this
round puts the features where the committee's gains have come from — inside the learners
(v17: −0.28% from four new combination CatBoosts; v18: −0.06% from four more). The user
directed that minor improvements are worth shipping.

## Learners (`--extra-features data/external/adsblol/features_v19`, eleven features, 314 in total)

| variant | recipe | worker |
| --- | --- | --- |
| lgb_x19_s24 / s25 / s26 | timeline_v1 LightGBM + v19 features, seeds 20260924–26 | arcw4 (first) |
| cat_ctr2_x19_s27 | ctr2 + v19 features, seed 20260927, 20k cap | arcw2 |
| cat_ctr2_x19_s28 | ctr2 + v19 features, seed 20260928 | arcw3 |
| cat_ctr3_x19_s29 | ctr3 + v19 features, seed 20260929 | arcw1 |
| cat_ctr2_x19_s30 | ctr2 + v19 features, seed 20260930 | arcw4 (after the LightGBMs) |

Feature tables: `tools/build_adsb_v19_features.py` on all thirteen months from
`data/external/adsblol/matches` (the learners' own pass-2 tables, 1,332 stands; the screen
had used `matches_v11`), 26–37 roll-start points and 415–476 imputed stand codes per month;
every table hash-bound in the declaration and in each run record
(`inputs.extra_feature_tables_sha256`). Runner: `tools/run_adsb_learner.py` with the new
`--extra-features` / `--extra-tag` options (commit c9f08ff; snapshot
`docs/adsb_v20/run_adsb_learner_declared.py`); launcher syncs the tables to the worker.

Ensembles: `ens_v20_lgb_x19` (three new LightGBMs), `ens_v20_lgb_all` (six v15 + three new),
`ens_v20_ctr_x19` (the new CatBoosts), `ens_v20_ctr_all` (nine v18 + new); bags at ¼ / ¾:
`ens_v20_bag_q_all` (both groups refreshed), `ens_v20_bag_q_x19` (new learners alone),
`ens_v20_bag_q_lgb` (LightGBM group refreshed only), `ens_v20_bag_q_ctr` (CatBoost group
refreshed only). Candidates: the four bags at blend 0.7 plus `ens_v18_bag_q_70` as the
reproduction check.

## Fits

Chains launched 08:49–08:50 UTC 18 September on arcw1–arcw4 (D16ds_v7, 16 threads).
First record (lgb_x19_s24 seasonal, 08:53 UTC): 314 features, best iteration 1,552,
matched / unmatched 415.87 / 700.46 — indistinguishable from the v15 LightGBMs
(415.8–415.9 / 700.0–700.7). Results table to follow.

LightGBMs (all three fitted by 09:20 UTC, ~3.5 min per fold; mirrored, ensembles built by
member multiplicities, `ens_v20_bag_q_lgb` equal to the ¼ / ¾ formula to 3.6e-12 s):

| variant | seasonal matched / unmatched | forward matched / unmatched | refit rounds |
| --- | --- | --- | ---: |
| lgb_x19_s24 | 415.87 / 700.46 | 204.87 / 645.38 | 1,388 |
| lgb_x19_s25 | 415.24 / 700.42 | 204.25 / 644.32 | 1,348 |
| lgb_x19_s26 | 415.50 / 700.59 | 204.33 / 645.15 | 1,384 |
| (v15 LightGBMs, for reference) | 415.8–415.9 / 700.0–700.7 | 204.1–205.3 / 644.9–645.2 | |

Descriptive learner-level RMSE on the non-LIRF rows (Jan / Jul / Nov): ens_v15_lgb 328.47 /
217.07 / 162.91; ens_v20_lgb_x19 328.99 / 218.00 / 163.68; ens_v20_lgb_all 328.48 / 217.12 /
162.83; the LightGBM-refreshed committee ens_v20_bag_q_lgb 324.64 / 209.22 / 157.16 against
v17's ens_v18_bag_q 324.63 / 209.22 / 157.19. The v19 features do nothing for the LightGBM
learner; the round rests on the combination CatBoosts.
