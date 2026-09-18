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

CatBoosts (chains done 13:14–14:24 UTC; each worker mirrored and deallocated as it finished):

| variant | seasonal best iter | seasonal matched / unmatched | forward best iter | forward matched / unmatched | refit rounds | hours |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| cat_ctr2_x19_s27 | 12,985 | 415.70 / 695.95 | 14,308 | 201.40 / 645.55 | 13,646 | 5.4 |
| cat_ctr2_x19_s28 | 11,749 | 417.83 / 701.10 | 10,867 | 202.84 / 646.05 | 11,308 | 4.4 |
| cat_ctr3_x19_s29 | 12,063 | 415.75 / 699.55 | 13,307 | 202.50 / 645.65 | 12,685 | 5.0 |
| cat_ctr2_x19_s30 | 14,083 | 410.68 / 692.79 | 12,294 | 202.31 / 645.76 | 13,188 | 5.1 |
| (v18 seeds, reference) | | 414.9–419.7 / 696.7–700.5 | | 199.8–203.5 / 644.5–647.1 | | |

Individually inside the recorded band (cat_ctr2_x19_s30's seasonal fold is the best of any
CatBoost, its forward fold is mid-band). Ensembles by member multiplicities
(`ens_v20_bag_q_all` 468 entries, `_x19` 48, `_ctr` 312, `_lgb` 36), each equal to its
weighted-group formula to ≤ 3e-11 s. Learner-level RMSE on the non-LIRF rows (Jan / Jul /
Nov): ens_v18_ctr 325.19 / 209.48 / 158.24; the four new CatBoosts alone 325.08 / 209.46 /
158.35 (four learners matching nine — the features help per learner); all thirteen 325.04 /
209.31 / 158.01; the committees: v17's ens_v18_bag_q 324.63 / 209.22 / 157.19,
ens_v20_bag_q_all 324.53 / 209.11 / 157.00.

## Selection ([`development_results.json`](development_results.json)): nothing qualifies

`tools/run_adsb_v11.py --config configs/adsb_v20_v1.json` (14:35 UTC; the reproduction
check ens_v18_bag_q_70 scores ratio 1.00000):

| candidate | Jan | Jul | Nov | mean | paired-day intervals (s) |
| --- | ---: | ---: | ---: | ---: | --- |
| ens_v20_bag_q_all_70 | −0.021% | −0.020% | −0.046% | **−0.029%** | [−0.16, −0.02] / [−0.08, −0.03] / [−0.14, −0.04] |
| ens_v20_bag_q_ctr_70 | −0.022% | −0.021% | −0.044% | −0.029% | all below zero |
| ens_v20_bag_q_x19_70 | −0.033% | −0.017% | −0.022% | −0.024% | Jan / Jul / Nov include zero |
| ens_v20_bag_q_lgb_70 | +0.001% | +0.000% | −0.005% | −0.001% | include zero |

The pre-registered bar (mean ratio ≤ 0.9995 with every interval below zero) is not met:
the best candidate is 0.02 percentage points short. Readings: (1) the v19 features are null
inside the LightGBM learner and small inside CatBoost — four feature-bearing CatBoosts equal
the nine-member group, but the whole committee moves only 0.03%; (2) what is left is
committee growth, at a quarter of v18's step, which was a quarter of v17's. **No export,
no upload; v17 (241.5068) remains the incumbent.** The seven learners are banked in
`runs/adsb_v20/`. All workers deallocated by 14:30 UTC.
