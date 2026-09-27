# Leave-one-month-out development: tree side and v9 (27 September 2026, in progress)

**Why.** Before this round, the fold side (committee trees, context networks, LIRF tables) scored January and July 2025 from the
seasonal fold (7 fitting months) and November from the forward fold (9). The v9 side (cross-fit v9 + the 12-month LOMO residual)
fitted on 9–11 months. Only December compared both sides on equal footing (January–November refits), and v33's official gain matched
December. This round makes January and July ranking-like: every fold-side component is refitted on the eleven other 2025 months, as
the ranking's twelve-month refits are. The user approved it ("go").

**Declaration:** [`configs/lomo_dev_v1.json`](../../configs/lomo_dev_v1.json), declared at 13:40:03 UTC before any fit (SHA-256
`74105d76…`, commit `d69f331`). This session covers the tree side and the v9 side. The parallel session covers the 64 context
networks, on AWS GPUs.

## Tree side

- **Tool.** `tools/run_adsb_learner_lomo.py`, a sibling of `run_adsb_learner_dec.py`, which stays unedited.
- **Scope.** 29 leaves (22 in `ens_v20_bag_q_all`, 7 in `ens_v22_lirf_q`) × {2025-01, 2025-07} = 58 fits.
- **Recipe and rounds** are each leaf's twelve-month refit (`runs/adsb_v23/<leaf>/refit`): the recipe comes from the fold records,
  and the rounds are the median of the two folds' best iterations. No early stopping.
- **Population.** The twelve-month refit population (1,919,370 + 165,677 rows) minus every row whose movement month is the held month.
  - The tool refuses to run if a held row is in the fit.
  - The predicted rows must equal the leaf's seasonal rows for that month.
- **Tables.** `tools/build_lomo_tables.py` combines the leaves with `leaf_members` + `ensemble_prediction` (nanmean with
  multiplicities). Output: `runs/lomo_dev/ens/<table>/lomo/predictions.parquet`.
  - Applied to the seasonal leaves, the same rule reproduces the stored seasonal tables (2.4e-11 s / 0.0).
- **Reproduction check.** `lgb_clean` held out on 2025-12 fits exactly the January–November refit population. Its model is
  **byte-identical** to `runs/adsb_v15/lgb_clean/refit/model.txt` (SHA-256 `fab12511…`), so the fit path is exact.
  - Note: its December predictions differ from the cached `runs/december/learners` values on 19 of 165,677 rows (RMS 0.06 s, max
    13.5 s). These are pairs of departures minutes apart, a tie-order difference in the ADS-B neighbour features between
    `prc2026_december.scoring.adsb_table` and `run_adsb_learner.adsb_table`. It is noise; the December cache is left as it is.

## v9 side

- **v7.** `tools/run_v7_lomo.py` wraps the unedited `run_v7_dec.py` with single-month folds L01 / L07 (frozen specs, counts and seed).
- **v9.** `tools/build_lomo_v9.py` writes `runs/lomo_dev/v9/<month>.parquet` with v9 = max(LOMO v7, −12) on clock-present rows.
  - Missing-clock rows and the LIRF fill population keep the seasonal fold v9, as `build_v24_scratch.py` does. That is a disclosed
    caveat covering about 1.5% of rows.
- **Residual stage (correction to the declaration's `residual_stage` line).** The declaration said the residual stage needs no
  refit, because it is already an eleven-month LOMO trained on cross-fit v9. That is incomplete: adj2, cat and rnb take v9 as an input
  feature, so the stored predictions, made with cross-fit v9 inputs, cannot be reused. The parallel session refits adj2/cat for
  January and July on arcw2 (the same fits as shipped, trained on cross-fit v9) and re-predicts the held month with the LOMO v9. It
  re-predicts the rnb nets from their saved held-month models.

## Execution

- **Workers.** `tools/launch_lomo_dev.py` runs on Azure arcw0/1/3/4/5/6/7 (arcw2, option A, is untouched). One job at a time per
  worker, at 16 threads. Jobs are assigned longest-first, about 10.5 h per worker. Chains started 13:43–13:56 UTC.
- **Round E.** Its refit12 lanes were stopped, as the parallel session allowed. arcw4 waited for its refit step to finish first.
- **Local smoke tests** ran before the declaration (3,000 rows per month, 20 rounds): `lgb_clean`, `adsb_full_nolirf`,
  `cat_ctr2_x19_s27` and `cat_lirf_ctr2_nm_s36`.
