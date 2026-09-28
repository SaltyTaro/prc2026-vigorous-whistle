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
- **Completion.** All 61 jobs finished with 0 failures: 58 LOMO leaves, v7 L01/L07 and the reproduction check. The last chain ended
  at 00:39:46 UTC on 28 September.
- **Mirror.** The first full mirror, models included, was reaped for low local RAM after 19 jobs. The rest was mirrored lean
  (predictions + run.json only). The CatBoost/LightGBM LOMO models of arcw1 and arcw3–arcw7 stay on those workers' disks.
- **Close-down.** All seven workers were checked idle, then arcw0, arcw1 and arcw3–arcw7 were deallocated at about 01:20 UTC on the
  user's instruction ("close down the compute once they finish"). Their disks are kept. All eight VMs (arcw2 by the parallel
  session) report "VM deallocated" at 01:28 UTC.

## Tables, rows and the declared weight round (28 September 2026)

- **Tree tables** (parallel session, docs/lomo_dev/tables.json): runs/lomo_dev/ens/ens_v20_bag_q_all/lomo/predictions.parquet
  (SHA-256 a7f722fe...) and ens_v22_lirf_q/lomo (1d5e80de...), 344,419 rows each, seasonal row order; lean mirror (predictions +
  run.json of all 58 LOMO jobs, 126 files hash-checked); member rule rebuilds both seasonal tables within 1e-9. LOMO minus seasonal:
  committee Jan -7.76 / 34.2 s (mean / RMS), Jul -0.78 / 25.8; LIRF table Jan -6.26 / 36.6, Jul -1.21 / 30.9.
- **Rows** (tools/run_lomo_rows.py -> runs/lomo_dev/rows.parquet, SHA-256 9072bc73..., rows_summary.json): the fold path reproduces
  the fair rows exactly (<= 9.1e-13). On the equal footing the mix overtakes the v9 side on matched rows: January base 122.06 / mix
  121.22 (fold 128.03 / 129.59), July 153.84 / 155.29 (fold 155.91 / 157.57). v33 January 323.457, July 259.584.
- **Round** (configs/lomo_weights_v1.json, tools/run_lomo_round.py --score --select -> lomo_weights_v1_results.json; months LOMO
  January / July + fair December, November reported): **selected g_u300_cs50** (eligible: g_u300, g_u300_cs50), mean -0.0544%,
  Jan -0.032 / Jul -0.075 / Dec -0.057% (Nov -0.136%), intervals below zero in 3/3. That is two of v35's three shipped layers, so the
  equal-footing months confirm v35 under the unchanged rule; no new export. No weight candidate qualifies: network share 0.85 is
  Dec -0.134% but Jan / Jul intervals cross zero (Nov +0.040%); 1.0 is Dec -0.163% but Nov +0.231%; 0.65 worse everywhere; outer
  weights and the LIRF matched v9 share fail their intervals (o_m70 worse everywhere).
