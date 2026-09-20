# ADS-B v24: the residual stage on twelve months of cross-fitted v9, and v7 refit on twelve months (declared 20 September 2026)

Declaration `configs/adsb_v24_v1.json` (15:28:06 UTC, commit 376be7b), recorded before any cross-fit fit, month-table build or
candidate score. Incumbent: **v21, official 239.5766** (`vigorous-whistle_v21.parquet`, SHA-256 `d3173e24…1dc`, the v20 recipe with
its 29 leaf learners refit on twelve months). The user asked for the next minor gains on 20 September and chose to play them now rather
than hold them (best-submission scoring, no separate final phase, serial confirmation protects against the v10-style bundle loss).

## What the v21 stack leaves on the table

1. **The residual stage is a four-month model.** v12's residual LightGBM (`label − v9` on matched non-LIRF rows, weight 0.3 through the
   0.7 committee blend; removing it costs +0.95% on the development months) is trained on the in-scope rows of January, July, November
   and December 2025 only — the months with an out-of-sample v9. The learners behind the committee use all twelve months; the residual
   cannot without an out-of-sample v9 for the other eight months.
2. **v7 is still a January–November fit.** It carries weight 0.3 on every non-LIRF row, 0.15 on matched LIRF rows and 1.0 on LIRF rows
   without ADS-B (all of LIRF in July 2026). v23 refit the learners on twelve months and bought −0.545%; v7 was left at eleven.

## Prior evidence (before the declaration)

- `learning_curve.json` (`tools/residual_learning_curve.py`): the declared residual fitted on nested subsets of the other development
  months. December held out (blend RMSE on non-LIRF rows, 160.890 without the residual): November alone −3.03%, July alone −1.90%,
  January alone −2.22%, July+November −3.00%, January+November −3.21%, January+July −2.61%, all three −3.26%; the three-month set
  subsampled to one month's rows −2.63…−2.74%. November held out: December alone −3.17%, July+December −3.17%. July held out: November
  alone −1.07%, January+November −1.23%, all three −1.28%. Reading: **adjacent months matter most, off-season months add little, rows
  still help** (3.5× rows within the same mix: −0.65% blend). The ranking's July half has one summer month (July 2025) behind the
  residual; June and August 2025 are its adjacent months and are only reachable by cross-fitting v7.
- `v7_month_value.json` (`tools/v7_month_value.py`): on December's 164,028 clock-present departures, v7's forward-fold components
  (January–September) score 199.83, the seasonal-fold components 202.98, the January–November refit 196.58 (**−1.63% / −3.15%**),
  most at LIRF (306 → 296) and LTFM (211 → 207) — where v9 carries full or large weight. v7 responds to adjacent months as the learners
  did (`docs/december/month_value.json`).

## Design

- **Cross-fit** (`tools/run_v7_dec.py`, launcher `tools/launch_v7_dec.py`): v7's twelve clock-present components (cat_combined 5,971
  and lgb_combined 1,612 rounds, ten airport LightGBMs at the frozen counts, NM-residual targets, the frozen 276 predictors, seed
  20260913, `prc2026_timeline.native.fit_node` imported frozen) fitted in four month-group folds — A 01/05/09, B 02/06/10, C 03/07/11,
  D 04/08/12 — on the v23 twelve-month population (1,919,370 + 165,677 rows) minus the fold's months, predicting every clock-present
  departure of the held-out months. No early stopping, no held-out label read for fitting. One fold per worker (arcw0–arcw3).
- **Scratch tables** (`tools/build_v24_scratch.py`, `scratch/v24/`): v9 = max(cross-fit v7, −12) on clock-present rows; the eight months
  without a development fold contribute clock-present rows only (missing-clock rows have no out-of-sample v6); the four development months
  keep their fold tables with the fold v9 as `v9_fold`.
- **Development** (`tools/run_adsb_v24.py`): leave-one-month-out over January / July / November / December. Residual variants: `r3` (the
  incumbent's protocol), `r12` (twelve months, fold v9 on the development months), `r12x` (twelve months, cross-fit v9 everywhere),
  `r12x_m1332` (on the learners' 1,332-stand match tables), `r12x_season` (with `season_cos`). Eight fixed candidates:
  `incumbent_repro`, `v7x_r3` (lever 2 alone), `r12` (lever 1 alone), `v7x_r12x` (both, the configuration that ships), `v7x_r12x_m1332`,
  `v7x_r12x_season`, `v7x_r12x_lirf25` / `_lirf50` (plus 0.25 / 0.5 of `ens_v22_lirf_q` on unmatched LIRF non-population rows with
  ordinary clocks). Baseline: the v21 recipe's development predictions (330.3018 / 268.4237 / 191.6621 / 176.9893; the tool refuses to
  run if they do not reproduce). Rule (v22's): not worse on any month, mean ≥ 0.02%, paired-day intervals below zero on touched months;
  lowest mean ratio wins.
- **Ship**: twelve-month v7 refit (`--fold refit`, arcw0) replaces v7 on clock-present ranking rows only; missing-clock rows and the
  LIRF fill population keep the uploaded v9 exactly (the development protocol never scores the mixture on a new v7); the selected
  residual refit once on all twelve months; blends and floor exactly as v21; `tools/export_adsb_v24.py --version 22`,
  `tools/verify_adsb_v24.py --version 22`, `tools/audit_export.py --version 22`; upload only with the user's approval.

## Execution

(to be completed)
