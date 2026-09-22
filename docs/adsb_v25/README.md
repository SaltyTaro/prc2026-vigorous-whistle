# adsb_v25: the residual stage on a twelve-month leave-one-month-out (22 September 2026)

Incumbent: `vigorous-whistle_v22` (official 239.1335; `configs/adsb_v24_v1.json`, candidate `v7x_r12x_lirf25`). This round changes
only the residual stage of that recipe (the LightGBM on `label − v9` over matched non-LIRF rows, weight 0.3 through the 0.7 committee
blend). Declaration `configs/adsb_v25_v1.json` at **03:45:35 UTC** (commit `782cc46`), before any development fit at protocol.

## Why this stage

`docs/audit_20260922/README.md`: removing the residual stage from the v22 recipe costs +0.67 / +0.84 / +2.22 / +2.62% on Jan / Jul /
Nov / Dec, against +0.17 / +0.50 / +1.20 / +1.77% for the 29-learner committee — one LightGBM that fits in minutes carries 1.5× the
committee's marginal value on the covered months, and January 2026 is covered like December. The residual learning curve
(`docs/adsb_v24/learning_curve.json`) showed adjacent months dominate its value; several inner folds of the shipped residual stopped at
the 4,000-round cap; a CatBoost residual had never been tried. The v24 assets (cross-fitted v9 for all twelve months, the label-free
feature caches) make a twelve-fold leave-one-month-out of this stage possible for the first time.

## Design

- **Variants** (all: scope `matched_no_lirf`, feature set `nb`, `matches_v11`, target `label − cross-fit v9`, training = the eleven
  other months, rounds from the 5-fold day-grouped inner split, patience 200):
  `r12x` control (the shipped protocol, cap 4,000); `adj2` sample weights by circular calendar-month distance to the target month
  (1 → 2.0, 2 → 1.5, else 1.0); `adj4` (1 → 4.0, 2 → 2.0); `cap8k` (cap 8,000); `cat` CatBoost (depth 8, learning rate 0.05, l2 3,
  native categorical airport / stand / runway, cap 8,000). Weights apply to the training folds only; the inner validation is unweighted.
- **Candidates** (full v22 recipe with the residual replaced): `r12x_repro` (check, not selectable), `adj2`, `adj4`, `cap8k`, `cat`,
  `mean_r12x_cat`, `mean_adj2_cat` (row-wise mean of two members' residuals).
- **Two readings.** (1) The residual stage's own out-of-sample RMSE on its scope, on all twelve held-out months, pooled by rows.
  (2) On the four recipe months (the only months with out-of-sample committee tables) every candidate is scored on all rows against the
  v22 recipe's development predictions (`runs/adsb_v24/dev/predictions_<month>.parquet`, reproduced exactly: 328.8317 / 267.6083 /
  189.4823 / 176.5416).
- **Rule** (pre-registered): not worse on any recipe month; mean ratio ≤ 0.9998; paired-day interval below zero on every touched
  recipe month; pooled twelve-month scope RMSE strictly below the control's. Lowest mean ratio among eligible; otherwise keep v22.
- **Ship**: member refits on all twelve months (one model per ranking target month for a weighted member: January weights by distance
  from month 1, July from month 7; each ranking row takes its own month's model); v9, committee and LIRF blends exactly v22's; floor −12;
  export `vigorous-whistle_v23.parquet`; independent verification; audit; the user's approval before upload.

## Execution

- Before the declaration the tool's code path was smoke-tested on 30,000 subsampled rows at 60 rounds (outputs discarded; an
  index-alignment bug in the selection's pooled scope error was fixed; no candidate was scored at protocol). Disclosed in the declaration.
- arcw0–arcw3 started 03:48 UTC; 133 input files (558.6 MB: every declaration parent, the feature caches, tools, packages) synced to
  each; chains launched 03:52–03:55 UTC with 16 threads: arcw0 Dec / Apr / Aug, arcw1 Jan / May / Sep, arcw2 Jul / Feb / Jun,
  arcw3 Nov / Mar / Oct (recipe months first).
- CatBoost proved the slow step (~87 min per month on 16 threads: every inner fold ran to the 8,000-round cap), so at 05:33 UTC each
  first-wave chain was cut after its running month (the fit itself untouched), the second months were queued behind them with a
  waiting chain (arcw0 Apr, arcw1 May, arcw2 Feb, arcw3 Mar), and arcw4–arcw7 were started, synced (plus scikit-learn 1.7.2) and given
  the third months at 05:44 UTC (arcw4 Aug, arcw5 Sep, arcw6 Jun, arcw7 Oct). Eight workers, one month each in the second wave.
- The reproduction check `r12x_repro` equals the incumbent's development prediction on every row of December, January and November
  (0 differing rows): the pipeline and LightGBM determinism are confirmed. `cap8k` is identical to the control on those months (the
  8,000 cap only changed non-median inner folds).
