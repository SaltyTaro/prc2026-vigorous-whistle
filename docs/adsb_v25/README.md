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
- All twelve months finished between 07:23 and 07:36 UTC; each mirrored (`runs/adsb_v25/dev/`) and the worker deallocated as it
  finished. arcw0 was kept for the export.

## Development results (`development_results.json`, `runs/adsb_v25/dev/selection.json`)

Residual-scope RMSE (v9 + r on matched non-LIRF rows, out of sample), pooled over the twelve held-out months:

| variant | pooled scope RMSE | vs control | months better than control |
| --- | ---: | ---: | ---: |
| `r12x` (control) | 119.655 | — | — |
| `cap8k` | 119.655 | 0.000% | 0 (identical: the cap only moved non-median inner folds) |
| `adj2` | 119.610 | −0.038% | 7 / 12 |
| `adj4` | 119.921 | +0.222% | 3 / 12 |
| `cat` | 118.004 | −1.380% | 12 / 12 (every inner fold ran to the 8,000 cap) |
| mean(`adj2`, `cat`) | 117.309 | −1.961% | — |
| mean(`r12x`, `cat`) | 117.397 | −1.887% | — |

Full recipe on the four recipe months versus the v22 development predictions (328.8317 / 267.6083 / 189.4823 / 176.5416):

| candidate | Jan | Jul | Nov | Dec | mean | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `r12x_repro` (check) | 0 rows differ | 0 | 0 | 0 | 0.000% | check |
| `cap8k` | 0 | 0 | 0 | 0 | 0.000% | no |
| `adj2` | −0.011% | −0.008% | −0.048% | −0.030% | −0.024% | no (July interval reaches 0) |
| `adj4` | −0.004% | +0.005% | −0.056% | −0.028% | −0.021% | no |
| `cat` | −0.016% | −0.023% | −0.016% | −0.099% | −0.039% | no (Jan / Nov intervals cross 0) |
| **`mean_adj2_cat`** | **−0.024%** | **−0.027%** | **−0.068%** | **−0.109%** | **−0.057%** | **yes — selected** |
| `mean_r12x_cat` | −0.018% | −0.022% | −0.043% | −0.092% | −0.044% | yes |

Selected `mean_adj2_cat`: paired-day intervals [−0.15, −0.04] / [−0.10, −0.04] / [−0.18, −0.07] / [−0.27, −0.12] s, 22/31, 26/31,
27/30, 26/31 days better; no airport worse than +0.014% on any month. Reading: the CatBoost residual is the real gain (better on all
twelve months on its own scope, still improving at the cap) and the mean with the weighted LightGBM adds the ensemble effect; the
month-distance weights alone are marginal (7 of 12 months), `adj4` is worse than uniform, and a longer LightGBM cap changes nothing.
The 20:1 damping between the residual's own error and the full recipe (0.3 weight, the committee on the same rows) is why a 2% scope
gain is a 0.06–0.11% recipe gain. Expected official transfer: of the order of −0.1 to −0.25 s, December being the anchor.

## Ship

- `tools/export_adsb_v25.py --version 23 --stage fit --threads 16` on arcw0 from 07:40 UTC: the `adj2` member refit twice (January and
  July target weights) and the `cat` member once, on all twelve months' in-scope rows under the inner rule. The export's first attempt on
  arcw0 stopped at the v9-from-scratch check: on Linux the v9 LIRF mixture classifier early-stops differently (358 rounds against the
  shipped 454; 370 of 383 population rows round differently, up to 1,430 s), so the uploaded v9 reproduces only on the workstation, as
  it did for v22. The tool was therefore split into a fit stage (models saved with a hash-bound fit record) and an assemble stage that
  reloads exactly those models; the assemble stage, the verifier and the audit run on the workstation. `export_failed_v9_check.log`
  on arcw0 preserves the first attempt.
- Fit stage on arcw0 (16 threads, 07:40–09:23 UTC): `adj2` January target 3,074 rounds (inner 3,330 / 3,074 / 2,849 / 3,989 / 2,892;
  weight sum 1,397,680.5), `adj2` July target 2,630 rounds (weight sum 1,369,633.0), `cat` 8,000 rounds (every inner fold at the cap;
  5,566 s); 1,113,566 in-scope training rows each (the count v22's residual recorded). Models and `fit_record.json` mirrored to
  `runs/adsb_v25/export_v23/`; arcw0 deallocated at 09:24 UTC — every VM is deallocated.
- Assemble stage on the workstation (`--stage assemble`, 335 s; one runtime key added for `export_adsb_v11.ranking_table`'s
  neighbour-feature switch, which the declared `nb` feature set already lists): **`runs/submissions/vigorous-whistle_v23.parquet`,
  SHA-256 `f960b72e0b729d47729eed2272634ba8525e57cff9bad243291d96c8fd14fac7`**, 344,841 rows, minimum −12, maximum 120,549. Versus
  the uploaded v22: 194,207 rows change (all inside the residual scope), RMS 6.56 s, median 1 s, 99.9% within 63 s, largest 205 s
  (EHAM disruption-week rows); LIRF and LTFM untouched. `export.json`.
- **Verification** (`tools/verify_adsb_v25.py --version 23`, `verification.json`): declaration parents and receipt provenance; the
  selection belongs to the declaration and selected the exported candidate; v9 from scratch equal to the uploaded v9 after rounding and
  the export's v9 equal to v22's recorded ranking v9; the mirrored twelve-month v7 nodes re-predict 3,000 sampled clock-present rows
  with **difference 0.0**; the three saved models reloaded (tree counts equal the recorded rounds) and the residual recomputed on an
  independently rebuilt ranking table with each row taking its calendar month's `adj2` model, **difference 0.0**; the blend recomputed
  with plain arithmetic **equal to the export on every row**; every row outside the residual scope equal to the uploaded v22; the
  LIRF fill population equal to the uploaded v9 and missing-clock LIRF rows equal to v22; 1,113,566 training rows and both weight sums
  recounted from the scratch tables, match tables and the declared distance scheme.
- **Audit** (`tools/audit_export.py --version 23 --compare 22 21 9`, `export_audit_v23.json`): IDs exact, no NaN, 17 negative
  predictions (none below −12, as v22); extremes 944 / 37 / 12 / 2 rows above 1 h / 4 h / 12 h / 24 h (v22: 941 / 37 / 12 / 2); the
  same five January airport-months above the 2025 development means; the EHAM 2–7 January and LFPG 7 January disruption days as in
  v22; the LIRF fill population (383 rows) unchanged; negative-proxy rows p50 1,088 s; no row moved by more than 600 s.
- Upload only with the user's approval; the official score against v22 (239.1335) is the final test.
- The four-month development mean was −0.057%, the December anchor −0.109%; the transfer is
  again at or above the December read (v20: −0.50% → −0.29 s; v21: December-driven; v22: −0.25% → −0.19%; v23: −0.11% → −0.13%).
  **v23 is the incumbent. Never re-upload v1–v23.** Standings snapshot after scoring: `leaderboard_after_v23.txt` (first of 174,
  0.50 s ahead of second; the second- and third-placed teams both improved earlier today).
- Reading for the next round: the CatBoost residual was still improving at its 8,000-round cap on every inner fold of every month and
  every refit; a declared follow-up with a higher cap (or learning rate 0.1) and a second seed is the cheapest remaining item, after
  which the committee's CatBoost kinds are the next VM round. The month-distance weights are marginal on their own (7 of 12 months)
  and `adj4` is worse than uniform: no reason to weight the committee refits.
