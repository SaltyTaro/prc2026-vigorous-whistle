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

- **Cross-fit** (arcw0–arcw3, 15:29–16:00 UTC): CatBoost globals 1,202–1,270 s per fold on 1.53–1.56 M clock-present rows, then the
  LightGBM global and ten airport models; every node reproduces after reload (difference 0.0). The first runner tagged the prediction
  rows with a column named `month` — also the name of a frozen predictor — so every fold failed at the prediction stage after its
  twelve components were fitted (each fold's `launch.log` keeps the traceback). The tag was renamed `held_month` and the prediction
  stage resumed from the saved nodes (`--resume`, checked against the fold's fitting-row identities and the frozen counts; recorded in
  each `run.json`). Amendment recorded in the declaration before any prediction or score; the fitting code is unchanged. Predicted
  rows A 522,858 / B 512,548 / C 517,494 / D 532,147 (2,085,047 in total; clock-present 517,353 / 507,023 / 511,118 / 527,083).
- **Scratch tables** (`scratch_build.json`): on the development months' clock-present rows the cross-fit v9 (nine-month models) scores
  Jan 199.94 vs 204.51 for the seasonal-fold v9 (−2.24%), Jul 243.24 vs 244.73 (−0.61%), Nov 192.32 vs 195.45 (−1.60%); December
  196.89 vs 196.58 for the eleven-month refit (+0.16%, the cross-fit lacks April and August). The eight other months contribute
  142,622–188,852 clock-present rows each. A second amendment corrected the builder's coverage check (finite predictions versus
  clock-present departures; February's catalog holds one row that sits in another month's raw file); no value changed.
- **matches_v11 month tables** (`tools/build_month_matches.py`, workstation): February–June and August–October built with the frozen
  1,111-stand table (matched 38.5% in June, whose surface archive has 20 days, to 64.9% in October).
- **Development** (`tools/run_adsb_v24.py --held-out MONTH --threads 16`, one month per worker, launched 16:10–16:14 UTC): the three
  January/July/November jobs first stopped at the baseline reproduction check because the declaration carried four-decimal baseline
  figures against a 1e-6 tolerance; the reproduced values are the v22 selection's own full-precision values, the declaration now carries
  them (third amendment) and the jobs were relaunched. December's job (whose recorded value was exact) ran on under the pre-amendment
  file; the amended fields do not enter its computation.

## Development results (`development_results.json`, `runs/adsb_v24/dev/`)

Baseline = the v21 recipe's development predictions: 330.3018 / 268.4237 / 191.6621 / 176.9893 (Jan / Jul / Nov / Dec), reproduced
exactly by the tool. Fold v9 → cross-fit v9 on the whole month: 343.87 → 341.20 / 288.13 → 286.89 / 219.88 → 217.13 / 209.36 → 209.65.

| candidate | Jan | Jul | Nov | Dec | mean | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| incumbent_repro (check) | 0.000% | 0.000% | 0.000% | −0.024% | −0.006% | (check) |
| v7x_r3 — v7 from more months alone | −0.294% | −0.143% | −0.486% | −0.046% | −0.242% | no (Dec interval crosses zero) |
| r12 — twelve-month residual alone | −0.131% | −0.159% | −0.713% | −0.221% | −0.306% | yes |
| v7x_r12x — both | −0.424% | −0.292% | −1.148% | −0.221% | −0.521% | yes |
| **v7x_r12x_lirf25 — selected** | **−0.445%** | **−0.304%** | **−1.137%** | **−0.253%** | **−0.535%** | yes |
| v7x_r12x_lirf50 | −0.447% | −0.310% | −1.108% | −0.273% | −0.534% | yes |
| v7x_r12x_m1332 (1,332-stand tables) | −0.441% | −0.308% | −1.116% | −0.192% | −0.514% | yes |
| v7x_r12x_season (season_cos) | −0.416% | −0.301% | −1.153% | −0.168% | −0.509% | yes |

Paired-day 95% intervals of the RMSE difference are below zero on every month for every eligible candidate (selected: [−2.60, −0.96] /
[−1.03, −0.61] / [−2.60, −1.79] / [−0.62, −0.28] s). Every airport improves in every month for the selected candidate (EGLL December
unchanged); LIRF 408.0 → 404.1 / 635.7 → 635.0 / 448.5 → 445.5 / 348.5 → 347.9. Ordinary rows (label ≤ 6 h): 196.66 → 194.18 /
260.84 → 259.99 / 171.61 → 169.17 / 174.05 → 173.59. Twelve-month residual rounds (median of the four months): r12x 2,882.

Reading. The twelve-month residual is worth −0.31% on its own and −0.22% on December; November gains most (−0.71%) because the
cross-fitted October and December sit next to it — the mechanism the learning curve predicted. The cross-fit v7 adds about −0.2%
where it replaces a seven- or nine-month fold model (January, July, November); on December it is slightly weaker than the shipped
eleven-month refit (+0.16% on clock rows) and the combined candidate still gains −0.25%, so December is the conservative read for the
January half of the ranking. The 1,332-stand tables and the season feature buy nothing; the 0.25 blend on unmatched LIRF rows is a
small, mostly positive hedge (Jan −0.19% on LIRF, Nov +0.03%) that the rule selected by 0.001 pp over 0.5 and 0.013 pp over none.
Expected official transfer: of the order of −0.2…−0.4% (0.5–1 s), the December read being the honest anchor.
