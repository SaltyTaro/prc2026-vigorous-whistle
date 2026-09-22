# adsb_v26: further CatBoost members for the v23 residual stage (22 September 2026)

Incumbent: `vigorous-whistle_v23` (official 238.8197; `configs/adsb_v25_v1.json`, candidate `mean_adj2_cat`). Declaration
`configs/adsb_v26_v1.json` at **10:07:04 UTC** (commit `7a8413b`), before any development fit at protocol.

## Why

adsb_v25 found the CatBoost residual (learning rate 0.05, cap 8,000) still improving at the cap on every inner fold of every month and on
the shipped refit (inner best iterations 7,898–8,000), and the row-wise mean with the weighted LightGBM residual worth −1.96% on the
residual's own scope and −0.131 s officially. The cheapest remaining item is therefore the same model with more effective capacity
(learning rate 0.10 at the same cap) and a second seed of it, combined with the shipped members.

## Design

- **New variants** (scope `matched_no_lirf`, feature set `nb`, target `label − cross-fit v9`, eleven training months, inner rule
  unchanged): `cat_lr10` (depth 8, learning rate 0.10, l2 3, cap 8,000, seed 20260915) and `cat_lr10_s2` (seed 20260916).
- **Inherited members** (no refit): `adj2` and `cat` — their adsb_v25 development residual columns on all twelve months and the shipped
  v23 models are hash-bound parents.
- **Candidates** (row-wise mean of members): `repro` = [adj2, cat] (check), `adj2_lr10`, `adj2_cat_lr10`, `adj2_lr10x2`,
  `adj2_cat_lr10x2`, `lr10x2`, `cat_lr10x2`.
- **Rule**: as adsb_v25 — not worse on any recipe month vs v23's development predictions (328.7524 / 267.5352 / 189.3526 / 176.3497),
  mean ≤ 0.9998, paired-day intervals below zero on touched months, pooled twelve-month scope RMSE strictly below the control's
  (the incumbent's mean of adj2 and cat: 117.309). Lowest mean ratio among eligible; otherwise keep v23.
- **Execution**: (month, variant) jobs — 24 in all — spread over arcw0–arcw7 with the tool's fit stage (`tools/run_adsb_v26.py`);
  each month is combined and scored on the workstation by the `--score` stage once both partials exist; `--select` at the end.
- **Ship**: new members refit on twelve months (inner rule); inherited members = the shipped v23 models; export
  `vigorous-whistle_v24.parquet`; independent verification; audit; the user's approval before upload.

## Execution

- Smoke test of the fit / score / select stages on subsampled rows before the declaration (disclosed; outputs discarded; the control's
  pooled scope reproduced 117.309 and `repro` equalled the incumbent).
- Eight workers started 10:08 UTC, 148 files (818 MB) synced to each, chains launched 10:18–10:22 UTC: wave 1 `cat_lr10` on
  Dec / Jan / Jul / Nov / Feb / Mar / Apr / May, wave 2 `cat_lr10` on Jun / Aug / Sep / Oct plus `cat_lr10_s2` on the recipe months,
  wave 3 `cat_lr10_s2` on the rest.

## Result: nothing selected; v23 remains the incumbent

All four recipe months are scored (both partials of each exist and are final). Versus v23's development predictions
(328.7524 / 267.5352 / 189.3526 / 176.3497):

| candidate | Jan | Jul | Nov | Dec | mean | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `repro` (check) | 0 rows differ | 0 | 0 | 0 | 0.0000% | check |
| `adj2_cat_lr10` | **+0.0003%** | −0.0102% | −0.0179% | −0.0242% | −0.0130% | no (January; mean) |
| `adj2_cat_lr10x2` | **+0.0004%** | −0.0116% | −0.0160% | −0.0522% | −0.0198% | no (January; mean) |
| `adj2_lr10` | **+0.0016%** | −0.0117% | −0.0361% | −0.0207% | −0.0167% | no (January; mean) |
| `adj2_lr10x2` | **+0.0007%** | −0.0139% | −0.0301% | −0.0617% | −0.0263% | no (January) |
| `cat_lr10x2` | **+0.0059%** | −0.0107% | **+0.0074%** | −0.0537% | −0.0128% | no |
| `lr10x2` | **+0.0080%** | −0.0148% | −0.0054% | −0.0730% | −0.0213% | no (January; two intervals cross zero) |

The declared rule requires ratio ≤ 1.0 on **every** recipe month. January is worse for every candidate, so `selected_candidate`
is null and **v23 (238.8197) remains the incumbent**. The closest is `adj2_lr10x2` (mean −0.0263%, three months improving with
paired-day intervals below zero, January worse by +0.0007% = 0.0023 s).

Why: at learning rate 0.10 the CatBoost residual converges before the cap (5,111–6,408 rounds against 7,898–8,000 at 0.05) and is
worse than the shipped member on its own twelve-month scope in 8 of the 11 months fitted (January +1.15 s is the outlier;
better only in March −0.30, April −0.23 and July −0.29). The second seed behaves the same. Reading: the shipped residual's
"still at the cap" signal meant it wanted **more small steps**, not bigger ones — a longer run at learning rate 0.05 is the
faithful follow-up, not a faster one.

Execution: 24 jobs were declared; 16 were fitted (every recipe month for both variants, plus `cat_lr10` on Feb/Mar/Apr/Jun/Aug/Sep/Oct
and May, which remains on arcw7's disk). Once January's two partials were final the outcome was determined, so the chains were cut
before the last wave, every running fit was allowed to finish, each worker was mirrored and deallocated, and no further VM time was
spent. The eight months without a `cat_lr10_s2` partial are not needed for the decision (the rule fails on a recipe month); they would
be needed only to evaluate the pooled-scope condition of a `_s2` candidate. No refit, export or upload was made.
