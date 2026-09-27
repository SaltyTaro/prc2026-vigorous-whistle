# Fair development set (27 September 2026): v33 confirmed, no further weight selected

**Why this exists.** The shipped development values compare the two sides of the recipe on unequal footing.

- **v9 side:** v9 (4-fold cross-fit v7, 9 fitting months) and the residual stage (12-month leave-one-month-out, 11 months).
- **Fold side:** the committee, the context networks and the LIRF tables, fitted on 7 months for January/July and 9 for November.
- **December:** both sides are fitted on January–November.
- **Evidence it mattered:** v33's official gain (−0.374%) matched the December reading (−0.381%)
  ([`../outer_weight/README.md`](../outer_weight/README.md)).

**What was built.**

- **Declaration:** [`configs/fair_dev_v1.json`](../../configs/fair_dev_v1.json) (commit `4260a34`), plus the review
  [`declaration_addendum.md`](declaration_addendum.md).
- **Residual stage refitted on each fold's own fitting months:** `tools/run_fold_residual.py` on arcw0–7, 09:35–12:15 UTC. It covers adj2,
  cat and 16 rnb seeds, with training rows on cross-fit v9 as shipped and scored rows on the fold v9: seasonal fold v9, forward fold v9,
  and the January–November refit v9 for December.
- **End-to-end check:** the December fold's training set is identical to the shipped December fits. All 16 rnb seeds reproduce the
  shipped training histories exactly, and adj2 / cat reproduce the shipped inner best iterations and rounds exactly (3,369 and 7,991).
- **Rows:** `tools/run_fair_dev.py` → `runs/fair_dev/rows.parquet` ([`rows_summary.json`](rows_summary.json)). Its shipped path
  reproduces v30's development rows exactly on all four months, and every reproduction check is NaN-safe.

## Matched non-LIRF rows: v9 side (base) against the fold side (mix), RMSE in s

| month | base, shipped | base, fair | mix |
|---|---:|---:|---:|
| January | 122.41 | 128.03 | 129.59 |
| July | 153.97 | 155.91 | 157.57 |
| November | 115.77 | 119.02 | 120.66 |
| December | 113.60 | 113.47 | 112.41 |

The fair set removes 43–78% of the base's lead in January, July and November. December is unchanged.

## v33 on the fair set

- Against v30: January −0.085%, July −0.251%, November −0.312%, December −0.373%. Mean −0.255% (shipped set −0.264%).
- The weights are also confirmed directly: putting matched other rows back to 0.7 (`m70`) is worse in every month (mean +0.062%).

## Fair-set weight round — [`configs/fair_weights_v1.json`](../../configs/fair_weights_v1.json)

Declared before any fair row existed (commit `3568073`). Tool: `tools/run_fair_weights.py` → [`fair_weights_results.json`](fair_weights_results.json).
Control = v33, and its rebuild equals the fair v33 exactly.

| candidate | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---:|---:|---:|---:|---:|---:|
| u80 (unmatched other 0.8) | +0.010% | +0.036% | −0.040% | −0.067% | −0.015% | 1/4 |
| u90 | +0.043% | +0.112% | −0.046% | −0.086% | +0.006% | 0/4 |
| u100 | +0.099% | +0.228% | −0.016% | −0.059% | +0.063% | 0/4 |
| m50 (matched other 0.5) | −0.017% | −0.027% | −0.066% | +0.003% | −0.027% | 2/4 |
| m70 | +0.033% | +0.046% | +0.123% | +0.046% | +0.062% | 0/4 |
| lu30 (LIRF unmatched v9 0.30) | −0.012% | −0.009% | −0.018% | −0.016% | −0.014% | 3/4 |
| lu20 | −0.020% | −0.017% | −0.033% | −0.027% | −0.024% | 2/4 |
| lm05 (LIRF matched v9 0.05) | 0.000% | −0.003% | +0.006% | −0.036% | −0.008% | 1/3 |
| lu20_lm05 | −0.020% | −0.020% | −0.027% | −0.063% | −0.033% | 3/4 (Jul [−0.20, +0.07]) |
| u80_lu20_lm05 | −0.010% | +0.016% | −0.068% | −0.130% | −0.048% | 2/4 |

**Nothing is selected; v33 stays.**

- **Unmatched non-LTFM rows:** the fair set does not support a higher mix weight. January and July stay worse even on equal footing,
  so December's preference there is seasonal, not a provenance artifact.
- **LIRF:** less v9 now helps in every month (`lu20`, `lu30`, `lu20_lm05`), which reverses the shipped set's mixed picture. The effect
  is small (0.01–0.03%, about 0.02–0.07 s) and its intervals touch zero in one or two months.
- **Remaining gain:** the fold asymmetry's large payoff was v33 itself. What remains of it on these weights is at most about 0.05%.
