# Dig tunnel 2: residual ore test on the shipped chain (3 October 2026, 2025 only, nothing declared)

**Question.** Is anything still learnable in the shipped chain's development residuals, from any label-free input?

**Control.** `runs/scout/refit/ctl_v55m.parquet`, column `new`: the v55m model part, then v48, the extreme-day layer and v57. It
reproduces 322.0588 / 255.9946 / 184.0052 / 166.2473 s (January / July / November / December 2025; 672,428 rows).

**Screen.** `runs/dig/ore/screen.py` fits a month-blocked LightGBM on the clipped residual: label − new, clipped at ±3,600 s.

- Fitting uses three months and predicts the fourth.
- Early stopping uses an inner day split of the fitting months: every fifth day of the month.
- Settings are fixed: 31 leaves, minimum leaf 400, learning rate 0.03, L2 50, at most 800 trees.
- The prediction is applied at weights 0.25, 0.5 and 1.0, with the −12 s floor.
- Scores are complete months, extreme labels included.
- Intervals: paired take-off-day bootstrap of the RMSE difference in seconds, 2,000 replicates.
- January, July and December count as equal footing; November is supplementary.
- No official score, leaderboard, cloud or submission file was used.

**Features.** `runs/dig/ore/build_features.py` builds 436 label-free columns, in these groups:

| Group | What it holds |
|---|---|
| comp (41) | Stack members and their differences: v46-era components from `runs/lomo_dev`, v46…v55m from `runs/dev_model_part`, and the chain outputs. |
| pub (12) | Published day total per airport and predicted off-block day: the total, the count, the discrepancy against the chain and against v55m (total and per row), and the row's share of the predicted positive delay. |
| peer (10) | Airport-hour means of member disagreements, match share, the row's position in the day. |
| adsb (38) | Match, neighbour and v19 features. |
| tl_core (80) | Timeline bundle: identities and clocks, including 15 categorical columns (stand, destination, operator, …). |
| tl_traffic (151) | Timeline bundle traffic features. |
| tl_wx (61) | Timeline bundle weather features. |
| tl_nm (43) | Timeline bundle NM-clock features. |

Four arrival-at-stand occupancy columns are dropped from every reported run except the first exploratory one (the coordinator's
separate lever).

## Results (change of complete-month RMSE, %)

| Feature set | Weight | Jan [95% s] | Jul [95% s] | Nov | Dec [95% s] | Equal-footing mean |
|---|---|---|---|---|---|---|
| members' families (timeline + ADS-B), sanity | 0.5 | +0.031 | +0.068 | +0.007 | −0.003 | **+0.032** |
| everything | 0.5 | +0.015 [−0.07, +0.15] | −0.122 | −0.161 | −0.293 | −0.134 |
| everything except tl_core | 0.5 | −0.041 [−0.32, −0.04] | −0.298 [−0.94, −0.58] | −0.208 | −0.372 [−0.83, −0.39] | −0.237 |
| comp only | 0.5 | −0.052 [−0.41, −0.05] | −0.253 [−0.77, −0.51] | −0.185 | −0.319 [−0.73, −0.34] | −0.208 |
| **comp + pub** | 0.5 | −0.061 [−0.45, −0.08] | −0.316 [−1.00, −0.62] | −0.304 | −0.371 [−0.84, −0.40] | **−0.249** |
| comp + pub, seeds 11 / 12 | 0.5 | −0.060 / −0.055 | −0.320 / −0.338 | −0.287 / −0.299 | −0.395 / −0.377 | −0.258 / −0.257 |
| comp + pub, 3-seed bag | 0.25 | −0.039 [−0.27, −0.06] | −0.197 [−0.60, −0.40] | −0.178 | −0.229 [−0.49, −0.27] | −0.155 |
| comp + pub, 3-seed bag | 0.5 | −0.059 [−0.44, −0.08] | −0.329 [−1.03, −0.64] | −0.300 | −0.386 [−0.86, −0.42] | **−0.258** |
| comp + pub, single seed | 1.0 | −0.045 [−0.50, +0.04] | −0.376 | −0.364 | −0.453 | −0.291 |
| 3-seed bag, LIRF rows left at `new` | 0.5 | −0.068 [−0.46, −0.11] | −0.267 [−0.85, −0.53] | −0.201 | −0.220 [−0.53, −0.21] | −0.185 |
| comp without chain outputs + pub | 0.5 | −0.057 | −0.337 | −0.299 | −0.367 | −0.254 |
| comp without chain outputs (no pub) | 0.5 | −0.029 | −0.181 | −0.136 | −0.213 | −0.141 |
| v46-era components only | 0.5 | −0.018 [−0.21, +0.02] | −0.203 | −0.119 | −0.167 | −0.129 |

Leave-one-group-out from "everything", equal-footing mean at weight 0.5:

| Group left out | Equal-footing mean |
|---|---|
| comp | −0.005 |
| pub | −0.111 |
| peer | −0.143 |
| adsb | −0.134 |
| tl_core | **−0.237** (removing it helps) |
| tl_traffic | −0.160 |
| tl_wx | −0.136 |
| tl_nm | −0.146 |

The run with low-cardinality categoricals and numeric core added back to "everything except tl_core" gives −0.261.

## Reading

- **The ore is in the stack's own components plus the published day discrepancy.** Every input family the members were trained
  on gives about zero or worse: the sanity check is +0.03%.
- The high-cardinality categoricals overfit. Stand, destination and flight prefix take the most gain but cost January.
- The member disagreements carry the gain: component spread, lirf_net − lirf_tree, v55m − v46, nets − committee, stack − v9.
  The day discrepancy per row (published total against the chain's implied positive delay) and the row's share of it also matter.
- The chain's own outputs are not needed: comp without chain outputs + pub gives −0.254 against −0.249.
- The new members (v51m–v55m differences) add a little over the v46-era components alone (−0.141 against −0.129).
- It is stable across seeds. Every month improves at every weight from 0.25 to 0.5, and all three equal-footing intervals lie
  below zero.
- The changes are diffuse:
  - RMS move is 9–13 s and the largest move under 260 s.
  - The best ten rows hold 14–23% of the gain in July, November and December, and 38% in January (where the gain is small).
  - The gain spreads over LIRF_m, LTFM_u, LFPG_u, EGLL, LEMD and LIRF_u. LIRF population rows are untouched.
  - With every LIRF row left at the chain, the equal-footing mean is still −0.185% at weight 0.5.
- This differs from the 29 September learned-gate screen (v46 stack, non-LIRF, components only, which found nothing) in three
  ways: it includes LIRF non-population rows, it has the v51m–v55m members, and it adds the published-day discrepancy features.

**Verdict.** GO candidate by the stated bar: comp + pub at weight 0.5 (or 0.25) gives an equal-footing mean of −0.25% (−0.16% at
0.25). No month is worse, and three of three equal-footing intervals lie below zero. At the ranking's error level this is roughly
0.4–0.6 s if it transferred fully.

**Caveats before any declaration.**

- Component tables exist only for these four months, so there is no twelve-month check.
- The upstream OOF tables are not fully nested: January and July are 11-month LOMO, November and December "fair".
- Ranking components are twelve-month refits, so their disagreement patterns may differ from the development OOF ones. This is
  the stacking transfer risk.
- Applying it on the ranking needs the ranking value of every component used, and the same day-discrepancy definition (predicted
  off-block day, chain values).
- The weight is not chosen on the scored months beyond these three readings.

**Files** (scratch, `runs/dig/ore/`):

- `features.parquet`, `meta.parquet`, `groups.json`
- `res_*.json` and `pred_*.parquet` for every run
- `pred_comp_pub_bag3.parquet`, the bagged predictor
- `breakdown_*.json`
- `res_all_v1.json`, the first exploratory run with arrival-at-stand columns included: equal-footing mean −0.137 at 0.5,
  practically the same as without them.
