# LIRF unmatched-blend round (27 September 2026): nothing selected, v30's LIRF blends are kept

## Setup

- **Declaration:** [`configs/lirf_unmatched_blend_v1.json`](../../configs/lirf_unmatched_blend_v1.json), declared at 08:37:56 UTC
  (SHA-256 `d5fd3797…`). It was committed by the parallel session before any score.
- **Prior context:** the parallel session's scratch leave-one-month-out screen on the LIRF component table. It is disclosed in the
  declaration.
- **Tool:** `tools/run_lirf_blend_dev.py --score --select` → [`development_results.json`](development_results.json)
  (= `runs/lirf_blend/selection.json`).

v30's recipe (committee mix `c_e24x32_e32h` + `lctx_b100_up`) was rebuilt on the four recipe months. Only the
`lirf_nonpop_unmatched` (net, tree, v9) weights change; v30's are 0.35 / 0.25 / 0.40. No model was fitted.

**Reproduction:**

- v28, `c_e24x32_e32h` with v25's LIRF blends, and v27 + `lctx_b100_up` each match their recorded development RMSE to 1e-9 s.
- The control equals v30's development rows (`runs/error_map/dev_rows.parquet`) exactly: max |Δ| is 0.0 on every month.
- Control RMSE: January 325.6759, July 262.1126, November 185.4367, December 168.8039 s.
- Scope rows: 11,254 / 864 / 3,871 / 2,474.
- No candidate changes a row outside the scope.

## Results (ratio to v30 on all rows; paired-day 95% interval of the RMSE difference, s)

| candidate (net/tree/v9) | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---:|---:|---:|---:|---:|---:|
| `u50_00_50` | −0.0168% | −0.0184% | −0.0395% | −0.0168% | −0.0229% | 3/4 (Dec [−0.087, **+0.030**]) |
| `u60_00_40` | −0.0153% | −0.0285% | −0.0378% | −0.0289% | −0.0276% | 1/4 |
| `u60_10_30` | −0.0098% | −0.0289% | −0.0237% | −0.0321% | −0.0236% | 0/4 |
| `u70_00_30` | −0.0050% | −0.0343% | −0.0290% | −0.0326% | −0.0252% | 0/4 |

- Every candidate improves every month, and every mean clears the 0.02% bar.
- No candidate has all four intervals below zero, so the declared rule **selects nothing**. No v32 candidate is built.

**On the scope's own rows:**

- In January 2025, the dark-regime analogue for July 2026 LIRF, scope RMSE falls from 273.7 s to 272.8–273.4 s (−0.1 to −0.3%).
- In July, scope RMSE falls from 416.4 s to 403.7–409.7 s, but on only 864 rows.
- The gains shrink as the net weight rises in January. The heavier net weights gain more in July and December, with wider intervals.

**Reading.** The direction is consistent: dropping the tree table and raising the net helps on unmatched LIRF rows. The effect is
small against its day-to-day noise, however, and the unchanged rule rejects it. `u50_00_50` misses only on December, where the
upper bound is +0.030 s. Shipping it would be a post-hoc relaxation of the declared rule, which is the user's call and would have to
be recorded as such.
