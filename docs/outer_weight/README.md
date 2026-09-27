# Outer weight by scope (27 September 2026): selected `ltfm100_m60`, exported as v33 (not uploaded)

**Question.** v30's recipe gives the committee mix (tree committee + context networks at share 0.75) weight 0.7 against v9 + the
residual stage on every non-LIRF row. That 0.7 dates from the v17 recipe, before the residual network and the context networks. Is it still
right, and is it right everywhere?

**Why the question was open.**

- The development months compare the two sides on unequal footing. v9 (cross-fit v7) and the residual stage are 12-month
  leave-one-month-out, fitted on 11 months. The committee and the networks score January and July from the seasonal fold (7 fitting
  months) and November from the forward fold (9).
- So January, July and November lean toward the base. December is the only recipe month where both sides are fitted on the same months
  (January–November), as the ranking's 12-month refits are.
- Every round that moves weight toward the mix pays this bias in three of four months.

**Declaration:** [`configs/outer_weight_scope_v1.json`](../../configs/outer_weight_scope_v1.json), declared at 09:06:30 UTC (commit
`c7a4b6a`). It was declared after a scratch screen on the same four months, and the screen is disclosed in it: the candidate scores were
seen before the declaration. The unchanged rule decides.

- **Candidates:** five fixed ones, outer weight on LTFM rows (0.9 / 1.0) and on the other ADS-B-matched non-LIRF rows (0.7 / 0.65 / 0.6).
  Unmatched non-LTFM rows stay 0.7: in the screen, a higher weight there helps December (EGLL, LFPG) but hurts January and July.
- **Tool:** `tools/run_outer_weight_dev.py --score --select` → [`development_results.json`](development_results.json).
- **Reproductions:** the three recorded reproductions (v28, `c_e24x32_e32h` with v25's LIRF blends, v27 + `lctx_b100_up`) and v30's
  development rows each match to 1e-9 s or better, and the scope rebuild at 0.7 equals the control exactly.

## Results (ratio to v30 on all rows; paired-day 95% interval of the RMSE difference, s)

| candidate (LTFM / matched other) | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---:|---:|---:|---:|---:|---:|
| `ltfm90` (0.9 / 0.7) | −0.027% | −0.136% | −0.108% | −0.254% | −0.131% | 4/4 |
| `ltfm100` (1.0 / 0.7) | −0.026% | −0.173% | −0.124% | −0.337% | −0.165% | 3/4 (Jan [−0.25, +0.01]) |
| `ltfm90_m65` (0.9 / 0.65) | −0.067% | −0.174% | −0.219% | −0.282% | −0.186% | 4/4 |
| `ltfm100_m65` (1.0 / 0.65) | −0.066% | −0.211% | −0.235% | −0.365% | −0.219% | 4/4 |
| **`ltfm100_m60` (1.0 / 0.6)** | **−0.101%** | **−0.243%** | **−0.331%** | **−0.381%** | **−0.264%** | **4/4 — selected** |

`ltfm100_m60` intervals: January [−0.68, −0.14], July [−0.85, −0.44], November [−0.85, −0.39], December [−0.92, −0.42] s.

**Where the gain comes from.**

- **LTFM** (no ADS-B at all, where the context networks gained most) is better at 1.0 in every month, including the three months that
  favour the base. December LTFM RMSE falls from 193.6 to 190.0 s.
- **The matched-row weight 0.6** is supported in every month. In December, the unbiased month, it is small: matched rows 110.7 → 110.5 s.
- December (−0.38%) is the best proxy for the ranking's equal-footing refits.

## Export v33 (27 September, 09:08 UTC), not uploaded

- **Tool:** `tools/export_v33.py` → `runs/submissions/vigorous-whistle_v33.parquet`, SHA-256
  `ceb055981846cdb025f0f0efed45df588bca2f21b56a0a024aec77ce426bed48` ([`export_v33.json`](export_v33.json)).
- **Build, per the declared ship clause:**
  1. Rebuild the recipe on the final-file runner's dry-run detail and check that the rebuild at 0.7 equals the dry run's v30 (max |Δ| 0.0).
  2. Add the delta to v31's floats, which round exactly to the uploaded v31.
  3. Floor at −12 s.
- **What changed:** 230,410 rows. LTFM: 46,358 rows, mean −0.95 s, RMS 29.2 s. Matched non-LIRF rows elsewhere: 184,052 rows, mean
  −0.48 s, RMS 6.3 s. Max |Δ| 638 s. 0 rows outside the scope; every LIRF row and every unmatched non-LTFM row equals v31.
  v32's LIRF change (officially worse) is not included.
- **Independent check:** rebuilt from the recorded v30 detail's own components instead of the dry run, 193 rows differ by 1 s (float
  noise in the dry run, 0.031 s RMS), none by more.
- **Audit:** [`../export_audit/v33_audit.json`](../export_audit/v33_audit.json) is clean (row set, types, floor) with v31's shape. Against
  v31: RMS 11.9 s, LTFM 29.2 s, other airports 4–7 s, LIRF 0.
- **Upload:** only on the user's decision.

## v33 official result (27 September 2026): **228.7246, new best**

- **Upload:** approved by the user ("upload it"). The fourth upload of the UTC calendar day. Processed at 09:10:32 UTC on all 344,841
  rows. Records: [`official_result_v33.json`](official_result_v33.json) and [`leaderboard_after_v33.txt`](leaderboard_after_v33.txt).
- **Against v31 (229.5832):** −0.8586 s, **−0.374%**. The squared-error change is −1.36e8.
- **Against development:** the development mean was −0.264%, and December, the equal-footing month, gave −0.381%. The ranking year
  matches December almost exactly, while recent rounds transferred 6–28% of their development gain. This supports the fold-asymmetry
  reading: January, July and November understate anything that moves weight from v9 + residual toward the committee mix.
- Never re-upload v1–v33.
