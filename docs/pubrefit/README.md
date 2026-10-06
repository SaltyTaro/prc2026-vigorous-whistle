# Lever 6: the published-data layers refitted on v66's chain — closed, no GO (6 October 2026)

Final-week campaign, fork F. 2025 development months only; no official score, no leaderboard, nothing declared, built or uploaded.
Tool `tools/run_pubrefit.py`; scratch `runs/pubrefit/`; screen record [`screen.json`](screen.json); log [`HANDOFF.md`](HANDOFF.md).

## Question

v66's development chain (and its ranking file) applies the extreme-day layer (`configs/scout_extreme_shift_v1.json`) and v57's variance
re-allocation (`configs/scout_var_alloc_v1.json`) with parameters fitted on the v46 development stack: in
`tools/run_ctx_ownpoints_chain.step_chain` the fitting months carry the old stack and only the scored month carries v66's model part.
Does refitting either layer on v66's own chain residuals help?

## Control

The pinned rebuild of v66's development chain (variance re-allocation LightGBM `deterministic`, `force_col_wise`; tools/lirf_own_pinned.py),
scored through `tools/run_ownpoints_weight.score_chain` (stack_resid_v3 refitted, intruder bound, identity extension, own-motion v1, v2):
**320.1258 / 251.0731 / 182.1762 / 163.7131 s** (January / July / November / December 2025). Its layer values and final values equal fork
B's pinned identity rebuild exactly (max |diff| 0.0); the stored unpinned files differ by up to 2.9 s per row (layers) and 0.024% per month.

## Variants (each month-blocked, as the original layers' governing runs)

- **va**: variance and late models and k refitted on v66's chain rows of the other months; extreme-day values as shipped.
- **ext**: extreme-day calibration (c0, beta) refitted on v66's chain days of the other months; the shipped variance re-allocation
  correction on top.
- **both**: `tools/scout_var_alloc.layered` on the all-v66 table (both layers refitted). Code check: the va arithmetic fed with both's
  extreme-day values reproduces both exactly (0.0).

## Screen (change of complete-month RMSE against the control; paired take-off-day 95% interval, %)

| variant | January | July | November | December | equal-footing mean | GO |
|---|---|---|---|---|---|---|
| va | -0.0017 [-0.006, +0.004] | **+0.0106** [+0.001, +0.019] | +0.0010 | -0.0124 [-0.023, -0.002] | -0.0012% | no |
| ext | -0.0007 [-0.005, +0.004] | +0.0010 [-0.003, +0.005] | **+0.0096** [+0.002, +0.018] | -0.0041 [-0.013, +0.006] | -0.0013% | no |
| both | -0.0021 [-0.005, +0.002] | **+0.0087** [-0.001, +0.017] | +0.0063 | -0.0127 [-0.030, +0.004] | -0.0020% | no |

GO bar (fixed in HANDOFF.md before any reading): equal-footing mean <= -0.02%, no month worse than +0.005%, every equal-footing interval
below zero. Every variant misses the mean by a factor of ten and worsens July or November. The twelve-month clause is not feasible (v66's
chain exists only for the four development months).

## Reading

The refitted k barely moves (0.19-0.25 against the shipped 0.22-0.25) and the extreme-day betas are near their old values: the layers'
parameters depend on the label-side day structure more than on the model part, so the v46-fitted parameters are already right for v66.
**Closed.** Nothing to build.
