# Six-month refit of the variance re-allocation layer (`configs/sixmonth_varalloc_v1.json`): NOT accepted, no candidate 7

Declared (60044a6) before any fit; adaptive choice disclosed there (written after candidates 4 / 6 and after the late learners' six-month
readings). Tool `tools/run_sixmonth_varalloc.py read`; results [`varalloc_results_v1.json`](varalloc_results_v1.json); VM log
[`varalloc/read.log`](varalloc/read.log). No official score used; nothing uploaded.

**What was read.** The early chain's step 5 (`configs/scout_var_alloc_v1.json`: two LightGBMs for log squared residual and P(late), slope k)
is fitted on the four development months. P(pool) is that fit for any pool of months (models on the pool; k from the pool's out-of-month
weights). Tier B February / June 2025 rows (lane M's rows read) enter at step 5's input of the Tier B chain (its step-2 base, v48 and the
extreme-day layer, reproduced exactly).

**Control (exact).** P(four development months) gives k = 0.24186632360173135, equal to the shipped k, and its values on the Tier B file equal
the Tier B chain's step 5 (`runs/febjun_tierb/chain/work/steps/Y`) on every February and June row.

| held month | a | b | change [95% paired-day interval] | dSSE (s^2) | rows moved |
|---|---|---|---|---:|---:|
| January 2025 | Jul + Nov + Dec | + Feb + Jun 2025 | +0.0017% [-0.0023, +0.0104] | +5.49e5 | 152,589 |
| July 2025 | Jan + Nov + Dec | + Feb + Jun 2025 | -0.0011% [-0.0059, +0.0040] | -2.71e5 | 189,013 |
| February 2025 | four development months (shipped) | + June 2025 | **+0.0343% [+0.0031, +0.0747]** | +3.19e6 | 50,283 |
| June 2025 | four development months (shipped) | + February 2025 | +0.0075% [-0.0010, +0.0165] | +9.53e5 | 35,162 |

Summed dSSE +4.42e6. **Rule: not accepted** on all three clauses (sum positive; February and June above zero; February and June worse than
+0.005%). February's loss is mostly LTFM (+2.59e6); June's mostly LIRF (+1.15e6). k falls from 0.242 to 0.208-0.238 when Tier B rows join a
pool. Unlike the late learners, the variance re-allocation does not gain from the Tier B months (their cheaper model part changes the
residual variance the layer learns).

**Decision.** No candidate 7; **candidate 6 (`a7ac1f60...`) stays the recommendation.** Compute: arcw1 only, 13:28-13:36 UTC 12 October
(27 pool fits, 121 s); inputs uploaded and hash-checked (52 files), results mirrored and hash-checked, arcw1 deallocated; all eight VMs
deallocated.
