# LIRF own-points design (6 October 2026): NO GO, closed

Final-week campaign, lever 2 (fork B). The user approved the plan ("yes start on them now") and a GPU round if this design passed ("yes you
can the GPU"). Declaration [`configs/lirf_own_design_v1.json`](../../configs/lirf_own_design_v1.json) (07d1279, 05:25 UTC, before any fit).
Report [`design_report.json`](design_report.json) (`tools/lirf_own_design_report.py`). No GPU was launched for fits, no development round was declared, and
nothing was exported or uploaded.

## Question

v65 / v66 added the 15 own-aircraft turnaround columns (`data/external/adsblol/features_v19own`) to the non-LIRF context-network member.
LIRF's two networks never saw them. Those are the round-1 LIRF network C (`ens_ctx_lirf`) and the masked LIRF member M (`lm25_32`, v53-v55).
LIRF non-population rows hold about 32% of ordinary development squared error. Does either network gain from these columns?

## Design

- **Fits.** 4 seeds each of `ctx_lirf_own` (C plus the 15 columns, round-1 tool, 4 threads) and `lm25_32own` (M plus the columns, unmasked, 8 threads).
  - Each used the same tool, settings, seeds and fit partition as the saved design runs it is compared with: seasonal fit months, October monitored.
  - They ran on arcw0-3 (CPU) from 05:26 to 10:35 UTC.
  - Two refitted controls reproduce the saved runs exactly (`ctx_lirf` s1, `lm25_32` s21: identical `model.pt`, October max difference 0.0 s).
- **Readings.** All on LIRF non-population rows (`tools/lirf_own_predict.py`, workstation CPU). Every saved model reproduces its earlier transfer file to 2e-6 s RMS.
  - October.
  - January 2025: LIRF is naturally dark, with own motion on 0 rows.
  - July 2025 natural.
  - July 2025 thinned to July 2026's coverage (attempt a13), with the own columns set to their unidentified state.
  - December 2025.
- **Composition.** As shipped: N = (1 − w) C + w M, with w = 0.40 on matched rows, 0.55 on unmatched rows and 0.70 in dark readings.
- **Items.** The own variant replaces its network only on rows with own ground motion seen (`p_first_mv`). Dark months are therefore unchanged by construction.
- **Rule.** L2 = 0.65 v9 + 0.35 N. October must be at most −0.20% with its paired-day interval below zero. Natural July and December and both dark readings must be at most +0.01%.

## Result (L2 change against the shipped composition; paired-day 95% interval in seconds)

| item | October | Jan 2025 (dark) | July natural | July thinned | December | GO |
|---|---|---|---|---|---|---|
| M_routed | −0.096% [−0.71, +0.01] | 0 | −0.043% | 0 | −0.136% [−0.84, +0.13] | no (October) |
| C_routed | +0.044% | 0 | +0.032% | 0 | −0.079% | no |
| CM_routed | −0.028% | 0 | +0.017% | 0 | −0.181% | no |
| M_all (reported) | −0.038% | −0.199% | +0.006% | **+0.415% [+0.09, +3.50]** | −0.054% | — |

- Standalone RMSE on own-route rows gives no consistent gain:
  - M against Mown: October 348.13 / 348.21 s, July 427.92 / 430.27 s, December 271.76 / 270.46 s.
  - C against Cown: October 343.21 / 347.84 s, July 431.34 / 432.46 s, December 273.68 / 272.67 s.
- Unrouted, the own columns hurt the dark July regime (M_all +0.42%, interval above zero), so the routing was necessary.
- No item reaches the bar. **The lever is closed on evidence.** At LIRF the own points are sparse and already reflected in the ADS-B match features, which agrees with tunnel 18's small LIRF stacker gain.

## Pipeline finding kept for later rounds

- **The chain does not rerun exactly.** The variance re-allocation's LightGBM (`src/prc2026_scout/var_alloc.PARAMS`) picks row- or column-wise histograms by a timing test, so reruns under load differ from stored proxies:
  - Unpinned rerun of v65: −0.0075% to +0.0048% per month.
  - Pinned rerun of v66: −0.009% to +0.024% per month.
- **Pinned reruns are exact.** Adding `deterministic` and `force_col_wise` makes reruns byte-identical (two pinned identity rebuilds are equal).
- **The scorer itself reproduces exactly.** On v66's own chain files the stage-2 scorer reproduces the stored v66 proxy with max difference 0.0.
- **Why it matters.** Comparisons at the 0.02% bar should use a pinned control and candidate (`tools/lirf_own_pinned.py`, `tools/run_lirf_own_dev.py`).
  - Stored proxies carry noise of the bar's own size.
  - The prepared tools (`tools/run_lirf_own_dev.py`, `tools/export_lirf_own.py`) are committed but unused.

## Compute

- arcw0, arcw2 and arcw3 were deallocated at 09:15 UTC after their hash-checked mirror. arcw1 was deallocated at 10:36 UTC by the detached watcher `tools/lirf_own_watch_arcw1.sh`.
- One on-demand g6e.2xlarge head launch in us-east-2 at 09:55 UTC failed for capacity before creating anything. No GPU cost; no cluster exists.
