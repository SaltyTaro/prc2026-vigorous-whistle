# Stack residual learner (`configs/stack_resid_v1.json`) — governing run

Accepted by the declared rule: **False**. Results: [`results_v1.json`](results_v1.json); transfer check: [`transfer_check.json`](transfer_check.json). Exploration: [`docs/dig/residual_ore.md`](../dig/residual_ore.md).

| Month | Control RMSE | Candidate | Change | Paired-day 95% (s) | Rows moved | RMS move (s) | Max move (s) |
|---|---|---|---|---|---|---|---|
| 2025-01 | 322.0588 | 321.8842 | -0.0542% | [-0.407, -0.068] | 153,641 | 9.1 | 202 |
| 2025-07 | 255.9946 | 255.1942 | -0.3127% | [-0.998, -0.603] | 190,354 | 12.2 | 190 |
| 2025-11 | 184.0052 | 183.4426 | -0.3058% | [-0.834, -0.343] | 162,276 | 8.7 | 235 |
| 2025-12 | 166.2473 | 165.6372 | -0.3670% | [-0.828, -0.402] | 165,583 | 8.6 | 229 |

Equal-footing mean (January, July, December): **-0.2446%**.

Rule: no_month_worse = True; equal_footing_mean_at_most_-0.05pct = True; equal_footing_intervals_below_zero = True; transfer_guard_0.5x_2x = False.

Transfer guard (ranking prediction RMS over development out-of-month RMS, by airport): EDDF 1.91, EDDM 0.96, EGLL 0.75, EHAM 3.44, LEBL 1.15, LEMD 0.92, LFPG 1.02, LIRF 1.34, LSZH 1.38, LTFM 1.01.

Sensitivity (reported, not selectable): ranking rows outside the development support of the top features left at the chain (13,725 rows): ratios EDDF 0.97, EDDM 0.79, EGLL 0.67, EHAM 1.12, LEBL 0.72, LEMD 0.77, LFPG 0.87, LIRF 1.06, LSZH 0.96, LTFM 0.81.

No twelve-month check exists (the component tables cover only the four development months). No official score is used.

## v2: support-guarded form (configs/stack_resid_v2.json)

Declaration [`configs/stack_resid_v2.json`](../../configs/stack_resid_v2.json); results [`results_v2.json`](results_v2.json). Accepted by the declared rule: **True**. The support guard is part of the candidate (development: range from the fitting months; ranking: range from all four months; 13,821 ranking rows held at the chain).

| Month | Control RMSE | Candidate | Change | Paired-day 95% (s) | Rows moved | Held by guard | RMS move (s) | Max move (s) |
|---|---|---|---|---|---|---|---|---|
| 2025-01 | 322.0588 | 321.9156 | -0.0445% | [-0.316, -0.060] | 148,942 | 4,699 | 7.7 | 195 |
| 2025-07 | 255.9946 | 255.6749 | -0.1249% | [-0.389, -0.256] | 158,912 | 31,442 | 9.4 | 129 |
| 2025-11 | 184.0052 | 183.6738 | -0.1801% | [-0.463, -0.224] | 160,337 | 1,939 | 7.6 | 162 |
| 2025-12 | 166.2473 | 165.7008 | -0.3287% | [-0.707, -0.377] | 146,334 | 19,249 | 7.5 | 193 |

Equal-footing mean (January, July, December): **-0.1660%**.

Rule: no_month_worse = True; equal_footing_mean_at_most_-0.05pct = True; equal_footing_intervals_below_zero = True; transfer_guard_0.5x_2x = True.

Transfer guard after the support guard (ranking prediction RMS over development out-of-month RMS, by airport): EDDF 1.26, EDDM 1.07, EGLL 0.75, EHAM 1.24, LEBL 1.11, LEMD 0.89, LFPG 1.02, LIRF 1.20, LSZH 1.30, LTFM 1.16.

No twelve-month check exists. No official score is used.

## v60 built and uploaded at the user's word ("Build + upload v60 now", 3 October 2026)

v60 = v59's chain without the intruder bound (`runs/rematch_nonm/rank/Y`, c67f20d6...) -> this guarded residual layer
(`runs/stack_resid/dry_v60`, cff778e0...) -> the intruder stand bound last (`tools/run_intruder_bound.py --build --version 60`; receipt
`docs/intruder_bound/export_v60.json`, audit `docs/export_audit/v60_audit.json`, clean). SHA-256
`30021455005a712c241fd57354c7a57c65ea8e81cd7a440e7175d0f27598b95f`; 296,843 rows differ from v59, RMS 9.1 s, squared change 2.84e7.

**Official: 218.6404 on all 344,841 rows (processed 09:47:43 UTC 3 October 2026), -0.2079 s against v59 (218.8483); the official best.**
Record [`official_result_v60.json`](official_result_v60.json). Squared-error gain 3.14e7 = 1.11 x the squared change. Nothing is re-tuned from
the score. Never re-upload v1-v60.
