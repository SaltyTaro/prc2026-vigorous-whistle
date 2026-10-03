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

## v3: own-aircraft turnaround / pre-gap features (configs/stack_resid_v3.json)

Declaration [`configs/stack_resid_v3.json`](../../configs/stack_resid_v3.json); results [`results_v3.json`](results_v3.json); exploration [`docs/dig/pregap.md`](../dig/pregap.md). **Accepted by the declared rule: True.** In-chain against the v62 development proxy (the stacking layer replaced, then the intruder bound, identity extension and own-motion bound re-evaluated month-blocked); LIRF keeps v2's value; the control chain reproduces the v61 proxy and the own-motion results exactly.

| Month | Control (v62 proxy) | Candidate | Change | Paired-day 95% (s) | Days better | Rows moved | Held by guard | RMS move (s) | Max move (s) |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01 | 320.8946 | 320.6647 | -0.0717% | [-0.443, -0.142] | 31/31 | 137,836 | 6,158 | 7.8 | 237 |
| 2025-07 | 252.1483 | 251.9446 | -0.0808% | [-0.274, -0.143] | 29/31 | 146,563 | 33,946 | 7.5 | 463 |
| 2025-11 | 183.0653 | 182.6074 | -0.2501% | [-0.612, -0.340] | 30/30 | 149,033 | 3,138 | 7.6 | 485 |
| 2025-12 | 164.8560 | 164.3171 | -0.3269% | [-0.729, -0.398] | 31/31 | 134,375 | 20,293 | 7.8 | 560 |

Equal-footing mean (January, July, December): **-0.1598%**.

Rule: 1_no_month_worse = True; 2_equal_footing_mean_at_most_-0.05pct = True; 3_equal_footing_intervals_below_zero = True; 4_transfer_guard_0.5x_2x = True.

Transfer guard after both support guards (ranking prediction RMS over development out-of-month RMS; the nine airports where the learner applies): EDDF 1.18, EDDM 0.98, EGLL 0.83, EHAM 1.13, LEBL 1.12, LEMD 1.08, LFPG 1.05, LSZH 1.20, LTFM 1.22; 14,637 ranking rows held.

Sensitivity, reported and not selectable (the same learner applied at LIRF too): -0.0666% / -0.0575% / -0.2865% / -0.3448%, equal-footing -0.1563%.

No twelve-month check exists (as for v2). No official score is used.

### v3 ranking dry run and build (3 October 2026)

`tools/run_stack_resid_v3.py --dry-run --version 63`: rematch chain file (c67f20d6) -> stack v3 (LIRF rows from the v60 stack file) -> intruder
bound + identity extension (`tools/run_intruder_identity.dry_run`, no v60 check) -> own-motion bound (`tools/run_own_motion.dry_run`). The same
chain from v2's stack file (cff778e0) reproduces v61 `fe137f09...` and the uploaded v62 `cba143f8...` byte for byte first.

- `runs/stack_resid/dry_v63/vigorous-whistle_v63.parquet`, **SHA-256 `4cff09459d044e9830f9a2a1b3a0be11d3c11bbe5c5926dbb131e7f05449846f`**
  (2,225,997 bytes); steps: stack `e2bd8bba...`, identity `543310e2...` (5 identity rows), own-motion 96 rows (v62: 155; the stack
  already lifts some).
- Against the uploaded v62: 233,886 rows differ, squared change 2.27e7, RMS 8.1 s, max +341 / -458 s, mean -0.14 s. No LIRF row, no LIRF
  population row and neither guard row changes. By airport (rows / squared change): EGLL 33,077 / 6.76e6, LFPG 34,893 / 4.33e6, LSZH 17,854 /
  2.53e6, EDDF 22,108 / 1.95e6, LEMD 24,333 / 1.79e6, EHAM 26,680 / 1.76e6, EDDM 18,656 / 1.64e6, LTFM 33,888 / 1.11e6, LEBL 22,397 / 0.86e6.
- Guards on the ranking: 16,785 rows held (14,637 outside LIRF). Coverage shift (`results_v3.json` `coverage_shift`): EDDF own-motion share
  0.87 (dev) -> 0.66 (ranking), guard share 26% dev out-of-month -> 11% (January 20%, July 5%); EGLL 0.51 -> 0.86, guard 3.5% -> 2.2%.
  The transfer guard reads EDDF 1.18, EGLL 0.83.
- **Built at the user's word** ("if you get anything build and keep it ready, we will upload it when slots reset"): `--build --version 63`
  reran the dry run into `runs/stack_resid/build_v63/`, byte-identical, and wrote `runs/submissions/vigorous-whistle_v63.parquet`
  (`4cff0945...`); export record [`export_v63.json`](export_v63.json); audit [`../export_audit/v63_audit.json`](../export_audit/v63_audit.json)
  (integrity clean: 344,841 unique ids matching the ranking departures, no NaN, none below -12 s). **Not uploaded.**
- Development chain proxy for later layers: `runs/stack_resid/dev_v63_proxy.parquet` (`1628a1d6...`, `tools/write_v63_dev_proxy.py`; columns
  `v62`, `v63`; reproduces this run's month RMSEs exactly). own_motion_v2 re-read on it (`docs/own_motion/results_v2_v63.json`): **not accepted
  on the v63 chain** (December's interval upper end +0.014%).
