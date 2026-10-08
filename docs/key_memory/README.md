# Winter-season (W25) flight-number memory on v75 (`key_memory_v1`, 8 October 2026) — ACCEPTED: `km_9`; dry run v76 (not built)

Declaration `configs/key_memory_v1.json` (ccc9f91), committed before its governing run (and before season_pass_v2's result fixed the
base); tool `tools/run_key_memory.py` (light pandas, workstation); results [`results_v1.json`](results_v1.json); exploration
[`../audit_v72/`](../audit_v72/) (aud14, aud15, aud16). No official score was read or used. A build under `runs/submissions` and any upload
are the user's word.

## Finding

`FLIGHT_mvt` is not a model input. The 29 September screen ([`../clean_line_screens/README.md`](../clean_line_screens/README.md) section 1)
found that a flight number's residual repeats in November / December (correlation +0.040) but not in January / July (-0.01), read it as a
fold effect and closed the lever. Under the season mechanism (`docs/season_pass/`) it is W25 adjacency: on v74's development values the
residual of airport + flight number carries November -> December and December -> November, and every cross-season pair loses (aud14:
48 key / strength settings, eight keys).

## What it is

c = clip(0.25 x n / (n + 30) x mean of clip(label - base, ±1800) over the key's November + December 2025 non-fixed rows (labels below
20,000 s), ±120 s); new = max(base + c, -12) on non-fixed **January 2026** rows at the candidate's airports. Base = v75 (season_pass_v2's
selected file): development values = v74's plus w25b_5's out-of-month bags. July 2026, EGLL, the LIRF population and the guard rows keep v75.

## Not a clean test (disclosed in the declaration)

The key, the strength (beta 0.25, lambda 30) and the second candidate's airport set (without EGLL, which lost in both directions) were
chosen after the exploration.

## Result (complete-month RMSE change against v75's development values; each month corrected from the other's keys)

| candidate | November (from December) | December (from November) | mean | gain / squared change | Dec -> Jan 2025 (descriptive) | transfer (Jan 2026) |
|---|---:|---:|---:|---|---:|---|
| km_10 (all ten) | -0.0465 [-0.074, -0.026] | -0.0464 [-0.068, -0.023] | -0.0465 | 2.40 / 1.43 | +0.008 | 0.94-1.11 |
| **km_9** (without EGLL) | -0.0478 [-0.075, -0.028] | -0.0486 [-0.072, -0.024] | **-0.0482** | 2.90 / 1.71 | +0.006 | 0.94-1.11 |

Both pass clauses 1-3; the rule selects **km_9**.

## Ranking dry run (version provisional)

`runs/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet`, SHA-256
`a2ff395db4fdf22be86e1001f5dd97c8f27adc7b836fd7b4902690a576f9725c`: v75 with 106,965 January 2026 rows changed (LTFM 20,153, LFPG 16,186,
LEMD 13,088, EHAM 11,969, EDDF 11,831, LIRF 9,861 non-population, LEBL 9,748, EDDM 7,503, LSZH 6,626), squared change 2.28e6, max |change|
78 s, RMS 2-9 s per airport (LIRF 9.0, LFPG 6.1, LTFM 5.1). Independently recomputed by a separate implementation ([`verify_v75_v76.py`](verify_v75_v76.py)): 0 mismatches; no July row, no EGLL row, no kind-3 row (LIRF population, guard rows), minimum -12, int32, template order.

**Expected effect:** about -0.015 x g s: -0.03 to -0.045 s at development's g = 1.7-2.9; +0.015 s if noise.

## Build (only at the user's word)

`docs/key_memory/build_v76.py` (fill in USER_DECISION), then `tools/audit_export.py --version 76 --compare 75 --report
docs/export_audit/v76_audit.json`, then `tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v76.parquet --expected-sha256
a2ff395db4fdf22be86e1001f5dd97c8f27adc7b836fd7b4902690a576f9725c --calendar-day`; status `--report-dir docs/key_memory`.
