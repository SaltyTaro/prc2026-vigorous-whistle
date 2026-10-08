# Second winter-season (W25) pass on v74 (`season_pass_v2`, 8 October 2026) — ACCEPTED: `w25b_5`; dry run v75 (not built)

Declaration `configs/season_pass_v2.json` (a26288c), committed before the five-seed bags were fitted; tool `tools/run_season_pass2.py`
(4c9dcab; `--build` locally, `--fit` on Azure arcw0, `--score` locally); results [`results_v1.json`](results_v1.json); exploration
[`../audit_v72/`](../audit_v72/) (aud13). No official score was read or used. A build under `runs/submissions` and any upload are the
user's word.

## What it is

v74's season pass (`docs/season_pass/`) used the raw ADS-B inputs only and moved January 2026 at EGLL / LIRF / LEBL. This second pass
adds the stacker, neighbour and queue inputs (151 features), is fitted on v74's residuals, and moves January 2026 at the airports
where aud13 read gains on both adjacent pairs: one LightGBM bag per airport (five seeds, 400 rounds, target clip(label - v74, ±1800)),
fitted on November + December 2025 and applied to the airport's non-fixed **January 2026** rows only:
new = max(v74 + clip(0.5 x bag, ±120 s), -12). July 2026, every other airport, the LIRF population and the guard rows keep v74.

## Not a clean test (disclosed in the declaration)

The airport sets and the input set were chosen after aud13's single-seed reading (EGLL / LIRF / LEBL lose on top of v74's pass: a double
correction; LFPG / LTFM lose).

## Result (complete-month RMSE change against v74's development values; each month predicted by the bag fitted on the other)

| candidate | November (from December) | December (from November) | mean | gain / squared change | Dec -> Jan 2025 (descriptive) | transfer (Jan 2026) |
|---|---:|---:|---:|---|---:|---|
| w25b_4 (EDDF, EDDM, EHAM, LSZH) | -0.0357 [-0.051, -0.024] | -0.0365 [-0.048, -0.025] | -0.0361 | 1.47 / 1.01 | +0.004 | 1.49 / 1.31 / 1.07 / 0.99 |
| **w25b_5** (+ LEMD) | -0.0441 [-0.064, -0.029] | -0.0397 [-0.054, -0.027] | **-0.0419** | 1.35 / 0.76 | +0.003 | ... LEMD 0.99 |

Both pass clauses 1-3; the rule selects **w25b_5** (lowest mean).

## Ranking dry run (version provisional)

`runs/season_pass2/rank/dry_v75/vigorous-whistle_v75.parquet`, SHA-256
`a227c51ca90589bf6966f49ae94e2113394ee8f508bee0794d20871a800a6036`: v74 with 65,230 January 2026 rows changed (LEMD 15,918, EHAM 14,972,
EDDF 14,968, EDDM 10,052, LSZH 9,320), squared change 5.24e6, max |change| 66 s, mean change +0.9 / -0.1 / -1.3 / -0.2 / -0.7 s
(EDDF / EDDM / EHAM / LEMD / LSZH). Independently recomputed from the bags (`docs/key_memory/verify_v75_v76.py`): equal on every row; no July row,
no LIRF row, no kind-3 row (the 383 LIRF population rows by the raw rule and both guard rows are inside kind 3), minimum -12, int32,
template order. Predictions copied back with matching SHA-256 (`runs/season_pass2/vm_preds.sha256`: dev 3ff0a835..., rank 9a071e31...).

**Expected effect:** about -0.035 x g s: -0.03 to -0.05 s at development's g = 0.8-1.35 (v74's W25 pass scored 1.70); +0.035 s if noise.

## Build (only at the user's word)

`docs/season_pass2/build_v75.py` (fill in USER_DECISION), then `tools/audit_export.py --version 75 --compare 74 --report
docs/export_audit/v75_audit.json`, then `tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v75.parquet --expected-sha256
a227c51ca90589bf6966f49ae94e2113394ee8f508bee0794d20871a800a6036 --calendar-day`; status `--report-dir docs/season_pass2`.
The flight-number memory layer (`docs/key_memory/`, dry run v76) is built on this file.
