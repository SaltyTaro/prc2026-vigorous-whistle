# Second guarded residual pass on v71 (`stack_second_v1`, 8 October 2026) — ACCEPTED: `sp_hicov7`

Declaration `configs/stack_second_v1.json` (b77fa7f), committed before the five-seed bags were fitted; tool `tools/run_stack_second.py`
(e5cfd9f; `--build` locally, `--fit` on Azure arcw0, `--score` locally); results [`results_v1.json`](results_v1.json); the exploration that
led here is in [`../audit_v71/`](../audit_v71/). No official score was read or used. A build under `runs/submissions` and any upload are
the user's word.

## What it is

v70's guarded stacker applies 0.65 of its out-of-month prediction; v71's neighbour layers come after it. A second LightGBM pass on the
same label-free inputs (the stacker's 79 columns) plus the neighbour, queue and clock columns learns the error that remains:
new = max(v71 + clip(0.5 x bag, ±120 s), -12), bag = mean of five seeds (7, 11-14), 400 rounds, fitted on non-fixed rows of the seven
high-coverage airports EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LSZH. LIRF (own stacker learner, fill hedges), LFPG and LTFM (little ADS-B)
keep v71.

## Not a clean test (disclosed in the declaration)

The airport set was chosen after a single-seed audit showed LIRF worsening January / July (+5.4e6 / +14.1e6 s²) and LFPG / LTFM mixed;
the single-seed readings of both declared candidates were known (eq -0.1171% / -0.0914%). No fresh-month check is possible: the stacker's
inputs exist only for the four development months. The stacker-weight round had shown the same direction (more weight helps November /
December, hurts January / July overall).

## Result (complete-month RMSE change against v71's development values, paired-day 95% interval)

| candidate | January | July | November | December | eq mean | transfer (ranking / dev RMS) |
|---|---:|---:|---:|---:|---:|---|
| **sp_hicov7** | -0.0612 [-0.217, -0.025] | -0.0656 [-0.094, -0.038] | -0.1508 [-0.227, -0.092] | -0.2305 [-0.301, -0.176] | **-0.1191** | EDDF 1.78, EDDM 1.04, EGLL 0.81, EHAM 1.49, LEBL 1.52, LEMD 0.96, LSZH 1.10 |
| sp_nolirf | -0.0368 [-0.170, -0.006] | -0.0590 [-0.090, -0.025] | -0.1452 | -0.1887 [-0.239, -0.139] | -0.0948 | EHAM 2.04 outside (keeps v71), others 0.83-1.90 |

Both pass clauses 1-4; the rule selects **sp_hicov7** (lower eq mean). The control (v71's development values) was rebuilt and matched
`docs/nbr_state/dark_results_v1.json`'s RMSEs exactly; the tables equal the audit's inputs on every checked column. Gain per unit squared
change on 2025: 2.99 / 1.54 / 3.03 / 3.93.

## Ranking dry run (version provisional)

`runs/stack_second/rank/dry_v72/vigorous-whistle_v72.parquet`, SHA-256
`4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80`: v71 with 201,983 rows changed at the seven airports (January
2026: EDDF 14,312, EDDM 8,608, EGLL 17,431, EHAM 13,565, LEBL 9,743, LEMD 14,763, LSZH 9,450; July 2026: 18,869 / 11,458 / 18,378 /
18,763 / 15,985 / 18,262 / 12,396), squared change 1.82e7, max |change| 120 s, mean -0.8 s (per airport -2.2..+2.1 s, RMS 6-13 s).
Independently checked: no LIRF / LFPG / LTFM row, no fixed row, guard rows 47,705 / 25,499 unchanged, minimum -12, int32, template order.
Expected about -0.12 to -0.37 s if the ranking months behave like 2025.

**Transfer caveat.** The ranking stacker is fitted on all four development months, the development stackers on three; the second pass
learned the out-of-month error of the latter. The per-airport RMS ratios (0.81-1.78) say the corrections have the development scale.

## Compute

Tables built locally (seconds); 50 LightGBM fits on arcw0 (about four minutes), predictions copied back with matching SHA-256
(`runs/stack_second/vm_preds.sha256`); arcw0 deallocated, all eight VMs deallocated.

## Build (only at the user's word)

`docs/stack_second/build_v72.py` (fill in USER_DECISION), then `tools/audit_export.py --version 72 --compare 71 --report
docs/export_audit/v72_audit.json`, then `tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v72.parquet
--expected-sha256 4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80 --calendar-day`.

## Inspection before the build (8 October ~03:50 UTC; development labels and label-free ranking inputs only)

Scripts: [`inspect_breakdown.py`](inspect_breakdown.py), [`inspect_published_days.py`](inspect_published_days.py) (both re-derive the
ranking values from `preds_rank.parquet` and reproduce the dry run exactly).

* **Spread, not a few rows.** Development gain (SSE removed) is positive in 27 of 28 airport-months; the one loss is EGLL July
  (+0.66e6 s²). Gain / squared change per airport over the four months: EDDF 5.0, EDDM 2.8, EGLL 0.9, EHAM 3.4, LEBL 1.9, LEMD 3.2,
  LSZH 4.3. The top 1% of moved rows by |change| carry 37% of the squared change and 42% of the gain; 10 development rows and 25
  ranking rows reach the 120 s cap.
* **Ranking changes look like development changes.** Quantiles of the change (0.1 / 1 / 50 / 99 / 99.9%): development -52.8 / -23.5 /
  -1.0 / 23.4 / 52.9 s, ranking -59.0 / -26.6 / -0.9 / 29.6 / 68.9 s; share positive 0.367 / 0.376; mean -1.05 / -0.71 s. By airport
  and row class (unmatched / dark / gate-traced) the RMS ratio is 0.80-1.95; the largest are EHAM unmatched 1.95 (2,142 rows) and EDDF
  dark / gate-traced 1.72 / 1.82 (EDDF's ranking coverage is a third of 2025's).
* **Published day totals (EUROCONTROL DLY_ALL_PRE, label-free on the ranking).** Implied day delay Σ max(a - value, 0) against the
  published figure at the seven airports: development moves toward the published figure on 60.7% of 862 airport-days (sum of |gap|
  -5.2%), ranking on 57.8% of 436 (-1.4%); per airport the direction agrees except EDDF (0.75 development, 0.50 ranking). The day
  channel is not the main mechanism: development days moved away from the published figure still gain (-26.7e6 against -44.6e6 toward).
* **Expected effect.** ΔRMSE ≈ -g x S / (2 N R) = -0.122 x g s with S = 1.82e7. No alignment at all (g = 0... the change pure noise):
  +0.12 s. Development alignment g = 1.5-3.9: -0.19 to -0.48 s. Official alignments of earlier stacker layers (v60 1.11, v69 1.82,
  v63 3.15): -0.14 to -0.38 s.
