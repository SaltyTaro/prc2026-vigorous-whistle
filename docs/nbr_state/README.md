# Neighbour ADS-B ground state for unmatched departures (`nbr_state_v1`, 7 October 2026) — ACCEPTED, w75

Declaration `configs/nbr_state_v1.json` (bc8388c), committed before this run; tool `tools/run_nbr_state.py`; results
`results_v1.json`. No official score was read or used. A build under `runs/submissions` and any upload are the user's word.

## Mechanism

v70's residuals are correlated between departures close in time at an airport (lag-1 0.11-0.26, `docs/quick_screens/README.md`).
On ADS-B-matched departures the chain's value differs from the non-ADS-B v9 prediction by what ADS-B (and the later models)
see of the current ground state. For each unmatched departure:

- m = mean of (value - v9) over matched departures at the same airport within 900 s (at least 3, the row itself excluded);
- x = m - the airport-day mean of (value - v9) over matched departures (the day level stays with the published-total reconciliation);
- new = max(value + clip(w · b_airport · x, ±120 s), -12), b_airport a least-squares slope through the origin of the clipped
  residual (±1,800 s) on x, fitted on development months; applied last, after v70's step 11.

LIRF population rows, the guard rows and every matched row keep their value. LTFM has no matched
neighbours and is untouched.

## Result (complete-month RMSE change, paired-day 95% interval, seed 20261004)

| candidate | Jan 2025 | Jul 2025 | Nov 2025 | Dec 2025 | eq mean | Aug 2025 (fresh) | Sep 2025 (fresh) | Oct 2025 (fresh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w50 | -0.0104 [-0.037, -0.004] | -0.0509 [-0.079, -0.027] | -0.0217 | -0.0464 [-0.066, -0.028] | -0.0359 | -0.0186 [-0.027, -0.014] | -0.0171 [-0.034, -0.009] | -0.0451 [-0.063, -0.031] |
| **w75** | -0.0139 [-0.049, -0.006] | -0.0537 [-0.089, -0.023] | -0.0227 | -0.0609 [-0.090, -0.035] | **-0.0428** | **-0.0255** [-0.037, -0.019] | **-0.0240** [-0.048, -0.013] | **-0.0630** [-0.089, -0.043] |

Clauses 1-3 (development) pass for both; **clause 4 (binding, August-October 2025 never read for this lever, on the older twelve-month
predictor `rows12` 'stack' with the cross-fit v9 of `scratch/v24`, slopes from the four development months) passes for both, every
interval below zero.** Selection (lowest eq mean): **w75**.

Clause 5 (per airport, ranking RMS / development RMS, w75): EDDF 0.79, EDDM 1.07, EGLL 0.77, **EHAM 2.07 (outside: keeps v70)**,
LEBL 0.92, LEMD 0.51, LFPG 1.03, LIRF 0.90, LSZH 1.28. EHAM's ratio comes from January 2026's disruption days (2-7 January).

Gain per unit squared change on 2025: 3.74 / 0.92 / 0.68 / 2.49 (development), 3.65 / 5.77 / 4.87 (fresh).

## Ranking dry run (version provisional)

`runs/nbr_state/rank/dry_v71/vigorous-whistle_v71.parquet`, SHA-256 `7a0b1431eef1146d7afd40c728bea7af6b90a411e257ca277564866e410b84ed`:
v70 with 35,887 unmatched rows changed (January 2026: EDDF 2,830, EDDM 267, EGLL 2,138, LEBL 500, LEMD 552, LFPG 7,291, LIRF 1,943,
LSZH 552; July 2026: EDDF 4,520, EDDM 458, EGLL 503, LEBL 617, LEMD 4,810, LFPG 8,075, LIRF 63, LSZH 768), squared change 7.30e6,
max |change| 120 s, per-airport mean change -0.2..+0.8 s. Independently checked: template order and int32 dtype as v70; no `fixed`
scope row, no matched row, no EHAM or LTFM row moved; guard rows 47,705 / 25,499 unchanged; minimum value -12. Expected about
-0.05 to -0.28 s if the ranking months behave like 2025 (about -0.12 s at the 2025 median gain ratio).

## Exploration disclosed in the declaration

Before declaring, the development months were read for windows 900 / 1,800 s, weights 0.5 / 0.75 / 1.0, focal rows unmatched / all,
baselines v9 / base / mix / committee / nets (only v9 works: the others already carry ADS-B), a model-free gate taxi excess (weak),
gate-only neighbours (weak), with / without intercepts, raw / day-demeaned. The fresh months were not read before the declaration.

## Implementation notes

- rows12 and scratch/v24 flag the LIRF population only in the four development months. In the fresh months the population is held
  fixed by its raw rule (LIRF, no NM origin, no airport aircraft type), which reproduces the development flag on every December row
  and on July's 337 (plus 18 rows the development flag leaves out).
- 6,708 fresh rows12 rows have no cross-fit v9 (rows12 filled their 'stack' with a median); they can be corrected but are never
  neighbours.

## Build (only at the user's word)

`docs/nbr_state/build_v71.py` (set CHOICE "state" or "state+dark" and the user's words) copies the verified dry-run bytes to
`runs/submissions/vigorous-whistle_v71.parquet`, validates against the template and writes the receipt; then
`tools/audit_export.py --version 71 --report docs/export_audit/v71_audit.json` and
`tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v71.parquet --expected-sha256 <sha> --calendar-day`.

## Extension to dark rows: `nbr_dark_v1` (7 October 2026) — ACCEPTED

Declaration `configs/nbr_dark_v1.json` (56ca0a9), after the user's "ok run it"; tool `tools/run_nbr_dark.py`; results
`dark_results_v1.json`. On top of nbr_state_v1 w75 (rebuilt and checked against `results_v1.json` exactly), matched rows whose
pushback ADS-B did not trace (gate False) move by 0.75 x B x x, with ONE slope B for all airports (0.1172; month-blocked
0.109-0.131), cap 120 s.

| | Jan 2025 | Jul 2025 | Nov 2025 | Dec 2025 | eq mean | Aug 2025 (fresh) | Sep 2025 (fresh) | Oct 2025 (fresh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dark_pool_w75 vs nbr_state w75 | -0.0055 [-0.020, -0.002] | -0.0167 [-0.032, -0.002] | -0.0346 | -0.0387 [-0.056, -0.025] | -0.0203 | -0.0043 [-0.0066, -0.0029] | -0.0087 [-0.017, -0.005] | -0.0216 [-0.038, -0.011] |

Clauses 1-3 pass (clause 2 by 0.0003 points; the candidate was chosen after the reads listed in the declaration); **the binding
fresh clause passes, every interval below zero**; transfer 0.77-1.28 at every airport (none excluded). Gain per unit squared change
on 2025: 1.16-3.93.

Dry run `runs/nbr_state/rank/dry_v72/vigorous-whistle_v72.parquet`, SHA-256
`2dfb3b9f53284c7ae3a02882d19cfd61b19116b1b00504176551bcae8b82e6ba` = v70 + both layers: 124,159 dark rows changed against the v71 dry
run (squared change 4.09e6, max |change| 96 s, mean +0.03 s; no gate-traced, unmatched, fixed or guard row moved), 160,046 rows and
11.39e6 against v70. Expected about -0.06 s on top of nbr_state (about -0.18 s for both against v70, if 2025-like). It can be uploaded
directly as v71 (CHOICE "state+dark").
