# After v48: three further uses of the published figures (29 September 2026) — none passes, nothing declared

**State: v48 (222.6042) stands as the clean line's best. No new file was built.** The three ideas below were explored on 2025 data
against the declared procedure of `configs/daily_totals_v1.json` as control. None reaches the standing rule (no equal-footing month
worse, mean at least 0.02% better, every equal-footing paired-day interval below zero), so none was declared and no governing run was
made. Figures: [`followups.json`](followups.json), [`followup_heavy_LIRF.json`](followup_heavy_LIRF.json). Tool:
`tools/screen_daily_followups.py`. No official score is used anywhere.

The code that produced v48 was extended, not changed: with the new options off, the governing run's four months reproduce bit for bit
and a dry run of the export is byte-identical to the uploaded v48 (SHA-256 `714ce425...`).

## 1. Separate day shifts for tracked and untracked rows

One published total per airport-day, two shifts: one for the rows with an ADS-B surface match, one for the others, jointly Gaussian
with the airport's scales and correlation from the fitting months (`run_daily_totals.py --components tracked`,
`reconcile.solve_day(shift_cov=...)`).

The premise holds. The untracked rows' day shift is two to four times wider at the airports with good coverage, and the two shifts
are correlated:

| Airport | tracked (s) | untracked (s) | correlation |
|---|---|---|---|
| EDDF | 5.7 | 16.0 | 0.67 |
| EDDM | 4.4 | 19.6 | 0.64 |
| EHAM | 2.9 | 17.1 | 0.89 |
| LSZH | 7.1 | 23.5 | 0.51 |
| EGLL | 16.5 | 19.3 | 0.89 |
| LFPG | 18.4 | 20.0 | 0.71 |
| LIRF | 20.0 | 28.0 | 0.96 |

The result does not follow:

| Read | Months better | Mean change | Note |
|---|---|---|---|
| development stack, four months | 3 of 4 | −0.0015% | January −0.001%, July −0.017%, November −0.002%, **December +0.015%**; 20 cells better, 18 worse |
| twelve months, other predictor | 8 of 12 | −0.006% | November and December +0.011%; 52 cells better, 52 worse |

Splitting a day's shift differently between the two groups re-distributes an estimate that is itself noisy. **Closed.**

## 2. The monthly taxi-out source as the centre of the day shifts

In the airport-months that pass the gate of `configs/published_means_v2.json` the monthly source measures the level well: over the
sixteen eligible cells of the twelve-month table the level error is 13.5 s (root mean square) before and 2.8 s after its shift; over
the eight cells of the development stack 12.6 s and 3.5 s. The daily procedure does not fix that level by itself (12.6 s before,
11.9 s after, on the same eight cells).

Two forms were read (`centres_stack.json`, `centres_twelve.json` hold the cells):

| Form | Development stack (Jan / Jul / Nov / Dec) | Twelve months |
|---|---|---|
| prior mean of the day shift, sd from the day-to-day spread (`--centres`) | −0.002 / −0.010 / **+0.012** / +0.001% | 10 of 12 better, mean −0.034%, of which April −0.329% |
| the control's day shifts recentred on the monthly shift | −0.003 / −0.011 / +0.001 / −0.029% | 9 of 12 better, April −0.294% |
| recentred, EGLL only (the source's noise is 0.7 s there) | −0.003 / −0.020 / 0.000 / −0.029%, the three equal-footing intervals below zero | 7 of 9 better, none beyond −0.01% |

- As a prior mean the source changes little: the day-to-day spread of the shift (5–19 s) is nearly the whole of its spread, so the
  daily total overrides the centre.
- Recentring gains where the level error is large and costs a little where it is small. The mean over the equal-footing months is
  −0.014% (−0.017% for EGLL alone), below the 0.02% the rule asks for, and July's interval includes zero.
- One cell carries the twelve-month read: EHAM in April 2025, where the predictions were 40 s low, the monthly source said 42 s and
  the daily procedure had moved 2 s. Without April the mean is −0.007%.
- LIRF, the one cell where it matters most on the scored months, went the wrong way in its only 2025 case (July 2025, +0.30% of the
  cell's squared error). There the month's level error sits in the hedged rows, which the day shift does not move.

**Not declared.** On the scored months the form would move EGLL (both months) and LEBL (July) by 7–13 s more than v48 did, worth
about 0.03 s if the source holds, and LIRF in July by about 20–40 s, worth about 0.15 s if right and costing the same if wrong.

## 3. One tilt per airport-month for the heavy rows

Heavy rows (the LIRF fill population and similar) keep their prediction in v48. The idea: one common tilt of their scenario weights
per airport-month, read from the month's 31 totals jointly, instead of row by row and day by day (which was a coin flip in the
exploration before v48).

| LIRF month | Heavy rows | Best possible with labels (tilt / one common shift) | Label-free estimate |
|---|---|---|---|
| January 2025 | 53 | −1.8% / −2.7% of their squared error | **+8.6%** |
| July 2025 | 351 | 0.0% / −1.9% | **+4.0%** |
| November 2025 | 64 | −7.3% / −9.9% | −1.5% |
| December 2025 | 91 | 0.0% / −0.6% | **+1.5%** |

Even with the labels one parameter per month buys little, and the label-free estimate is worse than doing nothing in three of four
months. A regression of each day's discrepancy on the day's population stake recovers the month's true deviation with a correlation
of −0.4 to −0.5. **Closed.** Heavy-row updates outside the LIRF population were also re-read from the exploration before v48: −0.05
to +0.07% of a month's squared error with no consistent sign.

## What the scored months show, label-free

At LIRF in July 2026 both published sources say that the predictions are too high in aggregate: the monthly source by 45 s per
flight (its noise is 4 s), the daily totals by 110 hours of delay over the month (35 s per late flight). v48 moved the ordinary rows
by −6 s only, because 708 heavy rows can absorb the excess (351 in July 2025, where the same signature was the hedged population
being over-predicted by 43 hours in total). 2025 offers no procedure that turns this into a reliable row-level change: see 2 and 3.

## Commands

```
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --month 2025-07 --tag fu1_tracked --no-heavy-updates --components tracked
.venv\Scripts\python.exe -X utf8 -B tools\screen_daily_followups.py --centres
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --month 2025-07 --tag fu2_centre --no-heavy-updates --centres docs/daily_totals/centres_stack.json
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --report --tag fu2_centre --control daily_totals_v1
.venv\Scripts\python.exe -X utf8 -B tools\screen_daily_followups.py --heavy LIRF
.venv\Scripts\python.exe -X utf8 -B tools\screen_daily_followups.py --summary
```

A month run holds about 1.3 GB; on the 16 GB workstation at most four run at once.
