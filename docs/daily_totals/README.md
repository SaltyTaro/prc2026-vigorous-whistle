# Published daily totals of off-block delay: screen, declared round and v48 (29 September 2026)

**State: v48 was uploaded at the user's word ("upload v48") at 13:16 UTC on 29 September 2026 and officially scores 222.6042 on all
344,841 rows: 1.3297 s (0.594%) better than v46 (223.9339). It is the clean model line's best. Never re-upload v1–v48.** Record: `official_result_v48.json`.

The squared change against v46 was 1.529e8, so the result corresponds to an alignment of 1.17: the moves pointed the right way and
were smaller than the truth, as in July 2025 (1.17). That figure describes the result. It is not used to rescale anything: no
score-derived work is done on this line.

## The source, and what it is

EUROCONTROL's Performance Review Unit publishes, per airport and day, the all-causes pre-departure delay reported by airport operators
(`All_Pre-Departure_Delay.xlsx`, release 2026-09-16, daily to 31 July 2026, SHA-256 `9ab45cc5...`; columns `FLT_DEP_IFR_2`,
`DLY_ALL_PRE_2`). Against the supplied 2025 clocks the figure is, per airport and off-block day,

    sum over departures of max(off-block - schedule, 0)

| Airports | Published minus own clocks, per flight, every month of 2025 |
|---|---|
| EDDF, EDDM, EGLL, LEBL, LEMD, LIRF, LTFM | 0.0 to 0.3 s (one month at LEBL −1.1 s) |
| LSZH | 0 to −3 s |
| EHAM | −1 to −7 s |
| LFPG | +2 to +5 s |

It is therefore a daily aggregate of the clock that the ranking file hides. The terms of the data are EUROCONTROL's performance-data
disclaimer (source to be named, non-commercial, no modification without permission), which is not an open licence. The user decided to use it ("go", after
"yes use it" on the portal's monthly aggregates); `configs/daily_totals_v1.json` records the words. No official score is used anywhere.

The ranking file holds the same set of departures as the published counts: per airport-day the row count over the published count is
1.000 in the median (0.982 to 1.034), as in 2025.

## Screen (`tools/screen_daily_totals.py`, `docs/clean_line_screens/daily_totals.json`)

The discrepancy between the published total and the total the v46 stack implies (off-block = takeoff − prediction) correlates 0.94–1.00
with the true discrepancy at seven airports. Two simple uses do not pay:

| Use | January | July | November | December |
|---|---|---|---|---|
| discrepancy spread over rows by error variance | −0.005% | +0.06% | +1.3% | +2.2% |
| learned residual model with the discrepancy as a feature | −0.53% | −0.32% | −1.67% | +1.14% |

The day's sum is dominated by a few rows with very large errors, so a common shift sized from the raw discrepancy harms. Rows with an
error above 1,800 s hold 54% of the squared error; the total sees 20%; rows that are nearly alone on their airport-day hold 3%.

## Method (`src/prc2026_daily/reconcile.py`)

- **Row model.** Each row draws a scenario from the recorded outcomes of comparable rows (class × schedule-interval band × prediction
  band, from other months): ordinary residual, off-block = schedule, off-block before schedule, label far below the prediction. Rows
  with widely spread scenarios get scenario weights whose mean equals the row's prediction, so the stack's own hedge stays the prior.
- **Day model.** Rows are independent given a day shift of the ordinary labels, prior N(0, τ²) with τ per airport from other months
  (5–22 s). The published figure has a Gaussian error plus two wide components.
- **Posterior.** By convolution on a 30 s grid. Every row that is not heavy moves by the posterior mean of the day shift plus a
  first-order tilt. Heavy rows (fill and one-day candidates: sd of their positive-delay error above 900 s) enter the day's arithmetic
  and keep their prediction.
- An airport-day is read when the rows cover at least 90% of the departures behind the published figure.

## Exploration that chose the configuration (disclosed in the declaration)

| Run | January | July | November | December | Reading |
|---|---|---|---|---|---|
| all rows, uniform scenario weights | −0.13% | +20.4% | −6.2% | +27.5% | heavy rows moved to their cell's mixture |
| scenario weights matched to the prediction | −0.62% | +1.56% | −1.23% | +0.20% | nine airports better; the LIRF fill population unstable |
| **heavy rows keep their prediction (declared)** | **−0.102%** | **−0.322%** | **−0.318%** | **−0.403%** | every interval below zero |
| the same, τ × 1.5 | −0.116% | −0.424% | −0.326% | −0.372% | not sensitive |
| month prior from the month's own totals | −0.098% | −0.463% | −0.360% | −0.341% | worse on six of twelve months of the check; rejected |

Leaving the LIRF fill population out was decided after seeing its result. Two earlier population changes that passed development lost
on the leaderboard (v10, v43).

## Governing run (`docs/daily_totals/results_v1.json`)

Control: the v46 development stack. Fitting side of each month from the other three.

| Month | Control RMSE | Reconciled | Change | Paired-day 95% interval (s) | Alignment |
|---|---|---|---|---|---|
| January 2025 | 322.858 | 322.528 | −0.102% | [−0.73, −0.15] | 0.84 |
| July 2025 | 257.718 | 256.889 | −0.322% | [−1.06, −0.61] | 1.17 |
| November 2025 (reported) | 184.905 | 184.318 | −0.318% | [−0.80, −0.42] | 0.85 |
| December 2025 | 167.514 | 166.838 | −0.403% | [−1.11, −0.34] | 0.78 |

**Accepted**: no equal-footing month worse, mean −0.276%, every interval below zero. 37 of 40 airport-month cells are better (worse:
EHAM November +0.11% and December +0.46%, LEMD December +0.21%). The day shift alone gives −0.082 / −0.215 / −0.287 / −0.350%; the
tilt adds in every month.

The governing run repeats the configuration chosen in the exploration on the same months. It confirms arithmetic.

## Twelve-month check (`docs/daily_totals/check12_report.json`)

All twelve months of 2025 with another predictor (the out-of-sample v9 plus residual stage, v23's recipe), cells from the other eleven
months, the declared configuration.

| Month | Change | Month | Change |
|---|---|---|---|
| January | −0.136% | July | −0.393% |
| **February** | −0.765% | **August** | −0.162% |
| **March** | −0.243% | **September** | −0.124% |
| **April** | −0.317% | **October** | −0.321% |
| **May** | −0.289% | November | −0.202% |
| **June** | −0.305% | December | −0.346% |

Twelve of twelve months better, every paired-day interval below zero, mean −0.30%; 112 of 120 airport-month cells better. The eight
months in bold were not used in the exploration.

One correction was made to the check after its first run. The cross-fit tables leave out 14,092 departures of the eight non-recipe
months (0.7%, mean positive delay 66 minutes), which the published totals contain. On the tables as they are the check read seven of
eight months better (mean −0.16%). The run above enters those departures at their airport-month's median prediction; they are never
scored and never enter a cell.

## v48

`runs/submissions/vigorous-whistle_v48.parquet`, SHA-256 `714ce4257b9b224ee7720e7bcae5c9240bc829935a05aa541049314ab8e51e0e`
(`tools/export_daily_totals.py`, report `docs/daily_totals/export_v48.json`, audit `docs/export_audit/v48_audit.json`).
Base: the uploaded v46 (223.9339). Cells and scales from the four development months.

- 619 airport-days read; 648 rows lie outside a read airport-day and keep v46.
- 333,373 rows change: mean −7.1 s, mean absolute 13.8 s, 4,684 by more than 60 s, largest 818 s. No row of the LIRF fill population
  is moved by more than its day shift; heavy rows keep v46.
- The dry run and the export are byte-identical.

Mean day shift per airport-month (seconds; negative = predictions lowered):

| Airport | January 2026 | July 2026 |
|---|---|---|
| EDDF | +0.2 | −1.9 |
| EDDM | −2.2 | −5.7 |
| EGLL | −2.9 | −17.7 |
| EHAM | +0.2 | +0.9 |
| LEBL | −5.5 | −14.3 |
| LEMD | +2.6 | −4.2 |
| LFPG | +0.2 | −6.0 |
| LIRF | +10.2 | −5.8 |
| LSZH | +1.5 | +1.1 |
| LTFM | −30.2 | −17.7 |

| Case | Score |
|---|---|
| v46 today | 223.934 |
| the changes are right in full | 222.94 |
| at the development alignment (0.78 to 1.17) | 223.38 to 222.60 |
| the changes are noise | 224.92 |

### Relation to v47

v47 (monthly published taxi-out means, `docs/published_means/`) and v48 move the level of the same airport-months from two sources of
the same kind. They are alternatives on the same base, never layers of each other. The two sources agree in sign on v47's four cells;
v48 moves less there (EGLL July −17.7 s against −30.2 s, LEBL July −14.3 s against −22.7 s, EGLL January −2.9 s against −10.0 s) and
much less at LIRF in July (−5.8 s against −45.0 s), where every row is untracked and the model cannot tell a level shift from fills.

## Commands

```
.venv\Scripts\python.exe -X utf8 -B tools\screen_daily_totals.py --extract
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --score --declaration configs/daily_totals_v1.json
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --month 2025-03 --tag check12f --rows twelve --no-heavy-updates
.venv\Scripts\python.exe -X utf8 -B tools\export_daily_totals.py --version 48
```

The upload that was made:

```
.venv\Scripts\python.exe -X utf8 -B tools\prc_submit.py upload --file runs/submissions/vigorous-whistle_v48.parquet --expected-sha256 714ce4257b9b224ee7720e7bcae5c9240bc829935a05aa541049314ab8e51e0e --calendar-day
```

## Not done

- No layer in the final-file runner (`src/prc2026_final/stack.py`): it would need the published totals of the final file's period.
- The LIRF fill population and other heavy rows are not resolved by the totals; two months gained and two lost in the exploration.
- The monthly and the daily source are not combined. Three further uses of the published figures were explored after v48 and none
  passes: separate shifts for tracked and untracked rows, the monthly source as the centre of the day shifts, one tilt per
  airport-month for the heavy rows. Record: [`followups.md`](followups.md).
- Heavy rows moved by a tempered posterior (`configs/daily_totals_v2.json`, amended in `configs/daily_totals_v3.json`): every month
  read is better, the declared rule selected nothing (two intervals include zero). v49 and v50 were built on the user's word and are
  not uploaded; v50 is the recommended one. Record: [`heavy_rows.md`](heavy_rows.md).
