# Heavy rows and the published daily totals (`configs/daily_totals_v2.json`, `configs/daily_totals_v3.json`, 29 September 2026)

**State: two files are built on the user's word ("build it"), neither is uploaded. v50 is the recommended one. They are alternatives:
at most one of them may be uploaded, and only at the user's word.**

| File | What it is | SHA-256 | Against v48 |
|---|---|---|---|
| v49 | the cautious strength (power 0.10) as declared in `configs/daily_totals_v2.json` | `de96516e6eea43ddc44c3adf810756bbfd3449ef44a80af40ffc76e92b73bdaf` | 1,090 heavy rows differ, squared change 17.1e6 |
| **v50** | the same with the amendment of `configs/daily_totals_v3.json`: a heavy row that holds an NM clock keeps its value | `13a8c3a0844700ee5497ca712303cd29d1316a2ba39f354c0b4da6613be66325` | 442 heavy rows differ, squared change 10.0e6 |

The declared rule selected nothing in either round: every month read is better than v48's procedure, and the paired-day intervals of
July and December include zero by a small margin. Both files are the user's override of that rule. Results:
[`results_v2.json`](results_v2.json), [`results_v3.json`](results_v3.json). No official score is used.

## What changes against v48

v48 leaves the heavy rows at their prediction. A heavy row is one whose positive off-block delay is widely uncertain (sd above 900 s):
almost always a LIRF departure without a Network Manager record and aircraft type, which is either a schedule fill (off-block =
schedule, label = schedule interval) or an ordinary departure with a delay of hours. This round moves them:

- **Tempered posterior.** For each outcome of the row, the likelihood of the day's published total without the row enters to the
  power 0.10 or 0.25. The exact posterior (power 1) was a coin flip before v48.
- **What the total can tell.** A delay of hours adds itself to the day's total; a fill adds nothing. The total tells a delay from
  none. It cannot tell a fill from an off-block before schedule or a one-day label. The row's prior keeps its cell's proportions
  inside the outcomes with a delay and inside those without; only the weight between the two groups is matched to the prediction,
  and the outcomes without delay count as the schedule interval.
- **Bounds.** A heavy row moves between the mean of its delay outcomes and its schedule interval, never beyond. A row predicted
  within 120 s of its schedule interval (a declared fill) keeps its prediction. A row predicted above its schedule interval is not
  moved. Heavy rows never receive the day shift.

## Governing run (development stack, control = v48's procedure)

| Month | Power 0.10 | Interval (s) | Power 0.25 | Interval (s) |
|---|---|---|---|---|
| January 2025 | −0.160% | [−1.48, −0.08] | −0.301% | [−2.69, −0.17] |
| July 2025 | −0.129% | [−0.69, **+0.01**] | −0.261% | [−1.49, **+0.14**] |
| November 2025 (reported) | −0.194% | [−0.84, −0.07] | −0.350% | [−1.44, −0.16] |
| December 2025 | −0.215% | [−0.89, **+0.04**] | −0.378% | [−1.48, **+0.10**] |
| Mean, equal-footing months | −0.168% | | −0.313% | |

**Rule: neither candidate passes** (every equal-footing interval must lie below zero). All of the change is on the LIRF fill
population; heavy rows at other airports move by seconds. At the power 0.25 the fills gain 149e6 of squared error, the rows with a
real delay 82e6, one-day rows are not moved.

## Twelve months with the population in every month (`heavy_rows_twelve.json`)

The cross-fit tables of the v48 check lack the fill population in eight months. `tools/screen_daily_followups.py --population-table`
enters those 950 rows with a leave-one-month-out v9 mixture (1,487 population rows in all; the month's fill share runs from 0.32 to
0.76). LIRF's airport-days, control = v48's procedure on the same table. This table was part of the exploration.

| Form | Months better | Mean change | Intervals below zero |
|---|---|---|---|
| **declared form, power 0.10** | 12 of 12 | −0.310% | 10 of 12 |
| **declared form, power 0.25** | 12 of 12 | −0.519% | 9 of 12 |
| declared form, power 1.00 | 11 of 12 | −0.810% | 7 of 12 (July +0.50%) |
| declared form, power 0.25, each day reading another day's discrepancy | 7 of 12 | −0.137% | 0 of 12 |
| exponential-tilt prior, power 0.05 / 0.10 / 0.25 | 12 of 12 each | −0.207 / −0.339 / −0.536% | 9 / 8 / 6 |

About half of the population's gain survives when a day reads another day's discrepancy: an ordinary discrepancy, whichever day,
excludes a delay of many hours. The true totals are better than the other day's on the population rows in 12 of 12 months.

## How often a large move is right (2025)

| Table, power | Moves of 100–300 s | 300–1,000 s | over 1,000 s |
|---|---|---|---|
| development stack, 0.10 | 63 of 84 (75%) | 16 of 17 | 3 of 3 |
| development stack, 0.25 | 90 of 121 (74%) | 52 of 72 (72%) | 8 of 9 |
| twelve months, 0.10 | 110 of 137 (80%) | 26 of 33 (79%) | 11 of 11 |
| twelve months, 0.25 | 216 of 307 (70%) | 83 of 101 (82%) | 21 of 25 |

Moves up (towards the schedule interval: no unexplained delay in the day's total) are right about nine times in ten. Moves down
(the day's total holds a delay that the other rows do not explain) are right about six times in ten.

## The ranking rows, label-free (`dry_run_v49_temper_10.json`, `dry_run_v49_temper_25.json`)

Dry runs on the uploaded v46 values; nothing was written under `runs/submissions`.

| | Power 0.10 | Power 0.25 |
|---|---|---|
| heavy rows that differ from v48 | 1,090 (497 up, 593 down) | 1,103 |
| rows moved by more than 300 s | 15 (8 up, 7 down) | 49 (23 up, 26 down) |
| their share of the squared change | 83% | 94% |
| largest move | 2,617 s | 4,941 s |
| sum of squared change against v48 | 17.1e6 | 76.5e6 |
| of which LIRF July 2026 | 15.6e6 | 70.7e6 |
| pinned rows (unchanged) | 13 | 13 |
| heavy rows beyond their schedule interval | 0 | 0 |

Size of the stake, from v48's 222.6042 (arithmetic only; no score enters the file):

| Case | Power 0.10 | Power 0.25 |
|---|---|---|
| the moves are noise | 222.72 | 223.10 |
| the moves are the truth's size | 222.49 | 222.11 |
| the moves are right and as under-sized as in 2025 (alignment 3.5 / 2.0) | 221.94 | 221.10 |
| the moves are as wrong as v43's were (alignment −0.46) | 222.82 | 223.56 |

## Why the prior was changed during the exploration

With the first prior (exponential tilt, as for the other rows) the largest moves on the ranking rows, +1,500 to +2,500 s, fell on
rows predicted at or above their schedule interval, among them v40's fill pins. That prior gives such a row about a fifth of one-day
weight, and "no delay in the total" then moves it up. The look used no label and no score. The declared form removes the effect by
construction; on the twelve-month table it costs little (−0.310% against −0.339% at the power 0.10).

## v49, what its check found, and the amendment (v50)

v49 was built from the cautious strength as declared (`export_v49.json`, audit `docs/export_audit/v49_audit.json`, clean;
byte-identical to its dry run). An independent label-free check of the file against the supplied ranking columns found that one row
carries 40% of its squared change against v48 (6.9e6 of 17.1e6):

| Flight | Schedule interval | NM clocks before takeoff | v46 = v48 | v49 |
|---|---|---|---|---|

The row is not a fill candidate. It holds an NM record and its prediction follows that clock. It sits in the class that carries the
fill population's cells, as 17 of that class's 400 ranking rows and 25 of its 562 development rows do. In development those rows are
worth 0.7e6 of squared error against 124e6 on the population, and none of them has a clock one day early: the move has no analogue
in 2025.

**Amendment** (`configs/daily_totals_v3.json`, declared at 20:32 UTC before any run of it): a heavy row that holds an NM off-block
clock keeps its prediction. It still enters the day's arithmetic. No score and no label of the scored months is part of the reason.

| Read (power 0.10) | As declared in v2 | Amended (v3) |
|---|---|---|
| January 2025 | −0.160% | −0.159%, interval [−1.47, −0.07] |
| July 2025 | −0.129% | −0.129%, interval [−0.68, +0.02] |
| November 2025 (reported) | −0.194% | −0.190%, interval [−0.83, −0.06] |
| December 2025 | −0.215% | −0.215%, interval [−0.89, +0.04] |
| twelve months, LIRF | 12 of 12, −0.310% | 12 of 12, −0.305% (`heavy_rows_twelve_amended.json`) |

The rule is not passed, as before. **v50** is the amended procedure on the uploaded v46 values (`export_v50.json`, audit
`docs/export_audit/v50_audit.json`, clean):

| Rows of v50 against v48 | Rows that differ | Moved by more than 60 s | Largest | Squared change |
|---|---|---|---|---|
| LIRF fill population (no NM clock, no aircraft type) | 333 of 383 | 86 | 1,918 s | 9.14e6 |
| rows that hold an NM clock (none is heavy-updated) | 15,597 | 4 | 118 s | 0.88e6 |
| other rows | 249 | 0 | 46 s | 0.02e6 |

- 14 rows move by more than 300 s, all fill candidates without a clock: 7 up, 7 down.
- The nine rows that v40 pinned at their schedule interval are unchanged. No heavy row lies beyond its schedule interval.
- 671 heavy rows keep their value (pins and rows with a clock).
- With the options off the final code reproduces the uploaded v48 byte for byte, and with v2's settings v49.

Size of the stake of v50, from v48's 222.6042 (arithmetic only; no score enters the file):

| Case | Score |
|---|---|
| the moves are noise | 222.67 |
| the moves are the truth's size | 222.54 |
| the moves are right and as under-sized as in 2025 (alignment 3.5 to 4.2) | 222.21 to 222.12 |
| the moves are as wrong as v43's were (alignment −0.46) | 222.73 |

By direction (`heavy_rows_direction.json`): in 2025 the moves up gained on 95 of 125 rows (development stack) and 260 of 305
(twelve months); the moves down gained on 74 of 110 and 108 of 191, with their gain concentrated in a few months. On the ranking
rows v50 has 74 moves up (squared 6.1e6) and 66 moves down (3.0e6), 58 of the latter in July 2026, where the published totals
exceed what the predictions imply.

## Standing

- This is the third change of the LIRF fill population. The first two (v10, v43) passed development and lost on the leaderboard.
  Those were classifiers fitted on 2025. This one reads the published totals of the scored days themselves, and its direct
  analogue before v48 (the exact posterior) failed in development, which is why v48 left the rows alone.
- With the new options off, the code reproduces v1's governing run bit for bit and the uploaded v48 byte for byte.
- One file at most should be uploaded: choosing between strengths, or between v49 and v50, by their scores would be learning from
  the ranking. The recommendation is v50.

## Commands

```
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --score --declaration configs/daily_totals_v2.json
.venv\Scripts\python.exe -X utf8 -B tools\screen_daily_followups.py --population-table
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --month 2025-03 --tag p12L_b10 --rows runs/daily_totals/rows12p.parquet --airports LIRF --heavy-temper 0.1 --heavy-prior bounded --pin 120
.venv\Scripts\python.exe -X utf8 -B tools\screen_daily_followups.py --compare runs/daily_totals/rows12p.parquet p12L_control p12L_b10 p12L_b25
.venv\Scripts\python.exe -X utf8 -B tools\export_daily_totals.py --dry-run DIR --version 49 --declaration configs/daily_totals_v2.json --candidate temper_10 --user-override "TEXT"
.venv\Scripts\python.exe -X utf8 -B tools\run_daily_totals.py --score --declaration configs/daily_totals_v3.json
.venv\Scripts\python.exe -X utf8 -B tools\export_daily_totals.py --version 50 --declaration configs/daily_totals_v3.json --candidate temper_10 --user-override "TEXT"
```

Upload of v50, only at the user's word:

```
.venv\Scripts\python.exe -X utf8 -B tools\prc_submit.py upload --file runs/submissions/vigorous-whistle_v50.parquet --expected-sha256 13a8c3a0844700ee5497ca712303cd29d1316a2ba39f354c0b4da6613be66325 --calendar-day
```
