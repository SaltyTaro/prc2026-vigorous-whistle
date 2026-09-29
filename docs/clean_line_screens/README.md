# Clean-line screens (29 September 2026): flight-number history and raw-trace features — both closed

Descriptive screens on the development months only, run while looking for model levers on the clean line (v45 = 224.2398). Nothing is
declared, selected or exported; no official score is used. Rows: `runs/lomo_dev/rows.parquet` (January and July 2025 on the
eleven-month footing, November and December fair), residual = label − v33, labels below 20,000 s.

## 1. Flight-number history (`tools/screen_flight_number_residual.py`)

`FLIGHT_mvt` is not a model input (only its prefix and its shape are). Question: does the stack's residual repeat for the same flight
number? For each month, the statistic is the mean clipped residual of the same key in the other three months.

| key | coverage (n ≥ 5) | correlation with the month's residual (Jan / Jul / Nov / Dec) | shrunken correction, change in squared error |
|---|---|---|---|
| airport + flight number | 58–95% | −0.010 / −0.014 / +0.040 / +0.040 | +0.6% to +2.5% |
| airport + ATC callsign | 51–94% | −0.012 / −0.009 / +0.040 / +0.041 | +0.6% to +2.4% |
| airport + operator + destination | 90–97% | −0.012 / −0.016 / +0.002 / +0.014 | +0.6% to +1.5% |
| airport + operator + stand | 89–92% | −0.014 / −0.016 / +0.003 / +0.007 | +0.7% to +2.7% |
| airport + stand + runway (known to the models) | 97–98% | +0.009 / +0.010 / +0.030 / +0.026 | +0.6% to +2.1% |

The residual does not persist per key. The new keys read like the control key the models already hold, so the small November/December
slope is a shared fold effect. A correction raises the squared error at every shrinkage in every month. **Closed.**

## 2. Raw-trace features for matched rows (`tools/screen_trace_features.py`, `tools/screen_trace_residual.py`)

Question: does the raw adsb.lol ground trace hold information beyond the 38 ADS-B features, enough to justify a trace encoder?

- 38 new features per matched departure from its raw ground points (`data/external/adsblol/surface/`): path and straight-line length,
  stops (count, longest, first, last, where and when), standing time in the last 5/10/15 minutes, distance to the last ground point at
  seven times before takeoff, speed quantiles, sampling gaps. Tables: `runs/clean_line_screens/trace/features_<month>.parquet`.
- Month-blocked residual model (LightGBM, fitted on three months, read on the fourth) on ADS-B-matched non-LIRF rows. Control inputs:
  the stack's components, the match table's columns and the v19 features (35). Candidate: control plus the 38 trace features.

| held-out month | scope rows | scope share of the month's squared error | control, scope squared error | with trace features | trace minus control [paired-day 95%] |
|---|---|---|---|---|---|
| January | 76,081 | 20.3% | +0.64% | +0.42% | −0.22% [−0.64, +0.25] |
| July | 87,418 | 16.9% | −0.68% | −0.86% | −0.17% [−0.54, +0.23] |
| November | 107,423 | 33.2% | −0.45% | −1.13% | −0.68% [−1.05, −0.37] |
| December | 107,106 | 29.0% | −1.29% | −1.98% | −0.68% [−1.23, −0.22] |

(Full predicted residual added; [`trace_screen.json`](trace_screen.json) also holds the half-weight reading.)

- The trace features add 0.2–0.7% of the matched rows' squared error, which is **0.02% (January, July) to 0.10% (November, December)
  of the month's RMSE**. The January and July intervals include zero.
- No single feature stands out (the largest carries 2–3% of the model's gain).
- The control model's own gain is largest at EDDM (−6% and −12% of its matched squared error in November and December). EDDM's coverage
  rose from 37% to 97% matched during autumn 2025, so the fold models saw few matched EDDM rows. The twelve-month refits hold those rows;
  this is a development artefact, not a ranking lever.

A trace encoder would compete for the same 0.02–0.10%. **Closed as a lever of size; not worth a GPU build.**

## 3. Learned gate over the stack's components (`tools/screen_learned_gate.py`, after v46)

Every layer shipped since v35 is a hand-made disagreement gate, and only a linear stacking regression had been tested
(−0.069%, `docs/disagreement/README.md`). Question: does a learned gate find more? A small LightGBM (15 leaves, 300 rounds) is fitted on
three months' residuals of the **full v46 stack** and read on the fourth. Rows: non-LIRF, outside the callsign and queue layers.
`components` = the stack value, v9, base, committee, networks, mix, their three differences, match flag, airport. `context` adds hour,
weekday, the NM clock interval, schedule and EOBT intervals and the no-NM flag.

Change in month RMSE against the v46 stack, by the weight given to the predicted residual:

| held-out month | components 0.25 | components 0.5 | components 1.0 | context 0.5 |
|---|---|---|---|---|
| January (eleven-month footing) | −0.028% | −0.044% | −0.039% | −0.028% |
| July (eleven-month footing) | −0.006% | +0.007% | +0.093% | +0.014% |
| November | −0.043% | −0.052% | +0.034% | −0.063% |
| December (fair) | −0.042% | −0.044% | +0.074% | +0.015% |

- At its best weight the learned gate is worth 0.03–0.04% in three months and nothing in July. The full predicted residual worsens three
  of four months. The context features add nothing.
- LTFM worsens in every month under both feature sets (+0.03% to +0.60% of its own RMSE); the small gains sit at the covered airports.
- The EDDM artefact of section 2 shows again in November and December.

Out of month, the v46 stack's residual is not predictable from its own components or from the clock and schedule intervals.
**Closed.** ([`learned_gate_screen.json`](learned_gate_screen.json))

## 4. Published airport-month taxi-out aggregates as a level check (`tools/screen_published_means.py`)

**Status: a screen only. These are aggregates of the hidden target in the scored months.

Source: EUROCONTROL Performance Review Unit, "Taxi-Out Additional Time" workbook, version 15 September 2026, period to 31 July 2026
(`https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx`, retrieved 29 September 2026 09:25 UTC,
SHA-256 `82d119ed…` in [`published_means.json`](published_means.json); kept under `data/external/eurocontrol_aiu/`, not in the
repository). Per airport and month it gives the departures, the departures with valid data and, for those with a reference time, the
total reference and additional taxi-out minutes. P = their sum per flight is the published mean taxi-out of the valid flights.

**What the figures are, against the 2025 labels** (mean label over labels in (0, 7200] s minus P, twelve months):

| airport | offset, mean ± sd (s) | note |
|---|---|---|
| EGLL | −0.8 ± 0.6 | the published mean is the label mean |
| LEBL | 6.2 ± 2.4 | |
| EHAM | 3.2 ± 4.0 | |
| LIRF | 9.3 ± 4.0 | the labels above two hours (fills, one-day rows) are outside the figure |
| LEMD | 8.2 ± 6.4 | |
| EDDF, LSZH, EDDM, LFPG | 0–15 in summer, 20–165 in winter | the unit leaves out 5–20% of winter departures (valid share 0.80–0.95); the offset grows with that share, so they are the slow (de-iced) flights |
| LTFM | 40 ± 71 | February 2025 alone is 264 s off |

The share left out is itself informative: in January 2026 it is 31% at EDDM, 20% at LSZH, 13% at EDDF and 10% at LFPG, against 20%,
10%, 5% and 9% in January 2025.

**Development read** (the v46 stack; mapping fitted on the other eleven months of 2025; every prediction in (0, 7200] of the
airport-month shifted by −w × the estimated level error):

| month | w = 0.5, all airports | w = 1.0, all airports | estimate against the true level error |
|---|---|---|---|
| January | +0.032% | +0.139% | rms difference 16 s (EDDF 42 estimated, −3 true: the winter line extrapolates) |
| July | −0.075% | −0.080% | 7 s |
| November | −0.030% | +0.017% | 9 s |
| December | −0.022% | +0.020% | 8 s |

In 2025 the stack's level errors are small (mostly under 13 s), about the size of the method's own noise, so the shift does not pay in
development. Where the mapping is tight the estimate is right: EGLL 7.1 / 8.4 / 18.0 / 7.2 s estimated against 6.9 / 7.5 / 18.3 / 7.2 true.

**Scored months, implied level error of the uploaded v46** (no label involved):

| airport | January 2026 | July 2026 | reliability |
|---|---|---|---|
| EGLL | +10 s | **+30 s** | tight mapping |
| LEBL | +6 s | **+23 s** | tight |
| EHAM | +7 s | −6 s | tight |
| LIRF | +2 s | **+53 s** | the estimate ran 5–14 s high in development |
| LEMD | +27 s | +1 s | moderate |
| LFPG | +13 s | −2 s | loose (sd 18 s) |
| EDDF | +39 s | +9 s | January extrapolates the winter line |
| LSZH | +31 s | −1 s | January extrapolates |
| EDDM | +2 s | +6 s | January extrapolates |
| LTFM | +32 s | +16 s | unusable (sd 68–149 s) |

If every implied error beyond the mapping's noise were removed exactly, v46 would read 223.17 (−0.76 s). Counting only the cells with a
tight mapping and no extrapolation (EGLL, LEBL, LIRF July, LEMD January), the stake is about 6–8e7 of squared error, **0.4–0.5 s**.
The reassuring half of the result needs no decision: the January 2026 levels that no 2025 month could verify (EDDM, EHAM, LIRF) agree
with the published figures within 2–7 s.

## 5. Published daily totals of off-block delay (`tools/screen_daily_totals.py`)

EUROCONTROL's daily all-causes pre-departure delay per airport equals, on the 2025 clocks, the day's sum of max(off-block − schedule, 0)
within 0.0–0.3 s per flight at seven airports. Spreading the day's discrepancy over the rows by error variance is worse than the stack
(+0.06 to +2.2% in three of four months); a learned residual model with the discrepancy as a feature reads −0.53 / −0.32 / −1.67 /
+1.14%. Figures: [`daily_totals.json`](daily_totals.json). The screen led to a declared round and to v48:
[`docs/daily_totals/README.md`](../daily_totals/README.md).
