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
