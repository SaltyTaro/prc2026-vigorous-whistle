# Dig tunnel 4: LFPG and EGLL (3 October 2026): no GO, closed

Development months only (2025-01/07/11/12). Nothing declared, exported or uploaded; no official score or leaderboard read.
Control: the shipped chain's development proxy `runs/scout/refit/ctl_v55m.parquet` (`new`). Scratch: `runs/dig/lfpg_egll/`
(`build.py` features, `screen_lib.py`, `screen1.py`, `screen2.py`, `screen3.py`, `desc1.py`, `trace_probe.py`; results `screen1.json`, `screen2.json`).
Protocol: per-airport residual models fitted on three months (labels 0–10,800 s, residual clipped ±1,800 s), read on the fourth;
weights 0.25 / 0.5 / 1.0; change of the whole month's RMSE (all ten airports, all labels) with paired-day 95% intervals (UTC take-off days,
2,000 reps). GO bar: equal-footing mean ≤ −0.05%, no equal-footing month above +0.01%, two intervals below zero.

## Where the error sits (ordinary labels; share of each month's total SSE, Jan / Jul / Nov / Dec)

Classes by the hidden block time against the NM clock (diagnostic only): at LFPG unmatched the **early** class (block more than 300 s before the
NM clock; 16% of the rows, bias +241 s) holds 3.2 / 4.4 / 5.3 / 7.6% of the month's SSE, more than every other LFPG/EGLL cell. It is LIRF's
"early" mechanism (`docs/clean_line_screens/lirf_mechanism.md`) at Paris.

## Screens (all fail)

| Screen | Equal-footing mean, w 0.25 / 0.5 / 1.0 | Note |
|---|---|---|
| residual LightGBM, base inputs (prediction, clock intervals, runway, stand area, operator, wake, hour), unmatched rows | +0.005 / +0.027 / +0.123% | worse in every month at w ≥ 0.5 |
| + runway configuration / flow (30-min buckets from all take-offs and landings, minutes since a flow or configuration change, change during the predicted taxi) | +0.012 / +0.043 / +0.163% | |
| + per-runway take-off sequence and queue (gaps, counts in 10/20/30 min, busy-period position and length, arrivals on the runway, peers taxiing at the predicted off-block, take-offs during the predicted taxi, overtakes; iterated on the chain's predicted peer off-blocks) | +0.003 / +0.023 / +0.115% | |
| both, all rows of the airport | +0.013 / +0.054 / +0.225% | |
| stand × runway residual encoding, shrunk k = 50 / 200, within-airport contrast | k200 unmatched +0.001 / +0.005 / +0.022% | November and December −0.006…−0.014%, January and July worse |

Univariate bins confirm it: no configuration or queue bin has a residual of consistent sign across the four months; the only consistent cell
(LFPG, ≥ 15 peers taxiing to the same runway at the predicted off-block: +10 to +143 s) has about 1,000 rows in four months (≈ 1e6 of SSE).

**Early-class separation** (month-blocked LightGBM classifier, early vs agree, AUC Jan/Jul/Nov/Dec): LFPG unmatched, the chain's own lift
(prediction − NM proxy) 0.869/0.863/0.877/0.867; label-free inputs 0.818/0.790/0.835/0.823; + chain outputs 0.870/0.856/0.881/0.870;
+ configuration and queue 0.872/0.857/0.883/0.871. EGLL unmatched: 0.886/0.909/0.879/0.881 against at most 0.885/0.907/0.869/0.881.
The new inputs add ≤ 0.002 AUC: as at LIRF, the chain already holds what the inputs say about the class; the class itself is unobserved.

**Partial ADS-B traces of unmatched rows** (raw `data/external/adsblol/surface/`, three January days): EGLL unmatched rows have some point in
the two hours before take-off in 84% of cases, but a ground or below-200-ft point in only 26%, and those sit at or after the recorded take-off
(median first low point 28 s after it). LFPG: 0.1%. No queue-entry time exists for unmatched rows.

## Verdict

Closed at LFPG and EGLL: runway configuration/flow changes, reconstructed runway queues (including the iterated predicted-peer form), stand ×
runway encodings and partial traces add nothing transferable; the residual left is the early/late recording class, which no label-free input
separates beyond the chain. De-icing / cold mornings and per-runway bias were already closed (`docs/scout/README.md` §15,
`docs/clean_line_screens/runway_bias.md`).
