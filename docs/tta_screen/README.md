# Test-time adaptation screen (18 September 2026): closed

Question: in the ranking months, gate-covered rows carry a near-exact pushback time from
ADS-B, and arrival rows carry their taxi-in labels. Can either be used at prediction time
to correct the incumbent's predictions on the rows that lack their own surface data (dark
segments and unmatched rows, 21% of the development squared error) — an airport-day or
airport-hour shift estimated from those rows and applied to the others? Read-only
diagnostics on the three development months with the v18 recipe's predictions
(`ens_v20_bag_q_all` at 0.7 over v9 + the v12 residual; 323.327 / 208.016 / 156.216 on
Jan / Jul / Nov 2025); no model, no export.

## 1. Is there a day effect to correct?

In-sample oracle (subtract every group's mean error): airport-month means are worth
0.07–0.37% of a month's squared error on unmatched rows; airport-day means 0.67 / 1.56 /
0.99%; 3-hour blocks 2.8–6.3% (mostly noise: ~6 rows per block). Honest estimate by
split halves of each airport-day (covariance of the half-means):

| rows | month | split-half corr | day-effect SD | honest share of month SSE |
| --- | --- | ---: | ---: | ---: |
| unmatched | Jan / Jul / Nov | 0.10 / 0.47 / 0.41 | 27 / 31 / 25 s | 0.33 / 1.12 / 0.76% |
| dark | Jan / Jul / Nov | 0.62 / 0.53 / 0.78 | 17 / 27 / 20 s | 0.09 / 0.52 / 0.71% |

So a real airport-day component exists, worth ≈ 0.15–0.55% RMSE if it were known exactly.

## 2. Can the gate rows see it? No.

Pseudo-label on gate rows: ADS-B-implied taxi plus the airport's median label offset
(training-side quantity). The gate rows' pseudo-residual against the incumbent is ≈ 0 by
construction (medians −4 to −8 s; the learners already track the motion start on those
rows), so the usable signal is the gate rows' ADS-B delay anomaly relative to the base
model (pseudo-label − v9, per airport-day, minus the airport-month mean). Its correlation
with the incumbent's unmatched-row day error, over 485 airport-days with ≥ 10 rows of each
kind: **+0.01 / −0.05 / +0.09** (Jan / Jul / Nov), slopes 0.03 / −0.02 / 0.10; against
the dark-row day error (585 airport-days): +0.15 / −0.15 / −0.15. The incumbent absorbs
the ADS-B day signal through the neighbour features (10–30-minute windows of matched
neighbours); what remains on the other rows is orthogonal to it.

## 3. Can the arrivals see it? No.

Arrival taxi-in labels are in the ranking file. The airport-day median taxi-in anomaly
(relative to the airport-month median; 728 airport-days with ≥ 30 arrivals and ≥ 10
unmatched departures) against the unmatched-row day error: **−0.22 / +0.06 / −0.11**,
slopes −0.23 / +0.04 / −0.08 — small and of inconsistent sign (January's is negative:
where arrivals taxi longer, the incumbent already over-corrects). Dark rows −0.16 / +0.02 /
+0.07; gate rows −0.15 / +0.02 / −0.08. Airport-day-3-hour blocks (2,924 with ≥ 10
arrivals and ≥ 8 unmatched departures): −0.04 / +0.12 / −0.09. The 14 September
arrival-context campaign (`docs/arrivalctx/`) had already found no usable signal in
arrival taxi-in windows for the base model.

## Verdict

The day-level error that remains on dark and unmatched rows is real but invisible to the
two test-time signals available in the ranking months; both are already used by the
models at finer granularity. Test-time adaptation is closed on this evidence; no
declaration follows.
