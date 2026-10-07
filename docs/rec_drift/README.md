# Recording-drift screen: did an airport change how it records block times in 2026? (7 October 2026) — NO GO

Screen only: no declaration, no candidate file, no upload, no leaderboard score read. 2025 labels enter only as the recorded
off-block of gate-traced 2025 departures and through the v69 development proxy (`runs/stack_weight/dev/w65_bag10/proxy.parquet`,
c088e409...). The 2026 side is label-free.

## Hypothesis

The label is take-off minus the airport-recorded off-block. If an airport changed its recording procedure (A-CDM source, docking
system, handler) between 2025 and 2026, models trained on 2025 cannot see it. Arrivals expose such a change without labels: every
ARR row, in the ranking file too, carries the recorded in-block. A search of `docs/` found no earlier test of this.

## Method

- **Arrivals** (`tools/rec_drift_arrivals.py`, on arcw1-3 over all 413 day files, 50-80 s per worker): callsign (CALLSIGN_flt, else
  FLIGHT_mvt) at the arrival airport. The candidate ICAO whose air-to-ground transition is nearest the recorded landing (within 300 s)
  is kept. Two stop times are taken after touchdown. **A** is the first point with gs <= 1 kt within 40 m of the learned stand
  position (`stands.parquet`, the departure matcher's anchors), kept only if the previous point is <= 30 s earlier. **E** is the
  first point within 40 m of the stand (entering). The arrival offset is recorded in-block minus the stop.
  Matched arrivals: 868k (EDDF 163k, EDDM 154k, EGLL 49k, EHAM 261k, LEBL 164k, LEMD 154k, LFPG 16k, LIRF 5k, LSZH 64k, LTFM 0).
  Sharp stand stops: 233k. **LFPG, LIRF and LTFM have essentially none** (no stand positions, no ground coverage).
- **Departures** (`tools/rec_drift_summary.py`, local): gate rows of `matches2_<month>.parquet`. dep = recorded off-block (take-off -
  label) - motion start (2025). aobt = AOBT_3_flt - motion start (2025 and 2026). imp = (take-off - prediction) - motion start, with
  v70 in 2026 and the v69 proxy in its four 2025 months. to = recorded take-off - last ADS-B ground point.
- **Stand fixed effects (fe).** Each row minus its stand's 2025 median (stands with >= 20 rows in 2025). Departures use rows whose
  segment starts at the known stand. Without this, a change in which stands the receivers see reads as a recording change. Raw EDDF
  dep moved from -97 s (January 2025) to -49 s (November 2025). The fe reading moves only from -7.7 to +3.7 s, because the
  September 2025 coverage gain added other stands.
- **fe_dense.** As fe, restricted to tracks with more than 0.05 points per second. On sparse tracks the motion start is seen late.
  This is the source of the EDDF 2026 departure "shifts" below.

Tables: `runs/rec_drift/airport_month.csv` (n, median, 10%-trimmed mean, IQR, SE of the median for each airport x month x quantity
x reading). Everything else is in `results.json`. Arrival receipts (day-file hashes): `receipts/`.

## Results

**Matching check** (ADS-B touchdown minus recorded landing, medians). The values are stable per airport over 2025: EDDF 29-34 s,
EDDM 41-47, EHAM 26-32, LEBL 17-28, LEMD 14-17, LFPG 49-51, LIRF 67-78, LSZH 14-17. 2026 is inside or near those ranges, except
LSZH in July 2026 at 34 s. Its tracks are sparser that month (0.11 vs 0.18 points/s), so the first ground point is seen later.
The stop measure A needs a point within 30 s before the stop and is not affected.

**Arrival recording offset, fe, seconds** (2025 monthly range | January 2026 | July 2026). Raw levels: EDDF +33..+38, EDDM -28..-31,
EGLL -17..-22, EHAM -29..-32, LEBL -55..-65, LEMD -58..-69, LSZH +3..+8.

| airport | A: 2025 range | A: Jan 26 | A: Jul 26 | E: 2025 range | E: Jan 26 | E: Jul 26 |
|---|---|---:|---:|---|---:|---:|
| EDDF | -1.1..+0.7 | +1.1 | -2.1 | -2.9..+1.6 | -1.0 | -5.2 |
| EDDM | -1.5..+0.8 | +1.4 | -0.1 | -1.2..+1.2 | +1.3 | -0.6 |
| EGLL | -3.2..+2.1 | +1.5 | +1.0 | -1.2..+5.5 | +0.2 | +2.9 |
| EHAM | -1.2..+0.7 | +0.5 | -0.2 | -1.4..+0.7 | +0.5 | +0.5 |
| LEBL | -3.7..+5.6 | -1.6 | -1.1 | -4.1..+6.9 | -1.1 | -1.6 |
| LEMD | -1.9..+2.4 | +1.6 | -0.2 | -1.2..+1.1 | +0.1 | -0.2 |
| LSZH | -1.8..+1.9 | +0.6 | -0.6 | -1.0..+1.5 | +1.3 | -4.1 |

No airport's in-block recording moved by more than about 2 s (A) or 5 s (E, EDDF and LSZH in July 2026, with A not confirming it).

**AOBT_3 - motion start, fe** (2025 range | Jan 26 | Jul 26): EDDF -31..+18 | -43 | -89 (fe_dense: -13 | n < 100); EDDM -36..+44 |
+16 | +9; EGLL -18..+25 | +25 | +52; EHAM -13..+7 | +8 | +2; LEBL -31..+22 | -42 | +24; LEMD -20..+36 | -12 | -11; LIRF -36..+15 |
-38 | none; LSZH -13..+10 | +13 | +22 (fe_dense +6 | +7). AOBT_3 drifts much more than either recorded block clock. In December
2025, for example, LSZH's raw AOBT_3 offset moved by +30 s while its label offset stayed at -41 s. AOBT_3 drift is input drift,
not label-recording drift. Arrivals and AOBT_3 never agree on a 2026 shift.

**Co-movement on 2025** (correlation of monthly deviations, fe). The departure label offset itself barely moves in 2025: sd of
monthly fe medians is 0.6 s (EHAM, LSZH), 1.5 (EDDM), 2.9-3.6 (EDDF, LEBL, LEMD) and 10 (EGLL).
- **Arrival E vs departure offset:** r = 0.91 EDDF, 0.81 EDDM, 0.86 EHAM, 0.37 LEMD, about 0 at LEBL / LSZH; pooled r = 0.28.
  Slope (departure on arrival) is 0.6-2.1. The arrival reading tracks the departure offset where both move.
- **Arrival A vs departure offset:** r = 0.85 at EHAM, weak elsewhere; pooled -0.05.
- **AOBT_3 vs departure offset:** r = 0.57 at EGLL with slope 0.43 (fe_dense 0.34). Elsewhere slopes are <= 0.07 (LIRF 0.67,
  7 noisy months).

AOBT_3 movement carries into the labels only at EGLL. There, in 2025, AOBT_3 and the label offset both run about +30-50 s later in
summer.

**v70's implied offset** (fe_dense; 2025 proxy mean | Jan 26 | Jul 26): EDDF -4.6 | -9.7 | n < 100 (the fe reading with sparse
tracks gives -24 | -35: late-seen motion starts); EDDM +1.5 | +2.2 | +7.0; EGLL -0.9 | +2.2 | +19.5; EHAM -0.2 | -1.9 | +3.4;
LEBL -6.9 | -0.6 | -1.0; LEMD -10.7 | -9.6 | -4.2; LSZH -6.5 | -6.6 | -10.5; LIRF -24 | -1.7 (n 341).

No arrival shift d exists for v70 to follow, so the d / part / none question has nothing to measure. v70's largest change is EGLL
July (+20 s). It matches the label movement that EGLL's 2025 relation predicts from its AOBT_3 change (+39..+54 s x 0.34-0.43 =
+14..+23 s). The proxy's own 2025 pattern there (label - prediction on gate rows) is -12 to -17 s and stable across the AOBT_3 drift
of late 2025.

**Size.** The largest b^2 x departures from an arrival-indicated shift that v70 ignores is EDDF July 2026. Taking E's -4.6 s as
read gives 4.4e5 s^2. Scaling it by the 2025 departure-on-arrival slope of 2.1 gives 1.9e6 s^2 (20,650 departures). Every other
airport-month is below 3e5. Under the unsupported reading that v70's EGLL July change (+20 s) is all bias, the figure is 8.2e6;
EGLL's own 2025 co-movement puts the expected bias at about 0 +- 9 s (at most about 1.7e6).

## Verdict

**NO GO.** At the seven airports with arrival coverage (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LSZH), the in-block recording
convention in January and July 2026 is within about 2 s of 2025 at matched stands. No airport-month reaches the 5e6 s^2 bar.
AOBT_3 shifts in 2026 (EDDF, EGLL, LEBL, LSZH) are input drift. At EDDF they are a sparse-coverage artefact; at EGLL v70 follows
them as 2025 says the labels do. The screen cannot see LFPG, LTFM or LIRF, where most of the projected ranking error sits:
adsb.lol has no arrival stand stops there.

## Compute

arcw1-3 started at about 15:00 UTC on 7 October. 181 missing day files were copied from the workstation; they stay on the worker
disks, with hashes in the receipts. Each worker ran one extraction. All outputs were hash-checked against the receipts after copying
back. All three were deallocated by 15:20 UTC; all eight workers were checked deallocated at 15:24 UTC. Nothing ran on AWS.
