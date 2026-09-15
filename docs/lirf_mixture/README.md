# LIRF turnaround mixture (15 September 2026, v9 scored 267.4454; new official best)

Declared at **06:05 UTC on 15 September 2026** in
[`configs/lirf_mixture_v1.json`](declaration.json) after the diagnostic that
found the signal, and before the declared implementation was scored. It fits
one small classifier per fold on the fitting months' LIRF population rows (a
LightGBM with seven leaves; the only native fits) and leaves every frozen v7
model untouched. V7 (275.0570 seconds) was the incumbent; v8 (283.668) is not
used for any rule.

**Selected: `mixture_75`, scoring 314.2323 seasonal / 219.8750 forward against
the exactly reconstructed v7 development scores 322.8193 / 221.1382
(−2.660% / −0.571%)**, clearing the 0.5% margin on both folds. Paired-day 95%
intervals against v7: seasonal [−14.84, −3.92], forward [−3.12, +0.41]
(48/62 and 18/30 days lower). `runs/submissions/vigorous-whistle_v9.parquet`
(SHA-256 `25d3b296a2753852144f53a2c36a1e539031dbd6ef35b8be928d9e1fa7363521`)
was built, independently verified, and **uploaded with the user's approval at
05:00 UTC on 15 September; processed at 05:01:23 UTC: 267.4454 seconds RMSE on
all 344,841 rows, 7.6116 s (2.767%) better than v7 and the new official best.**
At 05:03 UTC the team ranks **12th of 141** (up from 19th); third place is
246.626, 8.27% away. Evidence: [`official_result.json`](official_result.json),
[`submission_remote_verification.json`](submission_remote_verification.json),
[`leaderboard_snapshot.json`](leaderboard_snapshot.json).

## The signal

LIRF departures with no NM record and no aircraft type (the population,
~1% of LIRF rows) have `BLOCK_TIME = SCHED_TIME` in about half the rows. The
stand's arrival history separates the two modes: when the scheduled departure
precedes the previous arrival's in-block at the same stand, the fill share is
65–83%; when the schedule falls within two hours after that in-block, it is
25–33% (base rate 49%). Arrival in-block times remain supplied in the ranking
data, so the features are label-free at prediction time.

## Method

- Population rows with schedule interval `s = MVT − SCHED ≤ 43200 s` receive
  `(1 − w)·v7 + w·[p_fill·s + (1 − p_fill)·normal(s)]`, with `w = 0.75`;
  `p_fill = P(fill)/(P(fill) + P(normal))` from a three-class classifier
  (fill / normal-or-other / one-day) and `normal(s)` the fitting-month median
  label of normal-mode rows in the row's schedule-interval band. Rows above
  43200 s stay exactly at v7, so the one-day mode is handled as before.
- Features: schedule interval, local and scheduled local hour, date shift, day
  of week, month, flight-number shape, flight prefix, stand prefix, runway,
  destination region, and three turnaround quantities from the stand's arrival
  in-blocks (schedule minus previous in-block, takeoff minus previous
  in-block, next in-block minus takeoff; previous means at least 1200 s before
  takeoff).
- Rounds come from date-grouped five-fold inner validation on fitting rows
  (seasonal 228, forward 460, export 454). Everything is floored at the
  fitting-period minimum label (−12 s).

## Results

| Candidate | Seasonal | Forward | Mean ratio |
| --- | ---: | ---: | ---: |
| v7 (reconstructed) | 322.8193 | 221.1382 | 1.0000 |
| `mixture_50` | 314.9228 | 219.9007 | 0.9850 |
| **`mixture_75`** | **314.2323** | **219.8750** | **0.9838** |
| `mixture_100` | 314.4108 | 220.1614 | 0.9848 |

Population SSE by true mode for `mixture_75`: seasonal fills 7.59e8 → 8.16e8,
normal/other 4.12e9 → 2.48e9, one-day 1.25e9 unchanged; forward fills
1.55e8 → 1.31e8, normal/other 1.01e8 → 0.52e8, one-day unchanged. Scoring
fill AUC 0.893 seasonal / 0.981 forward. These are reused development folds;
the floor alone contributes about 3.9 s of the seasonal gain (see
[`../postprocess/README.md`](../postprocess/README.md)).

## Export and verification

- Refit-period classifier on 1,399 population rows (655 fill / 733 normal or
  other / 11 one-day), saved at `runs/lirf_mixture/export_v9/classifier.txt`
  (SHA-256 `c8ed40b1209ceafb94ca44484d4fe9c82b7aea9624bae302757301b10d52058e`)
  with its categories, rounds and normal component in `classifier.json`.
- The ranking population has 383 rows (372 at or below 43200 s); v9 changes
  398 rows (372 mixture, 26 floored). Ranking `p_fill` quantiles: 0.02 / 0.07 /
  0.39 / 0.84 / 0.96 at 10/25/50/75/90%. Every other row equals the uploaded v7
  integer. Export report: [`export.json`](export.json).
- [`verification.json`](verification.json): an independent re-implementation
  of population, turnaround features, normal component, floor and mixture,
  using the saved booster, reproduces the export exactly; saved probabilities
  differ by 0.0.
- Selection receipt [`development_results.json`](development_results.json),
  SHA-256 `67c95f9e89e57d9c8b759b95d82916533684c8969a2655968d7401b874b10aa7`;
  model-code SHA-256 `47f2c4dfcd26d640ef061c1bdaf5883f481d5c1aec1ae9218040c8905ca7a911`
  (`src/prc2026_lirf_mixture/` and `tools/run_lirf_mixture.py`).
- Six new tests in `tests/test_lirf_mixture.py`; **307 tests pass** (`tests_local.xml`).

## Closed alternatives from the same session

- Lever C (blending saved variants into v7): cross-fold non-negative stacking
  transfers at most −0.13% and is negative on the missing route; the only
  clean fixed blend is `ladder:cat_long` at 25% on the clock route
  (−0.41%/−0.38% SSE, ordinary rows), worth about 0.5 s and needing an Azure
  refit. `nm_mask:cat_augmented` and `ladder_scale:missing_scale*` take their
  forward gains from the two one-day rows and were rejected.
- Lever D (January 2026 disruption): December 2025 contains no analog on any
  label-free signal (its worst day, EGLL 31 December, is far below EHAM
  3–6 January 2026); v7's day-level bias is flat across 2025's congestion
  range; snow/ice/freezing/fog codes are already features. No December label
  was read. Peer NM proxies explain only 0.15–0.3 of the residual correlation
  on ordinary missing-clock rows.

Tools: `tools/run_lirf_mixture.py`, `tools/export_lirf_mixture.py --version N`,
`tools/verify_lirf_mixture.py --version N`. No Azure worker, external data,
December label or leaderboard-derived rule was used.
