# Hour-effect screen (29 September 2026) — closed

Nothing is declared, selected,
exported or uploaded; no ranking row and no official score is read. Tool: `tools/screen_hour_effect.py`; report:
[`hour_effect.json`](hour_effect.json). Control: the **v46 development stack**, rebuilt with
`tools/run_share_gate_reread.full_stack` and the `n85_nd1200` mix; it reproduces the recorded RMSE on all four months
(322.8585 / 257.7185 / 184.9050 / 167.5137).

## What prompted it: a census of the stack's error

Rows: `runs/lomo_dev/rows.parquet`, residual = label − stack, labels below 10,800 s ("ordinary"; they hold 65% of the
development squared error, the other 35% sits in 78 rows).

| cell | rows | RMSE (s) | share of ordinary squared error |
|---|---:|---:|---:|
| LIRF, ADS-B matched | 31,230 | 429 | 22.5% |
| LTFM (all unmatched) | 91,338 | 210 | 15.7% |
| LFPG unmatched | 64,894 | 240 | 14.5% |
| LIRF unmatched | 18,623 | 378 | 10.4% |
| EGLL unmatched / matched | 33,085 / 46,546 | 235 / 181 | 7.1% / 5.9% |
| LEBL matched | 53,542 | 151 | 4.8% |
| every other cell | | | below 3.1% each |

- Rows with an error above 600 s are 1.2% of the rows and hold 55% of the ordinary squared error; 72% of them are
  under-predictions.
- **The error clusters by hour.** The airport-hour mean residual explains 9.6% of the ordinary squared error; shuffling
  the residuals within each airport-month leaves 4.2%. The excess, about 5.5 points, sits at LTFM (1.4), LIRF (1.3),
  LFPG (1.1) and EGLL (0.75). The airport-day excess is about 1 point, as on record.
- **The hour state is shared by matched and unmatched rows.** Within the same airport-hour (at least five rows of each)
  their mean residuals correlate +0.56 at EGLL (1,590 hours), +0.52 at LFPG (944), +0.52 at EDDM, +0.35 at LEBL and +0.27
  at LEMD.

## The screen

- **Unit:** airport × UTC hour of the take-off; 25,080 airport-hours.
- **Target:** the hour's mean residual, each residual clipped to ±3,600 s.
- **Inputs (69, all label-free and computable on the ranking file):** for the hour and the hours h−2 to h+2:
  departures and arrivals, the largest gap between take-offs, the stack's and its components' predictions and their
  disagreements, schedule intervals, NM off-block proxies, the share of rows without an NM clock, the ADS-B match share and
  the matched rows' ADS-B taxi against their prediction, arrival taxi-in against the airport-month median; hour of day,
  weekday, airport. No departure label or block time enters an input.
- **Model:** LightGBM with fixed settings, fitted on three months' hours (weighted by their rows), read on the fourth.
- **Reading:** RMSE of stack + w × predicted hour effect against the stack, on every row of the held-out month.

| held-out month | hour correlation | w = 0.5 [paired-day 95%, s] | w = 1.0 [paired-day 95%, s] | calendar only, w = 1.0 |
|---|---:|---|---|---:|
| January (eleven-month footing) | +0.033 | −0.010% [−0.14, +0.09] | +0.044% [−0.06, +0.48] | +0.033% |
| July (eleven-month footing) | +0.063 | +0.023% [−0.09, +0.21] | +0.140% [+0.05, +0.70] | +0.083% |
| December (fair) | +0.071 | +0.082% [−0.41, +0.89] | +0.506% [−0.39, +2.73] | +0.117% |
| November (nine-month fold side) | +0.042 | +0.016% [−0.05, +0.11] | +0.155% [+0.09, +0.49] | +0.082% |

Mean over the three equal-footing months: +0.032% at w = 0.5 and +0.230% at w = 1.0.

- The predicted hour effect correlates with the true hour mean residual at 0.03–0.07. Adding it **raises** the error in
  every month at full weight and in three of four at half weight.
- LTFM, where the hour structure is largest, worsens most (December +3.2% at w = 1.0).
- The inputs the model leans on are the components' disagreements, which the shipped gates already use.

**Reading.** The hour state is real but the supplied data and the ADS-B aggregates do not show it. It agrees with the
earlier day-level reading (`docs/tta_screen/README.md`), with the row-level learned gate (section 3 of
[`README.md`](README.md)) and with the peer-congestion lift (about 0.04 s). What the hour shares is most likely the wait
between off-block and first movement, which no input observes. **Closed.**

## Two by-products of the census (not levers as they stand)

1. **Rows without an NM record outside LIRF**: 1.2% of rows, 6.8% of the ordinary squared error, 84% of it in 428 rows with
   an error above 600 s. At six airports their schedule time is at least an hour before the block time in 90% of the rows:
   they are mostly heavily delayed or refiled flights. On record as examined; not reopened.
2. **Departure stops** (no take-off for at least 30 minutes after a busy half hour, with at least ten of the following
   departures scheduled before or during the stop): 16 events in the twelve months of 2025. The rows that left their stand
   before a stop are badly under-predicted (EGLL on 30 July 2025: labels of 5,300–5,600 s against 2,800–2,900 s), and for
   them the NM off-block clock is late (median proxy 600–1,700 s against labels of 2,200–3,700 s). But the events are of
   two kinds that no label-free descriptor separates: in six the queue waited through the stop (53–88% of the first
   fifteen departures had left their stand before it), in the others the aircraft were held at the gate (0–20%).
   - Tested rule: for the first K departures after the restart that were scheduled before the stop began and have no ADS-B
     trace from the stand, move the prediction a fraction α of the way up to take-off minus stop start.
   - Against the twelve-month out-of-sample baseline (cross-fitted v9 plus the leave-one-month-out residual): at best −0.06%
     of the squared error (K = 15, α = 0.25–0.5), 9 events better and 7 worse. Against the v46 stack on the five events of
     the development months: −0.08% to −0.12% of the squared error, all of it from EGLL on 30 July; both EDDF events worsen.
   - Arrivals during the stop, the restart's discharge rate and the stop's length do not tell the two kinds apart.
   **Closed as a rule.** The ranking months hold about a dozen daytime stops (EHAM on 5 and 7 January, LEMD on 28 January,
   EDDM on 18 July among them); a lift there would be a coin flip per event.
