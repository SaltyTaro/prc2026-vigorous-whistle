# Dig tunnel 17: case files on the worst development rows (3 October 2026, 2025 only, nothing declared)

No official score, no leaderboard, no upload, nothing under runs/submissions. Labels are read only to pick, classify and score
development rows. Scratch (git-ignored): `runs/dig/case_files/`. The scripts are `build.py`, `context.py`, `dump.py` (`dump.txt`),
`ownmove.py`, `classify.py` (`mechanisms.json`), `scan_move.py` (`move_all.parquet`), `screen_move.py` (`screen_move.json`) and
`rank_move.py` (`rank_triggers.parquet`).

## The rows

- 150 ordinary rows (label at most 3 h, LIRF population excluded) with the largest squared error of the v61 development proxy
  (`runs/dig/census61/census_rows.parquet`). They hold 10.1% of ordinary non-population squared error over January, July, November
  and December 2025.
- 50 more rows from the other eight 2025 months, ranked on the twelve-month predictor (`runs/daily_totals/rows12.parquet`, `stack`),
  with LIRF population rows removed (no `ADEP_flt`, no `AIRCRAFT_TYPE_mvt`).

Each row was read with:
- every supplied field and the schedule / AOBT_3 / LOBT / EOBT_1 / IOBT intervals to take-off;
- the stand's arrivals (in-block, type) and departures within 10 h;
- the runway's and the airport's take-off gaps;
- the day's published-delay discrepancy;
- the aircraft's own adsb.lol points, where the ICAO identity table (`runs/intruder_identity/ids.parquet`) names it.

## Mechanisms in the listed rows

Classes are assigned in priority order, using labels as a diagnostic only. The share is of the 200 rows' squared error, 4.24e9.

| mechanism | rows (dev4) | share | label-free trigger? |
|---|---:|---:|---|
| extreme published day (LTFM 19 / 23 February 2025 snow; older predictor) | 43 (3) | 39.2% | yes: published discrepancy; already the extreme-day layer |
| LIRF schedule fill, clock present | 54 (52) | 15.2% | no (closed, `clean_line_screens/lirf_mechanism.md`) |
| no NM record, long schedule interval | 20 (18) | 12.3% | interval only, already a model input (no separating trigger) |
| no NM record, block near schedule, then a long hold | 18 (13) | 12.2% | no |
| early record: block well before the NM clock | 26 (25) | 8.4% | no (tunnel 4: AUC +0.002) |
| early record: block near schedule, NM clock late (EHAM / LFPG / LSZH / LTFM fills) | 28 (28) | 7.9% | no |
| GA / helicopter at LSZH (block about schedule) | 2 (2) | 2.9% | type, but 2 rows |
| over-predictions, late record, tow, label <= 60 s, unexplained | 9 (9) | 1.9% | no |

The top errors are 95% under-predictions. Most are in classes already closed.

One recurring pattern was new. Several dark or unmatched rows (UAL263 EGLL 30 July, UPS22 LFPG 22 November and others) show
**the aircraft's own ground motion well before take-off, after its identified inbound blocked in at the same stand**, while v61
predicts a normal taxi. In those rows the shipped matcher's segment starts after the last coverage gap, so this earlier motion is
never seen. For the longest gaps (B above 1 h) the motion is usually not the pushback, so the long-B rows themselves are not
usable. The mechanism holds in the moderate band.

## Candidate: own-motion bound (EDDF / EGLL / LFPG)

**Trigger (label-free):**
1. The departure's ICAO identity is known.
2. The latest identified arrival of the same aircraft at that airport blocked in 600 s to 18 h before take-off, at the
   departure's own stand.
3. The row is not gate-traced by the shipped matcher (`adsb_gate != 1`).
4. The first ground point of that aircraft with gs >= 3 kt after in-block + 120 s lies B seconds before take-off, with
   60 < B <= 2,700.
5. The prediction is below B.

**Value:** B + o(airport), where o is the mean of (label - B) on triggered rows of the other months, shrunk to the pooled mean with
weight 20. On all twelve months o is EDDF +141, EGLL +299 and LFPG +495 s.

**Validity in 2025** (label >= B - 60 s): with the inbound at the same stand, B up to 2,700 s:

| airport | EDDF | EGLL | LFPG |
|---|---:|---:|---:|
| validity | 93-99.9% | 98-100% | 99-100% |

It collapses for B > 3,600 s or a different stand: tows and repositioning. On the triggered development rows validity is 96.6%.

**Screen**, month-blocked, against the v61 proxy, as the change in month RMSE with its paired-day 95% interval:

| variant | January | July | November | December | equal-footing mean | twelve months (older predictor) |
|---|---|---|---|---|---:|---|
| B <= 2,700, w 1.0 | −0.038% [−0.131, −0.014] | −0.111% [−0.195, −0.047] | −0.082% [−0.194, +0.019] | −0.163% [−0.251, −0.085] | **−0.104%** | 12/12 better, mean −0.061% |
| B <= 2,700, w 0.5 | −0.028% [−0.096, −0.011] | −0.077% [−0.131, −0.034] | −0.093% [−0.158, −0.043] | −0.127% [−0.185, −0.075] | −0.078% | 12/12, mean −0.047% |
| B <= 3,600, w 1.0 | −0.043% | −0.109% [−0.214, +0.005] | −0.012% [−0.23, +0.21] | −0.031% [−0.33, +0.32] | −0.061% | 9/12 |

Triggered rows: 76 / 64 / 92 / 95 a month. Their gain per unit of squared change (w 1.0) is 0.88:

| by airport | EDDF | EGLL | LFPG |
|---|---:|---:|---:|
| gain / squared change | 0.70 | 1.04 | 1.45 |

| by matcher state | dark (matched, not gate-traced) | unmatched |
|---|---:|---:|
| gain / squared change | 0.47 | 1.30 |

Requiring a second moving point within 120 s ("persistence") lowers the gain in every month.

**GO by the screen bar.** The equal-footing mean is −0.104% (w 1.0) or −0.078% (w 0.5), no month is worse, and three or four
intervals are below zero. Thresholds 2,700 s / 3 kt / 120 s / 600 s–18 h were set while reading the rows, before the screen. The
2,700 s cut was chosen on the 2025 validity table (labels read), and the four B-band / persistence / w variants above are all that
was screened: disclose this in any declaration.

**Ranking (read-only on v61):**
- 155 rows trigger: EDDF 89, EGLL 50, LFPG 16; January 79, July 76.
- Squared change 4.25e7 at w 1.0 (mean +411 s, max +2,269 s), 1.06e7 at w 0.5.
- At the development gain ratio (0.88) w 1.0 is worth about **−0.25 s**, and about −0.14 s at a ratio of 0.5.
- No LIRF population or guard row is touched.

**Order:** apply it last, next to the intruder bound. It raises rows only, and the stand bounds share that direction.
Identity is available only at EDDF / EGLL / LFPG / LTFM, and LTFM has no surface data. Extending identities to EHAM / EDDM / LSZH
/ LEBL / LEMD would widen the trigger, but their ground coverage is mostly gate-traced.

## Verdict

- The case files hold no other new triggerable mechanism. Everything else is a closed class (fills, early or late records, no-NM
  holds) or an extreme day the published layers already read.
- **The own-motion bound is a GO for a declared round.** Rule as above; candidates w 1.0 and w 0.5 at B <= 2,700; the control is
  the v61 chain; the usual four-part rule applies.
