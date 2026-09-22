# ADS-B coverage census (22 September 2026)

Read-only, label-free. Tool: `tools/audit_adsb_coverage.py` (replays the matcher's own steps for every
departure and records the first step that rejects the row). Reports: `census_probe.json` (2026-01),
`census_jul_dec.json` (2026-07, 2025-12), row tables under `runs/adsb_coverage/`. Nothing was fitted,
scored or uploaded, and no label was read.

Run because three consecutive residual-stage rounds (v25 shipped, v26 and v27 null) showed the model
side is close to exhausted, and the December audit put 44.5% of ordinary SSE on rows with no ADS-B match.

## What a match is worth

Measured on the shipped v23 development predictions against v9 alone, non-LIRF rows:

| class | December rows | v9 | v23 | v23 vs v9 | share of non-LIRF SSE |
|---|---|---|---|---|---|
| unmatched | 46,454 | 223.7 | 218.6 | **−2.3%** | **60.1%** |
| dark (matched, never seen standing) | 63,143 | 184.2 | 134.3 | −27.1% | 30.8% |
| gate (stand stop observed) | 44,012 | 169.7 | 87.0 | −48.7% | 9.0% |

July 2025: −1.0% / −21.1% / −35.4%, unmatched carrying 71.3%. The whole ADS-B stack converts to gain
only where there is a match; the rows carrying most of the error barely move.

## Why departures fail to match

| | 2026-01 | 2026-07 | 2025-12 |
|---|---|---|---|
| gate | 23.4% | 23.8% | 27.6% |
| dark | 45.2% | 38.9% | 42.9% |
| callsign absent at that airport | 12.9% | 19.4% | 5.2% |
| aircraft seen, never on the ground | 7.3% | 13.3% | 7.8% |
| airport absent from the archive that day | 6.2% | 0.0% | 11.9% |
| only one or two ground points (three needed) | 3.4% | 3.1% | 3.5% |
| segment rejected / no airborne point / no callsign | 1.5% | 1.5% | 1.2% |

Of the callsign-absent rows, 10,711 (Jan) and 23,243 (Jul) have a callsign with the **same flight number
and a different prefix** present at the same airport that day — the IATA/ICAO spelling difference. Most
of those are at airports where nothing else is observable (LTFM 12,351 of Jan's 19,720; in July LTFM
15,527 and LIRF 1,986 of 23,243), so the recoverable part is **~4,200 rows in January and ~5,700 in July**.

## Ground-point coverage is not stable, and the ranking months are the worst

Ground positions per airport per month (thousands), from the local archive:

| month | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-07 | 1,790 | 1,303 | 301 | 9,369 | 2,010 | 587 | 34 | 944 | 1,651 | 0 |
| 2025-11 | 2,200 | 2,908 | 1,101 | 8,650 | 1,922 | 2,599 | 180 | 313 | 1,250 | 0 |
| 2025-12 | 1,978 | 2,880 | 952 | 8,466 | 1,934 | 3,412 | 118 | 493 | 1,796 | 0 |
| **2026-01** | **755** | 3,674 | 1,430 | 8,709 | 1,412 | 3,219 | 107 | 403 | 1,825 | 0 |
| **2026-07** | **471** | 3,239 | 4,188 | 9,355 | 2,747 | 1,812 | 171 | **2** | 1,103 | 3 |

Three findings matter:

1. **Rome goes dark in the July ranking month.** LIRF has 403k ground points in January 2026 and
   **2.1k across 7 days** in July 2026 — the raw archive holds thousands of LIRF points on 1, 15 and
   20 July with **zero** ground positions, every one airborne at ≥125 ft and ≥114 kt. The July census
   shows it: 0.0% gate, 0.2% dark, 72.1% never-on-the-ground. LIRF is our highest-error airport and
   half the ranking set is July. Nothing in our code recovers this.
2. **Frankfurt falls to a third.** EDDF drops from ~2.0M ground points a month in late 2025 to 755k
   (Jan 2026) and 471k (Jul 2026). Its gate share falls from 38.0% in December to 7.8% in January and
   3.8% in July. Our models learn EDDF mostly from gate-rich months and are scored on gate-poor ones.
3. **LIRF has no ADS-B at all before May 2025** (zero ground points January–April), and LTFM has none
   in any month. Training coverage and ranking coverage are different regimes.

Four archive days were never downloaded (2025-05 has 27 of 31 days, 2025-06 has 20 of 30) — a download
gap, not a coverage gap, and the reason June's match rate is 38.5% against November's 70.8%.

## Learned stand positions: a circular flaw

`learn_stands` derives each stand's position **only from segments already classified as gate**. An
airport we rarely catch at the gate therefore never learns its stands, and so keeps failing to catch it:

| airport | learned stands | samples |
|---|---|---|
| EHAM | 187 | 15,172 |
| EDDF | 184 | 8,440 |
| LEMD | 183 | 5,057 |
| EDDM / EGLL | 153 | 13,268 / 4,507 |
| LEBL | 131 | 7,162 |
| LSZH | 74 | 8,091 |
| **LIRF** | **43** | 539 |
| **LFPG** | **3** | 10 |
| LTFM | 0 | 0 |

Of the dark rows, 71–72% have a known stand, but the first observed point is a median 273–404 m from it
and only 15–21% are within 100 m; the `at_known_stand` rescue needs 40 m. We already hold OpenStreetMap
parking positions (ODbL) for LIRF and LFPG from the source-geometry campaign.

## What this leaves

Recoverable with data already held, per ranking month: ~4–6k rows from callsign variants, ~5–6k rows
held out by the three-ground-point threshold (mostly LFPG), ~2–3k segment rejections, and some part of
the 15–21% of dark rows whose first point is within 100 m of a known stand. Rescued traces will be
thinner than current ones, so they will not earn the full −27%/−49%; the conversion rate has to be
measured, not assumed.

Not recoverable: LTFM entirely, LFPG's 37–40% never-on-the-ground rows, and LIRF in July 2026.

## The matcher fixes were tested and all three fail (22 September)

`src/prc2026_adsb_v12/matching.py` re-implements the matcher with each census failure point as an option
(the frozen module is untouched); `tools/probe_adsb_v12.py` measures the class shift and
`tools/validate_adsb_v12.py` tests whether the extra matches carry signal. The base configuration
reproduces the frozen matcher row for row, so the deltas below are real.

**Coverage gained** (four January 2026 days, 19,857 departures):

| fix | extra matched rows | dark to gate |
|---|---|---|
| accept one or two ground points | +4.53% | +3 |
| known-stand radius 40 m to 100 m | 0 | +1,413 (gate share 23.3% to 30.4%) |
| callsign variants (unambiguous, type must agree) | +0.49% | +35 |
| all three | +5.06% | +1,469 |

**Signal test** (four December 2025 days, 20,923 departures, against the actual taxi-out label). For a gate
row the matcher's own estimate of the pushback, takeoff minus motion start, should approximate the label:
the frozen matcher's gate rows sit at a median 63 s from it, its dark rows at 401 s.

- **One or two ground points: garbage.** The 878 newly matched rows have a median of **one** ground point
  and a median estimate error of **3,233 s**. A single stray ground ping is not a departure trail.
- **The 100 m radius is actively wrong.** Its 1,622 conversions have a median first point 64 m from the
  stand — already rolling. Their estimate error goes **84 s to 163 s** and only **1%** of them improve.
  The frozen matcher was right to leave them dark; at 84 s they were among its *best* dark rows.
- **Callsign variants add ~55 rows** over the point threshold, 3% of them gate. Too few to matter.

The census's "recoverable" buckets are therefore not recoverable: they are rows where the archive holds no
usable surface trail, not rows our code mishandles. The matcher lever is closed on evidence, with no refit,
no export and no worker time spent. What is left in this direction is the 14 archive days never downloaded
(training rows only) and better modelling of the rows that have no trail at all — for which the committee
already carries the neighbour features, and for which no specialist has ever been fitted.
