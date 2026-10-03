# Dig tunnel 18: own-aircraft turnaround points as soft features in the guarded stacking learner — GO (3 October 2026)

2025 development months only; nothing declared, built or uploaded; no official score or leaderboard read. Scratch `runs/dig/pregap/`
(`build.py` features, `screen.py` screens, `screen_*.json`, `decomp.json`, `rank_coverage.json`, `cand_*.parquet`).

## Why

Tunnel 17 (`case_files.md`) found that the shipped matcher's departure segment starts after the last coverage gap (> 600 s), so the
aircraft's own points between its inbound in-block and that gap never reach the models; tunnel 15 (`own_bounds.md`) showed that as HARD
bounds most of these points are unreliable (tows, early recording). This tunnel asks whether a learner can use them softly.

## Features (label-free, every airport)

- **Identity** as `tools/run_intruder_identity.build_ids`, extended from EDDF / EGLL / LFPG / LTFM to all ten airports: nearest adsb.lol
  point of the same callsign (CALLSIGN_flt, else the rematch key map) at the same airport within 600 s of the take-off / landing.
  Departures identified: 97-98% at EDDF / EDDM / EGLL / EHAM / LEBL / LSZH, 77-88% at LEMD / LFPG / LIRF (dev), LTFM ~0.
- **Inbound:** latest identified arrival of the same ICAO at the same airport, in-block 600 s .. 18 h before take-off (72% of dev rows,
  72% of ranking departures).
- **Own points** in [in-block + 120 s, take-off + 120 s]: count; take-off minus first point (any speed) and minus first ground point at
  >= 3 kt; take-off minus last parked point (ground, < 1 kt); parked within 150 m of the park position in the last 10 / 20 / 40 min;
  points before the last > 600 s coverage gap (count, take-off minus the last of them, their largest distance from the park position,
  the gap length); start of the post-gap segment; tow flag (inbound stand != departure stand); plus each time minus the chain's
  prediction (`new`). 19 columns, each with its own support guard (0.1-99.9% per airport of the fitting months), as tunnel 13.

## Screens (month-blocked, v2's settings: 3-seed bag, w 0.5, v2's support guard; carried to the v61 development proxy)

Change in complete-month RMSE against the **v61 development proxy** (`census61.md`; the candidate's change is added on the 99.98% of
rows the intruder / identity layers left alone); paired-day 95% interval in %.

| screen | January | July | November | December | equal-footing mean |
|---|---|---|---|---|---:|
| stage 2 control (46 v2 inputs only) | +0.016 [+0.001, +0.060] | +0.034 [+0.015, +0.053] | +0.022 | −0.039 | +0.004% |
| stage 2: 46 + own points, w 0.5 | −0.060 [−0.221, −0.021] | −0.011 [−0.057, +0.032] | −0.313 [−0.445, −0.217] | −0.403 [−0.536, −0.279] | −0.158% |
| stage 2: 46 + own points, w 1.0 | −0.016 [−0.135, +0.052] | **+0.241** [+0.149, +0.341] | −0.345 | −0.433 | −0.069% |
| **replace: v2's learner refitted with 46 + own points, w 0.5** | **−0.087 [−0.291, −0.038]** | **−0.078 [−0.122, −0.037]** | **−0.354 [−0.503, −0.244]** | **−0.394 [−0.512, −0.302]** | **−0.186%** |

The replace variant clears the GO bar (equal-footing mean −0.186% <= −0.05%, no month worse, all four intervals below zero). Against
stack_resid_v2's own development output (`base2`) the same variant reads −0.087 / −0.083 / −0.355 / −0.405%.

## Where the gain sits (replace variant; share of each month's v61 squared error removed)

| | January | July | November | December | rows | gain / squared change |
|---|---:|---:|---:|---:|---:|---:|
| identified, own motion seen | 0.151 | 0.127 | 0.617 | 0.659 | 373,721 | 2.70 |
| identified, no motion point | 0.022 | 0.027 | 0.083 | 0.106 | 109,800 | 2.54 |
| not identified | 0.001 | 0.001 | 0.008 | 0.021 | 188,907 | 0.41 |
| rows inside tunnel 17's own-motion trigger (approx.: same stand, first motion <= 2,700 s, v61 below it) | 0.064 | 0.087 | 0.225 | 0.225 | 28,054 | 3.73 |
| all other rows | 0.109 | 0.068 | 0.482 | 0.561 | 644,374 | 2.10 |

- By airport every equal-footing cell gains except LIRF (January −0.010, July −0.046 points of the month) and LEMD / LTFM January / July
  (±0.02); the largest are EGLL (0.09 / 0.06 / 0.24 / 0.38 points), EDDF, LFPG and LSZH.
- About a third of the gain overlaps tunnel 17's hard own-motion bound (declared separately); two thirds come from rows the bound does not
  touch, so the two are complementary but not additive one for one: a declared round should be scored on the chain with the bound in it.

## Coverage (ranking, read only; `rank_coverage.json`)

Inbound identified / first own motion seen: EDDF 0.87-0.92 / 0.65-0.68 (2025 dev 0.81-0.91 motion), EGLL 0.92-0.94 / 0.78-0.94 (dev
0.25-0.62), EHAM, EDDM, LEBL, LSZH 0.86-0.94 / 0.85-0.93, LEMD 0.88-0.90 / 0.60-0.88, LFPG 0.66-0.70 / 0.12-0.14 (dev 0-0.26), LIRF
January 0.88 / 0.64 and July 2026 0.16 / 0.001 (dark), LTFM 0-0.03 / 0. EGLL's 2026 traces are denser than 2025's and EDDF's sparser;
both lie inside the range of 2025's dev months, but a declared round should read a coverage-shift check.

## Verdict

**GO** for a declared round: the replace variant (stack_resid_v2's learner refitted with these 19 guarded features, w 0.5). Open before
a declaration: the twelve-month check (the stacking components exist only for the four dev months), the ranking-side features through
the stacking pipeline (`tools/run_stack_resid.py` ranking table) and the squared change on v61, the order against the own-motion bound,
and the coverage shift at EGLL / EDDF.
