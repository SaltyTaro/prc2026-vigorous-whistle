# Dig tunnel 19: case files, round two (3 October 2026, 2025 only, nothing declared)

No official score, no leaderboard, no upload, nothing under runs/submissions. Labels are read only to pick, classify and score development
rows. Scratch (git-ignored): `runs/dig/case_files2/`.

| script | output |
|---|---|
| `build.py` | the 200 rows |
| `context.py` | per-row context |
| `dump.py` | `dump.txt` |
| `classify.py` | `mechanisms.json` |
| `state8.py` | matcher state, eight airports |
| `scan8.py` | `move8_2025-MM.parquet` |
| `validity.py` | `validity.txt` |
| `screen.py` | first screen, `screen.json` |
| `v62proxy.py` | v62 proxy |
| `screen62.py` | governing screen, `screen62.json` |
| `rank_new5.py` | ranking count, `rank_new5.parquet` |

## The rows

The rows come from the eight airports with ADS-B surface coverage: EDDF, EGLL, LFPG, EHAM, LEMD, LEBL, LSZH and EDDM. Each row has an
ordinary label (at most 3 h). Excluded:
- tunnel 17's 200 rows;
- LIRF population rows and LIRF fill rows (block within 5 s of schedule);
- no-NM rows;
- extreme published days (|implied day shift| >= 80 s or |discrepancy| > 1e5 s);
- rows the own-motion bound v1 fires on.

There are 200 rows, with 1.34e9 squared error:
- 140 from January, July, November and December 2025, the worst by the v61 development proxy. They hold 6.0% of the eight airports'
  ordinary squared error after the exclusions.
- 60 from the other eight months, ranked on the twelve-month predictor.

Each row was read with:
- every supplied field and its clock intervals;
- the stand's arrivals and departures over 12 h, with type and identity marks;
- the runway's and the airport's take-off sequence and gaps;
- the aircraft's whole turnaround in adsb.lol, compressed into still / moving / airborne runs relative to take-off, including the
  points before the matcher's last coverage gap;
- other aircraft stationary within 60 m of our parking fix.

Identity comes from tunnel 11's tables (`runs/dig/position/ids_*.parquet`), which cover all eight airports; 84% of the rows are
identified.

## Mechanisms

Classes are assigned in priority order, with labels used as a diagnostic only. The share is of the 200 rows' squared error.

| mechanism | rows (dev months) | share | label-free trigger? |
|---|---:|---:|---|
| early record: block well before the NM clock, no own ground motion seen | 60 (38) | 33.5% | no (tunnel 4) |
| early record: block about at schedule, NM clock late (LFPG 34, EGLL 7; LEBL July 2025 rows sit on the schedule) | 49 (27) | 28.6% | no (fill-like; oracle-small, closed in September) |
| early record, where ADS-B shows the real move late (first motion far below the label; LEBL 22, LFPG 9) | 45 (39) | 16.1% | no: the motion is the real push, the record is early |
| own ground motion more than 20 min before take-off, outside v1's scope (B > 2,700 s, another airport or another stand) | 19 (13) | 10.4% | partly; see below |
| late record (block after the NM clock), over-predicted | 5 (3) | 3.6% | no |
| over-prediction, other | 10 (9) | 3.6% | no |
| own motion long before a short label (tow or reposition), over-predicted | 7 (7) | 2.2% | no |
| under-prediction, unexplained | 4 (3) | 1.7% | no |
| data anomaly (label <= 60 s) | 1 (1) | 0.3% | no |

88.5% of the rows are under-predictions. 78% of the squared error sits in the early-record classes, where the airport's off-block record
is 30–90 min before the aircraft actually moves. ADS-B confirms the late real move in 45 rows. No label-free input separates them.

The new pattern is in the fourth class. Departures at EDDM (4 Sep 2025, a 53-min departure stop), LSZH, LEMD, EDDF and EGLL show the
aircraft moving 40–170 min before take-off, with a label within minutes of that motion. Most of these have B > 2,700 s.

## Own-motion bound outside v1: validity in 2025

`scan8.py` repeats v1's trigger for every departure at the eight airports that the shipped matcher does not trace from the gate. The
trigger is:
- the identified inbound of the same aircraft blocked in 600 s to 18 h before take-off;
- the first ground point at >= 3 kt after in-block + 120 s gives B.

It adds two discriminators: the inbound's parking fix, and "return" (the aircraft is still within 60 m of that fix more than 300 s after
the move).

**Validity** is the share with label >= B - 60 s, over all 2025 rows not traced from the gate, with no return. It comes from
`validity.txt`.

| B band | same stand | different stand |
|---|---:|---:|
| 1–20 min | 99.5% (581,204) | 99.3% (31,755) |
| 20–45 min | 98.3% (18,184) | 89.1% (1,409) |
| 45–60 min | 77.4% | 26.5% |
| 60–90 min | 14.8% | 2.4% |
| above 90 min | 0–3% | 0–1% |

- **B > 45–60 min is closed.** The long-B rows in the case files are the valid minority, and no discriminator rescues the band:
  - longest hold after the move: 25–67% valid;
  - inbound ground time: 7–60%;
  - distance from the parking fix: 39–83%.
- **Same stand, B <= 2,700 s, no return** holds at least 90% in every month at every airport (EHAM the lowest, 95.0–98.7%).
- **Different stand** holds 90–100% in every month with enough rows, but only 81.5% on the rows where the prediction is below B
  (see below).

## Screen

The governing screen is month-blocked. The control is the **v62 development proxy**, built exactly as `tools/run_own_motion.py --score`
builds it: v61 proxy -> `om_b2700_w100`, reproducing the declared January / July / November / December -0.038 / -0.110 / -0.081 /
-0.163%. The twelve-month table is `after2` -> v1 (`after3`).

- **Value:** p + w (B + o(airport) - p), where o is the mean of label - B on the other months' triggered rows of the twelve-month table,
  shrunk to the pooled mean with weight 20.
- **Excluded rows:** LIRF population and guard rows.
- **Airports:** v1's airports are excluded by construction; EDDF, EGLL and LFPG rows of other stands are a separate variant.

| variant | Jan | Jul | Nov | Dec | eq. mean | twelve |
|---|---|---|---|---|---:|---|
| **new5** (EHAM, EDDM, LSZH, LEMD, LEBL), same stand, B <= 2,700, no return, **w 0.5** | −0.037% [−0.139, −0.012] | −0.048% [−0.091, −0.018] | −0.088% [−0.133, −0.052] | −0.081% [−0.128, −0.028] | **−0.0554%** | 12/12 better, 12/12 intervals < 0, mean −0.067% |
| new5, w 1.0 | −0.023% [−0.128, +0.039] | −0.043% [−0.110, +0.008] | −0.077% [−0.146, −0.010] | −0.035% [−0.160, +0.145] | −0.0338% | 12/12, 8 intervals < 0 |
| new4 (no EHAM), w 0.5 | −0.026% [−0.093, −0.009] | −0.038% [−0.082, −0.011] | −0.069% [−0.108, −0.038] | −0.069% [−0.107, −0.027] | −0.0444% | 12/12, 12 intervals < 0 |
| EHAM only, w 0.5 | −0.011% [−0.059, +0.008] | −0.010% [−0.021, +0.001] | −0.019% [−0.035, −0.005] | −0.012% [−0.028, +0.008] | −0.0112% | 12/12 |
| all 8 airports, different stand, w 0.5 | +0.008% | +0.025% | +0.054% | +0.042% | +0.025% | 0/12 better |

**new5 w 0.5:** about 1,000–1,340 rows a month. Gain per unit of squared change is 1.44 / 1.83 / 1.78 / 1.28 in the four development
months.

By airport, over the four development months:

| airport | gain / squared change | validity of the triggered rows |
|---|---:|---:|
| EDDM | 2.76 | 98% |
| LSZH | 2.78 | 99% |
| LEBL | 1.59 | 88% |
| EHAM | 0.88 | 90% |
| LEMD | 0.38 | 88% (43 rows) |

By B band, 600–1,800 s carries most of the gain (ratio 2.2–2.5); 60–600 s is neutral.

At w 1.0 the gain ratio collapses to 0.14–0.41: the bound plus offset is too aggressive here, the aircraft often sitting near the B line.

Offsets on all twelve months: EDDM +51, EHAM −18, LEBL +152, LEMD +88, LSZH +46 s.

The first screen used tunnel 17's own reconstruction of v1 as the control and gave identical development numbers. The candidate rows are
disjoint from v1's.

**Variants screened (all listed above):**
- new5 same stand, w 0.5 / 1.0;
- different stand at all eight airports, w 0.5 / 1.0;
- new5 combined with different stand, w 0.5 / 1.0 (worse than new5 alone);
- new4 without EHAM;
- EHAM alone.

The trigger thresholds (gs >= 3 kt, in-block + 120 s, 600 s – 18 h, B <= 2,700 s, same stand) are v1's. "No return" was added while
reading these rows, before any screen.

## Ranking (read-only on `runs/submissions/vigorous-whistle_v62.parquet`)

v62 equals v61 on every trigger row; v62 changes only 155 EDDF / EGLL / LFPG rows.

**new5 w 0.5:** 2,523 rows change, squared change 2.14e7, max move +1,029 s, mean +34 s.

| airport | rows (Jan / Jul) | share of squared change |
|---|---|---:|
| EHAM | 2,002 (681 / 1,321) | 30% |
| LSZH | 258 | 23% |
| EDDM | 130 | 21% |
| LEBL | 74 | 16% |
| LEMD | 59 | 10% |

EHAM's January squared change is concentrated on 10 and 20 January (1.0e6 each), then 6 and 16 January.

| gain / squared change | expected change |
|---:|---:|
| 1.0 | −0.14 s |
| 1.5 | −0.21 s |
| development ratios 1.3–1.8 | −0.18 to −0.25 s |

new4 (no EHAM): 521 rows, squared change 1.50e7, −0.10 to −0.15 s.

No LIRF population or guard row is touched, and no LIRF row at all.

## Verdict

- **GO: own-motion bound extended to EHAM / EDDM / LSZH / LEMD / LEBL** (same stand, not gate-traced, no return to the parking fix,
  60 < B <= 2,700 s, w 0.5, applied last after v1).
  - Equal-footing mean −0.0554%; every development month better, all four intervals below zero; twelve months 12/12 better with every
    interval below zero.
  - Ranking size about −0.14 to −0.25 s.
  - It would also pass v57's four-part rule as screened.
  - Weak spot: EHAM carries 79% of the ranking rows but only 30% of the squared change, and its own evidence is thin (eq. −0.011%,
    intervals touching zero). new4 without EHAM is the conservative alternative, at −0.044%, under the screen bar.
- **Closed:**
  - long-B own motion above 45–60 min (validity 15–77%, no discriminator);
  - different-stand own motion (worse in every month);
  - the early-record classes (78% of the listed squared error; the ADS-B motion confirms the record, not the label);
  - LEBL July block-at-schedule rows (fill-like; no trigger).
