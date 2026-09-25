# X-Plane stand catalogue probe (25 September 2026): null, lever closed

**Question.** Our matcher learns stand positions only from gate-classified ADS-B segments
(`learn_stands`), so LFPG has 3 learned stands and `adsb_first_dist_stand_m` is empty on almost every LFPG match
(`docs/adsb_coverage/README.md`). Does an external catalogue give LFPG valid stand positions, and does the resulting first-point
distance carry signal the residual stage has not already used?

**Result: no-go under the declared rule.** The catalogue's LFPG positions are right: they agree with OpenStreetMap to a
4.6 m median. But LFPG aircraft are first seen by ADS-B about 1.8 km from their stands. The distance only says how far along
the taxi the aircraft was when first seen, and the residual stage already knows that from the first point's timing. On
19,375 LFPG dark matches from 2025 it lowers the residual path's out-of-month RMSE by **0.32%** (bar: 3%). The
optimistic recipe-month bound is **−0.014%** (bar: −0.05%). Nothing was fitted, exported or uploaded.

## Source and licence

X-Plane Scenery Gateway airport data, **GNU GPL version 2 or (at your option) any later version**, copyright the Gateway
contributors, notice kept on redistribution. It is compatible with the GPLv3 release, but not used by any submission.
`tools/pull_xplane_gateway.py` fetched, for all ten airports, the latest version approved before 2025-01-01 (`pre_data`,
the primary catalogue) and the version recommended at fetch time. Receipts: [`sources.json`](sources.json); cache
`data/external/xplane_gateway/` (git-ignored). Stand positions are `apt.dat` row code 1300 (startup locations).
LFPG has 513 named locations, identical in the 2024 and 2026 versions (0 moved).

## Declaration

[`configs/xplane_stands_probe_v1.json`](../../configs/xplane_stands_probe_v1.json), 01:47 UTC, commit `7245dfc`, before any
distance, count or label was computed. One name normalisation for every airport (lead words `Stand`/`Gate`/`Ramp`, size
codes and spaces dropped, leading zeros removed: `Stand W 42` → `W42`, `S06` → `S6`), exact keys only.
Tool: `tools/probe_xplane_stands.py`; results: [`probe_results.json`](probe_results.json).

One deviation, recorded in the results: the residual stage's tables omit some departures (2,502 in June 2025), which
therefore have no out-of-month prediction. Part C excludes the 156 LFPG dark matches affected (June 30, August 35,
September 36, October 55).

## Part A: position controls (declared rule fails)

Learned ADS-B stand positions against the catalogue position with the same name (pass: ≥ 30 stands, median ≤ 25 m,
≥ 80% within 40 m):

| airport | stands compared | median | within 40 m | passes |
|---|---:|---:|---:|---|
| EDDF | 158 | 11.4 m | 72% | no |
| EDDM | 123 | 20.0 m | 80% | yes |
| EGLL | 149 | 67.1 m | 42% | no |
| EHAM | 162 | 12.3 m | 70% | no |
| LEBL | 128 | 9.3 m | 70% | no |
| LEMD | 160 | 12.8 m | 69% | no |
| LIRF | 41 | 66.3 m | 46% | no |
| LSZH | 40 | 13.7 m | 57% | no |

1 of 8 pass, so the catalogue is not accepted. Most names land within about 10–20 m, but 20–58% sit far away. These
controls cannot say which side is wrong: a learned position is the median anchor of gate segments, and at some stands
those anchors are not at the stand. At LFPG, the 350 gate-class anchors whose stand resolves (of 367) sit a **median 2,080 m** from their
named stand (5% within 40 m).

## Part B: LFPG coverage (label-free)

| month | LFPG departures | name resolves | matches | gate | dark | dark, resolved | dark first point, median | ≤ 40 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-06 | 20,887 | 93.7% | 522 | 11 | 511 | 489 | 1,761 m | 0 |
| 2025-07 | 21,712 | 92.3% | 1,569 | 41 | 1,528 | 1,419 | 1,984 m | 9 |
| 2025-11 | 18,646 | 92.2% | 6,797 | 59 | 6,738 | 6,084 | 1,901 m | 17 |
| 2025-12 | 19,778 | 92.3% | 4,773 | 39 | 4,734 | 4,278 | 1,852 m | 11 |
| 2026-01 | 18,129 | 91.9% | 3,726 | 34 | 3,692 | 3,369 | 1,661 m | 7 |
| 2026-07 | 21,743 | 91.9% | 4,589 | 51 | 4,538 | 4,157 | 1,699 m | 18 |

(January–May 2025 have no LFPG matches; August–October are in `probe_results.json`.) The name resolves for 92–94% of
departures. Hardly any dark match starts within the matcher's 40 m stand radius, so a catalogue-based gate rescue would
convert about 10 rows a month.

## Part C: signal on the residual stage (declared rule fails)

Target: what the residual stage's twelve-month leave-one-month-out prediction leaves, `label − (v9 + ½ adj2 + ½ cat)`
(`runs/adsb_v25/dev/predictions_<month>.parquet`). The distance correction `f1` is a shrunk decile mean, compared with
the intercept-only `f0`, both fitted on the other months.

| month | rows | RMSE f0 | RMSE f1 |
|---|---:|---:|---:|
| 2025-06 | 459 | 302.37 | 303.11 |
| 2025-07 | 1,419 | 216.80 | 215.15 |
| 2025-08 | 1,609 | 229.22 | 228.71 |
| 2025-09 | 1,933 | 218.05 | 218.36 |
| 2025-10 | 3,592 | 190.90 | 190.22 |
| 2025-11 | 6,084 | 236.85 | 235.62 |
| 2025-12 | 4,278 | 218.70 | 218.27 |
| pooled | 19,375 | 222.82 | 222.10 (ratio 0.9968) |

f1 is better in 5 of 7 months (rule: 75%). The recipe-month bound, assuming the residual path's whole reduction carried
into the recipe, is −0.0037% / −0.0304% / −0.0078% (Jul / Nov / Dec), mean −0.014%. **go_for_screen: false.**

## Descriptive checks after the decision

[`descriptive_checks.json`](descriptive_checks.json), `tools/describe_xplane_stands.py`; these change nothing in the decision.

- **apt.dat against OpenStreetMap** (the source-geometry campaign's cached current parking positions): LFPG 437 common
  stands, median **4.6 m**, 99.1% within 100 m, none beyond 500 m; LIRF 136 stands, median 6.9 m. Two independent maps
  agree, so the catalogue's name → position mapping is right. At LFPG and LIRF, part A's failures come from the
  ADS-B-learned side (LIRF: 66 m against learned stands, 6.9 m against OSM), and the null in part C is not caused by
  wrong positions.
- **Own stand against a shuffled stand** (LFPG, Nov/Dec 2025 and Jan/Jul 2026, 18,062 matches): the first point is closer
  to its own stand than to a random row's stand 62% of the time (medians 1,797 m against 2,163 m). The supplied names do
  carry location, but LFPG's first sightings are nearly two kilometres out. LFPG's 174 gate-class anchors sit 1,882 m
  from their stand: at LFPG, "gate" means a stationary run on the taxiways, not at the stand.

## What this closes

- apt.dat adds nothing the models can use at LFPG: the aprons are dark to ADS-B, so stand position cannot convert dark
  rows. This matches the OPDI probe (`docs/opdi/README.md`, same apron blind spots).
- At the other airports the catalogue could replace learned stands. But those airports already have 120–190 learned
  stands, and the v19 screen's stand-distance imputation was a no-go.
- Static stand geometry was already null (Rome geometry, +0.007%). The apt.dat lever is closed on evidence; v26 remains
  the incumbent.
