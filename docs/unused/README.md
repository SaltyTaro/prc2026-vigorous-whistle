# Unused-information census (fork U, 7 October 2026) — no GO

Question (user: "try out things you have not tried yet"): which supplied or cached fields never enter v69's chain, and does any carry
signal on v69's development residual? Read-only; nothing declared, built or uploaded.

## Supplied fields (30 columns, DEP and ARR rows): every one is consumed

| Field(s) | Where it enters |
|---|---|
| MVT_ID_mvt, FLIGHT_ID_mvt | keys, flight-group protection, no-NM routing (`src/prc2026/features.py`); MVT_ID order screened |
| FLIGHT_mvt | flight prefix category (`prc2026_phase3`), ADS-B key for no-NM rows (v59 rematch); number history closed |
| FLIGHT_RULE_mvt / _flt, ADES_FILED_flt, AIRCRAFT_TYPE_flt / _mvt | phase-1 categories and phase-3 mismatch flags |
| ADEP_mvt, ADES_mvt, PHASE_mvt | airport, routing, destination; arrivals feed the surface / context features |
| MVT_TIME_UTC_mvt | anchor of every timing feature (take-off is known) |
| BLOCK_TIME_UTC_mvt, TAXITIME_SEC_mvt | departure: label only; arrival: in-block and taxi-in in the context network's tokens (`src/prc2026_ctx/tokens.py`), stand / turnaround features, own points |
| SCHED_TIME_UTC_mvt, LOBT / IOBT / EOBT_1 / ARVT_1 / AOBT_3 / ARVT_3 _flt | schedule and NM clock families (`prc2026_phase3/4`, `nmclock`, `timeline`) |
| RUNWAY_mvt, STAND_mvt | categories, runway service / headways (`phase5r`), surface counts (`timeline`), stand history |
| CALLSIGN_flt | ADS-B matching key and identity |
| ADEP_flt, ADES_flt, MARKET_SEGMENT_flt, FLIGHT_TYPE_flt, WK_TBL_CAT_flt, AIRCRAFT_OPERATOR_flt | phase-1 categories (`CATEGORY_SOURCES`) |

## Cached external fields

| Source | Status |
|---|---|
| adsb.lol surface points: position, gs, ground flags, callsign | matcher, features_v19 / own points, context network, bounds |
| adsb.lol registration / type | identity (intruder identity) only; no taxi-time mechanism |
| **adsb.lol squawk** | **extracted by `tools/adsb_extract_day.py`, never used anywhere** -> probed below |
| OPDI events | LSZH stacker (v68); other airports / sequences closed |
| IEM METAR | weather package; remaining fields closed (`docs/weather_ceiling/`) |
| EUROCONTROL daily: all-causes pre-departure delay, IFR counts | v48 reconciliation, extreme-day, var-alloc, stacker day features |
| EUROCONTROL daily: ATC pre-departure delay, slot adherence, arrival ATFM delay | screened, closed (`docs/clean_line_screens/README.md` section 6, `docs/scout/README.md`) |
| EUROCONTROL monthly taxi-out / taxi-in additional time | v47 (not uploaded), published-means screens closed; taxi-in is arrival-side |
| OSM / OurAirports / X-Plane | runway bearings used; LIRF geometry closed, LFPG rejected, X-Plane closed |

## Squawk probe (`tools/unused_squawk_probe.py`, two development days 15 January and 15 July 2025, EDDF / EGLL / LFPG / LTFM)

Mechanism tested: the crew sets the assigned squawk at clearance, shortly before off-block; a change seen while the aircraft is still
parked could time the recorded off-block better than the first motion. On gate-traced rows the recorded off-block precedes the
first >= 3 kt point by a median 182 s (IQR 61-427 s).

- A squawk change inside the parked segment is visible on only 25 of 314 gate-traced rows (8%), 1.4% of the probed departures;
  aircraft are mostly invisible at the stand (6% show any parked point).
- On those 25 rows v69's development value already has RMSE 91.8 s (median absolute error 40 s); the squawk estimate (change + 12 s)
  has RMSE 1,592 s (median 199 s), the motion estimate (motion + 182 s) 1,388 s (median 133 s).
- Bound: even a perfect correction of that 1.4% of rows at 92 s RMSE moves about 0.2% of squared error (about 0.1% RMSE), below
  the screen bar, and the learner could not reach it. **Closed without a full screen.**

## Verdict

No unused supplied field exists; the only never-used cached field (squawk) has no usable signal. No GO, no declaration. arcw1 was
started for the screen and deallocated unused; the probe read two local surface files (seconds).
