# OPDI flight events as a surface-data source (probe of 18 September 2026)

Read-only probe on one file; no model, no submission, no December label read.

## Source

Flight events v0.0.2, continuous 10-day parquet files,
`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{start}_{start+10d}.parquet`
(168 files linked on https://www.opdi.aero/flight-event-data, from 2022-01-01 to the
file ending 2026-08-08, so both ranking months are covered; ~250 MB each). File used:
`flight_events_20251101_20251111.parquet` (8,855,673 events; stored under
`data/external/opdi/flight_events/`, git-ignored). Columns: `id`, `flight_id`, `type`,
`event_time`, `longitude`, `latitude`, `altitude`, `source` (OSN), `version`, `info`
(JSON; for surface events `osm_id`, `osm_aeroway`, `osm_ref`, `osm_airport`,
`opdi_time_in_use_s`, `osn_icao24`, `osn_flight_id` = callsign).

Event types in the file: `level-start/end`, `first_seen`/`last_seen`, `first/last-xing-fl50/70/100/245`,
`top-of-climb`, `top-of-descent`, `entry/exit-runway`, `entry/exit-taxiway`,
`entry/exit-parking_position`, `entry/exit-apron`, `entry/exit-threshold`, `take-off`,
`landing`, `entry/exit-hangar`, `entry/exit-deicing_pad`. Surface events come from
H3-resolution-12 matching of OSN positions against the OSM airport layout
(https://euctrl-pru.github.io/OPDI/pipeline_overview.html, Stage 04).

## Result 1 — no surface coverage at LFPG, LTFM or LIRF

Distinct flights with any surface event at the ten airports in the ten days, and the
event counts that matter for a departure:

| airport | flights | entry-taxiway | exit-parking_position | exit-apron | entry-runway |
| --- | ---: | ---: | ---: | ---: | ---: |
| EDDF | 11,042 | 26,306 | 5,707 | 1,159 | 11,032 |
| EDDM | 5,522 | 7,205 | 404 | 410 | 5,150 |
| EGLL | 5,920 | 1,288 | 3 | 3 | 5,927 |
| EHAM | 10,584 | 1,431 | 2 | 15 | 11,442 |
| LEBL | 5,182 | 2,578 | 4 | 2 | 5,262 |
| LEMD | 10,204 | 39,638 | 2,883 | 888 | 12,210 |
| **LFPG** | 1,044 | **82** | **5** | 7 | 1,049 |
| **LIRF** | 506 | 191 | 0 | 0 | 509 |
| LSZH | 6,316 | 32,666 | 8,583 | 3,525 | 6,991 |
| **LTFM** | 15 | 2 | 0 | 0 | 16 |

`first_seen` events on the ground inside the aerodrome box (altitude within 300 ft of
field elevation), ten days: LFPG 171, LTFM 0, LIRF 190, EGLL 406, EHAM 679, LEMD 1,019;
`last_seen` on the ground at LFPG 3,771 — OSN sees LFPG arrivals down to the runway but
almost nothing on the aprons, exactly the "arrival-only" pattern of adsb.lol. OPDI cannot
supply LFPG pushbacks (≈17 departures a day out of ≈600 have any ground first-seen).

## Result 2 — at the covered airports OPDI mostly duplicates adsb.lol, with a noisier stand exit

Joined to the 55,103 departures of 1–10 November 2025 by airport + callsign
(`osn_flight_id`) with the OPDI runway entry (or take-off) within [takeoff − 900 s,
takeoff + 60 s]; adsb.lol status from `matches_v11/matches2_2025-11.parquet`:

| airport | OPDI runway | OPDI stand exit | OPDI taxiway entry | adsb.lol matched | adsb.lol gate | OPDI stand exit where adsb.lol has no gate segment | OPDI surface where adsb.lol unmatched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EDDF | 0.90 | 0.15 | 0.49 | 0.96 | 0.40 | 0.08 | 0.02 |
| EDDM | 0.57 | 0.04 | 0.16 | 0.97 | 0.66 | 0.01 | 0.00 |
| EGLL | 0.49 | 0.00 | 0.00 | 0.59 | 0.14 | 0.00 | 0.00 |
| EHAM | 0.78 | 0.00 | 0.07 | 0.96 | 0.66 | 0.00 | 0.00 |
| LEBL | 0.67 | 0.00 | 0.21 | 0.97 | 0.31 | 0.00 | 0.01 |
| LEMD | 0.81 | 0.15 | 0.73 | 0.96 | 0.24 | 0.08 | 0.02 |
| LFPG | 0.14 | 0.00 | 0.01 | 0.33 | 0.00 | 0.00 | 0.00 |
| LIRF | 0.03 | 0.00 | 0.00 | 0.54 | 0.07 | 0.00 | 0.00 |
| LSZH | 0.91 | **0.61** | 0.88 | 0.90 | 0.13 | **0.52** | 0.08 |
| LTFM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

On the 1,294 rows with both an OPDI stand exit and an adsb.lol gate-covered segment,
the OPDI exit sits later than the ADS-B motion start (median +134 s at EDDF, +305 s at
LEMD, +8 s EDDM, −5 s LSZH; upper quartiles +165 to +453 s) and is a much worse
pushback proxy against the label (RMSE of label − implied taxi: EDDF 476 vs 167 s,
EDDM 424 vs 112, LSZH 490 vs 209, LEMD 630 vs 603). Where adsb.lol has no gate segment
but OPDI has a stand exit (LSZH 1,789 rows = 52% of its departures, EDDF 484, LEMD
450), label − implied taxi has median 275 s at LSZH (IQR 57–447), 321 s at EDDF (IQR
91–610), 696 s at LEMD (IQR 476–883); within ±120 s for 27–28% of rows at LSZH/EDDF and
0% at LEMD. That is the information content of a dark adsb.lol segment, not of a
gate-covered one.

## Verdict

OPDI is not the LFPG/LTFM/LIRF lever: its OSN feed has the same apron blind spots as
adsb.lol. Its only distinct contribution is a stand-exit timestamp for about half of
LSZH's departures (LSZH holds ≈1.8% of the incumbent's development SSE) and ≈8% of
EDDF's and LEMD's, worth at most a few tenths of a percent of RMSE. If a feature round
is run on the covered airports, three OPDI features (stand exit, first taxiway entry,
their presence) can join it as one candidate family; on its own it does not justify the
≈10 GB download of the 13 months or a declaration. Licence: open data with attribution
(OPDI / EUROCONTROL PRC / OpenSky Network) — record the attribution if used.
