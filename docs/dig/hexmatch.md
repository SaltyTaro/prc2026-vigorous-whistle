# Dig tunnel 12: ADS-B rematch by ICAO address — no GO, closed (3 October 2026)

Question: are departures at covered airports left unmatched because the transponder broadcast a stale or blank callsign on the ground,
so that a rematch by ICAO address (tunnel 10's identity table) would recover their taxi, as v59's flight-number rematch did?
Label-free census on 2025 and the 2026 ranking; labels only for the side check. Nothing declared, built or uploaded. Scratch:
`runs/dig/hexmatch/` (`census.py`, `reseg.py`, `spurious.py`, tables `census.parquet`, `reseg.parquet`, `spurious_sample.parquet`).

## The hypothesis is already handled by the shipped matcher

`src/prc2026_adsb/features.py::match_departures` finds the candidate ICAO addresses through any point carrying the departure's
callsign at that airport that day (climb-out included), then takes **all** ground points of that address in the 4 h before take-off,
whatever callsign they carry. A stale or blank ground callsign therefore does not stop a match.

## Census: identified at take-off but unmatched (EDDF / EGLL / LFPG, the airports of the identity table)

| airport | year | departures | unmatched | identified | identified and unmatched | of those: >= 3 ground points and climb-out | segment() returns a departure segment | gate segment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 2025 | 230,141 | 9.4% | 94.2% | 8,384 | 60.6% (5,083) | 25% | 6% |
| EDDF | 2026 | 36,317 | 22.2% | 96.6% | 6,936 | 33.4% (2,315) | 8% | 1% (24 rows) |
| EGLL | 2025 | 239,546 | 57.0% | 94.7% | 125,209 | 4.4% (5,460) | 0% | 0% |
| EGLL | 2026 | 39,840 | 7.2% | 97.3% | 1,735 | 27.4% (475) | 0% | 0% |
| LFPG | 2025 | 239,552 | 90.9% | 64.8% | 133,405 | 1.2% (1,558) | 0% | 0% |
| LFPG | 2026 | 39,872 | 79.2% | 80.3% | 23,623 | 4.1% (963) | 0% | 0% |

- Where these rows have ground points at all, they mostly carry **another** callsign (EGLL 2026 37%, LFPG 33%, EDDF 45% "other only"):
  the aircraft's own **arrival** taxi-in under the previous leg's callsign. The unchanged `segment()` cuts at the arrival, finds no
  departure points after it and returns nothing at EGLL / LFPG.
- At EDDF the re-segmented tracks give 24 gate segments in the whole ranking (14 in January, 10 in July) and a few dozen dark ones:
  nothing to declare. EDDF's 2026 unmatched share (22% against 9%) is receiver coverage, as on record.

**Closed:** unmatched rows at covered airports lack departure-taxi coverage, not a key.

## Side findings (small, closed)

- **Day boundary in the matcher.** Matching loads only the take-off day's surface file, whose points cover that UTC day exactly; a
  departure taking off shortly after 00:00 UTC loses ground points from before midnight. Ranking departures in the first hour of a UTC
  day: LTFM 535 (no ADS-B anyway), LEMD 243, LFPG 131, LEBL 92, LIRF 27, EHAM 25, others 0-1. Even if every one of the covered ones
  became matched, the effect is well under 0.01 s.
- **Spurious "airborne" points during taxi.** `segment()` starts a departure after the last airborne point before take-off - 60 s; a
  slow (< 60 kt) airborne point within the hour before take-off truncates the segment for 0.1-0.7% of matched departures on four sample
  days (EHAM highest, 16 of 2,447). On the development chain those 33 rows have RMSE 107 s against 162 s for the other matched rows
  (bias +31 s): no excess error to recover.
