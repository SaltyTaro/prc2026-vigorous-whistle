# ADS-B surface data: OpenSky probe and adsb.lol traces (15 September 2026)

Question: can OpenSky ADS-B data supply the off-block (block time) information
that dominates the remaining ordinary-row error? Probed with the REST API on a
standard account (OAuth2 client in `.env`, never printed). No model, no
submission, no December label. Read-only, outside the competition data.

## Result 1 — the flight and track products start at lift-off (no off-block information)

`tools/pull_opensky_flights.py` pulled `/flights/departure` for EDDF and LIRF,
March 2025 (16 two-day windows each, 30 credits per window). Coverage is
essentially complete (EDDF 17,328 records for 17,636 departures; LIRF 12,237 of
12,544) and the join by ICAO callsign + takeoff window matches **96.6% / 97.6%**
of our departures (`tools/audit_opensky_flights.py`, [`flights_audit.json`](flights_audit.json)).
But `firstSeen` is the **lift-off** moment: takeoff minus firstSeen is within
±30 s for 99.9% of matches, at 4–300 m altitude. It sits a median 792 s (EDDF)
/ 1,021 s (LIRF) *after* the airport block time, carries no residual
information beyond the NM proxy (correlation 0.03 / 0.09), and an affine
combination with the NM proxy improves the in-sample RMSE by under 3 s.
`/tracks/all` for six flights of 13 September 2026 starts at the same instant
with zero on-ground waypoints. These products therefore cannot help; they only
confirm that the supplied takeoff clock is exact.

## Result 2 — OpenSky does receive surface positions at seven of the ten airports

A live `/states/all` bounding-box query at 06:5x UTC on 15 September 2026
(on-ground flag, velocity > 2 m/s):

| Airport | on ground | moving | Airport | on ground | moving |
| --- | ---: | ---: | --- | ---: | ---: |
| EDDF | 7 | 7 | LEMD | 1 | 1 |
| EDDM | 8 | 7 | LFPG | 10 | 8 |
| EGLL | 10 | 6 | LIRF | 0 | 0 |
| EHAM | 12 | 11 | LSZH | 18 | 6 |
| LEBL | 15 | 8 | LTFM | 0 | 0 |

Surface ADS-B reaches OpenSky at EDDF, EDDM, EGLL, EHAM, LEBL, LFPG and LSZH,
not at LIRF or LTFM, and barely at LEMD. Historical surface positions live only
in OpenSky's historical database (`state_vectors_data4`, Trino), which needs a
data-access grant; the REST API serves states at most one hour into the past.

## What it could be worth, and what it cannot

- The seven covered airports hold about 38% of the seasonal fold's SSE (once
  the two garbage LFPG rows are set aside) and 47% of the forward fold's. If
  first surface motion pinned the block time to within a minute, halving the
  error there would cut total RMSE by roughly 10–15% — from 267 to about
  230–240 on the ranking, i.e. the range of the current top three.
- Nothing about that can be measured without the historical database: the
  REST API cannot reach the ground phase for any past date, and there are no
  labels for live data. LIRF (our largest artifact problem) has no coverage.
- Credits used today: about 1,100 of the 4,000 daily flights/tracks allowance
  and 10 of the states allowance.

Raw downloads are under `data/external/opensky/` (git-ignored) with per-window
metadata and hashes; the OpenSky account terms of use apply to them.

## Result 3 — adsb.lol globe_history has the surface phase, under an open licence

The user directed: fetch the data first, ask the organisers only if it works.
OpenSky's historical database is not reachable with our account (Trino rejects
the client-credentials token), so the source is **adsb.lol's daily
`globe_history` archives** (GitHub releases `adsblol/globe_history_YYYY`, tag
`vYYYY.MM.DD-planes-readsb-prod-0`, 2.5–4 GB per day of readsb traces for every
aircraft the network saw; **Open Database License**, CC0 feeder contributions;
README and licence files inside every archive). `tools/adsb_extract_day.py`
streams one day from GitHub on an Azure worker (≈350 MB/s), decodes every
trace and keeps only positions inside the ten aerodrome boxes that are on the
ground or below 3,000 ft (≈20–45 s per day; 5–14 MB per day retained). Ground
coverage: EHAM, EDDM, LEBL, LSZH, EGLL strong; EDDF moderate; LEMD variable;
LFPG weak; **LIRF and LTFM essentially none** (2025-03-10 was a German strike
day and is not representative for EDDF/EDDM).

**The airport block time is the ADS-B pushback start.** Where the gate is covered the offset between
first movement and block time has an interquartile range of 17–36 s at LSZH
and EDDM. Most segments, however, begin only after a coverage gap at the gate
("dark": the aircraft is already moving); anchoring on stand positions learned
from covered segments (`learn_stands`, label-free) raises the covered share
from 13% to 33%.

`src/prc2026_adsb/features.py` derives 16 label-free features per departure
(matched by ICAO callsign and takeoff time; ~97% match rate where coverage
exists): time from first surface point and from first movement to takeoff,
first ground speed, distance of the first point from the learned stand, stand
stop duration, taxi path length, taxi stops and stopped time, slow-start
duration, coverage gap, turnaround from the previous in-block, and point
counts. A first test on seven March 2025 days (day-grouped CV, simple baseline
without the v9 stack): EDDM −38%, LSZH −30%, EHAM −26%, LEBL −22%, EDDF −19%,
EGLL −8%, LEMD 0 (32% matched), no data at LFPG/LIRF/LTFM; −6.7% overall.

Extraction of every 2025 training month and the two 2026 ranking months runs
on arcw1–arcw4 (`tools/adsb_driver.sh`); matching runs per month with
`tools/build_adsb_features.py` (pass 1, learned stands, pass 2). The fold
evaluation against v9 is `tools/evaluate_adsb_residual.py`.
