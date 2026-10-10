# Winter de-icing screen for the final's February (10 October 2026)

Three of the four final rankings include February / June 2026. February has de-icing and snow days, so the question was whether
de-icing evidence could correct the chain on winter departures. This is a label-free screen. No model was fitted, no label or
official score was read, and nothing was uploaded. Numbers: [`results_v1.json`](results_v1.json).

## Stage 1: is there de-icing evidence we can link to departures?

- **OPDI flight events (`data/external/opdi/stack_v1`, 28 windows, every month we hold: 2024-12 to 2026-07).** The filter keeps the
  surface event types, including `entry-deicing_pad` / `exit-deicing_pad`, inside the ten airport boxes. Within those boxes there are
  **zero de-icing events in every window and month**. The events present are taxiway, runway, parking position, apron, hangar,
  take-off and first-seen. In the one raw all-Europe window on disk (1-11 November 2025), OPDI v0.0.2 has 70 de-icing pad events,
  all at EDDS (62) and EDDP (8). OPDI therefore detects pads only where its map has them, and none of our airports is among them.
  With no events, the OPDI stacker's linking has nothing to link: **zero de-iced departures in any month**.
- **Our OSM cache** (`data/external/osm/source_geometry_v1`, LFPG and LIRF only) has no `aeroway=deicing_pad` element in any of its
  ten snapshots.
- **A live OSM count** of `aeroway=deicing_pad` at the ten airports was attempted once on each of two Overpass endpoints (HTTP 504,
  HTTP 500) and not pursued. Even if pads are mapped there, using them would mean building pad geometry and ADS-B pass-through
  features: a new feature campaign, not a screen.

**Stage 2 (the residual screen) was not run.** The directive runs it only if stage 1 links enough de-iced departures, and stage 1
links none.

## Context: how wintry is February 2026 compared with the months the chain was read on?

Label-free METAR census (IEM, the chain's own weather source), as % of reports in the airport-month:
- **frozen:** codes SN / SG / PL / GS / FZ;
- **cold-wet:** at or below 3 °C with any precipitation, fog or mist;
- **frost:** at or below 0 °C.

| airport | frozen Jan 25 / Nov 25 / Dec 25 / **Jan 26** / **Feb 26** | cold-wet Jan 25 / Dec 25 / **Jan 26** / **Feb 26** |
|---|---|---|
| EDDF | 2.0 / 1.2 / 0.6 / **8.5** / **4.1** | 11.6 / 3.5 / **20.0** / **11.5** |
| EDDM | 6.7 / 5.5 / 10.4 / **12.0** / **7.4** | 20.3 / 33.3 / **34.9** / **23.4** |
| EHAM | 5.9 / 2.6 / 0.3 / **13.3** / **1.5** | 25.0 / 3.4 / **30.5** / **7.0** |
| LSZH | 5.8 / 4.0 / 3.2 / **17.2** / **6.4** | 18.8 / 20.6 / **44.8** / **20.0** |
| LTFM | 0.0 / 0.0 / 1.1 / **9.9** / **5.5** | 0.5 / 0.1 / **12.9** / **5.0** |
| LFPG | 3.3 / 1.5 / 0.1 / **1.7** / **0.3** | 22.0 / 3.4 / **11.6** / **2.8** |

EGLL, LEBL, LEMD and LIRF stay below about 1% frozen in every listed month.

- **February 2026 is milder than January 2026** at every airport on all three measures. January 2026 is a scored month already on the
  board.
- **February 2026's levels sit inside the range of the labelled 2025 winter months** (January / February / November / December 2025),
  on which the chain's weather inputs were trained. Two exceptions just above that range: EDDF (frozen 4.1% against a 2025 maximum of
  2.0%) and LTFM's frozen share (5.5%), which is below its February 2025 level of 19.2%.
- The models already carry these conditions as inputs: the weather campaign's IEM features, including snow and freezing codes, and
  the ADS-B taxi-stop features on matched rows. An earlier snow screen was closed.

## Verdict

**No de-icing lever is available from open event data at our airports, and no screen against the -0.05% bar could be run.** OPDI has
no de-icing events in our airport boxes in any month, and our map cache has no pads. February 2026 is not out of distribution in
weather terms (milder than January 2026, within 2025's winter range), so there is no case for a February-specific weather or
de-icing round either. **A declared round is not worth it.**
