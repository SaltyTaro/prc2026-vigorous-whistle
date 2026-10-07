# Source-class sweep 2: off-block and pushback timestamps (7 October 2026)

This was a bounded web-only check from 12:22 to 12:55 UTC, with no compute and no model change. The question: is
there an **openly licensed** source, other than the classes already covered, that gives a timestamp of off-block,
pushback, door closure or taxi start? It could be per departure or per stand, terminal or hour, or it could reveal the
mode of the airport record. Priority airports were LFPG, LTFM and LIRF (July 2026), then the dark rows at EGLL, LEBL,
LEMD and EDDF. A source also had to cover 2025 (for validation) and January or July 2026.

The sweep followed the lesson of [`docs/oooi_source/`](../oooi_source/README.md): survey source **classes**, not
datasets. Classes already covered were not surveyed again: ADS-B and MLAT networks, the adsb.lol data-link archive,
airframes.io, OPDI, EUROCONTROL AIU workbooks, OSM / X-Plane / OurAirports, IEM / ERA5, UK CAA, airport and national
portals with monthly totals, and OpenSky ([`docs/adsb_coverage/`](../adsb_coverage/),
[`docs/opensky/`](../opensky/README.md)). All reads were public and sequential. No account, credential, form or contact
was used. No bulk download was made: the largest reads were three 0.5 MB OSM trackpoint pages and one 77 kB archived
web page, kept outside the repository.

## Verdict: no usable source

No candidate passes all four gates: open licence, a per-departure ground timestamp, 2025 history, and January or July
2026. Every class falls into one of three groups.

* **The signal exists but the licence fails.** ATC audio has it, and so do airline flight-status APIs.
* **The licence is open but the content cannot time an off-block.** This group covers noise maps, satellite images,
  emissions inventories and national catalogues.
* **Nothing exists.** A-CDM and SESAR datasets, scraped flight-board archives, NOTAM history and research
  repositories have nothing for the ten airports in 2025–2026.

**Closest near-misses:**

1. **ATC voice archives (LiveATC.net).** A pushback or start-up clearance on the ground frequency is a per-flight event
   timed to the second, and the archive claims 4–5 years of retention. Three things block it:
   * **No open licence.** The site sells audio data, and scripted reads of its terms and feed lists return HTTP 403.
   * **No feeds at the priority airports.** "the UK, Spain, and France ... don't allow third-party rebroadcasting of
     ATC frequencies", which rules out LFPG, EGLL, LEBL and LEMD. A listing archived in August 2025 shows "no public
     feed for LIRF", for the same reason. LTFM and EDDF were not confirmed.
   * **It is the ATC side, not the airport record.**
2. **Airline flight-status APIs** (the Air France-KLM API is branded "Open Data"; Lufthansa and Turkish Airlines run
   similar developer APIs). They carry actual times of the hub carrier's own flights. Blocks: access is "accessible
   following registration" with an API key, no licence is stated, and history back to 2025 is not documented. No
   account was created to test it.
3. **OpenStreetMap public GPS traces.** These are timestamped traces on the aprons, some from passengers in taxiing
   aircraft. In January and July of 2025 and 2026 there are at most one or two departures per airport-month: LTFM has
   one taxi-like trace on 13 July 2025 and a 10-point one on 28 July 2026, and LFPG and LIRF have none. The licence of
   the traces themselves is not stated.

## Sources checked

| # | Class | Source | What it holds / time resolution | 2025 | Jan / Jul 2026 | Priority airports | Licence (quoted where stated) | Verdict, reason |
|---|---|---|---|---|---|---|---|---|
| 1 | ATC voice archives | LiveATC.net ([AIN interview](https://backend.ainonline.com/aviation-news/air-transport/2025-12-15/how-liveatc-went-live), [LIRF listing note](https://rotatepilot.com/live-atc/lirf)) | Recorded ground / tower audio; pushback clearance to the second | Retention 4–5 years claimed | as 2025 | None at LFPG, LIRF, EGLL, LEBL, LEMD; LTFM / EDDF unconfirmed | None open: "the sale of audio data"; "the UK, Spain, and France ... don't allow third-party rebroadcasting" | **NOT USABLE**: licence, no feeds, ATC-side event |
| 2 | ATC speech corpora | ATCO2 and LiveATC-derived sets on Hugging Face | Transcribed ATC speech | No (recorded 2020–2022) | No | No | Mostly untagged or "other" | **NOT USABLE**: wrong years and airports |
| 3 | Airline flight-status APIs | [Air France-KLM Flight Status](https://developer.airfranceklm.com) (also LH, TK) | Actual times of own flights; minutes | Not documented | Not documented | AF at LFPG | None stated; "accessible following registration"; API key | **NOT USABLE**: registration, no licence, no history |
| 4 | Archived flight boards | Wayback captures of parisaeroport.fr / istairport.com / adr.it | One instant per capture | 0–1 captures a month | IST 6 / 2, CDG 2 (Jul); ADR times out | LFPG, LTFM, LIRF | None (third-party copyright) | **NOT USABLE**: the 18 Jul 2026 CDG capture holds no flight number or time (board loads by API) |
| 5 | Git-scraped boards | GitHub repository search | Two Heathrow scrapers (2009, 2017), live chat apps for IST | No | No | No | n/a | **NOT USABLE**: nothing archived |
| 7 | Dataset hubs | Hugging Face ([flight-delays-europe-2023-2025](https://huggingface.co/datasets/345rf4gt56t4r3e3/flight-delays-europe-2023-2025)) | OPDI flight-list fields (first_seen / last_seen); ADS-B sets are adsb.lol mirrors | 2023–2025 | No | sighting only | "apache-2.0" tag on OPDI-derived re-upload | **NOT USABLE**: no off-block, no 2026 |
| 8 | Research repositories | Zenodo (API, 15 queries) | IST triple-runway CO2 set (no files), global hourly ADS-B aggregate | Aggregates | No | No | CC BY 4.0 | **NOT USABLE** |
| 9 | Research repositories | Figshare, Dryad, Mendeley Data | Nothing relevant | No | No | No | n/a | **NOT USABLE** |
| 10 | Surface benchmarks | Amelia-48, TartanAviation | Surface tracks, ATC speech | US only | No | No | research releases | **NOT USABLE**: wrong region |
| 11 | Noise monitoring | Bruitparif ([SURVOL](https://survol.bruitparif.fr/), data.gouv maps) | Overflight noise events; open data = Ln maps | Viewer; maps 2020 | Viewer | LFPG (airborne) | Maps "lov2" (Licence Ouverte 2.0); event exports unreadable | **NOT USABLE**: airborne flyovers after take-off |
| 12 | Noise monitoring | [Aeroporti di Roma](https://www.adr.it/rumore-aeroportuale) | Annual reporting only | No | No | LIRF | none found | **NOT USABLE** |
| 13 | Noise monitoring | Istanbul Airport (İGA / DHMİ) | EIA chapter only | No | No | LTFM | none found | **NOT USABLE** |
| 14 | Noise / track viewers | Frankfurt INAA, city analyses | Viewer, monthly PDFs | viewer | viewer | EDDF | none found | **NOT USABLE** |
| 15 | A-CDM / SESAR / Horizon | Research using TOBT / TSAT / AOBT; ENAV FCO A-CDM leaflet | Non-public samples (one week at Schiphol) | No dataset | No | No | n/a | **NOT USABLE** |
| 16 | Slot coordinators | COHOR (also ACL, Assoclearance) | Start-of-season statistics | season | season | LFPG | none stated | **NOT USABLE**: aggregates |
| 17 | National / EU catalogues | data.europa.eu (all EU national portals), datos.gob.es, GovData, data.gouv.fr, ENAIRE, ENAC / ENAV, DHMİ | Monthly totals (e.g. Barcelona monthly operations by airline), noise maps, AIP sets | monthly | monthly | totals | CC BY 4.0 / dl-by-de 2.0 / Licence Ouverte on aggregates | **NOT USABLE**: no per-flight or per-hour ground record |
| 18 | NOTAM archives | current-NOTAM sites, paid APIs, EAD | Closures, validity periods | no open archive | no | – | none open | **NOT USABLE**: no history, wrong event |
| 19 | Satellite imagery | Copernicus Sentinel-2 | One 10 m image per pass, 5-day revisit | yes | yes | all | "free, full and open access" | **NOT USABLE**: one instant per few days |
| 20 | Webcams | [nyuuzyou/airportwebcams](https://huggingface.co/datasets/nyuuzyou/airportwebcams) and the feeds it lists | URL index, no images or times | no archive | no archive | index only | index CC0; feeds keep owners' terms | **NOT USABLE** |
| 21 | Crowd GPS traces | OSM trackpoints API at the aprons | Timestamped traces, seconds | LTFM 1 (13 Jul) | LTFM 1 short (28 Jul) | LFPG / LIRF: 0 in Jan / Jul | not stated for traces | **NOT USABLE**: ≤ 2 departures per airport-month |
| 22 | Emissions inventories | Climate TRACE (also EU ETS, CORSIA) | Monthly per airport | monthly | monthly | all | CC BY 4.0 | **NOT USABLE**: monthly, modelled |
| 23 | Organiser / team pages | [data page](https://prc-data-challenge-2026.netlify.app/data.html), a team page | Same three files; no external sources listed | – | – | – | – | **NOT A SOURCE** |

The following were considered but not checked online, because they fail on content:

* **Citizen sensors.** Seismic networks (Raspberry Shake) and citizen noise sensors (Sensor.Community) cannot see a
  pushback or identify a flight.
* **CC-licensed photo archives** (Wikimedia Commons, Flickr) hold a few pushback photos a year, with unreliable EXIF
  times.
* **Ground-vehicle ADS-B** (tugs) is heard by the same receivers as the aircraft, so it is dark where the aircraft are.
* **SATCOM ACARS** exists only in airframes.io, which is not open. HFDL is already in the adsb.lol archive.
* **Airport A-CDM manuals and AIP AD 2.20 texts on how AOBT is captured** are documentation, not per-flight data.
  Stand-level biases were already closed ([`docs/dig/README.md`](../dig/README.md)).

## Receipts

[`receipts/licence_excerpts.txt`](receipts/licence_excerpts.txt) holds every licence and content quote with its URL and
retrieval time. Lines marked "(search summary)" come from search-result summaries, not from the page itself. One search
log per catalogue sits beside it:

* `kaggle_hf_search.txt`
* `zenodo_search.txt`
* `figshare_dryad_search.txt`
* `data_europa_eu_search.txt`
* `national_portals.txt`
* `datagouv_noise_trajectories.txt`
* `github_repo_search.txt`
* `wayback_fids_cdx.txt`
* `osm_gps_traces.txt`

The machine-readable table is [`sources.json`](sources.json).

## What a 2025 validation sample would need (none applies)

None of the sources clears the licence gate together with per-departure content, so no validation sample was
prepared. Nothing in this folder changes a model, a file or an upload.
