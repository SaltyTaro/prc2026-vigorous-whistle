# Per-flight OUT / off-block source check (ACARS OOOI and airport portals), 7 October 2026

Earlier source surveys are not repeated (ADS-B networks, OPDI, EUROCONTROL AIU
workbooks, OSM / X-Plane, IEM / ERA5, UK CAA; see [`docs/adsb_coverage/`](../adsb_coverage/)). No account, credential,
contact or form was used. About 9.7 MB of CC0 sample files were read (30 files, listed in
[`receipts/sample_scan/sample_files.txt`](receipts/sample_scan/sample_files.txt); kept outside the repository).

## Verdict: no usable source

Nothing checked passes all four gates (open licence, per-flight OUT at the three airports, 2025 history, January or
July 2026). Airport and national portals publish monthly totals at most. Community ACARS data has an open licence in one
case only (adsb.lol, CC0), and that archive has no receiver close enough to hear an aircraft on the ground at CDG,
Istanbul or Fiumicino.

**Closest near-miss: the adsb.lol Aircraft Data Links archive** (ACARS / VDL2 / HFDL, GitHub releases, **CC0 1.0**).
It clears the licence gate, and ARINC OOOI reports (labels QP = OUT, QQ = OFF, QR = ON, QS = IN) are present and easy
to decode. Three things block it:

1. **No receiver in range of the three airports' surfaces.** Six sampled windows (15 July 2026 07:16–08:55 and
   09:55–11:19 UTC; 15 July, 15 November and 15 December 2025 and 15 January 2026, about 07–09 UTC) hold 4–123 feeder
   ids each. All are in the UK, the US, Australia, Asia and some continental sites (Amsterdam, Berlin, Rennes,
   Saint-Nazaire, Stockholm, Copenhagen, Warsaw, Budapest, Austria, Switzerland, Madrid, Faro, northern Italy,
   Thessaloniki). None is near Paris, Istanbul or Rome. The two feeders whose heard aircraft positions come closest to CDG
   (`portsmouthms`, `DEI-LFKZ`) hear aircraft only in the air, 11–26 km from CDG at the nearest. The positions they hear
   centre 340–430 km away. The only Italian feeder (`CHANGEME-LIDL`) hears nothing closer than 362 km to FCO, and the
   Thessaloniki feeder nothing closer than 395 km to IST. OUT (QP) is sent at the gate, so it is lost. Of the 13 OOOI messages carrying LFPG / LTFM / LIRF route codes, none is an OUT or OFF at
   those airports. They are Ryanair's LIRF→EGBB ON / IN reports and its EGBB→LIRF OUT / OFF, heard in Birmingham, plus
   one LFPG→LFRN IN report heard in Rennes.
2. **Gaps in coverage.** The archive starts on **14 June 2025**, so it has no January 2025. It is empty from 2 March to
   13 May 2026. **January 2026 is nearly empty:** 6.8 MB a day against 87 MB a day in December 2025, and the 15 January
   sample hour holds 37 messages from 4 feeders. July 2025, November 2025, December 2025 and July 2026 are complete
   (every day present; [`adl_monthly_volume.csv`](receipts/sample_scan/adl_monthly_volume.csv)).
3. **Only fragments reach the archive after departure.** Air France sends a label-**2N** report in flight, which UK and
   Channel Islands feeders hear when a westbound flight passes. It carries two HHMM times, for example
   `AF0020/15150807LFPGKLAX  07170753` and `AF098G/15151004LFPGKIAH  09170941`, plausibly OUT and OFF. That is one CDG
   departure per sampled hour on 15 July 2026, so roughly 10–30 a day, or a few hundred a month out of about 20,000 LFPG
   departures. It appears in July 2026 only: none was seen in the 2025 or January 2026 samples. Nothing documents the
   meaning of the two times (airframes' decoder has no 2N plugin). Ryanair's OFF report (QQ) also repeats the OUT time.
   It is sent at take-off, though, and no feeder is near Rome to hear the LIRF departures. Turkish Airlines and ITA
   messages heard over Europe (position, weather and fault reports) carry no OUT time.

**Plausible coverage if it were pursued:** LFPG perhaps 300–900 departures in July 2026 (Air France westbound long-haul
and Channel routes, 2N only, meaning unverified); LTFM 0; LIRF 0; January 2026 about 0. To check 2025 one would have to
download the ACARS (and possibly VDL2) streams for July, November and December 2025 (283 / 526 / 697 MB of ACARS,
830 / 1,668 / 1,818 MB of VDL2 compressed; a bulk download, the coordinator's decision). Then keep label `2N` messages
matching `^AF\w{4}/\d{8}LFPG[A-Z]{4}\s+\d{8}`, join them by date and Air France callsign to the supplied LFPG departures,
and test (a) whether the second time equals the supplied take-off minute and (b) how the first time relates to the
recorded off-block. On the samples, the yield in 2025 is uncertain and could be zero, and the reachable rows are Air
France hub departures, not the dark or early-record population the review identified. **Recommendation: close.**

## Sources checked

| # | Source | URL | What it holds | 2025 coverage | Jan / Jul 2026 coverage | LFPG / LTFM / LIRF | Licence (quoted, where stated) | Access | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | adsb.lol Aircraft Data Links archive | https://github.com/adsblol/aircraft-data-links-2025 , https://github.com/adsblol/aircraft-data-links-2026 ; docs https://www.adsb.lol/docs/open-data/aircraft-data-links/ | Decoded ACARS, VDL2 and HFDL messages (acarsdec / dumpvdl2 / dumphfdl JSON) from volunteer feeders, with feeder id, frequency, label, flight, registration, text and VDL2 air/ground bit. OOOI labels QP/QQ/QR/QS present. | From 14 Jun 2025; Jul, Nov and Dec 2025 complete; **no Jan 2025** | **Jan 2026 nearly empty** (6.8 MB/day vs 87 in Dec); Jul 2026 complete (163 MB/day); gap 2 Mar–13 May 2026 | No feeder in ground range of any of the three. LFPG: Air France 2N in-flight report (two times, meaning undocumented) for ~1 westbound departure per sampled hour, Jul 2026 only. LTFM 0. LIRF 0. | "This data is released under CC0 1.0 Universal (CC0 1.0) Public Domain Dedication." (each release body); "All collected data will be published under the CC0 (Creative Commons Zero) license on GitHub releases" (https://www.adsb.lol/docs/open-data/aircraft-data-links/) | Public download, no account (GitHub releases, ~72,000 zstd JSONL files) | **NOT USABLE.** Licence passes, but no gate-side OUT at the three airports, no Jan 2025, Jan 2026 nearly empty. Near-miss. |
| 2 | Airframes.io (app, API, historical message search) | https://app.airframes.io , https://docs.airframes.io/api/licensing , https://docs.airframes.io/api/pricing | Global community ACARS / VDL2 / HFDL / satcom message search and API | Depth not stated | Not stated | Not checked (licence fails first) | "Bulk redistribution, resale, or commercial use beyond your access tier requires prior arrangement." / "The Airframes API and its licensing terms are still being finalized." / free tier: "Personal and non-commercial use" | API for active feeders only; commercial tiers in development | **NOT USABLE.** No open licence; feeder-only, personal and non-commercial; terms not final |
| 3 | ACARSHub, acarsdec, dumpvdl2, acars_router (receiver software) | e.g. https://www.adsb.lol/docs/open-data/aircraft-data-links/ (lists the decoders) | Software for running one's own receiver; it shows local messages | No public archive found | None | Only where someone runs a receiver | n/a (software) | Self-hosted | **NOT A SOURCE.** No historical archive; a new receiver cannot record 2025 or Jan/Jul 2026 |
| 4 | OpenSky Network historical database (Trino) | https://raw.githubusercontent.com/open-aviation/pyopensky/master/src/pyopensky/schema.py | Mode S / ADS-B tables (state vectors, flights, raw replies, FLARM) | n/a | n/a | n/a | Research-only access (project already refused; [`docs/adsb_coverage/external_sources_20260926.md`](../adsb_coverage/external_sources_20260926.md)) | Account plus approval (denied) | **NOT USABLE.** No ACARS / VDL2 table in the documented schema, and access is denied |
| 5 | Zenodo (searches: ACARS, VDL2, OOOI, HFDL, off-block) | https://zenodo.org/api/records?q=ACARS | Two ATRDS 2025 papers on TOBT / off-block (PDF only) and unrelated records | No message dataset | None | None | Papers CC BY-SA 4.0 (not data) | Public | **NOT USABLE.** No decoded ACARS / OOOI dataset exists there |
| 6 | Kaggle / GitHub dataset search (web search only) | web search, 7 Oct 2026 | Nothing beyond #1 found | – | – | – | – | – | **NOT USABLE.** Nothing found |
| 7 | Commercial OOOI feeds (OAG, FlightAware AeroAPI, Cirium, SITA / ARINC) | https://www.oag.com/hubfs/Inbound-Services/OAG-ACARS-OOOI-Message-Types-and-Examples.pdf , https://www.flightaware.com/commercial/aeroapi/AeroAPI_Standard_License.pdf | Airline OOOI and gate times | Yes (paid) | Yes (paid) | Likely | Commercial licences (not opened) | Paid contract | **NOT USABLE.** Not openly licensed |
| 8 | data.gouv.fr (France) | https://www.data.gouv.fr/api/1/datasets/?q=aeroport | Airport traffic totals (CDG passengers and movements, last updated 2014), noise and geometry layers; "Transport aérien" stub (2019, 0 files) | Aggregates only | None | LFPG totals only | "Trafic aéroport Paris-CDG": licence `fr-lo` (2014); "Transport aérien": `"license": "notspecified"` | Public | **NOT USABLE.** No per-flight off-block |
| 9 | Groupe ADP (Paris airports) | https://www.data.gouv.fr/datasets/transport-aerien-1/informations ; VITRAIL viewer (earlier survey) | Live flight information and a trajectory viewer | No open history | No | – | No open-data licence found | Website viewer | **NOT USABLE.** No open per-flight data |
| 10 | DHMİ (Turkish State Airports Authority) statistics | https://www.dhmi.gov.tr/Lists/Istatislikler/Attachments/415/TÜM%20UÇAK.pdf | Monthly airport traffic statistics (PDF) | Monthly totals | Monthly totals | LTFM totals | No open licence stated | Public PDF | **NOT USABLE.** Aggregates, no licence |
| 11 | Istanbul municipality open data (İBB) | https://data.ibb.gov.tr/api/3/action/package_search?q=havalimani | 6 datasets of monthly flights, passengers and freight for Turkish airports | Last modified 2024-11 | No | LTFM totals | "Istanbul Metropolitan Municipality Open Data License" | Public | **NOT USABLE.** Monthly aggregates only |
| 12 | İGA (Istanbul Airport operator) | web search | Nothing found beyond live flight pages and commercial APIs | – | – | – | – | – | **NOT USABLE.** No open data found |
| 13 | Aeroporti di Roma (FCO operator) | web search | No open-data portal found | – | – | – | – | – | **NOT USABLE.** Nothing found |
| 14 | ENAC open data (Italy) | https://www.enac.gov.it/open-data/ | Certification registers only (examiners, CAMO / CAO / AMO / MTOA organisations) | No flight data | No | – | "dati in formato aperto o Open Data liberamente utilizzabili dall'intera collettività" (no named licence) | Public | **NOT USABLE.** No flight data |
| 15 | dati.gov.it (Italian national catalogue) | https://www.dati.gov.it/opendata/api/3/action/package_search?q=fiumicino%20voli | Regional and municipal traffic totals (Milan, Palermo, MIT 2012); "fiumicino voli" returns 0 | Aggregates | No | No LIRF per-flight data | CC BY 4.0 on the aggregates | Public | **NOT USABLE.** No per-flight data |
| 16 | EUROCONTROL Aviation Data Repository for Research (ADRR) | https://www.eurocontrol.int/dashboard/rnd-data-archive | NM flight records and trajectories | 2015–2025 | **None for 2026** | All ECAC airports (NM-derived, the same family as the supplied AOBT_3 / LOBT) | "The ADRR tool is open for all R&D use. Please consult the Terms of Use that have to be signed off by each individual user." | Account plus approval plus signed terms | **NOT USABLE.** No 2026; not an open licence; not an independent OUT observation |

Licence excerpts and retrieval times are in [`receipts/`](receipts/): `adsblol_aircraft_data_links_licence.txt`,
`adsblol_adl_github_releases.txt`, `airframes_terms.txt` and `portals_and_archives.txt`. The sample-scan evidence
(feeder lists, origin-coded messages, Air France 2N messages, sample file hashes, archive volume by month and by day)
is in [`receipts/sample_scan/`](receipts/sample_scan/). The machine-readable table is [`sources.json`](sources.json).

## What the samples show (adsb.lol archive)

| Window (UTC) | Read | Messages | Feeder ids | OOOI-label messages | With an LFPG / LTFM / LIRF route code | OUT or OFF at LFPG / LTFM / LIRF |
|---|---|---:|---:|---|---:|---:|
| 15 Jul 2026 07:16–08:55 | 3 ACARS and 3 VDL2 files in full (5.35 MB) | 65,417 | 123 | QQ 47, QP 18, QS 15, QR 10 | 13 | 0 |
| 15 Jul 2026 09:55–11:19 | 3 ACARS files in full (1.36 MB) | 17,900 | 71 | QQ 8, QP 6, QS 3, QR 2 | 0 | 0 (one Air France 2N) |
| 15 Jul 2025 ~07–09 | first ≤400 kB of each stream | 7,811 | 25 | QQ 3, QP 2, QR 1, QS 1 | 0 | 0 |
| 15 Nov 2025 ~07–08 | first ≤400 kB of each stream | 17,667 | 61 | QQ 11, QP 1 | 0 | 0 |
| 15 Dec 2025 ~07–08 | first ≤400 kB of each stream | 9,980 | 72 | QQ 7, QS 1 | 0 | 0 |
| 15 Jan 2026 ~05–13 | all 4 files of the window (7 kB) | 37 | 4 | none | 0 | 0 |

OOOI is decodable where a receiver hears the gate. Ryanair's Birmingham departure FR02FU, for example, sends QP
`EGBBLIRF0752 212` (OUT 07:52) and then QQ `EGBBLIRF08080752` (OFF 08:08, OUT 07:52). KLM sends QP `EHAMFACT0809...`.
The archive's receivers are in the wrong places for the three target airports.

## Not done

No account was created, no airframes, FlyItalyADSB or EUROCONTROL request was made, and no bulk download was run.
No supplied competition data was read, so the meaning of the Air France 2N times is not validated against labels.
Nothing in this folder changes a model, a file or an upload.
