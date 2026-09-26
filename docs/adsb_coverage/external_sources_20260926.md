# External data survey, 26 September 2026

The user asked whether every external data source had been searched, because the OpenSky historical-database request will most
likely not be granted. Two desk-research passes (public web pages, licences and dataset listings only; about 130 lookups; no
accounts, credentials, contacts or bulk downloads) covered flight-position archives and airport / ANSP / EUROCONTROL open data,
followed by one probe. The prize rule is that external data must be "openly available under an open source license" and documented
(eligibility page).

## Result: no open source improves on adsb.lol where the error is

The weak airports are LTFM (no surface coverage), LFPG (~20% of departures matched), EGLL (partial) and LIRF in July 2026 (surface
receiver dark). No openly licensed source gives surface positions or per-flight off-block times there.

- **Position archives.** airplanes.live and adsb.fi are non-commercial only (and our 18 September probe found them weaker at LFPG and
  LTFM); ADS-B Exchange is proprietary (JETNET terms); OpenSky's scientific datasets are research-only and cover one day of 2026;
  TheAirTraffic already feeds adsb.lol; adsb.aero and AeroWhere are derived from adsb.lol; Flightradar24 / FlightAware / RadarBox
  are commercial; the PRC 2025 trajectories (CC BY 4.0) stop in October 2025; Heathrow WebTrak / ADP VITRAIL are viewers without
  surface data or an open licence. LTFM has no open source anywhere.
- **adsb.lol `mlatonly` releases (ODbL) — probed and closed.** `tools/probe_adsblol_mlatonly.py` streamed the mlatonly releases of
  2026-01-15 and 2026-07-15 through the extractor's decoder: **zero ground points at every airport on both days** (up to 238 airborne
  aircraft per box). They hold multilateration positions of aircraft in the air only ([`mlatonly_probe.json`](mlatonly_probe.json)).
  The same prod files show LIRF on 2026-07-15 with 321 aircraft in the box and no ground point (a local surface receiver went dark
  while airborne coverage continued) and LTFM with 444 aircraft and one ground aircraft.
- **FlyItalyADSB (CC BY-SA 4.0) — conditional lead for LIRF July 2026 only.** Its own archive is self-serve for about the last
  20 days; older data is "available for university or institutional research upon request". Using it would need a contact (the
  user's decision) and a republished extract to count as openly accessible. Given that LIRF's airborne coverage continued in July 2026,
  the gap is most likely a local ground receiver, which the Italian network would not have either unless it had its own feeder at
  Fiumicino. Not pursued.
- **Airport / ANSP / EUROCONTROL data.** EUROCONTROL's performance portal (ansperformance.eu) publishes daily per-airport datasets to
  August 2026 — arrival ATFM delay by cause (incl. weather and de-icing), pre-departure delays, taxi-out additional time — under
  terms that are non-commercial and need approval for modifications (not an open licence). Taxi-out statistics, planning taxi times
  and all-cause pre-departure delay are aggregates of the target's own timestamps and are excluded outright, as is Schiphol's Flight
  API (airport-reported off-block times). Airport sites (Fraport, Munich, Heathrow, Zurich, Aena, DHMI) offer PDFs, live viewers or
  monthly totals without open licences. Runway in use per hour is already in the supplied data (`RUNWAY_mvt`).
- **Open weather beyond METAR.** ERA5 (CC BY 4.0 since July 2025; all ten airports, hourly, snowfall and snow depth) and national
  services — DWD (EDDF/EDDM, CC BY 4.0), KNMI (EHAM, CC BY 4.0), Météo-France (LFPG, Licence Ouverte), MeteoSwiss (LSZH, free with
  attribution) — add snow depth and precipitation amounts. The only clean new signal found; expected value modest (snow and freezing
  METAR codes are already features, and 2025 has no analogue of the EHAM early-January 2026 disruption).

## Decisions

- adsb.lol stays the only position source; `mlatonly` closed by the probe.
- EUROCONTROL daily delay data: not used without an organiser answer on its licence (the user decides whether to ask).
- FlyItalyADSB: not pursued (needs a contact; the LIRF gap looks like a local receiver).
- Open weather (ERA5 / national services): the one candidate for a later screen, if CPU rounds leave room.
