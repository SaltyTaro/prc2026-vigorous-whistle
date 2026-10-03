# Tunnel 1: label-free audit of the 2026 ranking file (3 October 2026)

Question: does the ranking file hold a 2026-only regime that the shipped v57 chain mishandles (as EHAM January 2026 no-clock rows did
before v35, and LFPG July EOBT_1 before v31)? Ranking inputs are read label-free; 2025 labels only for testing a fix. Scratch: `runs/dig/regime/`.

## FIND: no-NM departures at EDDF / EHAM / LSZH can be matched to ADS-B with a learned flight-number -> callsign key

**Mechanism.** Rows without a Network Manager record have no `CALLSIGN_flt`; the ADS-B matcher keys them on `FLIGHT_mvt`. At EDDF,
EHAM and LSZH that field is the flight number, not the ATC callsign (it differs from `CALLSIGN_flt` on 97% / 98% / 73% of the NM rows
there; at the other seven airports on 0-2%). So these airports' no-NM rows almost never match: ranking match rate 1-3% at EDDF / EHAM,
26-28% at LSZH, against 76-99% at EDDM / EGLL / LEBL / LEMD. v35's callsign fix mapped the IATA prefix only,
which misses alphanumeric callsigns and leading zeros.

**Key.** For each (airport, `FLIGHT_mvt`), the most frequent `CALLSIGN_flt` over every NM-present departure of 2025 and the ranking
file (label-free). Then the unchanged `prc2026_adsb.features.segment` on the surface track with that callsign around take-off
(`runs/dig/regime/rematch.py`, a copy of `tools/adsb_rematch_nonm.py` with the learned key and a faster callsign filter; it reproduces
its own unfiltered output exactly). Aircraft types agree with the ADS-B track on 98.6% of the rematched ranking rows.

**Why 2026 makes it matter.** The disruption days are no-NM days: EHAM 2-6 January (20-40% of departures without NM), LSZH 8, 10 and
22 January (Swiss flights without NM in blocks of hours, 9-15% of the day). On those days the rematched ADS-B taxi sits well above v57
(EHAM 2-6 January median +240 to +590 s, 5-9 taxi stops, 1-1.5 h stationary: de-icing queues).

**Independent corroboration (label-free).** On 9 of 11 airport-days with at least five new rows, the change the layer makes to the day's
implied positive delay has the sign of the remaining published-minus-implied discrepancy after v57 (correlation 0.66); e.g. EHAM 4 January
-3.49 h of a -3.83 h gap, 5 January -2.62 of -2.83 h, 3 January +1.64 of +5.03 h.

**Candidate layer.** Rows with no NM record at EDDF / EHAM / LSZH whose learned-key track is a gate segment starting at the row's known
stand (`at_known_stand`; gate segments off the stand are de-icing-pad stops, median 208 s short) and that no earlier key matched:
prediction := 0.5 x prediction + 0.5 x (take-off - ADS-B motion start + o_airport), o from other months (EDDF +98, EHAM +1, LSZH +44 s).

| development month (shipped chain `new`) | rows | squared change S | SSE change | RMSE change | paired-day 95% (s) | gain / S |
|---|---:|---:|---:|---:|---|---:|
| January 2025 | 73 | 5.95e6 | -1.60e7 | -0.050% | [-0.445, -0.021] | 2.69 |
| July 2025 | 241 | 8.34e6 | -5.14e6 | -0.021% | [-0.101, -0.018] | 0.62 |
| November 2025 | 39 | 0.66e6 | -1.25e6 | -0.011% | [-0.039, -0.007] | 1.89 |
| December 2025 | 77 | 1.58e6 | -1.93e6 | -0.021% | [-0.074, +0.004] | 1.22 |

Equal-footing mean -0.031%; all four months better, three intervals below zero. All twelve airport-month cells improve (gain / S 0.46-2.96);
moves above 100 s point the right way 91-95% of the time. At weight 0.75 July falls to gain / S 0.08 (one row of the v35 kind): 0.5 is
the robust weight. EHAM 5 January 2025 (a winter disruption day, 16 rows) carries half of January's gain.

**Twelve months** (`runs/dig/regime/eval12.py`; older v23-recipe predictor of `runs/daily_totals/rows12.parquet`, offsets from the other
eleven months, same filter and weight): **12 of 12 months better, all 12 paired-day intervals below zero**, mean -0.024%; gain / S per
month 2.85 / 2.95 / 2.39 / 1.70 / 2.93 / 2.49 / 0.74 / 2.61 / 2.34 / 2.58 / 2.13 / 1.61 (January -0.077%, October -0.050%, the rest
-0.003 to -0.032%). 8,331 no-NM rows at the three airports in 2025, 1,176 new gate rows.

**Ranking (dry arithmetic on v57, read only).** 343 rows: EHAM 246 January + 50 July, LSZH 28 + 11, EDDF 5 + 3. Squared change
S = 1.17e8 (EHAM January 93%; top 30 rows 50%). At the 2025 gain / S range 0.62-2.95 that is **-0.47 to -2.3 s** (about -1.8 s at
the twelve-month median 2.4); it is neutral at
gain / S = 0 and costs +0.77 s if the moves carried no information (gain / S = -1; the lowest 2025 cell is 0.46).
Candidate rows: `runs/dig/regime/rematch_candidate_rank.parquet` (MVT, airport, flight, learned callsign, v57, ADS-B taxi, 0.5 blend).

**Next test.** (1) Declare the layer (key, filter, weight 0.5, other-month offsets) and apply it in the model part before the published
chain, i.e. before v48's reconciliation, the extreme-day layer and v57's re-allocation (as v35's callsign fix sits), then rerun the chain:
the day totals then see the corrected rows instead of re-spreading their discrepancy. (2) Extend the same key to EHAM's NM-present rows?
No: those already match through `CALLSIGN_flt`. (3) Rows the learned key reaches with a dark (non-gate) segment are left alone (closed
"lower bound" screen).

## Other findings (no lever)

- **Clock conventions 2025 -> 2026** (per airport-month; AOBT/LOBT/IOBT/EOBT_1 minus schedule and take-off, seconds grids, equality with
  schedule): nothing new beyond the known LFPG July EOBT_1 and LTFM IOBT changes. AOBT_3 with non-zero seconds rises at LEBL (3.6%) and
  LFPG (2.8%) in January 2026; in 2025 such rows behave like the others (label minus proxy within 70 s of the zero-second rows). LTFM's
  non-zero-second AOBT rows (take-off minus AOBT about 4,400 s) are garbage clocks the shipped models already ignore (v57 ~900 s, 2025 labels
  ~900 s).
- **Runways and stands**: 2026 runway shares stay inside 2025's monthly range (EGLL easterly 09R 38-46%, LFPG July back to 27L after the
  2025 works); EGLL 09L departures 0.6% in July 2026 (new, ~120 rows). Unseen stands: EDDM July 7.9% and EDDF July 1.5% (known), elsewhere
  <= 0.3%.
- **Published discrepancy left after v57** (published minus implied delay per late flight): 2026 airport-month spreads match 2025's
  except the disruption days. Largest: EHAM 6 January -145 s (z -13 against 2025), LTFM 19 January +173 s (5 flights fewer in our rows
  than in the figure; outside the extreme-day layer), EDDF 10 January -49 s, LSZH 8 / 10 January. Row-level value of a further day shift:
  at most ~0.3 s at slope 1 over all |z| > 3 days, ~0.1 s at the 2025 slope 0.6; the general rule is closed (scout section 3).
- **Day-boundary pairs**: LEBL 1/2 July (counts 580/581 and 589/588, discrepancy +41.6k / -41.2k s), LEBL 20/21 July and LIRF 1/2 July are
  one long-delayed flight each whose predicted off-block falls on the other side of midnight. The row's own error is ~350 s; the cost is the
  wrongly spread day shift on two days (at most ~0.02 s each). Count-day mechanism (closed after v56); not reopened.
- **Other anomalies**: 222 late rows have a predicted off-block within 10 minutes of midnight (negligible); 18 rows at or below 0 s (NM
  clocks after take-off) and 62 helicopter rows below 120 s (fine); the v57 day assignment agrees with `bday` on all but 6 rows.
