# Dig tunnel 6: the published-total gaps on v59 (3 October 2026, label-free on the ranking; nothing declared or built)

Question: v59's rematch came from EHAM's 2-6 January 2026 days, whose implied off-block delay disagreed with EUROCONTROL's published
daily totals. Do other 2026 airport-days disagree beyond 2025's range, and does a label-free row-level mechanism explain them?
Scratch `runs/dig/gaps/` (`gaps.py`, `diag.py`, `eham.py`, `found_bound.py`, `count_aware.py`). No official score, no leaderboard.

Discrepancy per airport-day = published total - c0 (airport median of published minus the labels' own total, 2025 days with dn = 0) -
implied total (sum of max(a - prediction, 0) over in-figure rows, IFR only at EHAM/LSZH/LEBL), by the predicted off-block day, per row.
2025 reference: the four development months on the v55m model part (`stack`) and the shipped chain (`new`), |dn| <= 2.

## Outlier days (after the chain; 2025 chain spread per row: MAD 0.8-5.5 s)

| day | per row (s) | reading |
|---|---:|---|
| EDDF 10 Jan | -47 | heavy delay day (median a 2,704 s vs 1,455), composition normal; I = +50 s, below the extreme-day threshold |
| EHAM 2/3/5/6/29 Jan | -35 / +38 / -29 / -112 / -28 | the disruption: 36-74 no-NM rows a day still without a usable track (below) |
| EDDM 18 Jul | -30 | departure stop on record (hour-effect screen) |
| LEBL 1/2 Jul, 20 Jul; LIRF 1/2 Jul | +72 / -70, +69; +60 / -55 | day-boundary pairs: one long-delayed flight's delay on the other side of midnight |
| LTFM 19 Jan (snow), LTFM 31 Jan | +140, +130 | 19 Jan: dn = +5 so the extreme-day layer skipped it; 31 Jan: month end (dn = 74) |
| LSZH 10 Jan, LFPG 7 Jan, LFPG 14/15 Jul, LEBL 14/16 Jul | +78, +69, +48/+44, +38/+34 | over-predicted taxi on delay-heavy days; per-day stake 1-3e6 |
| LIRF July (23/24/26 negative, 1/6/11/15 positive) | -69..+64 | the dark month with heavy population rows (guarded) |

Uniform per-day corrections of all these days together are worth at most ~3e7 squared error (~0.1-0.2 s), most of it already the
extreme-day layer's mechanism.

## Candidates

1. **Found-but-unqualified rematch tracks** (no-NM rows whose learned callsign found an ADS-B track that is not a gate segment at the
   known stand: EHAM 2-6 Jan 145 rows; 2026 total ~900 rows at EDDF/EHAM/LSZH): the track's motion start bounds the label from below.
   2025 validity is 94-100% only for implied taxis up to 2,400 s (EHAM dark segments 80-85%) and collapses above (0-67%: segments of
   earlier movements). Clamping 2026 violators up to 2,400 s: 141 rows, S 6.1e6; up to 7,200 s: S 1.4e8 but only 8 of 73 airport-days
   move toward the published totals and the summed gap widens (EHAM 3/4/5 Jan overshoot). **Closed.**
2. **Count-aware extreme-day eligibility** (the layer skips days whose count difference dn has the "wrong" sign; with the noise model's
   count term, ~1,700-1,900 s per flight, the implied shift I' stays >= 80 s): EHAM 6 Jan (I' +113), LTFM 15/19/28 Jan (-83 / -199 /
   -82), plus LEBL 1/2/20 Jul, which are day-boundary pairs and must not shift. Without the pairs: 4 days, S ~1.7e7 at the airports'
   betas (~0.05-0.1 s if right). 2025 analogs (dn != 0, |I'| >= 80, dev months): LTFM 26 Jul and 7 Dec agree in sign and size (need
   +44 / -128 vs I' +104 / -97), EDDM 1 Jan does not (need -26 vs -217); the rest are month-end count artifacts. **Small, thin evidence.**
3. **Day-boundary pairs** (LEBL 1/2 Jul, 20/21 Jul; LIRF 1/2 Jul): one row's predicted off-block day differs from its true one; the two
   days' discrepancies are artifacts that also enter v57's re-allocation (I_d up to +-80 s on ~580 rows) and would mislead any
   count-aware extension. A label-free pair repair (reassign the single row whose implied delay matches the pair) is clean but worth
   ~1e5-1e6 per pair. Prerequisite for 2, not a lever alone.
4. **The rest of EHAM's disruption gap**: 82 no-NM rows on 2-6 Jan have a learned callsign but no ADS-B track at all, 7 no learned
   callsign. No data to recover them. Closed.

No candidate reaches 0.1 s with evidence. Next test, if any: 2 + 3 together as a declared extension of the extreme-day layer, read on
the twelve-month table (all 2025 days with dn != 0 and |I'| >= 80, pairs excluded) before any ranking file.
