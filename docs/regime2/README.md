# Lever 8: second label-free regime audit of the 2026 ranking file on v68's chain (7 October 2026) — no find

Question: does the 2026 ranking file hold a systematic difference from the 2025 development rows that v68's chain mishandles and that
label-free inputs can fix (the class of dig tunnel 1's no-NM flight-number key, v59)? Ranking inputs read label-free; 2025 labels only to
test a fix; no official score, no leaderboard, nothing declared, built or uploaded; LIRF population and guard rows untouched. Read-only
analyses on existing tables (no fit, no model inference); no VM was needed or started. Scripts: [`scripts/`](scripts/) (run from the
repository root). Control: `runs/opdi_stack_lszh/dev_v68_proxy.parquet` (938a91cd...), ranking `runs/submissions/vigorous-whistle_v68.parquet`.

## What was compared (2026 airport-months against the 2025 monthly range)

| check | result |
|---|---|
| Raw departure columns (all 28 inputs: missingness, callsign / flight / stand / type codings, seconds grids, IFR) [`missing.py`] | inside 2025's range except the known EHAM January no-NM share (5.0% vs <= 2.5%) and EGLL July numeric callsigns (0.53 vs <= 0.49, harmless) |
| Arrival in-block recording (fill share at schedule, whole-minute share) | identical to 2025 at every airport (LIRF arrival fills 0.079 / 0.097 vs 0.079-0.116) |
| Unseen categories (operator, type, destination, stand, runway) | operators 1-3% (mostly no-NM), stands EDDM July 7.9% / EDDF July 1.4% (known), types LIRF <= 1.8% |
| ADS-B v19 + own-points feature tables [`feat_shift.py`] | known coverage shifts only: EGLL much denser in 2026 (July p_n 99 vs <= 22, runway-queue counts ~2x), EDDF sparser (counts -35%), LIRF July dark, stand imputation EDDF July 14% / EDDM July 8% (unseen stands); no missing table, no stale cache (surface files 15/19 Sep, tokens 27 Sep, own points 3 Oct) |
| Stacker support guard (stack_resid_v3 coverage report) | ranking guard shares at or below development at every airport: no fallback to weaker paths |
| Published daily totals | all 620 ranking airport-days present and used by the reconciliation |
| Own-points member delta (v66 vs v64 model part) | 2026 inside the dev range except EHAM January (disruption days, RMS 111 s) and January at EDDF / LFPG / LSZH (about 2x) |

## Candidate finds, sized label-free, screened on 2025

1. **Month-start context.** The ranking block is built alone, so own-points inbound links on 1 January / 1 July 2026 miss the previous
   day's arrivals (no 30 June 2026 surface file; 31 December's arrivals sit in the training block): inbound share 0.60 on day 1 vs 0.74-0.77
   on other days (2025 day-1 drop 0.05-0.08, January 2025 0.15). About 900 rows per month lose own points; at the measured per-row value of
   the own-points layers this is under 0.01 s. Not pursued.
2. **Monthly level against EUROCONTROL's published taxi-out means** [`levels.py`, `levels_dev.py`]: v68's level error after the method bias,
   2026: EGLL July +9.8 s, January +4.5 s (mapping noise 0.7 s), LIRF July +38.2 s (noise 4.1 s), EHAM January +8.3 s; LSZH / EDDM / EDDF /
   LEMD January lie outside the mapping's range of the left-out share (u 0.20 / 0.31 / 0.13 / 0.03 against 2025 maxima 0.10 / 0.20 / 0.05 /
   0.02: an extreme de-icing month) and are not estimable. The declared published-means procedure read on the v68 proxy: January -0.0002%,
   July +0.0127%, November -0.0017%, December +0.0193% (noise limit 8 s; -0.0149% in December at 5 s). EGLL's shifts help in all four months
   (SSE -2.7e6 / -1.3e6 / -2.6e5 / -1.8e5) but the equal-footing mean is about -0.008% (bar -0.02%); LIRF's estimate overshot July 2025 (11.2 s
   against -2.3 s on ordinary rows), so its +38 s is not usable. Ranking stake of an EGLL-only layer about 2.4e6 SSE (about -0.016 s). Not
   declared (below the bar).
3. **Own-points member at LTFM** (no ADS-B there): its dev alignment is negative at LTFM in all four months (gain / S -0.5 to -0.8), but
   reverting it at LTFM reads -0.002 / -0.012 / -0.009 / -0.030% at the model part and +0.004 / +0.003 / +0.015 / -0.024% carried to the chain
   (the stacker absorbs it). Not a lever.

Closed earlier and not repeated: snow / de-icing lift, peer congestion, day-level ADS-B bias, EGLL density offset, published day shifts,
matcher fixes (docs/disagreement/, docs/adsb_coverage/, docs/dig/).

## Verdict

**No v59-class find.** The 2026 file has no coding, key, format, time-zone, day-boundary or missing-table failure that sends rows to a
weaker path; its differences from 2025 are ADS-B coverage (EGLL denser, EDDF sparser, LIRF July dark) and January 2026's weather (u at
LSZH / EDDM / EDDF far beyond 2025), all already screened. The one supported correction (EGLL's monthly level, about -0.016 s) is below the
bar. Nothing declared, no dry run, no machine started.
