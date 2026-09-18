# Export audit: vigorous-whistle_v16 (official 241.75), 18 September 2026

Read-only audit of the incumbent upload (`runs/submissions/vigorous-whistle_v16.parquet`,
SHA-256 `b6a7ef5e…1ec23`) with `tools/audit_export.py --version 16`; full figures in
[`v16_audit.json`](v16_audit.json). The reference is the incumbent recipe's own
development predictions on Jan/Jul/Nov 2025 (reconstructed from the recorded v9,
residual and learner tables; reproduces 330.469 / 271.782 / 192.417). No ranking label
exists and none was used; December untouched. Purpose: find anything silent in the file
we are defending with — malformed rows, extrapolation, garbage inputs passed through.

## Verdict: nothing to fix

No malformed rows, no negative or unfloored predictions, extremes are the designed LIRF
fill hedge, disruption days are carried by ADS-B evidence, and the ADS-B-informed rows
behave as in the fitting months. One item is an extrapolation to note, not an error
(EDDF's July coverage regime, below).

## 1. File

344,841 rows, `MVT_ID_mvt` unique and equal to the set of ranking departures of Jan +
Jul 2026 (0 missing, 0 extra), `TAXITIME_SEC_mvt` int32, no NaN, min −12 (the floor;
23 rows), max 120,549. Differences to earlier uploads: RMS 15.9 s vs v15 (no row differs by
more than 600 s), 64.4 s vs v11, 109.9 s vs v9 (LIRF identical to v9 by design).

## 2. Extremes and the LIRF population

| threshold | ranking rows | ranking share | dev prediction share | dev label share |
| --- | ---: | ---: | ---: | ---: |
| > 1 h | 942 | 0.27% | 0.13% | 0.24% |
| > 4 h | 37 | 0.011% | 0.007% | 0.010% |
| > 12 h | 12 | 0.003% | 0.003% | 0.003% |
| > 24 h | 2 | 0.001% | 0 | 0.002% |

The 40 largest predictions are all LIRF (39 population rows, 1 clock-present row whose
AOBT_3 sits 86,340 s before take-off); every one is v9's mixture value (prediction ≈
0.99 × schedule interval for the 43 population rows delayed > 4 h, against 0.91 on the
2025 development months — the classifier leans further to the fill in 2026, the
direction the v8 lesson supports). The two predictions above 24 h are fills for rows
whose schedule interval itself exceeds 24 h (111,654 s and 87,484 s); there is no one-day
hedge in the file. The doubled share of > 1 h predictions relative to the development
predictions comes from the January 2026 disruption days and LIRF's July population
(276 rows against 52–60 in Jan/Nov 2025).

## 3. Garbage inputs

169 rows with a negative NM proxy (AOBT_3 after take-off): predictions median 1,099 s,
max 3,228 s, none negative. 11 LFPG "gate" rows whose ADS-B segment implies a taxi of
more than 5,000 s (stale arrival segments): predictions median 969 s, max 1,408 s — the
learners ignore the stale segment. Rows without an NM record: EHAM January 5.0% (the
disruption; 1.2–2.3% in 2025), everything else 0.4–3.0%, within the 2025 range.

## 4. Airport-days out of line with their month

Flagged by a robust 3-sigma rule on daily mean predictions:

- **EHAM 3–7 Jan 2026** (means 1,770–2,364 s against 809 normal; no-NM share up to 40%).
  On these days the matched rows' predictions track the ADS-B-implied taxi (e.g. 4 Jan:
  2,467 s predicted, 2,274 s implied, the usual EHAM offset) and the unmatched rows are
  lifted to 2,000–2,240 s by the neighbour features, against v9's 1,340–1,780. Consistent.
- **LFPG 1–7 and 29 Jan 2026** (up to 2,080 s), **EDDF 6/10/26/29 Jan**, **EDDM 8/10/25/30 Jan**,
  **LSZH 8/10 Jan**, **LTFM 12/18/19 Jan** (1,500–2,100 s), **EGLL 4 Jan, 12 Jul**, LIRF 5 Jan
  and 2/20/24 Jul, LEMD 5 Jan: winter-weather and single disrupted days; the January
  cluster is why five airport-month means sit 17–25% above their 2025 counterparts
  (EDDF, EDDM, EHAM, LSZH, LTFM January). Nothing in 2025 can score these days; they are
  carried by the NM, weather, ADS-B and neighbour inputs the models were fitted on.
- **EHAM 4/6/28 Jul 2026** low days (≈630 s against 899): quiet-day or runway-use pattern,
  not an anomaly.

## 5. ADS-B-informed rows: coverage and behaviour, ranking versus fitting months

adsb.lol matched / gate-covered share per airport-month:

| airport | 2026-01 | 2026-07 | 2025-01 | 2025-07 | 2025-11 |
| --- | --- | --- | --- | --- | --- |
| EDDF | 0.79 / 0.08 | 0.76 / 0.04 | 0.90 / 0.12 | 0.92 / 0.17 | 0.96 / 0.40 |
| EDDM | 0.97 / 0.68 | 0.97 / 0.61 | 0.95 / 0.64 | 0.38 / 0.22 | 0.98 / 0.65 |
| EGLL | 0.88 / 0.25 | 0.97 / 0.47 | 0.70 / 0.17 | 0.23 / 0.03 | 0.69 / 0.19 |
| EHAM | 0.93 / 0.59 | 0.96 / 0.64 | 0.94 / 0.42 | 0.95 / 0.64 | 0.96 / 0.63 |
| LEBL | 0.95 / 0.30 | 0.96 / 0.35 | 0.95 / 0.21 | 0.93 / 0.30 | 0.97 / 0.34 |
| LEMD | 0.96 / 0.24 | 0.74 / 0.18 | 0.00 / 0.00 | 0.37 / 0.11 | 0.96 / 0.19 |
| LSZH | 0.94 / 0.36 | 0.94 / 0.18 | 0.93 / 0.32 | 0.91 / 0.29 | 0.94 / 0.28 |
| LIRF | 0.80 / 0.13 | 0.00 / 0.00 | 0.00 / 0.00 | 0.94 / 0.07 | 0.66 / 0.08 |

Prediction minus the ADS-B-implied taxi on gate-covered rows, median (p10…p90), seconds:

| airport | 2026-01 | 2026-07 | 2025-01 | 2025-07 | 2025-11 |
| --- | --- | --- | --- | --- | --- |
| EDDF | 116 (33…478) | **288 (35…582)** | 94 (34…439) | 94 (37…437) | 70 (28…337) |
| EDDM | 56 (27…255) | 56 (34…253) | 58 (29…222) | 57 (34…258) | 56 (29…245) |
| EGLL | 345 (151…622) | 242 (137…540) | 415 (156…681) | 443 (126…1314) | 301 (152…639) |
| EHAM | 17 (−25…271) | 8 (−28…239) | 19 (−24…276) | 14 (−25…250) | 12 (−25…244) |
| LEBL | 53 (27…266) | 70 (32…274) | 63 (32…348) | 58 (26…291) | 60 (31…246) |
| LEMD | 156 (83…446) | 160 (80…453) | – | 186 (90…479) | 158 (80…442) |
| LSZH | 54 (20…318) | 70 (30…346) | 49 (20…156) | 46 (19…234) | 58 (24…319) |

Every airport-month sits inside its 2025 band except EDDF July 2026's gate rows (784
rows, 4% of EDDF's July departures), where the median offset is 288 s. EDDF's ranking
months resemble Jan/Jul 2025 rather than Nov 2025 (gate share 8% / 4% against 12% / 17%
/ 40%, median points per segment 15 / 10 against 25 / 32 / 51, complete-match share
0.78 / 0.80 against 0.89 / 0.94 / 0.97): thinner coverage, most pronounced in July. The
dark rows — 95% of EDDF's matched rows in the ranking — carry a median offset of 438–442 s
against 402 / 402 / 254 in the fitting months, i.e. in line with the seasonal months. So
the learners are operating at the thin end of the coverage regimes they were fitted on,
not outside them; no action follows, but EDDF July is the one place where the ADS-B
contribution rests on fewer points per segment than usual.

EGLL's July coverage is the densest EGLL has ever had (gate 47%, matched 97%) and its
gate offset (242 s) sits below the 2025 values, consistent with segments that start
nearer the actual pushback; the block-before-motion convention at EGLL (recorded on 18
Sep in the session notes) is being applied by the learners as fitted.

## Tool

`tools/audit_export.py` (read-only; report JSON only). Rerun for any later export with
`--version N`; earlier versions to diff against via `--compare`.
