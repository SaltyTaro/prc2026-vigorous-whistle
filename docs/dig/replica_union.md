# Dig tunnel 16: union of adsb.lol's daily replicas (3 October 2026, label-free, nothing declared)

No official score, no leaderboard, no upload, nothing under runs/submissions. Scratch `runs/dig/replica/` (git-ignored):
`extract_replica.py` (the shipped `tools/adsb_extract_day.py` decode and boxes, release tag as a parameter), `compare.py`,
`compare_all.py`, `compare_all.csv`, `tags_2026.txt`, `new/new_YYYYMMDD.parquet` (rows newly matched under the union),
`out/surface_YYYYMMDD_staging-0.parquet`. Extraction ran on arcw0 (started twice, deallocated after each run; all eight workers
deallocated at the end).

## Question

`tools/adsb_extract_day.py` reads only the `-prod-0` release. adsb.lol also publishes `-staging-0` (and `-mlatonly-0`, probed
earlier) for most days. Batch 5's check compared match rates of staging-preferred and
prod-preferred days in prod-0 data; it never tested the union of both replicas on the same day.

## What the replicas are

- Ranking days: 62 have prod-0 and mlatonly-0; 52 have staging-0. There is no staging replica for 13 to 22 January 2026, and no
  `-0tmp` tags for any ranking day.
- The two replicas carry the same feed, sampled independently. On 8 January 2026, prod-0 has 1,332,343 in-box points and staging-0
  has 1,331,480, but their de-duplicated union has 1,430,532 (the same tracks at interleaved timestamps). The prod-0 re-extraction
  reproduces the shipped day file byte for byte (13,746,573 bytes).
- Archive sizes agree within 0.05% on most days. They diverge on 23, 26, 30 and 31 January and on 11, 24 and 25 July. On 23 January
  prod-0 itself is short (2.20 GB against neighbouring days' 2.6-3.2 GB), and staging-0 is truncated (one 2.0 GB part; extraction
  ends with "unexpected end of data" after 33,501 traces). Even so, the truncated staging replica fills a prod-0 outage.

## Shipped base matcher, prod-0 against the union

Method: `match_departures` with the learned stands (`data/external/adsblol/matches/stands.parquet`), ranking departures of each of
the 52 days, prod-0 alone against prod-0 + staging-0. The v35 IATA fix and the v59 rematch are not applied, so read the counts as
relative. No row loses its match, and the motion start of an already matched row moves by more than 30 s on 6 rows in total.

| month | airport | departures (52 days) | newly matched | share | new gate | upgraded to gate |
|---|---|---:|---:|---:|---:|---:|
| Jan 2026 | EDDF | 10,641 | 94 | 0.88% | 1 | 0 |
| | EDDM | 7,464 | 38 | 0.51% | 29 | 1 |
| | **EGLL** | 13,012 | 141 | **1.08%** | 24 | 0 |
| | EHAM | 10,474 | 40 | 0.38% | 29 | 1 |
| | LEBL | 8,524 | 60 | 0.70% | 23 | 1 |
| | LEMD | 11,510 | 71 | 0.62% | 29 | 0 |
| | **LFPG** | 12,381 | 184 | **1.49%** | 1 | 0 |
| | LIRF | 7,713 | 74 | 0.96% | 10 | 0 |
| | LSZH | 6,731 | 40 | 0.59% | 11 | 0 |
| Jul 2026 | **EDDF** | 20,650 | 265 | **1.28%** | 4 | 2 |
| | EDDM | 15,013 | 65 | 0.43% | 39 | 10 |
| | EGLL | 20,557 | 131 | 0.64% | 54 | 10 |
| | EHAM | 21,950 | 104 | 0.47% | 69 | 10 |
| | LEBL | 17,879 | 132 | 0.74% | 44 | 12 |
| | **LEMD** | 20,152 | 201 | **1.00%** | 21 | 3 |
| | **LFPG** | 21,743 | 299 | **1.38%** | 1 | 0 |
| | LIRF | 15,838 | 2 | 0.01% | 0 | 0 |
| | LSZH | 13,155 | 80 | 0.61% | 23 | 0 |

LTFM gains nothing (no surface coverage in either replica).

- **Total:** 2,021 newly matched departures, 412 of them gate-traced, plus 50 matched rows upgraded to gate. That is 0.68% of the
  52 days' departures.
- **Concentration:** two prod-0 outage days carry most of the rich new tracks: 23 January (354 new, 130 gate) and 11 July (185 new,
  58 gate, 27 upgraded). On ordinary days the union adds 10-50 rows a day.
- **Marginal tracks:** at EDDF, LFPG, LIRF and LEMD-July the new rows are mostly tracks the union lifts over the matcher's 3-point
  minimum (median 3 points; LFPG's first point a median 73-83 min before take-off). They carry little information. At EDDM, EGLL,
  EHAM, LEBL, LSZH and LEMD-January the new tracks are full (median 25-161 points, 80-100% observed more than 300 s before take-off).
- 2025 check (15 July 2025): 51 new matched of 6,120 (0.8%), 7 gate. The dev months would gain at a similar rate.

## Sizing (label-free, on the uploaded v61)

The 378 newly gate-traced rows at their known stand were sized with v59's rematch form, 0.5 x v61 + 0.5 x (ADS-B taxi + offset):

| offset | squared change S | largest move | mean move by airport (s) |
|---|---:|---:|---|
| 0 s | 9.8e6 | +1,444 s (one LFPG row) | EGLL −166, LEMD −109, LIRF −299 (10 rows), EDDM / EHAM / LEBL / LSZH −42 to −52 |
| +50 s | 8.6e6 | +1,469 s | EGLL −141, LEMD −84, others −17 to −27 |

v61 sits above the ADS-B taxi on these rows, most of all on the outage days. 23 January holds 125 of the 378 rows and 11 July 52;
those rows went through the unmatched route. At the gain-to-S ratios seen so far (0.9-2.55), the layer is worth about
0.8e7-2.2e7 squared error, roughly **−0.05 to −0.15 s**. A full rebuild that also feeds the roughly 900 rich non-gate new tracks to
the networks could add somewhat more. The marginal 3-point tracks are worth little either way.

## Verdict: GO by the directive's 1% bar (five airport-months), modest size

Five airport-months clear 1%: EGLL January, LFPG January, EDDF July, LEMD July and LFPG July. The value sits in the 378 new gate rows
(EDDM, EGLL, EHAM, LEBL, LEMD, LSZH, and the two outage days), not in the LFPG / EDDF marginal tracks.

### What using it would take

1. **Cheap path (recommended first): a v59-style layer.**
   - Union surface for the dev months (2025 January, July, November, December): staging-0 extraction is about 40-90 s per day on
     arcw0, three in parallel, about 30 min for the roughly 124 days.
   - Rows newly gate-traced at their known stand under the union get 0.5 x pred + 0.5 x (ADS-B taxi + o_airport), with o fitted on
     the other months (`tools/run_rematch_nonm.py`'s offset rule).
   - Declare and read it on the dev chain proxy (`runs/dig/census61/dev_v61.parquet`) under v57's rule. Apply it in the model part
     before v48, like the rematch, or last, like the intruder bound; decide in the declaration.
   - Ranking: the 378 rows above (plus the 50 upgraded rows if the declaration includes them).
2. **Full rebuild:**
   - Write union day files for the 52 ranking days, keeping prod-0 where no staging exists, into a new surface directory (do not
     overwrite `data/external/adsblol/surface`).
   - Re-run the matcher (matches2 with stands), the v35 callsign fix, the v59 rematch and the v19 / context features.
   - Re-run inference of every ADS-B-consuming member: residual stage, committee, nn2, the dl32 / m25dl32 / m25_32 context networks,
     the LIRF networks and the stacker.
   - Then run the chain, as in `docs/final_file/v61_chain.md` (its ADS-B step reads `data/external/adsblol/surface`; it needs a
     surface-directory override).
   - Dev months would need the same union and fold re-predictions for a fair reading. This is hours of CPU and a GPU-free inference
     pass, but its marginal value over path 1 is the rich dark tracks only.
