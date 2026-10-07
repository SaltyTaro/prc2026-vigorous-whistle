# Data-link OUT pin screen (adsb.lol Aircraft Data Links, CC0), 7 October 2026

A bounded **screen**, not a declared round. No candidate file, nothing under `runs/submissions`, no upload, and no
leaderboard score was read. 2025 labels were used as development data. Question: do the in-flight data-link messages
in the adsb.lol archive pin the **recorded off-block** of LFPG, LTFM or LIRF departures well enough to beat v69, and how
many January and July 2026 ranking rows would they cover? It follows the source check in
[`docs/oooi_source/README.md`](../oooi_source/README.md).

## Verdict: GO for a declared LFPG round (guarded pin or 50/50 blend); the unguarded pin fails December

* **Two message types carry the airline OUT time of LFPG departures, and it is the airport's recorded off-block.**
  The Air France label **2N** report (`AF098G/15151004LFPGKIAH  09170941`) holds OUT in its first time and OFF in its
  second. The easyJet label **8D** delay report (`0028,93,...,0028,1919,LFPG,EGAA`) holds the delay durations. Scheduled
  time plus their sum is OUT; the last clock is an ETA, not OUT. Recorded off-block minus OUT: within 10 s for 58% (2N) and
  70% (8D) of matched rows, within 60 s for 73% / 81%, within 5 min for 95% / 96%. This holds on the early records
  too (264 rows where the recorded off-block precedes AOBT_3 by more than 300 s: 73% within 60 s). 2N's OFF time is the
  supplied take-off within 60 s for 96% of rows.
* **Against v69 on the matched 2025 rows** (month-blocked offset per source, fitted on the other two months), the
  guarded pin improves every month: July −0.041%, November −0.140%, December −0.378% of complete-month RMSE. The 50/50
  blend with v69 gives −0.050% / −0.154% / −0.343%, and every day-block bootstrap interval of the blend's gain is above
  zero. Without a guard, a single 8D row breaks December (+1.32%): airport schedule 05:00, airline schedule about
  08:15. This is the guard's main caveat (see below).
* **Ranking coverage (label-free):** July 2026 has **700 LFPG rows** (2N 329, 8D 371): 436 unmatched and unseen, 199
  dark, 62 unmatched but seen, 3 gate. January 2026 has 83 LFPG rows. LIRF has 5 July rows. One of them is a LIRF
  population row, so the layer should be **LFPG only**. LTFM has none. The guard rows 213142471 and 213142469 are not hit.
* Against the bar (at least −0.02% of complete-month RMSE on two of three months, no month worse, at least ~200 July 2026
  rows), the guarded pin and the guarded blend pass. The unguarded pin and the unguarded blend fail December.

## What was run

1. **Extraction on arcw0** (10:00–10:15 UTC, then deallocated; [`tools/oooi_pin_extract.py`](../../tools/oooi_pin_extract.py),
   [`tools/oooi_pin_pack.py`](../../tools/oooi_pin_pack.py)). The public GitHub API listing returned 72,009 assets, 25,499 of
   which are in the target months (July, November and December 2025, January and July 2026), counted by file start date
   with the previous day included. These are all the ACARS, VDL2 and HFDL files: 11.75 GB compressed and 137.7 M messages.
   Each file was downloaded into memory, hashed and decompressed with `zstd`, then filtered. Every label 2N and QP / QQ
   / QR / QS message was kept, plus every message whose text names LFPG, LTFM or LIRF and has at least two 4-digit groups.
   Kept: 563,872 messages, 257,969 distinct texts. Each file's SHA-256 matches GitHub's `digest`, with 0 mismatches.
   Receipts: [`receipts/listing_receipt.json`](receipts/listing_receipt.json) (listing page hashes) and
   [`receipts/archive_files.csv.gz`](receipts/archive_files.csv.gz) (URL, size, SHA-256 and line count per file).
2. **Parsing and join, local** ([`tools/oooi_pin_join.py`](../../tools/oooi_pin_join.py)). Messages are kept when their
   origin is LFPG, LTFM or LIRF. Each HHMM clock is anchored to the latest UTC instant at or before the message
   time, which handles midnight crossings. The 2N layout is `flight/DD(dep) DD(msg) HHMM(msg) ORIG DEST  OUT OFF`. The
   join key is origin, destination, ICAO prefix (AF→AFR, U2→EZY, EC→EJU) and the callsign suffix without leading zeros
  . The supplied take-off must fall between OUT − 1 h and the message time
   + 5 min (for 8D: message − 8 h to message + 5 min). Only label-free fields are used. Development: 1,554 departures
   (1,536 by suffix, 18 by a unique route and time window; 187 messages with candidates but no suffix match and no
   unique fallback, and 1 ambiguous message, were dropped). 87–97% of the 2N reports and 98–100% of the 8D reports in
   the five months found their departure.
3. **Evaluation, local** ([`tools/oooi_pin_eval.py`](../../tools/oooi_pin_eval.py), [`tools/oooi_pin_report.py`](../../tools/oooi_pin_report.py)).
   It joins on MVT_ID_mvt to the v69 development proxy `runs/stack_weight/dev/w65_bag10/proxy.parquet` (SHA-256 c088e409…; every matched
   row is present). Pin = take-off − OUT + offset(source), with the offset taken as the mean, or the median, of label −
   (take-off − OUT) over the other two months. Blend = 0.5 pin + 0.5 v69. The oracle is an ideal correction capped at
   ±300 s. Month SSE = rows × RMSE² as given (July 190,713 × 250.835826², November 162,332 × 181.883484², December
   165,677 × 163.353697²).

## Matched development rows (2025)

| month | 2N | 8D | other | total | of which early records |
|---|---:|---:|---:|---:|---:|
| July | 209 | 35 | 1 QP (LTFM) | 245 | 60 |
| November | 282 | 286 | 1 QP | 569 | 101 |
| December | 313 | 427 | 0 | 740 | 103 |

The two QP rows are left out of the evaluation (one is misparsed). The 2N and 8D rows include 7 LIRF and 2 LTFM easyJet
8D rows, with small effect. Proxy classes of the evaluated rows: LFPG_u 993, LFPG_m 550, LIRF 7, LTFM 2.

## How the times relate (all 1,552 2N / 8D rows)

| quantity | n | median | 95th pct | ≤10 s | ≤60 s | ≤120 s | ≤300 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| recorded off-block − OUT, 2N | 804 | +4 s | +298 s | 57.7% | 72.9% | 88.7% | 95.3% |
| recorded off-block − OUT, 8D (sched + Σdelays) | 748 | +1 s | +237 s | 69.9% | 80.6% | 89.7% | 96.3% |
| take-off − OFF, 2N | 804 | 0 s | +59 s | 90.1% | 95.8% | 99.8% | 100% |
| AOBT_3 − OUT, 2N (for contrast) | 799 | +120 s | +900 s | 8.3% | 22.2% | 34.9% | 65.2% |
| early records: recorded off-block − OUT | 264 | +3 s | +242 s | 58.0% | 73.5% | 87.5% | 95.5% |

The recorded times sit on whole minutes with ±6 s jitter, so "≤10 s" means equal at minute resolution. The tail is
one-sided: the recorded off-block is later than the airline OUT (1% of 2N rows by more than 776 s). This tail is the
pin's main error. The mean offset (label − (take-off − OUT)) is −55 to −73 s for 2N and −18 to −25 s for 8D (guarded).

## Pin versus v69 on matched rows (mean offset, month-blocked)

Gain = v69 SSE − candidate SSE on the matched rows (s²). The complete-month figure is the change of complete-month RMSE.

| variant | month | rows (pinned) | v69 RMSE | pin RMSE | blend RMSE | pin gain s² | pin: % month SSE / RMSE | blend: % month SSE / RMSE | oracle ±300 s: RMSE |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| unguarded | Jul | 244 (244) | 290.9 | 211.1 | 189.6 | 9.78e6 | 0.082% / −0.041% | 0.099% / −0.050% | −0.053% |
| unguarded | Nov | 568 (568) | 228.5 | 191.6 | 160.7 | 8.83e6 | 0.164% / −0.082% | 0.279% / −0.140% | −0.217% |
| unguarded | Dec | 740 (740) | 260.1 | 475.9 | 267.8 | −1.18e8 | −2.66% / **+1.320%** | −0.068% / **+0.034%** | −0.340% |
| plausible 300–3600 s | Jul | 244 (243) | 290.9 | 275.5 | 243.8 | 2.13e6 | 0.018% / −0.009% | 0.051% / −0.026% | −0.053% |
| plausible 300–3600 s | Nov | 568 (567) | 228.5 | 188.0 | 160.4 | 9.60e6 | 0.179% / −0.089% | 0.280% / −0.140% | −0.217% |
| plausible 300–3600 s | Dec | 740 (732) | 260.1 | 177.5 | 188.3 | 2.68e7 | 0.606% / −0.303% | 0.539% / −0.270% | −0.340% |
| **source guard** | Jul | 244 (244) | 290.9 | 211.1 | 189.6 | 9.78e6 | 0.082% / **−0.041%** | 0.099% / **−0.050%** | −0.053% |
| **source guard** | Nov | 568 (565) | 228.5 | 160.8 | 152.1 | 1.50e7 | 0.279% / **−0.140%** | 0.308% / **−0.154%** | −0.217% |
| **source guard** | Dec | 740 (733) | 260.1 | 150.4 | 163.5 | 3.33e7 | 0.754% / **−0.378%** | 0.685% / **−0.343%** | −0.340% |

Source guard (label-free): pin a 2N row only if |take-off − OFF| ≤ 120 s (OFF confirms the match) and take-off − OUT
≥ 300 s. Pin an 8D row only if take-off − OUT is between 300 and 3,600 s and, where the NM clock exists, |AOBT_3 − OUT|
≤ 1,800 s. Otherwise keep v69. Day-block bootstrap (2,000 draws), 95% interval of the gain, source guard: pin July
[−6.3, +28.7]e6 (includes zero), November [+3.9, +24.5]e6, December [+17.2, +52.9]e6. Blend: July [+2.7, +24.5]e6,
November [+11.4, +22.2]e6, December [+17.3, +46.6]e6. Days on which the pin is better: 27/31, 25/30, 27/31. Median
offsets give the same picture (`results.json`, `median|source`). Gains concentrate on the early records: v69 RMSE
481 / 360 / 587 s against pin 138 / 137 / 295 s. They also concentrate on unmatched LFPG rows: December LFPG_u 262 s
against 106 s.

The pin can beat the ±300 s oracle in December because it also corrects errors larger than 300 s.

**Caveat:** the source guard and the 300–3,600 s window were written after inspecting the December outliers. Those are
one 8D row with a schedule mismatch of about 3.4 h, two more 8D rows 39–60 min off (the airport schedule differs
from the airline's), and long 2N taxi-outs on 5 December (labels 4,300–5,900 s) that the pin got right. A declared round should fix the guard and the weight
(pin or blend) before reading. The archive also holds **August, September and October 2025** (labels exist; not
read here) as a fresh check of the guard's false-pin rate and of the time relation.

## Ranking coverage, January and July 2026 (label-free)

| month | airport | rows | 2N | 8D | other | rows after source guard | unm_unseen | dark | unm_seen | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | LFPG | 84 | 33 | 50 | 1 QP | 83 | 50 | 23 | 9 | 1 |
| 2026-07 | LFPG | 704 | 329 | 372 | 2 QP, 1 text 5Z | 700 | 436 | 199 | 62 | 3 |
| 2026-07 | LIRF | 5 | 0 | 5 | 0 | 5 (1 is a LIRF population row: exclude) | 5 | – | – | – |

These are out of 18,129 / 21,743 LFPG ranking departures in January / July. Label-free drift check, 2026 against 2025:
2N's OFF matches take-off within 60 s on 97% of rows (96% in 2025). The take-off − OUT distribution is unchanged. AOBT_3 −
OUT has moved closer, from a 2N median of 120–180 s in 2025 to 20 s in July 2026 (240 s in January 2026); this does not enter the pin. If 2026
rows behave like the 2025 rows (about 3.8e4 s² gained per pinned row), the 783 LFPG rows are worth about 3e7 s². In
July 2026 about 2.6e7 s² is roughly 0.2% of a July-2025-sized month SSE, or about −0.1% RMSE. This is an estimate, not
a reading.

## Files

* `docs/oooi_pin/results.json`: every number above (time relations by source and month, all six variants per month
  with bootstrap intervals, by-class and by-source breakdowns, join statistics, ranking coverage by class, drift check,
  archive receipt summary, input hashes).
* `docs/oooi_pin/receipts/`: GitHub listing hashes and the per-file URL / SHA-256 receipt (25,499 files).
* `data/external/adsblol_links/target_origin_messages.csv.gz`: the parsed LFPG / LTFM / LIRF-origin messages (CC0
  source).
* `runs/oooi_pin/` (git-ignored): `messages_all.parquet` (all 563,872 kept copies), `messages_unique.parquet`,
  `parsed.parquet`, `joined_dev.parquet`, `joined_rank.parquet`, `eval.json`, logs. The unfiltered archive was not stored.
  The kept-message JSONL (468 MB) remains on arcw0 under `runs/oooi_pin/`.
* Scripts: `tools/oooi_pin_extract.py`, `tools/oooi_pin_pack.py` (arcw0), `tools/oooi_pin_join.py`,
  `tools/oooi_pin_eval.py`, `tools/oooi_pin_report.py` (local).

arcw0 was started shortly before 10:00 UTC and deallocated at 10:15 UTC; its final state is `VM deallocated`. No other VM
and no AWS resource was used, and `.env` was not read.

## Declared round oooi_pin_v1 (governing run 10:30 UTC, 7 October 2026)

Declaration: [`configs/oooi_pin_v1.json`](../../configs/oooi_pin_v1.json) (0010550 + 3f093f0, declared 10:21:01 UTC,
SHA-256 f40bf9a4…), run exactly as written by [`tools/run_oooi_pin.py`](../../tools/run_oooi_pin.py). Numbers are in
[`results_v1.json`](results_v1.json). Scope: LFPG departures matched to a 2N or 8D report that pass the screen's source
guard, verbatim. Candidate value = max(w × pin + (1 − w) × chain, −12) on those rows; every other row keeps the chain.
No fixed row (LIRF population, guard) is in scope. Intervals: `tools/run_intruder_bound.interval`, 2,000 replicates over
UTC take-off days, seed 20261004, on complete-month RMSE change. The proxy's `day` equals the take-off day on 100% of
rows.

**Clean months (clause 4).** August, September and October 2025 were downloaded and filtered on arcw0 (10:23–10:28
UTC). That is 17,995 files, 5.71 GB and 62.4 M messages, with 0 digest mismatches. The filtered tables were copied back
with matching SHA-256 (`messages_unique.parquet` bb297728…, `messages_all.parquet` c6137ae3…), and arcw0 was
**deallocated at 10:30 UTC (state `VM deallocated`)**. The first run's files on arcw0 also hash-match the local copies.
With 35 GB free, nothing needed removing. Receipts: [`receipts/fresh_listing_receipt.json`](receipts/fresh_listing_receipt.json)
and [`receipts/fresh_archive_files.csv.gz`](receipts/fresh_archive_files.csv.gz).

Code changes since 8de5d74 are parameters only, recorded in [`receipts/param_diff_v1.patch`](receipts/param_diff_v1.patch).
`oooi_pin_extract.py list` takes `--months` (8a3f27f2…). `oooi_pin_join.py` takes `--run-dir` and `--months`
(849e33b3…), and its defaults reproduce the screen's `joined_dev`, `joined_rank`, `parsed` and `join_stats` exactly.

In-scope rows: development July 242, November 560, December 731. Fresh August 373, September 461, October 493.
Ranking January 83, July 700.

| candidate | Jan | Jul [95%] | Nov [95%] | Dec [95%] | eq mean | c1 | c2 | c3 | Aug [95%] | Sep [95%] | Oct [95%] | c4 | c5 ratio | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w100 | 0 | −0.0405 [−0.128, **+0.026**] | −0.1359 [−0.238, −0.038] | −0.3823 [−0.583, −0.206] | −0.1409 | ✓ | ✓ | **✗** | −0.0088 [−0.031, +0.009] | −0.0203 [−0.039, −0.011] | −0.0525 [−0.090, −0.011] | ✓ (2/3) | 0.737 ✓ | no |
| w75 | 0 | −0.0522 [−0.129, **+0.001**] | −0.1645 [−0.253, −0.093] | −0.4025 [−0.613, −0.233] | −0.1516 | ✓ | ✓ | **✗** | −0.0199 [−0.040, −0.009] | −0.0238 [−0.044, −0.014] | −0.0738 [−0.102, −0.050] | ✓ (3/3) | 0.738 ✓ | no |
| **w50** | 0 | −0.0494 [−0.108, −0.011] | −0.1514 [−0.221, −0.100] | −0.3454 [−0.529, −0.205] | −0.1316 | ✓ | ✓ | ✓ | −0.0222 [−0.041, −0.012] | −0.0216 [−0.040, −0.013] | −0.0722 [−0.093, −0.057] | ✓ (3/3) | 0.737 ✓ | **yes** |

All figures are percent changes of complete-month RMSE. Development control: the v69 proxy (July 250.8358, November
181.8835, December 163.3537). Fresh control: rows12 `stack` (August 396.814, September 428.445, October 237.961). Offsets for
the fresh months and the ranking come from July, November and December: 2N −62.26 s, 8D −22.84 s. Pinned rows with
|pin − label| > 1,800 s: August 1 of 373 (0.27%), September 0 of 461, October 0 of 493. Clause 5 compares the RMS of the
ranking change on in-scope rows (w50 102.4 s, after rounding) with the RMS of the out-of-month development change (138.9 s).

**Selection: w50** is the only candidate that passes clauses 1–5. w100 and w75 fail clause 3 because July's interval
includes zero.

**Dry run** (not a submission; nothing under `runs/submissions`, no upload). The uploaded v69 (192c4a30…) with w50 on
the 783 in-scope LFPG rows, `np.rint` into int32, written to `runs/oooi_pin/rank/dry_v70/vigorous-whistle_v70.parquet`.
- SHA-256 **f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f**.
- 775 rows change: January 83 (unm_unseen 50, dark 23, unm_seen 9, gate 1) and July 692 (unm_unseen 432, dark 196,
  unm_seen 61, gate 3).
- Squared change S = 8.21e6 s² against v69. Largest move +927 s, smallest −609 s, mean +2.2 s.
- No LIRF, LIRF population or guard row (213142471, 213142469) changes, and every other row is byte-identical to v69.
- The receipt sits next to the file.
