# Data-link OUT formats at all ten airports (adsb.lol Aircraft Data Links, CC0), screen of 7 October 2026

This is a **screen**: no declaration, no candidate file, nothing under `runs/submissions`, no upload, and no leaderboard
use. It follows the declared round `oooi_pin_v1` ([`docs/oooi_pin/README.md`](../oooi_pin/README.md), shipped as v70).
The question: besides Air France 2N and easyJet 8D at LFPG, which data-link formats in the archive carry an OUT time at
EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH or LTFM? How well do they match the recorded off-block, and would
a guarded w50 pin pay?

## Verdict: NO GO for every format; the EGLL family is a near miss

* Nine formats carry an OUT-like time at the ten airports. Four meet the match-quality bar after their offset (centred
  ±30 s ≥ ~50%, ±2 min ≥ ~85%, ≥ ~100 scored 2025 matches): **BA_MVT16, UA_OUT13, MVA_AD (Aer Lingus) and EZY_8D**.
  QP_VS just misses it (49% / 87%). Apart from EZY_8D they are **all at Heathrow**. Their messages appear only from
  October 2025, after a receiver near LHR came online. July 2025 therefore has no rows, and the bar ("two of July /
  November / December") becomes "both November and December".
* Guarded w50 pin, change of complete-month RMSE, November / December 2025 (July has no rows):
  * BA_MVT16 −0.0520% / −0.0144%
  * UA_OUT13 −0.0167% / −0.0011%
  * MVA_AD −0.0150% / −0.0009%
  * QP_VS −0.0132% / −0.0022%
  * **EGLL_family**, one pin per departure from these four: **−0.0969% / −0.0186%**. It misses December by 0.0014
    points; every day-bootstrap interval of its gain is above zero.
  * EZY_8D, mostly EHAM, is **worse** in all three months: +0.0028% / +0.0201% / +0.0398%.

  At EGLL and EHAM, v69 is already good on these rows (RMSE 65–165 s, mostly gate-traced), so a pin gains little.

  January 2026 has almost nothing (the archive is nearly empty that month). Nothing usable was found at LTFM, LIRF,
  LEMD, EDDF, EDDM or LSZH. The hub carriers' messages there (KL, LH, LX, IB, TK, AZ) are position, ETA, engine or
  weather reports with no OUT time.
* Against the bar (−0.02% or better on two of July / November / December, no month worse, ≥ ~200 ranking rows), no
  format passes. The EGLL family and BA_MVT16 have the coverage, but only November clears −0.02%.

## What was run

1. **Extraction on arcw0** ([`tools/oooi_ext_extract.py`](../../tools/oooi_ext_extract.py),
   [`tools/oooi_ext_pack.py`](../../tools/oooi_ext_pack.py)). It covered July–December 2025, January 2026 and July 2026:
   all 43,118 ACARS / VDL2 / HFDL files, 17.33 GB, 198.6 M messages, with 0 GitHub digest mismatches. Each file was
   streamed, hashed and filtered in memory. Kept: every QP / QQ / QR / QS / 2N / 8D message; every text naming one of
   the ten ICAO codes, or as a separate word their IATA codes, within 16 characters of a run of four or more digits;
   and every MVT-style AD / AA / EA group. That gave 5.53 M messages, 2.31 M distinct within a month. Everything was
   copied back with matching SHA-256 ([`results.json`](results.json) → `archive.copy_back_sha256`). Receipts:
   [`receipts/listing_receipt.json`](receipts/listing_receipt.json) and
   [`receipts/archive_files.csv.gz`](receipts/archive_files.csv.gz). The VM's `data/raw` files are hash-identical to
   the local ones.
2. **Census and label-free join** on the VM ([`tools/oooi_ext_scan.py`](../../tools/oooi_ext_scan.py)).
   * Census by (label, two-character airline, text skeleton, route kind) × month × airport:
     [`census_top40_departure_routes.csv`](census_top40_departure_routes.csv) and
     [`census_departure_routes_by_airport.csv`](census_departure_routes_by_airport.csv).
   * Join keys: callsign suffix with leading zeros stripped; UTC day ±1; origin and destination when the text names
     them; take-off within [message − 14 h, message + 3 h].
   * The airline prefix (e.g. BA → BAW, EI → EIN, U2 → EZY; 144 airlines) is learned from unique route-confirmed
     matches, and only unique candidates are kept. Result: 501,433 matched messages, 491 ambiguous dropped.
3. **Token scan** on July / November / December 2025 labels only. Every HHMM-like group of each matched text was
   anchored to the instant nearest the take-off and compared with the recorded off-block and the take-off. Results
   per (label, airline, token): `runs/oooi_ext/scan/token_stats_*.csv`.
4. **Explicit parsers** ([`tools/oooi_ext_formats.py`](../../tools/oooi_ext_formats.py)) for the formats the scan
   flagged.
   * The label-free guard is uniform and was fixed before any pin was scored: take-off − OUT within 180–3,600 s, and
     |take-off − OFF| ≤ 180 s where an OFF time exists.
   * Two choices were made after reading the token scan (label-based, July / November / December), before any pin
     was scored: QP_VS (Virgin's QP, the best airline within QP_ORIG) and the EGLL_family membership and order.
     Neither guard was written after looking at pin outliers.
   * August–October 2025 block times and labels are blanked in `formats.parquet`; those months are only counted.
5. **Screen** ([`tools/oooi_ext_screen.py`](../../tools/oooi_ext_screen.py)) against the v69 development proxy
   (`runs/stack_weight/dev/w65_bag10/proxy.parquet`, c088e409…). Value = max(0.5 × pin + 0.5 × v69, −12). The offset
   is the mean of label − (take-off − OUT) per format, fitted on the other scored months. Rows in oooi_pin_v1's scope,
   LIRF population rows and the two guard rows are excluded. The results are assembled by
   [`tools/oooi_ext_report.py`](../../tools/oooi_ext_report.py).

## Formats found

Counts are matched departures. Scored months (July, November, December 2025) carry labels; August–October are counted
only. Quality is block − OUT on the scored months; "centred" means after subtracting the format's median.

| format | label / airline | airport(s) | 2025 Jul / Aug / Sep / Oct / Nov / Dec | 2026 Jan / Jul | median | raw ±30 s | centred ±30 s / ±2 min / >5 min | OFF vs take-off ±60 s |
|---|---|---|---|---|---:|---:|---|---:|
| BA_MVT16 `MVT01085606EGLLLIPE…` | 16 BA | EGLL | 0/0/0/0/455/162 | 0/620 | −120 s | 0% | **55% / 90%** / 1.5% | – |
| UA_OUT13 `/13 OUT EVENT / EGLL KIAH 13 125324` | 13 UA | EGLL | 0/0/0/33/104/28 | 0/161 | −81 s | 3% | **61% / 89%** / 2.3% | – |
| MVA_AD `MVA EIN151/… AD0645/0659 EA0758 EIDW` | 80 EI (+1 EB at LEMD) | EGLL | 0/4/0/31/170/38 | 0/128 | −55 s | 41% | **51% / 95%** / 1.9% | 96% |
| EZY_8D (sched + Σdelays; non-LFPG) | 8D U2/EC/DS | EHAM 887, LSZH 111, EDDM 58, LIRF 31, LEBL 24, EDDF 22, LEMD 13 | 99/121/149/220/172/187 | 4/196 | +45 s | 26% | **64% / 89%** / 7.2% | – |
| QP_VS `EGLLKJFK0829  53  1161/0828` | QP VS | EGLL | 0/0/0/43/134/36 | 0/117 | −55 s | 31% | 49% / 87% / 7.1% | – |
| SK_OUTRP `/OUTRP/EGLL/ESSA/12/0858/` | 8A SK | EGLL | 0/0/0/23/92/18 | 0/83 | −62 s | 6% | 53% / 76% / 11.8% | – |
| KL_BIS (OUT OFF ON IN line) | 10 KL | EHAM | 15/14/13/11/19/37 | 0/30 | +67 s | 4% | 75% / 89% / 5.6% (n 71) | 93% |
| QP_ORIG (all airlines' QP) | QP many | EGLL 1038, EHAM 70, … | 5/2/7/176/371/108 | 0/455 | −55 s | 22% | 40% / 75% / 14% | – |
| QQ_FR (OFF + OUT in QQ) | QQ many | EGLL 333, … | 1/2/5/27/148/36 | 0/129 | −57 s | 3% | 10% / 16% / 83% | 95% |
| AF_2N outside LFPG | 2N AF | EHAM | 1/0/1/0/0/0 | 0/0 | – | – | n 1 | – |

Smaller OUT-bearing formats seen in the token scan (fewer than 100 matches) are Delta 4I / 4L / 4M (engine-run and
movement times, EGLL), CX 12, QR / TK QP, Oman Air 24 (`TKO01` 2N-like) and United 12 (doors and brake). BA's label 11
`OUT01…` carries no time that matches the recorded off-block (best ±2 min share 56%). At their own hubs, the KLM
(EHAM), Lufthansa (EDDF / EDDM), Swiss (LSZH), Iberia (LEMD), Vueling (LEBL), THY (LTFM), ITA (LIRF), Ryanair and Wizz
messages heard are position, ETA, engine or weather reports. The best ±2 min share of any token with at least 30 matched departures is at most 47% (LH 47%, IB 47%, VY 38%, LX 33%, AZ 12%, FR 11%, W6 18%). These airlines' only
matching OUT times come from small samples away from their hubs: TK QP at EGLL (n 33, 85%), VY QQ at EGLL (n 53, 81%)
and KL_BIS (n 71).

## Screen: guarded w50 pin against the v69 proxy

| format | Nov rows | Nov v69 / pin / w50 RMSE | Nov change | Dec rows | Dec v69 / pin / w50 RMSE | Dec change | Jul | ranking rows (July 2026 by class) | verdict |
|---|---:|---|---:|---:|---|---:|---|---|---|
| BA_MVT16 | 455 | 148 / 78 / 97 | −0.0520% | 160 | 140 / 106 / 108 | −0.0144% | no rows | 611 (gate 310, dark 272, unm_unseen 29) | NO GO (one month) |
| UA_OUT13 | 103 | 164 / 77 / 98 | −0.0167% | 28 | 122 / 118 / 106 | −0.0011% | no rows | 159 (gate 99, dark 58, 2 unm) | NO GO |
| MVA_AD | 170 | 124 / 90 / 77 | −0.0150% | 38 | 65 / 49 / 46 | −0.0009% | no rows | 127 (dark 104, gate 19, unm_unseen 4) | NO GO |
| QP_VS | 133 | 153 / 138 / 113 | −0.0132% | 36 | 97 / 84 / 63 | −0.0022% | no rows | 116 (dark 81, gate 34, unm_seen 1) | NO GO |
| **EGLL_family** | 861 | 146 / 92 / 96 | **−0.0969%** | 262 | 124 / 99 / 96 | **−0.0186%** | no rows | **1,013** (dark 515, gate 462, unm_unseen 34, unm_seen 2) | NO GO (near miss) |
| EZY_8D | 172 | 98 / 235 / 149 | +0.0201% | 185 | 129 / 299 / 189 | +0.0398% | +0.0028% (97 rows) | 198 (EHAM 160, LSZH 18, …) | NO GO (worse) |
| SK_OUTRP | 89 | 118 / 171 / 108 | −0.0018% | 18 | 78 / 234 / 136 | +0.0025% | no rows | 83 | NO GO |
| KL_BIS | 19 | 23 / 82 / 43 | +0.0002% | 37 | 171 / 182 / 173 | +0.0003% | −0.0008% (15) | 30 | NO GO |
| QP_ORIG | 365 | 143 / 276 / 168 | +0.0260% | 104 | 94 / 237 / 139 | +0.0123% | +0.0008% (5) | 447 | NO GO (worse) |
| QQ_FR | 30 | 193 / 665 / 371 | +0.0281% | 8 | 90 / 751 / 389 | +0.0129% | 0 (1) | 26 | NO GO (worse) |

* **Offsets:** BA_MVT16 +139 to +156 s and UA_OUT13 +98 to +126 s. In both, the airline's OUT trails the recorded
  off-block.
* **Day-bootstrap 95% intervals of the gain:** above zero for BA_MVT16, UA_OUT13, MVA_AD, QP_VS and EGLL_family in
  both months. EGLL_family December: [+2.0e5, +3.1e6] s².
* **Oracle (±300 s cap) on the same rows:** EGLL_family −0.158% / −0.042%, BA_MVT16 −0.085% / −0.032%.
* **EGLL_family by proxy class:** in November, EGLL_u rows gain about 24 thousand s² per row (198 → 123 s) against
  about 10 thousand on EGLL_m rows (138 → 92 s). The 549 dark or unseen July 2026 rows might therefore gain more than
  the development rows suggest. That is an untested estimate.

**August–October 2025, counted and not scored** (guarded, after exclusions, the clean months for any later
declaration, control rows12 `stack`):
- EGLL_family: 4 / 0 / 106
- EZY_8D: 121 / 147 / 214
- KL_BIS: 14 / 13 / 11
- MVA_AD: 4 / 0 / 31
- UA_OUT13: 0 / 0 / 33
- QP_VS: 0 / 0 / 42
- SK_OUTRP: 0 / 0 / 23
- BA_MVT16: 0 / 0 / 0

At Heathrow only October would be a clean month.

## Files

* [`results.json`](results.json) holds the archive summary and copy-back hashes, the census (by month, airport,
  label and airline), join statistics, per-format quality and the full screen per format and month: rows, offsets,
  RMSEs, gains, bootstrap intervals, oracle, the >1,800 s count, ranking coverage by month / airport / class and the
  August–October counts.
* `runs/oooi_ext/` (git-ignored) holds `messages_unique.parquet` (2.31 M messages, ccd18627…),
  `scan/matched.parquet`, `scan/formats.parquet`, `scan/tokens_dev.parquet`, the token tables, `screen.json` and
  `v70_pinned_ids.parquet`.
* Scripts: `tools/oooi_ext_extract.py`, `tools/oooi_ext_pack.py`, `tools/oooi_ext_scan.py` and
  `tools/oooi_ext_formats.py` (run on arcw0); `tools/oooi_ext_screen.py` and `tools/oooi_ext_report.py` (local).

**VMs:** only arcw0 was started (about 11:30 UTC). It was deallocated at 11:54 UTC after the hash-checked copy back.
arcw0, arcw1, arcw2 and arcw3 all read `VM deallocated`. No AWS, no `.env`, and no edits to
`configs/oooi_pin_v1.json` or the shipped tools.
