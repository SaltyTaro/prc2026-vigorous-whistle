# ADS-B v21_v2: the LIRF-inclusive LightGBM blend on matched LIRF rows, exported as v19 (18 September 2026)

`docs/adsb_v21/README.md` records `adsb_v21_v1`: three LightGBMs fitted on every airport with only
the LIRF fill population excluded, blended on matched LIRF non-population rows with ordinary clocks,
plus a five-seed bag of the v12 residual stage. Its selection
([`../adsb_v21/development_results.json`](../adsb_v21/development_results.json)) scored every LIRF
candidate at −0.19% to −0.30% mean against v18 (July −0.43 … −0.74%, November −0.13 … −0.17%,
every interval below zero on both months) and the seed bag at −0.002% (null), and selected nothing
only because January 2025 has no row in the LIRF blend scope (LIRF had no ADS-B that month): a
candidate that leaves a month untouched cannot have that month's interval "below zero".

## Declaration `configs/adsb_v21_v2.json` (post-hoc, recorded as such; commit 003a5d2)

Identical learners, ensembles, candidates, baseline (v18's development predictions) and bar
(not worse on any month, mean ratio ≤ 0.9998); the residual seed bag dropped (null; the residual is
v12's single booster again, so the reproduction candidate is exact); the paired-day interval
condition applied to the months where the candidate differs from v18 on at least one row
(`interval_condition_touched_months_only`, `rows_differing_from_baseline` recorded per month).
Declared with the v21_v1 scores in hand (its declaration and selection are hash-bound parents).

## Selection ([`development_results.json`](development_results.json))

Reproduction check `ens_v20_bag_q_all_70`: ratio 1.0 on every month. Selected
**`lirf_ens3_70`** = the v18 recipe everywhere except on LIRF non-population rows with a matched
surface segment and |NM proxy| and |schedule interval| under six hours, where the prediction is
0.3 · v9 + 0.7 · mean(lgb_lirf_s31, lgb_lirf_cap_s32, lgb_lirf_nm_s33):

| month | v18 | candidate | change | paired-day interval (s) | days better | scope rows |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Jan 2025 | 330.302 | 330.302 | 0 | untouched (LIRF unmatched all month) | – | 0 |
| Jul 2025 | 271.528 | 269.512 | **−0.742%** | [−3.53, −0.99] | 27 / 31 | 13,992 |
| Nov 2025 | 192.187 | 191.878 | **−0.161%** | [−0.57, −0.09] | 18 / 30 | 7,401 |

Mean ratio 0.99699 (−0.30%). LIRF alone: 652.0 → 641.4 s (−1.62%) in July, 451.7 → 449.8 s
(−0.42%) in November. Runner-up `lirf_nm_50` (the NM-residual learner alone at 0.5) −0.275%;
the unrestricted-scope record `lirf_ens_50_all` +1.87% (the one-day row). No other airport
changes.

## Export, verification, audit

`tools/export_adsb_v11.py --version 19` → **`runs/submissions/vigorous-whistle_v19.parquet`,
SHA-256 `27a3a8fcb0d2608372c7a5cb2b1c07fb6a26ab891e370737e276b4ef6d1411b3`**, 344,841 rows,
min −12, max 120,549, residual refit rounds 1,173 ([`export.json`](export.json)). LIRF ranking
rows in the blend scope: **8,781 of 26,899** (8,669 in January 2026, 34 in July 2026 — LIRF is 80%
matched in January and 0.2% in July). `tools/verify_adsb_v11.py --version 19` passed: receipts, v9
reproduced from scratch, ADS-B features reproduced, independent recomputation equal to the export,
rows outside the scopes equal to v9, saved residual difference 0.0, 300 of 300 segment
re-derivations agree ([`verification.json`](verification.json)). `tools/audit_export.py --version
19` ([`export_audit_v19.json`](export_audit_v19.json)): identical to v18 outside LIRF; on LIRF RMS
difference 37.2 s (changed rows: median −5 s, 1–99% −158 … +181 s), three rows moved by more
than 600 s (fill-hedge rows where both models sit between the NM clock and the schedule
interval), none by more than 1,800 s; extremes, negative-proxy rows, the LIRF population and the
disruption days are unchanged from v18.

## Upload

Pending the user's approval. (Recorded when done.)
