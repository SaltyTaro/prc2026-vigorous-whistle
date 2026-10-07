# val2026: do the chain's components combine at LFPG in 2026 as 2025 says? (screen, 7 October 2026)

A **screen**: nothing declared, no candidate file, nothing under `runs/submissions`, no upload, no leaderboard score read. 2025 labels
are used as development data only. Local only (peak about 2 GB, 2 threads); no VM, no AWS.

Question: the July 2026 LFPG departures that carry an Air France 2N or easyJet 8D data-link OUT time
([`docs/oooi_pin/`](../oooi_pin/README.md), CC0 source) are the first open-observation validation set inside a scored month. On those
rows, with the pin `take-off - OUT + offset` as the target, do v69's components rank and combine the way 2025 says (label and pin as
target)? Is any blend weight at LFPG unseen or dark rows mis-set for 2026?

## Verdict: NO GO

No blend weight reads outside 2025's range for 2026 with an interval that excludes the 2025 readings. Four readings come close. Each
depends on one to five rows, misses the family-wise interval, or sits on the one direction where the pin is biased against the label in
2025:

| reading (July 2026, pin) | 2026 dw* [95%] | 2025 all-row (label) / pinned (pin) | why not a GO | S on all July 2026 rows of the class (full move / excess over 2025) |
|---|---|---|---|---|
| dark, network share | −1.44 [−1.87, −0.30] | −0.31 / −0.29 (2025 months −0.62..+0.50) | one row holds 51% of the numerator; without the five largest rows −0.08; the interval holds 2025's November −0.62; the family-wise difference interval [−2.03, +0.48] | 1.43e7 / 8.9e6 (6.3e5 at the interval end) |
| all classes, published-data steps (C − v55m) | +1.61 [+0.19, +3.12] | +0.16 / −0.14 (months −0.17..+0.55; July 2025 pinned pin +1.51) | the only 2025 direction where the pin disagrees with the label (label − pin −0.45 [−0.86, −0.04]); family-wise [−0.64, +3.99]; without the top five rows +0.44 | 1.22e7 / 9.9e6 (1.6e5) |
| unseen, outer weight (mix vs v9) | −0.22 [−0.57, +0.11] | +0.08 / +0.22 (months −0.01..+0.15) | the interval holds the 2025 all-row value; without the top five rows +0.16 (sign flips) | 8.9e6 / 1.7e7 (0) |
| EOBT_1 repair weight (2026 only, shipped 0.5) | −0.19 [−0.50, +0.18] (all) | no 2025 counterpart | the interval includes the shipped 0.5 | 3.3e6 |

`dw*` = optimal change of the shipped weight along the chain's blend direction (shipped + dw* = implied weight); intervals over UTC
take-off days.

The pin is a faithful proxy for comparing components. On the 1,532 guarded 2025 pinned rows, its error against the label is 152 s RMS
(median 35 s, one-sided tail), but it ranks the components like the label (Spearman 0.98 unseen, 0.94 dark, 1.00 all). Pair weights
and chain-level weights agree between label and pin within their intervals in every direction except the published-data steps.

## Components compared (paths and SHA-256 in `docs/val2026/components_receipt.json`)

| component | 2025 development rows (LFPG, Jan / Jul / Nov / Dec) | 2026 ranking rows (LFPG, Jan / Jul) |
|---|---|---|
| `v9` (non-ADS-B path), `residual`, `committee` (tree committee), `nets_old` (v30's 0.25 e24 + 0.25 e24b + 0.5 e32 table) | `runs/lomo_dev/rows.parquet` (9072bc73...) | `runs/final_dryrun/run/recipe/detail.parquet` (2e79c78b...); EOBT_1-repaired copies `runs/final_lfpg_eobt/run/recipe/detail.parquet` (a7345583...) |
| `own` (own_all context network, 16 seeds) / `msk` (masked m25_32, 16 seeds; unmatched rows only) | member tables of `configs/ctx_ownpoints_dev_v1.json` / `configs/adsb_ctx3_mask_dev_v1.json`, per month | the same declarations' `ranking` tables |
| `table` = own (w 1.00) then msk at 0.40; `mix` = v46's share 0.85 with the 1,200 s gate; `outer` = 0.3 v9 + 0.7 mix (unmatched), 0.4 (v9 + residual) + 0.6 mix (matched) | rebuilt here (`src/prc2026_final/share_gate.py`, `network_member.py`) | rebuilt here |
| `v55m` (model part) | `runs/pubrefit/chain/ship/dmp.parquet` (5a5c3373...) | `runs/final_dryrun/run/stack/detail.parquet` v55m + `steps/model_part/member_delta.parquet` d52; rounds to the shipped model part `steps/model_part/vigorous-whistle_v66.parquet` (73eef8da...) except 213 rows by ±1 s |
| `C` (after the no-NM rematch, v48 reconciliation, extreme-day, variance re-allocation) | `ship/cand.parquet` 'new' (bdf959c9...) | `runs/ctx_ownpoints_weight/rank/dry_v66/steps/Y/vigorous-whistle_v66.parquet` (12a50ea0...) |
| stacker (`p3`, ten-seed v3 bag; v69 = C + 0.65 p3) | `runs/stack_weight/dev/preds_<m>.parquet` | `runs/stack_weight/rank/preds.parquet` (8083ffcf...) |
| `v69` | `runs/stack_weight/dev/w65_bag10/proxy.parquet` (c088e409...) | `runs/submissions/vigorous-whistle_v69.parquet` (192c4a30...); v70 (f1d17747...) for the half-weight on pinned rows |
| class | `runs/hunt/census_dev.parquet` (e53ab521...) | `runs/hunt/census_rank.parquet` (a5e6e807...) |

Every component exists on both sides, except the EOBT_1 repair, which is 2026 only (LFPG July: v31 = v30 + 0.5 × (v30_repaired −
v30), non-zero on 21,575 rows). The recipe-level
reconstruction equals the dev v52m exactly on 53% of LFPG rows. The rest differ by the u300 / callsign / queue / stretch layers (90th
percentile 12 s).

Pins: `tools/run_oooi_pin.scope` (the declared source guard, verbatim) on `runs/oooi_pin/joined_{dev,rank}.parquet` (SHA-256 as in
`docs/oooi_pin/results_v1.json`). Offsets are month-blocked on 2025; on 2026 they are the declared 2N −62.26 s and 8D −22.84 s. Pinned
rows (not hard-moved):

| period | unm_unseen | dark | unm_seen | gate |
|---|---:|---:|---:|---:|
| 2025 Jul / Nov / Dec | 861 | 541 | 127 | 3 |
| 2026-07 | 436 | 199 | 62 | 3 |
| 2026-01 | 50 | 23 | 9 | 1 |

## A. Component RMSE (s)

| set | rows | v9 | committee | nets_old | own | msk | table | mix | outer | v55m | C | v69 |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| 2025 all Jul/Nov/Dec unseen, label | 41,569 | 251.5 | 242.8 | 240.0 | 241.8 | 238.7 | 239.4 | 234.5 | 235.0 | 232.8 | 231.6 | 230.8 |
| 2025 pinned unseen, label | 861 | 282.2 | 276.1 | 264.5 | 269.9 | 263.8 | 266.2 | 265.6 | 267.5 | 267.2 | 265.9 | 264.1 |
| 2025 pinned unseen, pin | 861 | 313.9 | 309.5 | 299.3 | 305.3 | 296.8 | 300.8 | 300.3 | 301.6 | 300.5 | 298.7 | 297.0 |
| **2026-07 pinned unseen, pin** | 436 | 226.8 | 229.8 | 228.5 | 229.4 | 226.1 | 227.4 | 226.1 | 223.3 | 222.7 | 220.7 | **220.1** |
| 2025 all Jul/Nov/Dec dark, label | 12,947 | 245.0 | 229.4 | 238.9 | 241.5 | – | 241.5 | 232.8 | 221.0 | 218.3 | 217.1 | 217.3 |
| 2025 pinned dark, label | 541 | 237.0 | 221.8 | 219.8 | 219.8 | – | 219.8 | 216.9 | 211.3 | 209.4 | 210.5 | 209.4 |
| 2025 pinned dark, pin | 541 | 268.1 | 252.8 | 253.1 | 252.9 | – | 252.9 | 250.2 | 245.7 | 244.4 | 244.8 | 244.0 |
| **2026-07 pinned dark, pin** | 199 | 159.7 | **139.3** | 168.1 | 171.1 | – | 171.1 | 163.7 | 161.8 | 166.1 | 160.2 | 159.6 |

* **Unseen, July 2026.** The chain steps still improve one after another (outer 223.3, v55m 222.7, C 220.7, v69 220.1), as in 2025,
  and v69 is the best single column. The masked member beats own_all, and the committee is the worst ADS-B column, as in 2025. Unlike
  2025, v9 sits level with the ADS-B members instead of 13–17 s behind. Three rows drive this: without the three largest |v69 − pin|
  rows (1,853 / 1,572 / 884 s), v9 is 195.4 against mix 182.9 and v69 182.6, which is 2025's ordering.
* **Dark, July 2026.** The committee alone (139.3) looks 20 s better than v69. Without it, v69 is 134.5, the committee 134.6 and the table 141.3. Without the second-largest row
  as well, they are all level (133–136).
* The EOBT_1-repaired components are worse against the pin than the unrepaired ones (unseen v30_r 228.3 against v30 224.2; v9_r 233.7
  against 226.8). This is the repair-weight reading in section C; it is not significant.
* **January 2026** (83 rows): two rows dominate every column. own_all reads best (215.8), committee 384.5 and masked member 342.5. Without
  the top two rows these become 148.7 / 211.2 / 194.3, and v69 189.5. The intervals span several weight units, so no reading.

## B. Component-level pair weights w* (0 = first, 1 = second; shipped in brackets)

| set | outer v9 → mix (0.7 / 0.6) | share committee → table (0.85) | mask own → msk (0.40) | own nets_old → own (1.00) |
|---|---|---|---|---|
| 2025 all unseen, label | +0.89 [+0.80, +0.97] | +0.70 [+0.61, +0.79] | +0.83 [+0.58, +1.08] | +0.25 [−0.02, +0.55] |
| 2025 pinned unseen, label / pin | +1.05 / +1.01 | +0.81 / +0.81 | +1.07 / +1.41 | −0.15 / −0.33 |
| **2026-07 pinned unseen, pin** | +0.53 [+0.16, +0.88] | +0.59 [+0.14, +0.95] | +1.03 [−0.40, +2.19] | +0.32 [−0.71, +1.34] |
| 2025 all dark, label | +0.39 [+0.24, +0.60] | +0.41 [+0.25, +0.69] | – | +0.29 [−0.01, +0.72] |
| 2025 pinned dark, label / pin | +0.52 / +0.54 | +0.54 / +0.50 | – | +0.50 / +0.52 |
| **2026-07 pinned dark, pin** | +0.61 [+0.03, +1.14] | −0.13 [−0.33, +0.56] | – | +0.08 [−0.85, +1.74] |

These are local blends of two columns. They ignore the rest of the chain, which is why the component-level outer weight (2025 ~0.9)
differs from the shipped 0.7: the later layers absorb part of it. Repair v30 → v30_r, July 2026: unseen +0.29 [−0.14, +0.80], dark +0.16
[−0.44, +0.75] (shipped 0.5).

## C. Chain-level weight change dw* on v69 (`docs/val2026/results.json`, `C_chain_weights`, `E_july2026_against_2025`)

The direction is the recipe-level derivative of v69 with respect to each weight:

* outer: mix − base
* share: w_o (table − committee), off the gate
* mask: w_o s (msk − own)
* own: w_o s (1 − 0.4 [msk]) (own − nets_old)
* stacker: p3
* published steps: C − v55m
* repair: v30_r − v30
* bias: 1

The estimate is dw* = Σ d (t − v69) / Σ d². July 2026 against 2025:

| class / direction | 2026-07 pin | 2025 all rows, label (pooled; month range) | 2025 pinned, pin | 2026 − 2025 all-row [95%] | family-wise (21 tests) |
|---|---|---|---|---|---|
| unseen outer | −0.22 [−0.57, +0.11] | +0.08 (−0.01..+0.15) | +0.22 | −0.30 [−0.65, +0.03] | [−0.82, +0.20] |
| unseen share | −0.09 [−0.61, +0.36] | −0.07 (−0.15..+0.07) | +0.19 | −0.02 [−0.55, +0.45] | – |
| unseen mask | +1.03 [−1.41, +3.01] | +0.15 (−0.21..+0.42) | +1.57 | +0.87 [−1.52, +2.92] | – |
| unseen own | −0.52 [−3.17, +2.16] | −0.03 (−0.55..+0.74) | −2.21 | −0.49 [−3.23, +2.21] | – |
| unseen stacker | +0.21 [−1.45, +2.06] | +0.15 (−0.11..+0.44) | +0.84 | +0.05 [−1.62, +1.95] | – |
| unseen published | +0.96 [−0.17, +1.96] | +0.17 (−0.06..+0.62) | +0.87 | +0.78 [−0.38, +1.80] | – |
| dark outer | −0.10 [−0.56, +0.31] | −0.18 (−0.28..+0.22) | −0.11 | +0.08 [−0.42, +0.51] | – |
| dark share | −1.44 [−1.87, −0.30] | −0.31 (−0.62..+0.50) | −0.29 | −1.13 [−1.72, +0.01] | [−2.03, +0.48] |
| dark own | −0.96 [−2.56, +1.44] | −0.43 (−0.77..+0.75) | −0.20 | −0.52 [−2.18, +1.89] | – |
| dark published | +3.03 [−1.27, +5.74] | +0.30 (−0.25..+0.73) | −0.87 | +2.73 [−1.68, +5.55] | – |
| all published | +1.61 [+0.19, +3.12] | +0.16 (−0.17..+0.55) | −0.14 | +1.45 [+0.01, +2.99] | [−0.64, +3.99] |
| all repair (2026 only) | −0.19 [−0.50, +0.18] | – | – | – | – |

Pin against label on the same 2025 pinned rows (label − pin, all classes): outer +0.01 [−0.05, +0.08], share +0.03 [−0.07, +0.15],
mask −0.24 [−0.57, +0.12], own +0.08 [−0.23, +0.40], stacker −0.03 [−0.29, +0.20], **published −0.45 [−0.86, −0.04]**, bias +1.0 [−7.5,
+8.2] s. The pin overstates the published-data direction by about 0.45, so the July 2026 +1.61 corresponds to about +1.2 on the label
scale. July 2025's pinned rows also read +1.51 on the pin and +0.71 on the label.

No 2026 interval excludes every 2025 reading (`interval_2026_excludes_every_2025_reading` false for all 21). Two difference intervals
miss zero at 95%: all-class published-data +1.45 [+0.01, +2.99], and unseen outer against the pinned 2025 pin reading −0.44 [−0.86,
−0.02]. That is about what 21 tests give by chance, neither survives the family-wise interval, and dropping the five largest rows moves
each back to 2025's range.

## D. Implied change on all July 2026 LFPG rows of the class (against v70; v70's pinned rows count at half)

Largest: dark share 1.43e7 s² on 4,527 rows (RMS 56 s), published-data steps on all 21,737 rows 1.22e7 s² (RMS 24 s), unseen outer
8.9e6 on 15,372 rows. The excess over the 2025 reading is 8.9e6 / 9.9e6 / 1.7e7. At the end of the 2026 interval nearest zero it is
6.3e5 / 1.6e5 / 0. Every other direction is below 8e6 (`D_implied_change_july2026`). None reaches 1e7 s² with an interval that clears
2025.

## What this adds

* On July 2026 LFPG unseen rows (436 pinned), v69 is the best column and every chain step improves on the one before, as on 2025.
  Coverage-regime weights read on 2025 (outer, network share, masked and own_all members, stacker) hold within noise in a scored month,
  measured on real 2026 observations rather than the thinned simulation. This matches the closed member-weight result
  (`docs/member_thin/`, `docs/member_weights/`).
* The EOBT_1 repair weight, set in September without any 2026 label, now has a first open-observation reading. The implied weight is
  0.31 [0.00, 0.68] (all classes; component-level 0.16–0.29), which includes the shipped 0.5.
* The pin is usable as a target for comparing components. It is not usable for day-level directions (the published-data steps), where
  2025 shows a bias.

## Files

* `tools/val2026_components.py`: the component tables (`runs/val2026/components_dev.parquet` 6bff5818...,
  `components_rank.parquet` 0b75cbab...; receipt `runs/val2026/components.json`, copied to `docs/val2026/components_receipt.json`, with every input's SHA-256 including the 160 member tables)
* `tools/val2026_screen.py`: readings A–E (`docs/val2026/results.json`; console `runs/val2026/screen_console.txt`), 2,000 day
  replicates, seed 20261007
* Rerun: `.venv\Scripts\python.exe -X utf8 -B tools/val2026_components.py`, then `tools/val2026_screen.py` (about 2 min, about 2 GB)
