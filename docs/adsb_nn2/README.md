# adsb_nn2 — more network seeds and a LIRF-inclusive network (from 24 September 2026, 01:15 UTC)

**Status: SELECTED `k32_m30_u25` (mean −0.152% over the four recipe months, every month better, every paired-day interval below
zero). Twelve-month refits running; export, verification and audit follow; upload only with the user's approval. v24 (235.3074) is
the incumbent until then.**

## Why

v24's network (adsb_nn) moved the official score by −1.47%, entirely through the non-LIRF committee table. Two follow-ups:

1. **More seeds.** The October seed curve was still improving at 8 seeds (single 171.5 → mean of 8 164.5 s); v24 used 16.
2. **LIRF.** Every LIRF row is still v24's tree-and-v9 recipe: LIRF is 29–45% of the development SSE outside January 2025, and
   the ranking has 17,718 unmatched and 8,781 matched non-population LIRF rows in the declared blend scopes (15,515 of the
   unmatched ones in July 2026, when LIRF has no ADS-B ground coverage). v24's network was fitted without LIRF.

adsb_v29 (fits nothing, same morning) first re-read the v24 blend numbers and kept them (`docs/adsb_v29/README.md`).

## Design (October only) — `configs/adsb_nn2_design_v1.json`, `docs/adsb_nn2/design_report.json`

New tool `tools/run_adsb_nn2.py`: v24's network tool (hash-bound to adsb_nn_v1, untouched) plus one variant setting,
`target_offset` (`none` = the v24 maths exactly; `nm_clock` = learn label − NM clock proxy where the clock is present).
Two variants, fitted on the seasonal fit partition and read on October only (no recipe or ranking month):

- **nmres** (6 seeds): the v24 network on the NM-clock residual target, the transformation that gave the trees v6's −14%.
- **nnl** (2 seeds): the v24 network with LIRF's ordinary rows in fitting (fill population excluded, as the v22 LIRF CatBoosts).

Readings (committee October RMSE 163.413 s on 171,532 non-LIRF rows):

| run | network | best blend with the committee | recipe proxy | error corr. |
|---|---|---|---|---|
| base (v24 nnb) s1–s8 | 170.67–172.55 | −0.73 … −1.00% | −0.33 … −0.44% | 0.89 |
| nmres s1–s6 | 173.06–174.68 | −0.51 … −0.70% | −0.18 … −0.28% | 0.89 |
| nnl s1, s2 | 172.00, 172.18 | −0.78%, −0.68% | −0.33%, −0.28% | 0.89–0.90 |
| base mean of 8 | 164.51 | −1.44% | −0.66% | 0.93 |
| nmres mean of 6 | 167.53 | −0.91% | −0.34% | 0.92 |

Equal compute, mean of base 1..k + nmres 1..k against base 1..2k: worse at every k = 1..4 by 0.17–0.26 pp (blend) and
0.12–0.16 pp (recipe proxy). **nmres is dropped; the new committee seeds are plain v24 seeds.** nnl's non-LIRF readings do not
both lie in the base range, so **nnl stays out of the committee pool**.

On October LIRF non-population rows against the cross-fitted v9 (descriptive): the two-seed nnl mean alone is worse (376.7 vs
360.3 s) but blends — matched rows best at 0.2 (355.5 → 353.4), **unmatched rows best at 0.5 (368.2 → 358.5, −2.6%)**. v24's
network, which never saw LIRF, is useless there (548 s alone). Because v24's structure caps any table at 0.25 on unmatched LIRF
rows, the declaration gives the LIRF network explicit per-scope weights.

## Declaration — `configs/adsb_nn2_v1.json` (01:44:52 UTC, commit `7e050b9`)

- Fits: 16 further v24-recipe seeds 117–132 (`tools/run_adsb_nn.py`, `configs/adsb_nn_network_v1.json`, variant hash identical
  to v24's) → `ens_new`; 16 nnl seeds 201–216 (`tools/run_adsb_nn2.py`, `configs/adsb_nn2_network_v1.json`) → `ens_nnl`.
  Each seed: seasonal, forward and January–November refit folds, as v24's networks.
- 18 candidates: committee pool {v24's 16 seeds, 32 seeds} × matched LIRF (v9, tree, net) ∈ {(0.15, 0.85, 0), (0.15, 0.70, 0.15),
  (0.15, 0.55, 0.30)} × unmatched LIRF ∈ {(0.75, 0.25, 0), (0.50, 0.25, 0.25), (0.25, 0.25, 0.50)}. Network share 0.5 and outer
  weight 0.7 unchanged. Control = nn50 (v24).
- Rule unchanged (no recipe month worse on all rows; mean ratio ≥ 0.02% below 1; every touched month's paired-day interval below
  zero; lowest mean ratio).
- Smoke test (`runs/adsb_nn2/smoke_dev`): every new ensemble replaced by v24's and every LIRF network weight zeroed; the control
  reproduced nn50 on all four months, every candidate equal to it (0.0), the independent LIRF-blend arithmetic check passing.

## Execution log

- 01:15 UTC arcw0–7 started. Design fits 01:19–01:40 (one per worker). Declaration 01:44:52.
- 01:45 UTC development chains launched (`tools/nn2_launch_dev.sh nnb`): two lanes per worker, each lane one committee seed and
  one nnl seed — development folds of both first, then their twelve-month refits (scheduling only; used only if selected).
  Two concurrent 8-thread jobs run at ~58 s/epoch against 38.5 s for one (1.33× throughput).
- Ship tooling prepared while the fits run: `tools/export_adsb_nn2.py` (dry-tested: its v24 rebuild equals the recorded v24
  values with difference 0.0 and the uploaded integers exactly) and `tools/verify_adsb_nn2.py` (scope recomputation dry-tested:
  8,781 / 17,718 rows, identical to the export's).
- 04:12 UTC all 32 seeds' development folds complete (no failure; a first progress waiter stopped on a stray ssh line, not a
  failure, and was re-armed). Mirrored (`tools/nn_mirror.sh`, `tools/nn2_mirror.sh`), seed means built
  (`tools/ensemble_adsb_full.py` → `runs/adsb_nn2/ens_new`, `ens_nnl`; mean pairwise seed correlation 0.97).

## Development results — `development_results.json`, `runs/adsb_nn2/dev/selection.json`

Whole-month RMSE change against the control (v24's development predictions), all rows; eligible = the declared rule:

| candidate (pool, LIRF net matched / unmatched) | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean | intervals < 0 | eligible |
|---|---|---|---|---|---|---|---|
| k32_m0_u0 (32 seeds only) | −0.019% | −0.004% | −0.034% | +0.002% | −0.014% | 2/4 | no |
| k16_m30_u0 (LIRF matched only) | 0.000% | −0.159% | −0.057% | −0.166% | −0.095% | 3/3 | yes |
| k16_m0_u25 (LIRF unmatched only) | −0.034% | −0.044% | −0.023% | −0.071% | −0.043% | 1/4 | no |
| k32_m15_u25 | −0.053% | −0.182% | −0.094% | −0.169% | −0.125% | 4/4 | yes |
| k32_m30_u0 | −0.019% | −0.163% | −0.090% | −0.164% | −0.109% | 4/4 | yes |
| **k32_m30_u25 — selected** | **−0.053%** | **−0.207%** | **−0.113%** | **−0.235%** | **−0.152%** | **4/4** | **yes** |
| k32_m30_u50 | −0.042% | −0.229% | −0.100% | −0.263% | −0.158% | 2/4 | no |

k32_m30_u25 in detail (RMSE, s): January 327.972 → 327.798 (interval −0.38 … −0.04, 22/31 days better); July 266.810 → 266.258
(−0.97 … −0.18, 18/31); November 188.327 → 188.114 (−0.35 … −0.08, 25/30); December 173.772 → 173.363 (−0.64 … −0.20, 25/31).
LIRF RMSE 404.07 → 402.83 / 635.04 → 632.19 / 445.52 → 444.61 / 347.93 → 345.10; non-LIRF within ±0.09 s.

Reading: **the gain is the LIRF-inclusive network.** Doubling the committee pool to 32 seeds is worth only −0.014% (seed
averaging has saturated at 16). On non-population LIRF rows the 16-seed nnl mean alone equals the LIRF tree table (January 282.9 vs
281.1 s, November 287.5 vs 287.6, December 282.5 vs 283.1; July 824 vs 553, where one-day rows dominate) and is a different model,
so it blends: the matched weight 0.30 (the grid's top) carries most of the gain; 0.25 on unmatched rows adds a consistent −0.03 …
−0.07%; 0.50 there has a lower mean but fails two intervals. January 2025 (LIRF without ADS-B, the July 2026 regime) gains only from
the unmatched weight and the pool. For the ranking: January 2026 has 8,747 matched LIRF rows like December; July 2026's LIRF rows are
all unmatched like January 2025.
