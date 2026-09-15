# v12_v4: v12 blended with the full-data learner (15 September 2026)

Declared in [`../../configs/adsb_v12_v4.json`](../../configs/adsb_v12_v4.json)
after v12 (official 246.4831, 3rd) and the null result of v12_v3 (the learner
prediction as a residual *feature*: −0.06% / −0.11% / −0.18%). The residual
model is exactly v12's (neighbour feature set, matched rows outside LIRF,
weight 1.0, refitted inside the run); each candidate then takes a fixed convex
combination with the no-LIRF full-data learner
([`../adsb_full/README.md`](../adsb_full/README.md)):

    value = (1 − w) · (v9 + residual) + w · adsb_full

on the blend scope — matched rows outside LIRF (`blend_matched_*`) or every
row outside LIRF (`blend_all_*`, where unmatched rows become
(1 − w) · v9 + w · adsb_full) — at w = 0.3 and 0.5. Rows outside both scopes
keep v9. Baseline: the recorded v12 development predictions
(332.993 / 274.585 / 198.544); rule: not worse on any month and at least 0.3%
better on average. Prior (descriptive) evidence: a 50/50 combination on all
non-LIRF rows gave 325.3 vs 326.3 (Jan), 212.1 vs 212.3 (Jul), 160.3 vs 164.5
(Nov) on those rows.

## Development ([`development_results.json`](development_results.json), `runs/adsb_v12_v4/selection.json`)

| Candidate | Jan 2025 | Jul 2025 | Nov 2025 | mean |
| --- | ---: | ---: | ---: | ---: |
| v12 baseline | 332.993 | 274.585 | 198.544 | — |
| `blend_matched_30` | 332.168 (−0.25%) | 273.877 (−0.26%) | 196.614 (−0.97%) | −0.49% |
| `blend_matched_50` | 332.187 (−0.24%) | 273.902 (−0.25%) | 196.229 (−1.17%) | −0.55% |
| **`blend_all_30`** | **331.826 (−0.35%)** | **273.881 (−0.26%)** | **195.769 (−1.40%)** | **−0.67%** |
| `blend_all_50` | 332.055 (−0.28%) | 274.438 (−0.05%) | 195.285 (−1.64%) | −0.66% |

All four are eligible; **`blend_all_30`** is selected (lowest mean ratio).
Paired-day bootstrap of the RMSE difference to v12: Jan [−2.34, −0.59] s,
Jul [−1.10, −0.34] s, Nov [−4.25, −1.83] s. Per airport on Nov 2025 versus
v12: LEMD −6.2%, EHAM −3.1%, EDDM −3.0%, LEBL −2.9%, LFPG −2.5%, EGLL −2.0%,
EDDF −1.5%, LSZH −0.7%, LTFM −0.6% (every airport improves; LIRF unchanged).
Residual refit rounds 1,173 (as v12). Exported as **v13**.

## Export ([`export.json`](export.json))

`tools/export_adsb_v11.py --config configs/adsb_v12_v4.json --selection
runs/adsb_v12_v4/selection.json --version 13`: v9 rebuilt from scratch, the
v12 residual refitted on the 271,115 in-scope rows of the three months (1,173
rounds), the learner's refit ranking predictions attached and blended at
w = 0.3 on every non-LIRF row: **317,942 rows touched** (LIRF 26,899 keep v9).
Output `runs/submissions/vigorous-whistle_v13.parquet`, SHA-256
**`c3985c532c1f9ca70cdfd699fe4a97e5037604d15d387cb2e596abd087a85ad2`**, 344,841
rows, min −12, max 120,549 (inherited).

## Verification ([`verification.json`](verification.json))

`tools/verify_adsb_v11.py --version 13 --config configs/adsb_v12_v4.json
--run-root runs/adsb_v12_v4`: receipts and parents re-checked, v9 reproduced
from scratch, the ADS-B features and the attached learner prediction reproduced,
the residual recomputed from the saved booster (difference 0.0), the blend
recomputed row by row, rows outside both scopes equal the uploaded v9, and the
independent segment re-derivation agrees on **300/300** sampled matched
departures. Not uploaded at the time of writing (next allowance 00:00 UTC,
16 September).
