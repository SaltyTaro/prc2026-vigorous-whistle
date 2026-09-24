# adsb_v30 — the LIRF-inclusive network's weights above adsb_nn2's grid (24 September 2026)

**Status: NOTHING SELECTED. v25 (234.8575) remains the incumbent; its LIRF weights (network 0.30 on matched, 0.25 on unmatched
non-population LIRF rows) stay. No refit, export or upload.**

## Why

adsb_nn2 selected k32_m30_u25 (v25, official −0.191% vs v24). Its matched-LIRF network weight 0.30 was the top of the declared grid
and carried most of the gain, so the next zero-fit question was whether more weight pays.

## Protocol

Declaration `configs/adsb_v30_v1.json` (06:24:50 UTC, commit `7312be5`), tool `tools/run_adsb_v30.py` (adsb_nn2's scoring code,
imported unchanged), fits nothing. Grid: matched-LIRF network weight c ∈ {0.30, 0.45, 0.60, 0.75} as (0.15, 0.85 − c, c) × unmatched
c ∈ {0.25, 0.35} as (0.75 − c, 0.25, c); control = k32_m30_u25. Smoke test: every candidate collapsed onto the control, which
reproduced k32_m30_u25's recorded development RMSE on all four months with difference 0.0. Rule unchanged.

## Result — `development_results.json`, `runs/adsb_v30/dev/selection.json`

Whole-month RMSE change against the control (v25's development predictions):

| candidate | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| m30_u35 | −0.001% | −0.012% | +0.001% | −0.016% | −0.007% | 0/4 |
| m45_u25 | 0.000% | +0.084% | −0.001% | −0.031% | +0.013% | 0/3 |
| m45_u35 | −0.001% | +0.072% | +0.001% | −0.048% | +0.006% | 0/4 |
| m60_u25 | 0.000% | +0.276% | +0.017% | −0.028% | +0.066% | 0/3 |
| m75_u25 | 0.000% | +0.575% | +0.054% | +0.010% | +0.160% | 0/3 |

No candidate is eligible. **0.30 is the optimum of the matched weight, not the edge of a larger gain:** July (where the network
alone is far worse on LIRF, 824 vs 553 s) turns against any higher weight, November is flat, and only December wants a little more.
The unmatched weight 0.35 moves no month by more than 0.016%. The LIRF blend weights are closed with the current networks; a further
LIRF gain needs a better LIRF network, not more weight on this one.
