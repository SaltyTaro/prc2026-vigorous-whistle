# adsb_v29 — the v24 blend numbers re-read with the network in the committee table (24 September 2026)

**Status: NOTHING SELECTED. v24 (235.3074) remains the incumbent; its numbers (network share 0.5, outer weight 0.7 on matched and
on unmatched non-LIRF rows) stay. No refit, export or upload.**

## Why

On non-LIRF rows v24 is `0.3 × (v9 + residual) + 0.7 × [(1 − a) × committee + a × network]`, a = 0.5. The residual is zero on
unmatched rows, so 0.3 of every unmatched non-LIRF prediction is still v9. The 0.7 was chosen in v18 with a tree-only committee
table; the network moved that table by −1.46% on December, and adsb_nn's December reading preferred a larger network share.

## Protocol

Declaration `configs/adsb_v29_v1.json` (01:26:24 UTC, commit `25cdab7`), tool `tools/run_adsb_v29.py`, fits nothing. Twelve
points: a ∈ {0.5, 0.6} × outer weight on matched non-LIRF rows ∈ {0.7, 0.8} × on unmatched ∈ {0.7, 0.8, 0.9}; the control
(0.5, 0.7, 0.7) is nn50. Rule of adsb_v25/v26/v27/nn unchanged. Smoke test (`runs/adsb_v29/smoke_dev`, provisional declaration):
every candidate collapsed onto the control, which reproduced nn50's recorded development RMSE on all four months with difference
0.0; no candidate reading existed before the declaration except nn60 = a60_m70_u70, disclosed in it.

## Result — `development_results.json`, `runs/adsb_v29/dev/selection.json`

Whole-month RMSE change against the control (v24's development predictions), all rows:

| candidate | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean |
|---|---|---|---|---|---|
| a50_m70_u80 | +0.050% | +0.062% | −0.007% | −0.061% | +0.011% |
| a50_m70_u90 | +0.122% | +0.162% | +0.029% | −0.076% | +0.059% |
| a50_m80_u70 | +0.100% | +0.073% | +0.235% | +0.057% | +0.116% |
| a60_m70_u70 (= nn60) | +0.020% | +0.041% | +0.038% | −0.127% | −0.007% |
| a60_m70_u80 | +0.081% | +0.123% | +0.045% | −0.174% | +0.019% |
| (other five points) | worse | worse | worse | ≤ −0.121% | ≥ +0.113% |

No candidate is eligible: every one is worse than the control on at least two months. Raising the outer weight hurts January,
July and November on both row groups — v9 still earns its 0.3 as a diverse partner even on unmatched rows, and the residual stage
still earns it on matched rows. Only December prefers more network and more committee, the same month disagreement that excluded
nn60 in adsb_nn. The shipped weights sit at the optimum of three of the four months.

**Lesson:** re-weighting the existing tables is closed again; the next gain has to come from new members (adsb_nn2: more network
seeds, a LIRF-inclusive network for the LIRF tables, and an NM-clock-residual network if its October design reading earns it).
