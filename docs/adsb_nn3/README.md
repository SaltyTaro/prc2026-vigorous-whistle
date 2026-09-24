# adsb_nn3 — a LIRF-inclusive network on the NM-clock residual target (24 September 2026)

**Status: NOTHING SELECTED. v25 (234.8575) remains the incumbent. No export or upload. The twelve-month refits of the 16 nnlr seeds
ran for scheduling only and are not used.**

## Why

adsb_v30 found that v25's LIRF network (nnl) cannot take more than 0.30 of the matched LIRF blend: on a handful of July LIRF rows with
labels of 6,800–9,200 s it predicts 900–2,800 s where the tree table and v9 predict 5,300–7,700 s. The hypothesis was long NM clocks,
and the NM-clock residual target (`target_offset nm_clock`, tools/run_adsb_nn2.py) passes the clock through.

## Design (October only) — `configs/adsb_nn3_design_v1.json`, `design_report.json`

nnlr (nnl + residual target) seeds 1–6 and nnl seeds 3–4, read on October's two LIRF scopes against the cross-fitted v9 (best blend):
nnl mean of 4 356.427 s, nnlr mean of 4 356.864, mix nnl 1–2 + nnlr 1–2 356.386. The declared go rule held narrowly through the mix
(−0.04 s). **The premise failed:** on long rows (v9 ≥ 7,200 s) nnlr read 3,532 s against nnl 4,076 and v9 929.

The follow-up diagnosis (July development rows, the ones behind adsb_v30's result) showed why: those rows are **schedule fills
outside the v9 fill population** — the label equals the scheduled time to takeoff within seconds, the NM clock proxy is an ordinary
700–1,300 s and the aircraft type is present. There is no long clock to pass through; the tree table predicts toward the schedule
interval, every network an ordinary taxi.

## Declaration and development — `configs/adsb_nn3_v1.json` (07:06:42 UTC, commit `4cda32e`)

16 nnlr seeds (401–416) on arcw0–7 from 07:07 UTC, two lanes per worker; development folds complete 08:49 UTC; seed mean
`runs/adsb_nn3/ens_nnlr` (pairwise correlation 0.97). Twelve candidates: nnlr alone (`r_`) or averaged with nnl (`x_`), matched LIRF
network weight 0.30/0.45/0.60, unmatched 0.25/0.35; control v25. Smoke test passed before any fit (control = v25 exactly).

## Result — `development_results.json`, `runs/adsb_nn3/dev/selection.json`

| candidate | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| r_m30_u25 | +0.000% | −0.000% | +0.018% | −0.041% | −0.006% | 0/4 |
| r_m30_u35 | −0.002% | −0.011% | +0.023% | −0.056% | −0.012% | 0/4 |
| x_m30_u35 | −0.003% | −0.021% | +0.007% | −0.043% | −0.015% | 1/4 |
| x_m45_u35 | −0.003% | +0.048% | +0.006% | −0.091% | −0.010% | 1/4 |
| r_m60_u35 | −0.002% | +0.237% | +0.054% | −0.121% | +0.042% | 1/4 |

No candidate is eligible: November is worse for every candidate, July for every matched weight above 0.30. December again wants more
network weight. The residual target changes little because the failing rows are fills, not long clocks. Next: adsb_v31, a fill gate on
the LIRF network (tools/run_adsb_v31.py), declared on v25's nnl.
