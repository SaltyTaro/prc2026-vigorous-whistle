# adsb_v31 — a fill gate on the LIRF network (24 September 2026)

**Status: NOTHING SELECTED. v25 (234.8575) remains the incumbent. No export or upload.**

## Why

While diagnosing adsb_v30 on July 2025 development rows, the long non-population LIRF rows that make v25's LIRF network fail turned
out to be **schedule fills outside the v9 fill population** (label = scheduled time to takeoff within seconds, ordinary NM clock,
aircraft type present). The tree table predicts toward the schedule interval there; every network predicts an ordinary taxi
(adsb_nn3 confirmed the NM-clock residual target does not help). The gate returns the network's weight to the tree table on LIRF
scope rows where the tree table exceeds the NM clock proxy by more than T, and lets the network take more weight elsewhere.

## Protocol

Declaration `configs/adsb_v31_v1.json` (08:51:34 UTC, commit `6c52474`), tool `tools/run_adsb_v31.py` (adsb_nn2's scorer with the
gate in the LIRF blends and in the independent arithmetic check), fits nothing. Grid: T ∈ {1,800, 3,600} s × matched LIRF network
weight {0.30, 0.45, 0.60} × unmatched {0.25, 0.35}, on v25's nnl (adsb_nn3's network was not better). The July discovery is
disclosed in the declaration; January, November and December were inspected only through label-free gated-row counts
(T = 1,800: 35 / 270 / 22 / 33 rows in Jan / Jul / Nov / Dec 2025). Smoke test passed (control = v25 exactly). Rule unchanged.

## Result — `development_results.json`, `runs/adsb_v31/dev/selection.json`

| candidate | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| l_g3600_m30_u25 (gate only) | +0.001% | −0.109% | −0.003% | −0.000% | −0.028% | 0/4 |
| l_g3600_m45_u35 (best mean) | −0.000% | −0.161% | −0.003% | −0.048% | −0.053% | 0/4 |
| l_g3600_m60_u35 | −0.000% | −0.144% | +0.014% | −0.045% | −0.044% | 0/4 |
| l_g1800_m30_u25 | +0.005% | −0.035% | +0.017% | +0.058% | +0.011% | 0/4 |
| l_g1800_m45_u35 | +0.005% | −0.075% | +0.022% | +0.032% | −0.004% | 0/4 |

No candidate is eligible: no paired-day interval lies below zero on any month for any candidate. The 60-minute gate worsens no
month, but its gain sits almost entirely in July 2025 — the month the mechanism was found in; January and November barely move and
December only with the higher weights. The 30-minute gate also removes the network where it was right and worsens November and
December. **The rule's rejection is the right call: the evidence outside the discovery month is too thin to ship a row-level rule.**

## Execution note

The 16 nnlr seeds' twelve-month refits (adsb_nn3, scheduling only) were stopped when adsb_nn3 selected nothing and adsb_v31 was
declared on nnl (whose refits already existed); all workers were deallocated at 08:53 UTC. The partial refit12 directories on the
workers are unused.
