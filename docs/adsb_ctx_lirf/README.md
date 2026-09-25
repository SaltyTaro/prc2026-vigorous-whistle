# LIRF context network — round B design (25 September 2026): NO-GO by 0.05 points

**Question.** LIRF carries 11–45% of development SSE per recipe month, and v27's context network never touches it (fitted
without LIRF). Does a context network fitted with LIRF's ordinary rows beat the LIRF-inclusive plain network `nnl` (ens_nnl)
in v27's LIRF blends?

**Declaration:** [`configs/adsb_ctx_lirf_design_v1.json`](../../configs/adsb_ctx_lirf_design_v1.json), 12:15 UTC, commit
`a03c87f`, before any fit.
- **Variants:**
  - `ctx_lirf`: round-1 ctx settings with LIRF non-population rows in fitting, fill population excluded as for nnl; 4 seeds.
  - `ctx_lirf_noadsb`: the same without ADS-B token features (July 2026's LIRF has no ADS-B ground coverage); 2 seeds.
- **Fold:** the October design fold (seasonal fit partition), the same fit partition as the 16 nnl development seeds.
- **Execution:** CPU only, as a third 4-thread lane beside round A's two lanes on arcw0, 2, 3, 4, 5 and 6 (12:16–15:30 UTC).
  - arcw4's fit was paused (SIGSTOP) from 13:14 to 13:35 UTC so that v27's last refits could finish, then resumed.
  - The LIRF fits' own fit times (about 1.9 h) include the shared CPU. No setting changed.
- **Report:** [`design_report_1.json`](design_report_1.json) (`tools/ctx_lirf_design_report.py`).

**Rows:** 14,027 October LIRF non-population rows (8,815 matched, 5,212 unmatched). v9 alone scores 360.28 s.

| seeds | ctx_lirf alone | nnl alone | 0.7 v9 + 0.3 ctx_lirf | 0.7 v9 + 0.3 nnl | change (all / matched / unmatched) | paired-day interval |
|---:|---:|---:|---:|---:|---|---|
| 1 | 383.75 | 393.65 | 355.57 | 358.06 | −0.69 / −0.54 / −0.95% | [−4.46, −0.56] |
| 2 | 369.79 | 380.18 | 353.37 | 356.78 | −0.96 / −0.81 / −1.20% | [−4.74, −2.04] |
| 3 | 366.82 | 376.39 | 353.12 | 356.55 | −0.96 / −0.82 / −1.19% | [−4.72, −1.98] |
| 4 | 366.33 | 375.81 | 353.28 | 356.65 | **−0.95** / −0.73 / −1.29% | [−4.46, −2.18] |

- **Decision (declared rule): NO-GO.** The 4-seed blend change is −0.95%, against a bar of −1.00%.
- **The context network is consistently better than nnl on LIRF:**
  - every ctx_lirf seed beats every nnl seed alone (372.8–383.8 s against 384.7–393.7 s);
  - the blend advantage is steady at about −0.95% from 2 to 4 seeds;
  - the interval is well below zero;
  - the gain is largest on unmatched rows (−1.29%).

  It is smaller than the bar set before the fits.
- **No-ADS-B variant (2 seeds):** −0.05% better than ctx_lirf on all LIRF rows but worse on unmatched ones (358.74 against
  355.07 s), so ctx_lirf would have stayed the variant.
- **Fitting LIRF costs the other airports a little (L3):**
  - non-LIRF October RMSE at 4 seeds: 156.86 against round-1 ctx's 156.46 s;
  - recipe proxy at b = 1: −1.594% against −1.731%.

  A LIRF context network would therefore only join the LIRF blends, never replace v27's non-LIRF network table.
- **Expected size if pursued:** in v25, ens_nnl's LIRF weights added −0.19% at the recipe level. Replacing it with a
  network that is about 1% better on LIRF ordinary rows suggests roughly −0.05% to −0.15%.

Whether to run the development round regardless (a user override of the design gate, CPU only, with the development round's
own declared selection rule unchanged) is the user's decision. Nothing is scored on a recipe month here.
