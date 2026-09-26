# LIRF context network — round B: design NO-GO by 0.05 points (25 Sep), development selected under the user's override, exported as v29 (26 Sep)

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

## Development round (26 September) — [`configs/adsb_ctx_lirf_v1.json`](../../configs/adsb_ctx_lirf_v1.json), selected `lctx_b100_up`

Run under the user's override of the design gate ([`override_design_gate.json`](override_design_gate.json)); declared 16:51 UTC on
25 September, before any development fit. 16 seeds (301–316) of `ctx_lirf`, two lanes per Azure worker as its round-A lanes finished.
Candidates change only v27's two LIRF non-population blends; the development rule is unchanged. Scorer
`tools/run_adsb_ctx_lirf_dev.py` ([`development_results.json`](development_results.json), commit `a516fa5`).

| Candidate | Jan | Jul | Nov | Dec | Mean | Intervals < 0 | Eligible |
|---|---|---|---|---|---|---|---|
| lctx_b50 (half ens_ctx_lirf, half ens_nnl) | −0.021 | −0.061 | −0.046 | −0.042 | −0.042 | 4/4 | yes |
| lctx_b100 (ens_ctx_lirf replaces ens_nnl) | −0.038 | −0.108 | −0.082 | −0.071 | −0.075 | 4/4 | yes |
| **lctx_b100_up** (ens_ctx_lirf, net weights 0.40 matched / 0.35 unmatched) | −0.051 | −0.115 | −0.108 | −0.137 | **−0.103** | 4/4 | **selected** |

Percent RMSE change against v27 per recipe month. `lctx_b100_up`'s paired-day intervals: January [−0.32, −0.09], July [−0.55, −0.09],
November [−0.30, −0.11], December [−0.34, −0.12] s. LIRF improves on matched and unmatched rows in every month (for example July
matched 578.6 → 577.4 s, unmatched 1,182.1 → 1,177.6 s; December 285.6 → 284.2 / 511.9 → 509.7 s).

## Coverage-regime ship check — [`configs/adsb_ctx_lirf_regime_check_v1.json`](../../configs/adsb_ctx_lirf_regime_check_v1.json)

Declared 10:01 UTC, before any thinned score of ens_ctx_lirf (`tools/ctx_lirf_regime_check.py`, commit `5b9df4c`). **Passes on both
months.**
- **July 2025, LIRF thinned to its dark July 2026 coverage:** natural −0.115%, **thinned −0.390%**, interval [−1.81, −0.51]. With
  LIRF dark the gain more than triples: the thinned LIRF unmatched rows (15,125) go 644.2 → 638.7 s.
- **December (EDDF thinned only; LIRF natural):** natural −0.137%, thinned −0.133%, interval [−0.34, −0.11].
- The natural rebuilds reproduce v27 and the candidate within 0.0007 s ([`regime_check_2025-07.json`](regime_check_2025-07.json),
  [`regime_check_2025-12.json`](regime_check_2025-12.json)). A first run was stopped by the workstation's memory guard (host
  memory, not the tool) and re-run on the user's instruction.

## Export v29 (26 September, ~12:45 UTC)

[`tools/export_adsb_ctx_lirf.py`](../../tools/export_adsb_ctx_lirf.py) → `runs/submissions/vigorous-whistle_v29.parquet`
(SHA-256 `ccf304fbcd4d22ff17b2e0b7cf686c2d487de520e092dcfaa7e8831c736bc576`, 1,342,807 bytes). Record: [`export.json`](export.json).

- **Recipe:** v28 (round A's pool_s75 committee mix) with round B's LIRF non-population blends. Round A changed only non-LIRF rows,
  round B only LIRF non-population rows, as both declarations fixed.
- **Inputs:** ens_ctx_lirf is the mean of the 16 round-B seeds' twelve-month refits (2,083,560 fitted rows, 16 epochs each); every
  other ensemble equals v28's recorded column; one ADS-B ranking table (`1dcf5c7f…`, v28's); declared token content.
- **Control rebuild:** v28 rebuilt from its export detail — committee mix and values differ from the recorded v28 by 0.0 s; the
  integers equal the uploaded v28.
- **Change against v28:** 25,673 rows change (all LIRF non-population), RMS 29.7 s on LIRF (8.3 s over all rows), mean −3.2 s,
  median 9 s, p99 115 s, max 739 s. Matched scope 19.3 s RMS, unmatched 34.0 s; January 19.9 s, July 34.9 s. Every other row equals
  the uploaded v28.
- **Independent verification** ([`verification_v29.json`](verification_v29.json), `tools/verify_adsb_ctx_lirf.py`): all 16
  ctx_lirf networks reload on Windows and re-predict 5,000 sampled ranking rows within 0.0024 s; every ensemble column (six ensembles)
  equals the plain mean of its saved seeds (0.0 s); v9, residual, committee, committee mix and scope equal v28's; the rebuilt v28
  equals the recorded one (0.0 s). Plain arithmetic max(w_v9 v9 + w_tree tree + w_net net, −12) on the 26,499 LIRF non-population
  rows reproduces the recorded v28 with v25's weights and the exported v29 with the selected ones (both within 2e-12 s, integers
  equal); the other 318,342 rows equal the uploaded v28. The file contract holds.
- **Audit** ([`export_audit_v29.json`](export_audit_v29.json), `tools/audit_export.py --version 29 --compare 28 27 9`): v28's shape,
  the same flags, one row moves more than 600 s and none more than 1,800 s.
- **Expected transfer (not a prediction of the official score):** development −0.103% (December −0.137%, July −0.115%); the July
  check with LIRF dark reads −0.390%, the closest regime to July 2026.
