# Chain-wide layer-strength audit (fork R, 7 October 2026) — no layer clears the screen bar

**Question.** Lever 11 found the stacker under-applied at its a-priori weight. Is any other layer of v69's chain applied at the wrong
strength?
Nothing is declared, built or uploaded.

Tool `tools/layer_strength.py`; outputs [`screen.json`](screen.json), [`table.json`](table.json); run on Azure arcw4 (08:09–08:17 UTC VM
clock). The full table is `runs/layer_strength/table.parquet` (`4db078fc...`, git-ignored).

## Method

- **`--build`** rebuilds v69's development chain with every intermediate column. The model part comes from the pinned v66 chain files
  `runs/pubrefit/chain/ship` (`dmp` v46 -> v51m ... v55m, `ctl` / `cand` stack -> v48 -> ext -> new). The stacking value is w 0.65 with
  ten seeds, from fork K's stored predictions. The tail (intruder bound, identity extension, own-motion v1, own-motion v2) is
  `run_stack_resid_v3.Chain.run` and `run_ctx_ownpoints_chain.own_motion_v2`, with a per-layer multiplier added.
- **Identity:** the rebuilt final value equals v69's development proxy `runs/stack_weight/dev/w65_bag10/proxy.parquet` (`c088e409...`) with
  a maximum difference of **0.0**.
- **The strength k.** For layer j with change d_j (after minus before), k = 1 is the shipped strength. The per-month estimate is
  k* = 1 + <r, d> / <d, d>, with r = label minus the v69 dev proxy. The month-blocked k for month m is fitted on the other three months.
  The held month is then read with v57's readings: change % and the paired-day 95% interval (`run_intruder_bound.interval`, seed 20261004).
- **Tail layers** (stacker and the four bounds) are exact in-chain: the rest of the tail is rerun with the layer scaled, and the local
  response comes from k = 1.1.
- **Upstream layers** (members, rematch, published layers) use first-order propagation. Rows that a later full-weight bound replaces get 0.
  Rows that own-motion v2 changes get one half. With alpha = 0.4, the published layers absorb 40% of a change's airport off-block-day mean.
- **Approximations, stated:** the stacker is not refitted when an upstream layer is scaled, and the published layers are not rerun.
  A layer that cleared the bar would need a real in-chain round.
- **Screen bar** (as all final-week screens): eq mean ≤ −0.05%, no month worse than +0.01%, at least two equal-footing intervals below zero.

## Month RMSE after each layer (development, s)

| after | Jan 2025 | Jul 2025 | Nov 2025 | Dec 2025 |
|---|---:|---:|---:|---:|
| v46 stack | 322.8585 | 257.7185 | 184.9050 | 167.5137 |
| own_all member (v51m) | 321.7597 | 256.1838 | 183.9403 | 166.2210 |
| masked member (v52m) | 321.7440 | 256.1860 | 183.8118 | 166.0372 |
| LIRF masked member (v53m) | 321.7094 | 256.1845 | 183.8367 | 166.0054 |
| LIRF dark re-read (v54m) | 321.7003 | 256.1845 | 183.8367 | 166.0054 |
| LIRF matched member (v55m) | 321.7003 | 256.1827 | 183.8024 | 165.9297 |
| no-NM rematch | 321.6031 | 256.1710 | 183.7913 | 165.9173 |
| v48 reconciliation | 321.2830 | 255.2888 | 183.2968 | 165.2796 |
| extreme-day | 321.2844 | 255.0644 | 183.2968 | 165.2796 |
| variance re-allocation | 321.1705 | 254.7141 | 183.2335 | 165.1738 |
| stacker (0.65, ten seeds) | 320.7738 | 254.0599 | 182.2658 | 163.9770 |
| intruder bound | 320.2563 | 251.1797 | 181.9226 | 163.6578 |
| identity extension | 320.0695 | 250.9436 | 181.8997 | 163.4508 |
| own-motion v1 | 320.0488 | 250.8731 | 181.9176 | 163.3654 |
| own-motion v2 = v69 dev proxy | **320.0147** | **250.8358** | **181.8835** | **163.3537** |

## Result per layer

The "In-chain held-month change" columns are the month-blocked k applied in-chain (tail layers) or first-order with alpha 0.4 (upstream).
"Below 0" counts equal-footing intervals entirely below zero.

| layer | k* Jan / Jul / Nov / Dec | month-blocked k | held-month change % Jan / Jul / Nov / Dec | eq mean | below 0 | strength tested before |
|---|---|---|---|---:|---:|---|
| own_all member | 1.18 / 1.26 / 0.98 / 1.03 | 1.11 / 1.07 / 1.17 / 1.16 | −0.009 / −0.016 / +0.026 / +0.014 | −0.0036% | 1 | lever A (0.55–1.00) |
| masked member | 0.69 / 0.73 / 0.86 / 1.10 | 0.87 / 0.86 / 0.81 / 0.75 | −0.003 / −0.004 / −0.002 / +0.015 | +0.0024% | 0 | v52 round |
| LIRF masked member | 0.88 / 0.44 / −0.30 / 1.58 | 0.57 / 0.76 / 0.98 / 0.52 | +0.001 / −0.001 / −0.000 / +0.007 | +0.0024% | 0 | v53 round |
| LIRF dark re-read | 0.16 / – / – / – | – | 0 | 0.0000% | 0 | v54 grid (dark months only) |
| LIRF matched member | – / −0.06 / 2.43 / 3.74 | 1.05 / 3.21 / 0.83 / 0.37 | 0.000 / +0.069 / +0.002 / +0.023 | +0.0309% | 0 | v55 grid |
| no-NM rematch | 1.69 / 0.55 / 1.17 / 0.49 | 0.57 / 1.44 / 0.93 / 0.98 | +0.009 / +0.016 / +0.000 / −0.000 | +0.0082% | 0 | 0.25–1.0 explored |
| v48 reconciliation | 0.69 / 1.20 / 0.71 / 0.66 | 0.86 / 0.68 / 0.85 / 0.88 | −0.005 / +0.026 / −0.013 / −0.023 | −0.0008% | 2 | tau / prior / beta variants |
| extreme-day | −0.46 / 1.83 / – / – | 1.83 / −0.46 / 1.82 / 1.82 | +0.000 / +0.070 / 0 / 0 | +0.0232% | 0 | calibrated; refit read (lever F) |
| variance re-allocation | 1.39 / 1.85 / 0.79 / 0.97 | 1.24 / 1.05 / 1.44 / 1.37 | −0.004 / −0.004 / +0.032 / +0.013 | +0.0020% | 1 | k fitted; refit read (lever F) |
| stacker | 1.10 / 1.03 / 1.13 / 1.44 | 1.16 / 1.20 / 1.17 / 1.09 | −0.001 / +0.007 / −0.024 / −0.032 | −0.0084% | 1 | lever 11 (0.5–1.0, 3 / 10 seeds) |
| intruder bound | 1.30 / 1.09 / 0.94 / 0.82 | 1.05 / 1.04 / 1.08 / 1.10 | −0.003 / −0.005 / +0.004 / +0.014 | +0.0019% | 0 | w 1.0 declared |
| identity extension | 1.17 / 1.13 / 0.99 / 1.10 | 1.11 / 1.14 / 1.14 / 1.14 | −0.001 / −0.001 / +0.000 / −0.001 | −0.0011% | 0 | none (one-sided bound) |
| own-motion v1 | 0.97 / 1.10 / 0.45 / 0.87 | 0.72 / 0.64 / 0.98 / 0.70 | +0.000 / +0.005 / −0.002 / +0.001 | +0.0021% | 0 | w 1.0 declared |
| own-motion v2 | 0.83 / 1.00 / 0.88 / 0.59 | 0.82 / 0.77 / 0.81 / 0.90 | −0.000 / +0.001 / −0.000 / −0.003 | −0.0009% | 0 | w 0.5 declared, 1.0 worse |

At fixed k ∈ {0.75, 1.25, 1.5}, the best reading in `screen.json` is the stacker at k 1.25 in-chain: eq −0.0196%, one interval below
zero. That is about weight 0.81, close to lever 11's w80_bag10 (−0.0655% against v68, with intervals crossing zero).

**Verdict: no layer clears the screen bar.** The best month-blocked readings are the stacker at −0.0084% and the own_all member at
−0.0036%, both an order of magnitude below the −0.05% bar. The only layer with k* above 1 in all four months is the stacker
(1.03–1.44); its month-blocked gain is negligible and July gets worse. Every other layer's k* changes side of 1 between months.

At development level, v69's chain sits at its strengths within noise. Upstream rows are first order (no stacker refit, no published-layer rerun); with gains this small,
a real in-chain rerun would not change the verdict.

No official score, no audit, nothing under runs/submissions; arcw4 was deallocated after the hash-checked mirror (`out.tar` `fc56837d...`).
