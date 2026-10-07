# Lever 12: LFPG-specialised context network — design NO GO (7 October 2026)

Fork N, at the user's word ("start Paris-only network bet"). Question: does an own-points context network fitted on LFPG rows only improve the
shipped member own_all (v66 / v69 network table at weight 1.00) on LFPG, the largest projected ranking error pot (rows never seen on the ground,
6.4e9 s², `docs/hunt/README.md`)? No official score, leaderboard read, scoring-month label or ranking row entered any step. Hand-off log:
[`HANDOFF.md`](HANDOFF.md).

## Declaration

[`configs/lfpg_net_design_v1.json`](../../configs/lfpg_net_design_v1.json) (7121baf, 04:38 UTC, before any fit), GPU authorisation
`infra/aws/gpu_authorization_20261007b.json`. Two variants, each a matched-row and an unmatched-row network (the own_all pair), 8 seeds (9-16),
October design fold, LFPG rows only (`rows.exclude_airports` = the other nine airports), unchanged runner `tools/run_adsb_ctx3.py`:
`lfpg_own` (dl32own / m25dl32own settings) and `lfpg_b512` (batch 512, lr 0.001, dropout 0.2). Control = lever 5's GPU fits of
dl32own / m25dl32own (variant hashes equal on all 16). Items: the specialist mixed into own_all on LFPG rows at c = 0.5 / 1.0. Bar: +0.05 pts
of the recipe proxy at b = 1.0, LFPG RMSE below the control, paired-day interval below zero. October LFPG coverage (matched 19.5%, gate 0.3%)
equals the ranking months' (20.6% / 21.1%), so the October reading is already at 2026 LFPG coverage; no separate thinned reading.

## Result ([`design_report.json`](design_report.json), `tools/lfpg_net_design_report.py`)

October, 20,493 LFPG rows, 8-seed means:

| | LFPG RMSE (s) | matched | unmatched | error corr. with control |
|---|---:|---:|---:|---:|
| control own_all (global fit) | 216.70 | 196.09 | 221.41 | |
| lfpg_own (LFPG only) | 227.71 | 203.32 | 233.24 | 0.935 |
| lfpg_b512 (LFPG only) | 227.83 | 203.29 | 233.39 | 0.946 |

| item | LFPG vs control | gain at b = 1.0 (pts) | interval (s) | GO |
|---|---:|---:|---|---|
| lfpg_own_c50 | +0.87% | -0.101 | [+0.097, +0.285] | no |
| lfpg_own_c100 | +5.08% | -0.363 | [+0.447, +0.898] | no |
| lfpg_b512_c50 | +1.17% | -0.113 | [+0.088, +0.338] | no |
| lfpg_b512_c100 | +5.14% | -0.362 | [+0.378, +0.957] | no |

- The LFPG-only networks are about 5% worse than the global member on LFPG, matched and unmatched alike; even a 50/50 mix worsens LFPG
  unmatched rows (+1.2 / +1.5%) and the proxy (intervals strictly above zero). Only the matched LFPG rows gain slightly at c = 0.5
  (-1.05 / -0.72%), 20% of the rows.
- Reading: as for the unmatched-row tree specialists (`docs/adsb_unm/`), the global fit already serves LFPG better than an LFPG-only fit;
  the other airports' rows carry information LFPG's 140,707 fitting rows cannot replace. Neither more optimiser steps nor more regularisation
  (lfpg_b512) changes that.
- **Verdict: NO GO.** No development round declared, no ranking file.

## Compute

No Azure VM was started. The workstation ran orchestration, git, hash checks and the design report.
