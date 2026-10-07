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

# Lever 12b: matched-LFPG-rows development round — NOT accepted (7 October 2026)

At the user's word ("ok do it", ~05:40 UTC, relayed by the coordinator). Declaration
[`configs/lfpg_net_dev_v1.json`](../../configs/lfpg_net_dev_v1.json) (cc20721, 05:34 UTC, before any fit; network config
`configs/lfpg_net_network_v1.json`), **post-hoc narrowing disclosed** (chosen after the October design: only matched LFPG rows gained at c = 0.5).
One candidate, no weight search: on LFPG rows with an ADS-B match, own_all -> 0.5 own_all + 0.5 lfpg_dl32own (own_all's matched-row network fitted
on LFPG rows only, 16 seeds 641-656, LOMO folds as own_all's round); every other row keeps own_all; member weight 1.00. Amendment 1 (6ba67fd,
before any candidate table or reading): numeric control-chain identity and a same-path rebuilt control as the governing comparison (see below).
Tool `tools/run_lfpg_net_dev.py`; results [`results_v1.json`](results_v1.json), stage 1 [`results_stage1.json`](results_stage1.json).

Rows changed by the member: 2025-01 **0** (LFPG has no ADS-B match in January 2025), 2025-07 1,569, 2025-11 6,797, 2025-12 4,773, ranking 8,315.

## Identities

- Ranking: steps 1-5 of the unpinned export path with own_all equal v66's shipped steps (the float side table Y/rows.parquet to 5.8e-11 s,
  Linux vs Windows), and with fork K's stored ranking stack predictions **the path reproduces v69 (192c4a30...) byte for byte** on arcw1.
- Development: (a) the pinned control chain rebuilt on arcw1 equals runs/pubrefit/chain/ship to 4.5e-13 s (dmp / rows_all byte-identical);
  (b) w65_bag10 through this tool's scoring path on the ship files with fork K's predictions = the v69 dev proxy exactly (0.0); (c) the control
  rebuilt through the whole VM path (stacking predictions refitted on the rebuilt chain) = the v69 dev proxy to 4.5e-13 s (1,225 rows in the last
  bits). The governing comparison is candidate vs (c).

## Stage 1 (member level, information only)

| month | change | paired-day 95% (s) | rows |
|---|---|---|---|
| January 2025 | 0 | [0, 0] | 0 |
| July 2025 | -0.0055% | [-0.068, +0.032] | 1,569 |
| November 2025 | -0.049% | [-0.235, +0.018] | 6,797 |
| December 2025 | **+0.031%** | [+0.019, +0.089] | 4,773 |

(The stage-1 tool wrote its result and then stopped in a console print on January's empty alignment; the result file is complete.)

## Stage 2 (governing; v69's chain, variance re-allocation pinned, stacker w 0.65 ten seeds re-fitted)

| month | control (rebuilt) | candidate | change | paired-day 95% (%) | LFPG rows change |
|---|---|---|---|---|---|
| January 2025 | 320.0147 | 320.0070 | -0.0024% | [-0.015, +0.001] | -0.000% |
| July 2025 | 250.8358 | 250.8253 | -0.0042% | [-0.028, +0.016] | -0.082% |
| November 2025 | 181.8835 | 181.7191 | -0.0904% | [-0.201, -0.020] | -0.459% |
| December 2025 | 163.3537 | 163.4201 | **+0.0406%** | [-0.001, +0.095] | **+0.193%** |

v57's clauses: (1) no month worse than 0.005%: **no** (December +0.041%); (2) equal-footing mean +0.0113% (bar -0.02%): **no**; (3) every
equal-footing interval below zero: **no**. Transfer guard (member change RMS, ranking / development LFPG matched rows): 50.1 / 52.9 s = 0.95, ok.
Against the declared control (v69 dev proxy) the changes are identical to the shown precision. **NOT accepted; no ranking dry run; v69 stays.**
The January change is stacker-refit noise only (no member row changes there).

## Compute

Azure: arcw1 (06:36-07:47 UTC; temp-disk workspace
/mnt/scratch/prc, inputs fetched lazily through the workstation's read-only file server and an SSH reverse tunnel, hash-listed in
runs/lfpg_net/vm_fetch.jsonl), outputs mirrored as `runs/lfpg_net/mirror/out.tar` (SHA-256 8f353af3...) and extracted; arcw1 deallocated, server
and tunnel stopped; arcw2 started and deallocated unused. The workstation built the member tables (light) and orchestrated.
