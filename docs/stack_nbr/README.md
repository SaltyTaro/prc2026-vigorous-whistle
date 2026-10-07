# Neighbour-state columns in the guarded stacker (`configs/stack_nbr_v1.json`): NOT accepted (7 October 2026)

The round was declared in 9eb3e44 before any reading. Its governing run is [`tools/run_stack_nbr.py`](../../tools/run_stack_nbr.py)
(b92d3a5). That tool is a copy of `tools/run_stack_queue.py` with the queue switch replaced by a neighbour switch, `--nbr off|on`. With
the switch on, each stacker learner gets the three columns from `tools/nbr_columns.columns` (nx, nxk, nn). The two v71 neighbour layers
are re-applied with `tools/run_nbr_state.py`'s functions (state, slope, apply, constants, dev_table) and `tools/run_nbr_dark.py`'s
(with_gate, W_STATE). Their `main()` arithmetic is replicated with the base values replaced, and neither file was edited. The run used
arcw1 only, from 22:43 to 23:34 UTC, with driver [`tools/stack_nbr_remote.py`](../../tools/stack_nbr_remote.py). Results are in
[`results_v1.json`](results_v1.json), and the run log is in [`HANDOFF.md`](HANDOFF.md). No official score or leaderboard was used.
Nothing was written under `runs/submissions`, and nothing was uploaded.

## Decision: neither candidate accepted; v71 stays

- **sn65** (stacker with the columns, w 0.65, ten seeds, then steps 7-11) fails clauses 1, 2 and 3. July is worse by +0.0364%, with
  an interval entirely above zero, and the equal-footing mean is +0.0130%.
- **sn65_layers** (sn65, then nbr_state_v1 w75 and nbr_dark_v1 re-applied on sn65's values) passes clause 1. It fails clause 2: the
  equal-footing mean is −0.0156% against a −0.02% bar. It also fails clause 3, because July's interval [−0.0178, +0.0183] includes
  zero.
- Clause 4 (the stacker transfer guard) holds for the columns-on bags.

No candidate was selected, so no ranking candidate file was written: `runs/stack_nbr/rank/dry_v72/` does not exist.

## Identity checks (run first; all hold exactly)

- **Development, columns off.** w 0.65 with ten seeds, then steps 7-11, equals the control's base `runs/oooi_lhr/dev_v70_proxy.parquet`
  (51aaa6a6...) on every row, with max |diff| **0.0**. Its step-10 value equals v69's development proxy (c088e409...) with max |diff| 0.0.
  The 120 per-seed columns-off predictions of all four months equal stack_queue_v1's columns-off predictions exactly, and those are
  identical to stack_weight_v1's.
- **Development, layers.** The control is v70's development proxy with nbr_state_v1 w75 and nbr_dark_v1. It reproduces
  `docs/nbr_state/dark_results_v1.json`'s development RMSEs with difference **0.0** in all four months. The intermediate nbr_state
  RMSEs differ by 0.0, the final per-airport slopes by at most 8.3e-17, and the pooled and month-blocked dark slopes by 0.0. The
  columns-off path with both layers equals the control values with max |diff| 0.0.
- **Ranking.** The columns-off per-seed predictions equal `runs/stack_weight/rank/preds.parquet` (8083ffcf...) exactly. Steps 6-10 give
  v69 (192c4a30...), and step 11 gives **v70 `f1d17747...` byte for byte**. Both layers on that file, with slopes from the columns-off
  development values, give the nbr_state file 7a0b1431... and **v71 `2dfb3b9f...` byte for byte**. The exclusions match v71's:
  nbr_state excludes EHAM, and nbr_dark excludes no airport.

## Development reading against the control (v71's development values)

Paired take-off-day 95% intervals are in %, from `tools/run_intruder_bound.interval` with 2,000 replicates and seed 20261004. Every row
of every complete month is scored.

### sn65

| month | control RMSE | sn65 | change | interval | rows changed | squared change |
|---|---:|---:|---:|---|---:|---:|
| January 2025 | 319.9525 | 319.9425 | −0.0031% | [−0.0194, +0.0164] | 150,412 | 4.54e6 |
| July 2025 | 250.5357 | 250.6268 | +0.0364% | [+0.0109, +0.0651] | 170,213 | 1.39e7 |
| November 2025 | 181.5040 | 181.4770 | −0.0149% | [−0.0416, +0.0074] | 161,238 | 4.10e6 |
| December 2025 | 162.6274 | 162.6368 | +0.0058% | [−0.0285, +0.0361] | 157,418 | 3.65e6 |

The equal-footing mean (January, July, December) is **+0.0130%**. Clause 1 fails (July, and December at +0.0058%), clause 2 fails, and
clause 3 fails. By airport-month, the changes run from −0.138% (EGLL November) to +0.823% (EDDM December), and EDDF worsens in
January (+0.32%) and July (+0.54%).

### sn65_layers

| month | control RMSE | sn65_layers | change | interval | rows changed | squared change |
|---|---:|---:|---:|---|---:|---:|
| January 2025 | 319.9525 | 319.9102 | −0.0132% | [−0.0425, −0.0045] | 150,408 | 4.13e6 |
| July 2025 | 250.5357 | 250.5380 | +0.0009% | [−0.0178, +0.0183] | 170,184 | 5.51e6 |
| November 2025 | 181.5040 | 181.4509 | −0.0292% | [−0.0591, −0.0047] | 161,237 | 2.94e6 |
| December 2025 | 162.6274 | 162.5714 | −0.0344% | [−0.0635, −0.0076] | 157,415 | 2.95e6 |

The equal-footing mean is **−0.0156%**. Clause 1 holds, while clause 2 (bar −0.02%) and clause 3 (July's interval) fail. By
airport-month, the changes run from −0.232% (EGLL December) to +0.487% (EDDM December).

Inside the stacker, the neighbour columns absorb part of the layers' signal. Refitted on sn65's values, the nbr_state slopes fall at
every airport, for example EGLL 0.256 → 0.142 and LFPG 0.213 → 0.116, and the pooled dark slope falls from 0.117 to 0.074. The layers
on top turn sn65's January, November and December readings into small gains and bring July back to +0.0009%. sn65_layers's
equal-footing mean against v71 (−0.0156%) is the same as the kernel-replacement near miss in the quick screens, again below the bar.

## Clause 4 (stacker transfer guard)

This is the RMS of the ranking ten-seed bag prediction (guarded, fixed rows at 0) over the RMS of the out-of-month guarded development
prediction:

- v3 bag: EDDF 1.37, EDDM 1.01, EGLL 0.82, EHAM 1.23, LEBL 1.13, LEMD 1.12, LFPG 1.09, LSZH 1.22, LTFM 1.12.
- v2 bag at LIRF: 1.11.
- OPDI bag on the LSZH rows where it is used: 1.04.

All ratios lie inside [0.5, 2.0], so clause 4 holds. It applies to both candidates because they share the bags.

## The layers' transfer guards for sn65_layers (for the record)

These were computed label-free on sn65's ranking chain: steps 6-11 on the columns-on bags in `runs/stack_nbr/rank/on_chain/`. That is
an intermediate, not a candidate file. The guard compares the RMS of the ranking change with the RMS of the out-of-month development
change, per airport:

- **nbr_state w75:** EDDF 0.80, EDDM 1.26, EGLL 0.70, EHAM 2.11, LEBL 0.95, LEMD 0.50 (0.497), LFPG 0.89, LIRF 0.88, LSZH 1.37.
  **EHAM and LEMD would be excluded**; v71 excluded only EHAM.
- **nbr_dark:** EDDF 0.81, EDDM 1.01, EGLL 1.01, EHAM 1.27, LEBL 0.98, LEMD 1.05, LFPG 1.26, LIRF 0.75, LSZH 1.01. No airport would be
  excluded.
- On that intermediate, the layers change 146,389 rows: 29,420 by nbr_state and 116,969 by nbr_dark, with a squared change of 5.57e6.
  The intermediate's SHA-256 is 394af390...

## Input checks (declaration; not material)

- **Development matched flag.** The stacker development table's `matched` equals the v70 development proxy's cls ending in '_m' on
  672,058 of 672,428 rows (99.945%). All 370 others are LIRF population rows: True in the stacker table, LIRF_pop in the proxy, so none
  lies outside the fixed rows. This holds for both stacker development tables (`run_stack_weight.dev_inputs` and `rank_inputs`). As
  declared, the columns use the stacker table's flag and the layers use the proxy's flag.
- **Ranking matched flag.** `rk['matched']` equals the recipe scope in {matched_no_lirf, lirf_nonpop_matched} on 344,841 of 344,841
  rows (100%).
- **v9.** The development stacker table's 'v9' equals `runs/lomo_dev/rows.parquet`'s 'v9' on every row (100%). For information only:
  the ranking stacker table's 'v9' equals the recipe's on 337,333 of 344,841 rows. The others are the LFPG EOBT-repaired rows, where
  the stacker table averages the repaired run's v9, as v66's chain does.
- **Column coverage.** nx is defined on 475,342 development rows and 259,791 ranking rows; nxk on 498,260 and 271,516. The mean nn is
  10.2 on development and 10.9 on ranking. The learners have 49, 68 and 82 columns.

## Compute and records

- The run used arcw1 only, with workspace `/mnt/scratch/prcn` on its temp disk (nvme2n1, formatted ext4):
  - It is a copy of `~/prc-v67` plus `git archive` of b92d3a5 (`runs/stack_nbr/sn_code.tgz`, 2be92a35...).
  - All 735 cached runs/data files were re-hashed: 723 equal the workstation and 0 differ. The other 12 were VM-only ctx_own2
    markers and logs, which were deleted.
  - Nineteen missing inputs were fetched lazily through a reverse tunnel, with hashes checked (`runs/stack_nbr/vm_fetch.jsonl`).
  - Software: Python 3.14.2, LightGBM 4.7.0, numpy 2.4.2, pandas 3.0.0, pyarrow 23.0.1.
- A preflight before the driver checked the tool's layer functions on v70's development proxy and on a copy of the uploaded v70. It
  reproduced the development RMSEs and v71 byte for byte. No fit was run and no candidate was read.
- Wall time was 51 minutes: 10 fit jobs of 30 LightGBM fits each, at most six jobs at a time.
- Results were mirrored as `runs/stack_nbr/mirror/out.tar` (c082235a..., hash equal) and extracted. arcw1 was deallocated at 23:38 UTC,
  the file server was stopped, and the tunnel closed with the VM. All eight VMs are deallocated.
- The code hashes in the results file are those of the VM's LF checkout (git blobs).
