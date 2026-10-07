# Runway take-off-sequence columns in the guarded stacker (`configs/stack_queue_v1.json`): NOT accepted (7 October 2026)

The round was declared at 15:38:16 UTC (b900f23 + 1938ce4), before any reading. Its governing run is
[`tools/run_stack_queue.py`](../../tools/run_stack_queue.py) (125064d). That tool imports `tools/run_stack_weight.py`'s functions
unchanged and adds one switch, `--queue off|on`, which appends the seven queue columns to each learner's feature matrix. The run
used arcw1 only, from 15:57 to 16:41 UTC, with driver [`tools/stack_queue_remote.py`](../../tools/stack_queue_remote.py) (e1c65cf).
Results are in [`results_v1.json`](results_v1.json); the run log is in [`HANDOFF.md`](HANDOFF.md). No official score or leaderboard
was used. Nothing was written under `runs/submissions` and nothing was uploaded.

## Decision: not accepted; v70 stays

Clause 1 holds, but clauses 2 and 3 fail: the equal-footing mean is −0.0036% against a −0.02% bar, and no interval is below zero.
Clause 4 holds. Because the candidate was not accepted, no ranking candidate file was written; `runs/stack_queue/rank/dry_v71/`
does not exist.

## Identity checks (run first; both hold)

- **Development.** The queue-off path (w 0.65, ten seeds, then steps 7-10, then step 11) equals the control
  `runs/oooi_lhr/dev_v70_proxy.parquet` (51aaa6a6...) on every row, with max |diff| **0.0**. Its step-10 value also equals v69's
  development proxy (c088e409...) exactly. January's 30 queue-off per-seed predictions equal `runs/stack_weight/dev/preds_2025-01.parquet`
  exactly.
- **Ranking.** The queue-off per-seed predictions equal `runs/stack_weight/rank/preds.parquet` (8083ffcf...) exactly. Steps 6-10 give
  v69 (192c4a30...), and step 11 gives **v70 `f1d17747...` byte for byte**.

## Development reading (q65_bag10 against v70's development proxy)

Paired take-off-day 95% intervals are in %, from `tools/run_intruder_bound.interval` with 2,000 replicates and seed 20261004.

| month | control RMSE | q65_bag10 | change | interval | rows changed | squared change |
|---|---:|---:|---:|---|---:|---:|
| January 2025 | 320.0147 | 320.0145 | −0.0001% | [−0.0126, +0.0125] | 147,919 | 2.15e6 |
| July 2025 | 250.7120 | 250.6852 | −0.0107% | [−0.0227, +0.0008] | 158,697 | 3.13e6 |
| November 2025 | 181.6080 | 181.5887 | −0.0106% | [−0.0316, +0.0096] | 159,122 | 1.36e6 |
| December 2025 | 162.7895 | 162.7893 | −0.0002% | [−0.0320, +0.0321] | 145,010 | 1.93e6 |

- The equal-footing mean (January, July, December) is **−0.0036%**.
- Clause 1 (no month worse by more than 0.005%) holds. Clause 2 (mean ≤ −0.02%) fails. Clause 3 (every equal-footing interval below
  zero) fails.
- By airport-month, the changes run from −0.098% (LFPG July) to +0.112% (LSZH July), with no consistent sign by airport.
  `airport_month_change_pct` in the results file has the full breakdown.
- The standalone exploration read −0.0181% on the same control. Inside the stacker, the seven columns add almost nothing, which suggests that the stacker's
  existing inputs already carry most of that signal.

## Clause 4 (transfer guard; for the record)

This is the RMS of the ranking ten-seed bag prediction (guarded, fixed rows at 0) over the RMS of the out-of-month guarded development
prediction:

- v3 bag: EDDF 1.34, EDDM 0.99, EGLL 0.84, EHAM 1.23, LEBL 1.12, LEMD 1.05, LFPG 1.08, LSZH 1.23, LTFM 1.11.
- v2 bag at LIRF: 1.13.
- OPDI bag on the LSZH rows where it is used: 1.05.

All ratios lie inside [0.5, 2.0], so clause 4 holds.

## Details

- **Queue columns.** These are `queue_columns`, which reproduces the exploration's `build` exactly on July 2025's file. Development
  rows take them from their own month's training file; every development row is a departure of that file. Ranking rows take them from
  `data/raw/ranking.parquet` treated as one file, as declared. In that file, 430 ranking rows have a gap that crosses January to July
  2026 (the first or last few take-offs on each runway). The learners have 53, 72 and 86 columns.
- **Unchanged from the declaration.** Guards are computed on the unchanged X46, own-point and OPDI columns. LightGBM settings,
  month-blocking and the seeds 20261003, 11-19 are as declared. Step 11 uses `run_oooi_pin`'s functions (offsets from labels and
  take-off − OUT only), and the joined tables were hash-checked against `docs/oooi_pin/results_v1.json`.
- **Compute.** The run used arcw1 only, with workspace `/mnt/scratch/prcq` on its temp disk:
  - It is a copy of `~/prc-v67` plus `git archive` of 125064d.
  - All 735 cached runs/data files were re-hashed: 723 equal the workstation and 0 differ. The other 12 were VM-only markers, which
    were deleted.
  - Fifteen missing inputs were fetched lazily through a reverse tunnel, with hashes checked (`runs/stack_queue/vm_fetch.jsonl`).
  - Software: Python 3.14.2, LightGBM 4.7.0, numpy 2.4.2, pandas 3.0.0, pyarrow 23.0.1.
  - Wall time was 44 minutes: 16 fit jobs of 30 LightGBM fits each, at most six jobs at a time.
  - Results were mirrored as `runs/stack_queue/mirror/out.tar` (`0f5a8f4d...`, hash equal) and extracted. arcw1 was deallocated at
    16:45 UTC, and the file server and tunnel were stopped.
  - arcw0, arcw2 and arcw3 were started at 15:41 only to inspect their disks. None held the chain workspace, and all three were
    deallocated at 15:45. All eight VMs are deallocated.
- **Code hashes.** The code hashes in the results file are those of the VM's LF checkouts (git blobs), not the CRLF copies on this
  workstation.
