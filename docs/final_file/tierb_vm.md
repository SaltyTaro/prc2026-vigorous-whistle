# Tier B on arcw6: execution record (10 October 2026)

The user asked at about 17:30 UTC to keep the remaining Tier B work off the workstation ("use the vm's only as i have less memory").
Lane M (model part) and lane C (chain) had not computed anything yet; their local runners were stopped and both lanes ran on **arcw6**.
The workstation only copied files, compared hashes and started remote jobs. Scripts: [`tierb_vm/`](tierb_vm/) (working copies under
`runs/febjun_tierb/vm/`, git-ignored).

## Setup

- arcw6 restarted 17:40 UTC (its own tree fits had been mirrored, hash-checked (21 files) and the VM deallocated at 17:30). The repository
  copy was moved to the VM's local NVMe (`mkfs.ext4 /dev/nvme2n1`, `/mnt/scratch/repo`) and **bind-mounted over the repository path**, so
  `Path.resolve().relative_to(ROOT)` holds everywhere. The NVMe is wiped at deallocation: every output was copied back first.
- Library versions equal the
  workstation's except scikit-learn (1.7.2 on the VM, 1.9.0 here); Python 3.12 there, 3.14 here.
- Inputs: 11 uploads (`upload.sh`: `tar` over `ssh`, then SHA-256 of every file on both sides; all `UPLOAD_OK`), about 22 GB. The first
  15 GB came from a seed list (the earlier final-chain staging lists, the 2025 surface days, the Tier B directories and the paths both
  Tier B tools quote); the rest was found by running every step under `strace` (`vmrun.sh`), which lists every path a step looked for and
  did not find -- including optional probes a step would skip silently -- and by the paths quoted in the tools and configurations of
  each lane.

## Validation before the real run

- **Lane M**, January / July reproductions (`validate_M.sh`): `trees`, `nets`, `rows`, `chain` checks identical to this workstation's
  `runs/febjun_tierb/model/checks/*.json`; `rematch` identical except one 2.3e-13 m float in a taxi-path difference.
- **Lane C**, the code-path stand-in rerun (`validate_C.sh`): byte-identical to the workstation's stand-in (`logs/standin2.log`): model
  part `21f2d65c`, early chain final `8ca9dbc6` with every intermediate step equal, and all eight late outputs.
- No step of either validation looked for a file that exists here but was missing on arcw6.

## Real run (`real_vm.sh`, then `real_vm_resumeC.sh`)

| UTC | event |
|---|---|
| 19:32 | started; own-points member tables built (8 tables, every row finite); waiting for the trees |
| 22:21 | last tree fit copied back (`relay_trees.sh`): 26 tree files uploaded, all 116 hash-equal on both machines, `TREES_READY` |
| 22:23 | lane M: tree record check 58/58, `trees`, `rows`, `chain`, `read` (about one minute in all) |
| 22:23 | lane C: `root` done; `c1` failed on the own-points weight lookup; fixed (03c476e), resumed 22:24 |
| 22:45 | lane C `early` (steps 2-10) done |
| 22:51 | `late` and `read` done: `REAL_DONE` |
| 23:00 | 269 output files copied back, all hash-equal (`mirror_back2.sh`) |

The `read` step looked for five files absent on arcw6 (a December bundle, two selection records); they were uploaded and the readings
recomputed: identical. Results: [`tierb_model.md`](tierb_model.md), [`tierb_chain.md`](tierb_chain.md), decision in
[`febjun_check.md`](febjun_check.md).

## Tree VMs

After the last tree fit, `tree_dealloc.sh` compared every tree output (run record, predictions, model) on its VM with the local mirror:
arcw0 30, arcw1 27, arcw3 27, arcw4 24, arcw5 18, arcw7 27 files, all equal; each VM deallocated (22:24-22:31 UTC). arcw0 / arcw1 held
their outputs on NVMe bind mounts, hence the check before deallocating.

## Pitfalls met

- File lists written by Python on Windows had CRLF endings; `tar -T` then finds nothing. Lists are written with `newline="\n"`.
- `pkill -f PATTERN` over `ssh` matches the remote shell running it (whose command line contains the pattern) and kills the session; the
  same self-match made the first VM-shutdown pass report every VM busy. Processes are now matched by exact PID or by an anchored
  `ps -eo args | grep -c "^..."`.
- A failed `early` leaves `steps/rematch/base/` behind, and the tool refuses to overwrite it: `restart_C.sh` cleans the work directory
  except step 0's key map and matches.
- Every long job ran as a detached process (`nohup` /
  `setsid`), so only waiters were lost, never work.
