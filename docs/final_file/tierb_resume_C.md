# Tier B lane C: resume note (10 October 2026, 15:06 UTC)

## Done

- **Declaration committed before any reading:** `configs/febjun_tierb_readings_chain_v1.json` (1d678fb). Record `docs/final_file/tierb_chain.md`
  (dce3a76), checks in `docs/final_file/tierb_chain_checks/`.
- **Reproductions (before the declaration):** own_all routing on the 16 stored January 2025 fits = the stored development member tables
  exactly; the OPDI builder replica = `runs/opdi_stack/features_rank.parquet` exactly; the month-blocked steps 7-10 on F with rows12 `stack`
  reproduce the stored twelve-month readings (intruder bound -1.0858 / -0.2102% vs -1.0859 / -0.2103%; identity extension, own-motion v1 and
  v2 equal to four decimals). The whole chain ran end to end on a stand-in base twice (no reading).
- **Inputs already built (label-free):** F `runs/febjun_tierb/chain/input/febjun2025.parquet` (`8a0de5bd...`, with January 2025's arrivals
  as context), template `febjun2025_template.parquet` (`7d23346e...`); step 0 (key map + F's no-NM matches) in
  `runs/febjun_tierb/chain/work/steps/rematch/`; OPDI events for February / June 2025 in `data/external/opdi/stack_v1/` (sources.json updated).
- **Network inputs:** all 80 Tier B fits are local and hash-checked, including dl32own / m25dl32own s601-604.

## Inputs still needed

| from | files |
|---|---|
| lane M | `runs/febjun_tierb/model/chain_2025-02.parquet`, `chain_2025-06.parquet` (columns of `tools/febjun_tierb_model.py chain`; `v55m` = the model part), and `docs/final_file/tierb_model_results.json` (M's `read` done; the guard also waits for no `febjun_tierb_model.py` process) |
| lane G | `runs/febjun_nets/nets/{dl32own,m25dl32own}_s60{1..4}/lomo_2025-0{2,6}/{run.json,predictions.parquet}` (present) |

## The guarded run (running detached)

`runs/febjun_tierb/chain/real_guarded.sh` (bash, pid 38591 at 15:05 UTC, log `runs/febjun_tierb/chain/logs/real.log`). It waits for the
inputs above, then runs each step only when free physical memory (`/proc/meminfo` MemFree) is at least its estimated peak + 1 GB, skipping a
step whose output exists:

| step | command (`.venv/Scripts/python.exe -X utf8 -B tools/febjun_tierb_chain.py ...`) | waits for free RAM | output marker | stand-in wall time |
|---|---|---:|---|---:|
| own-member | `own-member` | 2.5 GB | `runs/febjun_tierb/chain/own/member_build.json` | < 1 min |
| root | `root` | 3.0 GB | `runs/febjun_tierb/chain/root/input.json` | < 1 min |
| c1 | `c1` | 3.0 GB | `runs/febjun_tierb/chain/work/steps/model_part/step1.json` | ~2 min |
| early (steps 2-10) | `early` | 4.5 GB | `runs/febjun_tierb/chain/work/chain_tierb.json` | 23-29 min |
| late (steps 11-18) | `late` | 3.5 GB | `km` recorded in `work/late/late_chain.json` | ~3 min |
| readings | `read` | 2.5 GB | `docs/final_file/tierb_chain_results.json` | ~1 min |

It writes `REAL_DONE` at the end, or a line containing `Traceback-or-failure` (and the Python traceback) and stops. The peak estimates come
from the stand-in runs (free memory stayed at or above 2.7 GB with about 3 GB free at the start; the v48 subprocess used 0.8 GB, the parent
up to about 2 GB in the stacker and own-motion steps); they were not measured exactly, hence the 1 GB margin. Everything runs on the
workstation: the chain reads 2025 development tables only this machine holds (a VM would need the staging of `lane_V.md`).

After `REAL_DONE`: fill `docs/final_file/tierb_chain.md` (results section) from `tierb_chain_results.json`, commit both with an explicit
pathspec, report the readings and any switch-off. `read` refuses unless the declaration is committed and unchanged.

## Pitfalls

- **Never edit `real_guarded.sh` while it runs** (bash reads it incrementally). To change it: kill the bash pid, edit, relaunch with
  `nohup bash runs/febjun_tierb/chain/real_guarded.sh >> runs/febjun_tierb/chain/logs/real.log 2>&1 &`.
- **A failed step leaves partial outputs** and the tools never overwrite. Re-running `early` needs `runs/febjun_tierb/chain/work/` emptied
  except `steps/rematch/` (step 0, label-free, kept); re-running `late` needs `work/late/` removed; `c1` needs `work/steps/model_part/` removed;
  `root` needs `runs/febjun_tierb/chain/root/` removed.
- **Free RAM:** the user's own applications leave 0.7-1.3 GB free at times; the guard then waits (it logs every 10 minutes). Do not run it
  while lane M runs (the guard checks only once, before the first step).
- `work_standin/`, `root_standin/` and `work_repro12/` are code-path / reproduction runs; never read them as Tier B readings.
- Governing population: lane M's rows read (clock-present departures with an out-of-sample v9); placeholder rows pass through the chain
  and are reported, never governing (declaration `readings.rows_read`).
