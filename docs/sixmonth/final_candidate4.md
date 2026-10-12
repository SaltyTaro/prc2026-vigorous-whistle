# Final file candidate 4 (12 October 2026): built, checked, NOT uploaded

> **January / July reading added the same day ([`devmonths.md`](devmonths.md), `configs/sixmonth_devmonths_v1.json`):** with the Tier B rows
> added, C6 + C14 lower squared error on a held development month for January 2025 (-1.27e7 s²) and July 2025 (-1.60e7). The pre-set rule
> keeps candidate 4's values in every month, so no month splice is made. December 2025 reads +9.4e5 (C6's LIRF part), November -4.3e6.

Final candidate 3's chain (`docs/lirf_cap/README.md`, `b673791d...`), with the **six-month learners of the C6 guarded stacker and the C14
second pass** as well as C15's. Declared as a **disclosed override** in [`configs/sixmonth_override_v1.json`](../../configs/sixmonth_override_v1.json)
(def3367) before any candidate-4 value: `configs/sixmonth_refit_v1.json`'s rule did **not** accept C6 or C14, because February 2025's interval
includes zero (C6 February -0.0325% [-0.0735, +0.0041], June -0.1016% [-0.1750, -0.0380]; C14 February -0.0062% [-0.0187, +0.0046], June -0.0144%
[-0.0240, -0.0050]). **Candidate 3 stays the fallback; which file is uploaded is the user's decision.**

| file | SHA-256 | rows |
|---|---|---:|
| **candidate 4** `runs/final_blind/final_candidate4/chain_out_cap.parquet` | `8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42` | 670,790 |
| its late-chain output before the cap `runs/final_blind/late_c4/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet` | `ab25a196e74a325618eda095918b653aadedef3479267a83e06e46b2c56ab849` | 670,790 |
| its early-chain output `runs/final_blind/early_c4/vigorous-whistle_v9002.parquet` | `d5834fcbbd9293c06a4e36ccefe8b8d35f5cde11056e07eebee3e2c9dce6b068` | 670,790 |
| six-month step-6 preds `runs/sixmonth/c4/stack_preds_6m.parquet` (323,246 Tier B rows added) | `48a184a648b759d85a5c3648415712d058ac78ebf3284ce65c01086dc23e6a83` | |
| candidate 3 (unchanged, fallback) | `b673791dfdaf8e909175001387183946236075280c9b3457822b52157c9cebc8` | 670,790 |

## How it was built (arcw0, 08:16-09:30 UTC on 12 October)

- **VM.** arcw0 (its OS-disk repository already held most chain inputs). The repository was overlaid on the local NVMe
  (`mount -t overlay`, lower = the OS-disk repository, upper = `/mnt/scratch/up`), so every write went to the NVMe and every path stays a real
  path under the repository. Code: `git -c core.autocrlf=false archive HEAD` + `.git` (HEAD 25b13f2, containing def3367 / dba57dc).
- **Inputs.** `make_want.py` lists 4,793 files (19.8 GB): the earlier Tier B / six-month staging lists without the model-part folders, plus the
  final-file runner tables, early_9002, the 2025 and January / February / May / June / July 2026 surface days, the data-link messages, the Tier B
  tables and the P2 learners. 2,540 were already on arcw0 with equal SHA-256 (0 differed); 2,138 (10.4 GB) were uploaded in four `tar` streams
  (`upload.sh`), every file hash-equal on both sides.
- **Driver.** `tools/final_early_resume6.py` (dba57dc) copies early_9002's steps 0-5 and caches byte for byte into a fresh directory and runs the
  chain's own `configure()`, `step6()` (with `--stack-preds`) and `steps_7_to_10()`. `tools/run_sixmonth.py fit-final --override
  configs/sixmonth_override_v1.json` (def3367) uses the six-month learner for a layer the readings did not accept only if that committed file names
  it; the default path is unchanged. Script: [`c4/run_c4.sh`](c4/run_c4.sh); every step under `c4/vmrun.sh` (strace).

| step | UTC | result |
|---|---|---|
| **control, early** (`early_ctl`): resume at step 6 with candidate 3's own step-6 preds | 08:33-08:38 | **`4c393769...` = early_9002's output, byte for byte** |
| C6 six-month fit (`stack6m`, 30 bags, 323,246 Tier B rows) | 08:35-09:01 | preds `48a184a6...`; the tool printed the OVERRIDE notice |
| **control, late** (`late_ctl`): full late chain 11-18 on the control's output, shipped C14 fit, six-month C15 | 08:38-09:18 | **`f11ca28f...` = candidate 2, byte for byte** |
| candidate early (`early_c4`): resume at step 6 with the six-month preds | 09:02-09:10 | `d5834fcb...` |
| candidate late (`late_c4`): full late chain, six-month C14 (`--override`) and C15, then the LIRF cap | 09:18-09:29 | `ab25a196...` -> cap -> **`8fc8828d...`** |

The control's C14 fit was paused (SIGSTOP / SIGCONT) while the C6 fit ran: two 16-thread LightGBM jobs on 16 cores slow each other badly
(seen before on arcw2). Pausing does not change a fit; the control's byte-for-byte result confirms it. Every late layer selects what it selected
for candidates 1-3: OUT pin w50, nbr_state w75, nbr_dark, sp_hicov7, t3_ro_3, w25_e3, w25b_5, km_9, no selection flag.

## Checks (label-free; [`c4/checks_c4.json`](c4/checks_c4.json))

- **Uploader rehearsal** `tools/prc_submit.py upload-final --check-only`: `checked_not_uploaded`, 670,790 rows, exact
  `final_submitting.parquet` order, int32; no final object in the team bucket; organiser ETags unchanged
  ([`c4/vigorous-whistle_final.check_c4.json`](c4/vigorous-whistle_final.check_c4.json)). The shared
  `runs/submissions/vigorous-whistle_final.check.json` was restored byte for byte afterwards (`e7633168...`).
- Range -12 to 111,666, 71 negative values, no NaN.
- **Against candidate 3:** 552,619 rows differ (the step-6 stacker alone moves 487,921 rows of the early output, squared change 2.66e7), squared
  change **4.41e7**, max |difference| 533 s, mean -0.40 s. By month: January 1.28e7, February 6.53e6, June 1.01e7, July 1.47e7. By airport:
  LIRF 1.30e7 (non-population rows only), EDDF 7.5e6, LSZH 5.1e6, LTFM 3.9e6, EGLL 3.6e6, EHAM 3.4e6, LFPG 2.9e6, LEMD 2.4e6, LEBL 1.3e6,
  EDDM 0.8e6. Rows changed by airport and month are in `checks_c4.json`.

## Mirror and state

202 files mirrored from arcw0, hash-equal on both sides ([`c4/mirror.local.sha`](c4/mirror.local.sha)): `runs/final_blind/final_candidate4/`,
`late_c4/`, early_c4's new steps and record, both controls' records and outputs, the six-month preds and the logs. The overlay and NVMe were
unmounted and **arcw0 deallocated**; all eight VMs are deallocated. arcw0's OS-disk repository was not modified.

## Upload (only at the user's word; one upload, never overwritten)

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate4/chain_out_cap.parquet --expected-sha256 8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42

Fallback (candidate 3, without the overridden learners):

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate3/chain_out_cap.parquet --expected-sha256 b673791dfdaf8e909175001387183946236075280c9b3457822b52157c9cebc8
