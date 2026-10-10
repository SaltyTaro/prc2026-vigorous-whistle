# Final file candidate 2 (10-11 October 2026): built, checked, NOT uploaded

Candidate 1 (`docs/final_file/final_candidate.md`, `30e116a9...`) with one change. The third pass (`stack_third_v1` `t3_ro_3`) uses
its six-month learner: fitted on the development table plus both Tier B months' rows read. Configs/sixmonth_refit_v1.json accepted
that refit (`docs/sixmonth/results_v1.json`, 0a9ab7a: February -0.0039% [-0.0070, -0.0013], June -0.0073% [-0.0151, -0.0019]). The
stacker (C6) and the second pass (C14) keep the shipped learners. Record: `final_candidate2.json`.

| file | SHA-256 | rows |
|---|---|---:|
| **candidate 2** `runs/final_blind/final_candidate2/chain_out_km.parquet` (work: `runs/final_blind/late_c2/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet`) | `f11ca28f7503a98cad9211facafe30d7f4185a589aefdb655372896d59e2669e` | 670,790 |
| candidate 1 (unchanged, not modified) | `30e116a9a0ae4bdf5f2150938272b571bb20b980ac18a7e3bb3f0c48374f381b` | 670,790 |

## How it was built (arcw6, 23:36-23:47 UTC on 10 October)

1. **Staging.** Candidate 1's late work directory `late_9002` was checked locally against `runs/final_blind/chain_9002.sha`: 55 of 55
   files equal. It was uploaded to arcw6 with the early-chain output and the runner recipe: 58 files, SHA-256 equal on both sides.
   arcw6 already held every other input (`runs/stack_third`, `runs/season_pass*`, `runs/key_memory`, `data/raw/final_*`, the Tier B
   tables).
2. **Work copies.** `runs/sixmonth/vm/make_c2_work.py` copies `late_9002` to a new directory and removes the third pass, the season
   passes and key memory, with their `late_chain.json` outputs. It re-points the remaining paths to the copy. `late_9002` itself is
   never modified.
3. **Control:** `runs/final_blind/late_c2_control`, `st_build`, `fit --stage st` (shipped learner), `st_score`, `sp_*`, `sp2_*`, `km`.
   **It reproduces candidate 1 byte for byte: `30e116a9...`.** The st / sp / sp2 / km outputs and every parquet are equal to
   `late_9002`'s; the only files that differ are 8 JSON records (timestamps, work paths).
4. **Candidate 2:** `runs/final_blind/late_c2`, the same steps with `tools/run_sixmonth.py fit-final --kind st --target
   runs/final_blind/late_c2/stack_third` in place of `fit --stage st`.
   - The fit offered 323,246 Tier B rows.
   - `preds_rank` `a4659e15...`; the development predictions are the stored ones (`68aa91d3...`).
   - Every layer selects what it selected for candidate 1: sp_hicov7, t3_ro_3, w25_e3, w25b_5, km_9, no selection flag.
   - The third pass's transfer ratios: EDDF 1.54 / EHAM 1.05 / LSZH 1.04, against 1.61 / 1.05 / 1.03 for candidate 1.

Every step ran under strace. The only paths it looked for and did not find were the outputs it then wrote.

## Checks (label-free; `runs/final_blind/final_candidate2/checks_c2.json`, `month_means_c2.json`)

- **Uploader rehearsal** `tools/prc_submit.py upload-final --check-only` returns before any `put_object`:
  - 670,790 rows, exact `final_submitting.parquet` order, int32;
  - minimum -12, maximum 120,636, 80 negative values (candidate 1: 78);
  - organiser ETags unchanged, no final object in the team bucket.

  Its log overwrote `runs/submissions/vigorous-whistle_final.check.json`. That file was restored byte for byte to candidate 1's record;
  candidate 2's copy is `runs/final_blind/final_candidate2/vigorous-whistle_final.check_c2.json`.
- **Against candidate 1:** 142,400 rows differ, **all at EDDF / EHAM / LSZH** (all four months). Max |difference| 60 s, squared change
  2.88e6, RMS 4.5 s on the changed rows, mean +0.16 s. Through the stages:
  - the third pass changes 143,284 rows, all at EDDF / EHAM / LSZH;
  - the January-only layers move no row elsewhere: season_pass's ranking predictions are unchanged, and sp2 / km end at 142,400 rows;
  - **0 rows** change outside EDDF / EHAM / LSZH, in any month.
- No NaN, minimum -12.
- **Monthly plausibility** (`tools/final_month_means.py`): the flags are the same four as candidate 1's (EDDF / EDDM / LSZH January,
  LIRF July). The EDDF gaps rise by 0.9-3.2 s: February 52.6 -> 55.8 s, against a 2025 range of 0.3-52.9 s and a 30 s margin, so not
  flagged. EHAM and LSZH move by at most 0.3 s.

| airport | Jan rows changed | Feb | Jun | Jul | squared change (all months) |
|---|---:|---:|---:|---:|---:|
| EDDF | 13,134 | 13,384 | 16,538 | 17,494 | 1.92e6 |
| EHAM | 10,600 | 11,073 | 14,165 | 14,588 | 4.56e5 |
| LSZH | 6,693 | 6,135 | 8,995 | 9,601 | 5.06e5 |

## Mirror and state

- **Mirror:** 87 files mirrored from arcw6, hash-equal on both sides (`runs/sixmonth/vm/c2_mirror.{local,remote}.sha`):
  - `runs/final_blind/final_candidate2/*`;
  - `late_c2`'s new subdirectories and `late_chain.json`;
  - `late_c2_control`'s records;
  - the logs (`runs/sixmonth/logs/c2*`).

  `late_c2`'s unchanged subdirectories are byte copies of `late_9002` (`late_c2.sha`).
- **Not uploaded.**

      .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate2/chain_out_km.parquet --expected-sha256 f11ca28f7503a98cad9211facafe30d7f4185a589aefdb655372896d59e2669e
