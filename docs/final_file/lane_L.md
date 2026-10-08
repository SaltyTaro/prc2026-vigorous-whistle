# Lane L: chain steps 11-18 (v70 -> v76's layers) on any movement file (8 October 2026)

Part of the final-round plan ([`final_round_plan.md`](final_round_plan.md)). Steps 11-18 are the data-link OUT pin (v70), neighbour state
and its dark-row extension (v71), the second and third guarded residual passes (v72, v73), the W25 season pass (v74), season_pass_v2 (the
v75 dry run) and the W25 flight-number memory (v76). Until now each read a shipped `runs/submissions/vN` file by hash and the January / July
2026 matches.

## Result

- **Rehearsal on `data/raw/ranking.parquet` from the shipped v69: every step reproduces byte for byte.** v70 `f1d17747...`, the
  nbr_state dry run `7a0b1431...`, v71 `2dfb3b9f...`, v72 `4280c399...`, v73 `649e9972...`, v74 `85e57d05...`, the v75 dry run
  `a227c51c...` and **v76 `a2ff395d...`**. So are every intermediate ranking table (stack_second `rank_table`, `raw_rank`, `rank_t3`,
  `rank_sp`, `rank_sp2`) and every refitted ranking bag (`preds_rank` of the four passes, refitted on Azure arcw1 from the stored
  development tables). Report: `runs/final_late/rehearsal_ranking/check.json` (SHA-256 `ade889c2...`, `all_equal: true`).
- **New-file smoke test (July 2026 only, a movement file cut from the ranking file):** the whole chain runs. Steps 11-15 apply; the three
  W25 steps pass their input through (the file has no 2026-01 rows). Output in template order, int32, minimum -12. Against the full-file
  run's July rows, 912 rows differ (max 37 s, RMS 0.61 s), all at EHAM. That is the declared transfer guard evaluated on the file being
  predicted: on the ranking file nbr_state's EHAM ratio was 2.07 (January's disruption), so EHAM kept the base value; on July alone it is
  0.89, so the layer applies there. LEMD sits at 0.51, near the guard's 0.5 edge.

## How it works

`tools/final_late_chain.py` runs each layer's **own governing-run code**, with the tool's module constants pointed at a work directory.
- **What it points them at:** the movement file, the base file (the previous step's output), the runner's recipe detail, the ADS-B
  folders and the new file's stacker inputs.
- **Development side:** unchanged. Stored development tables and development predictions are copied after a hash check against the
  governing receipts (`tables.json`, `vm_preds.sha256`). Nothing on that side is rebuilt from other data.
- **Ranking bags:** refitted deterministically on Azure. This is the same LightGBM 4.7.0 call as shipped; only the development bags are
  skipped.
- **Declared rules on a new file:** each layer's rule is applied to the new file as it was to the ranking file, transfer guards included.
  A layer that selects nothing passes its input through.
- **Selection flag:** each step's record flags a selection that differs from the shipped one (`selection_flag`). In both runs here
  there was none.
- **Work directory:** must lie under the repository, because the tools write repository-relative paths into their receipts. Nothing is
  uploaded, and nothing is written under `runs/submissions`.

```
python tools/final_late_chain.py init  --work runs/final_late/<name> --raw <movement file> --base <step-10 output> --recipe <runner>/recipe/detail.parquet
                                       [--months M,M] [--matches-dir DIR] [--features-own-dir DIR] [--messages FILE] [--stacker-rank FILE]
python tools/final_late_chain.py step  --work W --name oooi | nbr_state | nbr_dark | ss_build          (workstation, light)
python tools/final_late_chain.py fit   --work W --stage ss                                             (Azure: copy W/late_chain.json, W/stack_second/{dev_table,rank_table,preds_dev_stored}.parquet)
python tools/final_late_chain.py step  --work W --name ss_score | raw | st_build
python tools/final_late_chain.py fit   --work W --stage st                                             (Azure: W/stack_third/{dev_t3,rank_t3,preds_dev_stored}.parquet)
python tools/final_late_chain.py step  --work W --name st_score | sp_build
python tools/final_late_chain.py fit   --work W --stage sp      (Azure; no-op without 2026-01 rows)
python tools/final_late_chain.py step  --work W --name sp_score | sp2_build
python tools/final_late_chain.py fit   --work W --stage sp2     (Azure; no-op without 2026-01 rows)
python tools/final_late_chain.py step  --work W --name sp2_score | km
python tools/final_late_chain.py check --work W                 (rehearsal on ranking.parquet only)
```

The final step's output is `cfg["outputs"]["km"]["file"]` in `W/late_chain.json`.
- **Wall time:** each workstation step takes seconds to a few minutes. Each ranking-only fit takes 1-3 minutes on a D16ds_v7.
- **Fit runner:** `runs/final_late/rfitw.sh <work> <stage>` on the VM.

## Changes to the tools

All changes are optional; the defaults are the shipped values.
- **Paths lifted into module constants** (same default values):
  - `tools/oooi_pin_join.py`: `RANKING`.
  - `tools/run_nbr_state.py`: `RANKING`.
  - `tools/run_nbr_dark.py`: `RANK_MONTHS`, `MATCHES`, `DARK_RESULTS`.
  - `tools/run_stack_second.py`: `RANKING`, `RANK_MONTHS`, `MATCHES`.
  - `tools/run_stack_third.py`: `RAW_DIR`.
  - `tools/run_key_memory.py`: `RANKING`.
  - `docs/audit_v72/raw_build.py`: its fixed `D:/` root is now found from the file, and the computation moved into `build()`. Run as a
    script it writes the same two files. The nanosecond slip in its four `m_` time columns is kept on purpose, because v73 depends on it.
- **`run_stack_second.build()` split** into the development half and `build_rank()`, both unchanged. Three new options, each defaulting
  to `None`:
  - `DEV_TABLE_FROM`: copy a stored development table byte for byte.
  - `STACKER_RANK`: read the ranking stacker inputs from a file instead of `run_stack_weight.rank_inputs()`.
  - `DEV_PREDS_FROM`: in `fit()`, copy stored development predictions and fit only the ranking bags.
- **`DEV_PREDS_FROM` added** to `run_stack_third`, `run_season_pass` and `run_season_pass2`.
- **Fits never predict on an empty set.** For the season passes this is the declared January-only rule. The shipped runs' sets are not
  empty, so their bags are unchanged.
- **Equivalence of the defaults:** shown by reading the code. The governing tools were not re-run in default mode, because that would
  rewrite the shipped `runs/`, `docs/` artefacts. The rehearsal exercises the same functions with overridden paths.

| argument | content | from |
|---|---|---|
| `--raw` | the new movement file (schema of `data/raw/ranking.parquet`, all phases) | organisers |
| `--base` | the **step-10 output** for that file: runner `--layers v55m` -> steps 2-10 with v69's step 6 (weight 0.65, ten seeds). A full submission in template order, int32. Fill pins OFF in the final file (plan decision 2) | Lane E / main |
| `--recipe` | the runner's `<root>/recipe/detail.parquet` (`MVT_ID_mvt, airport, month, scope, v9` for every departure; `scope` in `matched_no_lirf`, `lirf_nonpop_matched`, `fixed`, ...) | runner |
| `--stacker-rank` | `MVT_ID_mvt` + the 79 stacker columns of the 'po' learner for the file's departures: `X79r` of `run_stack_weight.rank_inputs()` run on the new file, named and ordered like the `s_*` columns of `runs/stack_second/dev_table.parquet` (on the ranking file: `final_late_chain.py stacker` writes it, `95a5eedc...`) | Lane E |
| `--matches-dir` | folder with `matches2_<month>.parquet` for the file's months. Columns: `gate`, `takeoff`, `segment_start`, `motion_start`, `arrival_last_ground`, `last_ground`, `first_gs`, `stop_seconds`, `first_lat`, `first_lon`, `taxi_path_m`, `taxi_stops`, `taxi_stopped_s`, `slow_start_s`, `gap_before_s`, `points`, `at_known_stand`, `first_dist_stand_m`, `anchor_dist_stand_m`. The runner writes its matches under its own root: point this at them, or copy them with this naming | runner / main |
| `--features-own-dir` | folder with `features_v19_<month>.parquet` (features_v19own's 26 columns) for those months | Lane E (`build_ownpoints_features.py`) |
| `--messages` | the new months' data-link `messages_unique.parquet` (`tools/oooi_pin_extract.py run --months ...`, then `tools/oooi_pin_pack.py`). It must not overlap the stored months. Without it the LFPG pin finds no reports for the new months and passes its input through | main |

**W25 layers on new months.** The declared rules of season_pass_v1, season_pass_v2 and key_memory_v1 act on 2026-01 rows only.
- On any other month they pass their input through, including February and March 2026, which are also winter season W25. Their
  declarations fix the target month; they do not change.
- Applying them to February or March would need a new declaration and a development reading (for example December -> February adjacency).
  That would be a design change, outside this lane.

## Not done here

- The early chain (steps 1-10 on a new file) and the stacker-input interface on a new file are Lane E's.
- No test was added to `tests/`. The rehearsal and the smoke run are the evidence.
- Work directories: `runs/final_late/rehearsal_ranking/` (1.2 GB) and `runs/final_late/smoke_july/` (0.7 GB), git-ignored. The VM copies
  on arcw1 were removed after the hash-checked copy back.
