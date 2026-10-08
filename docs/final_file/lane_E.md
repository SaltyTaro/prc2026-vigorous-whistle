# Lane E: the early chain on a new movement file (8 October 2026)

Final-round plan: [`final_round_plan.md`](final_round_plan.md). Lane E makes the **early chain** — v66's step 1 (the own-points member),
steps 2-10, with v69's step 6 — run on any movement file, and proves it on `data/raw/ranking.parquet`. Steps 11-18 (v70-v76) are Lane L's.
Nothing here uploads or writes under `runs/submissions`; no official score is read.

## What changed

| file | change | default behaviour |
|---|---|---|
| `tools/final_own_member.py` (new) | `features`: the 15 own-aircraft turnaround columns of the run's movement file (`tools/build_ownpoints_features.py`'s `load` / `identify` / `features`, unchanged, on the file alone, with the relearned key map); `predict`: the 32 own_all networks (dl32own / m25dl32own x seeds 601-616, twelve-month refits, GPU-trained) on the CPU via `prc2026_final.networks.predict_learner` on the runner's learner frame + the own columns, routed per seed by the runner's `adsb_matched` into `D/members/s<seed>.parquet` (the form of `runs/ctx_ownpoints_dev/member/s<seed>/ranking.parquet`); `compare`: ranking file only, against the shipped tables | new tool |
| `tools/final_early_chain.py` (new) | the orchestrator, steps 0-10 (below), with switches `--step1-source runner\|shipped`, `--members runner\|shipped`, `--published-table/--workbook` or `--no-published`, `--stack-preds`, `--rebuild-caches`, `--stop-after` | new tool |
| `tools/run_opdi_stack.py` | module constant `RANKING` (the movement file of the `rank` features) instead of the inline `data/raw/ranking.parquet` | identical (same path) |
| `tools/opdi_stack_fetch.py` | `--need START:END ...` adds the windows of a final file's period | identical (23 windows, checked) |

No scientific setting, declaration, model or shipped file changed. The own-point builder, the identity builders, the rematch tool, the
stacker and the bounds are called unchanged; the orchestrator only sets their module path constants (as `tools/final_chain_overrides.py`
does for steps 2-5, whose shipped flags are used as they are).

## The orchestrator (`tools/final_early_chain.py`)

    # a new file F (template T), runner root R (tools/final_predict.py run --layers v55m done, members stage included), its v55m file M:
    python tools/final_own_member.py features --root R --keymap O/rematch/key_map.parquet     # after step 0 below wrote the key map
    python tools/final_own_member.py predict  --root R --threads 16                          # Azure: ~32 s per network on a D16
    python tools/final_early_chain.py --raw F --template T --runner R --model-part M --out O --version N \
        [--published-table P --workbook W --workbook-sha256 S | --no-published]

| step | what | how the run's inputs reach it |
|---|---|---|
| 0 | key map + no-NM matches: `run_rematch_nonm.py --matches --out O/rematch` | `--raw F`; key map relearned from 2025 + F |
| 1 | model part M' = `export_ctx_ownpoints.step_model_part`'s arithmetic (own_all at 1.00 after the combination, then the masked member at 0.40; LIRF rows untouched) | `--step1-source runner`: v34 from `R/stack/detail.parquet`, the recipe detail `R/recipe/detail.parquet`, `export_stack`'s three readers from R's rematch / residual matches / peers (`prc2026_final.stack.run_tables`); `--members runner`: combination / masked members from `R/members/members.parquet`, own_all from `final_own_member.py`'s tables; the model part = M |
| 2 | no-NM rematch base B: `run_rematch_nonm.py --ranking-base --base M'` | `--raw F --template T` |
| 3-5 | v48 reconciliation, extreme-day, variance re-allocation (shipped tools, their final-file flags) | `--raw --template --detail R/recipe/detail.parquet --published-table P [--workbook W --workbook-sha256 S]` |
| 6 | v69's stacker: `run_stack_weight.rank_inputs` -> three learners x ten seeds -> `rank_value` at 0.65 | `run_opdi_stack_decl.SH = O/steps`; `run_stack_resid.RUNNER / REPAIRED / RANKING / MODEL_PART = R / R's repair root / F / M`; own-point cache `O/cache/v3_features` (shipped `dev`, `rank` rebuilt); OPDI features `O/cache/opdi` (shipped `dev`, `rank` rebuilt from `data/external/opdi/stack_v1`); key map = step 0's; `screen_daily_totals.TABLE = P` |
| 7-9 | `run_stack_resid_v3.chain_from_stack`: intruder bound + identity extension, own-motion v1 | identity table `O/cache/intruder_identity/ids.parquet` (`run_intruder_identity.build_ids`, new-file mode: the shipped 2025 rows + F); matches `O/cache/matches` (R's learner matches + the frozen stands table) |
| 10 | `run_own_motion_v2.dry_run` -> `O/vigorous-whistle_vN.parquet` | tunnel-11 identity tables `O/cache/position` (the shipped `ids_2025-*` + `ids_ranking` rebuilt from F by `run_own_motion_v2.build_ids`) |

`--no-published` skips steps 3-6 (`v61_chain.md`: without EUROCONTROL's daily figures for the period the stacker's published-day features
would be missing on every row); steps 7-10 then run on B. Every exporting module takes T (`export_submission` wrapped, as
`final_chain_overrides.py --template`). `chain.json` records inputs, every intermediate's SHA-256, cache hashes, timings and code hashes.

## Rehearsal on the ranking file (8 October, workstation; the chain's inputs live here)

**(1) Generic path with the shipped sources reproduces v69 byte for byte.** `runs/final_early_chain/rehearsal_shipped3/`:
`--step1-source shipped --members shipped --rebuild-caches --stack-preds runs/stack_weight/rank/preds.parquet` (every new-file redirection
on: relearned key map, own-point / OPDI caches, both identity tables and the match directory rebuilt under `O/cache`; step 6 reuses the
shipped per-seed predictions after checking rows, base and guard columns against the recomputed ones; the v69 rehearsal of 7 October
already showed the ten-seed refit reproduces them exactly). Final `192c4a30...` = v69. Every intermediate equals the shipped dry runs
(`runs/ctx_ownpoints_weight/rank/dry_v66`, `runs/stack_weight/rank/dry_v69`):

| step | file | SHA-256 (= shipped) |
|---|---|---|
| 1 model part | `steps/model_part/vigorous-whistle_v66.parquet` | `73eef8da...` (member delta `30965f4c...` equal too) |
| 2 rematch base | `steps/rematch/base/...` | `e0ffdbc4...` (relearned key map `79276c48...` = `runs/rematch_nonm/key_map.parquet`) |
| 3 v48 | `steps/R/` | `cf3b9f8e...` |
| 4 extreme-day | `steps/X/` | `02a3e44c...` |
| 5 var-alloc | `steps/Y/` | `12a50ea0...` |
| 6 stacker | `steps/stack/vigorous-whistle_v69.parquet` | `467ccb74...` |
| 7-8 identity | `steps/identity/` | `e0b24196...` |
| 9 own-motion v1 | `steps/own/` | `f6ab8d8b...` |
| 10 own-motion v2 | `steps/om2/` = final | **`192c4a30...` = v69** |

Rebuilt caches against the shipped ones: identity table `3097dd06...` and `ids_ranking` `b8cf4ce0...` byte-identical; OPDI `features_rank`
`eb9896f8...` byte-identical; own-point `rank` cache equal in rows, order and values (file bytes differ). The stacker inputs for Lane L
(`steps/stack/stacker_rank.parquet`) = `95a5eedc...`, the file `tools/final_late_chain.py stacker` writes. Wall 742 s (step 0 51 s, 1 17 s,
3-5 159 s, 6 83 s with reused predictions, 7-10 353 s).

**(2) Own columns and the CPU member.** `final_own_member.py features` on the ranking file (64.5 s) reproduces all 15 own columns of
`data/external/adsblol/features_v19own` exactly (values and NaN pattern, 344,841 rows); `export-v19own` reproduces both monthly tables'
values; the runner's 11 v19 columns equal the shipped ones. `predict` on Azure arcw0 (32 networks, 16 threads, ~32 s each; outputs mirrored
with matching SHA-256): against the shipped GPU predictions, per network RMS 0.10-0.20 s (p99 at most 0.30 s, one row at 77 s); the routed
16-seed mean RMS 0.054 s, p99 0.078 s, max 15.5 s, 43 rows above 1 s, the same rows without a value (`runs/final_early_chain/own_ranking/compare.json`).
The runner's other CPU members are of the same size (RMS 0.05-0.07 s, `runner.md`).

**(3) Rounded effect on the files.** Against the shipped intermediates (`runs/final_early_chain/q_own_cpu`, `q_runner`):

| run | step | rows differing | max | squared change |
|---|---|---:|---:|---:|
| CPU own_all only (shipped step-1 inputs, shipped combination / masked tables) | model part | 2,032 (99.6% within 2 s) | 8 s | 2.2e3 s^2 |
| full runner path (runner v34 / detail / tables / members, CPU own_all, the runner's own v55m `v955` `0d3dc265...`) | model part | 10,669 | 14 s | 1.2e4 |
| | 2 rematch | 10,663 | 14 s | 1.2e4 |
| | 3 v48 | 12,026 | 88 s | 2.2e4 |
| | 5 var-alloc | 12,052 (99.6% within 2 s) | 88 s | 2.2e4 (about 0.0002 s of RMSE) |

The runner's v955 itself differs from the shipped v55 model part on 9,777 rows (max 101 s, the LEMD u300 flip); in the model part the
member replaces that row's table, so the 101 s row is gone.

**(4) The whole new-file path, end to end** (`runs/final_early_chain/newfile_e2e/`): `--raw` = a byte copy of the ranking file
(`runs/final_dryrun/input/final.parquet`) and `--template` = a byte copy of the template, so every new-file branch runs (the shipped tools
get `--raw / --template`, every exporting module takes the template, all caches rebuilt); `--step1-source runner --members runner`, the
CPU own_all member, the runner's own v55m `v955`, step 6 refitted (three learners x ten seeds, 317 s at the shipped 4 threads). Wall 1,015 s.
Final `dea1cc9c...`; against v69: **17,351 rows differ, 95.1% by at most 2 s, max 89 s (LIRF), squared change 4.4e4 s^2 (about 0.0003 s
of RMSE)**; by airport the largest are LIRF 8,542 rows (the runner's v53m-v55m float chaining, as in `runner.md`), LTFM 1,993, EGLL 1,274.
The refitted per-seed stacker predictions differ from the shipped ones by RMS 0.50 s (slightly different inputs, early stopping).
`--no-published` (steps 3-6 skipped; `runs/final_early_chain/nopub_smoke/`) runs through to step 10; it is the declared fallback, not a
reproduction (it drops the reconciliation and the stacker: 333,120 rows differ from v69). Summary: [`lane_E_rehearsal.json`](lane_E_rehearsal.json).

## For the final file (commands)

    # 1. runner: tools/final_predict.py inspect / external / extract-adsb / pull-weather / refit-adj2 / run --layers v55m --no-fill-pins
    #    -> R, its v55m file M (R/final/vigorous-whistle_vN.parquet)
    # 2. key map (step 0 alone): the chain's first step writes O/steps/rematch/key_map.parquet; or run it first:
    python tools/run_rematch_nonm.py --matches --raw F --template T --out O/steps/rematch
    # 3. own_all member (CPU; Azure: copy R's input.json / features / adsb / tokens and runs/ctx_ownpoints_dev/nets/*/refit12; ~17 min on a D16)
    python tools/final_own_member.py features --root R --keymap O/steps/rematch/key_map.parquet
    python tools/final_own_member.py predict  --root R --threads 16
    python tools/final_own_member.py export-v19own --root R --out O/features_v19own      # Lane L's --features-own-dir
    # 4. OPDI events for the period, if published (optional): python tools/opdi_stack_fetch.py --need YYYY-MM-DD:YYYY-MM-DD
    # 5. the chain (step 0 is idempotent: run_rematch_nonm keeps existing files)
    python tools/final_early_chain.py --raw F --template T --runner R --model-part M --out O --version N         --published-table P --workbook W --workbook-sha256 S        # or --no-published
    # -> O/vigorous-whistle_vN.parquet (Lane L's --base), O/steps/stack/stacker_rank.parquet (--stacker-rank), R/adsb/learners (--matches-dir)

## Remaining gaps (not solved here)

- **Where the chain runs.** Steps 3-10 read the shipped 2025 development tables and caches (`runs/fair_dev`, `runs/lomo_dev`,
  `runs/daily_totals`, `runs/scout`, `runs/ctx_ownpoints_weight/chain`, `runs/stack_resid/v3_features/dev`, `runs/opdi_stack/features_dev`,
  `runs/dig/census61`, ...), which exist on this workstation (the rehearsals ran here; 742 s / 1,015 s). A VM run needs that set synced first
  (not done; arcw0 only ran the 32-network prediction). The own-member prediction is the only step that needs a VM for speed.
- **Published figures.** Steps 3-5 change nothing on airport-days EUROCONTROL has not published; step 6 has no partial mode: on a period
  published only in part, its published-day features are NaN on the unpublished rows (LightGBM's missing branch; the support guard ignores
  NaN). With `--no-published` there is no stacker table
  for Lane L's late passes either.
- **OPDI.** Without event extracts for the period, `o_has` is 0 on every row and the OPDI learner moves nothing (LSZH rows take the v3
  learner); whether OPDI v0.0.2 publishes the final months in time is unknown.
- **Surface days.** The identity tables, own points and both rematches read `data/external/adsblol/surface/surface_<day>.parquet`
  (the runner's `extract-adsb` puts them there); a day missing there leaves its rows unidentified, silently.
- **Float noise.** Runner-sourced inputs differ from the shipped GPU / integer chain by about 0.0003 s of RMSE on the ranking file
  (rehearsal 4); threshold rules can flip single rows (the 89 s LIRF row).
