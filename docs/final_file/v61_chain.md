# The v61 chain on a new movement file (final-file readiness, 3 October 2026)
> Later files: the v66 chain (steps added after v61, intermediates and rehearsal) is [`v66_chain.md`](v66_chain.md); this page stays the v61 record.

v61 = 218.6142 official is built in eight steps: the final-file runner's model part, then seven separate tools. This page gives the
exact order, what each step reads, what each tool hard-coded, and the optional flags added on 3 October so the whole chain runs on a
new file with the ranking file's schema. **Every flag defaults to the shipped behaviour**: with no flag each tool reproduces its shipped
file byte for byte (checked below). Nothing here uploads or writes under `runs/submissions`; a submission build stays the user's word.

Shared overrides (`tools/final_chain_overrides.py`, accepted by every post-runner tool):

| flag | replaces | used by |
|---|---|---|
| `--raw FILE` | `data/raw/ranking.parquet` | all seven tools |
| `--detail FILE` | `runs/final_dryrun/run/recipe/detail.parquet` (scope per departure) | v48 reconciliation, extreme-day, var-alloc |
| `--published-table FILE` | `data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet` | v48, extreme-day, var-alloc, stacking |
| `--template FILE` | `data/raw/submitting.parquet` | every tool that exports through `export_submission`; the intruder tools' `--build` |

## Inputs a new period needs

| input | how | used by |
|---|---|---|
| the movement file and its template | from the organisers | everything |
| adsb.lol surface day files for every day of the file and the day before, in `data/external/adsblol/surface/surface_YYYYMMDD.parquet` | `tools/final_predict.py external` / `extract-adsb --execute` (2-4 GB download per day; an Azure worker is the recommended host, copy the parquet files back) | runner (ADS-B features, callsign rematch), step 2 (no-NM rematch), step 7 (identity table) |
| IEM weather for the new station-months | `tools/final_predict.py pull-weather --execute` (one request at a time), then `--weather-inventory` | runner |
| EUROCONTROL `All_Pre-Departure_Delay.xlsx` covering the new period (DLY_ALL_PRE_2, FLT_DEP_IFR_2 per airport-day; same terms as for v48, the user's decision of 29 September) | download from eurocontrol.int/performance/data; extract with `screen_daily_totals.extract()` into a NEW parquet path (set `WORKBOOK` and `TABLE` on the module first; do not overwrite the shipped extract) | steps 3-6 |
| (only for months without a saved booster) the declared residual adj2 refit | `tools/final_predict.py refit-adj2 --month M` | runner |

If EUROCONTROL has not yet published the period's daily figures (the 16 September 2026 release ended on 31 July 2026), the
reconciliation, the extreme-day layer and the variance re-allocation change nothing (no airport-day is "complete"), and the stacking layer
must be **skipped**, because its published-day features would be missing for every row (outside its development support). The chain then is:
runner model part -> no-NM rematch -> intruder bound -> identity extension.

## The commands, in order

`R` = the runner's root for the new file, `F` = the new movement file, `T` = its template, `P` = the new published parquet, `W` = the
new workbook, `O` = a scratch directory (never under `runs/submissions`). `N` is any free version number for the dry-run file names.

1. **Model part (runner, v55m)** (`docs/final_file/runner.md`):

       tools/final_predict.py inspect --root R --raw F --template T
       tools/final_predict.py external --root R           # then extract-adsb / pull-weather / refit-adj2 as it lists
       tools/final_predict.py run --root R --layers v55m --version N [--threads 12 --workers 8] [--weather-inventory ...]
   -> `R/final/vigorous-whistle_vN.parquet` (call it M; M_SHA its SHA-256); `R/recipe/detail.parquet`; the repair stage writes
   `R/repair/edited_ids.parquet` and, when it flags rows, a repaired run (`--repair-root` or `R/repair/run`).

2. **No-NM ADS-B rematch** (`configs/rematch_nonm_v1.json`; EDDF / EHAM / LSZH no-NM rows; key map relearned from 2025 + F; 2025
   matches and offsets stay those of `runs/rematch_nonm/`):

       tools/run_rematch_nonm.py --matches --raw F --out O/rematch
       tools/run_rematch_nonm.py --ranking-base --raw F --out O/rematch --base M --base-sha256 M_SHA
   -> `O/rematch/base/vigorous-whistle_v55m_rematch.parquet` (= B, SHA B_SHA), `O/rematch/key_map.parquet`.

3. **v48's reconciliation** (`configs/daily_totals_v1.json`):

       tools/export_daily_totals.py --base B --version N --dry-run O/R --raw F --detail R/recipe/detail.parquet \
           --published-table P --template T --workbook W --workbook-sha256 <W's SHA-256> --no-incumbent-report

4. **Extreme-day layer** (`configs/scout_extreme_shift_v1.json`):

       tools/scout_extreme_shift.py --base B --recon O/R --version N --dry-run O/X --raw F --detail R/recipe/detail.parquet --published-table P --template T

5. **Variance re-allocation** (`configs/scout_var_alloc_v1.json`):

       tools/scout_var_alloc.py --base B --recon O/R --version N --previous O/X/vigorous-whistle_vN.parquet --previous-sha256 <its SHA> \
           --dry-run O/Y --raw F --detail R/recipe/detail.parquet --published-table P --template T
   -> `O/Y/vigorous-whistle_vN.parquet` (= C, SHA C_SHA) and `O/Y/rows.parquet`.

6. **Guarded stacking residual** (`configs/stack_resid_v2.json`; components from the runner's stage outputs):

       tools/run_stack_resid.py --declaration configs/stack_resid_v2.json --dry-run --version N --base C --base-sha256 C_SHA \
           --model-part B --model-part-sha256 B_SHA --model-part-original M --chain-rows O/Y/rows.parquet \
           --runner R --repaired <R/repair/run, or none> --out O/stack --raw F --published-table P --template T
   -> `O/stack/vigorous-whistle_vN.parquet` (= D, SHA D_SHA).

7. **Identity table** for the new file's movements (2025 rows kept from the shipped table):

       tools/run_intruder_identity.py --build-ids --raw F --ids O/ids.parquet --keymap O/rematch/key_map.parquet

8. **Intruder stand bound + identity extension** (`configs/intruder_bound_v1.json`, `configs/intruder_identity_v1.json`; both last):

       tools/run_intruder_identity.py --dry-run --version N --base D --base-sha256 D_SHA --raw F --ids O/ids.parquet --out O/final --no-v60-check
   -> `O/final/vigorous-whistle_vN.parquet`: the v61-equivalent file. (It applies the accepted different-type bound itself and then the
   identity extension; `tools/run_intruder_bound.py --dry-run ... --raw F --out ...` alone gives the v60-equivalent.)

9. **Own-motion bound** (`configs/own_motion_v1.json`, accepted 3 October; applied after step 8, last; not part of v61, the layer of
   the v62 dry run `runs/own_motion/dry_v62/`, `cba143f8...`). It needs the step-7 identity table, the new file's `matches2_<YYYY-MM>.parquet`
   (the runner's ADS-B matching; `adsb_gate`) and the surface day files of every take-off day and the day before:

       tools/run_own_motion.py --dry-run --version N --base O/final/vigorous-whistle_vN.parquet --base-sha256 <its SHA> --raw F \
           --ids O/ids.parquet --matches-dir <directory with the new months' matches2 files> --out O/own
   -> `O/own/vigorous-whistle_vN.parquet`. Offsets stay those fitted on the twelve 2025 months. With no flag the tool reproduces the v62
   dry run on v61 byte for byte.

A submission build (`--build`) remains the user's word. For a new file run step 8 once more without `--out` (default directory
`runs/intruder_identity/dry_v<N>`), then `--build --version N --user-decision "<words>"` with the same flags (`--raw F --ids ... --base D
--base-sha256 D_SHA --no-v60-check --template T`). `tools/prc_submit.py` itself still checks the ranking file's template and organiser
ETags: an upload of a final file needs its own small change there (not made).

## What each tool hard-coded, and the flag now available

| tool | hard-coded before | flag |
|---|---|---|
| `tools/run_rematch_nonm.py` | `RANKING`, output dir `runs/rematch_nonm`, `BASE` = the shipped v55 model part | `--raw`, `--out` (2025 matches/offsets always from `runs/rematch_nonm`), `--base/--base-sha256`; `--dev/--score` refuse `--out` |
| `tools/export_daily_totals.py` | `RANKING`, `export_v33.DRY` (runner detail), the workbook SHA of the declaration, the published extract, the template, a comparison with the uploaded v48 | `--raw`, `--detail`, `--published-table`, `--template`, `--workbook/--workbook-sha256`, `--no-incumbent-report` |
| `tools/scout_extreme_shift.py` | inline ranking path (FLIGHT_RULE), plus v48's `ranking_table` | `--raw`, `--detail`, `--published-table`, `--template` |
| `tools/scout_var_alloc.py` | inline ranking path (departure features), default previous step = uploaded v55 | the same four (+ the existing `--previous`) |
| `tools/run_stack_resid.py` | runner `runs/final_dryrun/run`, repaired run `runs/final_lfpg_eobt/run`, the shipped v55 model part, inline ranking path, output dir | `--runner`, `--repaired` (or `none`), `--model-part-original`, `--model-part/--model-part-sha256`, `--chain-rows`, `--out`, `--raw`, `--published-table`, `--template` |
| `tools/run_intruder_bound.py` | inline ranking path; `--out` accepted only absolute paths | `--raw`, `--template` (build); `--out` now resolves under the project root |
| `tools/run_intruder_identity.py` | inline ranking path, identity table, key map, the v60 reproduction check, output dir | `--raw`, `--ids`, `--keymap`, `--no-v60-check`, `--out`, `--template` (build) |
| `tools/run_own_motion.py` (step 9) | base = the uploaded v61 | `--raw`, `--ids`, `--matches-dir`, `--base/--base-sha256`, `--out`, `--template` (build) |

Fixed inputs that stay as they are for any period: the 2025 development tables (cells, scales, models, offsets are all fitted on 2025),
`data/external/adsblol/matches_v11/stands.parquet`, the declarations. The two guard rows exist only in the
ranking file; on a new file the guard is inert. Month lists in these tools are the 2025 development months only; the new file's months
come from its rows (no `RANKING_MONTHS` constant is involved after the runner).

## Rehearsal on a copy of the ranking file (`runs/final_v61_rehearsal/`)

`raw/ranking.parquet` and `raw/submitting.parquet` are byte-identical copies (9db75e3a..., 0d383408...); every step ran through the
override flags pointing at the copies.

**(a) From the shipped v55m model part** (`runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet`, 266b8c93...) - every intermediate
equals the shipped one byte for byte:

| step | file | SHA-256 (= shipped) |
|---|---|---|
| 2 rematch | key map, matches (equal frames), base | `75957f82...` |
| 3 v48 | file, values, days | `ebb25a48...` (174 s) |
| 4 extreme-day | | `97f893e3...` |
| 5 var-alloc | file, rows | `c67f20d6...` (= v59 before the bound) |
| 6 stacking | | `cff778e0...` |
| 7 identity table | | `3097dd06...` (byte-identical to the shipped table) |
| 8 bound (alone) | | `30021455...` = v60 |
| 8 bound + identity | | **`fe137f09...` = v61** |

**(b) From the runner's own v55m** (`runs/final_dryrun/run/final/vigorous-whistle_v955.parquet`, 0d3dc265...; the realistic path for a new
file, `--model-part-original` set to it, `--no-v60-check`): `runs/final_v61_rehearsal/from_runner/final/vigorous-whistle_v61.parquet`
(9ad2e1f1...). Against v61: 11,400 rows differ, 99.6% by at most 2 s, sum of squared differences 3.1e4 (about 0.0002 s of RMSE).
9,492 of them are the rows where the runner's v55m already differs from the shipped model part (9,777 rows, float noise, RMS 1.45 s);
1,908 more move through the day shifts and the stacking features. None of the 56 stand-bound rows differ.

**Defaults unchanged** (no override flag; only the output location given): v48 / extreme-day / var-alloc / stacking reproduce
`ebb25a48`, `97f893e3`, `c67f20d6`, `cff778e0`; the intruder bound reproduces v58 `eabd3bd6` (base v57), v59 `2a84b713` (base c67f20d6) and
v60 `30021455` (base cff778e0); the identity extension reproduces v61 `fe137f09`; `run_rematch_nonm.py --matches` keeps its files.

## Still manual for a real final file

- ADS-B day extraction for new dates (Azure worker, `extract-adsb --execute`) and the IEM weather pull: the runner prints the commands.
- The EUROCONTROL workbook covering the period: download, extract to a new parquet, pass `--published-table`/`--workbook`; if the period is
  not yet published, skip steps 3-6 (see above).
- `tools/prc_submit.py` for a final file (template / ETag checks are tied to the ranking files).
- The runner's LFPG EOBT_1 repair: whether its detector flags anything in the new period decides `--repaired` for step 6.
