# Tier B lane M (model part v30 -> v55m): resume note (paused 10 October 2026, ~13:00 UTC)

Lane M was paused at the coordinator's request, waiting for inputs. Nothing has been read on February / June 2025.

## Done

- **Readings declaration committed: `configs/febjun_tierb_readings_model_v1.json`, commit `4fb2ad6`**, made before any
  February / June table existed. It governs R3, R5a/b, R6, R8, R10, R11, R12a/b and R13a/b/c, one shipped step each. It does not
  read R4 (v31 = v30 in 2025), R7 (v36 is not in the final chain) or R9 (the fill pins are off).
- **Tool: `tools/febjun_tierb_model.py`**, same commit. Every January / July 2025 reproduction is exact, max |difference| 0.0
  (outputs in `runs/febjun_tierb/model/checks/{trees,nets,rows,rematch,chain}.json`):
  - tree tables;
  - 16-seed network means and the routed members;
  - frame path against `runs/fair_dev/rows_shipped.parquet` (v30 / v33 ≤ 9.1e-13 s);
  - assembly against `runs/lomo_dev/rows.parquet`;
  - January rematch against `out_arcw5`;
  - the split stack against `full_stack`;
  - the stored RMSEs of v30, v33, v45, v46, v51m, v52m and v53m / v54m / v55m on their routes.
- **Label-free inputs already built for February / June:**
  - `runs/febjun_tierb/model/rematch/rematch_2025-02.parquet`, SHA-256 `d9f1e6ad…`: 326 no-NM EDDF / EHAM departures;
  - `rematch_2025-06.parquet`, SHA-256 `d3cc9e69…`: 704 departures, 422 rows out, because 1–10 June 2025 have no adsb.lol
    surface file, as in every 2025 match table.
- **Label-free facts:**
  - Rows read: February 142,622 of 143,731, June 180,640 of 183,142 (clock-present rows only).
  - LIRF in **February 2025 is dark**: 0 of 10,640 non-population rows matched. So v54m acts on February and v55m has no
    February route.
  - LIRF in June 2025 is covered: 8,696 matched, 5,804 unmatched, share 0.60. So v54m moves no June row.

## Inputs still needed (at ~13:00 UTC)

| input | path | expected | local at pause |
|---|---|---|---|
| tree LOMO fits (29 leaves × 2 months) | `runs/febjun_lomo/learners/<leaf>/lomo_2025-0{2,6}/{predictions.parquet,run.json}` | 58 | 31 (no partial directory) |
| network LOMO fits (lane G) | `runs/febjun_nets/nets/<variant>_s<seed>/lomo_2025-0{2,6}/predictions.parquet` | 80 | 65 |

- February held rows: 143,732, including the LTFM 28 February 23:59:57 row that sits in the March file.
- **Network fits, per month:**
  - `ctx_e24_s117`–`s120` (ens_ctx_e24) and `ctx_e24_s133`–`s136` (ens_ctx_e24b);
  - `ctx_e32_s401`–`s404`, `ctx_lirf_s301`–`s304`;
  - `dl32_s601`–`s604`, `m25dl32_s601`–`s604`;
  - `m25_32_s701`–`s704`, `lm25_32_s801`–`s804`.
  - These are the `TIERB_NETS` constant in the tool; check the hashes against lane G's `runs/febjun_nets/mirror.json`.

## Commands, in order, once both inputs are complete locally

From the repository root (each step takes minutes and needs under ~3 GB RAM; run them one after another):

    .venv\Scripts\python.exe -X utf8 -B tools\febjun_tierb_model.py trees     # -> runs/febjun_tierb/model/trees/<table>_<month>.parquet
    .venv\Scripts\python.exe -X utf8 -B tools\febjun_tierb_model.py nets      # -> runs/febjun_tierb/model/nets/<month>.parquet (seed means + routed members)
    .venv\Scripts\python.exe -X utf8 -B tools\febjun_tierb_model.py rows      # -> rows_<month>.parquet, excluded_<month>.parquet
    .venv\Scripts\python.exe -X utf8 -B tools\febjun_tierb_model.py chain     # -> chain_<month>.parquet (every layer column; v55m = the model part), chain_record.json
    .venv\Scripts\python.exe -X utf8 -B tools\febjun_tierb_model.py read      # -> docs/final_file/tierb_model_results.json; prints each layer and the switch-offs

- The default months are 2025-02 and 2025-06; `read` must take both together, because one generator per layer runs over
  February, then June.
- `rematch` is already done; rerun it only if its outputs go missing.
- `read` refuses unless the declaration is committed and unchanged.
- Optional re-checks (January / July, about 1–3 min each): `trees --check`, `nets --check`, `rows --check`, `rematch --check`,
  `chain --check`.
- Afterwards: put the readings into `docs/final_file/tierb_model.md`, then commit `docs/final_file/tierb_model_results.json` and the
  doc with an explicit pathspec.
- `runs/febjun_tierb/model/chain_<month>.parquet` is the chain lane's input (its `v55m` column, plus every component).

## Pitfalls found

- **Member routing** uses the learner-directory match flag (`runs/adsb_ctx2/inputs/adsb_features_arcw0`,
  `build_combo_member.matched_flags`), not the rows' `matches_v11` flag, exactly as the member rounds did.
- **`nets` needs all 40 fits of a month:** `net_means` stops on a missing file or on seeds that predict different rows.
- **`rows` stops on any non-finite** v9, base, mix, committee or nets value. None is expected on clock-present rows.
- **Excluded rows:** missing-clock departures and the LIRF fill population have no out-of-sample v9 for February / June
  (`scratch/v24` holds clock-present rows only). The declaration excludes them from the readings, and they are listed in
  `excluded_<month>.parquet`.
- **v54m in February:** January and February 2025 are dark LIRF months, so the dark re-read acts there. This is the regime it
  was declared for.
- **Applying a switch-off:** the tool only reads.
- **Waiter stopped at the pause:** the polling script (`scratchpad/laneM_wait.sh`, background task `bhtc46chb`, bash PID 28744)
  ran `launch_febjun_lomo.py mirror` every 20 minutes. It was stopped; nothing of lane M runs unattended.
