# Tier B, model part (lane M): February / June 2025 out-of-sample v30 -> v55m

Status: **complete** (10 October 2026, read 22:23 UTC on arcw6): **nothing switched off**. Umbrella: `configs/febjun_tierb_v1.json`; parent rule `configs/febjun_check_v1.json`.
Readings declaration (committed before any February / June reading): `configs/febjun_tierb_readings_model_v1.json`.

## Construction (chosen)

The development rows layout every model-part round used (`runs/lomo_dev/rows.parquet`, `tools/run_lomo_rows.py`), extended to
February / June 2025, then the development stack functions in shipping order:

- **rows**: `tools/run_adsb_v24.month_table(load_frame(..., "cross_fit"), adsb_features(...))` for the frame (labels, day, airport,
  floor, matched, LIRF scopes, cross-fit v9: fold B held out February / June / October); the v9 side's residual = v26's `rnet_w90`
  combination of the twelve-month LOMO adj2 / cat (`runs/adsb_v25/dev/predictions_<m>.parquet`) and the 16 rnet seeds' held-month
  predictions (`runs/adsb_rnn/rnb_s5xx/<m>/`); committee / LIRF tree = the Tier B LOMO tree tables (member rule of
  `tools/build_lomo_tables.py`); networks = the Tier B 4-seed LOMO means (e24 117-120, e24b 133-136, e32 401-404, lirf 301-304);
  `run_lomo_rows.assemble` for base / mix / lirf_tree / lirf_net / v30 / v33.
- **chain**: v30 -> v31 (= v30, the EOBT_1 detector cannot fire on 2025) -> v33 -> v34 -> v35 (`run_mix_gate.stack`: u300, callsign
  fix 50%, queue trust 0.75) -> v38 (two-sided stretch) -> v40 (= v38, fill pins off) -> v45 (LIRF gate) -> v46 (share gate n85_nd1200)
  -> v51m (combination member w40) -> v52m (masked member w40) -> v53m / v54m / v55m (`src/prc2026_final/lirf_members.deltas`).
  The steps v35 -> v45 are `tools/run_share_gate_reread.full_stack` split at its layer boundaries (checked equal to it).
- **callsign fix input**: `tools/adsb_rematch_nonm.py` (unchanged, through `prc2026_final.stack.rematch`) on the no-NM EDDF / EHAM
  departures of February / June 2025; queue segments by `prc2026_final.stack.queue_segments` on `matches_v11` + those rows.
- **rows read**: every February / June 2025 departure with an out-of-sample v9 (the cross-fit tables cover the clock-present rows);
  missing-clock rows and the LIRF fill population have no out-of-sample v9 for these months and are listed, not read.

## Files for the chain lane (lane C)

| file | content |
|---|---|
| `runs/febjun_tierb/model/chain_<month>.parquet` (`2025-02`, `2025-06`) | per row: `MVT_ID_mvt`, month, day, airport, label, floor, matched, lirf_m, lirf_u, the components (v9, residual, base, committee, nets, mix, lirf_tree, lirf_net, ens_ctx_e24 / e24b / e32 / lirf, combo_all, mask_unmatched, lirf_mask) and every layer column (`off_r3`, v30, v31, v33, v34, v35, v38, v40, v45, v46, v51m, v52m, v53m, v54m, v55m). **`v55m` is the model part the chain starts from.** |
| `runs/febjun_tierb/model/excluded_<month>.parquet` | the month's departures without an out-of-sample v9 (MVT_ID_mvt, reason) |
| `runs/febjun_tierb/model/trees/<table>_<month>.parquet`, `runs/febjun_tierb/model/nets/<month>.parquet` | the Tier B component tables (MVT_ID_mvt + columns) |

Tool: `tools/febjun_tierb_model.py` (subcommands `trees`, `nets`, `rematch`, `rows`, `chain`, `read`).

## Results (10 October 2026, 22:23 UTC; `docs/final_file/tierb_model_results.json`)

Run on arcw6, not on the workstation (the user's PC was short of memory; execution record [`tierb_vm.md`](tierb_vm.md)). Before the
real run the January / July checks were repeated on arcw6: trees, nets, rows and chain identical to this workstation's `checks/*.json`,
rematch identical except one 2.3e-13 m float in a taxi-path difference. The 58 tree fits were hash-checked on both machines before `trees`
(116 files) and passed the record check (held month, fit months, held rows, declaration, not smoke). Rows read: February 142,622 of
143,732, June 180,640 of 183,142. Paired-day 95% intervals (2,000 replicates, generator seeded 20261004), in percent of the month's RMSE.

| reading | layer | Feb 2025 change [95% interval] | rows moved | Jun 2025 change [95% interval] | rows moved | off? |
|---|---|---|---:|---|---:|---|
| R3 | v30 recipe, committee side and mix: v30 against the v9 side alone (off_r3 = max(v9 + residual on matched non-LIRF rows, floor)) | -5.550% [-7.085, -3.742] | 142,617 | -4.233% [-4.953, -3.584] | 180,629 | no |
| R5a | v33 outer weights (LTFM 1.0 / matched 0.6 / unmatched 0.7) | -0.957% [-1.532, -0.314] | 93,077 | -0.214% [-0.285, -0.154] | 84,580 | no |
| R5b | v34 LIRF unmatched v9 share 0 | -0.012% [-0.164, +0.207] | 10,640 | -0.053% [-0.132, +0.029] | 5,804 | no |
| R6 | v35: u300 gate, callsign fix 50%, queue trust 0.75 | -0.014% [-0.052, +0.018] | 70 | -0.057% [-0.119, -0.001] | 171 | no |
| R8 | v38 two-sided tail stretch | -2.111% [-3.296, -0.770] | 73,997 | -0.121% [-0.276, +0.037] | 91,787 | no |
| R10 | v45 LIRF disagreement gate (v40 = v38, fill pins off) | -0.038% [-0.223, +0.075] | 11 | -0.121% [-0.265, -0.010] | 55 | no |
| R11 | v46 network share 0.85 + networks-far-below gate | +0.038% [-0.144, +0.229] | 131,977 | -0.060% [-0.105, -0.010] | 166,129 | no |
| R12a | v51m combination member w40 | +0.260% [-0.131, +0.665] | 131,977 | -0.252% [-0.334, -0.172] | 166,129 | no |
| R12b | v52m masked member w40 | -0.393% [-0.785, -0.003] | 56,864 | -0.130% [-0.215, -0.044] | 104,745 | no |
| R13a | v53m masked LIRF member w55 (LIRF non-population unmatched rows) | -0.077% [-0.167, +0.013] | 10,640 | -0.039% [-0.085, +0.002] | 5,804 | no |
| R13b | v54m dark re-read w70_s750 (only in LIRF months with a non-population match share below 5%, read from the rows; it moves no row in a covered month) | -0.010% [-0.063, +0.042] | 10,640 | +0.000% [+0.000, +0.000] | 0 | no |
| R13c | v55m matched-row LIRF member mw40 | +0.000% [+0.000, +0.000] | 0 | +0.003% [-0.034, +0.040] | 8,696 | no |

**Rule outcome: no model-part layer is switched off in either month** (`switched_off` empty). The worst reading is R12a (v51m combination
member) in February, +0.26% with an interval that includes zero; it stays on. Everything with an interval wholly below zero helps.
