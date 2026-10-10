# Tier B, model part (lane M): February / June 2025 out-of-sample v30 -> v55m

Status: **in progress** (10 October 2026). Umbrella: `configs/febjun_tierb_v1.json`; parent rule `configs/febjun_check_v1.json`.
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
