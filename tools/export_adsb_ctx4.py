"""Export the 32-epoch round's selected candidate (configs/adsb_ctx_v4.json, g_thirds) on top of the incumbent v29.

v29 = v28's committee mix (round A pool_s75: 0.5 ens_ctx + 0.5 ens_ctx_e24 at share 0.75) with round B's LIRF non-population blends
(configs/adsb_ctx_lirf_v1.json lctx_b100_up). The 32-epoch round changes only the committee mix's network table (non-LIRF rows); round B's
LIRF blends are kept exactly. Steps:
1. Every ensemble's ranking column is the mean of its declared seeds' twelve-month refits (tools/export_adsb_nn2.ensemble_table): the
   inherited ensembles must equal v29's recorded columns to 1e-9 s; ens_ctx_e32 is the sixteen GPU seeds 401-416. Each refit record carries
   the declared variant, code, configuration and seed, and its saved model the recorded hash; all seeds one ADS-B ranking table (v29's);
   every context seed's token files the declared content.
2. v29 is rebuilt from its recorded components (runs/adsb_ctx/export_v29/ranking_values.parquet, hash-checked against
   docs/adsb_ctx_lirf/export.json) on the development scorer's code path: the committee mix must equal v29's recorded one to 1e-9 s, the
   values the recorded v29 to 1e-6 s and the uploaded v29 integers exactly.
3. The selected committee mix replaces pool_s75 on the same code path (round B's LIRF blends unchanged); floor; nearest-integer export
   through the frozen submission contract. Every LIRF row must equal the uploaded v29.
The coverage-regime ship check (configs/adsb_ctx_regime_check_v4.json) must have passed on both thinned months, and round D
(configs/adsb_ctx_v3.json) must not have selected a candidate that has not been combined with this one. Nothing is uploaded here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from export_adsb_nn2 import ensemble_table  # noqa: E402
from export_adsb_v11 import ranking_table  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_v11 import candidate_values, check_parents  # noqa: E402
from run_lirf_mixture import MOVEMENT_COLUMNS  # noqa: E402

RANKING_MONTHS = ["2026-01", "2026-07"]
REGIME_MONTHS = ["2025-07", "2025-12"]
CHECK = "configs/adsb_ctx_regime_check_v4.json"
ROUND_B = "configs/adsb_ctx_lirf_v1.json"
ROUND_B_SELECTION = "runs/adsb_ctx/dev_lirf/selection.json"
ROUND_D_SELECTION = "runs/adsb_ctx/dev3/selection.json"
RECORDED = ["ens_ctx", "ens_ctx_e24", "ens_ctx_lirf", "ens_new", "ens_nn", "ens_nnl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_v4.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_ctx/dev4/selection.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("docs/adsb_ctx_lirf/export.json"), help="v29's export record")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx3/export.json"))
    parser.add_argument("--control-only", action="store_true", help="stop after the v29 control reproduction (no ens_ctx_e32, no export)")
    args = parser.parse_args()
    started = time.time()
    decl, decl_b = read_json(ROOT / args.declaration), read_json(ROOT / ROUND_B)
    if decl["audit_used"] or decl_b["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    selection_b = read_json(ROOT / ROUND_B_SELECTION)
    if selection_b["declaration_sha256"] != object_hash(decl_b) or selection_b["selected_candidate"] != "lctx_b100_up":
        raise ValueError("Round B's selection is not v29's lctx_b100_up.")
    lirf_spec = decl_b["candidates"]["lctx_b100_up"]
    control_spec = decl["candidates"]["control"]
    name, candidate, regime = None, None, {}
    if not args.control_only:
        selection = read_json(ROOT / args.selection)
        if selection["declaration_sha256"] != object_hash(decl) or not selection["selected_candidate"]:
            raise ValueError("The selection does not belong to this declaration or selected nothing.")
        name = selection["selected_candidate"]
        if not selection["candidates"][name]["eligible"] or name == "control":
            raise ValueError("The selected candidate is not eligible.")
        candidate = decl["candidates"][name]
        check = read_json(ROOT / CHECK)
        regime = {m: read_json(ROOT / f"docs/adsb_ctx/regime_check3_{m}.json") for m in REGIME_MONTHS}
        if check["candidate"]["name"] != name or any(r["declaration_sha256"] != sha256_file(ROOT / CHECK) or not r["gate_month"]["passes"] for r in regime.values()):
            raise ValueError("The coverage-regime ship check has not passed on both thinned months for this candidate.")
        if (ROOT / ROUND_D_SELECTION).exists() and read_json(ROOT / ROUND_D_SELECTION).get("selected_candidate"):
            raise ValueError("Round D selected a candidate: a combined candidate must be declared and checked before either ships.")
    config = merged_config(decl)
    config_b = merged_config(decl_b)
    for new in decl_b["new_networks"]:
        config["extra_feature_tables"][new], config["ensembles"][new] = config_b["extra_feature_tables"][new], config_b["ensembles"][new]
    networks = {**decl["networks"], **{n: decl_b["networks"][n] for n in decl_b["new_networks"]}}
    check_parents(config)

    # recorded components of v29
    v29_record = read_json(ROOT / args.incumbent)
    if v29_record["selected_candidate"] != "lctx_b100_up" or sha256_file(ROOT / v29_record["ranking_values"]) != v29_record["ranking_values_sha256"]:
        raise ValueError("The recorded v29 ranking values changed.")
    v29_detail = pd.read_parquet(ROOT / v29_record["ranking_values"])
    ids = v29_detail[ID].to_numpy()
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("The recorded v29 values are not in the ranking departure order.")
    v9, residual, v29_recorded, mix_recorded = (v29_detail[c].to_numpy(dtype=float) for c in ["v9", "residual", "v29", "committee_mix"])
    runtime = {**config, "matches_dir": config["matches_dirs"]["v11"], "residual_model": {"neighbour_features": False},
               "extra_feature_tables": {k: v for k, v in config["extra_feature_tables"].items() if k not in config["ensembles"]}}
    stands = pd.read_parquet(ROOT / config["matches_dirs"]["v11"] / "stands.parquet")
    table, _ = ranking_table(runtime, departures, stands, v9)
    if not np.array_equal(table[ID].to_numpy(), ids):
        raise ValueError("Ranking table order differs from the departures.")
    floor = float(pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / "2025-12.parquet", columns=["floor"])["floor"].iloc[0])
    if floor != -12.0:
        raise ValueError("Unexpected floor.")
    committee = table[config["committee_table"]].to_numpy(dtype=float)
    if not np.array_equal(committee, v29_detail["committee"].to_numpy(dtype=float)):
        raise ValueError("The committee table differs from v29's.")
    learners = {}
    used = sorted(e for e in config["ensembles"] if e in RECORDED or not args.control_only)
    for ensemble in used:
        table[ensemble], learners[ensemble] = ensemble_table(config["ensembles"][ensemble], ids)
    for ensemble in [e for e in used if e in RECORDED]:
        if float(np.max(np.abs(table[ensemble].to_numpy() - v29_detail[ensemble].to_numpy(dtype=float)))) > 1e-9:
            raise ValueError(f"The {ensemble} mean differs from v29's recorded one.")
    caches = {r["adsb_ranking_table_sha256"] for e in used for r in learners[e].values()}
    if len(caches) != 1 or caches != {v29_record["adsb_ranking_table_sha256"]}:
        raise ValueError(f"The ensembles' seeds used different ADS-B ranking tables, or not v29's: {caches}.")
    for ensemble in [e for e in used if e in networks]:
        for leaf in learners[ensemble]:
            if read_json(ROOT / leaf / "refit12" / "run.json")["inputs"]["tokens"]["files"] != networks[ensemble]["tokens_manifest_content"]:
                raise ValueError(f"{leaf}: the refit used token files whose content differs from the declared tokens.")
    if sha256_file(ROOT / v29_record["output"]) != v29_record["receipt"]["sha256"]:
        raise ValueError("The uploaded v29 file changed.")
    uploaded = pd.read_parquet(ROOT / v29_record["output"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    blends = None

    def recipe(spec: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
        nonlocal blends
        column = f"committee_mix:{label}"
        table[column] = mixed(committee, weighted(table, spec["network"]), float(spec["network_share"]))
        blends = lirf_blends(config, table, {**config["v25_recipe"], "lirf": lirf_spec["lirf"], "lirf_network": lirf_spec["lirf_network"]}, "lctx_b100_up")
        values, _ = candidate_values(table, v9, residual, spec_for(config, column, blends), floor)
        return values, table[column].to_numpy(dtype=float)

    control, control_mix = recipe(control_spec, "v29")
    reproduction = {"committee_mix_max_abs": float(np.nanmax(np.abs(control_mix - mix_recorded))), "values_max_abs": float(np.max(np.abs(control - v29_recorded))),
                    "integers_equal_uploaded": bool(np.array_equal(np.rint(control).astype(np.int32), uploaded))}
    print({"control_reproduction": reproduction}, flush=True)
    if (not np.array_equal(np.isnan(control_mix), np.isnan(mix_recorded)) or reproduction["committee_mix_max_abs"] > 1e-9 or reproduction["values_max_abs"] > 1e-6
            or not reproduction["integers_equal_uploaded"]):
        raise ValueError(f"The rebuilt control does not reproduce v29: {reproduction}")
    if args.control_only:
        return
    values, candidate_mix = recipe(candidate, name)
    exported = np.rint(values).astype(np.int32)
    airports = table["airport"].to_numpy()
    lirf = airports == "LIRF"
    if not np.array_equal(exported[lirf], uploaded[lirf]):
        raise ValueError("A LIRF row differs from the uploaded v29.")
    scope = v29_detail["scope"].to_numpy()
    run_dir = ROOT / "runs/adsb_ctx" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    lirf_columns = {c: v29_detail[c].to_numpy(dtype=float) for c in v29_detail.columns if str(c).startswith("lirf_mix:lctx_b100_up")}
    detail = pd.DataFrame({ID: ids, "airport": airports, "month": row_month, "scope": scope, "v9": v9, "residual": residual,
                           "committee": committee, config["lirf_table"]: table[config["lirf_table"]].to_numpy(dtype=float), **{e: table[e].to_numpy(dtype=float) for e in used},
                           "committee_mix_v29": control_mix, "committee_mix": candidate_mix, **lirf_columns, "v29": control, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite a versioned submission.")
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": decl["experiment_id"], "declaration_sha256": object_hash(decl),
                  "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": name, "selected_by": "declared rule", "candidate": candidate,
                  "round_b": {"declaration_sha256": object_hash(decl_b), "selection_sha256": sha256_file(ROOT / ROUND_B_SELECTION), "selected_candidate": "lctx_b100_up"},
                  "ship_check": {"declaration": CHECK, "months": {m: {"thinned_change_pct": r["regimes"]["thinned"]["change_pct"], "passes": r["gate_month"]["passes"]}
                                                                  for m, r in regime.items()}},
                  "hardware": {"ens_ctx_e32": networks["ens_ctx_e32"]["hardware"]},
                  "ensembles": {e: {k: config["ensembles"][e][k] for k in ["variant_sha256", "code_sha256", "config_sha256", "seeds"]} for e in used},
                  "learners": learners, "recipe": "v29 with the 32-epoch round's committee mix (g_thirds)",
                  "incumbent": {"submission": "vigorous-whistle_v29.parquet", "official_rmse_seconds": 229.6196},
                  "external_data": decl["external_data"], "audit_used": False, "floor_seconds": floor}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)),
              "selected_candidate": name, "candidate": candidate, "learners": learners, "control_reproduction": reproduction, "lirf_rows_unchanged": True,
              "adsb_ranking_table_sha256": next(iter(caches)), "token_content_sha256": {e: networks[e]["tokens_manifest_content"] for e in used if e in networks},
              "ranking_values": (run_dir / "ranking_values.parquet").relative_to(ROOT).as_posix(), "ranking_values_sha256": sha256_file(run_dir / "ranking_values.parquet"),
              "network_tables": {"correlation_e32_vs_pool": float(np.corrcoef(table["ens_ctx_e32"].to_numpy(dtype=float)[~lirf],
                                                                              weighted(table, control_spec["network"])[~lirf])[0, 1])},
              "versus_v29": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds_all_rows": float(np.sqrt(np.mean(delta ** 2))),
                             "rmse_of_change_seconds_non_lirf": float(np.sqrt(np.mean(delta[~lirf] ** 2))), "mean_change_seconds_non_lirf": float(delta[~lirf].mean()),
                             "absolute_change_quantiles_seconds_non_lirf": {str(q): float(np.quantile(np.abs(delta[~lirf]), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                             "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))},
                             "rms_change_by_month": {m: float(np.sqrt(np.mean(delta[row_month == m] ** 2))) for m in RANKING_MONTHS}},
              "seconds": round(time.time() - started, 1)}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["output", "selected_candidate"]}, report["versus_v29"]["changed_rows"], round(report["versus_v29"]["rmse_of_change_seconds_non_lirf"], 3))


if __name__ == "__main__":
    main()
