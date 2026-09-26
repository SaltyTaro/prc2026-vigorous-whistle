"""Export round B's selected candidate (configs/adsb_ctx_lirf_v1.json) on top of the incumbent v28: v28 with new LIRF non-population blends.

Round B changes only the two LIRF non-population blends (their network table and (v9, tree, net) weights); round A (v28) changed only the
committee mix, which does not reach LIRF rows. The declaration's ship step applies both selections, each exactly as selected:
1. Every ensemble's ranking column is the mean of its declared seeds' twelve-month refits (tools/export_adsb_nn2.ensemble_table): the
   inherited ens_nn / ens_new / ens_nnl, ens_ctx and ens_ctx_e24 as recorded for v28, and ens_ctx_lirf the sixteen round-B seeds; each
   refit record must carry the declared variant, code, configuration and seed and its saved model the recorded hash; all seeds one ADS-B
   ranking table (v28's); every context seed's token files the declared content.
2. v28 is rebuilt from its recorded components (runs/adsb_ctx/export_v28/ranking_values.parquet, hash-checked against
   docs/adsb_ctx/export_v2.json) with the development scorer's code path: round A's pool_s75 committee mix and v25's LIRF blends. The
   committee mix must equal v28's recorded one to 1e-9 s, the values the recorded v28 to 1e-6 s and the uploaded v28 integers exactly.
3. Round B's selected LIRF blends replace v25's on the same code path; floor; nearest-integer export through the frozen submission
   contract. Every row outside the two LIRF non-population scopes must equal the uploaded v28.
The coverage-regime ship check (configs/adsb_ctx_lirf_regime_check_v1.json) must have passed on both thinned months. Nothing is uploaded here.
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
CHECK = "configs/adsb_ctx_lirf_regime_check_v1.json"
ROUND_A = "configs/adsb_ctx_v2.json"
ROUND_A_SELECTION = "runs/adsb_ctx/dev2/selection.json"
RECORDED = ["ens_ctx", "ens_ctx_e24", "ens_new", "ens_nn", "ens_nnl"]
LIRF_SCOPES = ("lirf_nonpop_matched", "lirf_nonpop_unmatched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_lirf_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_ctx/dev_lirf/selection.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("docs/adsb_ctx/export_v2.json"), help="v28's export record")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx_lirf/export.json"))
    parser.add_argument("--control-only", action="store_true", help="stop after the v28 control reproduction (no ens_ctx_lirf, no export)")
    args = parser.parse_args()
    started = time.time()
    decl, decl_a = read_json(ROOT / args.declaration), read_json(ROOT / ROUND_A)
    if decl["audit_used"] or decl_a["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    selection_a = read_json(ROOT / ROUND_A_SELECTION)
    if selection_a["declaration_sha256"] != object_hash(decl_a) or selection_a["selected_candidate"] != "pool_s75":
        raise ValueError("Round A's selection is not v28's pool_s75.")
    committee_spec = decl_a["candidates"]["pool_s75"]
    control_lirf = {"lirf": decl["control"]["lirf"], "lirf_network": decl["control"]["lirf_network"]}
    name, candidate = None, None
    regime = {}
    if not args.control_only:
        selection = read_json(ROOT / args.selection)
        if selection["declaration_sha256"] != object_hash(decl) or not selection["selected_candidate"]:
            raise ValueError("The selection does not belong to this declaration or selected nothing.")
        name = selection["selected_candidate"]
        if not selection["candidates"][name]["eligible"] or name == "control":
            raise ValueError("The selected candidate is not eligible.")
        candidate = decl["candidates"][name]
        check = read_json(ROOT / CHECK)
        regime = {m: read_json(ROOT / f"docs/adsb_ctx_lirf/regime_check_{m}.json") for m in REGIME_MONTHS}
        if check["candidate"]["name"] != name or any(r["declaration_sha256"] != sha256_file(ROOT / CHECK) or not r["gate_month"]["passes"] for r in regime.values()):
            raise ValueError("The coverage-regime ship check has not passed on both thinned months for this candidate.")
    config = merged_config(decl_a)
    config_b = merged_config(decl)
    for new in decl["new_networks"]:
        config["extra_feature_tables"][new], config["ensembles"][new] = config_b["extra_feature_tables"][new], config_b["ensembles"][new]
    networks = {**decl_a["networks"], **{n: decl["networks"][n] for n in decl["new_networks"]}}
    check_parents(config)
    v25_recipe = config["v25_recipe"]
    if control_lirf != {"lirf": v25_recipe["lirf"], "lirf_network": v25_recipe["lirf_network"]}:
        raise ValueError("The control's LIRF blends are not v25's (those of v27 and v28).")

    # recorded components of v28
    v28_record = read_json(ROOT / args.incumbent)
    if v28_record["selected_candidate"] != "pool_s75" or sha256_file(ROOT / v28_record["ranking_values"]) != v28_record["ranking_values_sha256"]:
        raise ValueError("The recorded v28 ranking values changed.")
    v28_detail = pd.read_parquet(ROOT / v28_record["ranking_values"])
    ids = v28_detail[ID].to_numpy()
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("The recorded v28 values are not in the ranking departure order.")
    v9, residual, v28_recorded, mix_recorded = (v28_detail[c].to_numpy(dtype=float) for c in ["v9", "residual", "v28", "committee_mix"])
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
    if not np.array_equal(committee, v28_detail["committee"].to_numpy(dtype=float)):
        raise ValueError("The committee table differs from v28's.")
    learners = {}
    used = sorted(e for e in config["ensembles"] if e in RECORDED or not args.control_only)
    for ensemble in used:
        table[ensemble], learners[ensemble] = ensemble_table(config["ensembles"][ensemble], ids)
    for ensemble in [e for e in used if e in RECORDED]:
        if float(np.max(np.abs(table[ensemble].to_numpy() - v28_detail[ensemble].to_numpy(dtype=float)))) > 1e-9:
            raise ValueError(f"The {ensemble} mean differs from v28's recorded one.")
    caches = {r["adsb_ranking_table_sha256"] for e in used for r in learners[e].values()}
    if len(caches) != 1 or caches != {v28_record["adsb_ranking_table_sha256"]}:
        raise ValueError(f"The ensembles' seeds used different ADS-B ranking tables, or not v28's: {caches}.")
    for ensemble in [e for e in used if e in networks]:
        for leaf in learners[ensemble]:
            if read_json(ROOT / leaf / "refit12" / "run.json")["inputs"]["tokens"]["files"] != networks[ensemble]["tokens_manifest_content"]:
                raise ValueError(f"{leaf}: the refit used token files whose content differs from the declared tokens.")
    if sha256_file(ROOT / v28_record["output"]) != v28_record["receipt"]["sha256"]:
        raise ValueError("The uploaded v28 file changed.")
    uploaded = pd.read_parquet(ROOT / v28_record["output"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    table["committee_mix:v28"] = mixed(committee, weighted(table, committee_spec["network"]), float(committee_spec["network_share"]))

    def recipe(lirf: dict, label: str) -> np.ndarray:
        blends = lirf_blends(config, table, {**v25_recipe, "lirf": lirf["lirf"], "lirf_network": lirf["lirf_network"]}, label)
        values, _ = candidate_values(table, v9, residual, spec_for(config, "committee_mix:v28", blends), floor)
        return values

    control = recipe(control_lirf, "v28")
    control_mix = table["committee_mix:v28"].to_numpy(dtype=float)
    reproduction = {"committee_mix_max_abs": float(np.nanmax(np.abs(control_mix - mix_recorded))), "values_max_abs": float(np.max(np.abs(control - v28_recorded))),
                    "integers_equal_uploaded": bool(np.array_equal(np.rint(control).astype(np.int32), uploaded))}
    print({"control_reproduction": reproduction}, flush=True)
    if (not np.array_equal(np.isnan(control_mix), np.isnan(mix_recorded)) or reproduction["committee_mix_max_abs"] > 1e-9 or reproduction["values_max_abs"] > 1e-6
            or not reproduction["integers_equal_uploaded"]):
        raise ValueError(f"The rebuilt control does not reproduce v28: {reproduction}")
    if args.control_only:
        return
    values = recipe(candidate, name)
    exported = np.rint(values).astype(np.int32)
    airports = table["airport"].to_numpy()
    scope = v28_detail["scope"].to_numpy()
    inside = np.isin(scope, LIRF_SCOPES)
    if not np.array_equal(exported[~inside], uploaded[~inside]):
        raise ValueError("A row outside the LIRF non-population scopes differs from the uploaded v28.")
    if (airports[inside] != "LIRF").any():
        raise ValueError("A LIRF non-population scope row is not at LIRF.")
    run_dir = ROOT / "runs/adsb_ctx" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    lirf_columns = {c: table[c].to_numpy(dtype=float) for c in table.columns if str(c).startswith("lirf_mix:")}
    detail = pd.DataFrame({ID: ids, "airport": airports, "month": row_month, "scope": scope, "v9": v9, "residual": residual,
                           "committee": committee, config["lirf_table"]: table[config["lirf_table"]].to_numpy(dtype=float), **{e: table[e].to_numpy(dtype=float) for e in used},
                           "committee_mix": control_mix, **lirf_columns, "v28": control, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite a versioned submission.")
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": decl["experiment_id"], "declaration_sha256": object_hash(decl),
                  "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": name, "selected_by": "declared rule", "candidate": candidate,
                  "round_a": {"declaration_sha256": object_hash(decl_a), "selection_sha256": sha256_file(ROOT / ROUND_A_SELECTION), "selected_candidate": "pool_s75"},
                  "ship_check": {"declaration": CHECK, "months": {m: {"thinned_change_pct": r["regimes"]["thinned"]["change_pct"], "passes": r["gate_month"]["passes"]}
                                                                  for m, r in regime.items()}},
                  "ensembles": {e: {k: config["ensembles"][e][k] for k in ["variant_sha256", "code_sha256", "config_sha256", "seeds"]} for e in used},
                  "learners": learners, "recipe": "v28 with round B's LIRF non-population blends", "incumbent": {"submission": "vigorous-whistle_v28.parquet",
                                                                                                                  "official_rmse_seconds": 230.0668},
                  "external_data": decl["external_data"], "audit_used": False, "floor_seconds": floor}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded
    lirf = airports == "LIRF"
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)),
              "selected_candidate": name, "candidate": candidate, "learners": learners, "control_reproduction": reproduction, "rows_outside_lirf_scopes_unchanged": True,
              "adsb_ranking_table_sha256": next(iter(caches)), "token_content_sha256": {e: networks[e]["tokens_manifest_content"] for e in used if e in networks},
              "ranking_values": (run_dir / "ranking_values.parquet").relative_to(ROOT).as_posix(), "ranking_values_sha256": sha256_file(run_dir / "ranking_values.parquet"),
              "lirf_network_vs_v28": {"correlation_lirf_nonpop": float(np.corrcoef(weighted(table, candidate["lirf_network"])[inside], weighted(table, control_lirf["lirf_network"])[inside])[0, 1])},
              "versus_v28": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds_all_rows": float(np.sqrt(np.mean(delta ** 2))),
                             "rmse_of_change_seconds_lirf": float(np.sqrt(np.mean(delta[lirf] ** 2))), "mean_change_seconds_lirf": float(delta[lirf].mean()),
                             "absolute_change_quantiles_seconds_scope": {str(q): float(np.quantile(np.abs(delta[inside]), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                             "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_scope": {s: float(np.sqrt(np.mean(delta[scope == s] ** 2))) for s in LIRF_SCOPES},
                             "rms_change_lirf_by_month": {m: float(np.sqrt(np.mean(delta[lirf & (row_month == m)] ** 2))) for m in RANKING_MONTHS}},
              "seconds": round(time.time() - started, 1)}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["output", "selected_candidate"]}, report["versus_v28"]["changed_rows"], round(report["versus_v28"]["rmse_of_change_seconds_lirf"], 3))


if __name__ == "__main__":
    main()
