"""Export round A's selected candidate (configs/adsb_ctx_v2.json) as the next versioned submission: v27 with a new committee-mix network table and share.

Steps (declared in configs/adsb_ctx_v2.json, "ship"):
1. Every ensemble's ranking column is the mean of its declared seeds' twelve-month refits (tools/export_adsb_nn2.ensemble_table): ens_nn,
   ens_new and ens_nnl as recorded for v25-v27, ens_ctx as recorded for v27, ens_ctx_e24 the sixteen round-A seeds; each refit record
   must carry the declared variant, code, configuration and seed and its saved model the recorded hash; all seeds one ADS-B ranking table;
   every context seed's token files the declared content.
2. v27 is rebuilt from its recorded components: v9 and the residual from its export detail (runs/adsb_ctx/export_v27/ranking_values.parquet,
   hash-checked against docs/adsb_ctx/export.json), the label-free ranking table (tools/export_adsb_v11.ranking_table) for the committee
   and LIRF tables; the committee and every recorded ensemble must equal v27's detail; with the development scorer's code path (mixed,
   lirf_blends with v25's LIRF weights, spec_for, candidate_values) the control's committee mix must equal v27's recorded one to 1e-9 s,
   its values the recorded v27 to 1e-6 s and the uploaded v27 integers exactly.
3. The selected candidate's network table and share replace the control's on the same code path; floor; nearest-integer export through
   the frozen submission contract. LIRF rows must equal the uploaded v27 (the committee mix does not reach them).
The coverage-regime ship check (configs/adsb_ctx_regime_check_v2.json) must have passed on both thinned months. Nothing is uploaded here.
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
CHECK = "configs/adsb_ctx_regime_check_v2.json"
RECORDED = ["ens_ctx", "ens_new", "ens_nn", "ens_nnl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_v2.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_ctx/dev2/selection.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("docs/adsb_ctx/export.json"), help="v27's export record")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx/export_v2.json"))
    parser.add_argument("--control-only", action="store_true", help="stop after the v27 control reproduction (no new ensembles, no export)")
    args = parser.parse_args()
    started = time.time()
    decl = read_json(ROOT / args.declaration)
    if decl["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    selection = read_json(ROOT / args.selection)
    if selection["declaration_sha256"] != object_hash(decl) or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name = selection["selected_candidate"]
    if not selection["candidates"][name]["eligible"] or name == "control":
        raise ValueError("The selected candidate is not eligible.")
    candidate, control_spec = decl["candidates"][name], decl["candidates"]["control"]
    if control_spec != {"network": decl["control"]["network_table"], "network_share": decl["control"]["network_share"]}:
        raise ValueError("The control is not v27's committee mix.")
    check = read_json(ROOT / CHECK)
    regime = {m: read_json(ROOT / f"docs/adsb_ctx/regime_check2_{m}.json") for m in REGIME_MONTHS} if not args.control_only else {}
    if not args.control_only:
        if check["candidate"]["name"] != name or any(r["declaration_sha256"] != sha256_file(ROOT / CHECK) or not r["gate_month"]["passes"] for r in regime.values()):
            raise ValueError("The coverage-regime ship check has not passed on both thinned months for this candidate.")
    config = merged_config(decl)
    check_parents(config)
    v25_recipe = config["v25_recipe"]

    # recorded components of v27
    v27_record = read_json(ROOT / args.incumbent)
    if sha256_file(ROOT / v27_record["ranking_values"]) != v27_record["ranking_values_sha256"]:
        raise ValueError("The recorded v27 ranking values changed.")
    v27_detail = pd.read_parquet(ROOT / v27_record["ranking_values"])
    ids = v27_detail[ID].to_numpy()
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("The recorded v27 values are not in the ranking departure order.")
    v9, residual, v27_recorded, mix_recorded = (v27_detail[c].to_numpy(dtype=float) for c in ["v9", "residual", "v27", "committee_mix"])
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
    if not np.array_equal(committee, v27_detail["committee"].to_numpy(dtype=float)):
        raise ValueError("The committee table differs from v27's.")
    learners = {}
    used = sorted(e for e in config["ensembles"] if e in RECORDED or not args.control_only)
    for ensemble in used:
        table[ensemble], learners[ensemble] = ensemble_table(config["ensembles"][ensemble], ids)
    for ensemble in [e for e in used if e in RECORDED]:
        if float(np.max(np.abs(table[ensemble].to_numpy() - v27_detail[ensemble].to_numpy(dtype=float)))) > 1e-9:
            raise ValueError(f"The {ensemble} mean differs from v27's recorded one.")
    caches = {r["adsb_ranking_table_sha256"] for e in used for r in learners[e].values()}
    if len(caches) != 1 or caches != {v27_record["adsb_ranking_table_sha256"]}:
        raise ValueError(f"The ensembles' seeds used different ADS-B ranking tables, or not v27's: {caches}.")
    # each worker rebuilt its tokens (manifests differ in timestamps); the recorded per-file content hashes must equal the declaration's
    for ensemble in [e for e in used if e in decl["networks"]]:
        for leaf in learners[ensemble]:
            if read_json(ROOT / leaf / "refit12" / "run.json")["inputs"]["tokens"]["files"] != decl["networks"][ensemble]["tokens_manifest_content"]:
                raise ValueError(f"{leaf}: the refit used token files whose content differs from the declared tokens.")
    if sha256_file(ROOT / v27_record["output"]) != v27_record["receipt"]["sha256"]:
        raise ValueError("The uploaded v27 file changed.")
    uploaded = pd.read_parquet(ROOT / v27_record["output"]).set_index(ID)[TARGET].reindex(ids).to_numpy()

    def recipe(spec: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
        column = f"committee_mix:{label}"
        table[column] = mixed(committee, weighted(table, spec["network"]), float(spec["network_share"]))
        values, _ = candidate_values(table, v9, residual, spec_for(config, column, lirf_blends(config, table, v25_recipe, "v27")), floor)
        return values, table[column].to_numpy(dtype=float)

    control, control_mix = recipe(control_spec, "control")
    reproduction = {"committee_mix_max_abs": float(np.nanmax(np.abs(control_mix - mix_recorded))), "values_max_abs": float(np.max(np.abs(control - v27_recorded))),
                    "integers_equal_uploaded": bool(np.array_equal(np.rint(control).astype(np.int32), uploaded))}
    print({"control_reproduction": reproduction}, flush=True)
    if (not np.array_equal(np.isnan(control_mix), np.isnan(mix_recorded)) or reproduction["committee_mix_max_abs"] > 1e-9 or reproduction["values_max_abs"] > 1e-6
            or not reproduction["integers_equal_uploaded"]):
        raise ValueError(f"The rebuilt control does not reproduce v27: {reproduction}")
    if args.control_only:
        return
    values, mix = recipe(candidate, name)
    exported = np.rint(values).astype(np.int32)
    airports = table["airport"].to_numpy()
    lirf = airports == "LIRF"
    if not np.array_equal(exported[lirf], uploaded[lirf]):
        raise ValueError("LIRF rows differ from the uploaded v27.")
    run_dir = ROOT / "runs/adsb_ctx" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    detail = pd.DataFrame({ID: ids, "airport": airports, "month": row_month, "scope": v27_detail["scope"].to_numpy(), "v9": v9, "residual": residual,
                           "committee": committee, **{e: table[e].to_numpy(dtype=float) for e in used},
                           "committee_mix_v27": control_mix, "committee_mix": mix, "v27": control, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite a versioned submission.")
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": decl["experiment_id"], "declaration_sha256": object_hash(decl),
                  "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": name, "selected_by": "declared rule", "candidate": candidate,
                  "ship_check": {"declaration": CHECK, "months": {m: {"thinned_change_pct": r["regimes"]["thinned"]["change_pct"], "passes": r["gate_month"]["passes"]}
                                                                  for m, r in regime.items()}},
                  "ensembles": {e: {k: config["ensembles"][e][k] for k in ["variant_sha256", "code_sha256", "config_sha256", "seeds"]} for e in used},
                  "learners": learners, "recipe": "v27 with the selected committee-mix network table and share", "incumbent": decl["control"]["official"],
                  "external_data": decl["external_data"], "audit_used": False, "floor_seconds": floor}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)),
              "selected_candidate": name, "candidate": candidate, "learners": learners, "control_reproduction": reproduction, "lirf_rows_unchanged": True,
              "adsb_ranking_table_sha256": next(iter(caches)), "token_content_sha256": {e: decl["networks"][e]["tokens_manifest_content"] for e in used if e in decl["networks"]},
              "ranking_values": (run_dir / "ranking_values.parquet").relative_to(ROOT).as_posix(), "ranking_values_sha256": sha256_file(run_dir / "ranking_values.parquet"),
              "network_vs_v27_network": {"correlation_non_lirf": float(np.corrcoef(weighted(table, candidate["network"])[~lirf], weighted(table, control_spec["network"])[~lirf])[0, 1])},
              "versus_v27": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                             "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))},
                             "rms_change_by_month": {m: float(np.sqrt(np.mean(delta[row_month == m] ** 2))) for m in RANKING_MONTHS}},
              "seconds": round(time.time() - started, 1)}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["output", "selected_candidate"]}, report["versus_v27"]["changed_rows"], round(report["versus_v27"]["rmse_of_change_seconds"], 3))


if __name__ == "__main__":
    main()
