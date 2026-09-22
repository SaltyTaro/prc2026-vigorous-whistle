"""Export the adsb_v25 selected candidate as the next versioned submission (v23): v22's recipe with only the residual stage replaced.

Steps (all declared in configs/adsb_v25_v1.json, "ship"):
1. v9 on the ranking reproduced exactly as v22 did: v9 from scratch (frozen v7, January-November LIRF mixture, equal to the uploaded v9 after rounding),
   then the twelve-month v7 of runs/adsb_v24/v7/refit on every clock-present row (v9 = max(v7, floor)); the LIRF fill population and missing-clock rows
   keep the reproduced v9. The result must equal v22's recorded ranking v9 exactly.
2. Every member variant of the selected candidate's residual is refit on ALL twelve months' in-scope rows (scratch/v24 frames, cross-fit v9) with rounds from
   the inner rule applied once per refit (tools/run_adsb_v25.fit_variant, unchanged): one model for an unweighted member, one model per ranking target month
   (January, July) for a weighted member; each ranking row takes the residual of its own calendar month's model; a mean candidate averages its members.
3. Blend exactly as v22 (0.7 ens_v20_bag_q_all on non-LIRF rows; 0.85 ens_v22_lirf_q on matched LIRF non-population rows; 0.25 ens_v22_lirf_q on unmatched
   LIRF non-population rows; the twelve-month ranking tables of v23); floor at -12; nearest-integer export through the frozen submission contract.
   Every row outside the residual scope must equal the uploaded v22 after rounding.
The two stages can run on different hosts: `--stage fit` fits and saves the member models (runs/adsb_v25/export_v<N>/fit_record.json) and stops;
`--stage assemble` reloads exactly those saved models instead of fitting and does everything else; the default runs both in one process.
Nothing is uploaded here.
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
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_adsb.features import DEPARTURE_COLUMNS
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_submission.contract import export_submission
from export_adsb_v11 import ranking_table
from run_adsb_v11 import candidate_values, check_parents, scope_mask
from run_adsb_v24 import ALL_MONTHS, adsb_features, load_frame, month_table
from run_adsb_v25 import fit_variant, predict_variant
import lightgbm as lgb
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population

RANKING_MONTHS = ["2026-01", "2026-07"]
MODEL_CODE = ["tools/export_adsb_v25.py", "tools/run_adsb_v25.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "tools/export_adsb_v11.py", "src/prc2026_adsb/features.py"]


def save_model(model: dict, path_stem: Path) -> str:
    if model["kind"] == "lightgbm":
        path = path_stem.with_suffix(".txt")
        model["model"].save_model(str(path), num_iteration=model["rounds"])
    else:
        path = path_stem.with_suffix(".cbm")
        model["model"].save_model(str(path))
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v25_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_v25/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v25/export.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--threads", type=int, default=None, help="defaults to the declared export thread count")
    parser.add_argument("--stage", choices=["fit", "assemble", "both"], default="both", help="fit: save the member models and stop; assemble: reload them and export")
    args = parser.parse_args()
    started = time.time()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    config_sha256 = object_hash(config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != config_sha256 or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name = selection["selected_candidate"]
    if not selection["candidates"][name]["eligible"] or name in config["check_candidates"]:
        raise ValueError("The selected candidate is not eligible.")
    threads = args.threads or config["threads"]["export"]
    candidate = config["candidates"][name]
    members = candidate["residual"] if isinstance(candidate["residual"], list) else [candidate["residual"]]
    scope = config["residual_scope"]
    matches_root = ROOT / config["matches_dirs"]["v11"]
    config["matches_dir"] = config["matches_dirs"]["v11"]  # export_adsb_v11.ranking_table reads the residual's match directory from here
    config["residual_model"]["neighbour_features"] = True  # runtime switch of export_adsb_v11.ranking_table; the declared `nb` feature set lists the nb_* columns (both keys are added after the declaration's hash is taken)
    stands = pd.read_parquet(matches_root / "stands.parquet")
    mixture_config = read_json(ROOT / config["parents"]["lirf_mixture_config"]["path"])
    if object_hash(mixture_config) != read_json(ROOT / config["parents"]["lirf_mixture_selection"]["path"])["config_sha256"]:
        raise ValueError("The v9 declaration does not match its selection receipt.")
    # 2. Residual members refit on all twelve months (stage fit): one model per member, or per ranking target month for a weighted member.
    run_dir = ROOT / "runs/adsb_v25" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_path = run_dir / "fit_record.json"
    if args.stage in ("fit", "both"):
        if fit_path.exists():
            raise ValueError(f"Preserve the existing fit record: {fit_path}")
        train = pd.concat([month_table(load_frame(config, m, "cross_fit"), adsb_features(m, matches_root, ROOT / args.cache), m) for m in ALL_MONTHS], ignore_index=True)
        fitted = {}
        for member in members:
            variant = config["residual_variants"][member]
            targets = RANKING_MONTHS if variant.get("weights") else ["all"]
            fitted[member] = {}
            for target in targets:
                model = fit_variant(train, config, variant, target if target != "all" else "2025-01", threads)  # the target month only matters through the weights
                file = save_model(model, run_dir / f"residual_{member}_{target}")
                fitted[member][target] = {"file": file, "sha256": sha256_file(run_dir / file), "kind": model["kind"], "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"],
                                          "training_rows": model["rows"], "weights": model["weights"], "weight_sum": model["weight_sum"], "weight_by_distance_rows": model["weight_by_distance"],
                                          "target_month": target, "categories": model["categories"], "feature_set": model["feature_set"], "scope": model["scope"], "seconds": model["seconds"]}
                print(member, target, {k: fitted[member][target][k] for k in ["rounds", "training_rows", "seconds"]}, flush=True)
        write_json(fit_path, {"created_at_utc": utc_now(), "config_sha256": config_sha256, "selection_sha256": sha256_file(ROOT / args.selection), "candidate": name, "members": members,
                              "threads": threads, "training_rows": int(len(train)), "models": fitted, "model_code_sha256": object_hash({p: sha256_file(ROOT / p) for p in MODEL_CODE})})
        if args.stage == "fit":
            print({"stage": "fit", "record": fit_path.relative_to(ROOT).as_posix(), "models": {m: {t: f["rounds"] for t, f in fitted[m].items()} for m in members}, "seconds": round(time.time() - started, 1)})
            return
    fit_record = read_json(fit_path)
    if fit_record["config_sha256"] != config_sha256 or fit_record["selection_sha256"] != sha256_file(ROOT / args.selection) or fit_record["candidate"] != name or fit_record["members"] != members:
        raise ValueError("The fit record does not belong to this declaration, selection or candidate.")
    fitted = fit_record["models"]
    threads = fit_record["threads"]
    # 1. v9 exactly as v22: from scratch, then the twelve-month v7 on clock-present rows.
    fitting = movements([f"2025-{m:02}" for m in range(1, 12)])
    v9_model, v9_component, v9_summary = fit_population_model(fitting, mixture_config)
    floor = v9_summary["floor_seconds"]
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    ids = departures[ID].to_numpy()
    v7_frozen = pd.read_parquet(ROOT / config["parents"]["v7_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    _, _, _, interval, v9_mixture, index = score_population(v9_model, v9_component, ranking, ids, mixture_config)
    v9_frozen, _ = apply_candidate(v7_frozen, index, interval, v9_mixture, mixture_config["candidates"]["mixture_75"]["mixture_weight"], mixture_config["mixture"]["schedule_interval_threshold_seconds"], floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v9_frozen).astype(np.int32), uploaded_v9):
        raise ValueError("The uploaded v9 does not reproduce from scratch.")
    population = np.zeros(len(ids), dtype=bool)
    population[index] = True
    run = ROOT / "runs/adsb_v24/v7/refit"
    v7_record = read_json(run / "run.json")
    if v7_record["stage"] != "refit_twelve_months" or v7_record["held_out_months"] or v7_record["predicted_ids_sha256"] != identity_hash(pd.Series(ids)) or v7_record["fit_months"] != ALL_MONTHS:
        raise ValueError("The twelve-month v7 refit does not match the ranking or the declaration.")
    v7_new = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    clock = np.isfinite(v7_new)
    if clock.sum() != v7_record["predicted_clock_rows"] or (clock & population).any():
        raise ValueError("Twelve-month v7 coverage disagrees with the clock-present route.")
    v9 = np.where(clock, np.maximum(v7_new, floor), v9_frozen)
    v22_values = pd.read_parquet(ROOT / config["parents"]["v22_ranking_values"]["path"]).set_index(ID).reindex(ids)
    if float(np.max(np.abs(v22_values["v9"].to_numpy(dtype=float) - v9))) > 1e-9:
        raise ValueError("v9 differs from v22's recorded ranking v9.")
    v7_report = {"run_sha256": sha256_file(run / "run.json"), "predictions_sha256": sha256_file(run / "ranking_predictions.parquet"), "clock_rows": int(clock.sum()), "clock_fit_rows": v7_record["clock_fit_rows"],
                 "component_iterations": v7_record["component_iterations"], "code_sha256": v7_record["code"]["sha256"]}
    table, _ = ranking_table(config, departures, stands, v9)
    if not np.array_equal(table[ID].to_numpy(), ids):
        raise ValueError("Ranking table order differs from the departures.")
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    if set(row_month) != set(RANKING_MONTHS):
        raise ValueError("Ranking rows are not the two declared months.")
    member_residuals = {}
    for member in members:
        residual = np.zeros(len(ids))
        for target, spec in fitted[member].items():
            path = run_dir / spec["file"]
            if sha256_file(path) != spec["sha256"]:
                raise ValueError(f"Saved model {spec['file']} differs from the fit record.")
            if spec["kind"] == "lightgbm":
                loaded = lgb.Booster(model_file=str(path))
            else:
                from catboost import CatBoostRegressor
                loaded = CatBoostRegressor()
                loaded.load_model(str(path))
            model = {"kind": spec["kind"], "model": loaded, "categories": spec["categories"], "rounds": spec["rounds"], "scope": spec["scope"], "feature_set": spec["feature_set"]}
            rows = np.ones(len(ids), dtype=bool) if target == "all" else row_month == target
            values = predict_variant(model, table, config)
            residual[rows] = values[rows]
        member_residuals[member] = residual
    residual = np.mean([member_residuals[m] for m in members], axis=0)
    inside = scope_mask(table, scope)
    residual[~inside] = 0.0
    # 3. Blends, floor, export.
    item = {"weight": 1.0, "scope": scope, "blend_with": config["blend_with"], "extra_blends": config["extra_blends"]}
    values, touched = candidate_values(table, v9, residual, item, floor)
    uploaded_v22 = pd.read_parquet(ROOT / config["parents"]["v22_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    exported = np.rint(values).astype(np.int32)
    lirf_missing_clock = table["airport"].eq("LIRF").to_numpy() & ~clock
    if not np.array_equal(exported[~inside], uploaded_v22[~inside]):
        raise ValueError("Rows outside the residual scope differ from the uploaded v22.")
    if not np.array_equal(exported[population], uploaded_v9[population]) or not np.array_equal(exported[lirf_missing_clock], uploaded_v22[lirf_missing_clock]):
        raise ValueError("Rows the declaration keeps fixed (LIRF fill population, missing-clock LIRF rows) differ from the uploaded values.")
    record = {"candidate": name, "members": members, "models": fitted, "scope": scope, "feature_set": config["residual_variants"][members[0]]["feature_set"], "matches_dir": config["matches_dirs"]["v11"],
              "blend_with": config["blend_with"], "extra_blends": config["extra_blends"], "weight": 1.0, "floor_seconds": floor, "v9_provenance": "cross_fit (v22)",
              "training_months": [f"{m}:cross_fit" for m in ALL_MONTHS], "ranking_months": RANKING_MONTHS, "features": list(config["residual_model"]["feature_sets"]["nb"])}
    write_json(run_dir / "residual_model.json", record)
    detail = pd.DataFrame({ID: ids, "airport": table["airport"].to_numpy(), "month": row_month, "v9_frozen": v9_frozen, "v9": v9, "residual": residual, "in_scope": inside, "touched": touched, f"v{args.version}": values})
    for member in members:
        detail[f"residual:{member}"] = member_residuals[member]
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    model_hashes = sorted(f["sha256"] for m in fitted.values() for f in m.values())
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": config_sha256, "selection_sha256": sha256_file(ROOT / args.selection),
                  "selected_candidate": name, "v9_provenance": "cross_fit (v22)", "twelve_month_v7": v7_report,
                  "residual": {"members": members, "models": {m: {t: {k: f[k] for k in ["file", "sha256", "kind", "rounds", "training_rows", "weight_sum"]} for t, f in fitted[m].items()} for m in members},
                               "feature_set": record["feature_set"], "matches_dir": record["matches_dir"], "training_months": record["training_months"], "record_sha256": sha256_file(run_dir / "residual_model.json")},
                  "residual_models_sha256": model_hashes, "residual_model_sha256": object_hash(model_hashes), "rounds": {m: {t: f["rounds"] for t, f in fitted[m].items()} for m in members},
                  "training_rows": fitted[members[0]][list(fitted[members[0]])[0]]["training_rows"], "weight": 1.0, "scope": scope, "feature_set": record["feature_set"],
                  "blend_with": config["blend_with"], "extra_blends": config["extra_blends"], "ranking_rows_in_scope": int(inside.sum()), "ranking_matched_rows": int(table["adsb_matched"].eq(1).sum()),
                  "stands": int(len(stands)), "external_data": config["external_data"], "parents": config["parents"], "incumbent": config["incumbent"],
                  "model_code_sha256": object_hash({p: sha256_file(ROOT / p) for p in MODEL_CODE}), "audit_used": False, "native_fit_count": sum(len(m) for m in fitted.values()), "threads": threads}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded_v22
    airports = table["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)), "selected_candidate": name,
              "twelve_month_v7": v7_report, "v9_equal_to_v22": True, "rows_in_scope": int(inside.sum()), "matched_rows": int(table["adsb_matched"].eq(1).sum()), "rows_outside_scope_equal_v22": True,
              "population_rows_unchanged": True, "lirf_missing_clock_rows_unchanged": True,
              "threads": threads, "residual": {m: {t: {k: f[k] for k in ["kind", "rounds", "inner_best_iterations", "training_rows", "weight_sum", "weight_by_distance_rows", "seconds"]} for t, f in fitted[m].items()} for m in members},
              "versus_v22": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]}, "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))},
                             "rms_change_by_month": {m: float(np.sqrt(np.mean(delta[row_month == m] ** 2))) for m in RANKING_MONTHS}},
              "seconds": round(time.time() - started, 1)}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "changed_vs_v22": report["versus_v22"]["changed_rows"], "rms_change": round(report["versus_v22"]["rmse_of_change_seconds"], 3),
           "max_change": report["versus_v22"]["max_absolute_change_seconds"], "minimum": receipt["minimum_prediction_seconds"], "maximum": receipt["maximum_prediction_seconds"]})


if __name__ == "__main__":
    main()
