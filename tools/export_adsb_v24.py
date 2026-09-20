"""Export the adsb_v24 selected candidate as the next versioned submission (v22): v21's recipe with the selected v9 provenance and residual.

Steps (all declared in configs/adsb_v24_v1.json, "ship"):
1. v9 on the ranking reproduced from scratch exactly as v21 did (frozen v7, January-November LIRF mixture classifier, floor); its rounding must equal the uploaded v9.
2. If the selected candidate uses cross-fit v9: the twelve-month v7 (tools/run_v7_dec.py --fold refit, mirrored to runs/adsb_v24/v7/refit) replaces v7 on every clock-present
   ranking row, v9 = max(v7, floor) there; missing-clock rows and the LIRF fill population keep the reproduced v9 exactly.
3. The selected residual variant is refit once on ALL twelve months' declared rows (development months with the selected v9 provenance, the eight others with cross-fit v9,
   all in scope matched_no_lirf) with rounds from the day-grouped five-fold inner rule (run_adsb_v11.fit_residual, unchanged).
4. Blend exactly as v21 (0.7 ens_v20_bag_q_all on every non-LIRF row; 0.85 ens_v22_lirf_q on matched LIRF non-population rows with ordinary clocks; both the twelve-month
   ranking tables of v23) plus any selected extra blend; floor at -12; nearest-integer export through the frozen submission contract.
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
from run_adsb_v11 import candidate_values, check_parents, fit_residual, predict_residual, scope_mask
from run_adsb_v24 import ALL_MONTHS, FRAME_OF, adsb_features, load_frame, month_table
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v24_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_v24/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v24/export.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    args = parser.parse_args()
    started = time.time()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name = selection["selected_candidate"]
    candidate = config["candidates"][name]
    variant = config["residual_variants"][candidate["residual"]]
    dir_key, feature_set, scope = variant["matches_dir"], variant["feature_set"], config["residual_scope"]
    matches_root = ROOT / config["matches_dirs"][dir_key]
    config["matches_dir"] = config["matches_dirs"][dir_key]  # export_adsb_v11.ranking_table reads the residual's match directory from here
    stands = pd.read_parquet(matches_root / "stands.parquet")
    mixture_config = read_json(ROOT / config["parents"]["lirf_mixture_config"]["path"])
    if object_hash(mixture_config) != read_json(ROOT / config["parents"]["lirf_mixture_selection"]["path"])["config_sha256"]:
        raise ValueError("The v9 declaration does not match its selection receipt.")
    # 1. v9 reproduced from scratch (frozen v7 + January-November mixture), equal to the uploaded v9 after rounding.
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
    # 2. Twelve-month v7 on clock-present rows when the candidate uses cross-fit v9.
    v9 = v9_frozen.copy()
    v7_record = None
    if candidate["v9"] == "cross_fit":
        run = ROOT / "runs/adsb_v24/v7/refit"
        v7_record = read_json(run / "run.json")
        if v7_record["stage"] != "refit_twelve_months" or v7_record["held_out_months"] or v7_record["predicted_ids_sha256"] != identity_hash(pd.Series(ids)) or v7_record["fit_months"] != ALL_MONTHS:
            raise ValueError("The twelve-month v7 refit does not match the ranking or the declaration.")
        v7_new = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
        clock = np.isfinite(v7_new)
        if clock.sum() != v7_record["predicted_clock_rows"] or (clock & population).any():
            raise ValueError("Twelve-month v7 coverage disagrees with the clock-present route.")
        v9 = np.where(clock, np.maximum(v7_new, floor), v9_frozen)
        v7_record = {"run_sha256": sha256_file(run / "run.json"), "predictions_sha256": sha256_file(run / "ranking_predictions.parquet"), "clock_rows": int(clock.sum()), "clock_fit_rows": v7_record["clock_fit_rows"],
                     "component_iterations": v7_record["component_iterations"], "code_sha256": v7_record["code"]["sha256"]}
    # 3. Residual on all twelve months' declared rows.
    parts, training_months = [], []
    for m in ALL_MONTHS:
        src = candidate["v9"] if m in FRAME_OF else "cross_fit"
        parts.append(month_table(load_frame(config, m, src), adsb_features(m, matches_root, ROOT / args.cache), m))
        training_months.append(f"{m}:{src}")
    train = pd.concat(parts, ignore_index=True)
    model = fit_residual(train, config, scope, feature_set)
    rounds, booster, scoped_rows = model["rounds"], model["booster"], int(model["rows"])
    # 4. Ranking features, residual, blends, floor.
    table, _ = ranking_table(config, departures, stands, v9)
    day = departures[TIME].dt.dayofyear.to_numpy(dtype=float)
    table["season_cos"] = np.cos(2 * np.pi * (day - 1) / 365.25)
    residual = predict_residual(model, table, config, scope)
    inside = scope_mask(table, scope)
    matched = table["adsb_matched"].eq(1).to_numpy()
    item = {"weight": 1.0, "scope": scope, "blend_with": config["blend_with"], "extra_blends": [*config["extra_blends"], *candidate.get("extra_blends", [])]}
    values, touched = candidate_values(table, v9, residual, item, floor)
    uploaded_v21 = pd.read_parquet(ROOT / config["parents"]["v21_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    lirf_missing_clock = table["airport"].eq("LIRF").to_numpy() & ~np.isfinite(table["proxy_s"].to_numpy(dtype=float))
    if not np.array_equal(np.rint(values[population]).astype(np.int32), uploaded_v9[population]) or not np.array_equal(np.rint(values[lirf_missing_clock]).astype(np.int32), uploaded_v21[lirf_missing_clock]):
        raise ValueError("Rows the declaration keeps fixed (LIRF fill population, missing-clock LIRF rows) differ from the uploaded values.")
    run_dir = ROOT / "runs/adsb_v24" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "residual_model.txt"), num_iteration=rounds)
    write_json(run_dir / "residual_model.json", {"categories": model["categories"], "rounds": rounds, "inner_best_iterations": model["inner_best_iterations"], "training_rows": scoped_rows, "training_months": training_months,
                                                  "scope": scope, "feature_set": feature_set, "matches_dir": config["matches_dirs"][dir_key], "blend_with": config["blend_with"], "extra_blends": item["extra_blends"], "models": ["residual_model.txt"],
                                                  "features": list(config["residual_model"]["feature_sets"][feature_set]), "weight": 1.0, "floor_seconds": floor, "v9_provenance": candidate["v9"],
                                                  "feature_importance_gain": dict(zip(booster.feature_name(), map(float, booster.feature_importance("gain"))))})
    detail = pd.DataFrame({ID: ids, "airport": table["airport"].to_numpy(), "v9_frozen": v9_frozen, "v9": v9, "residual": residual, "in_scope": inside, "touched": touched, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": object_hash(config), "selection_sha256": sha256_file(ROOT / args.selection),
                  "selected_candidate": name, "v9_provenance": candidate["v9"], "twelve_month_v7": v7_record, "residual": {"variant": candidate["residual"], "feature_set": feature_set, "matches_dir": config["matches_dirs"][dir_key],
                  "rounds": rounds, "training_rows": scoped_rows, "training_months": training_months, "residual_model_sha256": sha256_file(run_dir / "residual_model.txt")},
                  "residual_model_sha256": sha256_file(run_dir / "residual_model.txt"), "residual_models_sha256": [sha256_file(run_dir / "residual_model.txt")], "rounds": rounds, "training_rows": scoped_rows,
                  "weight": 1.0, "scope": scope, "feature_set": feature_set, "blend_with": config["blend_with"], "extra_blends": item["extra_blends"], "ranking_rows_in_scope": int(inside.sum()),
                  "ranking_matched_rows": int(matched.sum()), "stands": int(len(stands)), "external_data": config["external_data"], "parents": config["parents"], "incumbent": config["incumbent"],
                  "model_code_sha256": object_hash({p: sha256_file(ROOT / p) for p in ["tools/export_adsb_v24.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "tools/export_adsb_v11.py", "src/prc2026_adsb/features.py"]}),
                  "audit_used": False, "native_fit_count": 1 + (12 if v7_record else 0)}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = np.rint(values).astype(np.int32)
    delta = exported.astype(float) - uploaded_v21
    airports = table["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)), "selected_candidate": name,
              "v9_provenance": candidate["v9"], "twelve_month_v7": v7_record, "rows_in_scope": int(inside.sum()), "matched_rows": int(matched.sum()), "population_rows_unchanged": True, "lirf_missing_clock_rows_unchanged": True,
              "v9_reproduced_from_scratch": True, "v9_changed_rows_vs_frozen": int((np.abs(v9 - v9_frozen) > 1e-9).sum()),
              "residual": {"variant": candidate["residual"], "rounds": rounds, "inner_best_iterations": model["inner_best_iterations"], "training_rows": scoped_rows, "training_months": training_months},
              "versus_v21": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]}, "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))}},
              "seconds": round(time.time() - started, 1)}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "rounds": rounds, "residual_training_rows": scoped_rows, "changed_vs_v21": report["versus_v21"]["changed_rows"],
           "rms_change": round(report["versus_v21"]["rmse_of_change_seconds"], 3), "max_change": report["versus_v21"]["max_absolute_change_seconds"], "minimum": receipt["minimum_prediction_seconds"], "maximum": receipt["maximum_prediction_seconds"]})


if __name__ == "__main__":
    main()
