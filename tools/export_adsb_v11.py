"""Export the selected ADS-B surface residual candidate on top of v9 as the next versioned submission.

Reconstructs the float v9 ranking predictions from scratch (frozen v7 plus the
v9 mixture refitted on the refit months; their rounding must equal the uploaded
v9), fits the declared residual model on the in-scope rows of the three
development months at the declared refit rounds (the scope is the selected
candidate's: matched rows, or matched rows plus rows with a matched runway
neighbour), applies it to the in-scope ranking departures, floors, and exports.
Rows outside the scope must equal the uploaded v9. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_adsb.features import DEPARTURE_COLUMNS, FEATURE_NAMES, NEIGHBOUR_NAMES, departure_features, neighbour_features
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_postprocess.rules import row_context
from prc2026_submission.contract import export_submission
from run_adsb_v11 import attach_extra, check_parents, extra_tables, feature_list, matches_dir, model_code_receipt, month_table, predict_residual, prepare_matrix, scope_mask
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population
import lightgbm as lgb

RANKING_MONTHS = ["2026-01", "2026-07"]


def ranking_table(config: dict, departures: pd.DataFrame, stands: pd.DataFrame, v9: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free ADS-B feature table of the ranking departures (plus neighbour context when declared) and the matches used."""
    matches = pd.concat([pd.read_parquet(matches_dir(config) / f"matches2_{month}.parquet") for month in RANKING_MONTHS], ignore_index=True)
    feats = departure_features(departures, matches, stands).merge(departures[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    if config["residual_model"].get("neighbour_features", False):
        feats = feats.merge(neighbour_features(departures, matches), on=ID, validate="one_to_one")
    context = row_context(departures)
    table = feats.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    table["airport"] = departures["ADEP_mvt"].to_numpy()
    table["v9"] = v9
    table["proxy_s"] = context["proxy_s"].to_numpy()
    table["schedule_interval_s"] = context["schedule_interval_s"].to_numpy()
    table["hour"] = departures[TIME].dt.hour + departures[TIME].dt.minute / 60
    return attach_extra(table, extra_tables(config, "ranking")), matches


def feature_columns(config: dict) -> list[str]:
    extra = [name for name in config.get("extra_feature_tables", {})]
    return [*FEATURE_NAMES, *(NEIGHBOUR_NAMES if config["residual_model"].get("neighbour_features", False) else []), *extra, *[f"{n}_minus_v9" for n in extra]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v11_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_v11/selection.json"))
    parser.add_argument("--scratch", type=Path, default=Path("scratch"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v11/export.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or selection["audit_used"]:
        raise ValueError("Selection does not belong to this declaration.")
    if selection["model_code"]["sha256"] != model_code_receipt(ROOT)["sha256"]:
        raise ValueError("Scientific code changed after selection.")
    if not selection["selected_candidate"]:
        raise ValueError("No candidate was selected; nothing to export.")
    candidate = config["candidates"][selection["selected_candidate"]]
    weight, scope, feature_set = candidate["weight"], candidate.get("scope", "matched"), candidate.get("features")
    neighbours = bool(config["residual_model"].get("neighbour_features", False))
    check_parents(config)
    mixture_config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    if object_hash(mixture_config) != read_json(ROOT / config["parents"]["lirf_mixture_selection"]["path"])["config_sha256"]:
        raise ValueError("The v9 declaration does not match its selection receipt.")
    # Float v9 on the ranking, reproduced from scratch.
    fitting = movements(config["refit_months"])
    v9_model, v9_component, v9_summary = fit_population_model(fitting, mixture_config)
    floor = v9_summary["floor_seconds"]
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    ids = departures[ID].to_numpy()
    v7 = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    _, _, _, interval, v9_mixture, index = score_population(v9_model, v9_component, ranking, ids, mixture_config)
    v9, _ = apply_candidate(v7, index, interval, v9_mixture, mixture_config["candidates"]["mixture_75"]["mixture_weight"], mixture_config["mixture"]["schedule_interval_threshold_seconds"], floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v9).astype(np.int32), uploaded_v9):
        raise ValueError("The uploaded v9 does not reproduce from scratch.")
    # Residual model on the in-scope rows of all three development months.
    stands = pd.read_parquet(matches_dir(config) / "stands.parquet")
    seasonal = pd.read_parquet(ROOT / args.scratch / "seasonal.parquet")
    forward = pd.read_parquet(ROOT / args.scratch / "forward.parquet")
    extra = extra_tables(config, "development")
    tables = [month_table(month, seasonal if month in ["2025-01", "2025-07"] else forward, stands, matches_dir(config), neighbours, extra) for month in config["development_months"]]
    train = pd.concat(tables, ignore_index=True)
    scoped_train = train[scope_mask(train, scope)].reset_index(drop=True)
    spec = config["residual_model"]
    x, categories = prepare_matrix(scoped_train, config, feature_set=feature_set)
    y = (scoped_train[TARGET] - scoped_train["v9"]).to_numpy(dtype=float)
    rounds = int(selection["refit_rounds"])
    booster = lgb.train(dict(spec["params"]), lgb.Dataset(x, y, categorical_feature=spec["categorical"]), rounds)
    model = {"booster": booster, "categories": categories, "rounds": rounds, "feature_set": feature_set}
    # Ranking ADS-B features and residual.
    table, _ = ranking_table(config, departures, stands, v9)
    residual = predict_residual(model, table, config, scope)
    inside = scope_mask(table, scope)
    matched = table["adsb_matched"].eq(1).to_numpy()
    values = np.maximum(v9 + weight * residual, floor)
    if not np.array_equal(np.rint(values[~inside]).astype(np.int32), uploaded_v9[~inside]):
        raise ValueError("Rows outside the scope differ from the uploaded v9 export.")
    run_dir = ROOT / args.selection.parent / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "residual_model.txt"), num_iteration=rounds)
    write_json(run_dir / "residual_model.json", {"categories": categories, "rounds": rounds, "training_rows": int(len(scoped_train)), "scope": scope, "feature_set": feature_set, "features": feature_list(config, feature_set),
                                                  "weight": weight, "floor_seconds": floor, "feature_importance_gain": dict(zip(booster.feature_name(), map(float, booster.feature_importance("gain"))))})
    detail = table[[ID, "airport", *feature_columns(config)]].copy()
    detail["v9"] = v9; detail["residual"] = residual; detail["in_scope"] = inside; detail[f"v{args.version}"] = values
    detail.to_parquet(run_dir / "ranking_features.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": object_hash(config),
                  "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": selection["selected_candidate"], "weight": weight, "scope": scope, "feature_set": feature_set,
                  "residual_model_sha256": sha256_file(run_dir / "residual_model.txt"), "rounds": rounds, "training_rows": int(len(scoped_train)),
                  "ranking_rows_in_scope": int(inside.sum()), "ranking_matched_rows": int(matched.sum()), "stands": int(len(stands)), "external_data": config["external_data"],
                  "parents": config["parents"], "incumbent": config["incumbent"], "model_code_sha256": selection["model_code"]["sha256"],
                  "audit_used": False, "refit_months": config["refit_months"], "native_fit_count": 2}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = np.rint(values).astype(np.int32)
    changed = exported != uploaded_v9
    by_airport = table.groupby("airport").agg(rows=("adsb_matched", "size"), matched=("adsb_matched", "sum"), gate=("adsb_gate", "mean"))
    by_airport["in_scope"] = pd.Series(inside, index=table.index).groupby(table["airport"]).sum()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)),
              "ranking_ids_sha256": identity_hash(ids), "scope": scope, "rows_in_scope": int(inside.sum()), "matched_rows": int(matched.sum()),
              "changed_rows_vs_v9": int(changed.sum()), "rows_outside_scope_match_v9_export": True, "v9_reproduced_from_scratch": True,
              "by_airport": {a: {"rows": int(r.rows), "matched": int(r.matched), "in_scope": int(r.in_scope), "gate_share": None if np.isnan(r.gate) else float(r.gate)} for a, r in by_airport.iterrows()},
              "residual_quantiles_in_scope": {str(q): float(np.quantile(residual[inside], q)) for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]},
              "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(values - v9), q)) for q in [0.5, 0.9, 0.99, 0.999]},
              "rounds": rounds, "training_rows": int(len(scoped_train))}
    if "v11_export" in config["parents"]:
        uploaded_v11 = pd.read_parquet(ROOT / config["parents"]["v11_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
        delta = exported.astype(float) - uploaded_v11
        report["versus_v11"] = {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))),
                                "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                                "changed_rows_outside_matched": int(((delta != 0) & ~matched).sum())}
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "scope": scope, "in_scope": report["rows_in_scope"], "matched": report["matched_rows"],
           "changed_vs_v9": report["changed_rows_vs_v9"], "versus_v11": report.get("versus_v11"), "minimum": receipt["minimum_prediction_seconds"],
           "maximum": receipt["maximum_prediction_seconds"], "rounds": rounds, "in_scope_by_airport": {a: r["in_scope"] for a, r in report["by_airport"].items()}})


if __name__ == "__main__":
    main()
