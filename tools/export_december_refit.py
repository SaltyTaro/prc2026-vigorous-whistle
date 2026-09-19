"""Export the v20 recipe with the twelve-month refit learners and a four-month residual stage as the next versioned submission (adsb_v23_v1).

v9 is rebuilt from scratch (frozen v7 plus the January-November LIRF mixture; its rounding must equal the uploaded v9). Every declared
table is rebuilt for the ranking as the mean of finite twelve-month refit predictions over its frozen flattened leaf list. The residual
(v12's nb model, matched rows outside LIRF) is fitted on the in-scope rows of January, July, November and December 2025 with rounds from
the day-grouped five-fold inner rule on that training set, applied to the in-scope ranking rows, blended exactly as v20, floored, exported.
Rows outside every scope must equal the uploaded v9. Nothing is uploaded.
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
from prc2026_adsb.features import DEPARTURE_COLUMNS
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_submission.contract import export_submission
from export_adsb_v11 import feature_columns, ranking_table
from run_adsb_v11 import candidate_values, check_parents, fit_residual, matches_dir, model_code_receipt, month_table, predict_residual, scope_mask
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population

RANKING_MONTHS = ["2026-01", "2026-07"]


def leaf_name(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def flattened_members(root: Path, run_dir: str) -> list[str]:
    if (root / run_dir / "refit" / "run.json").exists():
        return [run_dir]
    return [leaf for member in read_json(root / run_dir / "run.json")["learners"] for leaf in flattened_members(root, member)]


def build_ranking_ensembles(config: dict, ids: np.ndarray) -> dict:
    """runs/adsb_v23/ens/<name>/refit/ranking_predictions.parquet from the twelve-month leaf refits; returns per-table member lists and hashes."""
    base = read_json(ROOT / config["parents"]["v22_v1_declaration"]["path"])
    leaf_predictions, leaf_hashes = {}, {}
    record = {}
    for name, spec in config["extra_feature_tables"].items():
        members = flattened_members(ROOT, Path(base["extra_feature_tables"][name]["development"][0]).parent.parent.as_posix())
        for leaf in set(members):
            if leaf not in leaf_predictions:
                path = ROOT / config["leaf_learners"][leaf]["refit_output"] / "ranking_predictions.parquet"
                frame = pd.read_parquet(path)
                if not np.array_equal(frame[ID].to_numpy(), ids):
                    raise ValueError(f"{leaf}: twelve-month refit predictions are not in the ranking order.")
                leaf_predictions[leaf] = frame["adsb_full"].to_numpy(dtype=float)
                run = read_json(path.parent / "run.json")
                if run["stage"] != "refit_twelve_months" or run["december_rows"] != 165677 or run["rows"] != 1919370 + 165677 or run["inputs"]["source_run"] != leaf:
                    raise ValueError(f"{leaf}: not a twelve-month refit of this learner.")
                leaf_hashes[leaf] = {"predictions_sha256": sha256_file(path), "run_sha256": sha256_file(path.parent / "run.json"), "rounds": run["rounds"], "fitted_rows": run["fitted_rows"]}
        wide = np.column_stack([leaf_predictions[m] for m in members])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(wide, axis=1)
        out = ROOT / spec["ranking"]
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: ids, "adsb_full": mean}).to_parquet(out, index=False, compression="zstd")
        write_json(out.parent / "run.json", {"created_at_utc": utc_now(), "rule": "mean of finite twelve-month refit predictions over the frozen flattened leaf list", "learners": members, "sources": {m: leaf_hashes[m] for m in set(members)}})
        record[name] = {"members": len(members), "distinct_leaves": len(set(members)), "all_nan_rows": int(np.isnan(mean).sum()), "sha256": sha256_file(out)}
    return {"tables": record, "leaves": leaf_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v23_v1.json"))
    parser.add_argument("--scratch", type=Path, default=Path("scratch"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v23/export.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    candidate = config["candidates"][config["selected_candidate"]]
    weight, scope, feature_set = candidate["weight"], candidate.get("scope", "matched"), candidate.get("features")
    item = {"weight": weight, "scope": scope, "blend_with": candidate.get("blend_with"), "extra_blends": candidate.get("extra_blends", [])}
    neighbours = bool(config["residual_model"].get("neighbour_features", False))
    mixture_config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    if object_hash(mixture_config) != read_json(ROOT / config["parents"]["lirf_mixture_selection"]["path"])["config_sha256"]:
        raise ValueError("The v9 declaration does not match its selection receipt.")
    # Float v9 on the ranking, reproduced from scratch (January-November mixture, unchanged).
    fitting = movements([f"2025-{m:02}" for m in range(1, 12)])
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
    # Twelve-month ranking tables.
    ensembles = build_ranking_ensembles(config, ids)
    # Residual model on the in-scope rows of the four residual months (December from the opened scratch table; no extra tables are residual features).
    stands = pd.read_parquet(matches_dir(config) / "stands.parquet")
    folds = {"seasonal": pd.read_parquet(ROOT / args.scratch / "seasonal.parquet"), "forward": pd.read_parquet(ROOT / args.scratch / "forward.parquet"), "december": pd.read_parquet(ROOT / args.scratch / "december.parquet")}
    frame_of = {"2025-01": "seasonal", "2025-07": "seasonal", "2025-11": "forward", "2025-12": "december"}
    if float(folds["december"]["floor"].iloc[0]) != floor:
        raise ValueError("December scratch floor differs from the v9 floor.")
    tables = [month_table(month, folds[frame_of[month]], stands, matches_dir(config), neighbours, []) for month in config["residual_months"]]
    train = pd.concat(tables, ignore_index=True)
    model = fit_residual(train, config, scope, feature_set)
    rounds, booster = model["rounds"], model["booster"]
    scoped_rows = int(model["rows"])
    # Ranking ADS-B features, residual, blend.
    table, _ = ranking_table(config, departures, stands, v9)
    residual = predict_residual(model, table, config, scope)
    matched = table["adsb_matched"].eq(1).to_numpy()
    values, inside = candidate_values(table, v9, residual, item, floor)
    if not np.array_equal(np.rint(values[~inside]).astype(np.int32), uploaded_v9[~inside]):
        raise ValueError("Rows outside the scope differ from the uploaded v9 export.")
    run_dir = ROOT / "runs/adsb_v23" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "residual_model.txt"), num_iteration=rounds)
    write_json(run_dir / "residual_model.json", {"categories": model["categories"], "rounds": rounds, "inner_best_iterations": model["inner_best_iterations"], "training_rows": scoped_rows, "training_months": config["residual_months"],
                                                  "scope": scope, "feature_set": feature_set, "blend_with": candidate.get("blend_with"), "extra_blends": candidate.get("extra_blends", []), "models": ["residual_model.txt"],
                                                  "features": list(config["residual_model"]["feature_sets"][feature_set]) if feature_set else list(config["residual_model"]["features"]), "weight": weight, "floor_seconds": floor,
                                                  "feature_importance_gain": dict(zip(booster.feature_name(), map(float, booster.feature_importance("gain"))))})
    detail = table[[ID, "airport", *feature_columns(config)]].copy()
    detail["v9"] = v9; detail["residual"] = residual; detail["in_scope"] = inside; detail[f"v{args.version}"] = values
    detail.to_parquet(run_dir / "ranking_features.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": object_hash(config), "selected_candidate": config["selected_candidate"], "weight": weight, "scope": scope,
                  "feature_set": feature_set, "blend_with": candidate.get("blend_with"), "extra_blends": candidate.get("extra_blends", []), "residual_model_sha256": sha256_file(run_dir / "residual_model.txt"),
                  "residual_models_sha256": [sha256_file(run_dir / "residual_model.txt")], "rounds": rounds, "training_rows": scoped_rows, "residual_months": config["residual_months"], "refit_months": config["refit_months"],
                  "ranking_rows_in_scope": int(inside.sum()), "ranking_matched_rows": int(matched.sum()), "stands": int(len(stands)), "external_data": config["external_data"], "parents": config["parents"],
                  "incumbent": config["incumbent"], "model_code_sha256": model_code_receipt(ROOT)["sha256"], "twelve_month_tables": ensembles["tables"], "twelve_month_leaves": ensembles["leaves"],
                  "audit_used": False, "native_fit_count": 1 + len(ensembles["leaves"])}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = np.rint(values).astype(np.int32)
    v20 = pd.read_parquet(ROOT / config["parents"]["v20_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    delta = exported.astype(float) - v20
    by_airport = table.groupby("airport").agg(rows=("adsb_matched", "size"), matched=("adsb_matched", "sum"))
    by_airport["in_scope"] = pd.Series(inside, index=table.index).groupby(table["airport"]).sum()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(ids), "scope": scope,
              "rows_in_scope": int(inside.sum()), "matched_rows": int(matched.sum()), "changed_rows_vs_v9": int((exported != uploaded_v9).sum()), "rows_outside_scope_match_v9_export": True, "v9_reproduced_from_scratch": True,
              "residual": {"rounds": rounds, "inner_best_iterations": model["inner_best_iterations"], "training_rows": scoped_rows, "months": config["residual_months"]},
              "twelve_month_tables": ensembles["tables"], "twelve_month_leaves": ensembles["leaves"],
              "versus_v20": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[table["airport"].eq(a).to_numpy()] ** 2))) for a in sorted(table["airport"].unique())}},
              "by_airport": {a: {"rows": int(r.rows), "matched": int(r.matched), "in_scope": int(r.in_scope)} for a, r in by_airport.iterrows()}}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "rounds": rounds, "residual_training_rows": scoped_rows, "versus_v20": report["versus_v20"]["changed_rows"],
           "rms_change": round(report["versus_v20"]["rmse_of_change_seconds"], 3), "minimum": receipt["minimum_prediction_seconds"], "maximum": receipt["maximum_prediction_seconds"]})


if __name__ == "__main__":
    main()
