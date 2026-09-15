"""Independently reconstruct weather covariates, full scores, blends and native predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIME
from prc2026.validation import evaluate, identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_weather.bundle import load_bundle
from prc2026_weather.features import FEATURE_NAMES, INPUT_COLUMNS, PREVIOUS_NAMES, WEATHER_NAMES, load_observations, runway_geometry, verify_external, weather_features
from prc2026_weather.inference import node_prediction, pipeline_prediction, airports_prediction, legacy_prediction
from prc2026_weather.training import checked_runs, model_code_receipt, reference_predictions, validate_config
from select_weather import collect
from verify_phase5 import same_metrics
from refit_weather import eligible_recipe


def difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Native comparisons require matching finite arrays.")
    return float(np.max(np.abs(left-right))) if left.size else 0.


def sample(features):
    indexes = np.unique(np.r_[np.linspace(0, len(features)-1, min(1000, len(features)), dtype=int),
                              np.flatnonzero(features["lirf_missing_record"].eq(1)),
                              np.flatnonzero(features["wx_missing"].eq(1))])
    return features.iloc[indexes].reset_index(drop=True)


def verify(root, base, config_path, output, prior, followup, require_refit=False):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    external = verify_external(root, manifest["descriptor"]["config"])
    observations = load_observations(root, external["weather"])
    runway_path = next(i["path"] for i in external["runways"]["objects"] if i["path"].endswith("runways.csv"))
    geometry = runway_geometry(pd.read_csv(root/runway_path))
    feature_rows, recomputed_rows, coverage = 0, 0, {}
    for source, rows in manifest["rows"].items():
        frame = pd.read_parquet(base/"features"/source)
        previous = pd.read_parquet(root/manifest["previous_bundle"]/"features"/source)
        if list(frame) != [ID, *FEATURE_NAMES] or len(frame) != rows:
            raise ValueError("Weather schema or row count changed.")
        pd.testing.assert_frame_equal(frame[[ID, *PREVIOUS_NAMES]], previous, check_exact=True)
        if identity_hash(frame[ID]) != manifest["source_ids_sha256"][source]:
            raise ValueError("Weather population changed.")
        chosen = sample(frame)
        raw = pd.read_parquet(root/"data/raw"/source, columns=INPUT_COLUMNS)
        query = chosen[[ID]].merge(raw, on=ID, how="left", validate="one_to_one", sort=False)
        rebuilt = weather_features(query, observations, geometry)
        pd.testing.assert_frame_equal(rebuilt, chosen[[ID, *WEATHER_NAMES]], check_exact=True)
        coverage[source] = {"rows": len(frame), "weather_missing_rows": int(frame["wx_missing"].sum()),
                            "external_runway_bearing_rows": int(frame["wx_runway_external_bearing"].sum())}
        feature_rows += len(frame)
        recomputed_rows += len(chosen)
    if coverage != manifest["coverage"]:
        raise ValueError("Feature coverage changed.")
    audited = read_json(root/"docs/competition/weather_20260914/coverage.json")["overall"]
    for population in ["ranking", "january_november_training"]:
        selected = [v for k, v in coverage.items() if (k == "ranking.parquet") == (population == "ranking")]
        rows = sum(v["rows"] for v in selected)
        covered = rows-sum(v["weather_missing_rows"] for v in selected)
        if rows != audited[population]["rows"] or covered != audited[population]["covered_rows"]:
            raise ValueError("Features differ from the independent pre-model coverage audit.")
    experiments = checked_runs(root, output, config, manifest)
    if set(experiments) != {(v, f) for v in config["variant_order"] for f in config["development_folds"]}:
        raise ValueError("All declared weather fits are required for verification.")
    native, run_hashes, scores = {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
        features, truth = load_partition(base, parent, catalog, assignment, "score", FEATURE_NAMES)
        v4, airports, _ = reference_predictions(root, config, fold, features, prior, followup)
        for name, frame in [("v4", v4), ("airports", airports)]:
            pd.testing.assert_frame_equal(frame, pd.read_parquet(output/"references"/f"{fold}_{name}.parquet"), check_exact=True)
        chosen = sample(features)
        for variant in config["variant_order"]:
            directory, run = experiments[(variant, fold)]
            node = read_json(directory/"node.json")
            if node["spec"] != config["variants"][variant] or node["iterations"] != run["best_iterations"]:
                raise ValueError("Native node does not match its declared run.")
            for role in ["fit", "early_stop", "score"]:
                if run[role+"_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                    raise ValueError("Protected fitting/stopping/scoring identities changed.")
                if role != "score" and (node[role+"_rows"] != run[role+"_rows"] or node[role+"_ids_sha256"] != run[role+"_ids_sha256"]):
                    raise ValueError("Native model used different fitting/stopping rows.")
            component = pd.read_parquet(directory/"component_predictions.parquet")
            if len(component) != len(features) or identity_hash(component[ID]) != run["score_ids_sha256"]:
                raise ValueError("Incomplete native scoring predictions.")
            expected = chosen[[ID]].merge(component, on=ID, validate="one_to_one", how="left")["prediction"].to_numpy()
            delta = difference(node_prediction(directory, node, chosen, config["threads"]), expected)
            if delta > 1e-8:
                raise ValueError("Saved native weather model does not reproduce.")
            native[directory.name] = {"rows": len(chosen), "max_difference_seconds": delta}
            aligned = features[[ID]].merge(component, on=ID, validate="one_to_one", how="left")["prediction"].to_numpy()
            values = .5*aligned+.5*airports["prediction"].to_numpy()
            saved = pd.read_parquet(directory/"predictions.parquet")
            expected = features[[ID]].merge(saved, on=ID, validate="one_to_one", how="left")["prediction"].to_numpy()
            if difference(values, expected) > 1e-8 or len(saved) != len(features):
                raise ValueError("Saved complete candidate composition changed.")
            metrics = evaluate(truth, saved, original["ranking_month_weights"] if fold == "seasonal" else None)
            same_metrics(metrics, read_json(directory/"metrics.json"))
            scores[directory.name] = metrics["overall"]["rmse_seconds"]
            run_hashes[directory.name] = sha256_file(directory/"run.json")
    selection = read_json(output/"selection.json")
    rebuilt = collect(root, base, config_path, output)
    for key in ["config_sha256", "model_code_sha256", "selection_code_sha256", "paired_day_code_sha256", "bundle_id", "selected_candidate", "refit_iterations"]:
        if selection[key] != rebuilt[key]:
            raise ValueError("Weather selection no longer reproduces: "+key)
    same_metrics(rebuilt["candidates"], selection["candidates"])
    refits = []
    for filename in sorted(output.glob("*/run.json")):
        report = read_json(filename)
        if report["stage"] != "refit":
            continue
        verify_receipts(filename.parent, report["artifacts"])
        if (report["audit_used"] or report["refit_months"] != config["refit_months"]
            or report["config_sha256"] != object_hash(config) or report["bundle_id"] != manifest["bundle_id"]
            or report["model_code"]["sha256"] != model_code_receipt(root)["sha256"]
            or report["selection_sha256"] != sha256_file(output/"selection.json")
            or report["refit_code_sha256"] != sha256_file(root/"tools/refit_weather.py")):
            raise ValueError("Refit provenance changed.")
        candidate = eligible_recipe(config, selection, manifest, model_code_receipt(root))
        if report["variant"] != selection["selected_candidate"]:
            raise ValueError("Refit does not implement the selected candidate.")
        months = catalog[TIME].dt.strftime("%Y-%m")
        protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
        ids = catalog.loc[months.isin(config["refit_months"]) & ~catalog["FLIGHT_ID_mvt"].isin(protected), ID]
        if report["fit_rows"] != len(ids) or report["fit_ids_sha256"] != identity_hash(ids):
            raise ValueError("Refit consumed different rows or protected flight groups.")
        ranking = pd.read_parquet(base/"features/ranking.parquet")
        saved = pd.read_parquet(filename.parent/"ranking_predictions.parquet")
        if len(saved) != len(ranking) or identity_hash(saved[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking predictions must retain the complete template population.")
        chosen = sample(ranking)
        expected = chosen[[ID]].merge(saved, on=ID, how="left", validate="one_to_one")["prediction"].to_numpy()
        delta = difference(pipeline_prediction(filename.parent, read_json(filename.parent/"pipeline.json"), chosen, config["threads"]), expected)
        if delta > 1e-8:
            raise ValueError("Final native pipeline does not reproduce ranking predictions.")
        cards = read_json(filename.parent/"nodes.json")
        if set(cards) != set(candidate["family_weights"]):
            raise ValueError("Refit has undeclared native components.")
        new_values = np.zeros(len(chosen))
        for variant, weight in candidate["family_weights"].items():
            node = cards[variant]
            if (node["spec"] != config["variants"][variant] or node["iterations"] != selection["refit_iterations"][variant]
                or node["fit_ids_sha256"] != identity_hash(ids) or node["fit_rows"] != len(ids) or node["early_stop_rows"] != 0):
                raise ValueError("Refit native component differs from the frozen recipe or population.")
            new_values += weight*node_prediction(filename.parent, node, chosen, config["threads"])
        source = prior/config["reference_refit_run"]
        if sha256_file(source/"run.json") != config["reference_refit_sha256"]:
            raise ValueError("Original v4 refit changed.")
        verify_receipts(source, read_json(source/"run.json")["artifacts"])
        old = read_json(source/"pipeline.json")
        airport_pipeline = next(m["pipeline"] for m in old["members"] if m["name"] == "catboost_long")
        proposed = .5*new_values+.5*airports_prediction(source, airport_pipeline, chosen, config["threads"])
        weight = candidate["new_weight"]
        from prc2026_weather.features import CORE_NAMES
        composed = weight*proposed+(1-weight)*legacy_prediction(source, old, chosen[[ID, *CORE_NAMES]], config["threads"])
        if difference(composed, expected) > 1e-8:
            raise ValueError("Final pipeline does not implement the declared ensemble weights.")
        refits.append({"run_id": filename.parent.name, "run_sha256": sha256_file(filename), "rows": len(chosen), "max_difference_seconds": delta})
    if require_refit and not refits:
        raise ValueError("A verified refit was required but is missing.")
    return {"created_at_utc": utc_now(), "passed": True, "audit_used": False, "bundle_id": manifest["bundle_id"],
            "model_code_sha256": model_code_receipt(root)["sha256"], "verification_code_sha256": sha256_file(Path(__file__)),
            "feature_rows_checked": feature_rows, "weather_feature_rows_recomputed": recomputed_rows, "coverage": coverage,
            "native_checks": native, "development_scores": scores, "development_run_sha256": run_hashes,
            "blend_fold_scores_checked": 2*len(selection["candidates"]), "selection_sha256": sha256_file(output/"selection.json"),
            "selected_candidate": selection["selected_candidate"], "refits": refits}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/weather_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/weather"))
    parser.add_argument("--prior-runs", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--followup-runs", type=Path, default=Path("runs/phase5_followup"))
    parser.add_argument("--output", type=Path, default=Path("docs/weather/verification.json"))
    parser.add_argument("--require-refit", action="store_true")
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/"data/weather/latest.json")["bundle"])
    with ResourceMonitor() as monitor:
        report = verify(root, base, root/args.config, root/args.runs, root/args.prior_runs, root/args.followup_runs, args.require_refit)
    report["resources"] = monitor.stats
    write_json(root/args.output, report)
    print({k: report[k] for k in ["passed", "feature_rows_checked", "weather_feature_rows_recomputed", "blend_fold_scores_checked", "selected_candidate"]})


if __name__ == "__main__":
    main()
