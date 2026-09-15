"""Independently verify real traffic windows and all saved phase-3 run receipts."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, PHASE, TIME
from prc2026.phase1 import load_partition as original_partition
from prc2026.validation import evaluate
from prc2026_models.experiments import load_model, predict
from prc2026_phase3.bundle import feature_code_receipt, load_bundle, load_parent, validate_feature_config
from prc2026_phase3.encoding import transform
from prc2026_phase3.features import CONTEXT_COLUMNS, clean_codes
from prc2026_phase3.training import model_code_receipt


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--output", type=Path, default=Path("docs/phase3/verification.json"))
    parser.add_argument("--parent-manifest", type=Path,
                        help="Archived producing-machine parent manifest, for cross-platform verification.")
    args = parser.parse_args()
    base = root / (args.bundle or read_json(root / "data/phase3/latest.json")["bundle"])
    parent_equivalence = None
    if args.parent_manifest:
        manifest = read_json(base / "manifest.json")
        archive = root / args.parent_manifest
        if sha256_file(archive) != manifest["descriptor"]["parent_manifest_sha256"]:
            raise ValueError("Archived producing-machine parent manifest hash mismatch.")
        parent = root / manifest["parent_bundle"]
        original, catalog = load_parent(root, parent)
        producing = read_json(archive)
        # Every field except creation time and measured host/resource metadata
        # must match, including all 43 byte hashes, input/code identities and IDs.
        descriptive = {"created_at_utc", "preparation"}
        local_identity = {k: v for k, v in original.items() if k not in descriptive}
        producing_identity = {k: v for k, v in producing.items() if k not in descriptive}
        for identity in [local_identity, producing_identity]:
            identity["sources"] = [{k: v for k, v in source.items() if k != "wall_seconds"}
                                    for source in identity["sources"]]
        if local_identity != producing_identity:
            raise ValueError("Local/producing parent data, code, or frozen split identities differ.")
        if manifest["audit_used"] or manifest["code"]["sha256"] != feature_code_receipt(root)["sha256"]:
            raise ValueError("Feature source or audit policy mismatch.")
        if manifest["bundle_id"] != object_hash(manifest["descriptor"]):
            raise ValueError("Phase-3 bundle descriptor mismatch.")
        validate_feature_config(manifest["descriptor"]["config"], original)
        verify_receipts(base, manifest["artifacts"])
        parent_equivalence = {"matching_data_code_split_identity_sha256": object_hash(local_identity),
                              "producing_manifest_sha256": sha256_file(archive),
                              "local_manifest_sha256": sha256_file(parent / "manifest.json"),
                              "only_allowed_differences": sorted(descriptive | {"sources[*].wall_seconds"})}
    else:
        manifest, parent, original, catalog = load_bundle(root, base)
    config = manifest["descriptor"]["config"]
    parts = []
    for record in original["artifacts"]:
        if not record["path"].startswith("context/"):
            continue
        part = pd.read_parquet(parent / record["path"], columns=CONTEXT_COLUMNS)
        months = config["ranking_months"] if record["path"] == "context/ranking.parquet" else config["training_months"]
        parts.append(part.loc[part[TIME].dt.strftime("%Y-%m").isin(months)])
    context = pd.concat(parts, ignore_index=True)
    for name in ["airport", "RUNWAY_mvt", "STAND_mvt"]:
        context[name] = clean_codes(context[name])
    # Reduce independent brute-force comparisons to one airport at a time.
    airport_context = {a: g for a, g in context.groupby("airport")}
    rng, comparisons, sampled_ids = np.random.default_rng(20260913), 0, []
    for source in manifest["rows"]:
        raw = pd.read_parquet(root / "data/raw" / source,
                              columns=[ID, PHASE, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"])
        months = config["ranking_months"] if source == "ranking.parquet" else config["training_months"]
        raw = raw.loc[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").isin(months)].sort_values(TIME)
        chosen = raw.iloc[np.unique(np.r_[0, len(raw)-1, rng.integers(0, len(raw), 2)])]
        saved = pd.read_parquet(base / "features" / source).set_index(ID)
        for _, query in chosen.iterrows():
            sampled_ids.append(float(query[ID]))
            data = airport_context[query["ADEP_mvt"]]
            t, row = query[TIME], saved.loc[query[ID]]
            runway = clean_codes(pd.Series([query["RUNWAY_mvt"]])).iloc[0]
            stand = clean_codes(pd.Series([query["STAND_mvt"]])).iloc[0]
            for phase, prefix, key, windows in [
                ("DEP", "airport_dep", None, (5, 15, 30, 60)),
                ("ARR", "airport_arr", None, (5, 15, 30, 60)),
                ("DEP", "runway_dep", "RUNWAY_mvt", (15, 60)),
                ("ARR", "runway_arr", "RUNWAY_mvt", (15, 60)),
                ("ARR", "stand_arr", "STAND_mvt", (60,)),
            ]:
                events = data.loc[data[PHASE].eq(phase) & data[TIME].lt(t)]
                if key:
                    events = events.loc[events[key].eq(runway if key == "RUNWAY_mvt" else stand)]
                for window in windows:
                    expected = int(events[TIME].ge(t-pd.Timedelta(minutes=window)).sum())
                    if row[f"{prefix}_count_{window}m"] != expected:
                        raise ValueError("Real traffic count disagrees with independent boolean-filter calculation.")
                    comparisons += 1
            arrivals = data.loc[data[PHASE].eq("ARR") & np.isfinite(data["arrival_taxi_in_s"])
                                & data["arrival_block_time_utc"].ge(data[TIME])
                                & data["arrival_block_time_utc"].lt(t)]
            for prefix, events in [("completed_arrival_taxi", arrivals),
                                   ("runway_completed_arrival_taxi", arrivals.loc[arrivals["RUNWAY_mvt"].eq(runway)])]:
                window = events.loc[events["arrival_block_time_utc"].ge(t-pd.Timedelta(minutes=60))]
                if row[f"{prefix}_count_60m"] != len(window):
                    raise ValueError("Arrival completion availability/count mismatch.")
                mean = window["arrival_taxi_in_s"].to_numpy(dtype=float).mean() if len(window) else np.nan
                if not np.isclose(row[f"{prefix}_mean_60m"], mean, rtol=1e-6, atol=.01, equal_nan=True):
                    raise ValueError("Arrival taxi-in mean differs from independent calculation.")
                comparisons += 2
    runs, truths, rescored = [], {}, 0
    model_code = model_code_receipt(root)["sha256"]
    for path in sorted((root / args.runs).glob("*/run.json")):
        run = read_json(path)
        if run["bundle_id"] != manifest["bundle_id"] or run["audit_used"]:
            raise ValueError("Phase-3 run provenance mismatch.")
        verify_receipts(path.parent, run["artifacts"])
        if run["model_code"]["sha256"] != model_code:
            raise ValueError("Saved run source differs from current model code.")
        if run["max_model_reload_difference_seconds"] > 1e-9:
            raise ValueError("Saved-model reproduction exceeded tolerance.")
        local_reload_difference = None
        if run["stage"] == "development":
            fold = run["fold"]
            if fold not in {"seasonal", "forward"}:
                raise ValueError("Unapproved scoring fold.")
            if fold not in truths:
                assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
                _, truths[fold] = original_partition(parent, catalog, assignment, "score")
            predictions = pd.read_parquet(path.parent / "predictions.parquet")
            metric = evaluate(truths[fold], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            saved_metric = read_json(path.parent / "metrics.json")
            if not np.isclose(metric["overall"]["rmse_seconds"], saved_metric["overall"]["rmse_seconds"], rtol=1e-12):
                raise ValueError("Saved development RMSE does not reproduce locally.")
            rescored += 1
        elif run["stage"] == "refit":
            encoder = read_json(path.parent / "encoder.json")
            ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *encoder["feature_names"]]).iloc[:1000]
            values = predict(load_model(path.parent / "model.txt", "lightgbm"), transform(ranking, encoder), "lightgbm", 4)
            saved = pd.read_parquet(path.parent / "ranking_predictions.parquet")
            aligned = ranking[[ID]].merge(saved, on=ID, how="left", validate="one_to_one")
            local_reload_difference = float(np.max(np.abs(values-aligned["prediction"].to_numpy())))
            if not np.isfinite(local_reload_difference) or local_reload_difference > 1e-9:
                raise ValueError("Refit predictions do not reproduce on this platform.")
        runs.append({"run_id": run["run_id"], "stage": run["stage"],
                     "reload_difference_seconds": run["max_model_reload_difference_seconds"],
                     "local_refit_reload_difference_seconds": local_reload_difference})
    report = {"checked_at_utc": utc_now(), "bundle_id": manifest["bundle_id"],
              "feature_artifacts_verified": len(manifest["artifacts"]), "parent_artifacts_verified": len(original["artifacts"]),
              "real_departures_sampled": len(sampled_ids), "independent_window_comparisons": comparisons,
              "sampled_ids": sampled_ids, "runs_verified": runs, "audit_used": False,
              "development_runs_rescored": rescored, "parent_cross_platform_equivalence": parent_equivalence,
              "all_original_predictors_unchanged": "asserted for every row during feature preparation"}
    write_json(root / args.output, report)
    print({"sampled_departures": len(sampled_ids), "comparisons": comparisons, "verified_runs": len(runs),
           "development_runs_rescored": rescored, "parent_cross_platform_equivalence": bool(parent_equivalence)})


if __name__ == "__main__":
    main()
