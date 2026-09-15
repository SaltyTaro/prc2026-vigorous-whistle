"""Verify the feature chain, every development score, and saved model pipelines."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, PHASE, TIME
from prc2026.phase1 import load_partition as original_partition
from prc2026.validation import evaluate, identity_hash
from prc2026_phase4.bundle import load_bundle, scientific_parent
from prc2026_phase4.features import FEATURE_NAMES, PREVIOUS_NAMES, route_mask
from prc2026_phase4.training import model_code_receipt, pipeline_prediction, validate_config


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase4"))
    parser.add_argument("--output", type=Path, default=Path("docs/phase4/verification.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_v1.json"))
    args = parser.parse_args()
    base = root / (args.bundle or read_json(root / "data/phase4/latest.json")["bundle"])
    manifest, parent, original, catalog = load_bundle(root, base)
    declared_config = read_json(root / args.config)
    validate_config(declared_config, original["descriptor"]["protocol"])
    declared_config_hash = object_hash(declared_config)
    previous = root / manifest["previous_bundle"]
    metadata, checked_rows, offset_rows = [], 0, 0
    for source in manifest["rows"]:
        saved = pd.read_parquet(base / "features" / source)
        old = pd.read_parquet(previous / "features" / source, columns=[ID, *PREVIOUS_NAMES])
        pd.testing.assert_frame_equal(saved[[ID, *PREVIOUS_NAMES]], old, check_exact=True)
        raw = pd.read_parquet(root / "data/raw" / source,
                              columns=[ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
        months = manifest["descriptor"]["config"]["ranking_months" if source == "ranking.parquet" else "training_months"]
        raw = raw.loc[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").isin(months)].reset_index(drop=True)
        if not np.array_equal(saved[ID].to_numpy(), raw[ID].to_numpy()):
            raise ValueError("Original raw feature population/order changed.")
        interval = (raw[TIME]-raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(dtype=float)
        gate = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
        expected = np.zeros(len(raw), dtype="float32")
        expected[gate & np.isfinite(interval)] = interval[gate & np.isfinite(interval)]
        np.testing.assert_array_equal(saved["conditional_schedule_offset_s"].to_numpy(), expected)
        np.testing.assert_array_equal(saved["lirf_missing_record"].to_numpy(), gate.astype("float32"))
        if identity_hash(saved[ID]) != manifest["source_ids_sha256"][source]:
            raise ValueError("Feature ID receipt mismatch.")
        small = saved[[ID, "airport", "lirf_missing_record"]].copy()
        small["source_file"] = source
        metadata.append(small)
        checked_rows += len(raw)
        offset_rows += int(gate.sum())
    metadata = pd.concat(metadata, ignore_index=True)
    del raw, saved, old, small
    code = model_code_receipt(root)
    truths, sampled_features, records = {}, {}, []
    for path in sorted((root / args.runs).glob("*/run.json")):
        run = read_json(path)
        if run["audit_used"] or run["bundle_id"] != manifest["bundle_id"] or run["model_code"]["sha256"] != code["sha256"]:
            raise ValueError("Saved model provenance mismatch.")
        verify_receipts(path.parent, run["artifacts"])
        config = read_json(path.parent / "config.json")
        # Receipts use canonical sorted-key JSON. Execution order belongs to
        # the declared source configuration, whose entire content must match.
        if object_hash(config) != run["config_sha256"] or run["config_sha256"] != declared_config_hash:
            raise ValueError("Model configuration receipt mismatch.")
        pipeline = read_json(path.parent / "pipeline.json")
        if run["stage"] == "development":
            fold = run["fold"]
            if fold not in config["development_folds"]:
                raise ValueError("Unapproved scoring fold.")
            assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
            if fold not in truths:
                _, truths[fold] = original_partition(parent, catalog, assignment, "score")
            predictions = pd.read_parquet(path.parent / "predictions.parquet")
            metric = evaluate(truths[fold], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            saved_metric = read_json(path.parent / "metrics.json")
            if not np.isclose(metric["overall"]["rmse_seconds"], saved_metric["overall"]["rmse_seconds"], rtol=1e-12):
                raise ValueError("Complete development RMSE does not reproduce.")
            for role in ["fit", "early_stop", "score"]:
                if run[role + "_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                    raise ValueError("Run changed its protected split population.")
        elif run["stage"] == "refit":
            fold = "ranking"
            predictions = pd.read_parquet(path.parent / "ranking_predictions.parquet")
            if identity_hash(predictions[ID]) != original["ranking_ids_sha256"]:
                raise ValueError("Refit ranking IDs changed.")
            months = catalog[TIME].dt.strftime("%Y-%m")
            assignment = catalog[[ID]].copy()
            assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
            protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
            assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected), "partition"] = "reserved"
            if identity_hash(assignment.loc[assignment["partition"].eq("fit"), ID]) != run["fit_ids_sha256"]:
                raise ValueError("Refit fitting IDs changed.")
        else:
            raise ValueError("Unknown saved model stage.")
        for node in pipeline["nodes"].values():
            for role in ["fit", "early_stop"] if run["stage"] == "development" else ["fit"]:
                population = metadata.loc[metadata[ID].isin(assignment.loc[assignment["partition"].eq(role), ID])]
                selected = population.loc[route_mask(population, node["route"])]
                if len(selected) != node[role + "_rows"] or identity_hash(selected[ID]) != node[role + "_ids_sha256"]:
                    raise ValueError("Specialist fit/stop population differs from its recorded routing.")
        if fold not in sampled_features:
            candidates = metadata.loc[metadata[ID].isin(predictions[ID])]
            gated = candidates.loc[candidates["lirf_missing_record"].eq(1)]
            remaining = candidates.loc[~candidates[ID].isin(gated[ID])]
            chosen = pd.concat([gated, remaining.sample(n=min(1000-len(gated), len(remaining)), random_state=20260913)])
            parts = []
            for source, group in chosen.groupby("source_file"):
                data = pd.read_parquet(base / "features" / source, columns=[ID, *FEATURE_NAMES])
                parts.append(data.loc[data[ID].isin(group[ID])])
            sampled_features[fold] = pd.concat(parts, ignore_index=True)
        features = sampled_features[fold]
        restored = pipeline_prediction(path.parent, pipeline, features, threads=4)
        expected = features[[ID]].merge(predictions, on=ID, validate="one_to_one")["prediction"].to_numpy()
        difference = float(np.max(np.abs(restored-expected)))
        if not np.isfinite(difference) or difference > 1e-9:
            raise ValueError("Saved model pipeline does not reproduce on this machine.")
        records.append({"run_id": run["run_id"], "stage": run["stage"], "components": len(pipeline["nodes"]),
                        "reload_sample_rows": len(features), "reload_difference_seconds": difference,
                        "development_rmse_seconds": None if run["stage"] == "refit" else metric["overall"]["rmse_seconds"]})
    report = {"checked_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "bundle_id": manifest["bundle_id"],
              "declared_configuration_sha256": declared_config_hash,
              "parent_scientific_sha256": object_hash(scientific_parent(original)),
              "parent_artifacts_verified": len(original["artifacts"]), "previous_artifacts_verified": len(read_json(previous / "manifest.json")["artifacts"]),
              "feature_artifacts_verified": len(manifest["artifacts"]), "rows_with_all_86_prior_predictors_exactly_unchanged": checked_rows,
              "rows_with_independently_verified_offset": checked_rows, "gated_rows_including_ranking": offset_rows,
              "runs_verified": records, "development_runs_rescored": sum(r["stage"] == "development" for r in records),
              "reload_samples_include": "all gated LIRF scoring/ranking rows plus deterministic random rows, 1,000 per fold",
              "audit_used": False}
    write_json(root / args.output, report)
    print({"verified_rows": checked_rows, "verified_runs": len(records), "development_runs_rescored": report["development_runs_rescored"],
           "maximum_reload_difference_seconds": max((r["reload_difference_seconds"] for r in records), default=0.0)})


if __name__ == "__main__":
    main()
