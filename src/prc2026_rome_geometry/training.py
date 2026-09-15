"""Three declared models per fold; every reported candidate keeps full scoring IDs."""

from pathlib import Path
import argparse
import gc
import shutil

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import emit
from prc2026.validation import identity_hash, require_development_fold
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_nmclock.features import route_mask
from .bundle import load_bundle
from .common import configurations, complete_reference, focal_partition, model_code_receipt
from .native import fit_node
from .inference import node_prediction


def checked_runs(root, output, config, manifest):
    found = {}
    code = model_code_receipt(root)
    for path in sorted(output.glob("*/run.json")):
        report = read_json(path)
        if report["stage"] != "development":
            continue
        key = (report["fold"], report["variant"])
        if key in found or key[0] not in config["development_folds"] or key[1] not in config["variants"]:
            raise ValueError("Duplicate or undeclared geometry run.")
        if report["model_code"] != code or report["config_sha256"] != object_hash(config) or report["bundle_id"] != manifest["bundle_id"] or report["audit_used"]:
            raise ValueError("A completed run has changed inputs or model code.")
        verify_receipts(path.parent, report["artifacts"])
        found[key] = (path.parent, report)
    return found


def suite(root, fold, surface_runs, timeline_runs, output):
    _, config = configurations(root)
    base, manifest, previous, original_base, original, catalog, extra = load_bundle(root)
    require_development_fold(original["descriptor"]["protocol"], fold)
    full, full_truth, reference, original_airport, ref_metrics = complete_reference(root, previous, original_base, original, catalog, fold, surface_runs, timeline_runs)
    assignment = pd.read_parquet(original_base / "splits" / (fold + ".parquet"))
    features, truths, identity = {}, {}, {}
    for role in ["fit", "early_stop", "score"]:
        values, truth = focal_partition(root, previous, original_base, catalog, assignment, role, extra, original)
        identity[role] = {"focal_rows":len(values), "focal_ids_sha256":identity_hash(values[ID])}
        present = route_mask(values)
        features[role], truths[role] = values.loc[present].reset_index(drop=True), truth.loc[present].reset_index(drop=True)
        identity[role].update(clock_rows=len(features[role]), clock_ids_sha256=identity_hash(features[role][ID]))
    ref_dir = output / "references" / fold
    if not ref_dir.exists():
        ref_dir.mkdir(parents=True)
        reference.to_parquet(ref_dir / "v7.parquet", index=False, compression="zstd")
        original_airport.to_parquet(ref_dir / "airport_component.parquet", index=False, compression="zstd")
        write_json(ref_dir / "metrics.json", ref_metrics)
        write_json(ref_dir / "reference.json", {"config_sha256":object_hash(config), "score_ids_sha256":identity_hash(full[ID]),
            "complete_rows":len(full), "parent_selection_sha256":config["reference_selection_sha256"], "artifacts":receipts(ref_dir)})
    else:
        ref_receipt = read_json(ref_dir / "reference.json")
        if ref_receipt["config_sha256"] != object_hash(config):
            raise ValueError("Existing reference belongs to another declaration.")
        verify_receipts(ref_dir, ref_receipt["artifacts"])
        for name, expected in [("v7.parquet", reference), ("airport_component.parquet", original_airport)]:
            pd.testing.assert_frame_equal(pd.read_parquet(ref_dir / name), expected, check_exact=True)
    existing = checked_runs(root, output, config, manifest)
    expected_airport = features["score"][[ID]].merge(original_airport, on=ID, how="left", validate="one_to_one").prediction.to_numpy()
    for variant in ["control", "geometry", "geometry_queue"]:
        if (fold, variant) in existing:
            continue
        if shutil.disk_usage(root).free < config["execution"]["minimum_free_disk_before_fit_bytes"]:
            raise ValueError("Keep the declared free-disk margin before fitting.")
        run = new_run(output, fold + "_" + variant)
        with ResourceMonitor() as monitor:
            node = fit_node(run, variant, config["settings"], features["fit"], truths["fit"], features["early_stop"], truths["early_stop"])
            values = node_prediction(run, node, features["score"], config["threads"])
            pd.DataFrame({ID:features["score"][ID], "prediction":values}).to_parquet(run / "component_predictions.parquet", index=False, compression="zstd")
            control_difference = float(np.max(np.abs(values - expected_airport))) if variant == "control" else None
            if variant == "control" and control_difference > config["control_reload_maximum_difference_seconds"]:
                write_json(run / "control_failure.json", {"maximum_difference_seconds":control_difference, "iterations":node["iterations"]})
                raise ValueError("The matched control does not reproduce v7; stop before new variant fits.")
            write_json(run / "node.json", node)
            write_json(run / "config.json", config)
        report = {"created_at_utc":utc_now(), "run_id":run.name, "stage":"development", "fold":fold, "variant":variant,
            "bundle_id":manifest["bundle_id"], "config_sha256":object_hash(config), "model_code":model_code_receipt(root),
            "fit_identities":identity, "complete_score_rows":len(full), "complete_score_ids_sha256":identity_hash(full[ID]),
            "iterations":node["iterations"], "control_maximum_difference_seconds":control_difference,
            "resources":monitor.stats, "environment":model_environment(root), "newly_fitted_models":1,
            "audit_used":False, "target_filter":"none", "prediction_postprocess":"none", "artifacts":receipts(run)}
        write_json(run / "run.json", report)
        emit("rome_geometry_variant_complete", fold=fold, variant=variant, iterations=node["iterations"], control_difference=control_difference, resources=monitor.stats)
        gc.collect()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["seasonal", "forward"], required=True)
    parser.add_argument("--surface-runs", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--output", type=Path, default=Path("runs/rome_geometry"))
    args = parser.parse_args()
    suite(root, args.fold, root / args.surface_runs, root / args.timeline_runs, root / args.output)
