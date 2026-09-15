"""Frozen declarations, exact ID joins and original complete reference scores."""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prc2026.artifacts import object_hash, read_json, sha256_file, verify_receipts
from prc2026.features import ID, TIME, TARGET
from prc2026.validation import identity_hash, evaluate, require_development_fold
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_nmclock.features import route_mask
from prc2026_timeline.encoding import contract as previous_contract
from prc2026_surface_airports.training import model_code_receipt as previous_model_code
from .features import POLICY, INPUT_COLUMNS, STATIC_NAMES, QUEUE_NAMES, EXTRA_NAMES

FEATURE_CONFIG = "configs/features_rome_geometry_v1.json"
MODEL_CONFIG = "configs/rome_geometry_v1.json"
PREVIOUS_NAMES, CATEGORIES = previous_contract({"features":"combined", "target":"nm_residual"})


def configurations(root):
    feature, model = read_json(root / FEATURE_CONFIG), read_json(root / MODEL_CONFIG)
    if feature["feature_policy"] != POLICY or feature["input_columns"] != INPUT_COLUMNS:
        raise ValueError("The declared geometry input contract differs.")
    if feature["static_feature_names"] != STATIC_NAMES or feature["queue_feature_names"] != QUEUE_NAMES:
        raise ValueError("The declared geometry predictors differ.")
    if feature["audit_used"] or feature["departure_labels_used_in_shared_features"] or model["audit_used"]:
        raise ValueError("The reserved audit and departure labels cannot enter shared features.")
    if feature["training_months"] != [f"2025-{m:02}" for m in range(1, 12)] or feature["ranking_months"] != ["2026-01", "2026-07"]:
        raise ValueError("Only declared movement months are allowed.")
    if set(feature["snapshot_assignment"]) != set(feature["training_months"] + feature["ranking_months"]):
        raise ValueError("Snapshot assignment is incomplete.")
    for config, key in [(feature, "input_receipts_sha256"), (model, "feature_receipts_sha256")]:
        for name, digest in config[key].items():
            if sha256_file(root / name) != digest:
                raise ValueError("A declared parent changed: " + name)
    old = read_json(root / "configs/surface_airports_v1.json")
    if model["settings"] != old["settings"] or model["threads"] != 16 or model["seed"] != 20260913:
        raise ValueError("The matched v7 learner settings changed.")
    if model["development_folds"] != ["seasonal", "forward"] or model["target_filter"] != "none" or model["prediction_postprocess"] != "none":
        raise ValueError("Complete development folds and unfiltered labels are required.")
    if model["candidate_variants"] != ["geometry", "geometry_queue"] or model["candidate_weights"] != [.5, 1.] or model["minimum_improvement_fraction_each_fold"] != .005:
        raise ValueError("The four declared recipes or promotion margin changed.")
    if model["refit_months"] != feature["training_months"] or model["airport"] != "LIRF":
        raise ValueError("The airport or December protection changed.")
    return feature, model


def model_code_receipt(root):
    files = dict(previous_model_code(root)["files"])
    paths = list((root / "src/prc2026_rome_geometry").glob("*.py"))
    paths += [root / "tools/run_rome_geometry.py", root / "requirements-rome-geometry.txt"]
    files.update({path.relative_to(root).as_posix():sha256_file(path) for path in paths})
    return {"files":files, "sha256":object_hash(files)}


def aligned_values(features, prediction):
    if prediction[ID].duplicated().any() or set(features[ID]) != set(prediction[ID]):
        raise ValueError("Prediction identities must match the complete requested population.")
    result = features[[ID]].merge(prediction[[ID, "prediction"]], on=ID, how="left", validate="one_to_one", sort=False)
    if not np.isfinite(result.prediction.to_numpy()).all():
        raise ValueError("Every prediction must be finite.")
    return result


def add_features(frame, extra):
    if frame[ID].duplicated().any() or extra[ID].duplicated().any():
        raise ValueError("Duplicate feature identity.")
    focal = frame.airport.eq("LIRF")
    if not set(frame.loc[focal, ID]).issubset(set(extra[ID])):
        raise ValueError("A LIRF row lacks its geometry feature record.")
    result = frame.merge(extra[[ID, *EXTRA_NAMES]], on=ID, how="left", validate="one_to_one", sort=False)
    if not np.array_equal(result[ID], frame[ID]):
        raise ValueError("The feature join changed row order.")
    pd.testing.assert_frame_equal(result[[ID, *PREVIOUS_NAMES]], frame[[ID, *PREVIOUS_NAMES]].reset_index(drop=True), check_exact=True)
    return result


def focal_partition(root, parent_base, original_base, catalog, assignment, role, extra, original):
    if role not in {"fit", "early_stop", "score"}:
        raise ValueError("Only original development partitions are supported.")
    all_ids = assignment.loc[assignment.partition.eq(role), ID]
    meta = catalog.loc[catalog[ID].isin(all_ids) & catalog.airport.eq("LIRF")].copy()
    if meta.empty or meta[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
        raise ValueError("Empty or reserved focal partition.")
    features, labels = [], []
    for source, part in meta.groupby("source_file", sort=True):
        ids = part[ID].tolist()
        features.append(pq.read_table(parent_base / "features" / source, columns=[ID, *PREVIOUS_NAMES], filters=[(ID, "in", ids)]).to_pandas())
        labels.append(pq.read_table(original_base / "labels" / source, columns=[ID, TARGET], filters=[(ID, "in", ids)]).to_pandas())
    values = pd.concat(features, ignore_index=True)
    if len(values) != len(meta) or identity_hash(values[ID]) != identity_hash(meta[ID]):
        raise ValueError("Focal partition IDs changed.")
    truth = meta.merge(pd.concat(labels, ignore_index=True), on=ID, validate="one_to_one")
    truth = aligned_truth(values, truth)
    if not np.isfinite(truth[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Every signed/extreme focal label must be retained.")
    return add_features(values, extra), truth


def complete_reference(root, previous_base, original_base, original, catalog, fold, surface_runs, timeline_runs):
    require_development_fold(original["descriptor"]["protocol"], fold)
    assignment = pd.read_parquet(original_base / "splits" / (fold + ".parquet"))
    features, truth = load_partition(previous_base, original_base, catalog, assignment, "score", ["airport", "nmclock_available", "nmclock_proxy_s"])
    truth = aligned_truth(features, truth)
    if identity_hash(features[ID]) != original["folds"][fold]["identity_sha256"]["score"]:
        raise ValueError("Original complete scoring identities changed.")
    selected = read_json(root / "docs/surface_airports/development_results.json")
    timeline = read_json(root / "docs/timeline/development_results.json")
    run_info = selected["development_runs"][fold]
    surface = surface_runs / run_info["run_id"]
    if sha256_file(surface / "run.json") != run_info["run_sha256"]:
        raise ValueError("Frozen v7 airport run changed.")
    verify_receipts(surface, read_json(surface / "run.json")["artifacts"])
    airport_prediction = aligned_values(features, pd.read_parquet(surface / "predictions.parquet"))
    reference_dir = surface_runs / "references" / fold
    verify_receipts(reference_dir, read_json(reference_dir / "reference.json")["artifacts"])
    baseline = aligned_values(features, pd.read_parquet(reference_dir / "v6.parquet"))
    present = route_mask(features)
    globals_ = []
    for variant in ["lgb_combined", "cat_combined"]:
        info = timeline["development_runs"][variant][fold]
        source = timeline_runs / info["run_id"]
        if sha256_file(source / "run.json") != info["run_sha256"]:
            raise ValueError("Frozen v7 global run changed.")
        verify_receipts(source, read_json(source / "run.json")["artifacts"])
        globals_.append(aligned_values(features, pd.read_parquet(source / "predictions.parquet")).prediction.to_numpy())
    values = baseline.prediction.to_numpy().copy()
    # Match the frozen family iteration order to preserve floating-point output.
    values[present] = (.25 * globals_[1][present] + .25 * globals_[0][present]
                       + .5 * airport_prediction.prediction.to_numpy()[present])
    prediction = pd.DataFrame({ID:features[ID], "prediction":values})
    metrics = evaluate(truth, prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
    expected = selected["candidates"]["full_timeline_v6_100"]["folds"][fold]["metrics"]
    if abs(metrics["overall"]["rmse_seconds"] - expected["overall"]["rmse_seconds"]) > 1e-9:
        raise ValueError("The complete v7 reference score no longer reproduces.")
    return features, truth, prediction, airport_prediction, metrics
