"""Declared populations, weighting and immutable source receipts."""
from pathlib import Path
import numpy as np

from prc2026.artifacts import read_json, sha256_file, object_hash
from prc2026.validation import identity_hash
from prc2026.features import ID, TIME, TARGET
from .features import FEATURE_NAMES, CATEGORICAL_COLUMNS, AIRPORT_INPUTS, MASKED_FIELDS, UNCHANGED_NAMES


def configurations(root):
    config = read_json(root / "configs/nm_mask_v1.json")
    feature = read_json(root / config["feature_config"])
    if config["audit_used"] or config["automatic_submission"] or feature["audit_used"] or feature["departure_labels_used"]:
        raise ValueError("Audit use or automatic submission is outside this declaration.")
    if config["development_folds"] != ["seasonal", "forward"] or config["native_fit_count"] != 8:
        raise ValueError("The original two folds and eight jobs are required.")
    if config["candidate_weights"] != [.5, 1.] or config["candidate_count"] != 12:
        raise ValueError("Only the twelve declared candidate recipes are allowed.")
    if config["minimum_improvement_fraction_each_fold"] != .005:
        raise ValueError("The two-fold promotion margin changed.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("No target filtering or prediction clipping.")
    if config["refit_months"] != [f"2025-{month:02}" for month in range(1, 12)]:
        raise ValueError("December cannot enter refitting.")
    expected = {(fold, variant) for fold in config["development_folds"] for variant in config["variants"]}
    if len(config["jobs"]) != 8 or {(j["fold"], j["variant"]) for j in config["jobs"]} != expected:
        raise ValueError("Each declared variant needs both original folds.")
    if {j["worker"] for j in config["jobs"]} != {f"arcw{i}" for i in range(8)}:
        raise ValueError("Use exactly the eight allocated workers.")
    if feature["feature_names"] != FEATURE_NAMES or feature["categorical_columns"] != CATEGORICAL_COLUMNS:
        raise ValueError("The predictor contract changed.")
    if (feature["input_projection"] != AIRPORT_INPUTS or feature["masked_nm_fields"] != MASKED_FIELDS
            or feature["inherited_unchanged_features"] != UNCHANGED_NAMES):
        raise ValueError("The masking or inheritance contract changed.")
    for variant in config["variants"].values():
        if variant["kind"] not in {"lightgbm", "catboost"} or variant["fit_population"] not in {"natural", "augmented"}:
            raise ValueError("Unknown declared model or fitting cohort.")
        if variant["settings"]["threads"] != 16 or variant["settings"]["seed"] != config["seed"]:
            raise ValueError("Declared worker settings differ.")
    return feature, config


def code_receipt(root, *, feature=False):
    files = ["src/prc2026/artifacts.py", "src/prc2026/features.py", "src/prc2026/validation.py",
             "src/prc2026_phase3/features.py", "src/prc2026_phase4/features.py",
             "src/prc2026_phase5r/features.py", "src/prc2026_weather/features.py",
             "src/prc2026_weather/encoding.py", "requirements-phase1.txt", "requirements-phase2.txt",
             "requirements-nm-mask.txt", "src/prc2026_nm_mask/__init__.py", "src/prc2026_nm_mask/features.py",
             "src/prc2026_nm_mask/common.py"]
    files += (["src/prc2026_nm_mask/bundle.py", "tools/prepare_nm_mask.py"] if feature else
              ["src/prc2026_nm_mask/common.py", "src/prc2026_nm_mask/native.py",
               "src/prc2026_nm_mask/training.py", "tools/run_nm_mask.py"])
    hashes = {name: sha256_file(root / name) for name in sorted(files)}
    return {"files": hashes, "sha256": object_hash(hashes)}


def sample_weights(natural, population):
    natural = np.asarray(natural)
    if natural.dtype != bool or natural.ndim != 1 or not natural.any():
        raise ValueError("A nonempty boolean natural-missing cohort is required.")
    if population == "natural":
        if not natural.all():
            raise ValueError("A natural-only fit contains augmented rows.")
        return np.ones(len(natural), dtype=float)
    if population != "augmented" or natural.all():
        raise ValueError("Augmentation requires both declared cohorts.")
    return np.where(natural, .5, .5 * natural.sum() / (~natural).sum()).astype(float)


def validate_truth(features, truth):
    if features[ID].duplicated().any() or truth[ID].duplicated().any() or not np.array_equal(features[ID], truth[ID]):
        raise ValueError("Features and truth must align exactly one to one.")
    if not np.isfinite(truth[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("All signed/extreme targets must be finite and retained.")
    if truth[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
        raise ValueError("Reserved December labels cannot enter a development job.")


def compose(reference, predictions, eligible, weight):
    reference = np.asarray(reference, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    eligible = np.asarray(eligible)
    if eligible.dtype != bool or eligible.shape != reference.shape or len(predictions) != eligible.sum():
        raise ValueError("Candidate populations do not align.")
    if weight not in {.5, 1.} or not np.isfinite(reference).all() or not np.isfinite(predictions).all():
        raise ValueError("Candidate weights and predictions must follow the declaration.")
    result = reference.copy()
    result[eligible] = reference[eligible] + weight * (predictions - reference[eligible])
    return result
