"""Prepare portable jobs from original partitions and full v7 references."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd

from prc2026.artifacts import (read_json, write_json, sha256_file, verify_receipts,
                              object_hash, utc_now, ResourceMonitor)
from prc2026.features import ID, TIME, TARGET
from prc2026.validation import identity_hash, evaluate, require_development_fold
from prc2026_phase3.bundle import load_partition
from .bundle import load_bundle
from .features import FEATURE_NAMES, eligible_route
from .common import configurations, validate_truth, code_receipt


def aligned(features, values):
    if features[ID].duplicated().any() or values[ID].duplicated().any() or set(features[ID]) != set(values[ID]):
        raise ValueError("Exact one-to-one partition identities are required.")
    return features[[ID]].merge(values, on=ID, how="left", validate="one_to_one", sort=False)


def prepare_jobs(root):
    _, config = configurations(root)
    base, manifest = load_bundle(root)
    original_base = root / manifest["original_bundle"]
    original = read_json(original_base / "manifest.json")
    verify_receipts(original_base, original["artifacts"])
    catalog = pd.read_parquet(original_base / "train_catalog.parquet")
    destination = root / "data/nm_mask_jobs" / manifest["bundle_id"][:20]
    if destination.exists():
        raise ValueError("Job inputs already exist; do not replace them.")
    destination.mkdir(parents=True)
    (destination / "jobs").mkdir()
    source_hash = code_receipt(root)["sha256"]
    all_jobs, fold_receipts = [], {}
    with ResourceMonitor() as monitor:
        for fold in config["development_folds"]:
            require_development_fold(original["descriptor"]["protocol"], fold)
            assignment_path = original_base / "splits" / f"{fold}.parquet"
            assignment = pd.read_parquet(assignment_path)
            directory = destination / fold
            directory.mkdir()
            paths, receipts, identity = {}, {}, {}
            def save(name, frame):
                path = directory / name
                frame.to_parquet(path, index=False, compression="zstd")
                rel = path.relative_to(root).as_posix()
                paths[name] = rel
                receipts[rel] = sha256_file(path)
            for role in ["fit", "early_stop", "score"]:
                features, truth = load_partition(base, original_base, catalog, assignment, role, FEATURE_NAMES)
                truth = aligned(features, truth)
                parts = []
                for source in sorted(truth.source_file.unique()):
                    part = pd.read_parquet(base / "routing" / source)
                    parts.append(part.loc[part[ID].isin(features[ID])])
                routing = aligned(features, pd.concat(parts, ignore_index=True))
                natural = eligible_route(routing)
                truth["natural_missing"] = natural
                validate_truth(features, truth)
                expected = original["folds"][fold]["identity_sha256"][role]
                if identity_hash(features[ID]) != expected or not natural.any():
                    raise ValueError("A complete original partition changed or has no deployment cohort.")
                identity[role] = {"original_rows": len(features), "original_ids_sha256": expected,
                                  "natural_rows": int(natural.sum()), "natural_ids_sha256": identity_hash(features.loc[natural, ID])}
                if role == "fit":
                    save("fit_augmented_features.parquet", features)
                    save("fit_augmented_truth.parquet", truth)
                    save("fit_natural_features.parquet", features.loc[natural].reset_index(drop=True))
                    save("fit_natural_truth.parquet", truth.loc[natural].reset_index(drop=True))
                elif role == "early_stop":
                    save("early_stop_features.parquet", features.loc[natural].reset_index(drop=True))
                    save("early_stop_truth.parquet", truth.loc[natural].reset_index(drop=True))
                else:
                    save("score_features.parquet", features.loc[natural].reset_index(drop=True))
                    save("score_truth.parquet", truth)
                    reference_info = config["reference_predictions"][fold]
                    if sha256_file(root / reference_info["path"]) != reference_info["sha256"]:
                        raise ValueError("The frozen full v7 reference changed.")
                    reference = aligned(features, pd.read_parquet(root / reference_info["path"]))
                    save("reference.parquet", reference)
                    metrics = evaluate(truth, reference, original["ranking_month_weights"] if fold == "seasonal" else None)
                    expected_metrics = read_json(root / config["reference_selection"])["candidates"][config["reference_candidate"]]["folds"][fold]["metrics"]
                    if abs(metrics["overall"]["rmse_seconds"] - expected_metrics["overall"]["rmse_seconds"]) > 1e-9:
                        raise ValueError("The complete v7 baseline does not reproduce.")
                    metrics_path = directory / "reference_metrics.json"
                    write_json(metrics_path, metrics)
                    paths["reference_metrics.json"] = metrics_path.relative_to(root).as_posix()
                    receipts[paths["reference_metrics.json"]] = sha256_file(metrics_path)
                del features, truth, routing, parts
                gc.collect()
            fold_receipts[fold] = {"identity": identity, "assignment_sha256": sha256_file(assignment_path), "files": receipts}
            for item in [j for j in config["jobs"] if j["fold"] == fold]:
                variant = config["variants"][item["variant"]]
                population = variant["fit_population"]
                chosen = {"fit_features": paths[f"fit_{population}_features.parquet"],
                          "fit_truth": paths[f"fit_{population}_truth.parquet"],
                          **{key: paths[key + ".parquet"] for key in ["early_stop_features", "early_stop_truth", "score_features", "score_truth", "reference"]},
                          "reference_metrics": paths["reference_metrics.json"]}
                report = {**item, "created_at_utc": utc_now(), "bundle_id": manifest["bundle_id"],
                    "feature_manifest_sha256": sha256_file(base / "manifest.json"), "model_code_sha256": source_hash,
                    "config_sha256": sha256_file(root / "configs/nm_mask_v1.json"), "variant_configuration": variant,
                    "job_input_code_sha256": sha256_file(root / "src/prc2026_nm_mask/data.py"),
                    "files": {p: receipts[p] for p in chosen.values()}, "paths": chosen,
                    "original_partition_identity": identity, "assignment_sha256": sha256_file(assignment_path),
                    "ranking_month_weights": original["ranking_month_weights"] if fold == "seasonal" else None,
                    "audit_used": False, "target_filter": "none", "prediction_postprocess": "none"}
                job_path = destination / "jobs" / (item["job_id"] + ".json")
                write_json(job_path, report)
                all_jobs.append({**item, "path": job_path.relative_to(root).as_posix(), "sha256": sha256_file(job_path)})
            print(f"masked_jobs_prepared fold={fold} natural_fit={identity['fit']['natural_rows']} augmented_fit={identity['fit']['original_rows']}", flush=True)
    report = {"created_at_utc": utc_now(), "bundle_id": manifest["bundle_id"], "jobs": all_jobs,
              "folds": fold_receipts, "model_code_sha256": source_hash, "resources": monitor.stats,
              "audit_used": False, "competition_models_fitted": 0}
    write_json(destination / "manifest.json", report)
    write_json(root / "docs/nm_mask/jobs.json", report)
    return report


def load_job(root, path):
    _, config = configurations(root)
    job = read_json(path)
    expected = next((item for item in config["jobs"] if item["job_id"] == job["job_id"]), None)
    if expected is None or any(job[key] != expected[key] for key in expected):
        raise ValueError("Unknown worker/job assignment.")
    if job["config_sha256"] != sha256_file(root / "configs/nm_mask_v1.json") or job["model_code_sha256"] != code_receipt(root)["sha256"]:
        raise ValueError("The declared model configuration or source changed.")
    if job["job_input_code_sha256"] != sha256_file(root / "src/prc2026_nm_mask/data.py"):
        raise ValueError("The separately bound job preparation/loading source changed.")
    if job["variant_configuration"] != config["variants"][job["variant"]] or job["audit_used"]:
        raise ValueError("Job settings changed or use reserved data.")
    for name, digest in job["files"].items():
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()) or sha256_file(target) != digest:
            raise ValueError("A portable job input changed.")
    frames = {key: pd.read_parquet(root / value) for key, value in job["paths"].items() if value.endswith(".parquet")}
    validate_truth(frames["fit_features"], frames["fit_truth"])
    validate_truth(frames["early_stop_features"], frames["early_stop_truth"])
    score = frames["score_truth"]
    validate_truth(frames["score_features"], score.loc[score.natural_missing].reset_index(drop=True))
    if not np.array_equal(score[ID], frames["reference"][ID]):
        raise ValueError("Complete reference and scoring labels do not align.")
    populations = job["original_partition_identity"]
    fitting_key = "natural_ids_sha256" if job["variant_configuration"]["fit_population"] == "natural" else "original_ids_sha256"
    if identity_hash(frames["fit_features"][ID]) != populations["fit"][fitting_key]:
        raise ValueError("Fitting identities changed.")
    if identity_hash(frames["early_stop_features"][ID]) != populations["early_stop"]["natural_ids_sha256"] or not frames["early_stop_truth"].natural_missing.all():
        raise ValueError("The common October stopping population changed.")
    if not frames["early_stop_truth"][TIME].dt.strftime("%Y-%m").eq("2025-10").all():
        raise ValueError("Stopping labels are not exclusively the declared October period.")
    if identity_hash(score[ID]) != populations["score"]["original_ids_sha256"]:
        raise ValueError("The complete scoring population changed.")
    ids = [set(frames[name][ID]) for name in ["fit_features", "early_stop_features", "score_features"]]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Fitting, stopping and scoring IDs overlap.")
    return job, frames
