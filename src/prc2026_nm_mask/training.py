"""Run one fixed model/fold assignment on its authorized Azure worker."""
from pathlib import Path
import platform
import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, write_json, sha256_file, utc_now
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash, evaluate
from .data import load_job
from .common import sample_weights, compose, code_receipt
from .features import offset
from .native import fit_native, save_native, predict_native, environment, load_native


def train_job(root, job_path):
    job, data = load_job(root, job_path)
    if platform.system() != "Linux" or platform.node() != job["worker"]:
        raise ValueError("Competition model fitting is restricted to its assigned Azure worker.")
    directory = root / "runs/nm_mask" / job["job_id"]
    if directory.exists():
        raise ValueError("A job output already exists; never retrain over it.")
    directory.mkdir(parents=True)
    weights = sample_weights(data["fit_truth"].natural_missing.to_numpy(dtype=bool), job["variant_configuration"]["fit_population"])
    with ResourceMonitor() as monitor:
        model, encoder, history, iterations, parameters = fit_native(
            data["fit_features"], data["fit_truth"][TARGET].to_numpy(dtype=float) - offset(data["fit_features"]), weights,
            data["early_stop_features"], data["early_stop_truth"][TARGET].to_numpy(dtype=float) - offset(data["early_stop_features"]),
            job["variant_configuration"])
        save_native(directory, model, encoder, history, iterations, parameters)
        values = predict_native(model, encoder, data["score_features"], iterations)
        prediction = pd.DataFrame({ID: data["score_features"][ID], "prediction": values})
        prediction.to_parquet(directory / "component_predictions.parquet", index=False, compression="zstd")
        natural = data["score_truth"].natural_missing.to_numpy(dtype=bool)
        full = compose(data["reference"].prediction.to_numpy(), values, natural, 1.)
        complete = pd.DataFrame({ID: data["score_truth"][ID], "prediction": full})
        complete.to_parquet(directory / "predictions.parquet", index=False, compression="zstd")
        metrics = evaluate(data["score_truth"], complete, job["ranking_month_weights"])
        write_json(directory / "metrics.json", metrics)
        pd.DataFrame({ID: data["fit_features"][ID], "sample_weight": weights,
                      "natural_missing": data["fit_truth"].natural_missing}).to_parquet(directory / "fit_ids_weights.parquet", index=False, compression="zstd")
        data["early_stop_features"][[ID]].to_parquet(directory / "early_stop_ids.parquet", index=False)
        reloaded, saved_encoder, node = load_native(directory)
        difference = float(np.max(np.abs(predict_native(reloaded, saved_encoder, data["score_features"], node["iterations"]) - values)))
        if difference > 1e-8:
            raise ValueError("Native save/reload changed the complete eligible prediction set.")
    artifacts = [{"path": p.name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                 for p in sorted(directory.iterdir()) if p.is_file()]
    natural_fit = data["fit_truth"].natural_missing.to_numpy(dtype=bool)
    report = {"created_at_utc": utc_now(), "worker": job["worker"], "job_id": job["job_id"],
        "fold": job["fold"], "variant": job["variant"], "iterations": iterations,
        "model_code": code_receipt(root), "job_sha256": sha256_file(job_path),
        "feature_bundle_id": job["bundle_id"], "artifacts": artifacts, "environment": environment(),
        "fit_rows": len(weights), "fit_ids_sha256": identity_hash(data["fit_features"][ID]),
        "natural_fit_rows": int(natural_fit.sum()), "augmented_fit_rows": int((~natural_fit).sum()),
        "natural_weight_sum": float(weights[natural_fit].sum()), "augmented_weight_sum": float(weights[~natural_fit].sum()),
        "early_stop_rows": len(data["early_stop_features"]), "early_stop_ids_sha256": identity_hash(data["early_stop_features"][ID]),
        "complete_score_rows": len(complete), "complete_score_ids_sha256": identity_hash(complete[ID]),
        "native_prediction_rows": len(values), "native_reload_maximum_difference_seconds": difference,
        "complete_rmse_seconds": metrics["overall"]["rmse_seconds"], "resources": monitor.stats,
        "audit_used": False, "prediction_postprocess": "none", "target_filter": "none", "new_native_fits": 1}
    write_json(directory / "run.json", report)
    print(f"job_complete worker={job['worker']} job={job['job_id']} iterations={iterations} complete_rmse={report['complete_rmse_seconds']:.9f}", flush=True)
    return report
