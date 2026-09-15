"""Measure a fitted recipe's sensitivity to suspect pressure covariates, without labels."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026_weather.bundle import load_bundle
from prc2026_weather.inference import pipeline_prediction


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/weather/pressure_sensitivity.json"))
    args = parser.parse_args()
    directory = root/args.refit_run
    run = read_json(directory/"run.json")
    if run["stage"] != "refit" or run["audit_used"]:
        raise ValueError("Expected an audit-preserving refit.")
    verify_receipts(directory, run["artifacts"])
    base = root/(args.bundle or read_json(root/"data/weather/latest.json")["bundle"])
    manifest, _, _, _ = load_bundle(root, base)
    if run["bundle_id"] != manifest["bundle_id"]:
        raise ValueError("Refit and weather bundle differ.")
    ranking = pd.read_parquet(base/"features/ranking.parquet")
    pressure_bad = ranking["wx_pressure_hpa"].lt(850) | ranking["wx_pressure_hpa"].gt(1100)
    change_bad = ranking["wx_change60_pressure_hpa"].abs().gt(100) | pressure_bad
    chosen = ranking.loc[pressure_bad | change_bad].reset_index(drop=True)
    changed = chosen.copy()
    changed.loc[changed["wx_pressure_hpa"].lt(850) | changed["wx_pressure_hpa"].gt(1100), "wx_pressure_hpa"] = np.nan
    changed.loc[changed["wx_change60_pressure_hpa"].abs().gt(100) | chosen["wx_pressure_hpa"].lt(850) | chosen["wx_pressure_hpa"].gt(1100), "wx_change60_pressure_hpa"] = np.nan
    pipeline = read_json(directory/"pipeline.json")
    config = read_json(directory/"config.json")
    if len(chosen):
        original = pipeline_prediction(directory, pipeline, chosen, config["threads"])
        altered = pipeline_prediction(directory, pipeline, changed, config["threads"])
        saved = pd.read_parquet(directory/"ranking_predictions.parquet")
        expected = chosen[[ID]].merge(saved, on=ID, how="left", validate="one_to_one")["prediction"].to_numpy()
        if not np.allclose(original, expected, rtol=0, atol=1e-8):
            raise ValueError("Original native predictions do not reproduce.")
        delta = altered-original
    else:
        delta = np.empty(0)
    report = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "refit_run_sha256": sha256_file(directory/"run.json"),
              "bundle_id": manifest["bundle_id"], "ranking_rows": len(ranking), "affected_rows": len(chosen),
              "departure_labels_read": False, "audit_labels_used": False, "predictions_or_submission_modified": False,
              "perturbation": "Only suspect pressure outside 850-1100 hPa and hourly pressure changes over 100 hPa are replaced by missing values at inference.",
              "maximum_absolute_prediction_change_seconds": float(np.max(np.abs(delta))) if len(delta) else 0.,
              "rmse_difference_upper_bound_seconds": float(np.sqrt(np.dot(delta, delta)/len(ranking))),
              "bound_interpretation": "By the reverse triangle inequality, the absolute change in full-ranking RMSE between these two prediction vectors cannot exceed this value for any fixed hidden targets.",
              "limitation": "This is an input-sensitivity check of the fitted recipe. It does not evaluate a model retrained on pressure-filtered fitting data."}
    write_json(root/args.output, report)
    print(report)


if __name__ == "__main__":
    main()
