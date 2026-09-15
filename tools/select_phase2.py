"""Compare both frozen folds, diagnose errors, and freeze one refit recipe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.phase1 import load_partition
from prc2026.validation import evaluate
from prc2026_models.experiments import model_code_receipt


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/azure_arcw0_phase2"))
    parser.add_argument("--bundle", type=Path, default=Path("data/phase1/58d621b245de3f5e12dd"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase2/selection.json"))
    args = parser.parse_args()
    config = read_json(root / args.config)
    config_hash = object_hash(config)
    model_code = model_code_receipt(root)["sha256"]
    base = root / args.bundle
    bundle = read_json(base / "manifest.json")
    verify_receipts(base, bundle["artifacts"])
    baseline = read_json(root / "docs/phase1/benchmark_summary.json")
    if baseline["bundle_id"] != bundle["bundle_id"]:
        raise ValueError("Airport-mean reference uses a different feature/split bundle.")
    selected_runs = {}
    for filename in sorted((root / args.runs).glob("*/run.json")):
        run = read_json(filename)
        if run.get("stage") != "development" or run["config_sha256"] != config_hash:
            continue
        if run["bundle_id"] != bundle["bundle_id"] or run["model_code"]["sha256"] != model_code:
            raise ValueError("Run code or feature bundle differs from the configured baseline.")
        if run["audit_used"] or run["fold"] not in config["development_folds"]:
            raise ValueError("Audit data appears in baseline selection.")
        verify_receipts(filename.parent, run["artifacts"])
        selected_runs[(run["model_kind"], run["fold"])] = (filename.parent, run)
    expected = {(kind, fold) for kind in ["lightgbm", "catboost"] for fold in config["development_folds"]}
    if set(selected_runs) != expected:
        raise ValueError("Selection requires both models on both declared development folds.")
    catalog = pd.read_parquet(base / "train_catalog.parquet")
    truths = {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(base / "splits" / f"{fold}.parquet")
        _, truths[fold] = load_partition(base, catalog, assignment, "score")
    candidates, predictions = {}, {}
    for kind in ["lightgbm", "catboost"]:
        folds, ratios = {}, []
        for fold in config["development_folds"]:
            path, run = selected_runs[(kind, fold)]
            pred = pd.read_parquet(path / "predictions.parquet")
            metrics = evaluate(truths[fold], pred, bundle["ranking_month_weights"] if fold == "seasonal" else None)
            saved = read_json(path / "metrics.json")
            if not np.isclose(metrics["overall"]["rmse_seconds"], saved["overall"]["rmse_seconds"], rtol=1e-12):
                raise ValueError("Saved metrics do not reproduce from predictions.")
            reference = baseline["benchmarks"][fold]["metrics"]["overall"]["rmse_seconds"]
            ratio = metrics["overall"]["rmse_seconds"] / reference
            ratios.append(ratio)
            folds[fold] = {"run_id": run["run_id"], "best_iterations": run["best_iterations"],
                           "early_stop_rmse_seconds": run["early_stop_rmse_seconds"],
                           "rmse_ratio_to_airport_mean": ratio, "metrics": metrics,
                           "resources": run["resources"], "environment": run["environment"]}
            predictions[(kind, fold)] = pred
        candidates[kind] = {"folds": folds, "mean_relative_rmse": float(np.mean(ratios)),
                            "improves_both_folds": all(ratio < 1 for ratio in ratios)}
    eligible = [kind for kind, candidate in candidates.items() if candidate["improves_both_folds"]]
    if not eligible:
        raise ValueError("Neither baseline improves both folds; do not submit without further work.")
    chosen = min(eligible, key=lambda kind: candidates[kind]["mean_relative_rmse"])
    other = "catboost" if chosen == "lightgbm" else "lightgbm"
    uncertainty = {}
    for fold in config["development_folds"]:
        truth = truths[fold]
        merged = truth[[ID, TIME, TARGET]].merge(predictions[(chosen, fold)], on=ID, validate="one_to_one")
        merged = merged.rename(columns={"prediction": "chosen"})
        merged = merged.merge(predictions[(other, fold)], on=ID, validate="one_to_one")
        merged["day"] = merged[TIME].dt.strftime("%Y-%m-%d")
        merged["chosen_sse"] = (merged["chosen"] - merged[TARGET]) ** 2
        merged["other_sse"] = (merged["prediction"] - merged[TARGET]) ** 2
        grouped = merged.groupby("day").agg(chosen_sse=("chosen_sse", "sum"),
                                             other_sse=("other_sse", "sum"), rows=(ID, "count"))
        rng = np.random.default_rng(config["seed"])
        indexes = rng.integers(0, len(grouped), size=(2000, len(grouped)))
        rows = grouped["rows"].to_numpy()[indexes].sum(axis=1)
        delta = (np.sqrt(grouped["chosen_sse"].to_numpy()[indexes].sum(axis=1) / rows)
                 - np.sqrt(grouped["other_sse"].to_numpy()[indexes].sum(axis=1) / rows))
        uncertainty[fold] = {
            "comparison": f"{chosen} minus {other}", "paired_day_bootstrap_replicates": 2000,
            "days": len(grouped), "rmse_difference_95pct_interval_seconds":
                [float(x) for x in np.quantile(delta, [0.025, 0.975])],
            "interpretation": "Descriptive uncertainty on reused development data; not an independent final test",
        }
    iterations = int(np.rint(np.median([x["best_iterations"] for x in candidates[chosen]["folds"].values()])))
    selection = {
        "created_at_utc": utc_now(), "selected_model": chosen, "refit_iterations": iterations,
        "refit_months": config["refit_months"], "audit_used": False,
        "rule": config["selection"], "bundle_id": bundle["bundle_id"],
        "config_sha256": config_hash, "model_code_sha256": model_code,
        "selection_code_sha256": sha256_file(Path(__file__)),
        "candidates": candidates, "paired_day_uncertainty": uncertainty,
    }
    write_json(root / args.output, selection)
    write_json(root / "docs/phase2/development_results.json", selection)
    print(f"Selected {chosen}; refit {iterations} iterations on January-November; December remains reserved.")
    for kind, candidate in candidates.items():
        print(kind, {fold: value["metrics"]["overall"]["rmse_seconds"] for fold, value in candidate["folds"].items()})


if __name__ == "__main__":
    main()
