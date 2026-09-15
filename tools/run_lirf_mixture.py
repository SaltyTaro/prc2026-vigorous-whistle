"""Fit and score the declared LIRF turnaround mixture on both complete folds; select by the declared rule.

The classifier and the normal component use the fold's fitting months only. The
incumbent v7 development predictions are reconstructed from hash-checked frozen
runs (their scores must reproduce exactly). Nothing is uploaded.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import evaluate, identity_hash
from prc2026_lirf_mixture.features import ARRIVAL_COLUMNS, DEPARTURE_COLUMNS, classifier_features, label_modes, population_rows
from prc2026_lirf_mixture.model import apply_candidate, fit_classifier, mixture_values, normal_component, predict_modes
from run_postprocess import MONTH_FILES, PROTOCOL, check_parents, reconstruct_fold
from select_phase4 import paired_days

MOVEMENT_COLUMNS = sorted(set(DEPARTURE_COLUMNS + ARRIVAL_COLUMNS + [TARGET]))


def model_code_receipt(root: Path) -> dict:
    paths = sorted((root / "src/prc2026_lirf_mixture").glob("*.py")) + [root / "tools/run_lirf_mixture.py"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def movements(months: list[str]) -> pd.DataFrame:
    if "2025-12" in months:
        raise ValueError("December remains reserved.")
    return pd.concat([pd.read_parquet(MONTH_FILES[m], columns=MOVEMENT_COLUMNS) for m in months], ignore_index=True)


def fit_population_model(fitting: pd.DataFrame, config: dict) -> tuple[dict, dict, dict]:
    population = population_rows(fitting)
    features = classifier_features(population, fitting, config["turnaround_features"]["previous_arrival_margin_seconds"])
    labels = population[TARGET].to_numpy(dtype=float)
    interval = features["schedule_interval_s"].to_numpy()
    modes = label_modes(labels, interval, config)
    dates = population[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    model = fit_classifier(features, modes, dates, config)
    component = normal_component(labels, interval, modes, config)
    summary = {"fitting_population_rows": int(len(population)), "fitting_population_ids_sha256": identity_hash(population[ID]),
               "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"], "mode_counts": model["mode_counts"],
               "normal_component": component, "floor_seconds": float(fitting.loc[fitting[PHASE].eq("DEP"), TARGET].min())}
    return model, component, summary


def score_population(model: dict, component: dict, scoring_movements: pd.DataFrame, ids: np.ndarray, config: dict):
    population = population_rows(scoring_movements)
    features = classifier_features(population, scoring_movements, config["turnaround_features"]["previous_arrival_margin_seconds"])
    probabilities = predict_modes(model, features)
    interval = features["schedule_interval_s"].to_numpy()
    mixture = mixture_values(probabilities, interval, component)
    position = pd.Series(np.arange(len(ids)), index=ids)
    index = position.reindex(population[ID].to_numpy()).to_numpy()
    if np.isnan(index).any():
        raise ValueError("Population rows are not all in the scoring population.")
    return population, features, probabilities, interval, mixture, index.astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lirf_mixture_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/lirf_mixture/selection.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/lirf_mixture/development_results.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    for path in [ROOT / args.output, ROOT / args.report]:
        if path.exists():
            raise ValueError(f"Preserve the original selection: {path}")
    if config["audit_used"] or "2025-12" in config["refit_months"]:
        raise ValueError("Declaration does not respect the reserved period.")
    check_parents(config)
    protocol = read_json(PROTOCOL)["folds"]
    folds, candidates, reference_metrics = {}, {}, {}
    for fold in config["development_folds"]:
        scoring, truth, v7, _, _, expected = reconstruct_fold(fold, config)
        ids = scoring[ID].to_numpy()
        fitting = movements(protocol[fold]["fit_months"])
        model, component, summary = fit_population_model(fitting, config)
        scoring_movements = movements(protocol[fold]["score_months"])
        population, features, probabilities, interval, mixture, index = score_population(model, component, scoring_movements, ids, config)
        reference_metrics[fold] = evaluate(truth, pd.DataFrame({ID: ids, "prediction": v7}))
        truth_modes = label_modes(population[TARGET].to_numpy(dtype=float), interval, config)
        folds[fold] = {**summary, "rows": int(len(ids)), "score_ids_sha256": identity_hash(ids), "frozen_scores": expected,
                       "scoring_population_rows": int(len(population)), "scoring_population_modes": {str(k): int((truth_modes == k).sum()) for k in range(3)},
                       "scoring_fill_auc": float(_auc(truth_modes == 0, probabilities[:, 0]))}
        for name, spec in config["candidates"].items():
            values, counts = apply_candidate(v7, index, interval, mixture, spec["mixture_weight"],
                                             config["mixture"]["schedule_interval_threshold_seconds"], summary["floor_seconds"])
            prediction = pd.DataFrame({ID: ids, "prediction": values})
            metrics = evaluate(truth, prediction)
            ratio = metrics["overall"]["rmse_seconds"] / reference_metrics[fold]["overall"]["rmse_seconds"]
            paired = paired_days(truth, prediction, pd.DataFrame({ID: ids, "prediction": v7}), config["seed"])
            paired["comparison"] = "candidate minus reconstructed v7"
            pop_sse = {str(k): float((((values[index] - population[TARGET].to_numpy(dtype=float)) ** 2)[truth_modes == k]).sum()) for k in range(3)}
            v7_sse = {str(k): float((((v7[index] - population[TARGET].to_numpy(dtype=float)) ** 2)[truth_modes == k]).sum()) for k in range(3)}
            candidates.setdefault(name, {"spec": spec, "folds": {}})["folds"][fold] = {
                "metrics": metrics, "rmse_ratio_to_v7": ratio, "row_counts": counts, "paired_days": paired,
                "population_sse_by_true_mode": {"candidate": pop_sse, "v7": v7_sse}}
    margin = config["minimum_improvement_fraction_each_fold"]
    for name, item in candidates.items():
        ratios = [item["folds"][fold]["rmse_ratio_to_v7"] for fold in config["development_folds"]]
        item["mean_relative_rmse"] = float(np.mean(ratios))
        item["eligible"] = bool(all(r <= 1 - margin for r in ratios))
    eligible = [name for name, item in candidates.items() if item["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
    report = {"created_at_utc": utc_now(), "policy": config["prediction_postprocess"], "configuration": config, "config_sha256": object_hash(config),
              "model_code": model_code_receipt(ROOT), "audit_used": False, "native_fit_count": 2, "folds": folds,
              "reference_metrics": reference_metrics, "candidates": candidates, "selected_candidate": selected,
              "rule": config["selection"], "environment": environment(ROOT)}
    for path in [ROOT / args.output, ROOT / args.report]:
        write_json(path, report)
    print({"selected": selected,
           "candidates": {name: {fold: round(item["folds"][fold]["metrics"]["overall"]["rmse_seconds"], 4) for fold in item["folds"]} for name, item in candidates.items()},
           "v7": {fold: round(reference_metrics[fold]["overall"]["rmse_seconds"], 4) for fold in reference_metrics},
           "folds": {fold: {k: folds[fold][k] for k in ["rounds", "mode_counts", "scoring_population_rows", "scoring_fill_auc", "floor_seconds"]} for fold in folds}})


def _auc(positive: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


if __name__ == "__main__":
    main()
