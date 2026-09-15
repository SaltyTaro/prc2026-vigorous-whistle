"""Score the seven declared v10 candidates on both complete folds against the exactly reconstructed v9.

V7 and the no-clock fallback come from hash-checked frozen runs; the v9 mixture is
refitted on the fold's fitting months exactly as in the v9 campaign and must
reproduce its recorded score; cat_long fold predictions are the frozen ladder
runs. Two small classifiers are the only native fits. Nothing is uploaded.
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
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import evaluate, identity_hash
from prc2026_lirf_mixture.features import classifier_features, label_modes, population_rows
from prc2026_lirf_mixture.model import apply_candidate, mixture_values, normal_component
from prc2026_postprocess.rules import row_context
from prc2026_v10.features import DENSITY_NAMES, density_features
from prc2026_v10.model import ITEM_NAMES, apply_items, bagging_seeds, fit_bagged_classifier, predict_bagged_modes, select_candidate
from run_lirf_mixture import fit_population_model, movements, score_population
from run_postprocess import PROTOCOL, aligned, check_parents, reconstruct_fold
from select_phase4 import paired_days

LADDER_RUNS = {"seasonal": "ladder_seasonal_cat_long", "forward": "ladder_forward_cat_long"}


def model_code_receipt(root: Path) -> dict:
    paths = sorted((root / "src/prc2026_v10").glob("*.py")) + sorted((root / "src/prc2026_lirf_mixture").glob("*.py"))
    paths += [root / "tools/run_v10.py", root / "tools/run_lirf_mixture.py", root / "tools/run_postprocess.py"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def load_configs(config: dict) -> dict:
    mixture_config = read_json(ROOT / config["lirf_mixture_config"])
    if object_hash(mixture_config) != read_json(ROOT / config["parents"]["lirf_mixture_selection"]["path"])["config_sha256"]:
        raise ValueError("The v9 declaration does not match its selection receipt.")
    if config["items"]["density_classifier"]["mixture_weight"] != mixture_config["candidates"]["mixture_75"]["mixture_weight"]:
        raise ValueError("The declared mixture weight differs from v9's.")
    return mixture_config


def fit_density_model(fitting: pd.DataFrame, config: dict, mixture_config: dict) -> tuple[dict, dict, pd.DataFrame]:
    population = population_rows(fitting)
    margin = mixture_config["turnaround_features"]["previous_arrival_margin_seconds"]
    features = classifier_features(population, fitting, margin).merge(density_features(population, fitting, config), on=ID, validate="one_to_one")
    labels = population[TARGET].to_numpy(dtype=float)
    interval = features["schedule_interval_s"].to_numpy()
    modes = label_modes(labels, interval, mixture_config)
    dates = population[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    model = fit_bagged_classifier(features, modes, dates, mixture_config, bagging_seeds(config, mixture_config))
    component = normal_component(labels, interval, modes, mixture_config)
    return model, component, features


def score_density(model: dict, component: dict, scoring_movements: pd.DataFrame, config: dict, mixture_config: dict):
    population = population_rows(scoring_movements)
    margin = mixture_config["turnaround_features"]["previous_arrival_margin_seconds"]
    features = classifier_features(population, scoring_movements, margin).merge(density_features(population, scoring_movements, config), on=ID, validate="one_to_one")
    probabilities = predict_bagged_modes(model, features)
    interval = features["schedule_interval_s"].to_numpy()
    return population, features, probabilities, mixture_values(probabilities, interval, component)


def cat_long_predictions(fold: str, ids: np.ndarray, config: dict) -> np.ndarray:
    parent = config["parents"][LADDER_RUNS[fold] + "_predictions"]
    run = read_json(ROOT / config["parents"][LADDER_RUNS[fold] + "_run"]["path"])
    ladder = read_json(ROOT / config["parents"]["ladder_selection"]["path"])
    card = ladder["development_runs"][config["cat_long_refit"]["variant"]][fold]
    if run["variant"] != config["cat_long_refit"]["variant"] or run["fold"] != fold or run["stage"] != "development" or run["audit_used"]:
        raise ValueError("The ladder run is not the declared cat_long development fit.")
    if run["run_id"] != card["run_id"] or sha256_file(ROOT / config["parents"][LADDER_RUNS[fold] + "_run"]["path"]) != card["run_sha256"]:
        raise ValueError("The ladder selection does not reference this run.")
    if run["bundle_id"] != config["cat_long_refit"]["bundle_id"] or run["score_ids_sha256"] != identity_hash(ids):
        raise ValueError("The ladder run's bundle or scoring identities differ.")
    if run["best_iterations"] != card["iterations"]:
        raise ValueError("Recorded iterations disagree.")
    return aligned(ids, ROOT / parent["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v10_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/v10/selection.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/v10/development_results.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    for path in [ROOT / args.output, ROOT / args.report]:
        if path.exists():
            raise ValueError(f"Preserve the original selection: {path}")
    if config["audit_used"] or "2025-12" in config["refit_months"] or list(config["order_of_application"]) != ITEM_NAMES + ["floor"]:
        raise ValueError("Declaration does not match the implemented items.")
    check_parents(config)
    mixture_config = load_configs(config)
    v9_selection = read_json(ROOT / config["parents"]["lirf_mixture_selection"]["path"])
    protocol = read_json(PROTOCOL)["folds"]
    threshold = mixture_config["mixture"]["schedule_interval_threshold_seconds"]
    iterations = [read_json(ROOT / config["parents"][LADDER_RUNS[fold] + "_run"]["path"])["best_iterations"] for fold in config["development_folds"]]
    if config["cat_long_refit"]["iterations"] != int(np.rint(np.median(iterations))):
        raise ValueError("Declared refit iterations are not the rounded median of the development counts.")
    folds, candidates, reference_metrics = {}, {}, {}
    for fold in config["development_folds"]:
        scoring, truth, v7, fallback, _, expected = reconstruct_fold(fold, mixture_config)
        ids = scoring[ID].to_numpy()
        fitting = movements(protocol[fold]["fit_months"])
        scoring_movements = movements(protocol[fold]["score_months"])
        v9_model, v9_component, v9_summary = fit_population_model(fitting, mixture_config)
        population, _, v9_probabilities, interval, v9_mixture, index = score_population(v9_model, v9_component, scoring_movements, ids, mixture_config)
        floor = v9_summary["floor_seconds"]
        v9, v9_counts = apply_candidate(v7, index, interval, v9_mixture, config["items"]["density_classifier"]["mixture_weight"], threshold, floor)
        reference_metrics[fold] = evaluate(truth, pd.DataFrame({ID: ids, "prediction": v9}))
        recorded = v9_selection["candidates"][v9_selection["selected_candidate"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
        if abs(reference_metrics[fold]["overall"]["rmse_seconds"] - recorded) > 1e-6 or abs(recorded - config["incumbent"]["development_rmse_seconds"][fold]) > 1e-6:
            raise ValueError(f"Reconstructed v9 does not reproduce its recorded {fold} score.")
        density_model, density_component, _ = fit_density_model(fitting, config, mixture_config)
        if density_component != v9_component:
            raise ValueError("The normal component must be identical to v9's.")
        d_population, d_features, d_probabilities, density_mixture = score_density(density_model, density_component, scoring_movements, config, mixture_config)
        if not np.array_equal(d_population[ID].to_numpy(), population[ID].to_numpy()):
            raise ValueError("Density scoring population differs from the v9 population.")
        context = row_context(scoring)
        clock = context["clock_available"].to_numpy()
        negative = clock & (context["proxy_s"].to_numpy() < 0)
        cat_long = cat_long_predictions(fold, ids, config)
        truth_modes = label_modes(population[TARGET].to_numpy(dtype=float), interval, mixture_config)
        folds[fold] = {"rows": int(len(ids)), "score_ids_sha256": identity_hash(ids), "frozen_scores": expected,
                       "v9": {"rounds": v9_model["rounds"], "row_counts": v9_counts, "fitting_population_rows": v9_summary["fitting_population_rows"],
                              "scoring_fill_auc": float(_auc(truth_modes == 0, v9_probabilities[:, 0]))},
                       "density": {"seeds": density_model["seeds"], "rounds": density_model["rounds"], "inner_best_iterations": density_model["inner_best_iterations"],
                                   "mode_counts": density_model["mode_counts"], "scoring_fill_auc": float(_auc(truth_modes == 0, d_probabilities[:, 0])),
                                   "feature_importance_gain": {m["seed"]: dict(zip(m["booster"].feature_name(), map(float, m["booster"].feature_importance("gain")))) for m in density_model["members"]}},
                       "floor_seconds": floor, "clock_rows": int(clock.sum()), "negative_proxy_rows": int(negative.sum()),
                       "scoring_population_rows": int(len(population)), "scoring_population_modes": {str(k): int((truth_modes == k).sum()) for k in range(3)},
                       "cat_long_iterations": int(read_json(ROOT / config["parents"][LADDER_RUNS[fold] + "_run"]["path"])["best_iterations"])}
        for name, items in config["candidates"].items():
            values, counts = apply_items(v7, items, clock=clock, cat_long=cat_long, negative_proxy=negative, fallback=fallback, population_index=index,
                                         interval=interval, v9_mixture=v9_mixture, density_mixture=density_mixture, config=config, threshold=threshold, floor=floor)
            prediction = pd.DataFrame({ID: ids, "prediction": values})
            metrics = evaluate(truth, prediction)
            ratio = metrics["overall"]["rmse_seconds"] / reference_metrics[fold]["overall"]["rmse_seconds"]
            paired = paired_days(truth, prediction, pd.DataFrame({ID: ids, "prediction": v9}), config["seed"])
            paired["comparison"] = "candidate minus reconstructed v9"
            pop_sse = {str(k): float((((values[index] - population[TARGET].to_numpy(dtype=float)) ** 2)[truth_modes == k]).sum()) for k in range(3)}
            v9_sse = {str(k): float((((v9[index] - population[TARGET].to_numpy(dtype=float)) ** 2)[truth_modes == k]).sum()) for k in range(3)}
            candidates.setdefault(name, {"items": items, "folds": {}})["folds"][fold] = {
                "metrics": metrics, "rmse_ratio_to_v9": ratio, "row_counts": counts, "paired_days": paired,
                "changed_rows_vs_v9": int((values != v9).sum()), "population_sse_by_true_mode": {"candidate": pop_sse, "v9": v9_sse}}
    selected = select_candidate(candidates, config["development_folds"], config["minimum_improvement_fraction_each_fold"])
    report = {"created_at_utc": utc_now(), "policy": config["prediction_postprocess"], "configuration": config, "config_sha256": object_hash(config),
              "model_code": model_code_receipt(ROOT), "audit_used": False, "native_fit_count": len(config["development_folds"]) * (1 + len(bagging_seeds(config, mixture_config))), "density_feature_names": DENSITY_NAMES,
              "folds": folds, "reference_metrics": reference_metrics, "candidates": candidates, "selected_candidate": selected,
              "selected_items": candidates[selected]["items"] if selected else None, "rule": config["selection"], "environment": environment(ROOT)}
    for path in [ROOT / args.output, ROOT / args.report]:
        write_json(path, report)
    print({"selected": selected,
           "candidates": {name: {fold: (round(item["folds"][fold]["metrics"]["overall"]["rmse_seconds"], 4), round(100 * (item["folds"][fold]["rmse_ratio_to_v9"] - 1), 3),
                                        [round(v, 3) for v in item["folds"][fold]["paired_days"]["rmse_difference_95pct_interval_seconds"]]) for fold in item["folds"]} for name, item in candidates.items()},
           "v9": {fold: round(reference_metrics[fold]["overall"]["rmse_seconds"], 4) for fold in reference_metrics},
           "folds": {fold: {"v9_rounds": folds[fold]["v9"]["rounds"], "density_rounds": folds[fold]["density"]["rounds"], "fill_auc": (round(folds[fold]["v9"]["scoring_fill_auc"], 4), round(folds[fold]["density"]["scoring_fill_auc"], 4))} for fold in folds}})


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
