"""Score the declared extreme-row post-processing candidates on both complete folds.

Reconstructs the frozen v7 development predictions and the frozen no-clock
fallback from saved run artifacts (hash-checked), fits rule parameters on the
fold's fitting labels only, scores every declared candidate and applies the
declared selection rule. No native model is fitted and nothing is uploaded.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import evaluate, identity_hash
from prc2026_ladder.inference import compose
from prc2026_postprocess.rules import POLICY, RAW_COLUMNS, apply_rules, fit_parameters, row_context
from select_phase4 import paired_days

RAW = ROOT / "data/raw"
MONTH_FILES = {f"2025-{m:02}": RAW / f"training_2025-{m:02}-01_{2025 + (m == 12)}-{(m % 12) + 1:02}-01.parquet" for m in range(1, 13)}
PROTOCOL = ROOT / "configs/validation_v1.json"
SURFACE_RUNS = ROOT / "runs/azure_arcw0_surface_airports"
TIMELINE_RUNS = ROOT / "runs/azure_arcw0_timeline"
WEATHER_RUNS = ROOT / "runs/azure_arcw0_weather"


def model_code_receipt(root: Path) -> dict:
    paths = sorted((root / "src/prc2026_postprocess").glob("*.py")) + [root / "tools/run_postprocess.py"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def check_parents(config: dict) -> None:
    for name, parent in config["parents"].items():
        if sha256_file(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent artifact changed: {name}")


def departures(months: list[str], columns: list[str]) -> pd.DataFrame:
    if "2025-12" in months:
        raise ValueError("December remains reserved.")
    frames = []
    for month in months:
        frame = pd.read_parquet(MONTH_FILES[month], columns=sorted(set(columns) | {PHASE}))
        frames.append(frame.loc[frame[PHASE].eq("DEP")].drop(columns=[PHASE]))
    return pd.concat(frames, ignore_index=True)


def aligned(reference_ids: np.ndarray, path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    if frame[ID].duplicated().any():
        raise ValueError(f"Duplicate prediction IDs in {path}")
    series = frame.set_index(ID)["prediction"].reindex(reference_ids)
    if series.isna().any():
        raise ValueError(f"Predictions in {path} do not cover the fold.")
    return series.to_numpy(dtype=float)


def reconstruct_fold(fold: str, config: dict):
    """Frozen v7 candidate and frozen no-clock fallback on the fold's complete scoring set."""
    surface = read_json(ROOT / config["parents"]["incumbent_selection"]["path"])
    timeline = read_json(ROOT / config["parents"]["timeline_selection"]["path"])
    weather = read_json(ROOT / config["parents"]["fallback_selection"]["path"])
    recipe = surface["candidates"][surface["selected_candidate"]]
    if surface["selected_candidate"] != config["incumbent"]["recipe"] or weather["selected_candidate"] != config["fallback_recipe"]:
        raise ValueError("Parent selections disagree with the declaration.")
    ref_dir = SURFACE_RUNS / "references" / fold
    receipt = read_json(ref_dir / "reference.json")
    verify_receipts(ref_dir, receipt["artifacts"])
    reference = pd.read_parquet(ref_dir / "v6.parquet")
    ids = reference[ID].to_numpy()
    components = {}
    surface_run = SURFACE_RUNS / surface["development_runs"][fold]["run_id"]
    if sha256_file(surface_run / "run.json") != surface["development_runs"][fold]["run_sha256"]:
        raise ValueError("Frozen surface run changed.")
    verify_receipts(surface_run, read_json(surface_run / "run.json")["artifacts"])
    components["surface_airports"] = aligned(ids, surface_run / "predictions.parquet")
    for name in ["lgb_combined", "cat_combined"]:
        run = TIMELINE_RUNS / timeline["development_runs"][name][fold]["run_id"]
        if sha256_file(run / "run.json") != timeline["development_runs"][name][fold]["run_sha256"]:
            raise ValueError("Frozen timeline run changed.")
        verify_receipts(run, read_json(run / "run.json")["artifacts"])
        components[name] = aligned(ids, run / "predictions.parquet")
    weather_run = WEATHER_RUNS / weather["development_runs"]["cat_weather"][fold]["run_id"]
    fallback = aligned(ids, weather_run / "predictions.parquet")
    protocol = read_json(PROTOCOL)["folds"][fold]
    scoring = departures(protocol["score_months"], RAW_COLUMNS + [TARGET])
    scoring = scoring.set_index(ID).reindex(ids).reset_index()
    if scoring[TARGET].isna().any() or len(scoring) != len(ids):
        raise ValueError("Raw scoring rows do not cover the frozen reference population.")
    available = row_context(scoring)["clock_available"].to_numpy()
    v7 = compose(reference["prediction"], components, available, recipe["routes"], recipe["new_weight"])
    truth = scoring[[ID, TIME, TARGET]].assign(airport=scoring["ADEP_mvt"])
    expected = {"v7": recipe["folds"][fold]["metrics"]["overall"]["rmse_seconds"],
                "fallback": weather["candidates"][config["fallback_recipe"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]}
    for name, values in [("v7", v7), ("fallback", fallback)]:
        metrics = evaluate(truth, pd.DataFrame({ID: ids, "prediction": values}))
        if abs(metrics["overall"]["rmse_seconds"] - expected[name]) > 1e-9:
            raise ValueError(f"Reconstructed {name} does not reproduce the frozen {fold} score.")
        if name == "v7" and metrics["evaluation_ids_sha256"] != recipe["folds"][fold]["metrics"]["evaluation_ids_sha256"]:
            raise ValueError("Scoring identities differ from the frozen selection.")
    fitting = departures(protocol["fit_months"], RAW_COLUMNS + [TARGET])
    return scoring, truth, v7, fallback, fitting, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/postprocess_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/postprocess/selection.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/postprocess/development_results.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    for path in [ROOT / args.output, ROOT / args.report]:
        if path.exists():
            raise ValueError(f"Preserve the original selection: {path}")
    if config["audit_used"] or "2025-12" in config["refit_months"] or list(config["rule_order"]) != ["route_negative_proxy", "hedge_long_schedule", "floor"]:
        raise ValueError("Declaration does not match the implemented rules.")
    check_parents(config)
    folds, parameters, candidates, reference_metrics = {}, {}, {}, {}
    scored = {}
    for fold in config["development_folds"]:
        scoring, truth, v7, fallback, fitting, expected = reconstruct_fold(fold, config)
        parameters[fold] = fit_parameters(fitting, config)
        reference_metrics[fold] = evaluate(truth, pd.DataFrame({ID: scoring[ID], "prediction": v7}))
        folds[fold] = {"rows": int(len(scoring)), "score_ids_sha256": identity_hash(scoring[ID]),
                       "fitting_rows": int(len(fitting)), "fitting_ids_sha256": identity_hash(fitting[ID]),
                       "frozen_scores": expected}
        for name, rules in config["candidates"].items():
            values, counts = apply_rules(scoring, v7, fallback, parameters[fold], rules)
            prediction = pd.DataFrame({ID: scoring[ID], "prediction": values})
            metrics = evaluate(truth, prediction)
            ratio = metrics["overall"]["rmse_seconds"] / reference_metrics[fold]["overall"]["rmse_seconds"]
            paired = paired_days(truth, prediction, pd.DataFrame({ID: scoring[ID], "prediction": v7}), config["seed"])
            paired["comparison"] = "candidate minus reconstructed v7"
            candidates.setdefault(name, {"rules": rules, "folds": {}})["folds"][fold] = {
                "metrics": metrics, "rmse_ratio_to_v7": ratio, "row_counts": counts, "paired_days": paired}
            scored[(name, fold)] = values
    for name, item in candidates.items():
        ratios = [item["folds"][fold]["rmse_ratio_to_v7"] for fold in config["development_folds"]]
        item["mean_relative_rmse"] = float(np.mean(ratios))
        item["eligible"] = bool("floor" in item["rules"] and all(r <= 1.0 for r in ratios))
    eligible = [name for name, item in candidates.items() if item["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else "floor"
    report = {"created_at_utc": utc_now(), "policy": POLICY, "configuration": config, "config_sha256": object_hash(config),
              "model_code": model_code_receipt(ROOT), "audit_used": False, "native_fit_count": 0,
              "folds": folds, "fitted_parameters": parameters, "reference_metrics": reference_metrics,
              "candidates": candidates, "selected_candidate": selected, "selected_rules": candidates[selected]["rules"],
              "rule": config["selection"], "environment": environment(ROOT)}
    for path in [ROOT / args.output, ROOT / args.report]:
        write_json(path, report)
    print({"selected": selected, "parameters": {f: {k: parameters[f][k] for k in ["floor_seconds", "one_day_share", "one_day_taxi_median_seconds", "long_schedule_group_rows"]} for f in parameters},
           "candidates": {name: {fold: round(item["folds"][fold]["metrics"]["overall"]["rmse_seconds"], 6) for fold in item["folds"]} for name, item in candidates.items()},
           "v7": {fold: round(reference_metrics[fold]["overall"]["rmse_seconds"], 6) for fold in reference_metrics}})


if __name__ == "__main__":
    main()
