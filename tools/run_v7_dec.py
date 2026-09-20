"""Cross-fit and twelve-month refit of v7's twelve timeline components (the frozen `full_timeline_v6_100` recipe) with December 2025.

v7's clock-present prediction is 0.25 * CatBoost global + 0.25 * LightGBM global + 0.5 * airport LightGBM, all NM-residual targets on
the frozen 276 timeline predictors, fitted on January-November 2025 at the frozen October-selected counts (configs/surface_refit_v1.json).
This runner fits the same twelve components with the same specs, settings, seed and counts (prc2026_timeline.native.fit_node, imported
frozen) on the v23 twelve-month population (the frozen January-November refit rows plus the opened December build) in two modes:

  --fold A|B|C|D   cross-fit: the fold's three months are held out, the other nine fitted; predictions for every clock-present departure
                   of the held-out months -> <output>/<fold>/predictions.parquet [ID, month, prediction]. Together the four folds give an
                   out-of-sample v7 for every 2025 month (the residual stage's training input; missing-clock rows have no cross-fit value).
  --fold refit     all twelve months fitted; predictions for the clock-present ranking departures -> <output>/refit/ranking_predictions.parquet
                   (NaN on missing-clock rows, which keep the frozen v6 values exactly as v7 does).

Nothing is early-stopped, nothing is uploaded, no label of a held-out month is read for fitting. Fold months (declared):
A = 01/05/09, B = 02/06/10, C = 03/07/11, D = 04/08/12.

Correction (20 September, after the four folds' twelve components were fitted): the first version tagged the prediction rows with a string column
named `month`, which is also the name of a frozen timeline predictor, so the prediction stage failed on every fold before any value was computed
(`launch.log` of each fold keeps the traceback). The tag is now `held_month`; `--resume` performs the prediction stage from the saved nodes of a
run directory whose twelve components are complete (nodes.json present, no run.json), which is recorded in run.json. The fitting code is unchanged.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import ResourceMonitor, environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.features import route_mask
from prc2026_surface_airports.inference import airport_prediction
from prc2026_surface_refit.guards import refit_assignment
from prc2026_surface_refit.inference import WEIGHTS
from prc2026_timeline.bundle import load_bundle
from prc2026_timeline.features import FEATURE_NAMES
from prc2026_timeline.inference import node_prediction
from prc2026_timeline.native import fit_node

FOLDS = {"A": ["2025-01", "2025-05", "2025-09"], "B": ["2025-02", "2025-06", "2025-10"], "C": ["2025-03", "2025-07", "2025-11"], "D": ["2025-04", "2025-08", "2025-12"], "refit": []}
TRAINING_MONTHS = [f"2025-{m:02}" for m in range(1, 12)]
DECEMBER, DECEMBER_FILE = "2025-12", "training_2025-12-01_2026-01-01.parquet"
FROZEN = ["src/prc2026_timeline/native.py", "src/prc2026_timeline/encoding.py", "src/prc2026_timeline/inference.py", "src/prc2026_timeline/features.py",
          "src/prc2026_surface_airports/inference.py", "src/prc2026_surface_refit/inference.py", "src/prc2026_surface_refit/guards.py", "src/prc2026_nmclock/features.py"]


def code_receipt() -> dict:
    files = {p: sha256_file(ROOT / p) for p in FROZEN + ["tools/run_v7_dec.py"]}
    return {"files": files, "sha256": object_hash(files)}


def recipe() -> dict:
    """The frozen v7 recipe: specs, settings, seed and fixed counts, bound to the hashes recorded by the v7 refit declaration."""
    refit_config = read_json(ROOT / "configs/surface_refit_v1.json")
    for field in ["development_config", "timeline_config"]:
        if sha256_file(ROOT / refit_config[field]) != refit_config[field + "_sha256"]:
            raise ValueError(f"{field} changed since the v7 refit declaration.")
    timeline = read_json(ROOT / refit_config["timeline_config"])
    airports = read_json(ROOT / refit_config["development_config"])
    return {"bundle": ROOT / refit_config["feature_bundle"], "iterations": refit_config["component_iterations"],
            "globals": {name: timeline["variants"][name] for name in ["cat_combined", "lgb_combined"]},
            "globals_settings": {key: timeline[key] for key in ["threads", "seed", "lightgbm", "catboost"]},
            "airport_spec": airports["native_spec"], "airport_settings": airports["settings"], "airports": airports["airports"],
            "config_sha256": {"surface_refit": sha256_file(ROOT / "configs/surface_refit_v1.json"), "timeline": refit_config["timeline_config_sha256"], "surface_airports": refit_config["development_config_sha256"]}}


def predict(run: Path, nodes: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    """v7's clock-present blend from the fitted nodes (prc2026_surface_refit.inference.pipeline_prediction without the v6 branch)."""
    if not route_mask(features).all():
        raise ValueError("Only clock-present rows are predicted by the twelve components.")
    values = np.zeros(len(features), dtype=float)
    for name, weight in WEIGHTS.items():
        component = airport_prediction(run, nodes["airports"], features, threads) if name == "surface_airports" else node_prediction(run, nodes["globals"][name], features, threads)
        values += weight * component
    if not np.isfinite(values).all():
        raise ValueError("Every clock-present row needs a finite prediction.")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=sorted(FOLDS), required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v24/v7"))
    parser.add_argument("--december", type=Path, default=Path("data/december/d198dea10625b0a75977"))
    parser.add_argument("--threads", type=int, default=None, help="override the frozen thread count (does not change the fit)")
    parser.add_argument("--resume", action="store_true", help="prediction stage only, from the saved nodes of a complete fit (nodes.json present, no run.json)")
    args = parser.parse_args()
    started = time.time()
    run = ROOT / args.output / args.fold
    if (run / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {run}")
    if args.resume != (run / "nodes.json").exists() or (not args.resume and run.exists() and any(run.iterdir())):
        raise ValueError(f"--resume requires a complete fit without run.json in {run}; a fresh fit requires an empty directory.")
    held_out = FOLDS[args.fold]
    spec = recipe()
    if args.threads:
        spec["globals_settings"]["threads"] = args.threads
        spec["airport_settings"]["threads"] = args.threads
    threads = spec["globals_settings"]["threads"]
    manifest, parent, original, catalog = load_bundle(ROOT, spec["bundle"])
    december = read_json(ROOT / args.december / "manifest.json")
    if december["month"] != DECEMBER or december["timeline_bundle_id"] != manifest["bundle_id"]:
        raise ValueError("December build does not belong to this timeline bundle.")
    with ResourceMonitor() as monitor:
        # The v23 population: the frozen January-November refit rows (December flight groups protected, 1,919,370) plus December's 165,677.
        population = refit_assignment(catalog, original["descriptor"]["protocol"], TRAINING_MONTHS)
        features, truth = load_partition(spec["bundle"], parent, catalog, population, "fit", FEATURE_NAMES)
        if len(features) != 1919370:
            raise ValueError("The January-November refit population changed.")
        month = catalog.set_index(ID)[TIME].reindex(features[ID]).dt.strftime("%Y-%m").to_numpy()
        dec_features = pd.read_parquet(ROOT / args.december / "features" / DECEMBER_FILE, columns=[ID, *FEATURE_NAMES])
        dec_truth = pd.read_parquet(ROOT / args.december / "labels" / DECEMBER_FILE)
        if identity_hash(dec_features[ID]) != december["ids_sha256"] or len(dec_truth) != len(dec_features):
            raise ValueError("December feature/label identity changed.")
        features = pd.concat([features, dec_features], ignore_index=True)
        truth = pd.concat([truth[[ID, TARGET]], dec_truth[[ID, TARGET]]], ignore_index=True)
        month = np.concatenate([month, np.full(len(dec_features), DECEMBER)])
        if features[ID].duplicated().any() or len(features) != 2085047:
            raise ValueError("The twelve-month population changed.")
        fit_rows = ~np.isin(month, held_out)
        fit_features = features.loc[fit_rows].reset_index(drop=True)
        fit_truth = truth.loc[fit_rows].reset_index(drop=True)
        routed_mask = route_mask(fit_features)
        routed = fit_features.loc[routed_mask].reset_index(drop=True)
        routed_truth = fit_truth.loc[routed_mask].reset_index(drop=True)
        fit_months = sorted(set(month[fit_rows]))
        del fit_features, fit_truth
        gc.collect()
        run.mkdir(parents=True, exist_ok=True)
        nodes = read_json(run / "nodes.json") if args.resume else {"globals": {}, "airports": {}}
        if args.resume:
            if set(nodes["globals"]) != {"cat_combined", "lgb_combined"} or list(nodes["airports"]) != spec["airports"]:
                raise ValueError("The saved nodes are incomplete.")
            for name, card in [*nodes["globals"].items(), *nodes["airports"].items()]:
                expected = spec["iterations"]["globals"][name] if name in nodes["globals"] else spec["iterations"]["airports"][name]
                if card["iterations"] != expected or card["fit_ids_sha256"] != identity_hash(routed[ID] if name in nodes["globals"] else routed.loc[routed.airport.eq(name), ID]):
                    raise ValueError(f"Saved node {name} does not match the fold's fitting rows or the frozen count.")
        for name in [] if args.resume else ["cat_combined", "lgb_combined"]:
            nodes["globals"][name] = fit_node(run, name, spec["globals"][name], spec["globals_settings"], routed, routed_truth, iterations=spec["iterations"]["globals"][name])
            write_json(run / "nodes_progress.json", nodes)
            print(f"{args.fold}: {name} {nodes['globals'][name]['iterations']} rounds on {len(routed)} rows, {round(time.time() - started)} s", flush=True)
            gc.collect()
        for airport in [] if args.resume else spec["airports"]:
            mask = routed.airport.eq(airport)
            nodes["airports"][airport] = fit_node(run, "airport_" + airport, spec["airport_spec"], spec["airport_settings"], routed.loc[mask].reset_index(drop=True),
                                                  routed_truth.loc[mask].reset_index(drop=True), iterations=spec["iterations"]["airports"][airport])
            write_json(run / "nodes_progress.json", nodes)
            print(f"{args.fold}: airport {airport} {nodes['airports'][airport]['iterations']} rounds on {int(mask.sum())} rows, {round(time.time() - started)} s", flush=True)
            gc.collect()
        write_json(run / "nodes.json", nodes)
        fit_ids_sha256, clock_fit_rows = identity_hash(routed[ID]), int(len(routed))
        del routed, routed_truth
        gc.collect()
        # Held-out months (every departure of those months, not only the population) or the ranking.
        if held_out:
            extra_months = [m for m in held_out if m != DECEMBER]
            parts = []
            if extra_months:
                assignment = catalog[[ID]].copy()
                assignment["partition"] = np.where(catalog[TIME].dt.strftime("%Y-%m").isin(extra_months), "heldout", "other")
                held_features, _ = load_partition(spec["bundle"], parent, catalog, assignment, "heldout", FEATURE_NAMES)
                held_month = catalog.set_index(ID)[TIME].reindex(held_features[ID]).dt.strftime("%Y-%m").to_numpy()
                parts.append(held_features.assign(held_month=held_month))  # `month` is a frozen predictor; the tag must not shadow it
            if DECEMBER in held_out:
                parts.append(dec_features.assign(held_month=DECEMBER))
            target = pd.concat(parts, ignore_index=True)
            out_name = "predictions.parquet"
        else:
            ranking = pd.read_parquet(spec["bundle"] / "features" / "ranking.parquet", columns=[ID, *FEATURE_NAMES])
            if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
                raise ValueError("Ranking movement identities changed.")
            target = ranking.assign(held_month="ranking")
            out_name = "ranking_predictions.parquet"
        del features, truth, dec_features, dec_truth
        gc.collect()
        clock = route_mask(target)
        prediction = np.full(len(target), np.nan)
        if clock.any():
            prediction[clock] = predict(run, nodes, target.loc[clock].reset_index(drop=True), threads)
        # Reload check: every node re-predicts a sample from its saved files (fit_node already checked 1,000 fitting rows per node).
        sample = target.loc[clock].iloc[:2000].reset_index(drop=True)
        restored = predict(run, read_json(run / "nodes.json"), sample, threads)
        delta = float(np.max(np.abs(prediction[clock][:2000] - restored))) if len(sample) else 0.0
        if delta > 1e-8:
            raise ValueError("The saved nodes do not reproduce.")
        pd.DataFrame({ID: target[ID].to_numpy(), "month": target["held_month"].to_numpy(), "prediction": prediction}).to_parquet(run / out_name, index=False, compression="zstd")
    report = {"created_at_utc": utc_now(), "fold": args.fold, "held_out_months": held_out, "fit_months": fit_months, "stage": "cross_fit" if held_out else "refit_twelve_months",
              "recipe": "full_timeline_v6_100 clock-present components (0.25 cat_combined + 0.25 lgb_combined + 0.5 airport LightGBM), NM-residual targets, frozen 276 predictors",
              "component_iterations": spec["iterations"], "iterations_rule": "the frozen October-selected counts of configs/surface_refit_v1.json, unchanged", "seed": spec["globals_settings"]["seed"],
              "population": "frozen January-November refit rows (1,919,370) + December build (165,677) = 2,085,047; the fold's held-out months removed before fitting",
              "clock_fit_rows": clock_fit_rows, "clock_fit_ids_sha256": fit_ids_sha256, "predicted_rows": int(len(target)), "predicted_clock_rows": int(clock.sum()), "predicted_ids_sha256": identity_hash(target[ID]),
              "prediction_quantiles": {str(q): float(np.nanquantile(prediction, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}, "max_reload_difference_seconds": delta, "reload_sample_rows": int(len(sample)),
              "bundle_id": manifest["bundle_id"], "december_build_id": december["build_id"], "december_ids_sha256": december["ids_sha256"], "config_sha256": spec["config_sha256"],
              "code": code_receipt(), "threads": threads, "audit_used": False, "seconds": round(time.time() - started, 1), "resources": monitor.stats, "environment": environment(ROOT),
              "resumed_prediction_after_correction": bool(args.resume), "correction": "the first runner shadowed the frozen predictor `month` with the held-out tag at prediction time; the components were fitted once by that runner and predicted by this one" if args.resume else None}
    write_json(run / "run.json", report)
    print({k: report[k] for k in ["fold", "held_out_months", "clock_fit_rows", "predicted_rows", "predicted_clock_rows", "prediction_quantiles", "max_reload_difference_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
