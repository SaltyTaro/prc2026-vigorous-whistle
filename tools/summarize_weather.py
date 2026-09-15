"""Summarize the declared weather batch without fitting or altering selection."""
from pathlib import Path
import argparse
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_weather.features import WEATHER_NAMES


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/weather"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/weather"))
    args = parser.parse_args()
    runs = root/args.runs
    selection = read_json(runs/"selection.json")
    old = read_json(root/"docs/phase5/development_results.json")
    initial_ground = read_json(root/"docs/phase5r/development_results.json")
    results, fits = {}, []
    for variant, folds in selection["development_runs"].items():
        results[variant] = {}
        for fold, record in folds.items():
            path = runs/record["run_id"]
            node = read_json(path/"node.json")
            importance = read_json(path/node["importance_path"])
            top_weather = sorted(((n, v) for n, v in importance.items() if n in WEATHER_NAMES), key=lambda item: (-item[1], item[0]))[:15]
            if variant == "lgb_ground_weather":
                matched = initial_ground["development_runs"]["demand_service"][fold]["metrics"]
                matched_name = "phase5r_demand_service"
            else:
                matched_name = "catboost_long" if variant == "cat_weather" else "global_capacity"
                matched = old["initial_and_seed_runs"][matched_name]["folds"][fold]["metrics"]
            rmse = record["metrics"]["overall"]["rmse_seconds"]
            previous = matched["overall"]["rmse_seconds"]
            results[variant][fold] = {"rmse_seconds": rmse, "iterations": node["iterations"],
                                      "matched_reference": matched_name, "matched_reference_rmse_seconds": previous,
                                      "rmse_delta_to_matched_reference_seconds": rmse-previous,
                                      "weather_importance_share": sum(v for n, v in importance.items() if n in WEATHER_NAMES)/sum(importance.values()),
                                      "top_weather_importances": [{"feature": n, "importance": v} for n, v in top_weather],
                                      "importance_measure": "split gain" if node["model_kind"] == "lightgbm" else "CatBoost PredictionValuesChange"}
            fits.append(record["resources"])
    best_name = min(selection["candidates"], key=lambda n: (selection["candidates"][n]["mean_relative_rmse"], n))
    best = selection["candidates"][best_name]
    comparisons = {}
    for fold, entry in best["folds"].items():
        ref = selection["reference_metrics"][fold]
        comparisons[fold] = {"rmse_seconds": entry["metrics"]["overall"]["rmse_seconds"],
                             "v4_rmse_seconds": ref["overall"]["rmse_seconds"],
                             "rmse_delta_seconds": entry["metrics"]["overall"]["rmse_seconds"]-ref["overall"]["rmse_seconds"],
                             "paired_days": entry["paired_days"],
                             "airport_rmse_deltas_seconds": {a: m["rmse_seconds"]-ref["by_airport"][a]["rmse_seconds"] for a, m in entry["metrics"]["by_airport"].items()},
                             "target_band_sse_deltas": {a: m["sse"]-ref["by_target_band"][a]["sse"] for a, m in entry["metrics"]["by_target_band"].items()},
                             "nm_missing_sse_deltas": {a: m["sse"]-ref["by_nm_missing"][a]["sse"] for a, m in entry["metrics"]["by_nm_missing"].items()}}
    report = {"created_at_utc": utc_now(), "selection_sha256": sha256_file(runs/"selection.json"),
              "code_sha256": sha256_file(Path(__file__)), "audit_used": False,
              "development_fits": results, "candidate_count": selection["candidate_count"],
              "both_folds_improved_before_margin": [n for n, c in selection["candidates"].items() if all(f["rmse_ratio_to_v4"] < 1 for f in c["folds"].values())],
              "eligible_candidates": [n for n, c in selection["candidates"].items() if c["eligible"]],
              "selected_candidate": selection["selected_candidate"], "lowest_mean_ratio_candidate": best_name,
              "lowest_mean_ratio_comparisons": comparisons,
              "fit_resources": {"count": len(fits), "summed_wall_seconds": sum(r["wall_seconds"] for r in fits),
                                "summed_cpu_seconds": sum(r["cpu_seconds"] for r in fits),
                                "maximum_sampled_rss_bytes": max(r["peak_process_rss_bytes_sampled"] for r in fits)},
              "limitation": "Reused development scores and model importance do not establish independent generalization or causal weather effects."}
    write_json(root/args.report_dir/"analysis.json", report)
    print({"selected_candidate": report["selected_candidate"], "lowest_mean_ratio_candidate": best_name,
           "eligible_candidates": report["eligible_candidates"], "fit_resources": report["fit_resources"]})


if __name__ == "__main__":
    main()
