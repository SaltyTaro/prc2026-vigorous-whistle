"""Export the selected LIRF turnaround mixture as the next versioned submission.

Fits the declared classifier and normal component on the refit months' population
rows, applies the selected candidate to the frozen v7 ranking predictions, floors
at the refit-period minimum label, saves the fitted classifier, and checks that
every untouched row reproduces the uploaded v7 integer. Nothing is uploaded.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET
from prc2026.validation import identity_hash
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_submission.contract import export_submission
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, model_code_receipt, movements, score_population
from run_postprocess import check_parents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lirf_mixture_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/lirf_mixture/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/lirf_mixture/export.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or selection["audit_used"]:
        raise ValueError("Selection does not belong to this declaration.")
    if selection["model_code"]["sha256"] != model_code_receipt(ROOT)["sha256"]:
        raise ValueError("Scientific code changed after selection.")
    if not selection["selected_candidate"]:
        raise ValueError("No candidate was selected; nothing to export.")
    check_parents(config)
    spec = config["candidates"][selection["selected_candidate"]]
    fitting = movements(config["refit_months"])
    model, component, summary = fit_population_model(fitting, config)
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[c for c in MOVEMENT_COLUMNS if c != TARGET] + [TARGET])
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    ids = departures[ID].to_numpy()
    incumbent = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"]
    v7 = incumbent.reindex(ids).to_numpy(dtype=float)
    if len(incumbent) != len(ids) or not np.isfinite(v7).all():
        raise ValueError("Frozen ranking predictions do not cover the ranking departures.")
    population, features, probabilities, interval, mixture, index = score_population(model, component, ranking, ids, config)
    values, counts = apply_candidate(v7, index, interval, mixture, spec["mixture_weight"],
                                     config["mixture"]["schedule_interval_threshold_seconds"], summary["floor_seconds"])
    uploaded = pd.read_parquet(ROOT / "runs/submissions/vigorous-whistle_v7.parquet").set_index(ID)[TARGET].reindex(ids).to_numpy()
    untouched = values == v7
    if not np.array_equal(np.rint(values[untouched]).astype(np.int32), uploaded[untouched]):
        raise ValueError("Untouched rows differ from the uploaded v7 export.")
    run_dir = ROOT / "runs/lirf_mixture" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model["booster"].save_model(str(run_dir / "classifier.txt"), num_iteration=model["rounds"])
    write_json(run_dir / "classifier.json", {"categories": model["categories"], "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"],
                                             "mode_counts": model["mode_counts"], "normal_component": component, "floor_seconds": summary["floor_seconds"]})
    detail = pd.DataFrame({ID: population[ID].to_numpy(), "schedule_interval_s": interval, "p_fill": probabilities[:, 0], "p_normal": probabilities[:, 1],
                           "p_one_day": probabilities[:, 2], "mixture": mixture, "v7": v7[index], "v9": values[index],
                           "schedule_minus_previous_inblock_s": features["schedule_minus_previous_inblock_s"].to_numpy()})
    detail.to_parquet(run_dir / "population_predictions.parquet", index=False)
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": object_hash(config),
                  "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": selection["selected_candidate"], "spec": spec,
                  "classifier_sha256": sha256_file(run_dir / "classifier.txt"), "fitting_summary": {k: v for k, v in summary.items() if k != "normal_component"},
                  "normal_component": component, "row_counts": counts, "parents": config["parents"], "incumbent": config["incumbent"],
                  "model_code_sha256": selection["model_code"]["sha256"], "audit_used": False, "refit_months": config["refit_months"], "native_fit_count": 1}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    changed = ~untouched
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)),
              "ranking_ids_sha256": identity_hash(ids), "row_counts": counts, "untouched_rows_match_v7_export": True,
              "fitting_summary": summary, "ranking_population_rows": int(len(population)),
              "ranking_population_interval_bands": {str(k): int(v) for k, v in pd.cut(pd.Series(interval), config["mixture"]["normal_component_bands_seconds"]).value_counts().sort_index().items()},
              "p_fill_quantiles": {str(q): float(np.quantile(probabilities[:, 0], q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
              "changed_by_airport": {k: int(v) for k, v in departures.loc[changed, "ADEP_mvt"].value_counts().items()},
              "largest_changes": [{"id": float(ids[i]), "v7": float(v7[i]), "v9": float(values[i])} for i in np.argsort(-np.abs(values - v7))[:15]]}
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "row_counts": counts, "rounds": model["rounds"],
           "minimum": receipt["minimum_prediction_seconds"], "maximum": receipt["maximum_prediction_seconds"], "p_fill_median": report["p_fill_quantiles"]["0.5"]})


if __name__ == "__main__":
    main()
