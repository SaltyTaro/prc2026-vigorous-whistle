"""Export the selected v10 candidate as the next versioned submission.

Refits the v9 classifier and the declared bagged density classifier on the refit
months' population rows, reproduces the uploaded v9 integers from scratch, then
applies the selected items to the frozen v7 ranking predictions using the
mirrored cat_long refit and the frozen v5 fallback. Nothing is uploaded.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, PHASE, TARGET
from prc2026.validation import identity_hash
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_postprocess.rules import row_context
from prc2026_submission.contract import export_submission
from prc2026_v10.model import apply_items
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population
from run_postprocess import check_parents
from run_v10 import fit_density_model, load_configs, model_code_receipt, score_density


def mirrored_refit(config: dict, selection_path: Path, runs: Path) -> tuple[Path, dict]:
    candidates = [path.parent for path in runs.glob("*_refit_*/run.json")]
    if len(candidates) != 1:
        raise ValueError("Exactly one mirrored cat_long refit is required.")
    run_dir = candidates[0]
    run = read_json(run_dir / "run.json")
    spec = config["cat_long_refit"]
    if (run["stage"] != "refit" or run["variant"] != spec["variant"] or run["iterations"] != spec["iterations"] or run["audit_used"]
            or run["config_sha256"] != object_hash(config) or run["selection_sha256"] != sha256_file(selection_path)
            or run["bundle_id"] != spec["bundle_id"] or run["clock_fit_rows"] != spec["expected_clock_fit_rows"]
            or run["clock_fit_ids_sha256"] != spec["expected_clock_fit_ids_sha256"] or run["max_model_reload_difference_seconds"] > 1e-8
            or run["refit_months"] != config["refit_months"]):
        raise ValueError("The mirrored refit does not match the declaration and selection.")
    verify_receipts(run_dir, run["artifacts"])
    mirror = read_json(ROOT / "docs/v10/refit_mirror.json")
    if not mirror["passed"] or mirror["run_id"] != run_dir.name:
        raise ValueError("The mirror receipt does not cover this run.")
    return run_dir, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v10_v2.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/v10/selection_v2.json"))
    parser.add_argument("--refit-runs", type=Path, default=Path("runs/azure_arcw0_v10"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/v10/export.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or selection["audit_used"]:
        raise ValueError("Selection does not belong to this declaration.")
    if selection["model_code"]["sha256"] != model_code_receipt(ROOT)["sha256"]:
        raise ValueError("Scientific code changed after selection.")
    if not selection["selected_candidate"]:
        raise ValueError("No candidate was selected; nothing to export.")
    items = selection["selected_items"]
    check_parents(config)
    mixture_config = load_configs(config)
    threshold = mixture_config["mixture"]["schedule_interval_threshold_seconds"]
    refit_dir, refit_run = mirrored_refit(config, ROOT / args.selection, ROOT / args.refit_runs) if "cat_long_25" in items else (None, None)
    fitting = movements(config["refit_months"])
    v9_model, v9_component, v9_summary = fit_population_model(fitting, mixture_config)
    floor = v9_summary["floor_seconds"]
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[c for c in MOVEMENT_COLUMNS if c != TARGET] + [TARGET])
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    ids = departures[ID].to_numpy()
    v7 = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    fallback = pd.read_parquet(ROOT / config["parents"]["fallback_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(v7).all() or not np.isfinite(fallback).all():
        raise ValueError("Frozen ranking predictions do not cover the ranking departures.")
    population, _, v9_probabilities, interval, v9_mixture, index = score_population(v9_model, v9_component, ranking, ids, mixture_config)
    v9, _ = apply_candidate(v7, index, interval, v9_mixture, config["items"]["density_classifier"]["mixture_weight"], threshold, floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v9).astype(np.int32), uploaded_v9):
        raise ValueError("The uploaded v9 does not reproduce from scratch.")
    density_model, density_component, _ = fit_density_model(fitting, config, mixture_config)
    if density_component != v9_component:
        raise ValueError("The normal component must be identical to v9's.")
    d_population, d_features, d_probabilities, density_mixture = score_density(density_model, density_component, ranking, config, mixture_config)
    if not np.array_equal(d_population[ID].to_numpy(), population[ID].to_numpy()):
        raise ValueError("Density population differs from the v9 population.")
    context = row_context(departures)
    clock = context["clock_available"].to_numpy()
    negative = clock & (context["proxy_s"].to_numpy() < 0)
    cat_long = np.full(len(ids), np.nan)
    if refit_dir is not None:
        saved = pd.read_parquet(refit_dir / "ranking_predictions.parquet")
        if len(saved) != int(clock.sum()) or identity_hash(saved[ID]) != refit_run["ranking_clock_ids_sha256"]:
            raise ValueError("Mirrored cat_long predictions do not match the run record.")
        cat_long = saved.set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(cat_long), clock):
            raise ValueError("cat_long predictions must cover exactly the clock-available ranking rows.")
    values, counts = apply_items(v7, items, clock=clock, cat_long=cat_long, negative_proxy=negative, fallback=fallback, population_index=index,
                                 interval=interval, v9_mixture=v9_mixture, density_mixture=density_mixture, config=config, threshold=threshold, floor=floor)
    eligible_rows = np.zeros(len(ids), dtype=bool)
    eligible_rows[index[interval <= threshold]] = True
    untouched = ~clock & ~eligible_rows
    if not np.array_equal(np.rint(values[untouched]).astype(np.int32), uploaded_v9[untouched]):
        raise ValueError("Rows outside the declared items differ from the uploaded v9 export.")
    run_dir = ROOT / "runs/v10" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for member in density_model["members"]:
        path = run_dir / f"classifier_member_{member['seed']}.txt"
        member["booster"].save_model(str(path), num_iteration=member["rounds"])
        members.append({"seed": member["seed"], "file": path.name, "sha256": sha256_file(path), "rounds": member["rounds"],
                        "inner_best_iterations": member["inner_best_iterations"], "categories": member["categories"]})
    v9_model["booster"].save_model(str(run_dir / "v9_classifier.txt"), num_iteration=v9_model["rounds"])
    v9_classifier_sha256 = sha256_file(run_dir / "v9_classifier.txt")
    if v9_classifier_sha256 != read_json(ROOT / "runs/submissions/vigorous-whistle_v9.receipt.json")["provenance"]["classifier_sha256"]:
        raise ValueError("The refit v9 classifier does not reproduce the exported v9 classifier.")
    classifier_record = {"members": members, "seeds": density_model["seeds"], "feature_names": density_model["feature_names"], "mode_counts": density_model["mode_counts"],
                         "fitting_rows": density_model["fitting_rows"], "normal_component": density_component, "floor_seconds": floor,
                         "v9_classifier_sha256": v9_classifier_sha256, "v9_rounds": v9_model["rounds"]}
    write_json(run_dir / "classifier.json", classifier_record)
    p_fill = d_probabilities[:, 0] / (d_probabilities[:, 0] + d_probabilities[:, 1])
    detail = pd.DataFrame({ID: population[ID].to_numpy(), "schedule_interval_s": interval, "p_fill_v9": v9_probabilities[:, 0] / (v9_probabilities[:, 0] + v9_probabilities[:, 1]),
                           "p_fill": p_fill, "p_one_day": d_probabilities[:, 2], "v9_mixture": v9_mixture, "density_mixture": density_mixture,
                           "v7": v7[index], "v9": v9[index], f"v{args.version}": values[index],
                           **{name: d_features[name].to_numpy() for name in config["density_features"]["names"]}})
    detail.to_parquet(run_dir / "population_predictions.parquet", index=False)
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": object_hash(config),
                  "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": selection["selected_candidate"], "items": items,
                  "classifier_members_sha256": {str(m["seed"]): m["sha256"] for m in members}, "classifier_record_sha256": sha256_file(run_dir / "classifier.json"),
                  "cat_long_refit": None if refit_run is None else {"run_id": refit_run["run_id"], "run_sha256": sha256_file(refit_dir / "run.json"), "iterations": refit_run["iterations"],
                                                                   "ranking_predictions_sha256": sha256_file(refit_dir / "ranking_predictions.parquet")},
                  "fitting_summary": {"fitting_population_rows": v9_summary["fitting_population_rows"], "floor_seconds": floor, "v9_rounds": v9_model["rounds"],
                                      "density_rounds": density_model["rounds"], "mode_counts": density_model["mode_counts"]},
                  "row_counts": counts, "parents": config["parents"], "incumbent": config["incumbent"], "model_code_sha256": selection["model_code"]["sha256"],
                  "audit_used": False, "refit_months": config["refit_months"], "native_fit_count": 1 + len(members) + (1 if refit_run else 0)}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    changed = np.rint(values).astype(np.int32) != uploaded_v9
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)),
              "ranking_ids_sha256": identity_hash(ids), "row_counts": counts, "changed_rows_vs_v9": int(changed.sum()),
              "untouched_rows_match_v9_export": True, "v9_reproduced_from_scratch": True,
              "changed_by_airport": {k: int(v) for k, v in departures.loc[changed, "ADEP_mvt"].value_counts().items()},
              "ranking_population_rows": int(len(population)), "p_fill_quantiles": {str(q): float(np.quantile(p_fill, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
              "p_fill_v9_quantiles": {str(q): float(np.quantile(detail["p_fill_v9"], q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
              "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(values - v9), q)) for q in [0.5, 0.9, 0.99, 0.999]},
              "largest_changes": [{"id": float(ids[i]), "v9": float(v9[i]), f"v{args.version}": float(values[i]), "airport": str(departures.loc[i, "ADEP_mvt"])}
                                  for i in np.argsort(-np.abs(values - v9))[:15]],
              "classifier": {k: v for k, v in classifier_record.items() if k != "members"}, "member_rounds": density_model["rounds"]}
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "row_counts": counts, "changed_rows_vs_v9": report["changed_rows_vs_v9"],
           "minimum": receipt["minimum_prediction_seconds"], "maximum": receipt["maximum_prediction_seconds"], "density_rounds": density_model["rounds"]})


if __name__ == "__main__":
    main()
