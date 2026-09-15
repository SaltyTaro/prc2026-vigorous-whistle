"""Summarize every declared ladder candidate from saved complete metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def comparisons(reference, candidate):
    result = {}
    for fold in ["seasonal", "forward"]:
        old, new = reference[fold], candidate["folds"][fold]["metrics"]
        if old["overall"]["rows"] != new["overall"]["rows"]:
            raise ValueError("The complete comparison population changed.")
        result[fold] = {
            "rows": new["overall"]["rows"], "rmse_seconds": new["overall"]["rmse_seconds"],
            "improvement_seconds": old["overall"]["rmse_seconds"]-new["overall"]["rmse_seconds"],
            "improvement_fraction": 1-new["overall"]["rmse_seconds"]/old["overall"]["rmse_seconds"],
        }
        for family in ["by_airport", "by_nm_missing", "by_target_band", "by_month"]:
            result[fold][family] = {}
            for key, item in new[family].items():
                baseline = old[family][key]
                if item["rows"] != baseline["rows"]:
                    raise ValueError("A subgroup comparison changed its population.")
                result[fold][family][key] = {
                    "rows": item["rows"], "rmse_seconds": item["rmse_seconds"],
                    "rmse_delta_seconds": item["rmse_seconds"]-baseline["rmse_seconds"],
                    "sse_delta": item["sse"]-baseline["sse"],
                    "sse_fraction": item["sse"]/new["overall"]["sse"],
                }
        result[fold]["airports_improved"] = sum(
            item["rmse_delta_seconds"] < 0 for item in result[fold]["by_airport"].values())
    return result


def summarize(runs, selection_path):
    selection = read_json(selection_path)
    if selection["audit_used"] or selection["candidate_count"] != 36 or len(selection["candidates"]) != 36:
        raise ValueError("All 36 complete development candidates must be present.")
    resources, variants = [], {}
    for variant, folds in selection["development_runs"].items():
        variants[variant] = {}
        for fold, row in folds.items():
            path = runs/row["run_id"]/"run.json"
            if sha256_file(path) != row["run_sha256"]:
                raise ValueError("A producing run changed.")
            report = read_json(path)
            resources.append(report["resources"])
            reference = selection["reference_metrics"][fold]["overall"]["rmse_seconds"]
            score = row["metrics"]["overall"]["rmse_seconds"]
            variants[variant][fold] = {
                "rmse_seconds": score, "improvement_fraction": 1-score/reference,
                "october_selected_trees": row["iterations"],
            }
    ordered = sorted(selection["candidates"], key=lambda name: (
        selection["candidates"][name]["mean_relative_rmse"], name))
    frontier = []
    for name in ordered:
        score = [selection["candidates"][name]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
                 for fold in ["seasonal", "forward"]]
        dominated = any(
            all(selection["candidates"][other]["folds"][fold]["metrics"]["overall"]["rmse_seconds"] <= value
                for fold, value in zip(["seasonal", "forward"], score))
            and any(selection["candidates"][other]["folds"][fold]["metrics"]["overall"]["rmse_seconds"] < value
                    for fold, value in zip(["seasonal", "forward"], score))
            for other in ordered if other != name)
        if not dominated:
            frontier.append(name)
    table = []
    for name in ordered:
        item = selection["candidates"][name]
        table.append({"name": name, "eligible": item["eligible"],
                      "mean_relative_rmse": item["mean_relative_rmse"],
                      "improvement_fraction": {fold: 1-item["folds"][fold]["rmse_ratio_to_v6"]
                                               for fold in ["seasonal", "forward"]},
                      "rmse_seconds": {fold: item["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
                                       for fold in ["seasonal", "forward"]}})
    chosen, leading = selection["selected_candidate"], ordered[0]
    report = {
        "created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
        "selection_sha256": sha256_file(selection_path), "audit_used": False, "new_labels_read": False,
        "development_native_fit_count": len(resources), "candidate_count": len(table),
        "native_models": variants, "selected_candidate": chosen,
        "eligible_candidates": [name for name in ordered if selection["candidates"][name]["eligible"]],
        "candidates_improving_both_folds": [
            row["name"] for row in table if all(value > 0 for value in row["improvement_fraction"].values())],
        "candidates_by_mean_relative_rmse": table, "nondominated_candidate_names": frontier,
        "leading_mean_ratio_candidate": leading,
        "leading_candidate_diagnostics": comparisons(selection["reference_metrics"], selection["candidates"][leading]),
        "selected_diagnostics": comparisons(selection["reference_metrics"], selection["candidates"][chosen]) if chosen else None,
        "selected_paired_days": selection["selected_paired_days"], "refit_iterations": selection["refit_iterations"],
        "fit_resources": {
            "total_wall_seconds": sum(item["wall_seconds"] for item in resources),
            "total_cpu_seconds": sum(item["cpu_seconds"] for item in resources),
            "maximum_sampled_rss_bytes": max(item["peak_process_rss_bytes_sampled"] for item in resources),
        },
        "decision": "promote the declared selected recipe" if chosen else "retain v6; no refit or submission from this batch",
        "interpretation": "Complete reused development results, distinct from official ranking and the reserved December audit. Descriptive rankings and subgroups do not alter the declared two-fold promotion rule.",
    }
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/ladder"))
    parser.add_argument("--selection", type=Path, default=Path("runs/ladder/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/ladder/analysis.json"))
    args = parser.parse_args()
    path = root/args.output
    if path.exists():
        raise ValueError("Preserve the original ladder summary.")
    report = summarize(root/args.runs, root/args.selection)
    write_json(path, report)
    print({key: report[key] for key in [
        "candidate_count", "eligible_candidates", "candidates_improving_both_folds",
        "selected_candidate", "leading_mean_ratio_candidate", "fit_resources",
    ]})


if __name__ == "__main__":
    main()

