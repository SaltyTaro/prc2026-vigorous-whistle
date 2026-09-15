"""Summarize all six arrival-context candidates and matched earlier learners."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json
from summarize_ladder import comparisons


def summarize(runs, selection_path):
    selected = read_json(selection_path)
    if selected["audit_used"] or selected["candidate_count"] != 6 or len(selected["candidates"]) != 6:
        raise ValueError("All six complete declared candidates are required.")
    if sum(len(folds) for folds in selected["development_runs"].values()) != 4:
        raise ValueError("All four declared native fits are required.")
    resources, native = [], {}
    for name, folds in selected["development_runs"].items():
        native[name] = {}
        for fold, row in folds.items():
            base = runs / row["run_id"]
            if sha256_file(base / "run.json") != row["run_sha256"]:
                raise ValueError("A producing run changed.")
            run, node = read_json(base / "run.json"), read_json(base / "node.json")
            verify_receipts(base, run["artifacts"])
            importance = read_json(base / node["importance_path"])
            total = sum(importance.values())
            native[name][fold] = {"run_id": row["run_id"], "run_sha256": row["run_sha256"],
                                  "october_selected_trees": row["iterations"],
                                  "new_feature_importance_fraction": sum(value for key, value in importance.items() if key.startswith("arrival_")) / total,
                                  "leading_new_features": sorted([(key, value) for key, value in importance.items() if key.startswith("arrival_")], key=lambda item: (-item[1], item[0]))[:8]}
            resources.append(run["resources"])
    ordered = sorted(selected["candidates"], key=lambda name: (selected["candidates"][name]["mean_relative_rmse"], name))
    table = [{"name": name, "eligible": selected["candidates"][name]["eligible"],
              "rmse_seconds": {fold: row["metrics"]["overall"]["rmse_seconds"] for fold, row in selected["candidates"][name]["folds"].items()},
              "improvement_fraction": {fold: 1 - row["rmse_ratio_to_v6"] for fold, row in selected["candidates"][name]["folds"].items()}}
             for name in ordered]
    winner, leading = selected["selected_candidate"], ordered[0]
    return {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
            "comparison_code_sha256": sha256_file(ROOT / "tools/summarize_ladder.py"), "selection_sha256": sha256_file(selection_path),
            "audit_used": False, "new_labels_read": False, "completed_native_fits": 4, "candidate_count": 6,
            "selected_candidate": winner, "eligible_candidates": [name for name in ordered if selected["candidates"][name]["eligible"]],
            "candidates_by_mean_relative_rmse": table, "native_models": native,
            "leading_mean_ratio_candidate": leading,
            "leading_candidate_diagnostics": comparisons(selected["reference_metrics"], selected["candidates"][leading]),
            "selected_diagnostics": comparisons(selected["reference_metrics"], selected["candidates"][winner]) if winner else None,
            "matched_learner_comparisons": {name: comparisons(selected["matched_reference_metrics"][name], selected["candidates"][name + "_v6_100"]) for name in selected["development_runs"]},
            "refit_iterations": selected["refit_iterations"], "selected_paired_days": selected["selected_paired_days"],
            "fit_resources": {"total_wall_seconds": sum(row["wall_seconds"] for row in resources),
                              "total_cpu_seconds": sum(row["cpu_seconds"] for row in resources),
                              "maximum_sampled_rss_bytes": max(row["peak_process_rss_bytes_sampled"] for row in resources)},
            "decision": "Promote the fixed eligible recipe to a reproducible refit." if winner else "Retain v6; no refit or submission from this batch.",
            "interpretation": "Complete reused development sets. This batch changes only missing-clock predictions; matched learners isolate the new arrival features. Native importance is descriptive. These are not official scores or the reserved December audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/arrivalctx"))
    parser.add_argument("--selection", type=Path, default=Path("runs/arrivalctx/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/arrivalctx/analysis.json"))
    args = parser.parse_args()
    destination = ROOT / args.output
    if destination.exists():
        raise ValueError("Preserve the original arrival-context summary.")
    report = summarize(ROOT / args.runs, ROOT / args.selection)
    write_json(destination, report)
    print({key: report[key] for key in ["candidate_count", "eligible_candidates", "selected_candidate", "leading_mean_ratio_candidate", "fit_resources"]})


if __name__ == "__main__":
    main()
