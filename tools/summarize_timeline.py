"""Summarize the complete declared timeline batch without reading new labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026_timeline.features import FLOWN_NAMES, SURFACE_NAMES
from summarize_ladder import comparisons


def summarize(runs: Path, selection_path: Path) -> dict:
    selection = read_json(selection_path)
    config = selection["configuration"]
    if selection["audit_used"] or selection["candidate_count"] != 10 or len(selection["candidates"]) != 10:
        raise ValueError("All ten declared complete candidate recipes are required.")
    expected = {(name, fold) for name in config["variant_order"] for fold in config["development_folds"]}
    found = {(name, fold) for name, folds in selection["development_runs"].items() for fold in folds}
    if found != expected or len(found) != 8:
        raise ValueError("All eight completed native fits are required.")
    native, resources = {}, []
    for name, folds in selection["development_runs"].items():
        native[name] = {}
        for fold, saved in folds.items():
            base = runs / saved["run_id"]
            if sha256_file(base / "run.json") != saved["run_sha256"]:
                raise ValueError("A producing native run changed.")
            run, node = read_json(base / "run.json"), read_json(base / "node.json")
            verify_receipts(base, run["artifacts"])
            importance = read_json(base / node["importance_path"])
            groups = {"inherited": 0.0, "flown_timeline": 0.0, "surface_context": 0.0}
            for feature, value in importance.items():
                group = "flown_timeline" if feature in FLOWN_NAMES else "surface_context" if feature in SURFACE_NAMES else "inherited"
                groups[group] += value
            total = sum(groups.values())
            if not np.isfinite(total) or total <= 0:
                raise ValueError("Native feature importance is not finite and positive.")
            new_importance = {key: value for key, value in importance.items() if key in FLOWN_NAMES or key in SURFACE_NAMES}
            native[name][fold] = {
                "run_id": saved["run_id"], "run_sha256": saved["run_sha256"],
                "october_selected_trees": saved["iterations"],
                "global_replacement_complete_rmse_seconds": saved["metrics"]["overall"]["rmse_seconds"],
                "native_importance_kind": "gain" if node["model_kind"] == "lightgbm" else "prediction_value_change",
                "native_importance_fraction_by_family": {key: value / total for key, value in groups.items()},
                "leading_new_features": sorted(new_importance.items(), key=lambda item: (-item[1], item[0]))[:12],
                "resources": run["resources"],
            }
            resources.append(run["resources"])
    ordered = sorted(selection["candidates"], key=lambda name: (selection["candidates"][name]["mean_relative_rmse"], name))
    table = []
    for name in ordered:
        row = selection["candidates"][name]
        table.append({
            "name": name, "eligible": row["eligible"], "mean_relative_rmse": row["mean_relative_rmse"],
            "rmse_seconds": {fold: row["folds"][fold]["metrics"]["overall"]["rmse_seconds"] for fold in config["development_folds"]},
            "improvement_fraction": {fold: 1 - row["folds"][fold]["rmse_ratio_to_v6"] for fold in config["development_folds"]},
        })
    selected, leading = selection["selected_candidate"], ordered[0]
    matched = {
        name: comparisons(selection["matched_lgb_reference_metrics"], selection["candidates"][name + "_v6_100"])
        for name in ["lgb_flown", "lgb_surface", "lgb_combined"]
    }
    return {
        "created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
        "comparison_code_sha256": sha256_file(ROOT / "tools/summarize_ladder.py"),
        "selection_sha256": sha256_file(selection_path), "audit_used": False, "new_labels_read": False,
        "completed_native_fits": 8, "candidate_count": 10, "native_models": native,
        "selected_candidate": selected,
        "eligible_candidates": [name for name in ordered if selection["candidates"][name]["eligible"]],
        "candidates_improving_both_folds": [row["name"] for row in table if all(value > 0 for value in row["improvement_fraction"].values())],
        "candidates_by_mean_relative_rmse": table,
        "leading_mean_ratio_candidate": leading,
        "leading_candidate_diagnostics": comparisons(selection["reference_metrics"], selection["candidates"][leading]),
        "selected_diagnostics": comparisons(selection["reference_metrics"], selection["candidates"][selected]) if selected else None,
        "matched_lightgbm_feature_comparisons": matched,
        "selected_paired_days": selection["selected_paired_days"], "refit_iterations": selection["refit_iterations"],
        "completed_fit_resources": {
            "total_wall_seconds": sum(row["wall_seconds"] for row in resources),
            "total_cpu_seconds": sum(row["cpu_seconds"] for row in resources),
            "maximum_sampled_rss_bytes": max(row["peak_process_rss_bytes_sampled"] for row in resources),
        },
        "attempt_accounting": "Eight completed declared fits plus one earlier LightGBM training attempt repeated after the recorded provenance-import error. Completed-fit resource totals exclude that preserved partial attempt.",
        "decision": "Promote the declared eligible recipe to reproducible refit." if selected else "Retain v6; no refit or submission from this batch.",
        "interpretation": "Complete reused development scores, distinct from official scoring and the reserved December audit. Matched LightGBM comparisons retain the original airport models. Native importance is descriptive, not an independent estimate of incremental value. Subgroup findings do not change the declared selection rule.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--selection", type=Path, default=Path("runs/timeline/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/timeline/analysis.json"))
    args = parser.parse_args()
    destination = ROOT / args.output
    if destination.exists():
        raise ValueError("Preserve the original timeline summary.")
    report = summarize(ROOT / args.runs, ROOT / args.selection)
    write_json(destination, report)
    print({key: report[key] for key in ["candidate_count", "eligible_candidates", "candidates_improving_both_folds", "selected_candidate", "leading_mean_ratio_candidate", "completed_fit_resources"]})


if __name__ == "__main__":
    main()
