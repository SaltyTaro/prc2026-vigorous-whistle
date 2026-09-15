"""Summarize complete airport feature comparisons without reading new labels."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json
from summarize_ladder import comparisons


def summarize(runs, selection_path):
    selection = read_json(selection_path)
    if selection["audit_used"] or selection["native_fit_count"] != 20 or selection["candidate_count"] != 4 or set(selection["development_runs"]) != {"seasonal", "forward"}:
        raise ValueError("Both complete sets of ten fits and four recipes are required.")
    node_resources, fold_resources, native = [], [], {}
    for fold, row in selection["development_runs"].items():
        base = runs / row["run_id"]
        if sha256_file(base / "run.json") != row["run_sha256"]:
            raise ValueError("An airport run changed.")
        run = read_json(base / "run.json")
        verify_receipts(base, run["artifacts"])
        fold_resources.append(run["resources"])
        native[fold] = {}
        for airport, node in read_json(base / "nodes.json").items():
            importance = read_json(base / node["importance_path"])
            total = sum(importance.values())
            new = {key: value for key, value in importance.items() if key.startswith("surface_") or key.startswith("flight_arrival_") or key.startswith("flight_duration_")}
            native[fold][airport] = {"iterations": node["iterations"], "new_feature_importance_fraction": sum(new.values()) / total if total else 0., "leading_new_features": sorted(new.items(), key=lambda item: (-item[1], item[0]))[:8]}
            node_resources.append(node["resources"])
    ordered = sorted(selection["candidates"], key=lambda name: (selection["candidates"][name]["mean_relative_rmse"], name))
    table = [{"name": name, "eligible": selection["candidates"][name]["eligible"],
              "rmse_seconds": {fold: row["metrics"]["overall"]["rmse_seconds"] for fold, row in selection["candidates"][name]["folds"].items()},
              "improvement_fraction": {fold: 1 - row["rmse_ratio_to_v6"] for fold, row in selection["candidates"][name]["folds"].items()}}
             for name in ordered]
    winner, leading = selection["selected_candidate"], ordered[0]
    def totals(rows):
        return {"total_wall_seconds": sum(row["wall_seconds"] for row in rows), "total_cpu_seconds": sum(row["cpu_seconds"] for row in rows), "maximum_sampled_rss_bytes": max(row["peak_process_rss_bytes_sampled"] for row in rows)}
    return {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "selection_sha256": sha256_file(selection_path),
            "audit_used": False, "new_labels_read": False, "native_fit_count": 20, "candidate_count": 4,
            "selected_candidate": winner, "eligible_candidates": [name for name in ordered if selection["candidates"][name]["eligible"]],
            "candidates_by_mean_relative_rmse": table, "native_models": native,
            "leading_mean_ratio_candidate": leading, "leading_candidate_diagnostics": comparisons(selection["reference_metrics"], selection["candidates"][leading]),
            "selected_diagnostics": comparisons(selection["reference_metrics"], selection["candidates"][winner]) if winner else None,
            "refit_iterations": selection["refit_iterations"], "selected_paired_days": selection["selected_paired_days"],
            "native_fit_resources": totals(node_resources), "fold_resources": totals(fold_resources),
            "decision": "Proceed with the declared eligible refit." if winner else "Retain v6; this bounded search found no qualifying replacement.",
            "interpretation": "Matched original airport settings with the frozen combined timeline features. Complete reused development scores; no official-score claim or December audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--output", type=Path, default=Path("docs/surface_airports/analysis.json"))
    args = parser.parse_args()
    destination = ROOT / args.output
    if destination.exists():
        raise ValueError("Preserve the original airport analysis.")
    report = summarize(ROOT / args.runs, ROOT / args.runs / "selection.json")
    write_json(destination, report)
    print({key: report[key] for key in ["native_fit_count", "candidate_count", "selected_candidate", "eligible_candidates", "leading_mean_ratio_candidate", "native_fit_resources"]})


if __name__ == "__main__":
    main()
