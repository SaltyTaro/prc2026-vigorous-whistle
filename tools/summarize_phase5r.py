"""Summarize saved complete-set research outcomes without new model selection."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json

root = Path(__file__).resolve().parents[1]
source = root/"docs/phase5r/development_results.json"
current = read_json(source)
previous = read_json(root/"docs/phase5/development_results.json")
best = min(current["candidates"], key=lambda name: (current["candidates"][name]["mean_relative_rmse"], name))
report = {"created_at_utc": utc_now(), "source_sha256": sha256_file(source),
          "selected_candidate": current["selected_candidate"], "candidate_count": current["candidate_count"],
          "candidates_improving_both_folds_without_margin": sum(
              all(row["folds"][f]["rmse_ratio_to_v4"] < 1 for f in ["seasonal", "forward"])
              for row in current["candidates"].values()),
          "lowest_mean_relative_rmse_candidate": best,
          "lowest_mean_candidate_is_promoted": current["candidates"][best]["eligible"],
          "matched_lightgbm_comparison": {}, "lowest_mean_candidate_comparison": {},
          "audit_used": False, "official_best_unchanged": {"version": 4, "rmse_seconds": 326.6895, "rows": 344841}}
for fold in ["seasonal", "forward"]:
    baseline = previous["initial_and_seed_runs"]["global_capacity"]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
    proposed = current["development_runs"]["combined"][fold]["metrics"]["overall"]["rmse_seconds"]
    report["matched_lightgbm_comparison"][fold] = {"old_rmse_seconds": baseline, "new_rmse_seconds": proposed,
                                                "improvement_percent": 100*(1-proposed/baseline)}
    candidate = current["candidates"][best]["folds"][fold]
    airport_deltas = {a: m["rmse_seconds"]-current["reference_metrics"][fold]["by_airport"][a]["rmse_seconds"]
                      for a, m in candidate["metrics"]["by_airport"].items()}
    report["lowest_mean_candidate_comparison"][fold] = {
        "reference_rmse_seconds": current["reference_metrics"][fold]["overall"]["rmse_seconds"],
        "candidate_rmse_seconds": candidate["metrics"]["overall"]["rmse_seconds"],
        "rmse_change_percent": 100*(candidate["rmse_ratio_to_v4"]-1),
        "airports_improved": sum(delta < 0 for delta in airport_deltas.values()),
        "airport_rmse_change_seconds": airport_deltas, "paired_days": candidate["paired_days"]}
runs = [item for folds in current["development_runs"].values() for item in folds.values()]
report["resources"] = {"model_runs": len(runs), "total_run_wall_seconds": sum(r["resources"]["wall_seconds"] for r in runs),
                       "total_run_cpu_seconds": sum(r["resources"]["cpu_seconds"] for r in runs),
                       "max_run_peak_rss_bytes": max(r["resources"]["peak_process_rss_bytes_sampled"] for r in runs)}
report["code_sha256"] = sha256_file(Path(__file__))
write_json(root/"docs/phase5r/analysis.json", report)
print({k: report[k] for k in ["selected_candidate", "candidates_improving_both_folds_without_margin", "lowest_mean_relative_rmse_candidate", "resources"]})
