"""Write the adsb_v28_v1 declaration (configs/adsb_v28_v1.json) with its parent hashes, before any candidate is scored.

One number changes: the weight of the LIRF learner table on LIRF non-population rows that have no ADS-B match.
No model is fitted; every candidate is rebuilt from artifacts that already exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

ALL_MONTHS = [f"2025-{m:02}" for m in range(1, 13)]
RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_v28_v1.json")
    parser.add_argument("--smoke-note", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    v27 = read_json(ROOT / "configs/adsb_v27_v1.json")
    v23_receipt = read_json(ROOT / "runs/submissions/vigorous-whistle_v23.receipt.json")
    parents = {
        "v27_declaration": "configs/adsb_v27_v1.json", "v27_development_results": "docs/adsb_v27/development_results.json",
        "v26_declaration": "configs/adsb_v26_v1.json", "v25_declaration": "configs/adsb_v25_v1.json", "v25_selection": "runs/adsb_v25/dev/selection.json",
        "v24_declaration": "configs/adsb_v24_v1.json", "v24_selection": "runs/adsb_v24/dev/selection.json",
        "v23_export": "runs/submissions/vigorous-whistle_v23.parquet", "v23_receipt": "runs/submissions/vigorous-whistle_v23.receipt.json",
        "v23_fit_record": "runs/adsb_v25/export_v23/fit_record.json", "v23_export_record": "docs/adsb_v25/export.json",
        "v23_verification": "docs/adsb_v25/verification.json", "v23_official_result": "docs/adsb_v25/official_result.json",
        "v23_ranking_values": "runs/adsb_v25/export_v23/ranking_values.parquet",
        "v7_twelve_month_ranking_predictions": "runs/adsb_v24/v7/refit/ranking_predictions.parquet",
        "development_tool": "tools/run_adsb_v28.py", "v24_development_tool": "tools/run_adsb_v24.py", "residual_tool_frozen": "tools/run_adsb_v11.py",
        "adsb_feature_code": "src/prc2026_adsb/features.py", "december_scoring": "src/prc2026_december/scoring.py", "postprocess_rules": "src/prc2026_postprocess/rules.py",
        "coverage_census_ranking": "docs/adsb_coverage/census_probe.json", "coverage_census_jul_dec": "docs/adsb_coverage/census_jul_dec.json",
        "coverage_shift_estimate": "docs/adsb_coverage/coverage_shift.json",
        "lirf_mixture_config": "configs/lirf_mixture_v1.json", "lirf_mixture_selection": "runs/lirf_mixture/selection.json",
        "v9_export": "runs/submissions/vigorous-whistle_v9.parquet", "stands_v11": "data/external/adsblol/matches_v11/stands.parquet",
        "v7_ranking_predictions": v27["parents"]["v7_ranking_predictions"]["path"],
        "ens_v20_bag_q_all_ranking_v23": "runs/adsb_v23/ens/ens_v20_bag_q_all/refit/ranking_predictions.parquet",
        "ens_v22_lirf_q_ranking_v23": "runs/adsb_v23/ens/ens_v22_lirf_q/refit/ranking_predictions.parquet",
        "ens_v20_bag_q_all_seasonal": "runs/adsb_v20/ens_v20_bag_q_all/seasonal/predictions.parquet",
        "ens_v20_bag_q_all_forward": "runs/adsb_v20/ens_v20_bag_q_all/forward/predictions.parquet",
        "ens_v22_lirf_q_seasonal": "runs/adsb_v22/ens_v22_lirf_q/seasonal/predictions.parquet",
        "ens_v22_lirf_q_forward": "runs/adsb_v22/ens_v22_lirf_q/forward/predictions.parquet",
    }
    for m in ALL_MONTHS:
        parents[f"v25_dev_predictions_{m}"] = f"runs/adsb_v25/dev/predictions_{m}.parquet"
        parents[f"scratch_v24_{m}"] = f"scratch/v24/{m}.parquet"
        parents[f"matches_v11_{m}"] = f"data/external/adsblol/matches_v11/matches2_{m}.parquet"
    for m in ["2026-01", "2026-07"]:
        parents[f"matches_v11_{m}"] = f"data/external/adsblol/matches_v11/matches2_{m}.parquet"
    for name, path in v27["parents"].items():
        if name.startswith("december_leaf_runs__"):
            parents[name] = path["path"]
    config = {
        "experiment_id": "prc2026-adsb-v28-v1", "prediction_postprocess": "adsb-v28-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("The blend weight on LIRF non-population rows WITHOUT an ADS-B match. The coverage census "
                    "(docs/adsb_coverage/README.md) found that LIRF loses its surface coverage entirely in the July 2026 ranking month "
                    "(403k ground points in January 2026 against 2.1k over seven days in July; 0.0% gate, 0.2% dark, 72.1% never seen on "
                    "the ground), so essentially every LIRF non-population row of that month takes this branch, and LIRF is the "
                    "highest-error airport in the submission. The weight in force, 0.25, was chosen in adsb_v24 against 0.50 by a mean "
                    "ratio of 0.994652 to 0.994657 - a margin of five parts in a million on whole-month RMSE, when those months were "
                    "mostly LIRF-matched. The twelve development months contain 75,602 rows of this scope (11,254 in January 2025, when "
                    "LIRF had no ADS-B coverage at all), which is enough to decide the weight on the regime that actually matters. "
                    "Nothing else in the recipe changes and no model is fitted."),
        "incumbent": {"submission": "vigorous-whistle_v23.parquet", "submission_sha256": v23_receipt["sha256"], "official_rmse_seconds": 238.8197,
                      "recipe": "configs/adsb_v25_v1.json candidate mean_adj2_cat, with 0.85 * ens_v22_lirf_q on matched LIRF non-population rows and 0.25 on unmatched ones",
                      "official_rank": "not polled for this campaign"},
        "development_months": ALL_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("No fit. For each of the twelve months the shipped recipe is rebuilt from the cross-fitted v9 frame "
                               "(scratch/v24), the v23 residual columns (runs/adsb_v25/dev/predictions_*.parquet, members adj2 and cat, "
                               "row-wise mean) and the declared committee and LIRF development tables, with the weight on the touched "
                               "scope set to each candidate's value. Reading 1: RMSE on the touched scope, per month and pooled by rows "
                               "over all twelve months. Reading 2: RMSE on ALL rows of each month, which guards the rest of the recipe. "
                               "The control candidate reproduces the incumbent and its predictions are the baseline for every comparison."),
        "touched_scope": "lirf_nonpop_unmatched", "residual_scope": "matched_no_lirf", "residual_members": ["adj2", "cat"],
        "residual_columns": {m: f"runs/adsb_v25/dev/predictions_{m}.parquet" for m in ALL_MONTHS},
        "matches_dirs": {"v11": "data/external/adsblol/matches_v11"}, "cross_fit": v27["cross_fit"],
        "blend_with": v27["blend_with"], "extra_blends": v27["extra_blends"], "extra_feature_tables": v27["extra_feature_tables"],
        "candidates": {"w25": 0.25, "w40": 0.40, "w50": 0.50, "w60": 0.60, "w75": 0.75, "w100": 1.00},
        "control_candidate": "w25",
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio <= 1.0); the pooled twelve-month RMSE on "
                      "the touched scope is at least 0.5% below the control's; and the paired-day bootstrap 95% interval of the scope RMSE "
                      "difference against the control lies below zero in at least 8 of the twelve months. Among eligible candidates choose "
                      "the lowest pooled scope RMSE (ties by name); otherwise select nothing and keep v23. The bar is set on the touched "
                      "scope rather than on whole-month RMSE because the scope is a small part of most development months and a large part "
                      "of the July 2026 ranking month; the whole-row condition on the recipe months is what protects the rest of the recipe."),
        "minimum_scope_improvement_fraction": 0.005, "minimum_months_with_interval_below_zero": 8,
        "seed": 20260915, "threads": {"development": 1},
        "floor_seconds_rule": "minimum departure label over the refit months (-12), applied after the residual and blends",
        "ship": ("If a candidate is selected: the shipped v23 export is rebuilt with the same v9, residual models (the shipped v23 files, "
                 "hash-bound), committee and LIRF tables and only this weight changed, exported as vigorous-whistle_v24.parquet through the "
                 "frozen submission contract, verified by an independent re-implementation, audited (tools/audit_export.py), and uploaded "
                 "only with the user's approval. Rows outside the touched scope must equal the uploaded v23 after rounding."),
        "external_data": "adsb.lol globe_history (ODbL), unchanged; no new external data",
        "execution": {"declared_before": "any candidate score", "development_hosts": "the workstation only; no worker is started and no model is fitted",
                      "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note or "none"},
        "prior_evidence": {
            "v24_round": ("runs/adsb_v24/dev/selection.json: v7x_r12x_lirf25 mean ratio 0.994652 (Jan -0.4451%, Jul -0.3038%, Nov -1.1373%, "
                          "Dec -0.2530%) against v7x_r12x_lirf50 mean ratio 0.994657 (Jan -0.4467%, Jul -0.3095%, Nov -1.1081%, Dec -0.2729%); "
                          "both were eligible and 0.25 was selected on the mean. January 2025, the month whose LIRF rows are all unmatched, "
                          "preferred 0.50. This declaration discloses that those two numbers were read before it was written; no other weight "
                          "has ever been scored on any month."),
            "coverage": ("docs/adsb_coverage/README.md: LIRF is 0.0% gate / 0.2% dark in the July 2026 ranking month; the touched scope holds "
                         "11,254 / 10,640 / 12,482 / 13,952 rows in January-April 2025 and 864-8,350 in the rest, 75,602 in total."),
            "closed": ("residual model tuning (v25 shipped, v26 and v27 null), the committee, calibration, test-time adaptation, the context "
                       "scope over unmatched rows (v12), and the matcher fixes (docs/adsb_coverage/README.md); none is re-tested here."),
            "no_candidate_scored": "no weight other than 0.25 and 0.50 has been scored on any month, and neither on this scope alone",
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    }
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]))


if __name__ == "__main__":
    main()
