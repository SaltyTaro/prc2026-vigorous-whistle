"""Write the adsb_v25_v1 declaration (configs/adsb_v25_v1.json) with its parent hashes, before any development fit at full protocol.

The declaration inherits the v22 recipe (configs/adsb_v24_v1.json, candidate v7x_r12x_lirf25) and changes only the residual stage. Run with
--output to write a scratch copy (for the disclosed code-path smoke test); without it, the file is written once to configs/ and refuses to
overwrite an existing declaration.
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
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_v25_v1.json")
    parser.add_argument("--smoke-note", default=None, help="disclosure text of the smoke test run before the declaration")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    v24 = read_json(ROOT / "configs/adsb_v24_v1.json")
    v24_dev = {m: read_json(ROOT / "runs/adsb_v24/dev" / f"month_{m}.json") for m in RECIPE_MONTHS}
    v22_receipt = read_json(ROOT / "runs/submissions/vigorous-whistle_v22.receipt.json")
    residual_model = {k: v24["residual_model"][k] for k in ["kind", "target", "categorical", "params", "blend", "prediction"]}
    residual_model["feature_sets"] = {"nb": v24["residual_model"]["feature_sets"]["nb"]}
    residual_model["features"] = v24["residual_model"]["feature_sets"]["nb"]
    residual_model["rounds_cap"] = 4000
    residual_model["patience"] = 200
    residual_model["catboost_params"] = {"loss_function": "RMSE", "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "random_seed": 20260915, "border_count": 254}
    residual_model["rounds_rule"] = ("development: rounds per held-out month and variant from a 5-fold day-grouped inner split of the eleven training months (median best "
                                     "iteration, 200 rounds patience, the variant's cap; sample weights apply to the training folds only, the inner validation is unweighted); "
                                     "the held-out month is never used for rounds; export: the inner rule applied once per refit")
    residual_model["weights_rule"] = ("a variant with a `weights` scheme multiplies each training row's weight by the scheme value at the circular calendar-month distance "
                                      "between the row's month and the target month (the held-out month in development; the ranking row's calendar month, January or July, "
                                      "at export): {'1': w1, '2': w2, 'default': w}; every row of every training month is kept")
    residual_model["applies_to"] = "v22's residual stage (matched rows outside LIRF, target label minus cross-fit v9, weight 1.0, single model); nothing else in the recipe changes"
    parents = {
        "v24_declaration": "configs/adsb_v24_v1.json", "v24_selection": "runs/adsb_v24/dev/selection.json", "v24_development_results": "docs/adsb_v24/development_results.json",
        "v22_export": "runs/submissions/vigorous-whistle_v22.parquet", "v22_receipt": "runs/submissions/vigorous-whistle_v22.receipt.json",
        "v22_export_record": "docs/adsb_v24/export.json", "v22_verification": "docs/adsb_v24/verification.json", "v22_official_result": "docs/adsb_v24/official_result.json",
        "v22_residual_model": "runs/adsb_v24/export_v22/residual_model.txt", "v22_residual_record": "runs/adsb_v24/export_v22/residual_model.json", "v22_ranking_values": "runs/adsb_v24/export_v22/ranking_values.parquet",
        "v7_twelve_month_record": "runs/adsb_v24/v7/refit/run.json", "v7_twelve_month_ranking_predictions": "runs/adsb_v24/v7/refit/ranking_predictions.parquet", "v7_twelve_month_nodes": "runs/adsb_v24/v7/refit/nodes.json",
        "audit_20260922": "docs/audit_20260922/audit.json", "audit_20260922_tool": "tools/audit_december_v22.py",
        "development_tool": "tools/run_adsb_v25.py", "v24_development_tool": "tools/run_adsb_v24.py", "residual_tool_frozen": "tools/run_adsb_v11.py", "adsb_feature_code": "src/prc2026_adsb/features.py",
        "december_scoring": "src/prc2026_december/scoring.py", "postprocess_rules": "src/prc2026_postprocess/rules.py",
        "lirf_mixture_config": "configs/lirf_mixture_v1.json", "lirf_mixture_selection": "runs/lirf_mixture/selection.json", "v9_export": "runs/submissions/vigorous-whistle_v9.parquet",
        "v7_ranking_predictions": v24["parents"]["v7_ranking_predictions"]["path"], "stands_v11": "data/external/adsblol/matches_v11/stands.parquet",
        "ens_v20_bag_q_all_ranking_v23": "runs/adsb_v23/ens/ens_v20_bag_q_all/refit/ranking_predictions.parquet", "ens_v22_lirf_q_ranking_v23": "runs/adsb_v23/ens/ens_v22_lirf_q/refit/ranking_predictions.parquet",
        "ens_v20_bag_q_all_seasonal": "runs/adsb_v20/ens_v20_bag_q_all/seasonal/predictions.parquet", "ens_v20_bag_q_all_forward": "runs/adsb_v20/ens_v20_bag_q_all/forward/predictions.parquet",
        "ens_v22_lirf_q_seasonal": "runs/adsb_v22/ens_v22_lirf_q/seasonal/predictions.parquet", "ens_v22_lirf_q_forward": "runs/adsb_v22/ens_v22_lirf_q/forward/predictions.parquet",
        "scratch_seasonal": "scratch/seasonal.parquet", "scratch_forward": "scratch/forward.parquet", "scratch_december": "scratch/december.parquet",
    }
    for m in RECIPE_MONTHS:
        parents[f"v24_dev_month_{m}"] = f"runs/adsb_v24/dev/month_{m}.json"
        parents[f"v24_dev_predictions_{m}"] = f"runs/adsb_v24/dev/predictions_{m}.parquet"
    for m in ALL_MONTHS:
        parents[f"scratch_v24_{m}"] = f"scratch/v24/{m}.parquet"
        parents[f"matches_v11_{m}"] = f"data/external/adsblol/matches_v11/matches2_{m}.parquet"
    for m in ["2026-01", "2026-07"]:
        parents[f"matches_v11_{m}"] = f"data/external/adsblol/matches_v11/matches2_{m}.parquet"
    for name, path in v24["parents"].items():
        if name.startswith("december_leaf_runs__"):
            parents[name] = path["path"]
    config = {
        "experiment_id": "prc2026-adsb-v25-v1", "prediction_postprocess": "adsb-v25-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("The residual stage of the v22 recipe, developed on a twelve-month leave-one-month-out: sample weights by calendar-month distance to the target month, "
                    "a longer inner rounds cap, a CatBoost residual, and two-member means. Everything else (cross-fit / twelve-month v7, v9, the committee blend, the LIRF "
                    "blends, the floor) is exactly v22. Motivation: docs/audit_20260922 (removing the residual stage costs +0.67 / +0.84 / +2.22 / +2.62% on Jan / Jul / Nov / Dec, "
                    "1.5x the committee's marginal value on the covered months; the residual learning curve showed adjacent months dominate; several inner folds of the shipped "
                    "residual stopped at the 4,000-round cap)."),
        "incumbent": {"submission": "vigorous-whistle_v22.parquet", "submission_sha256": v22_receipt["sha256"], "official_rmse_seconds": 239.1335,
                      "recipe": "configs/adsb_v24_v1.json candidate v7x_r12x_lirf25: cross-fit / twelve-month v7 -> v9, the twelve-month residual r12x on matched non-LIRF rows, 0.7 ens_v20_bag_q_all on non-LIRF rows, 0.85 ens_v22_lirf_q on matched LIRF non-population rows, 0.25 ens_v22_lirf_q on unmatched LIRF non-population rows, floor -12",
                      "official_rank": "not polled for this campaign"},
        "development_months": ALL_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("Leave-one-month-out over all twelve months of 2025 for the residual stage. For a held-out month every variant is fitted on the in-scope rows of the "
                               "other eleven months (scratch/v24 frames: cross-fit v9 on clock-present rows, the provenance that ships; the label-free matches_v11 feature caches of "
                               "adsb_v24) and predicts the held-out month. Reading 1 (all twelve months): the residual stage's own out-of-sample RMSE on its scope (matched non-LIRF "
                               "rows, v9 + r against the label), pooled over the twelve months by rows. Reading 2 (the four recipe months, the only months with out-of-sample committee "
                               "tables): every candidate is built with the frozen blend arithmetic (tools/run_adsb_v11.candidate_values) and scored on ALL rows (every label, no clipping) "
                               "against the incumbent's development prediction, which the tool must reproduce exactly from runs/adsb_v24/dev/predictions_<month>.parquet. The blend "
                               "tables are the incumbent's out-of-sample fold predictions (January / July / November) and the cached January-November refit leaves (December), identical "
                               "for baseline and candidates. The eight other months score the residual stage only."),
        "development_baseline": {"name": "v22 recipe development predictions", "column": "v7x_r12x_lirf25", "predictions": {m: f"runs/adsb_v24/dev/predictions_{m}.parquet" for m in RECIPE_MONTHS},
                                 "recorded_rmse": {m: v24_dev[m]["candidates"]["v7x_r12x_lirf25"]["rmse"] for m in RECIPE_MONTHS},
                                 "recipe": v24["development_baseline"]["recipe"] + " -- then, as v22: cross-fit v9 on clock-present rows, the twelve-month residual r12x, and 0.25 ens_v22_lirf_q on unmatched LIRF non-population rows"},
        "residual_scope": "matched_no_lirf", "matches_dirs": {"v11": "data/external/adsblol/matches_v11"},
        "cross_fit": {"scratch": "scratch/v24", "note": "the adsb_v24 cross-fit frames, unchanged: v9 = max(cross-fit v7, floor) on clock-present rows; missing-clock rows and the LIRF fill population keep the fold / refit v9", "folds": v24["cross_fit"]["folds"]},
        "residual_model": residual_model,
        "control_variant": "r12x",
        "residual_variants": {
            "r12x": {"kind": "lightgbm", "feature_set": "nb", "note": "the shipped residual protocol, unchanged (control; on the four recipe months it repeats the adsb_v24 r12x fit)"},
            "adj2": {"kind": "lightgbm", "feature_set": "nb", "weights": {"1": 2.0, "2": 1.5, "default": 1.0}, "note": "adjacent months weighted 2, two months away 1.5"},
            "adj4": {"kind": "lightgbm", "feature_set": "nb", "weights": {"1": 4.0, "2": 2.0, "default": 1.0}, "note": "adjacent months weighted 4, two months away 2"},
            "cap8k": {"kind": "lightgbm", "feature_set": "nb", "rounds_cap": 8000, "note": "the control with the inner rounds cap raised to 8,000 (several shipped inner folds stopped at 4,000)"},
            "cat": {"kind": "catboost", "feature_set": "nb", "rounds_cap": 8000, "note": "CatBoost residual (depth 8, learning rate 0.05, l2 3, native categorical airport / stand / runway), same target, scope and inner rule"},
        },
        "candidates": {
            "r12x_repro": {"residual": "r12x", "note": "reproduction check of the incumbent's residual protocol; not selectable"},
            "adj2": {"residual": "adj2"}, "adj4": {"residual": "adj4"}, "cap8k": {"residual": "cap8k"}, "cat": {"residual": "cat"},
            "mean_r12x_cat": {"residual": ["r12x", "cat"], "note": "residual = mean of the control and the CatBoost residual"},
            "mean_adj2_cat": {"residual": ["adj2", "cat"], "note": "residual = mean of adj2 and the CatBoost residual"},
        },
        "check_candidates": ["r12x_repro"],
        "blend_with": v24["blend_with"], "extra_blends": [*v24["extra_blends"], *v24["candidates"]["v7x_r12x_lirf25"]["extra_blends"]],
        "extra_feature_tables": v24["extra_feature_tables"],
        "selection": ("For each candidate: RMSE on each of the four recipe months versus the incumbent's development prediction, and the pooled twelve-month residual-scope RMSE "
                      "of its residual versus the control's. Eligible when not worse on ANY recipe month (ratio <= 1.0), mean ratio over the four recipe months at most 0.9998 "
                      "(0.02%), the paired-day bootstrap interval of the RMSE difference below zero on every recipe month where the candidate differs from the baseline on at "
                      "least one row, and the pooled residual-scope RMSE strictly below the control's. Choose the eligible candidate with the lowest mean ratio over the four "
                      "recipe months (ties by name); otherwise select nothing and keep v22. r12x_repro is a check and cannot be selected. Per-airport, ordinary-row and "
                      "per-month scope figures are descriptive."),
        "minimum_mean_improvement_fraction": 0.0002, "require_paired_day_intervals_below_zero": True, "interval_condition_touched_months_only": True, "require_pooled_scope_improvement": True,
        "seed": 20260915, "threads": {"development": 16, "export": 8},
        "floor_seconds_rule": "minimum departure label over the refit months (-12), applied after the residual and blends",
        "ship": ("If a candidate is selected: every member variant of its residual is refit on ALL twelve months' in-scope rows (scratch/v24 frames, cross-fit v9) with rounds from the "
                 "inner rule applied once per refit; an unweighted member gives one model; a weighted member gives two models, one per ranking target month (January: weights by "
                 "circular distance from month 1; July: from month 7), and each ranking row takes the residual of its own calendar month's model; a mean candidate averages its members' "
                 "residuals row by row. v9 on the ranking is exactly v22's (v9 from scratch, the twelve-month v7 of runs/adsb_v24/v7/refit on clock-present rows, the LIRF fill "
                 "population and missing-clock rows unchanged); the blends are exactly v22's with the twelve-month ranking tables; floor -12; nearest-integer export as "
                 "vigorous-whistle_v23.parquet through the frozen submission contract; verified by an independent re-implementation and audited (tools/audit_export.py); uploaded "
                 "only with the user's approval; the official score against v22 (239.1335) is the final test."),
        "external_data": "adsb.lol globe_history (ODbL), unchanged; no new external data",
        "execution": {"declared_before": "any development fit at full protocol or candidate score", "development_hosts": "arcw0-arcw3 (three held-out months each, 16 threads); selection, export and verification on the workstation",
                      "azure_only": True, "automatic_submission": False,
                      "smoke_test": args.smoke_note or "none"},
        "prior_evidence": {
            "audit_20260922": "docs/audit_20260922/README.md: removing the residual stage costs +0.67 / +0.84 / +2.22 / +2.62% (Jan / Jul / Nov / Dec), removing the committee +0.17 / +0.50 / +1.20 / +1.77%; the single residual LightGBM equals the committee on matched rows; additive / hour biases and midnight rows <= 0.1%",
            "learning_curve": "docs/adsb_v24/learning_curve.json: adjacent months dominate the residual's value (December held out: November alone -3.03%, January -2.22%, July -1.90%); rows still help",
            "shipped_inner_rounds": "runs/adsb_v24/dev/month_*.json: r12x inner best iterations include 3,926 / 3,951 / 3,991 and 4,000 (the cap) on several folds",
            "closed": "residual seed bag (-0.002%), residual scale, the season_cos feature, the 1,332-stand tables, a second-stage residual on the committee, v19 surface features (all recorded in earlier declarations); none is re-tested here",
            "no_candidate_scored": "no variant or candidate had been fitted at full protocol or scored on any held-out month before this declaration",
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    }
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]))


if __name__ == "__main__":
    main()
