"""Write the adsb_v26_v1 declaration (configs/adsb_v26_v1.json) with its parent hashes, before any development fit at full protocol.

The declaration inherits the v23 recipe (configs/adsb_v25_v1.json, candidate mean_adj2_cat) and adds CatBoost residual members; the inherited
members' development residuals and shipped models are reused without refitting. Run with --output to write a scratch copy (for the disclosed
code-path smoke test); without it, the file is written once to configs/ and refuses to overwrite an existing declaration.
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
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_v26_v1.json")
    parser.add_argument("--smoke-note", default=None, help="disclosure text of the smoke test run before the declaration")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    v25 = read_json(ROOT / "configs/adsb_v25_v1.json")
    v25_sel = read_json(ROOT / "runs/adsb_v25/dev/selection.json")
    v23_receipt = read_json(ROOT / "runs/submissions/vigorous-whistle_v23.receipt.json")
    v23_fit = read_json(ROOT / "runs/adsb_v25/export_v23/fit_record.json")
    residual_model = dict(v25["residual_model"])
    residual_model["applies_to"] = "v23's residual stage (matched rows outside LIRF, target label minus cross-fit v9, weight 1.0, row-wise mean of member models); nothing else in the recipe changes"
    parents = {
        "v25_declaration": "configs/adsb_v25_v1.json", "v25_selection": "runs/adsb_v25/dev/selection.json", "v25_development_results": "docs/adsb_v25/development_results.json",
        "v23_export": "runs/submissions/vigorous-whistle_v23.parquet", "v23_receipt": "runs/submissions/vigorous-whistle_v23.receipt.json", "v23_export_record": "docs/adsb_v25/export.json",
        "v23_verification": "docs/adsb_v25/verification.json", "v23_official_result": "docs/adsb_v25/official_result.json", "v23_fit_record": "runs/adsb_v25/export_v23/fit_record.json",
        "v23_residual_record": "runs/adsb_v25/export_v23/residual_model.json", "v23_ranking_values": "runs/adsb_v25/export_v23/ranking_values.parquet",
        "v22_export": "runs/submissions/vigorous-whistle_v22.parquet", "v22_ranking_values": "runs/adsb_v24/export_v22/ranking_values.parquet",
        "v7_twelve_month_record": "runs/adsb_v24/v7/refit/run.json", "v7_twelve_month_ranking_predictions": "runs/adsb_v24/v7/refit/ranking_predictions.parquet", "v7_twelve_month_nodes": "runs/adsb_v24/v7/refit/nodes.json",
        "development_tool": "tools/run_adsb_v26.py", "v25_development_tool": "tools/run_adsb_v25.py", "v24_development_tool": "tools/run_adsb_v24.py", "residual_tool_frozen": "tools/run_adsb_v11.py", "adsb_feature_code": "src/prc2026_adsb/features.py",
        "december_scoring": "src/prc2026_december/scoring.py", "postprocess_rules": "src/prc2026_postprocess/rules.py",
        "lirf_mixture_config": "configs/lirf_mixture_v1.json", "lirf_mixture_selection": "runs/lirf_mixture/selection.json", "v9_export": "runs/submissions/vigorous-whistle_v9.parquet",
        "v7_ranking_predictions": v25["parents"]["v7_ranking_predictions"]["path"], "stands_v11": "data/external/adsblol/matches_v11/stands.parquet",
        "ens_v20_bag_q_all_ranking_v23": "runs/adsb_v23/ens/ens_v20_bag_q_all/refit/ranking_predictions.parquet", "ens_v22_lirf_q_ranking_v23": "runs/adsb_v23/ens/ens_v22_lirf_q/refit/ranking_predictions.parquet",
        "ens_v20_bag_q_all_seasonal": "runs/adsb_v20/ens_v20_bag_q_all/seasonal/predictions.parquet", "ens_v20_bag_q_all_forward": "runs/adsb_v20/ens_v20_bag_q_all/forward/predictions.parquet",
        "ens_v22_lirf_q_seasonal": "runs/adsb_v22/ens_v22_lirf_q/seasonal/predictions.parquet", "ens_v22_lirf_q_forward": "runs/adsb_v22/ens_v22_lirf_q/forward/predictions.parquet",
    }
    for member, targets in v23_fit["models"].items():
        for target, spec in targets.items():
            parents[f"v23_model_{member}_{target}"] = f"runs/adsb_v25/export_v23/{spec['file']}"
    for m in ALL_MONTHS:
        parents[f"v25_dev_month_{m}"] = f"runs/adsb_v25/dev/month_{m}.json"
        parents[f"v25_dev_predictions_{m}"] = f"runs/adsb_v25/dev/predictions_{m}.parquet"
        parents[f"scratch_v24_{m}"] = f"scratch/v24/{m}.parquet"
        parents[f"matches_v11_{m}"] = f"data/external/adsblol/matches_v11/matches2_{m}.parquet"
    for m in ["2026-01", "2026-07"]:
        parents[f"matches_v11_{m}"] = f"data/external/adsblol/matches_v11/matches2_{m}.parquet"
    for name, path in v25["parents"].items():
        if name.startswith("december_leaf_runs__"):
            parents[name] = path["path"]
    config = {
        "experiment_id": "prc2026-adsb-v26-v1", "prediction_postprocess": "adsb-v26-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("Further CatBoost members for the v23 residual stage: adsb_v25 found the CatBoost residual (learning rate 0.05) still improving at its 8,000-round cap on "
                    "every inner fold of every month and on the shipped refit (docs/adsb_v25/README.md), so the same model at learning rate 0.10 and a second seed of it "
                    "are fitted on the twelve-month leave-one-month-out and combined with the shipped members (adj2, cat) as row-wise means. The shipped members' "
                    "development residuals (runs/adsb_v25/dev) and shipped models (runs/adsb_v25/export_v23) are reused unchanged. Everything else is exactly v23."),
        "incumbent": {"submission": "vigorous-whistle_v23.parquet", "submission_sha256": v23_receipt["sha256"], "official_rmse_seconds": 238.8197,
                      "recipe": "configs/adsb_v25_v1.json candidate mean_adj2_cat: v22's v9, committee and LIRF blends with the residual = mean(adj2 month-weighted LightGBM per ranking target month, cat CatBoost lr 0.05 8,000 rounds)",
                      "official_rank": "not polled for this campaign"},
        "development_months": ALL_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("Leave-one-month-out over all twelve months of 2025 for the new residual members only: for a held-out month each new variant is fitted on the "
                               "in-scope rows of the other eleven months (scratch/v24 frames, cross-fit v9; the label-free matches_v11 feature caches of adsb_v24) and predicts "
                               "the held-out month; the inherited members' residuals for that month are the adsb_v25 development columns (residual:adj2, residual:cat), which "
                               "were produced by the same protocol. Reading 1 (twelve months): each variant's and each candidate mean's out-of-sample residual-scope RMSE, "
                               "pooled by rows. Reading 2 (the four recipe months): every candidate built with the frozen blend arithmetic and scored on ALL rows against the "
                               "incumbent's development prediction (v25's mean_adj2_cat column, reproduced exactly). Jobs are split by (month, variant) across workers and "
                               "combined per month by the tool's --score stage; no number is read before every partial of a month exists."),
        "development_baseline": {"name": "v23 recipe development predictions", "column": "mean_adj2_cat", "predictions": {m: f"runs/adsb_v25/dev/predictions_{m}.parquet" for m in RECIPE_MONTHS},
                                 "recorded_rmse": {m: v25_sel["candidates"]["mean_adj2_cat"]["months"][m]["rmse"] for m in RECIPE_MONTHS}},
        "inherited_variants": {"names": ["adj2", "cat"], "predictions": {m: f"runs/adsb_v25/dev/predictions_{m}.parquet" for m in ALL_MONTHS},
                               "shipped_models": {member: {target: f"runs/adsb_v25/export_v23/{spec['file']}" for target, spec in targets.items()} for member, targets in v23_fit["models"].items()},
                               "note": "the v23 residual members: development residuals and shipped models reused without refitting"},
        "residual_scope": "matched_no_lirf", "matches_dirs": {"v11": "data/external/adsblol/matches_v11"},
        "cross_fit": v25["cross_fit"], "residual_model": residual_model,
        "new_variants": {
            "cat_lr10": {"kind": "catboost", "feature_set": "nb", "rounds_cap": 8000, "params": {"learning_rate": 0.10, "random_seed": 20260915}, "note": "the v23 CatBoost residual at learning rate 0.10 (same depth 8, l2 3, cap 8,000, inner rule)"},
            "cat_lr10_s2": {"kind": "catboost", "feature_set": "nb", "rounds_cap": 8000, "params": {"learning_rate": 0.10, "random_seed": 20260916}, "note": "the same with seed 20260916"},
        },
        "control_members": ["adj2", "cat"],
        "candidates": {
            "repro": {"residual": ["adj2", "cat"], "note": "reproduction check of the incumbent (inherited columns only); not selectable"},
            "adj2_lr10": {"residual": ["adj2", "cat_lr10"], "note": "the lr-0.10 CatBoost replaces the lr-0.05 one"},
            "adj2_cat_lr10": {"residual": ["adj2", "cat", "cat_lr10"]},
            "adj2_lr10x2": {"residual": ["adj2", "cat_lr10", "cat_lr10_s2"]},
            "adj2_cat_lr10x2": {"residual": ["adj2", "cat", "cat_lr10", "cat_lr10_s2"]},
            "lr10x2": {"residual": ["cat_lr10", "cat_lr10_s2"], "note": "no LightGBM member"},
            "cat_lr10x2": {"residual": ["cat", "cat_lr10", "cat_lr10_s2"], "note": "no LightGBM member"},
        },
        "check_candidates": ["repro"],
        "blend_with": v25["blend_with"], "extra_blends": v25["extra_blends"], "extra_feature_tables": v25["extra_feature_tables"],
        "selection": ("For each candidate: RMSE on each of the four recipe months versus the incumbent's development prediction, and the pooled twelve-month residual-scope RMSE "
                      "of its residual mean versus the control's (the incumbent's mean of adj2 and cat on the same rows). Eligible when not worse on ANY recipe month (ratio <= 1.0), "
                      "mean ratio over the four recipe months at most 0.9998 (0.02%), the paired-day bootstrap interval of the RMSE difference below zero on every recipe month "
                      "where the candidate differs from the baseline on at least one row, and the pooled residual-scope RMSE strictly below the control's. Choose the eligible "
                      "candidate with the lowest mean ratio over the four recipe months (ties by name); otherwise select nothing and keep v23. repro is a check and cannot be "
                      "selected. Per-airport, ordinary-row and per-month scope figures are descriptive."),
        "minimum_mean_improvement_fraction": 0.0002, "require_paired_day_intervals_below_zero": True, "interval_condition_touched_months_only": True, "require_pooled_scope_improvement": True,
        "seed": 20260915, "threads": {"development": 16, "export_fit": 16},
        "floor_seconds_rule": "minimum departure label over the refit months (-12), applied after the residual and blends",
        "ship": ("If a candidate is selected: every NEW member of its residual is refit on ALL twelve months' in-scope rows (scratch/v24 frames, cross-fit v9) with rounds from the "
                 "inner rule applied once (one model each: the new members carry no month weights); the inherited members use exactly the shipped v23 models (hash-bound parents; "
                 "adj2 per ranking target month as in v23); the candidate's residual is the row-wise mean of its members; v9, committee and LIRF blends exactly v23's; floor -12; "
                 "nearest-integer export as vigorous-whistle_v24.parquet through the frozen submission contract; verified by an independent re-implementation and audited "
                 "(tools/audit_export.py); uploaded only with the user's approval; the official score against v23 (238.8197) is the final test."),
        "external_data": "adsb.lol globe_history (ODbL), unchanged; no new external data",
        "execution": {"declared_before": "any development fit at full protocol or candidate score", "development_hosts": "arcw0-arcw7 ((month, variant) jobs, 16 threads); scoring, selection, export assembly and verification on the workstation; the new members' export refits on workers",
                      "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note or "none"},
        "prior_evidence": {
            "v25_round": "docs/adsb_v25/README.md: cat (lr 0.05) -1.38% pooled scope, better on 12/12 months, inner best iterations 7,898-8,000 on every fold of every month and 7,999-8,000 on the shipped refit; mean(adj2, cat) -1.96% scope -> -0.057% recipe -> official -0.131%; adj2 alone marginal, adj4 worse, cap8k identical to the control",
            "closed": "residual seed bag for LightGBM (-0.002%), residual scale, season_cos, 1,332-stand tables, second-stage residual on the committee, v19 surface features; none is re-tested here",
            "no_candidate_scored": "no new variant or candidate had been fitted at full protocol or scored on any held-out month before this declaration",
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    }
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]))


if __name__ == "__main__":
    main()
