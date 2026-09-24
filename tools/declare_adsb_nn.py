"""Write the adsb_nn declaration (configs/adsb_nn_v1.json) with its parent hashes, before any development fit or score of the network.

The network recipe (one variant of a network configuration file), its seeds and the committee mixing weights are fixed here from
the October-only design readings. The v23 recipe, its tables and every other number are inherited unchanged from adsb_v28_v2
(the last declaration that rebuilt v23 on the four recipe months, whose control scores the control here must reproduce).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from run_adsb_nn import code_receipt, resolve_variant  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_nn_v1.json")
    parser.add_argument("--network-config", type=Path, required=True, help="configuration file holding the chosen variant")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True, help="network share inside the committee table for each candidate")
    parser.add_argument("--design-report", type=Path, action="append", required=True, help="design report(s) the choice rests on")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--smoke-note", default="none")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    v28 = read_json(ROOT / "configs/adsb_v28_v2.json")
    network_config = read_json(ROOT / args.network_config)
    variant = resolve_variant(network_config, args.variant)
    learners = [f"runs/adsb_nn/{args.variant}_s{seed}" for seed in args.seeds]
    control_rmse = {m: read_json(ROOT / f"runs/adsb_v28/dev/month_{m}.json")["candidates"]["w25"]["rmse_all_rows"] for m in RECIPE_MONTHS}
    parents = {name: spec["path"] for name, spec in v28["parents"].items() if name != "development_tool"}
    parents.update({
        "v28_declaration": "configs/adsb_v28_v2.json", "v28_selection": "runs/adsb_v28/dev/selection.json",
        **{f"v28_month_{m}": f"runs/adsb_v28/dev/month_{m}.json" for m in RECIPE_MONTHS},
        "development_tool": "tools/run_adsb_nn_dev.py", "v28_development_tool": "tools/run_adsb_v28.py",
        "network_tool": "tools/run_adsb_nn.py", "learner_data_path": "tools/run_adsb_learner.py", "ensemble_tool": "tools/ensemble_adsb_full.py",
        "network_package_init": "src/prc2026_nn/__init__.py", "network_package_preprocess": "src/prc2026_nn/preprocess.py",
        "network_package_model": "src/prc2026_nn/model.py", "network_package_train": "src/prc2026_nn/train.py",
        "network_tests": "tests/test_nn.py", "network_config": args.network_config.as_posix(), "design_config": "configs/adsb_nn_design_v1.json",
        "design_committee_reference": "runs/adsb_nn/design/committee_october.parquet", "design_committee_reference_record": "runs/adsb_nn/design/committee_october.json",
        "design_reference_tool": "tools/nn_committee_october.py", "design_report_tool": "tools/nn_design_report.py",
    })
    for i, path in enumerate(args.design_report):
        parents[f"design_report_{i + 1}"] = path.as_posix()
    network_table = "ens_nn"
    tables = dict(v28["extra_feature_tables"])
    tables[network_table] = {"column": "adsb_full", "run_dir": "runs/adsb_nn/ens_nn",
                             "description": "mean of the declared network seeds (tools/ensemble_adsb_full.py over the learner directories below)",
                             "development": ["runs/adsb_nn/ens_nn/seasonal/predictions.parquet", "runs/adsb_nn/ens_nn/forward/predictions.parquet"],
                             "ranking": "the mean of the declared seeds' refit12/ranking_predictions.parquet (twelve-month refits), built by the export"}
    candidates = {"control": 0.0, **{f"nn{int(round(100 * w)):02d}": float(w) for w in args.weights}}
    config = {
        "experiment_id": "prc2026-adsb-nn-v1", "prediction_postprocess": "adsb-nn-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("A neural-network member for the incumbent committee. The committee (ens_v20_bag_q_all: 9 LightGBMs at 1/4, 13 combination "
                    "CatBoosts at 3/4) carries 0.7 of every non-LIRF prediction in v23; every member is a gradient-boosted tree ensemble. The "
                    "network is fitted on the same 314 columns and rows and mixed into the committee table at a fixed share."),
        "incumbent": {"submission": "vigorous-whistle_v23.parquet", "submission_sha256": v28["incumbent"]["submission_sha256"], "official_rmse_seconds": 238.8197,
                      "recipe": v28["incumbent"]["recipe"], "official_rank": "not polled for this campaign"},
        "development_months": RECIPE_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("Each declared seed is fitted like a committee learner (tools/run_adsb_nn.py): the seasonal fold predicts January/July 2025, "
                               "the forward fold predicts November 2025, the January-November refit predicts December 2025 (written to the December "
                               "leaf cache). The seeds are averaged (tools/ensemble_adsb_full.py -> runs/adsb_nn/ens_nn). On each of the four months "
                               "the v23 recipe is rebuilt exactly as adsb_v28_v2 rebuilt it, with the committee table replaced by "
                               "(1 - a) * ens_v20_bag_q_all + a * ens_nn; a = 0 is the control and must reproduce v23's recorded development RMSE."),
        "committee_table": "ens_v20_bag_q_all", "network_table": network_table,
        "recipe_blend": {"scope": v28["blend_with"]["scope"], "weight": v28["blend_with"]["weight"]},
        "residual_scope": v28["residual_scope"], "residual_members": v28["residual_members"], "residual_columns": v28["residual_columns"],
        "matches_dirs": v28["matches_dirs"], "cross_fit": v28["cross_fit"], "extra_blends": v28["extra_blends"], "extra_feature_tables": tables,
        "candidates": candidates, "control_candidate": "control", "control_rmse": control_rmse,
        "network": {"config": args.network_config.as_posix(), "config_sha256": sha256_file(ROOT / args.network_config), "variant": args.variant,
                    "variant_settings": variant, "variant_sha256": object_hash(variant), "seeds": sorted(args.seeds), "learners": learners,
                    "code_sha256": code_receipt()["sha256"], "code_files": code_receipt()["files"],
                    "folds": {"seasonal": "fit 2025-02..06, 08, 09; monitor 2025-10; predict 2025-01, 2025-07",
                              "forward": "fit 2025-01..09; monitor 2025-10; predict 2025-11",
                              "refit": "fit 2025-01..11 at the same epoch budget; predict December 2025 and the ranking",
                              "refit12": "fit 2025-01..12 at the same epoch budget; predict the ranking (ship only)"},
                    "rows": "LIRF left out of fitting and monitoring, exactly as the committee learners; every row predicted",
                    "rationale": args.rationale},
        "minimum_mean_improvement_fraction": 0.0002,
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; and on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero. Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v23. This is the rule of adsb_v25/v26/v27 with the control in place of the baseline."),
        "seed": 20260915, "threads": {"network_fits": 8, "development": 1},
        "floor_seconds_rule": "minimum departure label over the refit months (-12), applied after the residual and blends",
        "ship": ("If a candidate is selected: every declared seed is refit on January-December 2025 at the same epoch budget "
                 "(tools/run_adsb_nn.py --fold refit12); the ranking network table is the seed mean; the shipped v23 ranking values are rebuilt "
                 "from their recorded components (v9, residual, committee and LIRF twelve-month tables) with only the committee table replaced "
                 "by the selected mix; a zero share must reproduce the uploaded v23 exactly; exported as the next version through the frozen "
                 "submission contract, verified independently (every saved network reloaded and re-predicted), audited (tools/audit_export.py), "
                 "and uploaded only with the user's approval. LIRF rows must equal the uploaded v23."),
        "external_data": "adsb.lol globe_history (ODbL), unchanged; no new external data",
        "execution": {"declared_before": "any development fit or score of the network", "network_hosts": "Azure arcw0-arcw7 (existing workers)",
                      "development_scoring_host": "the workstation", "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "design": ("configs/adsb_nn_design_v1.json protocol: October-only readings of networks fitted on the seasonal fit partition; no recipe "
                       "month or ranking month was predicted or read. The design reports are parents of this declaration."),
            "closed": ("residual-stage tuning (v25 shipped, v26/v27 null), committee seeds and re-weighting (v18/v20), calibration, test-time "
                       "adaptation, matcher fixes, blend weights (v28); none is re-tested here."),
            "committee_rounds": "every committee CatBoost stopped at 9-15k rounds under its 20k cap (runs/*/seasonal|forward/run.json)",
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    }
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", candidates)


if __name__ == "__main__":
    main()
