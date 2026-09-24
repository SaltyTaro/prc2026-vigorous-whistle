"""Write the adsb_nn3 declaration (configs/adsb_nn3_v1.json) with its parent hashes, before any development fit or score of nnlr.

adsb_nn3 fits 16 seeds of nnlr (adsb_nn2's LIRF-inclusive network on the NM-clock residual target; tools/run_adsb_nn2.py,
configs/adsb_nn3_network_v1.json) like every network before it (seasonal, forward, January-November refit; twelve-month refit for
shipping) and re-reads the LIRF blends with it. Every other table, the committee pool, the rule and the scoring code are adsb_nn2's; the
control is v25 (adsb_nn2's k32_m30_u25). The go decision comes from the October-only design report (configs/adsb_nn3_design_v1.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_adsb_nn2  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
SEEDS = list(range(401, 417))
NETWORKS = {"r": {"ens_nnlr": 1.0}, "x": {"ens_nnl": 0.5, "ens_nnlr": 0.5}}   # nnlr alone; equal mean with v25's nnl
MATCHED_NET = [0.30, 0.45, 0.60]    # (0.15, 0.85 - c, c), from the tree table as in adsb_nn2
UNMATCHED_NET = [0.25, 0.35]        # (0.75 - c, 0.25, c), from v9 as in adsb_nn2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_nn3_v1.json")
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--smoke-note", default="none")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    design = read_json(ROOT / args.design_report)
    if not design["decision"]["go"]:
        raise ValueError("The design decision is not to fit nnlr.")
    nn2 = read_json(ROOT / "configs/adsb_nn2_v1.json")
    selection = read_json(ROOT / "runs/adsb_nn2/dev/selection.json")
    if selection["selected_candidate"] != "k32_m30_u25":
        raise ValueError("The shipped v25 recipe is adsb_nn2's k32_m30_u25.")
    control = nn2["candidates"]["k32_m30_u25"]
    control_rmse = {m: selection["candidates"]["k32_m30_u25"]["months"][m]["rmse"] for m in RECIPE_MONTHS}
    net_config = read_json(ROOT / "configs/adsb_nn3_network_v1.json")
    settings = run_adsb_nn2.resolve_variant(net_config, "nnlr")
    ensembles = dict(nn2["ensembles"])
    ensembles["ens_nnlr"] = {"variant": "nnlr", "variant_settings": settings, "variant_sha256": object_hash(settings), "config": "configs/adsb_nn3_network_v1.json",
                             "config_sha256": sha256_file(ROOT / "configs/adsb_nn3_network_v1.json"), "code_sha256": run_adsb_nn2.code_receipt()["sha256"],
                             "code_files": run_adsb_nn2.code_receipt()["files"], "tool": "tools/run_adsb_nn2.py", "seeds": SEEDS,
                             "learners": [f"runs/adsb_nn3/nnlr_s{s}" for s in SEEDS]}
    if ensembles["ens_nnlr"]["code_sha256"] != nn2["ensembles"]["ens_nnl"]["code_sha256"]:
        raise ValueError("The network code changed since adsb_nn2.")
    tables = dict(nn2["extra_feature_tables"])
    tables["ens_nnlr"] = {"column": "adsb_full", "run_dir": "runs/adsb_nn3/ens_nnlr", "description": "mean of the 16 nnlr seeds",
                          "development": ["runs/adsb_nn3/ens_nnlr/seasonal/predictions.parquet", "runs/adsb_nn3/ens_nnlr/forward/predictions.parquet"],
                          "ranking": "the mean of the declared seeds' refit12/ranking_predictions.parquet, built by the export"}
    candidates = {"control": control}
    for key, lirf_network in NETWORKS.items():
        for cu in UNMATCHED_NET:
            for cm in MATCHED_NET:
                lirf = {"lirf_nonpop_matched": {"v9": control["lirf"]["lirf_nonpop_matched"]["v9"], "tree": round(0.85 - cm, 10), "net": cm},
                        "lirf_nonpop_unmatched": {"v9": round(0.75 - cu, 10), "tree": 0.25, "net": cu}}
                candidates[f"{key}_m{round(100 * cm)}_u{round(100 * cu)}"] = {"network": control["network"], "lirf_network": lirf_network, "lirf": lirf}
    parents = {name: spec["path"] for name, spec in nn2["parents"].items() if name not in ("development_tool", "declaration_tool")}
    leaves = [leaf for e in ("ens_new", "ens_nnl") for leaf in nn2["ensembles"][e]["learners"]]
    parents.update({
        "nn2_declaration": "configs/adsb_nn2_v1.json", "nn2_selection": "runs/adsb_nn2/dev/selection.json",
        **{f"nn2_month_{m}": f"runs/adsb_nn2/dev/month_{m}.json" for m in RECIPE_MONTHS},
        "nn2_development_tool": "tools/run_adsb_nn2_dev.py", "development_tool": "tools/run_adsb_nn3_dev.py", "declaration_tool": "tools/declare_adsb_nn3.py",
        **{f"{e}_{fold}": f"runs/adsb_nn2/{e}/{fold}/predictions.parquet" for e in ("ens_new", "ens_nnl") for fold in ("seasonal", "forward")},
        **{f"december_leaf_{leaf.replace('/', '__')}": f"runs/december/learners/{leaf.replace('/', '__')}.parquet" for leaf in leaves},
        "v30_declaration": "configs/adsb_v30_v1.json", "v30_selection": "runs/adsb_v30/dev/selection.json",
        "nn3_network_config": "configs/adsb_nn3_network_v1.json", "nn3_design_config": "configs/adsb_nn3_design_v1.json",
        "nn3_design_report": args.design_report.as_posix(), "nn3_design_report_tool": "tools/nn3_design_report.py", "nn2_chain": "tools/nn2_chain.sh",
        "v25_export": "runs/submissions/vigorous-whistle_v25.parquet", "v25_receipt": "runs/submissions/vigorous-whistle_v25.receipt.json",
        "v25_export_record": "docs/adsb_nn2/export.json", "v25_verification": "docs/adsb_nn2/verification.json", "v25_official_result": "docs/adsb_nn2/official_result.json",
    })
    config = {key: nn2[key] for key in ["committee_table", "lirf_table", "v24_network_table", "network_share", "recipe_blend", "residual_scope",
                                          "residual_members", "residual_columns", "matches_dirs", "cross_fit", "extra_blends",
                                          "minimum_mean_improvement_fraction", "seed", "floor_seconds_rule", "external_data"]}
    config.update({
        "experiment_id": "prc2026-adsb-nn3-v1", "prediction_postprocess": "adsb-nn3-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("A LIRF network that follows long NM clocks. v25's LIRF network (nnl, label target) predicts 900-2,800 s on July's long-clock "
                    "LIRF rows (labels 6,800-9,200 s) where the tree table and v9 follow the clock, so it cannot take more than 0.30 of the "
                    "matched LIRF blend (adsb_v30). nnlr learns label minus the NM clock proxy and restores it, so a long clock passes through."),
        "incumbent": {"submission": "vigorous-whistle_v25.parquet", "submission_sha256": sha256_file(ROOT / "runs/submissions/vigorous-whistle_v25.parquet"),
                      "official_rmse_seconds": 234.8575, "recipe": "configs/adsb_nn2_v1.json candidate k32_m30_u25", "official_rank": "not polled for this campaign"},
        "development_months": RECIPE_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("Each nnlr seed is fitted like every network before it (seasonal fold -> January/July 2025, forward fold -> November, "
                               "January-November refit -> December leaf cache); seed mean by tools/ensemble_adsb_full.py -> runs/adsb_nn3/ens_nnlr. "
                               "On each of the four months v25 is rebuilt by tools/run_adsb_nn2_dev.py's scoring code, unchanged, with the LIRF "
                               "network and the LIRF (v9, tree, net) weights set per candidate. The control must reproduce k32_m30_u25's recorded RMSE."),
        "extra_feature_tables": tables, "ensembles": ensembles,
        "candidates": candidates, "control_candidate": "control", "control_rmse": control_rmse,
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; and on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero. Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v25. The rule of adsb_v25/v26/v27/nn/v29/nn2/v30, unchanged."),
        "threads": {"network_fits": 8, "development": 1},
        "design": {"report": args.design_report.as_posix(), "decision": design["decision"], "rationale": args.rationale},
        "ship": ("If a candidate is selected: the 16 nnlr seeds' twelve-month refits (queued behind their development folds, scheduling only) "
                 "give the ranking table; v25 is rebuilt from its recorded components and must reproduce the uploaded v25 exactly; then the "
                 "selected LIRF network and weights are applied; exported as the next version through the frozen submission contract, "
                 "verified independently (every saved network reloaded and re-predicted, the NM-clock offset restored), audited, and uploaded "
                 "only with the user's approval. Non-LIRF rows must equal the uploaded v25."),
        "execution": {"declared_before": "any development fit or score of nnlr", "network_hosts": "Azure arcw0-arcw7 (existing workers)",
                      "development_scoring_host": "the workstation", "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "v30": "adsb_v30 (fits nothing): with nnl, the matched LIRF network weight 0.30 is the optimum; 0.45 worsens July by 0.084%.",
            "design": "configs/adsb_nn3_design_v1.json: October-only design fits of nnlr (6 seeds) and nnl (seeds 3-4); no recipe or ranking month read.",
            "not_retested": "nnl alone at other weights (adsb_v30); every lever closed before adsb_v30.",
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    })
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", list(candidates))


if __name__ == "__main__":
    main()
