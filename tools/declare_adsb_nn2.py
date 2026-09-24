"""Write the adsb_nn2 declaration (configs/adsb_nn2_v1.json) with its parent hashes, before any development fit or score of its networks.

Two design-dependent choices are arguments, fixed from the October-only design report (configs/adsb_nn2_design_v1.json's decision
rule): --committee-variant (nnb: further seeds of v24's network; nmres: the NM-clock-residual network) and --pool-nnl (whether the
LIRF-inclusive network's seeds may also join the committee-table pool). Everything else is inherited from adsb_nn_v1 (whose nn50 is
the control) and adsb_v29 (which kept the blend numbers).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_adsb_nn  # noqa: E402
import run_adsb_nn2  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
NEW_SEEDS = {"nnb": list(range(117, 133)), "nmres": list(range(301, 317))}
NNL_SEEDS = list(range(201, 217))
MATCHED_NET = [0.0, 0.15, 0.3]      # taken from the tree table: (0.15, 0.85 - c, c)
UNMATCHED_NET = [0.0, 0.25, 0.5]    # taken from v9: (0.75 - c, 0.25, c)


def ensemble(run_dir: str, description: str) -> dict:
    return {"column": "adsb_full", "run_dir": run_dir, "description": description,
            "development": [f"{run_dir}/seasonal/predictions.parquet", f"{run_dir}/forward/predictions.parquet"],
            "ranking": "the mean of the declared seeds' refit12/ranking_predictions.parquet (twelve-month refits), built by the export"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_nn2_v1.json")
    parser.add_argument("--committee-variant", choices=sorted(NEW_SEEDS), required=True)
    parser.add_argument("--pool-nnl", choices=["yes", "no"], required=True)
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--smoke-note", default="none")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    nn = read_json(ROOT / "configs/adsb_nn_v1.json")
    nn_selection = read_json(ROOT / "runs/adsb_nn/dev/selection.json")
    control_rmse = {m: nn_selection["candidates"]["nn50"]["months"][m]["rmse"] for m in RECIPE_MONTHS}
    old_config, new_config = read_json(ROOT / "configs/adsb_nn_network_v1.json"), read_json(ROOT / "configs/adsb_nn2_network_v1.json")
    if run_adsb_nn.code_receipt()["sha256"] != nn["network"]["code_sha256"]:
        raise ValueError("The v24 network code changed.")

    def spec(variant: str, seeds: list[int], learners: list[str], config_path: str, config: dict, module, tool: str) -> dict:
        settings = module.resolve_variant(config, variant)
        return {"variant": variant, "variant_settings": settings, "variant_sha256": object_hash(settings), "config": config_path,
                "config_sha256": sha256_file(ROOT / config_path), "code_sha256": module.code_receipt()["sha256"], "code_files": module.code_receipt()["files"],
                "tool": tool, "seeds": seeds, "learners": learners}

    ensembles = {"ens_nn": {**nn["network"], "tool": "tools/run_adsb_nn.py"}}
    ensembles["ens_nn"] = {k: ensembles["ens_nn"][k] for k in ["variant", "variant_settings", "variant_sha256", "config", "config_sha256", "code_sha256", "code_files", "tool", "seeds", "learners"]}
    seeds = NEW_SEEDS[args.committee_variant]
    if args.committee_variant == "nnb":
        ensembles["ens_new"] = spec("nnb", seeds, [f"runs/adsb_nn/nnb_s{s}" for s in seeds], "configs/adsb_nn_network_v1.json", old_config, run_adsb_nn, "tools/run_adsb_nn.py")
    else:
        ensembles["ens_new"] = spec("nmres", seeds, [f"runs/adsb_nn2/nmres_s{s}" for s in seeds], "configs/adsb_nn2_network_v1.json", new_config, run_adsb_nn2, "tools/run_adsb_nn2.py")
    ensembles["ens_nnl"] = spec("nnl", NNL_SEEDS, [f"runs/adsb_nn2/nnl_s{s}" for s in NNL_SEEDS], "configs/adsb_nn2_network_v1.json", new_config, run_adsb_nn2, "tools/run_adsb_nn2.py")

    tables = dict(nn["extra_feature_tables"])
    tables["ens_new"] = ensemble("runs/adsb_nn2/ens_new", f"mean of the {len(seeds)} new committee-pool seeds ({args.committee_variant})")
    tables["ens_nnl"] = ensemble("runs/adsb_nn2/ens_nnl", f"mean of the {len(NNL_SEEDS)} LIRF-inclusive seeds (nnl)")

    pools = {"k16": {"ens_nn": 1.0}, "k32": {"ens_nn": 0.5, "ens_new": 0.5}}
    if args.pool_nnl == "yes":
        pools["k48"] = {"ens_nn": 1 / 3, "ens_new": 1 / 3, "ens_nnl": 1 / 3}
    inherited = {b["scope"]: b["weight"] for b in nn["extra_blends"]}
    if inherited != {"lirf_nonpop_matched": 0.85, "lirf_nonpop_unmatched": 0.25}:
        raise ValueError("Unexpected inherited LIRF blends.")
    candidates = {}
    for pool, weights in pools.items():
        for cm in MATCHED_NET:
            for cu in UNMATCHED_NET:
                name = "control" if (pool == "k16" and cm == 0 and cu == 0) else f"{pool}_m{round(100 * cm)}_u{round(100 * cu)}"
                lirf = {"lirf_nonpop_matched": {"v9": 1 - inherited["lirf_nonpop_matched"], "tree": inherited["lirf_nonpop_matched"], "net": 0.0},
                        "lirf_nonpop_unmatched": {"v9": 1 - inherited["lirf_nonpop_unmatched"], "tree": inherited["lirf_nonpop_unmatched"], "net": 0.0}}
                if cm:
                    lirf["lirf_nonpop_matched"] = {"v9": 1 - inherited["lirf_nonpop_matched"], "tree": round(inherited["lirf_nonpop_matched"] - cm, 10), "net": cm}
                if cu:
                    lirf["lirf_nonpop_unmatched"] = {"v9": round(1 - inherited["lirf_nonpop_unmatched"] - cu, 10), "tree": inherited["lirf_nonpop_unmatched"], "net": cu}
                candidates[name] = {"network": weights, "lirf_network": {"ens_nnl": 1.0}, "lirf": lirf}

    parents = {name: spec_["path"] for name, spec_ in nn["parents"].items() if name != "development_tool"}
    parents.update({
        "nn_declaration": "configs/adsb_nn_v1.json", "nn_selection": "runs/adsb_nn/dev/selection.json",
        **{f"nn_month_{m}": f"runs/adsb_nn/dev/month_{m}.json" for m in RECIPE_MONTHS},
        "nn_development_tool": "tools/run_adsb_nn_dev.py", "development_tool": "tools/run_adsb_nn2_dev.py",
        "ens_nn_seasonal": "runs/adsb_nn/ens_nn/seasonal/predictions.parquet", "ens_nn_forward": "runs/adsb_nn/ens_nn/forward/predictions.parquet",
        "ens_nn_record": "runs/adsb_nn/ens_nn/run.json",
        **{f"december_leaf_{leaf.replace('/', '__')}": f"runs/december/learners/{leaf.replace('/', '__')}.parquet" for leaf in nn["network"]["learners"]},
        "v24_export": "runs/submissions/vigorous-whistle_v24.parquet", "v24_receipt": "runs/submissions/vigorous-whistle_v24.receipt.json",
        "v24_export_record": "docs/adsb_nn/export.json", "v24_verification": "docs/adsb_nn/verification.json", "v24_official_result": "docs/adsb_nn/official_result.json",
        "v29_declaration": "configs/adsb_v29_v1.json", "v29_selection": "runs/adsb_v29/dev/selection.json",
        "nn2_network_tool": "tools/run_adsb_nn2.py", "nn2_chain": "tools/nn2_chain.sh", "nn_chain": "tools/nn_chain.sh", "nn2_tests": "tests/test_nn2.py",
        "nn2_network_config": "configs/adsb_nn2_network_v1.json", "nn2_design_config": "configs/adsb_nn2_design_v1.json",
        "nn2_design_report": args.design_report.as_posix(), "declaration_tool": "tools/declare_adsb_nn2.py",
    })
    config = {
        "experiment_id": "prc2026-adsb-nn2-v1", "prediction_postprocess": "adsb-nn2-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("Further network members for the v24 recipe. (1) The committee-table network pool doubles from 16 to 32 seeds "
                    f"(new seeds: {args.committee_variant}); the October seed curve was still improving at 8. (2) LIRF rows are untouched by v24's "
                    "network (it never saw LIRF); a network fitted with LIRF's ordinary rows (nnl) enters the LIRF table on both LIRF blend scopes. "
                    "LIRF is 29-45% of the development SSE outside January."),
        "incumbent": {"submission": "vigorous-whistle_v24.parquet", "submission_sha256": sha256_file(ROOT / "runs/submissions/vigorous-whistle_v24.parquet"),
                      "official_rmse_seconds": 235.3074, "recipe": "configs/adsb_nn_v1.json candidate nn50; adsb_v29 kept its blend numbers",
                      "official_rank": "not polled for this campaign"},
        "development_months": RECIPE_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("Every declared seed is fitted like v24's networks: seasonal fold (predicts January/July 2025), forward fold (November), "
                               "January-November refit (December, written to the December leaf cache). Seed means by tools/ensemble_adsb_full.py. "
                               "On each of the four months the v24 recipe is rebuilt exactly as adsb_nn_v1 rebuilt nn50, with the committee "
                               "table's network replaced by the candidate's pool and, on the two LIRF blend scopes, explicit (v9, tree table, "
                               "network) weights: matched (0.15, 0.85 - c, c), the network taking weight from the tree table; unmatched "
                               "(0.75 - c, 0.25, c), the network taking weight from v9 (the October design reading's competitor there). "
                               "The control must reproduce nn50's recorded development RMSE."),
        "committee_table": nn["committee_table"], "lirf_table": "ens_v22_lirf_q", "v24_network_table": "ens_nn", "network_share": 0.5,
        "recipe_blend": nn["recipe_blend"], "residual_scope": nn["residual_scope"], "residual_members": nn["residual_members"],
        "residual_columns": nn["residual_columns"], "matches_dirs": nn["matches_dirs"], "cross_fit": nn["cross_fit"], "extra_blends": nn["extra_blends"],
        "extra_feature_tables": tables, "ensembles": ensembles,
        "candidates": candidates, "control_candidate": "control", "control_rmse": control_rmse,
        "minimum_mean_improvement_fraction": nn["minimum_mean_improvement_fraction"],
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; and on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero. Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v24. The rule of adsb_v25/v26/v27/nn/v29, unchanged."),
        "seed": nn["seed"], "threads": {"network_fits": 8, "development": 1}, "floor_seconds_rule": nn["floor_seconds_rule"],
        "design": {"report": args.design_report.as_posix(), "committee_variant": args.committee_variant, "pool_nnl": args.pool_nnl, "rationale": args.rationale},
        "ship": ("If a candidate is selected: every seed of every ensemble the candidate uses is refit on January-December 2025 at the same epoch "
                 "budget (--fold refit12); ranking tables are the seed means; the shipped v24 ranking values are rebuilt from their recorded "
                 "components with the control's tables and must reproduce the uploaded v24 exactly; then the selected pools are applied, "
                 "exported as the next version through the frozen submission contract, verified independently (every saved network reloaded "
                 "and re-predicted on its cached inputs), audited (tools/audit_export.py), and uploaded only with the user's approval. The "
                 "twelve-month refits may start as development chains finish, before the selection, for scheduling only; they are used only "
                 "if the selection needs them."),
        "external_data": "adsb.lol globe_history (ODbL), unchanged; no new external data",
        "execution": {"declared_before": "any development fit or score of the new networks", "network_hosts": "Azure arcw0-arcw7 (existing workers)",
                      "development_scoring_host": "the workstation", "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "design": ("configs/adsb_nn2_design_v1.json: October-only design fits of nmres (6 seeds) and nnl (2 seeds); no recipe month or ranking "
                       "month was predicted or read. The design report is a parent of this declaration."),
            "v29": "adsb_v29 (fits nothing) kept the network share 0.5 and the outer weight 0.7 on matched and unmatched non-LIRF rows.",
            "closed": ("residual-stage tuning (v25 shipped, v26/v27 null), committee tree seeds and re-weighting (v18/v20), calibration, test-time "
                       "adaptation, matcher fixes, the LIRF unmatched weight (v28), blend numbers (v29); none is re-tested here."),
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    }
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", list(candidates))


if __name__ == "__main__":
    main()
