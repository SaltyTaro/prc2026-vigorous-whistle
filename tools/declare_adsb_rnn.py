"""Write the adsb_rnn declaration (configs/adsb_rnn_v1.json) with its parent hashes, before any leave-one-month-out fit of the residual network.

adsb_rnn adds a neural-network member to v25's residual stage. The network (tools/run_adsb_rnn.py, configs/adsb_rnn_network_v1.json) is
fitted on the residual stage's twelve-month leave-one-month-out for every declared seed; per month its seed mean is combined with v25's
residual members adj2 and cat. The candidates are fixed here from the October design report alone: residual = (1 - w) x mean(adj2, cat)
+ w x network for the declared weights w (optionally with a declared committee-mix weight on non-LIRF rows). Everything else is v25
(adsb_nn2's k32_m30_u25); the control is v25 itself and must reproduce its recorded development RMSE. Scoring: tools/run_adsb_rnn_dev.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_adsb_rnn  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
ALL_MONTHS = [f"2025-{m:02}" for m in range(1, 13)]
NETWORK_CONFIG = "configs/adsb_rnn_network_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_rnn_v1.json")
    parser.add_argument("--design-report", type=Path, required=True)
    parser.add_argument("--networks", required=True, help='JSON {"name": {"variant": V, "seeds": [..]}}; learners runs/adsb_rnn/<variant>_s<seed>')
    parser.add_argument("--weights", required=True, help="JSON list of network weights w per network, e.g. [0.3, 0.5]")
    parser.add_argument("--recipe-weights", default="[]", help="JSON list of extra committee-mix weights (v25: 0.7), crossed with every w")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--smoke-note", default="none")
    parser.add_argument("--allow-smoke-design", action="store_true", help="code-path test only: accept a design report whose decision is not go")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    design = read_json(ROOT / args.design_report)
    if not design["decision"]["go"] and not args.allow_smoke_design:
        raise ValueError("The design decision is not to fit the residual network.")
    nn2 = read_json(ROOT / "configs/adsb_nn2_v1.json")
    selection = read_json(ROOT / "runs/adsb_nn2/dev/selection.json")
    if selection["selected_candidate"] != "k32_m30_u25":
        raise ValueError("The shipped v25 recipe is adsb_nn2's k32_m30_u25.")
    v25 = nn2["candidates"]["k32_m30_u25"]
    net_config = read_json(ROOT / NETWORK_CONFIG)
    code = run_adsb_rnn.code_receipt()
    networks = {}
    for name, spec in json.loads(args.networks).items():
        settings = run_adsb_rnn.resolve_variant(net_config, spec["variant"])
        networks[name] = {"variant": spec["variant"], "variant_settings": settings, "variant_sha256": object_hash(settings), "config": NETWORK_CONFIG,
                          "config_sha256": sha256_file(ROOT / NETWORK_CONFIG), "code_sha256": code["sha256"], "code_files": code["files"], "tool": "tools/run_adsb_rnn.py",
                          "seeds": list(spec["seeds"]), "learners": [f"runs/adsb_rnn/{spec['variant']}_s{s}" for s in spec["seeds"]]}
    weights, recipe_weights = json.loads(args.weights), json.loads(args.recipe_weights)
    candidates = {"control": {"residual": {"adj2": 0.5, "cat": 0.5}}}
    for name in networks:
        for w in weights:
            residual = {"adj2": round((1 - w) / 2, 12), "cat": round((1 - w) / 2, 12), name: w}
            candidates[f"{name}_w{round(100 * w)}"] = {"residual": residual}
            for o in recipe_weights:
                candidates[f"{name}_w{round(100 * w)}_o{round(100 * o)}"] = {"residual": residual, "recipe_weight": o}
    parents = {name: spec["path"] for name, spec in nn2["parents"].items() if name not in ("development_tool", "declaration_tool")}
    parents.update({
        "nn2_declaration": "configs/adsb_nn2_v1.json", "nn2_selection": "runs/adsb_nn2/dev/selection.json",
        **{f"nn2_month_{m}": f"runs/adsb_nn2/dev/month_{m}.json" for m in RECIPE_MONTHS},
        **{f"residual_columns_{m}": f"runs/adsb_v25/dev/predictions_{m}.parquet" for m in ALL_MONTHS},
        "v25_residual_declaration": "configs/adsb_v25_v1.json", "v25_residual_selection": "runs/adsb_v25/dev/selection.json",
        "network_config": NETWORK_CONFIG, "design_config": "configs/adsb_rnn_design_v1.json", "design_report": args.design_report.as_posix(),
        "design_report_tool": "tools/rnn_design_report.py", "network_tool": "tools/run_adsb_rnn.py", "chain": "tools/rnn_chain.sh",
        "nn2_development_tool": "tools/run_adsb_nn2_dev.py", "development_tool": "tools/run_adsb_rnn_dev.py", "declaration_tool": "tools/declare_adsb_rnn.py",
        "refit_tool": "tools/refit_adsb_rnn.py", "ranking_table": "runs/adsb_rnn/ranking_nb.parquet", "ranking_table_receipt": "runs/adsb_rnn/ranking_nb.json",
        "ranking_table_tool": "tools/rnn_ranking_table.py", "launcher": "tools/rnn_launch_dev.sh",
        "v25_export": "runs/submissions/vigorous-whistle_v25.parquet", "v25_receipt": "runs/submissions/vigorous-whistle_v25.receipt.json",
        "v25_export_record": "docs/adsb_nn2/export.json", "v25_official_result": "docs/adsb_nn2/official_result.json",
    })
    config = {key: nn2[key] for key in ["committee_table", "lirf_table", "v24_network_table", "network_share", "recipe_blend", "residual_scope",
                                          "residual_columns", "matches_dirs", "cross_fit", "extra_blends", "extra_feature_tables", "ensembles",
                                          "minimum_mean_improvement_fraction", "seed", "floor_seconds_rule", "external_data"]}
    config.update({
        "experiment_id": "prc2026-adsb-rnn-v1", "prediction_postprocess": "adsb-rnn-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("A neural-network member of the residual stage, the one component of the v25 recipe that has never had a network. The "
                    "residual stage (label - cross-fit v9 on matched rows outside LIRF) is v25's mean of adj2 (month-weighted LightGBM) and "
                    "cat (CatBoost); its hyper-parameters are exhausted (adsb_v26/v27), so a new model family is the remaining lever there."),
        "incumbent": {"submission": "vigorous-whistle_v25.parquet", "submission_sha256": sha256_file(ROOT / "runs/submissions/vigorous-whistle_v25.parquet"),
                      "official_rmse_seconds": 234.8575, "recipe": "configs/adsb_nn2_v1.json candidate k32_m30_u25", "official_rank": "not polled for this campaign"},
        "v25_recipe": v25, "residual_networks": networks,
        "development_months": ALL_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("Each declared network seed is fitted by tools/run_adsb_rnn.py on the residual stage's twelve-month leave-one-month-out "
                               "(the eleven other 2025 months' in-scope rows, cross-fit v9, no held-out label read during training); per month the "
                               "seed mean is the network column. tools/run_adsb_rnn_dev.py rebuilds v25 on the four recipe months with "
                               "tools/run_adsb_nn2_dev.py's functions (the control must reproduce k32_m30_u25's recorded RMSE to 1e-9 s) and "
                               "replaces only the residual (and, where declared, the committee-mix weight)."),
        "candidates": candidates, "control_candidate": "control",
        "control_rmse": {m: selection["candidates"]["k32_m30_u25"]["months"][m]["rmse"] for m in RECIPE_MONTHS},
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero; and its residual's pooled twelve-month scope RMSE is below the "
                      "control's (the residual rounds' condition). Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v25."),
        "threads": {"network_fits": 8, "development": 1},
        "design": {"config": "configs/adsb_rnn_design_v1.json", "report": args.design_report.as_posix(), "decision": design["decision"], "rationale": args.rationale},
        "ship": ("If a candidate is selected: every declared seed is refitted once on all twelve months' in-scope rows (same variant, epoch budget "
                 "and seed) and predicts the ranking's in-scope rows; the seed mean replaces the network column. v25 is rebuilt from its recorded "
                 "components and must reproduce the uploaded v25 exactly; only the residual (and a declared committee-mix weight) changes. "
                 "Exported as the next version through the frozen submission contract, verified independently (every saved network reloaded "
                 "and re-predicted), audited, and uploaded only with the user's approval. LIRF rows must equal the uploaded v25."),
        "execution": {"declared_before": "any leave-one-month-out fit of the residual network", "network_hosts": "Azure arcw0-arcw7 (existing workers)",
                      "development_scoring_host": "the workstation", "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "damping": "residual-scope gains reach the recipe damped about 20:1 (adsb_v25 -1.96% scope -> -0.057% recipe; adsb_v27 -0.218% -> -0.0135%).",
            "design": "configs/adsb_rnn_design_v1.json: October-only design fits (rnb and rwide, seeds 1-4); no recipe or ranking month read.",
            "smoke_disclosure": ("DISCLOSED: a first smoke test of tools/run_adsb_rnn_dev.py (on a scratch declaration, before the tool was fixed to "
                                 "collapse every candidate setting) also carried a committee-mix weight of 0.65 with v25's own residual and printed "
                                 "its recipe-month ratios (2025-01 -0.056%, 2025-07 -0.046%, 2025-11 -0.073%, 2025-12 +0.038%). adsb_v29 had only "
                                 "raised that weight. No committee-mix candidate is declared here: December (the best predictor of official "
                                 "transfer so far) prefers v25's 0.7, and this round changes the residual alone."),
            "not_declared": "the wide variant (rwide): better on the residual scope but weaker in the October recipe reading (-0.039% vs -0.075% at 4 seeds).",
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    })
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", list(candidates))


if __name__ == "__main__":
    main()
