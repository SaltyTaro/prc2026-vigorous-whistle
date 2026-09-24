"""Write the adsb_v31 declaration (configs/adsb_v31_v1.json) with its parent hashes, before any development score of a gated candidate.

adsb_v31 fits nothing. On the two LIRF blend scopes it gates the LIRF network: on rows where the LIRF tree table exceeds the NM clock
proxy by more than T seconds (the trees signal a schedule fill) the network's weight returns to the tree table. The grid crosses the
gate threshold with the matched/unmatched network weights and the LIRF network(s) named on the command line (fixed after adsb_nn3's
selection, before any gated score). Every table, ensemble, the rule and the scoring code are adsb_nn3's / adsb_nn2's; the control is
v25 (adsb_nn2's k32_m30_u25, no gate).

Disclosure (recorded in the declaration): the fill mechanism was found on July 2025 development rows (a recipe month) while
diagnosing adsb_v30; January, November and December were not inspected for it beyond label-free row counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
THRESHOLDS = [1800, 3600]
MATCHED_NET = [0.30, 0.45, 0.60]
UNMATCHED_NET = [0.25, 0.35]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_v31_v1.json")
    parser.add_argument("--lirf-networks", required=True, help='JSON {"key": {ensemble: weight}}, e.g. {"l": {"ens_nnl": 1.0}}')
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--smoke-note", default="none")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    networks = json.loads(args.lirf_networks)
    nn3 = read_json(ROOT / "configs/adsb_nn3_v1.json")
    nn2_selection = read_json(ROOT / "runs/adsb_nn2/dev/selection.json")
    control = nn3["candidates"]["control"]
    for weights in networks.values():
        if not set(weights) <= set(nn3["ensembles"]) or abs(sum(weights.values()) - 1) > 1e-12:
            raise ValueError("Every LIRF network must be a normalised mean of declared ensembles.")
    candidates = {"control": {**control, "gate": None}}
    for key, lirf_network in networks.items():
        for threshold in THRESHOLDS:
            for cu in UNMATCHED_NET:
                for cm in MATCHED_NET:
                    lirf = {"lirf_nonpop_matched": {"v9": control["lirf"]["lirf_nonpop_matched"]["v9"], "tree": round(0.85 - cm, 10), "net": cm},
                            "lirf_nonpop_unmatched": {"v9": round(0.75 - cu, 10), "tree": 0.25, "net": cu}}
                    candidates[f"{key}_g{threshold}_m{round(100 * cm)}_u{round(100 * cu)}"] = {"network": control["network"], "lirf_network": lirf_network,
                                                                                            "lirf": lirf, "gate": {"threshold_s": threshold}}
    parents = {name: spec["path"] for name, spec in nn3["parents"].items() if name not in ("development_tool", "declaration_tool")}
    parents.update({
        "nn3_declaration": "configs/adsb_nn3_v1.json", "nn3_selection": "runs/adsb_nn3/dev/selection.json",
        **{f"nn3_month_{m}": f"runs/adsb_nn3/dev/month_{m}.json" for m in RECIPE_MONTHS},
        "nn3_development_tool": "tools/run_adsb_nn3_dev.py", "nn2_development_tool": "tools/run_adsb_nn2_dev.py",
        "development_tool": "tools/run_adsb_v31.py", "declaration_tool": "tools/declare_adsb_v31.py",
        **{f"ens_nnlr_{fold}": f"runs/adsb_nn3/ens_nnlr/{fold}/predictions.parquet" for fold in ("seasonal", "forward")},
        **{f"december_leaf_{leaf.replace('/', '__')}": f"runs/december/learners/{leaf.replace('/', '__')}.parquet" for leaf in nn3["ensembles"]["ens_nnlr"]["learners"]},
    })
    config = {key: nn3[key] for key in ["committee_table", "lirf_table", "v24_network_table", "network_share", "recipe_blend", "residual_scope",
                                          "residual_members", "residual_columns", "matches_dirs", "cross_fit", "extra_blends", "extra_feature_tables",
                                          "ensembles", "minimum_mean_improvement_fraction", "seed", "floor_seconds_rule", "external_data", "incumbent"]}
    config.update({
        "experiment_id": "prc2026-adsb-v31-v1", "prediction_postprocess": "adsb-v31-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("A fill gate on the LIRF network. July's long non-population LIRF rows are schedule fills (label = scheduled time to takeoff "
                    "within seconds, NM clock proxy 700-1,300 s); the tree table predicts toward the schedule interval there and every network "
                    "an ordinary taxi, which is why v25's network cannot take more than 0.30 (adsb_v30). On scope rows where the tree table "
                    "exceeds the NM clock proxy by more than T the network's weight returns to the tree table; elsewhere the network may take "
                    "more weight. Fits nothing."),
        "development_months": RECIPE_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("On each of the four months v25 is rebuilt by tools/run_adsb_v31.py (tools/run_adsb_nn2_dev.py's scoring code with the "
                               "gate added to the LIRF blends and to the independent arithmetic check). The control must reproduce k32_m30_u25's "
                               "recorded RMSE."),
        "candidates": candidates, "control_candidate": "control",
        "control_rmse": {m: nn2_selection["candidates"]["k32_m30_u25"]["months"][m]["rmse"] for m in RECIPE_MONTHS},
        "gate": {"signal": "LIRF tree table (ens_v22_lirf_q) minus proxy_s (the NM clock proxy of the scope definitions)", "thresholds_s": THRESHOLDS,
                 "applies_to": "rows of lirf_nonpop_matched and lirf_nonpop_unmatched only; the network's weight moves to the tree table, v9's weight is unchanged",
                 "label_free_row_counts_T1800": {"2025-01": 35, "2025-07": 270, "2025-11": 22, "2025-12": 33, "ranking 2026-01": 28, "ranking 2026-07": 188}},
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; and on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero. Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v25. The rule of every round since adsb_v25, unchanged."),
        "threads": {"development": 1},
        "ship": ("If a candidate is selected: every twelve-month refit exists; v25 is rebuilt from its recorded components and must reproduce the "
                 "uploaded v25 exactly; the selected LIRF network, weights and gate are applied; exported as the next version, verified "
                 "independently (the gate recomputed from the export detail), audited, and uploaded only with the user's approval. Non-LIRF rows "
                 "must equal the uploaded v25."),
        "execution": {"declared_before": "any development score of a gated candidate", "development_scoring_host": "the workstation",
                      "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "discovery": ("DISCLOSED: the fill mechanism was found on July 2025 development rows (a recipe month): the ten July matched-scope rows "
                          "carrying 86% of nnl's deficit against the tree table have labels equal to the schedule interval within seconds and "
                          "ordinary NM clocks. January, November and December were inspected only through label-free counts of gated rows. The "
                          "thresholds (30 and 60 minutes) were fixed without scoring any gated candidate."),
            "v30": "with nnl and no gate, the matched weight 0.30 is the optimum (0.45: July +0.084%, December -0.031%).",
            "nn3": "adsb_nn3's selection (runs/adsb_nn3/dev/selection.json) is a parent; its LIRF networks are named on the command line.",
            "rationale": args.rationale,
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    })
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", len(candidates))


if __name__ == "__main__":
    main()
