"""Write the adsb_v30 declaration (configs/adsb_v30_v1.json) with its parent hashes, before any development score of its candidates.

adsb_v30 fits nothing. It re-reads the LIRF-inclusive network's weights in the LIRF blends above adsb_nn2's grid, because the
matched-LIRF weight 0.30 (the grid's top) carried most of v25's gain. Every table, ensemble, the committee pool (32 seeds), the rule
and the scoring code are adsb_nn2's; the control is adsb_nn2's selected k32_m30_u25 (v25).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
MATCHED_NET = [0.30, 0.45, 0.60, 0.75]    # (0.15, 0.85 - c, c): taken from the tree table, as in adsb_nn2
UNMATCHED_NET = [0.25, 0.35]              # (0.75 - c, 0.25, c): taken from v9, as in adsb_nn2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_v30_v1.json")
    parser.add_argument("--smoke-note", default="none")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    nn2 = read_json(ROOT / "configs/adsb_nn2_v1.json")
    selection = read_json(ROOT / "runs/adsb_nn2/dev/selection.json")
    if selection["selected_candidate"] != "k32_m30_u25":
        raise ValueError("The shipped v25 recipe is adsb_nn2's k32_m30_u25.")
    control = nn2["candidates"]["k32_m30_u25"]
    control_rmse = {m: selection["candidates"]["k32_m30_u25"]["months"][m]["rmse"] for m in RECIPE_MONTHS}
    candidates = {}
    for cu in UNMATCHED_NET:
        for cm in MATCHED_NET:
            name = "control" if (cm, cu) == (0.30, 0.25) else f"m{round(100 * cm)}_u{round(100 * cu)}"
            lirf = {"lirf_nonpop_matched": {"v9": control["lirf"]["lirf_nonpop_matched"]["v9"], "tree": round(0.85 - cm, 10), "net": cm},
                    "lirf_nonpop_unmatched": {"v9": round(0.75 - cu, 10), "tree": 0.25, "net": cu}}
            candidates[name] = {"network": control["network"], "lirf_network": control["lirf_network"], "lirf": lirf}
    if candidates["control"] != control:
        raise ValueError("The rebuilt control differs from adsb_nn2's k32_m30_u25.")
    parents = {name: spec["path"] for name, spec in nn2["parents"].items() if name not in ("development_tool", "declaration_tool")}
    leaves = [leaf for e in ("ens_new", "ens_nnl") for leaf in nn2["ensembles"][e]["learners"]]
    parents.update({
        "nn2_declaration": "configs/adsb_nn2_v1.json", "nn2_selection": "runs/adsb_nn2/dev/selection.json",
        **{f"nn2_month_{m}": f"runs/adsb_nn2/dev/month_{m}.json" for m in RECIPE_MONTHS},
        "nn2_development_tool": "tools/run_adsb_nn2_dev.py", "nn2_declaration_tool": "tools/declare_adsb_nn2.py",
        "development_tool": "tools/run_adsb_v30.py", "declaration_tool": "tools/declare_adsb_v30.py",
        **{f"{e}_{fold}": f"runs/adsb_nn2/{e}/{fold}/predictions.parquet" for e in ("ens_new", "ens_nnl") for fold in ("seasonal", "forward")},
        **{f"{e}_record": f"runs/adsb_nn2/{e}/run.json" for e in ("ens_new", "ens_nnl")},
        **{f"december_leaf_{leaf.replace('/', '__')}": f"runs/december/learners/{leaf.replace('/', '__')}.parquet" for leaf in leaves},
        "v25_export": "runs/submissions/vigorous-whistle_v25.parquet", "v25_receipt": "runs/submissions/vigorous-whistle_v25.receipt.json",
        "v25_export_record": "docs/adsb_nn2/export.json", "v25_verification": "docs/adsb_nn2/verification.json",
        "v25_official_result": "docs/adsb_nn2/official_result.json",
    })
    config = {key: nn2[key] for key in ["committee_table", "lirf_table", "v24_network_table", "network_share", "recipe_blend", "residual_scope",
                                          "residual_members", "residual_columns", "matches_dirs", "cross_fit", "extra_blends", "extra_feature_tables",
                                          "ensembles", "minimum_mean_improvement_fraction", "seed", "floor_seconds_rule", "external_data"]}
    config.update({
        "experiment_id": "prc2026-adsb-v30-v1", "prediction_postprocess": "adsb-v30-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("Re-read the LIRF-inclusive network's weights in the LIRF blends above adsb_nn2's grid. adsb_nn2 selected k32_m30_u25 "
                    "(v25, official 234.8575, -0.191% vs v24); its matched-LIRF network weight 0.30 was the grid's top and carried most of "
                    "the gain (k16_m30_u0 alone -0.095%), and on non-population LIRF rows the 16-seed network alone equals the LIRF tree "
                    "table (December 282.5 vs 283.1 s). Fits nothing."),
        "incumbent": {"submission": "vigorous-whistle_v25.parquet", "submission_sha256": sha256_file(ROOT / "runs/submissions/vigorous-whistle_v25.parquet"),
                      "official_rmse_seconds": 234.8575, "recipe": "configs/adsb_nn2_v1.json candidate k32_m30_u25", "official_rank": "not polled for this campaign"},
        "development_months": RECIPE_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("On each of the four months the v25 recipe is rebuilt by tools/run_adsb_nn2_dev.py's scoring code, unchanged, "
                               "with the LIRF (v9, tree, net) weights set per candidate: matched (0.15, 0.85 - c, c), unmatched "
                               "(0.75 - c, 0.25, c), as adsb_nn2 parameterised them. The control must reproduce k32_m30_u25's recorded RMSE."),
        "candidates": candidates, "control_candidate": "control", "control_rmse": control_rmse,
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; and on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero. Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v25. The rule of adsb_v25/v26/v27/nn/v29/nn2, unchanged."),
        "threads": {"development": 1},
        "ship": ("If a candidate is selected: no refit is needed (every twelve-month refit exists). The v25 ranking values are rebuilt from "
                 "their recorded components and must reproduce the uploaded v25 exactly; then the selected LIRF weights are applied, exported "
                 "as the next version through the frozen submission contract, verified independently (tools/verify_adsb_nn2.py), audited "
                 "(tools/audit_export.py), and uploaded only with the user's approval. Non-LIRF rows must equal the uploaded v25."),
        "execution": {"declared_before": "any development score of a candidate other than the control", "development_scoring_host": "the workstation",
                      "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "already_read": ("adsb_nn2 scored m in {0, 0.15, 0.30} x u in {0, 0.25, 0.50}; k32_m30_u50 (the u50 row, with m30) was -0.042 / "
                             "-0.229 / -0.100 / -0.263% against v24 with two paired-day intervals not below zero. No point of this grid "
                             "other than the control has been scored."),
            "closed": ("residual-stage tuning, committee tree seeds and re-weighting, calibration, test-time adaptation, matcher fixes, the "
                       "LIRF unmatched tree weight (v28), the non-LIRF blend numbers (v29), non-LIRF network seed averaging (nn2: 16 -> 32 "
                       "seeds -0.014%)."),
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    })
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", list(candidates))


if __name__ == "__main__":
    main()
