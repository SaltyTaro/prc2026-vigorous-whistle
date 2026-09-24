"""Write the adsb_v29 declaration (configs/adsb_v29_v1.json) with its parent hashes, before any development score of its candidates.

adsb_v29 fits nothing. It re-reads the v24 recipe's three blend numbers (the network share inside the committee table and the committee
table's outer weight on matched and on unmatched non-LIRF rows) on the four recipe months, because the committee table changed by
-1.46% (December) when the network entered it and the outer weights were chosen before any network existed. Every table, the
residual columns, the LIRF blends and the rule are inherited unchanged from adsb_nn_v1, whose selected nn50 is the control here.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
SHARES, MATCHED, UNMATCHED = [0.5, 0.6], [0.7, 0.8], [0.7, 0.8, 0.9]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/adsb_v29_v1.json")
    parser.add_argument("--smoke-note", default="none")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Preserve the existing declaration: {args.output}")
    nn = read_json(ROOT / "configs/adsb_nn_v1.json")
    nn_selection = read_json(ROOT / "runs/adsb_nn/dev/selection.json")
    if nn_selection["selected_candidate"] != "nn50":
        raise ValueError("The shipped v24 recipe is adsb_nn's nn50.")
    control_rmse = {m: nn_selection["candidates"]["nn50"]["months"][m]["rmse"] for m in RECIPE_MONTHS}
    candidates = {}
    for a, wm, wu in itertools.product(SHARES, MATCHED, UNMATCHED):
        name = "control" if (a, wm, wu) == (0.5, 0.7, 0.7) else f"a{round(100 * a)}_m{round(100 * wm)}_u{round(100 * wu)}"
        candidates[name] = {"network_share": a, "outer_matched": wm, "outer_unmatched": wu}
    parents = {name: spec["path"] for name, spec in nn["parents"].items() if name != "development_tool"}
    leaves = nn["network"]["learners"]
    parents.update({
        "nn_declaration": "configs/adsb_nn_v1.json", "nn_selection": "runs/adsb_nn/dev/selection.json",
        **{f"nn_month_{m}": f"runs/adsb_nn/dev/month_{m}.json" for m in RECIPE_MONTHS},
        "nn_development_tool": "tools/run_adsb_nn_dev.py", "development_tool": "tools/run_adsb_v29.py",
        "ens_nn_seasonal": "runs/adsb_nn/ens_nn/seasonal/predictions.parquet", "ens_nn_forward": "runs/adsb_nn/ens_nn/forward/predictions.parquet",
        "ens_nn_record": "runs/adsb_nn/ens_nn/run.json",
        **{f"december_leaf_{leaf.replace('/', '__')}": f"runs/december/learners/{leaf.replace('/', '__')}.parquet" for leaf in leaves},
        "v24_export": "runs/submissions/vigorous-whistle_v24.parquet", "v24_receipt": "runs/submissions/vigorous-whistle_v24.receipt.json",
        "v24_export_record": "docs/adsb_nn/export.json", "v24_verification": "docs/adsb_nn/verification.json",
        "v24_official_result": "docs/adsb_nn/official_result.json",
    })
    config = {
        "experiment_id": "prc2026-adsb-v29-v1", "prediction_postprocess": "adsb-v29-v1", "declared_at_utc": utc_now(), "audit_used": False,
        "purpose": ("Re-read the v24 recipe's blend numbers now that the committee table carries the network. On non-LIRF rows v24 is "
                    "0.3 x (v9 + residual) + 0.7 x [(1 - a) x committee + a x network] with a = 0.5; the residual is zero on unmatched rows, so "
                    "0.3 of every unmatched non-LIRF prediction (60% of the non-LIRF error) is still v9. The 0.7 was chosen with a tree-only "
                    "committee table (v18); the network moved that table by -1.46% on December, and adsb_nn's December reading preferred a "
                    "larger network share (nn60 -1.587% vs nn50 -1.462%). Fits nothing."),
        "incumbent": {"submission": "vigorous-whistle_v24.parquet", "submission_sha256": sha256_file(ROOT / "runs/submissions/vigorous-whistle_v24.parquet"),
                      "official_rmse_seconds": 235.3074, "recipe": "configs/adsb_nn_v1.json candidate nn50 (a = 0.5, outer weight 0.7 on all non-LIRF rows)",
                      "official_rank": "not polled for this campaign"},
        "development_months": RECIPE_MONTHS, "recipe_months": RECIPE_MONTHS,
        "development_scheme": ("On each of the four months the v24 recipe is rebuilt exactly as adsb_nn_v1 rebuilt nn50 (tools/run_adsb_nn_dev.py), "
                               "with the outer blend split into its matched and unmatched non-LIRF parts (identical values at equal weights) and "
                               "the three numbers set per candidate. The control (0.5, 0.7, 0.7) must reproduce nn50's recorded development RMSE."),
        "committee_table": nn["committee_table"], "network_table": nn["network_table"], "network": nn["network"],
        "residual_scope": nn["residual_scope"], "residual_members": nn["residual_members"], "residual_columns": nn["residual_columns"],
        "matches_dirs": nn["matches_dirs"], "cross_fit": nn["cross_fit"], "extra_blends": nn["extra_blends"], "extra_feature_tables": nn["extra_feature_tables"],
        "candidates": candidates, "control_candidate": "control", "control_rmse": control_rmse,
        "minimum_mean_improvement_fraction": nn["minimum_mean_improvement_fraction"],
        "selection": ("A candidate is eligible when: no recipe month is worse on ALL rows (ratio to the control <= 1.0); the mean ratio over the four "
                      "recipe months is at least 0.02% below 1; and on every month where it changes a row the paired-day bootstrap 95% interval of the "
                      "RMSE difference against the control lies below zero. Among eligible candidates choose the lowest mean ratio (ties by name); "
                      "otherwise select nothing and keep v24. The rule of adsb_v25/v26/v27/nn, unchanged."),
        "seed": nn["seed"], "threads": {"development": 1},
        "floor_seconds_rule": nn["floor_seconds_rule"],
        "ship": ("If a candidate is selected: no refit is needed (every table is v24's). The shipped v24 ranking values are rebuilt from their "
                 "recorded components (v9, residual, committee, network and LIRF twelve-month tables); the control numbers must reproduce the "
                 "uploaded v24 exactly; then the selected numbers are applied, exported as the next version through the frozen submission "
                 "contract, verified independently, audited (tools/audit_export.py), and uploaded only with the user's approval. LIRF rows "
                 "must equal the uploaded v24."),
        "external_data": "adsb.lol globe_history (ODbL), unchanged; no new external data",
        "execution": {"declared_before": "any development score of a candidate other than the control and nn60 (see prior_evidence)",
                      "development_scoring_host": "the workstation", "azure_only": True, "automatic_submission": False, "smoke_test": args.smoke_note},
        "prior_evidence": {
            "already_read": ("adsb_nn_v1 scored a = 0.6 at outer weights (0.7, 0.7) as nn60: against v23 -0.218 / -0.230 / -0.504 / -1.587% "
                             "(January / July / November / December) versus nn50's -0.238 / -0.271 / -0.542 / -1.462%, i.e. worse than the control "
                             "here on three of four months, so candidate a60_m70_u70 cannot qualify; it is kept in the grid for completeness. No "
                             "other point of this grid has been scored."),
            "closed": ("residual-stage tuning (v25 shipped, v26/v27 null), committee seeds and re-weighting (v18/v20), calibration, test-time "
                       "adaptation, matcher fixes, the LIRF unmatched weight (v28); none is re-tested here."),
        },
        "parents": {name: {"path": path, "sha256": sha256_file(ROOT / path)} for name, path in sorted(parents.items())},
    }
    write_json(args.output, config)
    print("written", args.output, "declared_at_utc", config["declared_at_utc"], "parents", len(config["parents"]), "candidates", len(candidates))


if __name__ == "__main__":
    main()
