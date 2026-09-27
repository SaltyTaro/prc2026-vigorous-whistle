"""Export a candidate of the disagreement-gate round (configs/disagreement_gate_v1.json) on top of v34.

Construction (the declared ship clause), on the pattern of tools/export_fair_weights.py:
- v33's floats are rebuilt exactly as export_fair_weights does (v31 floats + the v33 outer-weight delta) and must round to the uploaded v33;
  v34's floats = v33's + (recipe at v34's weights - recipe at v33's weights) on the dry-run detail, and must round to the uploaded v34;
- on the dry-run detail, D = committee mix - (v9 + residual stage); the gate's rows get delta = gated value - v34's recipe value
  (unm_up: max(mix, -12); mat_down: max(base, -12); lirf_up: max(column, -12) with the LIRF matched net:tree proportions); zero elsewhere;
- output = max(v34 floats + delta, -12) through export_submission as vigorous-whistle_v<N>;
- independent rebuild from the recorded v30 detail's own components (runs/adsb_ctx/export_v30/ranking_values.parquet), reported.

  --candidate NAME --dry-run OUT                  scratch export (never under runs/submissions)
  --candidate NAME --version N --user-override T  the user's recorded decision to ship a candidate the unchanged rule did not select
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import export_v33  # noqa: E402
from export_fair_weights import FLOOR, LIRF_NET, LIRF_TREE, V33_FILE, V33_LIRF, V33_OUTER, V33_SHA, recipe, scopes, stats  # noqa: E402

DECLARATION = "configs/disagreement_gate_v1.json"
RESULTS = "docs/disagreement/gate_v1_results.json"
V34_FILE, V34_SHA = "runs/submissions/vigorous-whistle_v34.parquet", "62b9f2a50cb9ac24600eebbfd6125af46473485c282e2211b7d7dc15fb6dcbff"
V34_LIRF = {"lirf_nonpop_matched": dict(V33_LIRF["lirf_nonpop_matched"]), "lirf_nonpop_unmatched": {"v9": 0.0, "net": 0.35 / 0.60, "tree": 0.25 / 0.60}}


def gate_delta(detail: pd.DataFrame, spec: dict, v34_recipe: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    s = scopes(detail)
    base = detail["v9"].to_numpy(dtype=float) + np.where(detail["scope"].eq("matched_no_lirf").to_numpy(), detail["residual"].to_numpy(dtype=float), 0.0)
    mix = detail["committee_mix"].to_numpy(dtype=float)
    v9 = detail["v9"].to_numpy(dtype=float)
    m = V34_LIRF["lirf_nonpop_matched"]
    column = (m["net"] * detail[LIRF_NET].to_numpy(dtype=float) + m["tree"] * detail[LIRF_TREE].to_numpy(dtype=float)) / (m["net"] + m["tree"])
    d = mix - base
    gated = v34_recipe.copy()
    masks = {"unm_up": s["unmatched_other"] & (d > spec.get("unm_up", np.inf)),
             "mat_down": s["matched_other"] & (d < -spec.get("mat_down", np.inf)),
             "lirf_up": s["lirf_nonpop_matched"] & (column - v9 > spec.get("lirf_up", np.inf))}
    gated[masks["unm_up"]] = np.maximum(mix[masks["unm_up"]], FLOOR)
    gated[masks["mat_down"]] = np.maximum(base[masks["mat_down"]], FLOOR)
    gated[masks["lirf_up"]] = np.maximum(column[masks["lirf_up"]], FLOOR)
    return gated - v34_recipe, masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", type=int, default=35)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    if args.candidate not in decl["candidates"]:
        raise ValueError(f"{args.candidate} is not a declared candidate.")
    spec = decl["candidates"][args.candidate]
    results = read_json(ROOT / RESULTS)
    if results.get("declaration_sha256") != object_hash(decl):
        raise ValueError("The recorded results are not for this declaration.")
    if args.dry_run is None:
        if results.get("selected_candidate") != args.candidate and not args.user_override:
            raise ValueError(f"The round did not select {args.candidate}; shipping it needs --user-override.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/disagreement/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    for path, sha in [(export_v33.V31_FILE, export_v33.V31_SHA), (V33_FILE, V33_SHA), (V34_FILE, V34_SHA)]:
        if sha256_file(ROOT / path) != sha:
            raise ValueError(f"The uploaded {path} changed.")

    rec = pd.read_parquet(ROOT / export_v33.V30_DETAIL)
    ids = rec[ID].to_numpy()
    dry = pd.read_parquet(ROOT / export_v33.DRY).set_index(ID).reindex(ids).reset_index()
    repaired = pd.read_parquet(ROOT / export_v33.REPAIRED, columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    if dry["v30"].isna().any() or np.isnan(repaired).any() or not (dry["scope"].to_numpy() == rec["scope"].to_numpy()).all():
        raise ValueError("The dry-run / repaired details do not align with the v30 detail.")
    lfpg = 0.5 * (repaired - dry["v30"].to_numpy(dtype=float))
    uploaded = {v: pd.read_parquet(ROOT / f).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
                for v, f in [(31, export_v33.V31_FILE), (33, V33_FILE), (34, V34_FILE)]}
    v31 = np.maximum(rec["v30"].to_numpy(dtype=float) + lfpg, FLOOR)
    v33 = np.maximum(v31 + export_v33.recipe(dry, 1.0, 0.6, 0.7) - export_v33.recipe(dry, 0.7, 0.7, 0.7), FLOOR)
    at33 = recipe(dry, V33_OUTER, V33_LIRF)
    at34 = recipe(dry, V33_OUTER, V34_LIRF)
    v34 = np.maximum(v33 + at34 - at33, FLOOR)
    for v, floats in [(31, v31), (33, v33), (34, v34)]:
        if not np.array_equal(np.rint(floats).astype(np.int64), uploaded[v]):
            raise ValueError(f"Rebuilt v{v} floats do not round to the uploaded v{v}.")

    delta, masks = gate_delta(dry, spec, at34)
    touched = masks["unm_up"] | masks["mat_down"] | masks["lirf_up"]
    if np.any(delta[~touched] != 0) or not np.isfinite(delta).all():
        raise ValueError("The gate delta touches rows outside its masks.")
    values = np.maximum(v34 + delta, FLOOR)
    provenance = {"experiment_id": decl["experiment_id"], "candidate": args.candidate, "dry_run": args.dry_run is not None,
                  "selected_candidate": f"{args.candidate} (user override)" if args.user_override else args.candidate,
                  "selected_by": f"user override of the unchanged rule: {args.user_override}" if args.user_override else "dry run",
                  "recipe": "v34 with the disagreement gate: rows where the committee mix exceeds v9 + residual by more than the threshold (unmatched non-LIRF non-LTFM) take the mix; delta from the dry-run detail added to v34's floats, floor -12",
                  "incumbent": {"submission": "vigorous-whistle_v34.parquet", "official_rmse_seconds": 228.5933, "sha256": V34_SHA},
                  "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "declaration_object_hash": object_hash(decl),
                  "development_results": RESULTS, "development_results_sha256": sha256_file(ROOT / RESULTS),
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL]},
                  "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - uploaded[34].astype(float)

    # independent rebuild from the recorded detail's own components (v34's recipe there, plus the LFPG effect and the gate)
    rec34 = recipe(rec, V33_OUTER, V34_LIRF)
    rdelta, rmasks = gate_delta(rec, spec, rec34)
    independent = np.maximum(rec34 + rdelta + lfpg, FLOOR)
    differing = np.rint(independent).astype(np.int64) != exported
    airport = rec["airport"].to_numpy()
    day = pd.read_parquet(ROOT / "runs/disagreement_rank_days.parquet", columns=[ID, "day", "clock"]).set_index(ID).reindex(ids)
    report = {"created_at_utc": utc_now(), "candidate": args.candidate, "version": args.version, "dry_run": args.dry_run is not None,
              "output": output.as_posix(), "receipt_sha256": receipt.get("sha256"),
              "checks": {"v31_v33_v34_floats_round_to_uploaded": True, "delta_outside_masks": 0},
              "gate_rows": {k: int(v.sum()) for k, v in masks.items()},
              "gate_rows_recorded_detail": {k: int(v.sum()) for k, v in rmasks.items()},
              "gate_rows_mask_disagreement_dry_vs_recorded": int((touched != (rmasks["unm_up"] | rmasks["mat_down"] | rmasks["lirf_up"])).sum()),
              "delta_float_on_gate_rows": {"mean_s": float(delta[touched].mean()), "rms_s": float(np.sqrt(np.mean(delta[touched] ** 2))), "max_s": float(delta[touched].max()),
                                           "min_s": float(delta[touched].min())},
              "rows_changed_vs_v34": int((change != 0).sum()), "rows_changed_outside_gate": int(((change != 0) & ~touched).sum()),
              "max_abs_change_s": float(np.abs(change).max()), "rms_change_all_rows_s": float(np.sqrt(np.mean(change ** 2))),
              "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "independent_rebuild": {"source": export_v33.V30_DETAIL, "rows_differing": int(differing.sum()),
                                      "rows_differing_on_gate_rows": int((differing & touched).sum()),
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported).max())}}
    g = pd.DataFrame({"month": rec["month"].to_numpy()[touched], "airport": airport[touched], "day": day["day"].to_numpy()[touched],
                      "clock": day["clock"].to_numpy()[touched], "delta": delta[touched]})
    report["gate_rows_by_airport"] = g.groupby(["month", "airport"]).agg(rows=("delta", "size"), mean_delta_s=("delta", "mean"),
                                                                        no_clock=("clock", lambda c: int((~c.astype(bool)).sum()))).reset_index().to_dict(orient="records")
    report["gate_rows_eham_by_day"] = g[g.airport == "EHAM"].groupby("day").agg(rows=("delta", "size"), mean_delta_s=("delta", "mean")).reset_index().to_dict(orient="records")
    if report["rows_changed_outside_gate"]:
        raise ValueError("Rows outside the gate changed.")
    write_json(report_path, report)
    print({k: report[k] for k in ["candidate", "gate_rows", "rows_changed_vs_v34", "max_abs_change_s", "rms_change_all_rows_s", "delta_float_on_gate_rows", "independent_rebuild",
                                  "gate_rows_mask_disagreement_dry_vs_recorded"]}, receipt.get("sha256"))
    print(pd.DataFrame(report["gate_rows_by_airport"]).round(1).to_string())


if __name__ == "__main__":
    main()
