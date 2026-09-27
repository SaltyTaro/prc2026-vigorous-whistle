"""Export a selection of the fair-set weight round (configs/fair_weights_v1.json) on top of v33, modelled on tools/export_v33.py.

Control = v33's weights: outer weight of the committee mix (against v9 + the residual stage) 1.0 on LTFM rows, 0.6 on the other
ADS-B-matched non-LIRF rows, 0.7 on the other unmatched non-LIRF rows; LIRF blends lirf_nonpop_matched v9 .15 / net .40 / tree .45 and
lirf_nonpop_unmatched v9 .40 / net .35 / tree .25. A candidate overrides outer weights and/or a LIRF v9 share s, keeping the control's
net:tree proportions: value = max(s * v9 + (1 - s) * (net_w * N + tree_w * T) / (net_w + tree_w), -12) with N = ens_ctx_lirf and
T = ens_v22_lirf_q. LIRF 'fixed' rows (fill population) keep v30.

Construction (the declared ship clause):
- the recipe is rebuilt on the final-file runner's dry-run detail (runs/final_dryrun/run/recipe/detail.parquet); at v30's weights (outer 0.7,
  lctx_b100_up LIRF blends) it must equal the dry run's v30 on every row (1e-9), and at the control its non-LIRF rows must equal
  tools/export_v33.recipe exactly;
- v33's floats are rebuilt as export_v33 does (v31 floats = recorded v30 detail + 0.5 x the LFPG July EOBT_1 effect, floor; + the v33 delta,
  floor) and must round exactly to the uploaded v33;
- delta = recipe(dry, candidate) - recipe(dry, control), which must be zero outside the candidate's scopes; output = max(v33 floats + delta, -12)
  through export_submission as vigorous-whistle_v<N>.
- Independent rebuild from the recorded v30 detail's own components (runs/adsb_ctx/export_v30/ranking_values.parquet): recipe(recorded,
  candidate) + 0.5 x the LFPG effect, floored; the count of integers differing from the export is reported (dry-run float noise).

  --candidate NAME [--version N]    requires runs/fair_weights/selection.json to have selected NAME under this declaration's object hash;
                                    writes runs/submissions/vigorous-whistle_v<N>.parquet (+ receipt) and docs/fair_dev/export_v<N>.json
  --candidate NAME --dry-run OUT    skips the selection check; writes the file and the report under OUT only (never under runs/submissions)
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

DECLARATION = "configs/fair_weights_v1.json"
SELECTION = "runs/fair_weights/selection.json"
V31_FILE, V31_SHA = export_v33.V31_FILE, export_v33.V31_SHA
V33_FILE, V33_SHA = "runs/submissions/vigorous-whistle_v33.parquet", "ceb055981846cdb025f0f0efed45df588bca2f21b56a0a024aec77ce426bed48"
DRY, REPAIRED, V30_DETAIL = export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL
FLOOR = -12.0
V33_OUTER = {"ltfm": 1.0, "matched_other": 0.6, "unmatched_other": 0.7}
V33_LIRF = {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.40, "net": 0.35, "tree": 0.25}}
V30_OUTER = {"ltfm": 0.7, "matched_other": 0.7, "unmatched_other": 0.7}
LIRF_NET, LIRF_TREE = "ens_ctx_lirf", "ens_v22_lirf_q"


def scopes(detail: pd.DataFrame) -> dict[str, np.ndarray]:
    airport, scope = detail["airport"].to_numpy(), detail["scope"].to_numpy()
    non_lirf = airport != "LIRF"
    matched = scope == "matched_no_lirf"
    return {"ltfm": airport == "LTFM", "matched_other": non_lirf & (airport != "LTFM") & matched,
            "unmatched_other": non_lirf & (airport != "LTFM") & ~matched,
            "lirf_nonpop_matched": scope == "lirf_nonpop_matched", "lirf_nonpop_unmatched": scope == "lirf_nonpop_unmatched",
            "lirf_fixed": (airport == "LIRF") & (scope == "fixed")}


def recipe(detail: pd.DataFrame, outer: dict, lirf: dict) -> np.ndarray:
    """v30's recipe with outer weights by scope and LIRF (v9, net, tree) blends; LIRF rows outside the two blend scopes keep v30."""
    s = scopes(detail)
    if (s["lirf_nonpop_matched"] | s["lirf_nonpop_unmatched"] | s["lirf_fixed"]).sum() != (detail["airport"].to_numpy() == "LIRF").sum():
        raise ValueError("A LIRF row is in no known scope.")
    base = detail["v9"].to_numpy(dtype=float) + np.where(detail["scope"].eq("matched_no_lirf").to_numpy(), detail["residual"].to_numpy(dtype=float), 0.0)
    mix = detail["committee_mix"].to_numpy(dtype=float)
    w = np.select([s["ltfm"], s["matched_other"], s["unmatched_other"]], [outer["ltfm"], outer["matched_other"], outer["unmatched_other"]], np.nan)
    values = detail["v30"].to_numpy(dtype=float).copy()
    non_lirf = s["ltfm"] | s["matched_other"] | s["unmatched_other"]
    values[non_lirf] = np.maximum((1 - w[non_lirf]) * base[non_lirf] + w[non_lirf] * mix[non_lirf], FLOOR)
    v9, net, tree = (detail[c].to_numpy(dtype=float) for c in ["v9", LIRF_NET, LIRF_TREE])
    for scope, spec in lirf.items():
        k = s[scope]
        if abs(spec["v9"] + spec["net"] + spec["tree"] - 1) > 1e-12 or min(spec.values()) < 0 or spec["net"] + spec["tree"] <= 0:
            raise ValueError(f"LIRF weights on {scope} must be non-negative and sum to one.")
        table = (spec["net"] * net[k] + spec["tree"] * tree[k]) / (spec["net"] + spec["tree"])
        values[k] = np.maximum(spec["v9"] * v9[k] + (1 - spec["v9"]) * table, FLOOR)
    if not np.isfinite(values).all():
        raise ValueError("The rebuilt recipe has non-finite values.")
    return values


def candidate_weights(control: dict, spec: dict) -> tuple[dict, dict, list[str]]:
    """The candidate's outer and LIRF weights (control's net:tree proportions kept) and the scopes it may touch."""
    outer = {**control["outer"], **spec.get("outer", {})}
    lirf = {}
    for scope, w in control["lirf"].items():
        if scope not in spec.get("lirf_v9", {}):  # untouched: the control's weights verbatim (re-normalising would add float noise)
            lirf[scope] = dict(w)
            continue
        s = spec["lirf_v9"][scope]
        lirf[scope] = {"v9": s, "net": (1 - s) * w["net"] / (w["net"] + w["tree"]), "tree": (1 - s) * w["tree"] / (w["net"] + w["tree"])}
    touched = list(spec.get("outer", {})) + list(spec.get("lirf_v9", {}))
    return outer, lirf, touched


def stats(change: np.ndarray, mask: np.ndarray) -> dict:
    c = change[mask]
    return {"rows": int(mask.sum()), "changed": int((c != 0).sum()), "mean_change_s": float(c.mean()) if len(c) else 0.0,
            "rms_change_s": float(np.sqrt(np.mean(c ** 2))) if len(c) else 0.0, "max_abs_change_s": float(np.abs(c).max()) if len(c) else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", type=int, default=34)
    parser.add_argument("--dry-run", type=Path, help="scratch directory: skip the selection check and write only there")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    if args.candidate not in decl["candidates"]:
        raise ValueError(f"{args.candidate} is not a declared candidate.")
    control = decl["control"]
    if control["outer"] != V33_OUTER or control["lirf"] != V33_LIRF:
        raise ValueError("The declaration's control is not v33's weights.")
    if args.dry_run is None:
        if not (ROOT / SELECTION).exists():
            raise ValueError(f"{SELECTION} does not exist; the round has not selected yet (use --dry-run for a test).")
        selection = read_json(ROOT / SELECTION)
        if selection.get("declaration_sha256") != object_hash(decl) or selection.get("selected_candidate") != args.candidate:
            raise ValueError(f"The round did not select {args.candidate} under this declaration.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/fair_dev/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    for path, sha in [(V31_FILE, V31_SHA), (V33_FILE, V33_SHA)]:
        if sha256_file(ROOT / path) != sha:
            raise ValueError(f"The uploaded {path} changed.")

    rec = pd.read_parquet(ROOT / V30_DETAIL)
    ids = rec[ID].to_numpy()
    dry = pd.read_parquet(ROOT / DRY).set_index(ID).reindex(ids).reset_index()
    repaired = pd.read_parquet(ROOT / REPAIRED, columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    if dry["v30"].isna().any() or np.isnan(repaired).any() or not (dry["airport"].to_numpy() == rec["airport"].to_numpy()).all() \
            or not (dry["scope"].to_numpy() == rec["scope"].to_numpy()).all():
        raise ValueError("The dry-run / repaired details do not align with the v30 detail.")
    lfpg = 0.5 * (repaired - dry["v30"].to_numpy(dtype=float))  # v31's LFPG July EOBT_1 blend (zero elsewhere)

    # v33 floats exactly as export_v33 builds them; they must round to the uploaded v33
    v31 = np.maximum(rec["v30"].to_numpy(dtype=float) + lfpg, FLOOR)
    uploaded31 = pd.read_parquet(ROOT / V31_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v31).astype(np.int64), uploaded31.astype(np.int64)):
        raise ValueError("Rebuilt v31 floats do not round to the uploaded v31.")
    v33_delta = export_v33.recipe(dry, 1.0, 0.6, 0.7) - export_v33.recipe(dry, 0.7, 0.7, 0.7)
    v33 = np.maximum(v31 + v33_delta, FLOOR)
    uploaded33 = pd.read_parquet(ROOT / V33_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v33).astype(np.int64), uploaded33.astype(np.int64)):
        raise ValueError("Rebuilt v33 floats do not round to the uploaded v33.")

    # reproductions of the recipe on the dry-run detail
    v30_lirf = {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.40, "net": 0.35, "tree": 0.25}}
    at_v30 = recipe(dry, V30_OUTER, v30_lirf)
    v30_check = float(np.max(np.abs(at_v30 - dry["v30"].to_numpy(dtype=float))))
    if not v30_check <= 1e-9:
        raise ValueError(f"The rebuild at v30's weights does not equal the dry run's v30 (max {v30_check}).")
    base = recipe(dry, control["outer"], control["lirf"])
    s = scopes(dry)
    non_lirf = s["ltfm"] | s["matched_other"] | s["unmatched_other"]
    ref33 = export_v33.recipe(dry, 1.0, 0.6, 0.7)
    if not np.array_equal(base[non_lirf], ref33[non_lirf]):
        raise ValueError("The control rebuild does not equal export_v33's construction on non-LIRF rows.")
    lirf_check = float(np.max(np.abs(base[~non_lirf] - dry["v30"].to_numpy(dtype=float)[~non_lirf])))
    if not lirf_check <= 1e-9:
        raise ValueError(f"The control rebuild does not equal the dry run's v30 on LIRF rows (max {lirf_check}).")

    outer, lirf, touched = candidate_weights(control, decl["candidates"][args.candidate])
    delta = recipe(dry, outer, lirf) - base
    in_scope = np.zeros(len(ids), dtype=bool)
    for scope in touched:
        in_scope |= s[scope]
    if np.any(delta[~in_scope] != 0):
        raise ValueError("The delta touches rows outside the candidate's scopes.")
    values = np.maximum(v33 + delta, FLOOR)

    provenance = {"experiment_id": decl["experiment_id"], "candidate": args.candidate, "dry_run": args.dry_run is not None,
                  "weights": {"outer": outer, "lirf": lirf}, "touched_scopes": touched,
                  "recipe": "v33 with the fair-set round's weights; delta from the dry-run detail (candidate minus the v33 control) added to v33's floats, floor -12",
                  "incumbent": {"submission": "vigorous-whistle_v33.parquet", "official_rmse_seconds": 228.7246, "sha256": V33_SHA},
                  "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "declaration_object_hash": object_hash(decl),
                  "selection": None if args.dry_run is not None else {"path": SELECTION, "sha256": sha256_file(ROOT / SELECTION)},
                  "inputs": {DRY: sha256_file(ROOT / DRY), REPAIRED: sha256_file(ROOT / REPAIRED), V30_DETAIL: sha256_file(ROOT / V30_DETAIL)},
                  "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy()
    change = exported.astype(float) - uploaded33.astype(float)
    changed = change != 0

    # independent rebuild from the recorded v30 detail's own components
    rec_v30_lirf = recipe(rec, V30_OUTER, v30_lirf)
    rec_check = float(np.max(np.abs(rec_v30_lirf - rec["v30"].to_numpy(dtype=float))))
    independent = np.maximum(recipe(rec, outer, lirf) + lfpg, FLOOR)
    differing = np.rint(independent).astype(np.int64) != exported.astype(np.int64)
    airport = rec["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "candidate": args.candidate, "version": args.version, "dry_run": args.dry_run is not None,
              "output": output.as_posix(), "receipt_sha256": receipt.get("sha256"), "weights": {"outer": outer, "lirf": lirf}, "touched_scopes": touched,
              "checks": {"v30_rebuild_on_dry_max_abs": v30_check, "control_non_lirf_equals_export_v33": True, "control_lirf_vs_dry_v30_max_abs": lirf_check,
                         "v31_floats_round_to_uploaded": True, "v33_floats_round_to_uploaded": True, "delta_outside_scopes": 0,
                         "recorded_detail_v30_rebuild_max_abs": rec_check},
              "delta_float": {k: {"rows": int(m.sum()), "mean_s": float(delta[m].mean()) if m.any() else 0.0,
                                  "rms_s": float(np.sqrt(np.mean(delta[m] ** 2))) if m.any() else 0.0} for k, m in s.items()},
              "rows_changed_vs_v33": int(changed.sum()), "rows_changed_outside_scopes": int((changed & ~in_scope).sum()),
              "max_abs_change_s": float(np.abs(change).max()), "rms_change_all_rows_s": float(np.sqrt(np.mean(change ** 2))),
              "by_scope": {k: stats(change, m) for k, m in s.items()},
              "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "independent_rebuild": {"source": V30_DETAIL, "rows_differing": int(differing.sum()),
                                      "note": "rows outside the scopes differ only through v33's own construction (dry-run float noise; export_v33's check found 193 rows at 1 s)",
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported.astype(np.int64)).max()),
                                      "rows_differing_outside_scopes": int((differing & ~in_scope).sum())}}
    if report["rows_changed_outside_scopes"]:
        raise ValueError("Rows outside the candidate's scopes changed.")
    write_json(report_path, report)
    print({k: report[k] for k in ["candidate", "rows_changed_vs_v33", "max_abs_change_s", "rms_change_all_rows_s", "independent_rebuild"]}, receipt.get("sha256"))
    for k, v in report["by_scope"].items():
        print(f"  {k:22s} rows {v['rows']:7d} changed {v['changed']:7d} mean {v['mean_change_s']:+8.3f} rms {v['rms_change_s']:8.3f} max {v['max_abs_change_s']:6.0f}")


if __name__ == "__main__":
    main()
