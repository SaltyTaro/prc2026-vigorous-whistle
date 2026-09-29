"""Export v46 = v45 with a thinned-regime candidate's committee mix (configs/thinned_regime_round_v1.json) run through the shipped stack.

The declared rule selected nothing; the export is the user's override and needs --user-override with the user's words.

Construction, on the ranking rows:
  1. tools/export_disagreement_gate.rebuild_v34: v31 / v33 / v34 floats, each rounding to its upload; the dry-run recipe detail.
  2. control = tools/export_stack.layers(v34, detail, ..., queue 0.75 regular, stretch 1500 / 900): v38's floats, which must round to
     the uploaded v38 on every row.
  3. candidate: the detail's committee_mix replaced by src/prc2026_final/share_gate.mix (network share, networks-far-below gate); v34's
     recipe recomputed on it; the same layers call on v34 + (recipe' - recipe) and the candidate detail, so the u300 gate, the callsign
     and queue layers and the stretch all act on the candidate's own values.
  4. v46 = the candidate's float where it differs from the control, v45's integer elsewhere. LIRF rows (fill pins, LIRF gate, blends)
     do not depend on the committee mix and must not change.
Independent rebuild from the recorded v30 detail's components. Nothing is uploaded.

  --candidate n85_nd1200 --dry-run DIR
  --candidate n85_nd1200 --version 46 --user-override TEXT
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
from prc2026_final import share_gate  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import export_v33  # noqa: E402
from export_disagreement_gate import V34_LIRF, rebuild_v34  # noqa: E402
from export_fair_weights import FLOOR, V33_OUTER, recipe, stats  # noqa: E402
from export_stack import layers  # noqa: E402

DECLARATION = "configs/thinned_regime_round_v1.json"
RESULTS = "docs/thinned_regime/results_v1.json"
V38 = ("runs/submissions/vigorous-whistle_v38.parquet", "61264cdb7f0a37a8e5cf943f8e950441df38b826132cfc1097f00ca9d0519fe4")
V45 = ("runs/submissions/vigorous-whistle_v45.parquet", "72d24155e6b6b8dbb4d6caf7a590679d3b242ad7c83793dcc3d28d5b88fd77fa", 224.2398)
STACK = {"qw": 0.75, "qsource": "regular", "mix_up": None, "peer": False, "stretch": 1500.0, "stretch_lower": 900.0}
CODE = ["tools/export_share_gate.py", "src/prc2026_final/share_gate.py", "tools/export_stack.py", "tools/export_disagreement_gate.py", "tools/export_fair_weights.py", "tools/export_v33.py"]


def uploaded(path: str, sha: str, ids: np.ndarray) -> np.ndarray:
    if sha256_file(ROOT / path) != sha:
        raise ValueError(f"The uploaded {path} changed.")
    values = pd.read_parquet(ROOT / path).set_index(ID)[TARGET].reindex(ids)
    if values.isna().any():
        raise ValueError(f"{path} does not cover every ranking row.")
    return values.to_numpy().astype(np.int64)


def stacks(detail: pd.DataFrame, v34: np.ndarray, at34: np.ndarray, ids: np.ndarray, share: float, gate: dict | None):
    """(control floats, candidate floats, control masks, gate mask) through tools/export_stack.layers."""
    committee, nets = detail["committee"].to_numpy(dtype=float), share_gate.network_table(detail)
    control_mix, _ = share_gate.mix(committee, nets, share_gate.CONTROL_SHARE)
    if not float(np.nanmax(np.abs(control_mix - detail["committee_mix"].to_numpy(dtype=float)))) <= 1e-9:
        raise ValueError("The detail's committee_mix is not the control mix.")
    candidate_mix, fire = share_gate.mix(committee, nets, share, gate)
    candidate_detail = detail.assign(committee_mix=candidate_mix)
    at_candidate = recipe(candidate_detail, V33_OUTER, V34_LIRF)
    control, masks = layers(v34, detail, at34, ids, STACK["qw"], STACK["qsource"], STACK["mix_up"], STACK["peer"], STACK["stretch"], STACK["stretch_lower"])
    candidate, _ = layers(v34 + (at_candidate - at34), candidate_detail, at_candidate, ids, STACK["qw"], STACK["qsource"], STACK["mix_up"], STACK["peer"],
                          STACK["stretch"], STACK["stretch_lower"])
    return control, candidate, masks, fire & (detail["airport"].to_numpy() != "LIRF")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", type=int, default=46)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    results = read_json(ROOT / RESULTS)
    if results.get("declaration_sha256") != object_hash(decl):
        raise ValueError("The recorded results are not for this declaration.")
    share, gate = share_gate.candidate_spec(decl, args.candidate)
    selected = results.get("selected_candidate")
    if args.dry_run is None:
        if selected != args.candidate and not args.user_override:
            raise ValueError(f"The round did not select {args.candidate}; shipping it needs --user-override with the user's words.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/thinned_regime/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")

    rec, ids, dry, lfpg, _, v34, at34 = rebuild_v34()
    v38, v45 = uploaded(*V38, ids), uploaded(V45[0], V45[1], ids)
    control, candidate, masks, fire = stacks(dry, v34, at34, ids, share, gate)
    if not np.array_equal(np.rint(control).astype(np.int64), v38):
        raise ValueError("The control stack does not round to the uploaded v38.")
    if not (np.isfinite(candidate).all() and np.isfinite(control).all()):
        raise ValueError("Non-finite stack values.")
    airport, month = dry["airport"].to_numpy(), dry["month"].to_numpy()
    changed = np.abs(candidate - control) > 1e-9
    if (changed & (airport == "LIRF")).any():
        raise ValueError("A LIRF row depends on the committee mix.")
    if not np.array_equal(v45[changed], v38[changed]):
        raise ValueError("A changed row carries a layer above v38 in v45.")
    values = np.where(changed, candidate, v45.astype(float))
    provenance = {"experiment_id": decl["experiment_id"], "candidate": args.candidate, "dry_run": args.dry_run is not None,
                  "selected_by": (f"the user's override of the declared rule, which selected nothing: {args.user_override}" if args.user_override
                                  else "the declared rule" if selected == args.candidate else "dry run"),
                  "recipe": f"v45 with the committee mix at network share {share}" + (f" and share {gate['share_inside']} where the committee sits more than {gate['threshold_s']} s above the networks" if gate else "")
                            + ", through the shipped stack (v34 weights, u300 gate, callsign fix 50%, queue trust 0.75, two-sided stretch 1500 / 900); LIRF rows unchanged; floor -12 s",
                  "incumbent": {"submission": Path(V45[0]).name, "official_rmse_seconds": V45[2], "sha256": V45[1], "note": "the clean model line's best"},
                  "declaration": {DECLARATION: {"sha256": sha256_file(ROOT / DECLARATION), "object_hash": object_hash(decl)}},
                  "development_results": {p: sha256_file(ROOT / p) for p in [RESULTS, f"docs/thinned_regime/reread_{args.candidate}.json"] if (ROOT / p).exists()},
                  "code": {p: sha256_file(ROOT / p) for p in CODE},
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL, V38[0], V45[0]]},
                  "external_data": "adsb.lol globe_history (ODbL) surface positions, as in v45", "audit_used": False, "floor_seconds": FLOOR,
                  "official_scores_used": "none"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - v45.astype(float)
    if ((change != 0) & ~changed).any():
        raise ValueError("A row outside the candidate's scope changed.")

    # independent rebuild from the recorded v30 detail's own components
    rec34 = recipe(rec, V33_OUTER, V34_LIRF)
    control_r, candidate_r, _, fire_r = stacks(rec, np.maximum(rec34 + lfpg, FLOOR), rec34, ids, share, gate)
    independent = np.where(np.abs(candidate_r - control_r) > 1e-9, candidate_r, v45.astype(float))
    differing = np.rint(independent).astype(np.int64) != exported
    n = len(ids)
    moved = change != 0
    anchored = masks["callsign"] | masks["queue"]
    report = {"created_at_utc": utc_now(), "version": args.version, "candidate": args.candidate, "network_share": share, "gate": gate, "base": 45,
              "dry_run": args.dry_run is not None, "output": output.as_posix(), "sha256": sha256_file(output), "receipt": receipt, "rows": n,
              "checks": {"control_rounds_to_uploaded_v38": True, "lirf_rows_changed": 0, "changed_rows_equal_v38_in_v45": True, "rows_changed_outside_scope": 0},
              "rows_changed_float": int(changed.sum()), "rows_changed_integer": int(moved.sum()), "gate_rows": int(fire.sum()),
              "gate_rows_by_month": {m: int((fire & (month == m)).sum()) for m in sorted(set(month))},
              "rows_changed_in_anchored_layers": int((moved & anchored).sum()),
              "mean_change_s": float(change[moved].mean()), "mean_abs_change_s": float(np.abs(change[moved]).mean()), "max_abs_change_s": float(np.abs(change).max()),
              "mean_change_gate_rows_s": float(change[fire].mean()) if fire.any() else 0.0, "sum_squared_change": float((change ** 2).sum()),
              "by_month": {m: stats(change, month == m) for m in sorted(set(month))}, "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "independent_rebuild": {"source": export_v33.V30_DETAIL, "rows_differing": int(differing.sum()), "gate_rows_mask_disagreement": int((fire != fire_r).sum()),
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported).max())},
              "rmse_if_every_changed_label_equals_the_new_value": float(np.sqrt((n * V45[2] ** 2 - (change ** 2).sum()) / n)),
              "rmse_if_every_changed_label_equals_the_old_value": float(np.sqrt((n * V45[2] ** 2 + (change ** 2).sum()) / n))}
    write_json(report_path, report)
    print({k: report[k] for k in ["sha256", "rows_changed_integer", "gate_rows", "gate_rows_by_month", "mean_change_s", "mean_abs_change_s", "max_abs_change_s", "mean_change_gate_rows_s",
                                  "sum_squared_change", "independent_rebuild"]})
    print({a: {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()} for a, e in report["by_airport"].items()})


if __name__ == "__main__":
    main()
