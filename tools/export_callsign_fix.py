"""Export a candidate of the callsign-fix round (configs/callsign_fix_v1.json) on top of v34 (optionally with the u300 gate).

- v34's floats are rebuilt by tools/export_disagreement_gate.rebuild_v34 (v31 / v33 / v34 must round to the uploads);
- start = v34 floats + the u300 gate delta (u300_cs*) or v34 floats (cs*);
- on the ranking rows in scope (tools/adsb_rematch_nonm.py output for the ranking: no existing key match, IATA->ICAO mapped gate segment),
  value = max(w * (takeoff - motion_start + offset[airport]) + (1 - w) * start, -12), offsets from tools/run_callsign_fix.offsets;
- output = start elsewhere, floored, through export_submission; independent rebuild from the recorded v30 detail's components.

  --candidate NAME --dry-run OUT
  --candidate NAME --version N --user-override TEXT
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
from export_disagreement_gate import V34_LIRF, V34_SHA, gate_delta, rebuild_v34  # noqa: E402
from export_fair_weights import FLOOR, V33_OUTER, recipe, stats  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402

DECLARATION = "configs/callsign_fix_v1.json"
RESULTS = "docs/disagreement/callsign_fix_v1_results.json"
GATE_DECLARATION = "configs/disagreement_gate_v1.json"


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

    rec, ids, dry, lfpg, uploaded, v34, at34 = rebuild_v34()
    off = offsets()
    if {k: round(v, 6) for k, v in off.items()} != {k: round(v, 6) for k, v in results["offsets_s"].items()}:
        raise ValueError("The offsets differ from the recorded round.")
    gate_spec = read_json(ROOT / GATE_DECLARATION)["candidates"]["u300"] if spec.get("gate") == "u300" else {}
    gdelta, gmasks = gate_delta(dry, gate_spec, at34)
    start = np.maximum(v34 + gdelta, FLOOR)

    scope = scope_table(decl).set_index(ID).reindex(ids)
    inside = scope["motion_start"].notna().to_numpy()
    airport = rec["airport"].to_numpy()
    if np.any(inside & (airport != scope["airport"].fillna("").to_numpy())):
        raise ValueError("A scope row sits at another airport in the ranking detail.")
    if np.any(inside & rec["scope"].eq("matched_no_lirf").to_numpy()):
        raise ValueError("A scope row is already ADS-B-matched in the ranking detail.")
    taxi = scope["takeoff"].to_numpy() - scope["motion_start"].to_numpy() + np.array([off.get(a, np.nan) for a in scope["airport"].fillna("")])
    w = spec["weight"]
    values = start.copy()
    values[inside] = np.maximum(w * taxi[inside] + (1 - w) * start[inside], FLOOR)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite export values.")
    touched = inside | (gdelta != 0)

    provenance = {"experiment_id": decl["experiment_id"], "candidate": args.candidate, "dry_run": args.dry_run is not None,
                  "selected_candidate": f"{args.candidate} (user override)" if args.user_override else args.candidate,
                  "selected_by": f"user override of the unchanged rule: {args.user_override}" if args.user_override else "dry run",
                  "recipe": ("v34" + (" with the u300 disagreement gate" if gate_spec else "") + f"; no-NM EDDF/EHAM departures newly linked to ADS-B by IATA->ICAO callsign mapping (gate segments) blend {w} x the observed ADS-B taxi (+ airport offset); floor -12"),
                  "incumbent": {"submission": "vigorous-whistle_v34.parquet", "official_rmse_seconds": 228.5933, "sha256": V34_SHA},
                  "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "declaration_object_hash": object_hash(decl),
                  "gate_declaration": GATE_DECLARATION if gate_spec else None,
                  "development_results": RESULTS, "development_results_sha256": sha256_file(ROOT / RESULTS), "offsets_s": off,
                  "external_data": "adsb.lol globe_history (ODbL) surface positions",
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL, *decl["rematch"]["outputs"]]},
                  "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - uploaded[34].astype(float)

    # independent rebuild from the recorded detail's own components
    rec34 = recipe(rec, V33_OUTER, V34_LIRF)
    rdelta, _ = gate_delta(rec, gate_spec, rec34)
    rstart = np.maximum(rec34 + rdelta + lfpg, FLOOR)
    independent = rstart.copy()
    independent[inside] = np.maximum(w * taxi[inside] + (1 - w) * rstart[inside], FLOOR)
    differing = np.rint(independent).astype(np.int64) != exported
    month = rec["month"].to_numpy()
    report = {"created_at_utc": utc_now(), "candidate": args.candidate, "version": args.version, "dry_run": args.dry_run is not None,
              "output": output.as_posix(), "receipt_sha256": receipt.get("sha256"), "offsets_s": off,
              "checks": {"v31_v33_v34_floats_round_to_uploaded": True},
              "gate_rows": int((gdelta != 0).sum()), "callsign_rows": {m: int((inside & (month == m)).sum()) for m in ["2026-01", "2026-07"]},
              "overlap_gate_and_callsign": int((inside & (gdelta != 0)).sum()),
              "callsign_rows_mean_change_s": float((values - start)[inside].mean()) if inside.any() else 0.0,
              "callsign_rows_observed_taxi_mean_s": float(taxi[inside].mean()) if inside.any() else None,
              "callsign_rows_start_mean_s": float(start[inside].mean()) if inside.any() else None,
              "rows_changed_vs_v34": int((change != 0).sum()), "rows_changed_outside_scope": int(((change != 0) & ~touched).sum()),
              "max_abs_change_s": float(np.abs(change).max()), "rms_change_all_rows_s": float(np.sqrt(np.mean(change ** 2))),
              "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "independent_rebuild": {"source": export_v33.V30_DETAIL, "rows_differing": int(differing.sum()),
                                      "rows_differing_in_scope": int((differing & touched).sum()),
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported).max())}}
    if report["rows_changed_outside_scope"]:
        raise ValueError("Rows outside the candidate's scope changed.")
    write_json(report_path, report)
    print({k: report[k] for k in ["candidate", "gate_rows", "callsign_rows", "overlap_gate_and_callsign", "callsign_rows_mean_change_s", "callsign_rows_observed_taxi_mean_s",
                                  "callsign_rows_start_mean_s", "rows_changed_vs_v34", "max_abs_change_s", "independent_rebuild"]}, receipt.get("sha256"))


if __name__ == "__main__":
    main()
