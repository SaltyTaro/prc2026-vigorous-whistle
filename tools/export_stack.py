"""Export the stacked candidate on v34: u300 gate (configs/disagreement_gate_v1.json) + callsign fix 50% (configs/callsign_fix_v1.json)
+ optionally queue trust (configs/queue_trust_v1.json) at weight W on its scope rows, restricted to regular ADS-B matches with
--queue-source regular (the round's callsign-matched queue row failed in July 2025; disclosed post-hoc restriction).

Construction: v34 floats rebuilt (tools/export_disagreement_gate.rebuild_v34; v31 / v33 / v34 must round to the uploads) -> + u300 gate delta
(dry-run detail) -> callsign rows = max(0.5 x observed taxi + 0.5 x start, -12) -> queue rows = max(W x observed taxi + (1 - W) x that, -12).
Independent rebuild from the recorded v30 detail's components. Nothing is uploaded.

  --queue W --queue-source regular|all --dry-run OUT
  --queue W --queue-source regular --version N --user-override TEXT
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
from run_queue_trust import offsets_all, segments  # noqa: E402

DECLS = ["configs/disagreement_gate_v1.json", "configs/callsign_fix_v1.json", "configs/queue_trust_v1.json"]
RESULTS = ["docs/disagreement/gate_v1_results.json", "docs/disagreement/callsign_fix_v1_results.json", "docs/disagreement/queue_trust_v1_results.json"]


def layers(v34_start: np.ndarray, detail: pd.DataFrame, at34: np.ndarray, ids: np.ndarray, qw: float | None, qsource: str):
    """(values, masks): v34 floats -> u300 -> callsign 50% -> queue trust."""
    gdelta, _ = gate_delta(detail, read_json(ROOT / DECLS[0])["candidates"]["u300"], at34)
    values = np.maximum(v34_start + gdelta, FLOOR)
    cs = scope_table(read_json(ROOT / DECLS[1])).set_index(ID).reindex(ids)
    off_cs = offsets()
    cs_in = cs["motion_start"].notna().to_numpy()
    cs_taxi = cs["takeoff"].to_numpy() - cs["motion_start"].to_numpy() + np.array([off_cs.get(a, np.nan) for a in cs["airport"].fillna("")])
    values = values.copy()
    values[cs_in] = np.maximum(0.5 * cs_taxi[cs_in] + 0.5 * values[cs_in], FLOOR)
    q_in = np.zeros(len(ids), dtype=bool)
    if qw is not None:
        seg = segments(["2026-01", "2026-07"]).reindex(ids)
        off = offsets_all()
        q_in = seg["in_scope"].fillna(False).to_numpy(dtype=bool)
        if qsource == "regular":
            q_in &= seg["source"].eq("regular").to_numpy()
        q_taxi = seg["taxi"].to_numpy() + np.array([off.get(a, np.nan) for a in seg["airport"].fillna("")])
        values[q_in] = np.maximum(qw * q_taxi[q_in] + (1 - qw) * values[q_in], FLOOR)
    return values, {"u300": gdelta != 0, "callsign": cs_in, "queue": q_in}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=float)
    parser.add_argument("--queue-source", choices=["regular", "all"], default="regular")
    parser.add_argument("--version", type=int, default=35)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    for d, r in zip(DECLS, RESULTS):
        if read_json(ROOT / r).get("declaration_sha256") != object_hash(read_json(ROOT / d)):
            raise ValueError(f"{r} is not the recorded result of {d}.")
    if args.dry_run is None:
        if not args.user_override:
            raise ValueError("None of the three rounds selected; shipping the stack needs --user-override.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/disagreement/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")

    rec, ids, dry, lfpg, uploaded, v34, at34 = rebuild_v34()
    values, m = layers(v34, dry, at34, ids, args.queue, args.queue_source)
    touched = m["u300"] | m["callsign"] | m["queue"]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite export values.")
    provenance = {"experiment_id": "stack_u300_cs50" + (f"_qt{int(round(100 * args.queue))}_{args.queue_source}" if args.queue is not None else ""),
                  "dry_run": args.dry_run is not None, "selected_by": f"user override of the unchanged rule: {args.user_override}" if args.user_override else "dry run",
                  "recipe": "v34 + u300 disagreement gate + callsign fix 50%" + (f" + queue trust {args.queue} ({args.queue_source} matches)" if args.queue is not None else ""),
                  "incumbent": {"submission": "vigorous-whistle_v34.parquet", "official_rmse_seconds": 228.5933, "sha256": V34_SHA},
                  "declarations": {d: sha256_file(ROOT / d) for d in DECLS}, "development_results": {r: sha256_file(ROOT / r) for r in RESULTS},
                  "external_data": "adsb.lol globe_history (ODbL) surface positions",
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL, *read_json(ROOT / DECLS[1])["rematch"]["outputs"]]},
                  "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - uploaded[34].astype(float)

    rec34 = recipe(rec, V33_OUTER, V34_LIRF)
    independent, _ = layers(np.maximum(rec34 + lfpg, FLOOR), rec, rec34, ids, args.queue, args.queue_source)
    differing = np.rint(independent).astype(np.int64) != exported
    airport = rec["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "version": args.version, "dry_run": args.dry_run is not None, "output": output.as_posix(), "receipt_sha256": receipt.get("sha256"),
              "queue": args.queue, "queue_source": args.queue_source, "layer_rows": {k: int(v.sum()) for k, v in m.items()},
              "rows_changed_vs_v34": int((change != 0).sum()), "rows_changed_outside_layers": int(((change != 0) & ~touched).sum()),
              "max_abs_change_s": float(np.abs(change).max()), "rms_change_all_rows_s": float(np.sqrt(np.mean(change ** 2))),
              "mean_change_by_layer_s": {k: float(change[v].mean()) if v.any() else 0.0 for k, v in m.items()},
              "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "independent_rebuild": {"source": export_v33.V30_DETAIL, "rows_differing": int(differing.sum()), "rows_differing_in_layers": int((differing & touched).sum()),
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported).max())}}
    if report["rows_changed_outside_layers"]:
        raise ValueError("Rows outside the layers changed.")
    write_json(report_path, report)
    print({k: report[k] for k in ["layer_rows", "rows_changed_vs_v34", "max_abs_change_s", "mean_change_by_layer_s", "independent_rebuild"]}, receipt.get("sha256"))


if __name__ == "__main__":
    main()
