"""Member step of the LIRF masked-member export (configs/adsb_ctx3_lirf_mask_dev_v1.json export_plan): v52's model part with the route's
ranking rows recomputed. The reconciliation with the published daily totals is applied afterwards, last, by
tools/export_daily_totals.py --base <this file>.

Base: v52's model part = the members' dry-run file before the reconciliation (runs/adsb_ctx3_mask/export_dry/vigorous-whistle_v52.parquet,
hash-bound below). Route: the ranking's lirf_nonpop_unmatched rows (scope of the final-file dry run's recipe detail, the same detail and
components v45's gate export read). Member M = the 16-seed mean of the refit12 ranking predictions. On the route:
value = base + (gated blend with network' - gated blend with ens_ctx_lirf), network' = (1 - w) ens_ctx_lirf + w M, v34's unmatched blend with
v45's gate u600 (src/prc2026_final/lirf_gate.py), floored at -12 s; every other row keeps the base integer. Dry run only: the output never
goes under runs/submissions (the final file comes from export_daily_totals). Nothing is uploaded.

  --dry-run DIR [--version 53] [--candidate NAME --user-override TEXT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import lirf_gate as lg  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

DECL = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
DETAIL = "runs/final_dryrun/run/recipe/detail.parquet"
TEMPLATE = "data/raw/submitting.parquet"
BASE = "runs/adsb_ctx3_mask/export_dry/vigorous-whistle_v52.parquet"
BASE_RECORD = "runs/adsb_ctx3_mask/export_dry/export_v52.json"
GATE = {"T": 600, "lean": {"v9": 0.0, "net": 0.2, "tree": 0.8}}
ROUTE = "lirf_nonpop_unmatched"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", type=Path, required=True)
    parser.add_argument("--version", type=int, default=53)
    parser.add_argument("--candidate")
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decl = read_json(ROOT / DECL)
    results = read_json(ROOT / decl["report"])
    selected = results["selected_candidate"]
    candidate = args.candidate or selected
    if candidate is None or candidate not in decl["candidates"]:
        raise ValueError(f"{candidate!r} is not a declared (or selected) candidate.")
    if candidate != selected and not args.user_override:
        raise ValueError("A non-selected candidate needs --user-override with the user's words.")
    out_dir = args.dry_run.resolve()
    if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
        raise ValueError("The member step never writes under runs/submissions.")
    output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base_sha = read_json(ROOT / BASE_RECORD)["sha256"]
    if sha256_file(ROOT / BASE) != base_sha:
        raise ValueError("v52's model part differs from its record.")
    w = float(decl["candidates"][candidate]["member_weight"])

    base = pd.read_parquet(ROOT / BASE)
    ids = base[ID].to_numpy()
    values = base[TARGET].to_numpy(dtype=float).copy()
    detail = pd.read_parquet(ROOT / DETAIL, columns=[ID, "airport", "month", "scope", "v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
    detail = pd.DataFrame({ID: ids}).merge(detail, on=ID, how="left", validate="one_to_one")
    if detail["scope"].isna().any():
        raise ValueError("The recipe detail does not cover every ranking row.")
    k = detail["scope"].eq(ROUTE).to_numpy()
    fits = decl["fits"]
    paths = [ROOT / fits["outputs"].split("<variant>")[0] / f"{v}_s{s}" / "refit12" / "ranking_predictions.parquet" for v in fits["variants"] for s in fits["seeds"]]
    missing = [p.as_posix() for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"refit12 member predictions missing: {len(missing)}")
    route_ids = ids[k]
    member = np.mean([pd.read_parquet(p).set_index(ID)["adsb_full"].reindex(route_ids).to_numpy(dtype=float) for p in paths], axis=0)
    if not np.isfinite(member).all():
        raise ValueError("The member does not cover every route row.")
    v9, net, tree = (detail.loc[k, c].to_numpy(dtype=float) for c in ["v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
    control, fire0 = lg.gated(v9, net, tree, "unmatched", GATE)
    candidate_values, fire1 = lg.gated(v9, (1 - w) * net + w * member, tree, "unmatched", GATE)
    diff = values[k] - control
    checks = {"route_rows": int(k.sum()), "base_minus_control_max_abs_s": float(np.max(np.abs(diff))), "base_minus_control_rows_over_1s": int((np.abs(diff) > 1.0).sum()),
              "gate_rows_control": int(fire0.sum()), "gate_rows_candidate": int(fire1.sum()),
              "member_minus_network_rms_s": float(np.sqrt(np.mean((member - net) ** 2))), "member_mean_s": float(member.mean()), "network_mean_s": float(net.mean())}
    if checks["base_minus_control_rows_over_1s"] > 0.01 * checks["route_rows"]:
        raise ValueError(f"v52's model part does not hold the shipped LIRF value on the route: {checks}")
    new = values.copy()
    new[k] = np.maximum(values[k] + (candidate_values - control), lg.FLOOR)
    provenance = {"experiment_id": decl["experiment_id"], "candidate": candidate, "member_weight": w, "dry_run": True, "step": "member (before the reconciliation)",
                  "selected_by": "the declared rule (docs/adsb_ctx3_lirf_mask/results_v1.json)" if candidate == selected else f"the user's decision: {args.user_override}",
                  "recipe": f"v52's model part + (gated blend with (1 - {w}) ens_ctx_lirf + {w} M - gated blend with ens_ctx_lirf) on {ROUTE} rows, floored at -12 s",
                  "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
                  "results": {decl["report"]: {"sha256": sha256_file(ROOT / decl["report"])}},
                  "code": {p: sha256_file(ROOT / p) for p in ["src/prc2026_final/lirf_gate.py", "tools/export_lirf_member.py"]},
                  "inputs": {DETAIL: sha256_file(ROOT / DETAIL), BASE: base_sha, **{p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}},
                  "audit_used": False, "external_data": "none for this step (adsb.lol / OSM inputs of the networks as before)"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": new}), ROOT / TEMPLATE, output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - values
    if ((change != 0) & ~k).any():
        raise ValueError("A row outside the route changed.")
    month = detail["month"].to_numpy()
    by_month = {m: {"route_rows": int((k & (month == m)).sum()), "rows_changed": int(((change != 0) & (month == m)).sum()),
                    "mean_change_s": float(change[k & (month == m)].mean()) if (k & (month == m)).any() else 0.0,
                    "mean_abs_change_s": float(np.abs(change[k & (month == m)]).mean()) if (k & (month == m)).any() else 0.0}
                for m in sorted(set(month))}
    report = {"created_at_utc": utc_now(), "declaration": DECL, "candidate": candidate, "member_weight": w, "base": BASE, "output": str(output), "sha256": sha256_file(output),
              "rows": int(len(ids)), "checks": checks, "rows_changed": int((change != 0).sum()), "max_abs_change_s": float(np.abs(change).max()),
              "sum_squared_change": float((change ** 2).sum()), "by_month": by_month, "receipt": receipt}
    write_json(report_path, report)
    print({x: report[x] for x in ("sha256", "rows_changed", "max_abs_change_s", "sum_squared_change", "by_month", "checks")})


if __name__ == "__main__":
    main()
