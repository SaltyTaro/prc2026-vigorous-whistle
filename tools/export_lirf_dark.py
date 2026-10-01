"""Member step of the LIRF dark-regime export (configs/lirf_dark_reread_v1.json export_plan): v53's model part with the dark months'
route rows recomputed at the selected (w, s). The reconciliation with the published daily totals follows, last, by
tools/export_daily_totals.py --base <this file>.

Base: v53's model part = the LIRF member step's dry-run file before the reconciliation (runs/adsb_ctx3_lirf_mask/export_dry/
vigorous-whistle_v53.parquet, hash-bound below). Route: the ranking's lirf_nonpop_unmatched rows (final-file dry run's recipe detail) in dark
LIRF months, i.e. months whose LIRF non-population rows have an ADS-B match share below 5% (July 2026; January 2026 is covered and keeps v53).
Member M = the 16-seed mean of the LIRF masked member's refit12 ranking predictions. On the route:
value = base + (candidate value - control value), control = v53's arithmetic (w 0.55, s 0.35/0.60, gate u600), candidate = the selected
(w, s) with the same gate (tools/run_lirf_dark.values), floored at -12 s; every other row keeps the base integer. Dry run only: the output
never goes under runs/submissions. Nothing is uploaded.

  --dry-run DIR [--version 54] [--candidate NAME --user-override TEXT]
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
from prc2026_final import lirf_gate as lg  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import run_lirf_dark as rld  # noqa: E402

DECL = "configs/lirf_dark_reread_v1.json"
MASK_DECL = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
DETAIL = "runs/final_dryrun/run/recipe/detail.parquet"
TEMPLATE = "data/raw/submitting.parquet"
BASE = "runs/adsb_ctx3_lirf_mask/export_dry/vigorous-whistle_v53.parquet"
BASE_RECORD = "runs/adsb_ctx3_lirf_mask/export_dry/export_v53.json"
ROUTE, MATCHED = "lirf_nonpop_unmatched", "lirf_nonpop_matched"
DARK_SHARE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", type=Path, required=True)
    parser.add_argument("--version", type=int, default=54)
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
        raise ValueError("v53's model part differs from its record.")
    w, share = float(decl["candidates"][candidate]["member_weight"]), float(decl["candidates"][candidate]["network_share"])

    base = pd.read_parquet(ROOT / BASE)
    ids = base[ID].to_numpy()
    values = base[TARGET].to_numpy(dtype=float).copy()
    detail = pd.read_parquet(ROOT / DETAIL, columns=[ID, "airport", "month", "scope", "v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
    detail = pd.DataFrame({ID: ids}).merge(detail, on=ID, how="left", validate="one_to_one")
    if detail["scope"].isna().any():
        raise ValueError("The recipe detail does not cover every ranking row.")
    month = detail["month"].to_numpy()
    scope = detail["scope"].to_numpy()
    shares = {}
    for m in sorted(set(month[detail["airport"].eq("LIRF").to_numpy()])):
        nm, nu = int(((scope == MATCHED) & (month == m)).sum()), int(((scope == ROUTE) & (month == m)).sum())
        shares[m] = {"matched": nm, "unmatched": nu, "match_share": nm / (nm + nu) if nm + nu else None, "dark": bool(nm + nu and nm / (nm + nu) < DARK_SHARE)}
    dark_months = [m for m, v in shares.items() if v["dark"]]
    if dark_months != ["2026-07"]:
        raise ValueError(f"Unexpected dark months {dark_months}: {shares}")
    k = (scope == ROUTE) & np.isin(month, dark_months)

    fits = read_json(ROOT / MASK_DECL)["fits"]
    paths = [ROOT / fits["outputs"].split("<variant>")[0] / f"{v}_s{s}" / "refit12" / "ranking_predictions.parquet" for v in fits["variants"] for s in fits["seeds"]]
    missing = [p.as_posix() for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"refit12 member predictions missing: {len(missing)}")
    route_ids = ids[k]
    member = np.mean([pd.read_parquet(p).set_index(ID)["adsb_full"].reindex(route_ids).to_numpy(dtype=float) for p in paths], axis=0)
    if not np.isfinite(member).all():
        raise ValueError("The member does not cover every route row.")
    s = {c: detail.loc[k, src].to_numpy(dtype=float) for c, src in (("v9", "v9"), ("net", "ens_ctx_lirf"), ("tree", "ens_v22_lirf_q"))}
    s["member"] = member
    control, fire0 = rld.values(s, rld.CONTROL["member_weight"], rld.CONTROL["network_share"], 600.0)
    cand, fire1 = rld.values(s, w, share, 600.0)
    diff = values[k] - control
    checks = {"dark_months": dark_months, "match_shares": shares, "route_rows": int(k.sum()), "base_minus_control_max_abs_s": float(np.max(np.abs(diff))),
              "base_minus_control_rows_over_1s": int((np.abs(diff) > 1.0).sum()), "gate_rows_control": int(fire0.sum()), "gate_rows_candidate": int(fire1.sum()),
              "candidate_minus_control_mean_s": float((cand - control).mean()), "candidate_minus_control_rms_s": float(np.sqrt(((cand - control) ** 2).mean()))}
    if checks["base_minus_control_rows_over_1s"] > 0.01 * checks["route_rows"]:
        raise ValueError(f"v53's model part does not hold v53's LIRF value on the route: {checks}")
    new = values.copy()
    new[k] = np.maximum(values[k] + (cand - control), lg.FLOOR)
    provenance = {"experiment_id": decl["experiment_id"], "candidate": candidate, "member_weight": w, "network_share": share, "dry_run": True,
                  "step": "dark-regime LIRF re-read (before the reconciliation)",
                  "selected_by": "the declared rule (docs/lirf_dark/results_v1.json)" if candidate == selected else f"the user's decision: {args.user_override}",
                  "recipe": f"v53's model part + (value at w {w}, s {share} - value at w 0.55, s 0.583; gate u600) on {ROUTE} rows of dark LIRF months {dark_months}, floored at -12 s",
                  "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
                  "results": {decl["report"]: {"sha256": sha256_file(ROOT / decl["report"])}},
                  "code": {p: sha256_file(ROOT / p) for p in ["src/prc2026_final/lirf_gate.py", "tools/run_lirf_dark.py", "tools/export_lirf_dark.py"]},
                  "inputs": {DETAIL: sha256_file(ROOT / DETAIL), BASE: base_sha, **{p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}},
                  "audit_used": False, "external_data": "none for this step (adsb.lol / OSM inputs of the networks as before)"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": new}), ROOT / TEMPLATE, output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - values
    if ((change != 0) & ~k).any():
        raise ValueError("A row outside the route changed.")
    report = {"created_at_utc": utc_now(), "declaration": DECL, "candidate": candidate, "member_weight": w, "network_share": share, "base": BASE,
              "output": str(output), "sha256": sha256_file(output), "rows": int(len(ids)), "checks": checks, "rows_changed": int((change != 0).sum()),
              "max_abs_change_s": float(np.abs(change).max()), "mean_change_s": float(change[k].mean()), "mean_abs_change_s": float(np.abs(change[k]).mean()),
              "sum_squared_change": float((change ** 2).sum()), "receipt": receipt}
    write_json(report_path, report)
    print({x: report[x] for x in ("sha256", "rows_changed", "max_abs_change_s", "mean_change_s", "mean_abs_change_s", "sum_squared_change")})
    print({x: checks[x] for x in checks if x != "match_shares"})


if __name__ == "__main__":
    main()
