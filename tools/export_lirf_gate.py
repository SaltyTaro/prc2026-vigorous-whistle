"""Export v45 = base file + the LIRF disagreement gate (configs/lirf_gate_v1.json's selected candidate) on the ranking's LIRF rows.

The ranking's LIRF components (v9, ens_ctx_lirf, ens_v22_lirf_q, scope) come from the final-file dry run's recipe detail. On the two LIRF
non-population scopes the v34 blend is rebuilt and checked against the base file (v40 holds exactly that blend, rounded); the gated rows
get value = base + (gated blend - v34 blend), floored at -12 s; every other row keeps the base integer. Nothing is uploaded.

  --version 45 [--base 40] [--user-override TEXT]   (the selected candidate needs no override; another candidate or base does)
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

DECL = "configs/lirf_gate_v1.json"
DETAIL = "runs/final_dryrun/run/recipe/detail.parquet"
TEMPLATE = "data/raw/submitting.parquet"
BASES = {40: ("runs/submissions/vigorous-whistle_v40.parquet", "da5b20c328ab6be5fca4a21c37cb7c80210397fba62d8395557a9b84d77d162a", 224.4588),
         }
SCOPES = {"lirf_nonpop_unmatched": "unmatched", "lirf_nonpop_matched": "matched"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", default=DECL)
    parser.add_argument("--candidate")
    parser.add_argument("--base", type=int, default=40, choices=sorted(BASES))
    parser.add_argument("--version", type=int, default=45)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    config = read_json(ROOT / args.declaration)
    if not config.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    report_in = read_json(ROOT / config["report"])
    selected = report_in.get("selected_candidate")
    candidate = args.candidate or selected
    if candidate not in config["candidates"]:
        raise ValueError(f"{candidate!r} is not a declared candidate.")
    if args.dry_run is None:
        if (candidate != selected or args.base != 40) and not args.user_override:
            raise ValueError("A non-selected candidate or a non-declared base needs --user-override with the user's words.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/lirf_gate/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    base_file, base_sha, base_rmse = BASES[args.base]
    if sha256_file(ROOT / base_file) != base_sha:
        raise ValueError(f"The uploaded v{args.base} changed.")
    spec = config["candidates"][candidate]

    base = pd.read_parquet(ROOT / base_file)
    ids = base[ID].to_numpy()
    values = base[TARGET].to_numpy(dtype=float).copy()
    detail = pd.read_parquet(ROOT / DETAIL, columns=[ID, "airport", "month", "scope", "v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
    detail = pd.DataFrame({ID: ids}).merge(detail, on=ID, how="left", validate="one_to_one")
    if detail["scope"].isna().any():
        raise ValueError("The recipe detail does not cover every ranking row.")
    gate_mask = np.zeros(len(ids), dtype=bool)
    delta = np.zeros(len(ids))
    checks = {}
    for scope_name, scope in SCOPES.items():
        k = detail["scope"].eq(scope_name).to_numpy()
        v9, net, tree = (detail.loc[k, c].to_numpy(dtype=float) for c in ["v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
        v34, _ = lg.gated(v9, net, tree, scope, None)
        gated_values, fire = lg.gated(v9, net, tree, scope, spec.get(scope))
        checks[scope_name] = {"rows": int(k.sum()), "gated": int(fire.sum()),
                              "base_minus_v34_blend_max_abs_s": float(np.max(np.abs(values[k] - v34))) if k.any() else 0.0,
                              "base_minus_v34_blend_rows_over_1s": int((np.abs(values[k] - v34) > 1.0).sum())}
        checks[scope_name]["gated_rows_base_minus_v34_blend_max_abs_s"] = float(np.max(np.abs(values[k][fire] - v34[fire]))) if fire.any() else 0.0
        if args.base == 40 and checks[scope_name]["base_minus_v34_blend_rows_over_1s"] > 0.01 * max(int(k.sum()), 1):
            raise ValueError(f"v40 does not hold the v34 blend on {scope_name}: {checks[scope_name]}")  # a few rows carry v35's queue-trust layer
        idx = np.flatnonzero(k)[fire]
        delta[idx] = (gated_values - v34)[fire]
        gate_mask[idx] = True
    new = np.where(gate_mask, np.maximum(values + delta, lg.FLOOR), values)
    provenance = {"experiment_id": config["experiment_id"], "candidate": candidate, "dry_run": args.dry_run is not None,
                  "selected_by": "the declared rule (docs/lirf_gate/results_v1.json)" if candidate == selected and args.base == 40 else f"the user's decision: {args.user_override}",
                  "recipe": f"v{args.base} + (gated LIRF blend - v34 LIRF blend) on gated rows ({candidate}: {spec}), floored at -12 s",
                  "incumbent": {"submission": Path(base_file).name, "official_rmse_seconds": base_rmse, "sha256": base_sha},
                  "declaration": {args.declaration: {"sha256": sha256_file(ROOT / args.declaration), "object_hash": object_hash(config)}},
                  "results": {config["report"]: {"sha256": sha256_file(ROOT / config["report"])}},
                  "code": {p: sha256_file(ROOT / p) for p in ["src/prc2026_final/lirf_gate.py", "tools/export_lirf_gate.py"]},
                  "inputs": {DETAIL: sha256_file(ROOT / DETAIL), base_file: base_sha}, "audit_used": False, "external_data": "none for this layer"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": new}), ROOT / TEMPLATE, output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - values
    if ((change != 0) & ~gate_mask).any():
        raise ValueError("A row outside the gate changed.")
    month = detail["month"].to_numpy()
    by_month = {m: {"gated": int((gate_mask & (month == m)).sum()), "mean_change_s": float(change[gate_mask & (month == m)].mean()) if (gate_mask & (month == m)).any() else 0.0,
                    "max_change_s": float(change[gate_mask & (month == m)].max()) if (gate_mask & (month == m)).any() else 0.0}
                for m in sorted(set(month))}
    n = len(ids)
    report = {"created_at_utc": utc_now(), "declaration": args.declaration, "candidate": candidate, "base": args.base, "output": str(output), "sha256": sha256_file(output),
              "dry_run": args.dry_run is not None, "rows": n, "checks": checks, "rows_changed": int((change != 0).sum()), "mean_change_s": float(change[gate_mask].mean()) if gate_mask.any() else 0.0,
              "mean_abs_change_s": float(np.abs(change[gate_mask]).mean()) if gate_mask.any() else 0.0, "max_abs_change_s": float(np.abs(change).max()),
              "sum_squared_change": float((change ** 2).sum()), "by_month": by_month, "receipt": receipt,
              "rmse_if_every_changed_label_equals_the_new_value": float(np.sqrt((n * base_rmse ** 2 - (change ** 2).sum()) / n)),
              "rmse_if_every_changed_label_equals_the_old_value": float(np.sqrt((n * base_rmse ** 2 + (change ** 2).sum()) / n))}
    write_json(report_path, report)
    print({k: v for k, v in report.items() if k in ("sha256", "rows_changed", "mean_change_s", "mean_abs_change_s", "max_abs_change_s", "sum_squared_change", "by_month", "checks")})


if __name__ == "__main__":
    main()
