"""Declared fair-set weight round (configs/fair_weights_v1.json): v33's recipe re-weighted between the v9 side and the fold side on the
fair development rows (runs/fair_dev/rows.parquet, tools/run_fair_dev.py). No model is fitted.

The control rebuild must equal the fair v33 column to 1e-9 s per row. Outer weights act on non-LIRF rows by scope (LTFM / other matched /
other unmatched); LIRF v9 shares act on lirf_nonpop_matched / lirf_nonpop_unmatched rows with the control's net/tree proportions; every
other row keeps the control.

  --score --select   -> runs/fair_weights/selection.json and docs/fair_dev/fair_weights_results.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402

DECLARATION = "configs/fair_weights_v1.json"
ROWS = "runs/fair_dev/rows.parquet"
OUT = ROOT / "runs/fair_weights"
REPORT = ROOT / "docs/fair_dev/fair_weights_results.json"
SEED = 20260927


def build(f: pd.DataFrame, outer: dict, lirf: dict) -> np.ndarray:
    airport, matched = f["airport"].to_numpy(), f["matched"].to_numpy()
    floor = f["floor"].to_numpy(dtype=float)
    non_lirf = airport != "LIRF"
    w = np.where(airport == "LTFM", outer["ltfm"], np.where(matched, outer["matched_other"], outer["unmatched_other"]))
    values = np.where(non_lirf, np.maximum((1 - w) * f["base"].to_numpy() + w * f["mix"].to_numpy(), floor), f["v33"].to_numpy())
    for scope, flag in [("lirf_nonpop_matched", "lirf_m"), ("lirf_nonpop_unmatched", "lirf_u")]:
        spec = lirf[scope]
        k = f[flag].to_numpy() & (airport == "LIRF")
        column = (spec["net"] * f["lirf_net"].to_numpy() + spec["tree"] * f["lirf_tree"].to_numpy()) / (spec["net"] + spec["tree"])
        values = np.where(k, np.maximum(spec["v9"] * f["v9"].to_numpy() + (1 - spec["v9"]) * column, floor), values)
    return values


def candidate_weights(control: dict, spec: dict) -> tuple[dict, dict]:
    outer = {**control["outer"], **spec.get("outer", {})}
    lirf = {k: dict(v) for k, v in control["lirf"].items()}
    for scope, share in spec.get("lirf_v9", {}).items():
        lirf[scope]["v9"] = share
    return outer, lirf


def main() -> None:
    global ROWS, OUT, REPORT
    if "--test-shipped" in sys.argv:  # code-path test on the shipped-provenance rows; never a result
        ROWS, OUT, REPORT = "runs/fair_dev/rows_shipped.parquet", ROOT / "runs/fair_dev/test_shipped", ROOT / "runs/fair_dev/test_shipped/report.json"
    if "--declaration" in sys.argv:  # a follow-up declaration on the same rows (e.g. configs/fair_lirf_v1.json)
        name = sys.argv[sys.argv.index("--declaration") + 1]
        stem = Path(name).stem
        OUT, REPORT = ROOT / "runs" / stem, ROOT / "docs/fair_dev" / f"{stem}_results.json"
    else:
        name = DECLARATION
    decl = read_json(ROOT / name)
    if isinstance(decl.get("control"), str):  # "v33 exactly as in configs/fair_weights_v1.json"
        decl = {**decl, "control": read_json(ROOT / DECLARATION)["control"]}
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    rows = pd.read_parquet(ROOT / ROWS)
    control = decl["control"]
    months = decl["recipe_months"]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "audit_used": False, "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        base = build(f, control["outer"], control["lirf"])
        if not np.isfinite(base).all():
            raise ValueError(f"The control rebuild has non-finite values on {month}.")
        check = float(np.max(np.abs(base - f["v33"].to_numpy())))
        if not check <= 1e-9:  # also catches NaN
            raise ValueError(f"The control rebuild does not equal the fair v33 on {month} (max {check}).")
        report["months"][month] = {"rows": int(len(f)), "control_rmse": rmse(y, base), "control_rebuild_max_abs": check,
                                   "v30_fair_rmse": rmse(y, f["v30"].to_numpy()), "v33_vs_v30_fair_pct": 100 * (rmse(y, base) / rmse(y, f["v30"].to_numpy()) - 1)}
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            values = build(f, *candidate_weights(control, spec))
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - base) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, base), "rows_differing_from_control": int(changed.sum()),
                                      "paired_days_vs_control": paired_days(days, y, values, base, SEED)}
        print(month, {"control": round(report["months"][month]["control_rmse"], 4), "v33_vs_v30_fair_pct": round(report["months"][month]["v33_vs_v30_fair_pct"], 4)},
              {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_month[month].items()}, flush=True)
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "months": {m: per_month[m][name] for m in months}}
        ratios = [entry["months"][m]["ratio_to_control"] for m in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in months if entry["months"][m]["rows_differing_from_control"] > 0]
        below = [m for m in touched if entry["months"][m]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        report["candidates"][name] = entry
    eligible = [n for n, e in report["candidates"].items() if e["eligible"]]
    report["selected_candidate"] = min(eligible, key=lambda n: (report["candidates"][n]["mean_ratio"], n)) if eligible else None
    report["eligible"] = eligible
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_fair_weights.py", "tools/run_fair_dev.py", "tools/run_fold_residual.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": report["selected_candidate"], "eligible": eligible})
    for n, e in report["candidates"].items():
        print(f"{n:15s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


if __name__ == "__main__":
    main()
